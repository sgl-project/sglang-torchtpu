# SPDX-License-Identifier: Apache-2.0
"""TP8 numerics of the Muse Spark prefill program against the CPU reference decoder.

Reuses test_muse_megakernel_numerics.py's seeded serving-layout weights, KV pools and its
teacher-forced reference layer: an extend chunk of q tokens per request is the reference's
q-row block over the cached context. fp4 experts go through the serving load and the
block-256 requant (the fused-RS MoE reads serving layouts), so SGLANG_TPU_MUSE_MK_QBLOCK=256 is
required. Run one process per device:

    eval $(python3 -m torch.tpu.distributed)
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
    SGLANG_TPU_MUSE_MK_QBLOCK=256 torchrun --nproc_per_node="$WORLD_SIZE" \\
        test/manual/tpu/test_muse_mk_prefill_numerics.py

Rank 0 prints one MK_PREFILL_NUMERICS line per (case, layer) and MK_PREFILL_NUMERICS_PASS when
every check holds; every rank prints MK_PREFILL_NUMERICS_DONE.
"""

import argparse
import json
import os
import shutil
import sys
import time
import types

import msgspec
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_muse_megakernel_numerics as M  # noqa: E402

TP = M.TP
# Measured (block-256 experts): per-row errors up to 2.6% / 3.6% on fp4 / fp8 layers, means
# 1.6-1.8% / 2.7-3.0%. fp8 layers' fused-RS rows are fp8-quantized (~2^-4 per element) and
# RPA tiles round to bf16; the decode megakernel's (1e-2, 2e-2) assume neither. A misindexed
# key or weight moves every row by O(10%).
_TOL_ROW_REL = (5e-2, 5e-2)
_TOL_MEAN_REL = (3.5e-2, 3.5e-2)
# That stream error also moves biased router scores by more than the decode test's 2e-3.
_NEAR_TIE = 1e-2
# The fp8 gathers add one e4m3 rounding (~2^-4) of the expert input and the post-expert
# projection per layer.
_FP8_GATHER_TOL_SCALE = 1.5


class Case(msgspec.Struct, frozen=True):
    name: str
    n_req: int
    q: int  # chunk tokens per request
    ctx: tuple  # cached tokens per request
    expert_group: int = 0
    fp8_gather: bool = False
    # M.Case fields the reference reads
    kv: str = "sal_fp8"


CASES = {
    c.name: c
    for c in (
        # one request, its chunk filling the 384-row bucket
        Case("pf_c1q384", 1, 384, (1500,)),
        # two requests and 64 padded rows; a context below one window
        Case("pf_c2q160_pad", 2, 160, (2600, 700), expert_group=2),
        # past one 2048-token window on both requests, e4m3 gathers
        Case("pf_c2q192_fp8", 2, 192, (5000, 2300), fp8_gather=True),
    )
}
# (case, flag): reruns that must be bit-identical to the case's run
DEFAULT_EXACT = (("pf_c1q384", "repeat"), ("pf_c1q384", "expert_group=2"))


def step_inputs(case: Case) -> M.StepInputs:
    ps = M._PAGES[case.kv]
    n_live = case.n_req * case.q
    rows = -(-n_live // 128) * 128
    pps = -(-(max(case.ctx) + case.q) // ps)
    # SWA ring of more than window + chunk tokens: the chunk's writes never land on a key
    # that one of its queries still reads
    swa_pages = -(-(M.WINDOW + case.q) // ps) + 1
    g = M._gen("prefill_step", case.name)
    pos = torch.zeros(rows, dtype=torch.int32)
    for j, c in enumerate(case.ctx):
        pos[j * case.q : (j + 1) * case.q] = torch.arange(c, c + case.q)
    full = torch.randperm(case.n_req * pps, generator=g).to(torch.int32) + 1
    ring = torch.randperm(case.n_req * swa_pages, generator=g).to(torch.int32) + 1
    p = torch.arange(pps)
    swa = torch.cat([ring[j * swa_pages + p % swa_pages] for j in range(case.n_req)])
    emb = torch.randn(rows, M.HIDDEN, generator=g)
    emb[n_live:] = 0
    return M.StepInputs(
        rows=rows,
        n_live=n_live,
        pps=pps,
        swa_pages=swa_pages,
        pos=pos,
        pf=full,
        psw=swa,
        emb=emb.to(torch.bfloat16),
        max_position=-(-(max(case.ctx) + case.q + 8) // 8) * 8,
    )


def _fp8_serving_refs(l: int, rank: int, device: torch.device) -> list:
    """One fp8 layer's experts as the serving load lays them out for fused-RS: fp8 [E, K, N]
    and channelwise scales (E, 2, N), the values M.decode_expert reads."""
    slabs = [M.expert_slabs(l, rank, e, "fp8") for e in range(M.N_EXPERTS)]

    def unwords(words):
        shifts = torch.arange(0, 32, 8, dtype=torch.int32)[None, :, None]
        b = ((words[:, None, :] >> shifts) & 0xFF).to(torch.uint8)
        return b.reshape(words.shape[0] * 4, words.shape[1]).view(torch.float8_e4m3fn)

    return [
        torch.stack([unwords(s[0]) for s in slabs]).to(device),
        torch.stack([s[1].float() for s in slabs]).to(device),
        torch.stack([unwords(s[2]) for s in slabs]).to(device),
        torch.stack([s[3].float() for s in slabs]).to(device),
    ]


class _PrefillRunner:
    """M._Runner's weights and pools, with the prefill program's fields on its fake
    MuseMegakernel."""

    def __init__(
        self, n_layers: int, global_layers: tuple, rank: int, max_position: int
    ):
        from sglang_tpu.muse_megakernel import kernel as K
        from sglang_tpu.muse_megakernel import ops

        self.ops = ops
        self.base = M._Runner(K, ops, n_layers, global_layers, rank, max_position)
        for l, kind in enumerate(self.base.kinds):
            if kind == "fp8":
                self.base.per_layer_experts[l] = _fp8_serving_refs(
                    l, rank, self.base.device
                )
        mk = self.base.mk
        mk.w4a16_layers = ops.w4a16_layers_of(self.base.kinds, M.tp_qblock())
        mk.prefill_fp8_gather = False
        mk.prefill_config = types.MethodType(ops.MuseMegakernel.prefill_config, mk)
        mk.prefill_callable_for = types.MethodType(
            ops.MuseMegakernel.prefill_callable_for, mk
        )
        self.last_cfg = None
        traced_config = mk.prefill_config

        def config(**kw):
            self.last_cfg = traced_config(**kw)
            return self.last_cfg

        mk.prefill_config = config
        ops._register_prefill_op()

    def step(self, case: Case, st: M.StepInputs, kv: list, expert_group: int):
        """(hidden [rows, H] bf16, aux [rows, L * H] bf16); kv updated in place."""
        self.base.mk.prefill_fp8_gather = case.fp8_gather
        self.last_cfg = None
        experts = self.base.experts(expert_group)
        d = self.base.device
        n = case.n_req
        meta = torch.tensor([st.n_live, self.base.rank], dtype=torch.int32)
        kv_lens = torch.tensor([c + case.q for c in case.ctx], dtype=torch.int32)
        cu_q = torch.arange(n + 1, dtype=torch.int32) * case.q
        dist_ = torch.tensor([n, n, n], dtype=torch.int32)
        capture = list(range(self.base.n_layers))
        op = torch.ops.sglang_tpu_pallas.muse_prefill

        def run(meta, pos, kv_lens, cu_q, dist_, pf, psw, emb, weights, experts, kv):
            return op(
                meta,
                pos,
                kv_lens,
                cu_q,
                dist_,
                pf,
                psw,
                emb,
                weights,
                experts,
                kv,
                capture,
            )

        torch._dynamo.reset()
        torch.tpu._clear_cache()
        compiled = torch.compile(run, fullgraph=True, dynamic=False)
        hid, aux = compiled(
            *(
                t.to(d)
                for t in (meta, st.pos, kv_lens, cu_q, dist_, st.pf, st.psw, st.emb)
            ),
            self.base.weights,
            experts,
            kv,
        )
        if self.last_cfg is None:
            raise RuntimeError(
                "the step ran a cached executable: the op was not traced"
            )
        return hid.cpu(), aux.cpu()


def _set_tolerances(fp8_gather: bool) -> None:
    scale = _FP8_GATHER_TOL_SCALE if fp8_gather else 1.0
    M.TOL_ROW_REL = tuple(t * scale for t in _TOL_ROW_REL)
    M.TOL_MEAN_REL = tuple(t * scale for t in _TOL_MEAN_REL)
    M.NEAR_TIE = _NEAR_TIE


def run_case(runner: _PrefillRunner, case: Case, rank: int, out_dir: str, exact: list):
    st = step_inputs(case)
    if st.max_position > runner.base.mk.max_position:
        raise ValueError(f"{case.name} needs {st.max_position} rope rows")
    t0 = time.time()
    kv = runner.base.pools(case, st)
    hid, aux = runner.step(case, st, kv, case.expert_group)
    new_kv = runner.base.new_kv(case, st, kv)
    t_kernel = time.time() - t0
    n = st.n_live
    cdir = os.path.join(out_dir, case.name)
    torch.save({"hid": hid, "aux": aux, "new_kv": new_kv}, f"{cdir}/rank{rank}.pt")
    exact_res = []
    for flag in exact:
        group = int(flag.split("=")[1]) if flag.startswith("expert_group=") else None
        if group is None and flag != "repeat":
            raise ValueError(f"unknown exact flag {flag}")
        # fresh pools: the step rewrites the chunk's KV rows
        hid2, aux2 = runner.step(
            case,
            st,
            runner.base.pools(case, st),
            case.expert_group if group is None else group,
        )
        exact_res.append(
            {
                "case": case.name,
                "flag": flag,
                "hid_identical": bool(torch.equal(hid[:n], hid2[:n])),
                "aux_identical": bool(torch.equal(aux[:n], aux2[:n])),
            }
        )
    with open(f"{cdir}/exact_rank{rank}.json", "w") as f:
        json.dump(exact_res, f)
    dist.barrier()
    data = [torch.load(f"{cdir}/rank{d}.pt") for d in range(TP)]
    _set_tolerances(case.fp8_gather)
    L = runner.base.n_layers
    aux0 = data[0]["aux"].float()
    for l in range(L):
        if l % TP != rank:
            continue
        stream_in = (
            M._bf(M._rms(st.emb[:n].float(), M.RMS_EPS))
            if l == 0
            else aux0[:n, (l - 1) * M.HIDDEN : l * M.HIDDEN]
        )
        res = M.check_layer(
            l,
            case,
            st,
            L,
            runner.base.global_layers,
            "shifted",
            stream_in,
            aux0[:n, l * M.HIDDEN : (l + 1) * M.HIDDEN],
            torch.stack([data[d]["new_kv"][l] for d in range(TP)]),
        )
        res["kernel_s"] = round(t_kernel, 1)
        with open(f"{cdir}/layer{l}.json", "w") as f:
            json.dump(res, f)
    if rank == 0:
        last = aux0[:n, (L - 1) * M.HIDDEN :]
        want = M._bf(M._rms(last, M.RMS_EPS, M.final_gain()))
        rel = float(M._row_rel(data[0]["hid"][:n].float(), want).max())
        fin = {
            "case": case.name,
            "final_max_row_rel": rel,
            "ranks_identical": all(
                torch.equal(data[0]["hid"][:n], data[d]["hid"][:n])
                and torch.equal(data[0]["aux"][:n], data[d]["aux"][:n])
                for d in range(1, TP)
            ),
        }
        fin["ok"] = rel <= M.TOL_HID_REL and fin["ranks_identical"]
        with open(f"{cdir}/final.json", "w") as f:
            json.dump(fin, f)
    dist.barrier()


def summarize(cases: list, n_layers: int, out_dir: str) -> bool:
    ok = True
    for case in cases:
        cdir = os.path.join(out_dir, case.name)
        for l in range(n_layers):
            with open(f"{cdir}/layer{l}.json") as f:
                r = json.load(f)
            ok &= r["ok"]
            print("MK_PREFILL_NUMERICS " + json.dumps(r), flush=True)
        with open(f"{cdir}/final.json") as f:
            r = json.load(f)
        ok &= r["ok"]
        print("MK_PREFILL_NUMERICS_FINAL " + json.dumps(r), flush=True)
        for d in range(TP):
            with open(f"{cdir}/exact_rank{d}.json") as f:
                for r in json.load(f):
                    good = r["hid_identical"] and r["aux_identical"]
                    ok &= good
                    if d == 0 or not good:
                        print(
                            "MK_PREFILL_NUMERICS_EXACT " + json.dumps({**r, "rank": d}),
                            flush=True,
                        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--global-layers", default="1,4")
    ap.add_argument("--out-dir", default="/tmp/mk_prefill_numerics")
    args = ap.parse_args()
    if M.tp_qblock() != 256:
        raise ValueError(
            "needs SGLANG_TPU_MUSE_MK_QBLOCK=256 (served block-256 experts)"
        )
    M._QBLOCK_REF = M._SERVED = True
    M._LayerRef.bf16_attention_io = True

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    if dist.get_world_size() != TP:
        raise RuntimeError(f"needs {TP} processes, got {dist.get_world_size()}")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // TP))
    torch._dynamo.config.cache_size_limit = 256
    torch.compiler.config.force_disable_caches = True

    cases = [CASES[c] for c in args.cases]
    global_layers = tuple(int(x) for x in args.global_layers.split(",") if x)
    if rank == 0:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        for c in cases:
            os.makedirs(os.path.join(args.out_dir, c.name))
    max_position = max(step_inputs(c).max_position for c in cases)
    t0 = time.time()
    runner = _PrefillRunner(args.layers, global_layers, rank, max_position)
    if rank == 0:
        print(f"MK_PREFILL_NUMERICS_SETUP weights {time.time() - t0:.1f}s", flush=True)
    dist.barrier()
    for case in cases:
        t0 = time.time()
        exact = [f for cn, f in DEFAULT_EXACT if cn == case.name]
        run_case(runner, case, rank, args.out_dir, exact)
        if rank == 0:
            print(
                f"MK_PREFILL_NUMERICS_CASE {case.name} {time.time() - t0:.1f}s",
                flush=True,
            )
    ok = True
    if rank == 0:
        ok = summarize(cases, args.layers, args.out_dir)
        print(
            "MK_PREFILL_NUMERICS_PASS" if ok else "MK_PREFILL_NUMERICS_FAIL", flush=True
        )
    print(f"MK_PREFILL_NUMERICS_DONE rank={rank}", flush=True)
    dist.barrier()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
