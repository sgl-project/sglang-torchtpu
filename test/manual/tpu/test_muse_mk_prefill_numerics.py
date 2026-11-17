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
import math
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

from sglang_tpu.envs import envs as tpu_envs  # noqa: E402

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
# Transposed RPA prefill tiles (SGLANG_TPU_RPA_TRANSPOSED): the fp8 MXU accumulates Q^T K at
# ~bf16 precision even with the hi/lo Q^T split; measured means 3.5% / 3.7% (pf_c1q384).
_TRANSPOSED_MEAN_REL = 4e-2


class Case(msgspec.Struct, frozen=True):
    name: str
    n_req: int
    q: int  # chunk tokens per request
    ctx: tuple  # cached tokens per request
    expert_group: int = 0
    fp8_gather: bool = False
    # M.Case fields the reference reads
    kv: str = "sal_fp8"
    # padded rows' multiple (16: decode-shaped SP steps, prefill_config's SP_ROW_ALIGN)
    row_align: int = 128


CASES = {
    c.name: c
    for c in (
        # one request, its chunk filling the 384-row bucket
        Case("pf_c1q384", 1, 384, (1500,)),
        # two requests and 64 padded rows; a context below one window
        Case("pf_c2q160_pad", 2, 160, (2600, 700), expert_group=2),
        # past one 2048-token window on both requests, e4m3 gathers
        Case("pf_c2q192_fp8", 2, 192, (5000, 2300), fp8_gather=True),
        # the served chunk: expert TP routes 4096 x 8 x 8 rows, past the fused-RS SMEM budget
        # unless each rank keeps only its own rows there
        Case("pf_c1q4096", 1, 4096, (3000,)),
        # the served chunk of a fresh request (context 0): the prefill latency baseline
        Case("pf_c1q4096_ctx0", 1, 4096, (0,)),
        # the same chunk after 128k cached tokens
        Case("pf_c1q4096_ctx128k", 1, 4096, (131072,)),
    )
}
# decode-shaped steps (one new token per request, 4096 cached) for the SP + EP decode target
# (--sp with SGLANG_TPU_MUSE_MK_SP_O_RS=1): R rows = R requests
CASES.update(
    {
        f"dec_r{r}": Case(f"dec_r{r}", r, 1, (4096,) * r, row_align=16)
        for r in (16, 32, 48, 64, 128)
    }
)
# (case, flag): reruns that must be bit-identical to the case's run
DEFAULT_EXACT = (("pf_c1q384", "repeat"), ("pf_c1q384", "expert_group=2"))


def step_inputs(case: Case) -> M.StepInputs:
    ps = M._PAGES[case.kv]
    n_live = case.n_req * case.q
    rows = -(-n_live // case.row_align) * case.row_align
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


def _sp_expert_refs(l: int, rank: int, kind: str, device: torch.device) -> list:
    """This rank's E / tp whole experts as the SP program reads them: every rank's slices
    side by side (w13: all gate columns, then all up columns, the halves gmm_v2's fused silu
    reads; w2's rows), u32 codes along K, scales [E / tp, K / block, 1, N] (fp4:
    SGLANG_TPU_MUSE_MK_QBLOCK blocks; fp8: w13 per column, w2 per column of each rank's
    INTER_LOCAL rows)."""
    il = M.INTER_LOCAL
    el = M.N_EXPERTS // TP

    def halves(gate, up):
        return torch.cat([gate, up], dim=1)

    out = [[], [], [], []]
    for e in range(rank * el, (rank + 1) * el):
        sl = [M.tp_expert_slabs(l, d, e, kind) for d in range(TP)]
        w13 = halves(
            torch.cat([s[0][:, :il] for s in sl], dim=1),
            torch.cat([s[0][:, il:] for s in sl], dim=1),
        )
        w2 = torch.cat([s[2] for s in sl], dim=0)
        if kind == "fp4":
            s13 = halves(
                torch.cat([s[1][:, :il] for s in sl], dim=1),
                torch.cat([s[1][:, il:] for s in sl], dim=1),
            )
            s2 = torch.cat([s[3] for s in sl], dim=0)
        else:
            s13 = halves(
                torch.cat([s[1][0, :il] for s in sl])[None],
                torch.cat([s[1][0, il:] for s in sl])[None],
            )
            s2 = torch.stack([s[3][0] for s in sl])
        for j, t in enumerate((w13, s13[:, None], w2, s2[:, None])):
            out[j].append(t)
    return [torch.stack(t).to(device) for t in out]


def _sp_replicate_pool(runner, n_layers: int, pool: bool = True) -> None:
    """wpool / wunpool / wo replicated: every rank's slices side by side, their scales too."""
    from sglang_tpu.muse_megakernel.common.consts import SMALL
    from sglang_tpu.muse_megakernel.common.scratch import WEIGHT_NAMES

    names = list(WEIGHT_NAMES)
    sm = runner.weights[names.index("smalls")].cpu()
    for name in ("wpool", "wunpool") if pool else ():
        full = []
        for l in range(n_layers):
            ranks = [M.dense_layer(l, d) for d in range(TP)]
            full.append(torch.cat([r[name] for r in ranks], dim=1))
            sc = torch.cat([r[name + "_s"] for r in ranks])
            sm[l, SMALL[name + "_s"]] = 0
            sm[l, SMALL[name + "_s"], : sc.numel()] = sc
        runner.weights[names.index(name)] = torch.stack(full).to(runner.device)
    # wo whole (the SP o-proj runs locally on every head): every rank's rows stacked, requantized
    # to one per-output-channel scale (each rank's slice had its own). The decode target
    # (SGLANG_TPU_MUSE_MK_SP_O_RS) reads its row block of the same tensor.
    full = []
    for l in range(n_layers):
        ranks = [M.dense_layer(l, d) for d in range(TP)]
        w = torch.cat([r["wo"].float() * r["wo_s"][None] for r in ranks])
        sc = w.abs().amax(0).clamp_min(1e-30) / 448.0
        full.append((w / sc).to(torch.float8_e4m3fn))
        sm[l, SMALL["wo_s"]] = 0
        sm[l, SMALL["wo_s"], : sc.numel()] = sc
    runner.weights[names.index("wo")] = torch.stack(full).to(runner.device)
    runner.weights[names.index("smalls")] = sm.to(runner.device)


def _deq_fp4(words: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """u32 words [E, K/8, N] of e2m1 codes along K, block scales [E, K / block, N] -> f32."""
    dev = words.device
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)[None, None, :, None]
    codes = (words[:, :, None, :] >> shifts) & 0xF
    e, kw, _, n = codes.shape
    v = M._E2M1.to(dev)[codes.long()].reshape(e, kw * 8, n)
    return v * scale.float().repeat_interleave(kw * 8 // scale.shape[1], dim=1)


def _deq_fp8(words: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """u32 words [E, K/4, N] of e4m3 codes along K, column scales [E, rows, N] (row 0) -> f32."""
    dev = words.device
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=dev)[None, None, :, None]
    b = ((words[:, :, None, :] >> shifts) & 0xFF).to(torch.uint8)
    e, kw, _, n = b.shape
    w = b.view(torch.float8_e4m3fn).float().reshape(e, kw * 4, n)
    return w * scale[:, :1].float()


def _exact_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """f32 a @ b for bf16-valued a, b, exact whatever the device's f32 matmul precision: b is
    split into bf16 hi + lo parts, each product of bf16 values exact in f32."""
    hi = b.to(torch.bfloat16).float()
    return a @ hi + a @ (b - hi).to(torch.bfloat16).float()


class _DeviceRef:
    """The teacher-forced reference layer on the ranks' TPUs: each rank computes its own head /
    column / expert slices from its resident weights, and dist sums or gathers them, mirroring
    M._LayerRef's bf16 rounding points (prefill: bf16 attention I/O, shifted softmax)."""

    EXPERT_CHUNK = 32  # experts dequantized per batched matmul

    def __init__(self, base, n_layers: int):
        from sglang_tpu.muse_megakernel.common.scratch import WEIGHT_NAMES

        self.base, self.dev, self.rank = base, base.device, base.rank
        self.w = dict(zip(WEIGHT_NAMES, base.weights))
        # this rank's slice of every expert, natural order (block-256 fp4 under QBLOCK)
        self.experts = [
            [
                torch.stack(t).to(self.dev)
                for t in zip(
                    *(M.tp_expert_slabs(l, base.rank, e, k) for e in range(M.N_EXPERTS))
                )
            ]
            for l, k in enumerate(base.kinds)
        ]

    def _row(self, l: int, name: str, n: int | None = None):
        from sglang_tpu.muse_megakernel.common.consts import SMALL

        r = self.w["smalls"][l][SMALL[name]]
        return r if n is None else r[:n]

    def _mv(self, x, l: int, name: str, n: int | None = None):
        w = self.w[name][l]
        return (M._bf(x) @ w.float()) * self._row(l, name + "_s", n or w.shape[1])

    @staticmethod
    def _sum(x):
        dist.all_reduce(x)
        return x

    @staticmethod
    def _cat_cols(x):
        parts = [torch.empty_like(x) for _ in range(TP)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=1)

    def _attention(self, l, case, st, qn, kb, vb, kv, is_global):
        q, ps = case.q, M._PAGES[case.kv]
        g = M.Q_HEADS // M.KV_HEADS
        outs = []
        for j in range(case.n_req):
            rows = slice(j * q, (j + 1) * q)
            pos = st.pos[rows].long()
            p_lo = int(pos[0])
            first = 0 if is_global else max(p_lo - M.WINDOW + 1, 0)
            kpos = torch.arange(first, p_lo)
            tab = st.pf if is_global else st.psw
            page = tab[j * st.pps + kpos // ps].long()
            cached = kv[page.to(self.dev), (kpos % ps).to(self.dev)]
            hd = M.HEAD_DIM
            keys = torch.cat([cached[:, :, 0], kb[rows].reshape(q, M.KV_HEADS, hd)])
            vals = torch.cat([cached[:, :, 1], vb[rows].reshape(q, M.KV_HEADS, hd)])
            allpos = torch.cat([kpos, pos]).to(self.dev)
            posd = pos.to(self.dev)
            live = allpos[None, :] <= posd[:, None]
            if not is_global:
                live &= posd[:, None] - allpos[None, :] < M.WINDOW
            qs = M._bf(qn[rows].reshape(q, M.KV_HEADS, g, hd))
            sc = torch.einsum("thgd,nhd->thgn", qs, keys) * M.ATTN_SCALE
            sc = sc.masked_fill(~live[:, None, None, :], -math.inf)
            pr = torch.exp(sc - sc.amax(-1, keepdim=True))
            acc = torch.einsum("thgn,nhd->thgd", M._bf(pr), vals)
            outs.append((acc / pr.sum(-1, keepdim=True)).reshape(q, -1))
        return torch.cat(outs)

    def _experts(self, l: int, xe, ids):
        """[n, top_k, He] bf16 values: per (row, slot) the sum over ranks of each rank's slice
        output, every slice rounded to bf16 first."""
        n, k_top = ids.shape
        he, il = M.EXPERT_HIDDEN, M.INTER_LOCAL
        w13w, s13, w2w, s2 = self.experts[l]
        deq = _deq_fp4 if self.base.kinds[l] == "fp4" else _deq_fp8
        flat = ids.reshape(-1).cpu()
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=M.N_EXPERTS)
        starts = torch.cumsum(counts, 0) - counts
        xe_pad = torch.cat([xe, torch.zeros(1, he, device=self.dev)])
        y = torch.zeros(n * k_top, he, device=self.dev)
        ch = self.EXPERT_CHUNK
        for c0 in range(0, M.N_EXPERTS, ch):
            cnt = counts[c0 : c0 + ch]
            top = int(cnt.max())
            if top == 0:
                continue
            cap = max(8, 1 << (top - 1).bit_length())
            pair = torch.full((ch, cap), n * k_top, dtype=torch.long)
            for i in range(ch):
                c = int(cnt[i])
                pair[i, :c] = order[int(starts[c0 + i]) : int(starts[c0 + i]) + c]
            pair_d = pair.to(self.dev)
            x = xe_pad[torch.div(pair_d, k_top, rounding_mode="floor").clamp(max=n)]
            w13 = deq(w13w[c0 : c0 + ch], s13[c0 : c0 + ch])
            w2 = deq(w2w[c0 : c0 + ch], s2[c0 : c0 + ch])
            gu = torch.bmm(x, w13)
            gate, up = gu[..., :il], gu[..., il:]
            yd = M._bf(torch.bmm(M._bf(gate * torch.sigmoid(gate) * up), w2))
            valid = pair < n * k_top
            y[pair_d[valid.to(self.dev)]] = yd[valid.to(self.dev)]
        return M._bf(self._sum(y)).reshape(n, k_top, he)

    def _ffn_tail(self, l, stream, y, sc, ids):
        he = M.EXPERT_HIDDEN
        ynorm = M._bf(M._rms(y * self._row(l, "post_expert_w", he), M.POST_EPS))
        w = torch.gather(sc, 1, ids)
        w = w / (w.sum(1, keepdim=True) + M.ROUTER_EPS) * M.ROUTE_SCALE
        u = M._bf((ynorm * w[:, :, None]).sum(1))
        f = self._cat_cols(M._bf(self._mv(u, l, "wunpool")))
        return self._row(l, "ffn_alpha") * stream + self._row(l, "ffn_beta") * M._bf(
            M._rms(f * self._row(l, "post_ffn_w"), M.POST_EPS)
        )

    def check_layer(self, l, case, st, global_layers, stream_in, out, new_kv) -> dict:
        """M.check_layer's result dict for one layer, on every rank (collectives inside)."""
        t0 = time.time()
        n, dev = st.n_live, self.dev
        is_global = l in global_layers
        qd, kd, hd = M.Q_HEADS * M.HEAD_DIM, M.KV_HEADS * M.HEAD_DIM, M.HEAD_DIM
        x_in = stream_in.to(dev)
        x = M._bf(M._rms(x_in, M.RMS_EPS) * self._row(l, "pre_attn_w"))
        qkvg = self._mv(x, l, "wqkvg")
        q, k = qkvg[:, :qd], qkvg[:, qd : qd + kd]
        v, og = qkvg[:, qd + kd : qd + 2 * kd], qkvg[:, qd + 2 * kd :]
        qn = M._heads_rms(q, M.Q_HEADS) * M.SCALE_QUERY_BY
        kn = M._heads_rms(k, M.KV_HEADS)
        if not is_global:
            rope = M.rope_table(st.max_position)[st.pos[:n].long()].to(dev)
            qn, kn = M._rope(qn, M.Q_HEADS, rope), M._rope(kn, M.KV_HEADS, rope)
        kn, v = M._bf(kn), M._bf(v)
        kv_dt = torch.float8_e4m3fn if case.kv == "sal_fp8" else torch.bfloat16
        kb = kn.to(kv_dt).float().to(torch.bfloat16).float()
        vb = v.to(kv_dt).float().to(torch.bfloat16).float()
        pages = M.n_pool_pages(case, st, l, global_layers)
        kv = M.kv_logical(case, l, self.rank, pages).to(dev)
        out_att = self._attention(l, case, st, qn, kb, vb, kv, is_global)
        attn = M._bf(M._heads_rms(M._bf(out_att), M.Q_HEADS))
        gated = M._bf(torch.sigmoid(M._bf(og)) * attn)
        a = M._bf(self._sum(M._bf(self._mv(gated, l, "wo"))))
        mid = self._row(l, "attn_alpha") * x_in + self._row(l, "attn_beta") * M._bf(
            M._rms(a, M.POST_EPS)
        )
        ffn_bf = M._bf(M._rms(mid, M.RMS_EPS, self._row(l, "pre_ffn_w")))
        pooled = self._cat_cols(M._bf(self._mv(ffn_bf, l, "wpool")))
        xe = M._bf(
            M._rms(pooled, M.RMS_EPS, self._row(l, "pre_expert_w", M.EXPERT_HIDDEN))
        )
        wr = self.w["wrouter"][l].float()
        sc = torch.sigmoid(_exact_dot(ffn_bf, wr) * M.ROUTER_MULT)
        biased = sc + self._row(l, "expert_bias", wr.shape[1])
        ids = torch.topk(biased, M.TOP_K, dim=1).indices
        want = self._ffn_tail(l, mid, self._experts(l, xe, ids), sc, ids)
        outd = out.to(dev)
        rel = M._row_rel(outd, want).cpu()
        forced = int(l > 0)
        tol_row = M.TOL_ROW_REL[forced]
        flips, unexplained = [], []
        biased_c, ids_c = biased.cpu(), ids.cpu()
        # every rank sees the same rows over tolerance: one batched collective pass for all
        # their near-tie alternatives keeps the ranks in step
        bad = torch.nonzero(rel > tol_row).flatten().tolist()
        cand = [
            (r, alt) for r in bad for alt in M._near_tie_swaps(biased_c[r], ids_c[r])
        ]
        best: dict = {}
        if cand:
            rows = torch.tensor([r for r, _ in cand], device=dev)
            ids2 = torch.stack([alt for _, alt in cand]).to(dev)
            w2 = self._ffn_tail(
                l, mid[rows], self._experts(l, xe[rows], ids2), sc[rows], ids2
            )
            e2 = M._row_rel(outd[rows], w2).cpu()
            for (r, _), e in zip(cand, e2.tolist()):
                best[r] = min(best.get(r, e), e)
        for r in bad:
            if r in best and best[r] <= tol_row:
                flips.append(r)
                rel[r] = best[r]
            else:
                unexplained.append(r)
        new_ref = torch.stack(
            [kb.reshape(-1, M.KV_HEADS, hd), vb.reshape(-1, M.KV_HEADS, hd)], 2
        )
        kv_rel = M._row_rel(new_kv.to(dev).reshape(n, -1), new_ref.reshape(n, -1))
        kv_max = kv_rel.max().reshape(1)
        dist.all_reduce(kv_max, op=dist.ReduceOp.MAX)
        res = {
            "case": case.name,
            "layer": l,
            "attn": "global" if is_global else "swa",
            "experts": self.base.kinds[l],
            "rows": n,
            "max_row_rel": float(rel.max()),
            "mean_row_rel": float(rel.mean()),
            "ref_rms": float(want.pow(2).mean().sqrt()),
            "kv_max_rel": float(kv_max),
            "route_flips": len(flips),
            "route_unexplained": unexplained,
            "ref_s": round(time.time() - t0, 1),
        }
        res["ok"] = bool(
            not unexplained
            and res["max_row_rel"] <= tol_row
            and res["mean_row_rel"] <= M.TOL_MEAN_REL[forced]
            and res["kv_max_rel"] <= M.TOL_KV_REL[case.kv][forced]
            and bool(torch.isfinite(outd).all())
        )
        return res


class _PrefillRunner:
    """M._Runner's weights and pools, with the prefill program's fields on its fake
    MuseMegakernel."""

    def __init__(
        self,
        n_layers: int,
        global_layers: tuple,
        rank: int,
        max_position: int,
        sp: bool = False,
        with_ref: bool = True,
        capture: tuple | None = None,
    ):
        from sglang_tpu.muse_megakernel import ops

        self.ops = ops
        self.base = M._Runner(ops, n_layers, global_layers, rank, max_position)
        self.sp = sp
        # aux layers the step returns (numerics checks every layer; all by default)
        self.capture = list(range(n_layers)) if capture is None else list(capture)
        # built before SP replaces the pool / unpool weights: it reads every rank's slices
        self.ref = _DeviceRef(self.base, n_layers) if with_ref else None
        # --sp-ep-layout: the base runner already holds decode EP's layout (M._EP: whole
        # experts, replicated pool / unpool in kperm order); only wo is replicated here
        ep_layout = sp and M._EP
        for l, kind in enumerate(self.base.kinds):
            if ep_layout:
                continue
            if sp:
                self.base.per_layer_experts[l] = _sp_expert_refs(
                    l, rank, kind, self.base.device
                )
            elif kind == "fp8":
                self.base.per_layer_experts[l] = _fp8_serving_refs(
                    l, rank, self.base.device
                )
        if sp:
            _sp_replicate_pool(self.base, n_layers, pool=not ep_layout)
        mk = self.base.mk
        mk.moe_mode = "ep" if sp else "tp"
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

    def _compiled(self, case: Case, st: M.StepInputs, expert_group: int):
        """(a freshly compiled prefill step, its device inputs before the pools)."""
        self.base.mk.prefill_fp8_gather = case.fp8_gather
        self.last_cfg = None
        experts = self.base.experts(0 if self.sp else expert_group)
        d = self.base.device
        n = case.n_req
        meta = torch.tensor([st.n_live, self.base.rank], dtype=torch.int32)
        kv_lens = torch.tensor([c + case.q for c in case.ctx], dtype=torch.int32)
        cu_q = torch.arange(n + 1, dtype=torch.int32) * case.q
        # RPA's [mixed, prefill, total] cumsum: decode-shaped steps (q = 1) are decode-only
        dist_ = torch.tensor([0, 0, n] if case.q == 1 else [n, n, n], dtype=torch.int32)
        capture = self.capture
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
        inputs = [
            t.to(d) for t in (meta, st.pos, kv_lens, cu_q, dist_, st.pf, st.psw, st.emb)
        ]
        compiled = torch.compile(run, fullgraph=True, dynamic=False)
        return compiled, [*inputs, self.base.weights, experts]

    def step(self, case: Case, st: M.StepInputs, kv: list, expert_group: int):
        """(hidden [rows, H] bf16, aux [rows, L * H] bf16); kv updated in place."""
        compiled, args = self._compiled(case, st, expert_group)
        hid, aux = compiled(*args, kv)
        if self.last_cfg is None:
            raise RuntimeError(
                "the step ran a cached executable: the op was not traced"
            )
        return hid.cpu(), aux.cpu()

    def bench(
        self, case: Case, st: M.StepInputs, kv: list, iters: int, profile_dir: str = ""
    ) -> float:
        """Mean wall ms of one prefill step after its compile and one warm step (each step
        rewrites the chunk's K/V in place); profile_dir: rank 0 then captures 3 more steps
        in xprof there."""
        compiled, args = self._compiled(case, st, case.expert_group)
        compiled(*args, kv)
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            compiled(*args, kv)
        torch.accelerator.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / iters
        if profile_dir:
            from pathlib import Path

            from sglang_tpu.profiler import start_tpu_profile, stop_tpu_profile

            trace = None
            if self.base.rank == 0:
                trace = start_tpu_profile(
                    output_dir=Path(profile_dir) / case.name, with_cpu=False
                )
            for _ in range(3):
                compiled(*args, kv)
            torch.accelerator.synchronize()
            if trace is not None:
                stop_tpu_profile(trace)
        return ms


def _set_tolerances(fp8_gather: bool) -> None:
    scale = _FP8_GATHER_TOL_SCALE if fp8_gather else 1.0
    M.TOL_ROW_REL = tuple(t * scale for t in _TOL_ROW_REL)
    M.TOL_MEAN_REL = tuple(t * scale for t in _TOL_MEAN_REL)
    if tpu_envs.SGLANG_TPU_RPA_TRANSPOSED.get():
        M.TOL_MEAN_REL = tuple(max(t, _TRANSPOSED_MEAN_REL) for t in M.TOL_MEAN_REL)
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
    # SP runs e4m3 activations (the fp8 gathers' rounding, and more): their scale too
    _set_tolerances(case.fp8_gather or runner.sp)
    L = runner.base.n_layers
    aux0 = data[0]["aux"].float()
    for l in range(L):
        stream_in = (
            M._bf(M._rms(st.emb[:n].float(), M.RMS_EPS))
            if l == 0
            else aux0[:n, (l - 1) * M.HIDDEN : l * M.HIDDEN]
        )
        # every rank runs every layer: the reference sums / gathers the ranks' slices
        res = runner.ref.check_layer(
            l,
            case,
            st,
            runner.base.global_layers,
            stream_in,
            aux0[:n, l * M.HIDDEN : (l + 1) * M.HIDDEN],
            new_kv[l],
        )
        res["kernel_s"] = round(t_kernel, 1)
        if rank == 0:
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
    # default: every case the program runs (the dec_* steps are SP-only)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--global-layers", default="1,4")
    ap.add_argument("--out-dir", default="/tmp/mk_prefill_numerics")
    # the sequence-parallel program (EP MoE on whole experts, replicated pool / unpool)
    ap.add_argument("--sp", action="store_true")
    # timing only: compile each case once, then time this many steps (no numerics)
    ap.add_argument("--bench-iters", type=int, default=0)
    # with --bench-iters: an xprof capture of 3 more steps per case (rank 0) under this dir
    ap.add_argument("--profile-dir", default="")
    # with --bench-iters: aux layers to return, comma-separated (empty: none); default all
    ap.add_argument("--capture", default=None)
    # SP on decode EP's weight layout (SGLANG_TPU_MUSE_MK_SP_EP_LAYOUT): the layout prefill SP
    # and both decode paths share; timing only for now (the reference reads TP slices)
    ap.add_argument("--sp-ep-layout", action="store_true")
    # debug: build the runner (weights, pools) and exit before any prefill step
    ap.add_argument("--setup-only", action="store_true")
    args = ap.parse_args()
    if args.capture is not None and not args.bench_iters:
        raise ValueError("--capture is for --bench-iters; numerics checks every layer")
    if args.sp:
        os.environ["SGLANG_TPU_MUSE_MK_PREFILL_SP"] = "1"
    if args.sp_ep_layout:
        if not (args.sp and args.bench_iters):
            raise ValueError("--sp-ep-layout needs --sp and --bench-iters")
        os.environ["SGLANG_TPU_MUSE_MK_SP_EP_LAYOUT"] = "1"
        M._EP = True
    if M.tp_qblock() != 256:
        raise ValueError(
            "needs SGLANG_TPU_MUSE_MK_QBLOCK=256 (served block-256 experts)"
        )
    M._QBLOCK_REF = M._SERVED = True
    # the served prefill attention tiles (the recipe's); override one with =0 to A/B it
    for k in (
        "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK",
        "SGLANG_TPU_RPA_EXP2_SOFTMAX",
        "SGLANG_TPU_RPA_ONES_V",
        "SGLANG_TPU_RPA_TRANSPOSED",
    ):
        os.environ.setdefault(k, "1")
    M._LayerRef.bf16_attention_io = True
    # the reference's f32 dots run on the TPU (_DeviceRef): keep them f32
    torch.set_float32_matmul_precision("highest")

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    if dist.get_world_size() != TP:
        raise RuntimeError(f"needs {TP} processes, got {dist.get_world_size()}")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // TP))
    torch._dynamo.config.cache_size_limit = 256
    torch.compiler.config.force_disable_caches = True

    names = args.cases or [n for n in CASES if args.sp or not n.startswith("dec_")]
    cases = [CASES[c] for c in names]
    global_layers = tuple(int(x) for x in args.global_layers.split(",") if x)
    if rank == 0:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        for c in cases:
            os.makedirs(os.path.join(args.out_dir, c.name))
    max_position = max(step_inputs(c).max_position for c in cases)
    t0 = time.time()
    runner = _PrefillRunner(
        args.layers,
        global_layers,
        rank,
        max_position,
        sp=args.sp,
        with_ref=not args.bench_iters,
        capture=None
        if args.capture is None
        else tuple(int(x) for x in args.capture.split(",") if x),
    )
    if rank == 0:
        print(f"MK_PREFILL_NUMERICS_SETUP weights {time.time() - t0:.1f}s", flush=True)
    dist.barrier()
    if args.setup_only:
        return
    if args.bench_iters:
        for case in cases:
            st = step_inputs(case)
            ms = runner.bench(
                case,
                st,
                runner.base.pools(case, st),
                args.bench_iters,
                args.profile_dir,
            )
            if rank == 0:
                print(
                    "MK_PREFILL_BENCH "
                    + json.dumps(
                        {
                            "case": case.name,
                            "rows": st.rows,
                            "sp": args.sp,
                            "layers": args.layers,
                            "iters": args.bench_iters,
                            "ms_per_step": round(ms, 3),
                            "ms_per_layer": round(ms / args.layers, 3),
                        }
                    ),
                    flush=True,
                )
        dist.barrier()
        return
    for case in cases:
        t0 = time.time()
        exact = [
            f
            for cn, f in DEFAULT_EXACT
            if cn == case.name and not (args.sp and f.startswith("expert_group"))
        ]
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
