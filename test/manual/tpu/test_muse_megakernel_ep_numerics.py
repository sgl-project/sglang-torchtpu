"""Numerics of the EP + sequence-parallel Muse decode step (muse_megakernel/kernel_ep.py)
against test_muse_megakernel_numerics.py's CPU reference: the same seeded weights, inputs,
teacher-forced per-layer checks and tolerances. One JAX process drives every device:
    python3 test/manual/tpu/test_muse_megakernel_ep_numerics.py --cases salfp8_c2q8_s
Prints one MK_EP_NUMERICS line per (case, layer) and MK_EP_NUMERICS_PASS when every check
holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Every rounding point as written: XLA may otherwise drop f32 -> fp8 -> f32 round trips.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_muse_megakernel_numerics as ref  # noqa: E402

from sglang_tpu.muse_megakernel import kernel_ep  # noqa: E402

TP = ref.TP
CASES = {
    **{
        n: ref.CASES[n]
        for n in (
            "salfp8_c1q1_s",
            "salfp8_c2q8_s",
            "salfp8_c2q8_l",
            "salfp8_c5q8_s",
            "salfp8_c16q8_s",
        )
    },
    "salfp8_c8q4_s": ref.Case(
        "salfp8_c8q4_s", "sal_fp8", 8, 4, tuple(1100 + 131 * i for i in range(8))
    ),
    "salfp8_c32q4_s": ref.Case(
        "salfp8_c32q4_s", "sal_fp8", 32, 4, tuple(900 + 97 * i for i in range(32))
    ),
}


def _np(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8).numpy()
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def _put(t: torch.Tensor, mesh: Mesh, spec: P, dtype=None) -> jax.Array:
    a = jnp.asarray(_np(t))
    if t.dtype == torch.float8_e4m3fn:
        a = jax.lax.bitcast_convert_type(a, kernel_ep.FP8)
    elif dtype is not None:
        a = a.astype(dtype)
    return jax.device_put(a, NamedSharding(mesh, spec))


def build_weights(mesh: Mesh, n_layers: int) -> tuple[dict, list]:
    """kernel_ep's weights and experts from the reference's per-(layer, rank) seeds."""
    rep_p, head_p = P(), P(kernel_ep.AXIS)
    w: dict[str, list] = {
        n: []
        for n in (
            *kernel_ep.REPLICATED_ROWS,
            "pre_expert_w",
            "post_expert_w",
            "expert_bias",
            "wrouter",
            "wpool",
            "wpool_s",
            "wunpool",
            "wunpool_s",
            "wqkvg",
            "wqkvg_s",
            "wo",
            "wo_s",
        )
    }
    experts = []
    kinds = ref._kinds(n_layers)
    for l in range(n_layers):
        t0 = time.time()
        rep = ref.replicated_layer(l)
        dense = [ref.dense_layer(l, d) for d in range(TP)]
        for name in (
            *kernel_ep.REPLICATED_ROWS,
            "pre_expert_w",
            "post_expert_w",
            "expert_bias",
        ):
            w[name].append(_put(rep[name], mesh, rep_p))
        w["wrouter"].append(_put(rep["wrouter"], mesh, rep_p, jnp.bfloat16))
        for name in (
            "wpool",
            "wunpool",
        ):  # replicated in full: the ranks' column slices
            w[name].append(
                _put(torch.cat([d[name] for d in dense], dim=1), mesh, rep_p)
            )
            w[name + "_s"].append(
                _put(torch.cat([d[name + "_s"] for d in dense]), mesh, rep_p)
            )
        for name in ("wqkvg", "wo"):
            w[name].append(_put(torch.stack([d[name] for d in dense]), mesh, head_p))
            w[name + "_s"].append(
                _put(torch.stack([d[name + "_s"] for d in dense]), mesh, head_p)
            )
        slabs = [
            [ref.expert_slabs(l, d, e, kinds[l]) for d in range(TP)]
            for e in range(ref.N_EXPERTS)
        ]
        experts.append(
            tuple(
                _put(
                    torch.stack(
                        [torch.stack([s[d][j] for d in range(TP)]) for s in slabs]
                    ),
                    mesh,
                    head_p,
                    jnp.bfloat16 if j % 2 else None,
                )
                for j in range(4)
            )
        )
        del slabs
        print(f"weights layer {l}: {time.time() - t0:.0f}s", flush=True)
    return w, experts


def new_kv(case: ref.Case, st: ref.StepInputs, pools: list, global_layers: tuple):
    """This step's K/V rows read back from every rank's pools: [L, tp, n, kvh, 2, hd]."""
    ps = ref._PAGES[case.kv]
    r = torch.arange(st.n_live)
    pos = st.pos[: st.n_live].long()
    out = []
    for l, pool in enumerate(pools):
        tab = st.pf if l in global_layers else st.psw
        page = tab[(r // case.q) * st.pps + pos // ps].long()
        p = torch.from_numpy(np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8)))
        x = p[
            :, page, :, :, :, pos % ps
        ]  # split advanced indices go first: [n, tp, ...]
        x = x.permute(1, 0, 2, 3, 4).contiguous().view(torch.float8_e4m3fn).float()
        out.append(x.reshape(TP, st.n_live, ref.KV_HEADS, 2, ref.HEAD_DIM))
    return torch.stack(out)


def run_case(
    case: ref.Case,
    mesh: Mesh,
    weights: dict,
    experts: list,
    n_layers: int,
    global_layers: tuple,
) -> list[dict]:
    assert case.kv == "sal_fp8", "kernel_ep reads the fp8 SEQ_ALONG_LANE pool only"
    st = ref.step_inputs(case)
    cfg = kernel_ep.EPConfig(
        tp=TP,
        n_layers=n_layers,
        global_layers=global_layers,
        kinds=ref._kinds(n_layers),
        q=case.q,
        n_req=case.n_req,
        rows=st.rows,
        page_size=ref._PAGES[case.kv],
        pages_per_seq=st.pps,
    )
    rp = P()
    acts = {
        "n_live": jax.device_put(jnp.int32(st.n_live), NamedSharding(mesh, rp)),
        "pos": _put(st.pos, mesh, rp),
        "pf": _put(st.pf, mesh, rp),
        "psw": _put(st.psw, mesh, rp),
        "emb": _put(st.emb, mesh, rp, jnp.bfloat16),
        "rope": _put(ref.rope_table(st.max_position), mesh, rp),
        "final_gain": _put(ref.final_gain(), mesh, rp),
    }
    pools = []
    for l in range(n_layers):
        pages = ref.n_pool_pages(case, st, l, global_layers)
        pools.append(
            _put(
                torch.stack(
                    [
                        ref.pool_layout(ref.kv_logical(case, l, d, pages), case.kv)
                        for d in range(TP)
                    ]
                ),
                mesh,
                P(kernel_ep.AXIS),
            )
        )
    t0 = time.time()
    hid, aux, pools = kernel_ep.step_fn(cfg, mesh)(acts, weights, experts, pools)
    hid = torch.from_numpy(np.asarray(hid.astype(jnp.float32))).to(torch.bfloat16)
    aux = torch.from_numpy(np.asarray(aux.astype(jnp.float32))).to(torch.bfloat16)
    t_kernel = time.time() - t0
    kv = new_kv(case, st, pools, global_layers)
    n = st.n_live
    results = []
    for l in range(n_layers):
        a = aux.float()
        stream_in = (
            ref._bf(ref._rms(st.emb[:n].float(), ref.RMS_EPS))
            if l == 0
            else a[:n, (l - 1) * ref.HIDDEN : l * ref.HIDDEN]
        )
        res = ref.check_layer(
            l,
            case,
            st,
            n_layers,
            global_layers,
            "shifted",
            stream_in,
            a[:n, l * ref.HIDDEN : (l + 1) * ref.HIDDEN],
            kv[l],
        )
        res["kernel_s"] = round(t_kernel, 1)
        results.append(res)
        print("MK_EP_NUMERICS " + json.dumps(res), flush=True)
    fin = ref.check_final(case, st, n_layers, hid, aux)
    print("MK_EP_NUMERICS_FINAL " + json.dumps(fin), flush=True)
    return [*results, fin]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--global-layers", default="1,4")
    args = ap.parse_args()
    global_layers = tuple(int(x) for x in args.global_layers.split(","))
    devices = sorted(jax.devices(), key=lambda d: d.id)[:TP]
    mesh = Mesh(np.asarray(devices), (kernel_ep.AXIS,))
    weights, experts = build_weights(mesh, args.layers)
    ok = True
    for name in args.cases:
        ok &= all(
            r["ok"]
            for r in run_case(
                CASES[name], mesh, weights, experts, args.layers, global_layers
            )
        )
    print("MK_EP_NUMERICS_PASS" if ok else "MK_EP_NUMERICS_FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
