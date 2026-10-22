# SPDX-License-Identifier: Apache-2.0
"""Checkpoint -> megakernel layout, built per rank on the host (numpy).

The megakernel owns the weights when it is enabled, so tensors are read
straight from the safetensors shards into the kernel layout; nothing goes
through sglang's own (differently sharded) parameters.
"""

import json
import os
import struct

import ml_dtypes
import numpy as np

from sglang_tpu.k3_megakernel.config import MoeLayout

_ST_DTYPES = {
    "BF16": ml_dtypes.bfloat16,
    "F32": np.float32,
    "F16": np.float16,
    "U8": np.uint8,
    "I32": np.int32,
    "I64": np.int64,
}
# E2M1 code -> value; bit 3 is the sign. Even K elements sit in the low nibble.
_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)


class Checkpoint:
    """Memory-mapped reader over a sharded safetensors checkpoint."""

    def __init__(self, path: str, prefix: str = "language_model.model."):
        self._path, self._prefix = path, prefix
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}

    def _header(self, shard: str):
        if shard not in self._headers:
            with open(os.path.join(self._path, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (8 + n, json.loads(f.read(n)))
        return self._headers[shard]

    def get(self, name: str) -> np.ndarray:
        full = self._prefix + name
        shard = self._weight_map[full]
        base, header = self._header(shard)
        meta = header[full]
        start, end = meta["data_offsets"]
        dtype = np.dtype(_ST_DTYPES[meta["dtype"]])
        return np.memmap(
            os.path.join(self._path, shard),
            dtype=dtype,
            mode="r",
            offset=base + start,
            shape=tuple(meta["shape"]),
        )


def _unpack_fp4(packed: np.ndarray) -> np.ndarray:
    """[..., K/2] u8 -> [..., K] f32 E2M1 values (scales not applied)."""
    lo = _E2M1[packed & 0x0F]
    hi = _E2M1[packed >> 4]
    return np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def reblock_fp8_per_column(packed: np.ndarray, scale: np.ndarray):
    """MXFP4 ``[N, K/2]`` + E8M0 ``[N, K/32]`` -> e4m3 ``[K, N]`` and an f32
    power-of-two scale per column ``[N]``, losslessly.

    Column ``n`` uses exponent ``S = max_g e[n, g] - 6`` so the largest E2M1
    value (6 * 2**6 = 384) stays under the e4m3 max (448); every value
    fp4 * 2**d with d >= -8 is exact in e4m3 (smallest subnormal 2**-9).
    Raises if a column's exponents span more than 14.
    """
    e = scale.astype(np.int32)
    col_exp = e.max(axis=1) - 6
    shift = e - col_exp[:, None]
    if shift.min() < -8:
        raise ValueError(
            f"MXFP4 exponent spread {(-shift.min()) + 6} too wide for a per-column e4m3 scale"
        )
    values = _unpack_fp4(packed)  # [N, K]
    k = values.shape[1]
    values = np.ldexp(
        values.reshape(values.shape[0], k // 32, 32), shift[:, :, None]
    ).reshape(values.shape)
    fp8 = values.astype(ml_dtypes.float8_e4m3fn)
    if not np.array_equal(fp8.astype(np.float32), values):
        raise ValueError("FP8 reblock is not exact")
    col_scale = np.ldexp(np.float32(1.0), col_exp - 127).astype(np.float32)
    return np.ascontiguousarray(fp8.T), col_scale


def _expert_gate_up(
    ckpt: Checkpoint, prefix: str, expert: int, lane: int, lane_inter: int
):
    """One expert's [K, 2 * lane_inter] e4m3 gate|up slice for ``lane`` and its column scales."""
    rows = slice(lane * lane_inter, (lane + 1) * lane_inter)
    outs, scales = [], []
    for w in ("w1", "w3"):
        packed = ckpt.get(f"{prefix}{expert}.{w}.weight_packed")[rows]
        scale = ckpt.get(f"{prefix}{expert}.{w}.weight_scale")[rows]
        fp8, col = reblock_fp8_per_column(np.asarray(packed), np.asarray(scale))
        outs.append(fp8)
        scales.append(col)
    return np.concatenate(outs, axis=1), np.concatenate(scales)[None, :]


def _expert_down(
    ckpt: Checkpoint, prefix: str, expert: int, lane: int, lane_inter: int
):
    """w2 K-slice for ``lane``: packed u8 ``[lane_inter/2, N]`` (sglang's gmm
    layout) and E8M0 exponents ``[lane_inter/32, N]``."""
    packed = ckpt.get(f"{prefix}{expert}.w2.weight_packed")
    scale = ckpt.get(f"{prefix}{expert}.w2.weight_scale")
    kp = slice(lane * lane_inter // 2, (lane + 1) * lane_inter // 2)
    ks = slice(lane * lane_inter // 32, (lane + 1) * lane_inter // 32)
    return np.ascontiguousarray(packed[:, kp].T), np.ascontiguousarray(scale[:, ks].T)


# Streamed dense projections that ``Relax.dense_fp8`` stores as e4m3 (every ``StreamPool`` use).
DENSE_FP8_FAMILIES = (
    "kda_proj",
    "kda_oproj",
    "mla_w1",
    "mla_qb",
    "mla_oproj",
    "dense_gu",
    "dense_down",
    "router",
    "latent_down",
    "latent_up",
    "shared_gu",
    "shared_down",
)
_E4M3_MAX = 448.0


def quantize_e4m3_columns(w: np.ndarray):
    """``[..., K, N]`` -> e4m3 ``[..., K, N]`` and f32 scales ``[..., 1, N]``, scale = amax_K |w| / 448
    (1 for all-zero columns), weight = round_to_nearest(w / scale)."""
    lead, (k, n) = w.shape[:-2], w.shape[-2:]
    q = np.empty(w.shape, ml_dtypes.float8_e4m3fn)
    scales = np.empty(lead + (1, n), np.float32)
    for idx in np.ndindex(*lead):
        m = np.asarray(w[idx], np.float32)
        amax = np.abs(m).max(axis=0, keepdims=True)
        sc = np.where(amax > 0, amax / np.float32(_E4M3_MAX), np.float32(1.0)).astype(
            np.float32
        )
        q[idx] = np.clip(m / sc, -_E4M3_MAX, _E4M3_MAX).astype(ml_dtypes.float8_e4m3fn)
        scales[idx] = sc
    return q, scales


def apply_dense_fp8(arrays: dict) -> dict:
    """Quantizes every ``DENSE_FP8_FAMILIES`` member of ``arrays`` in place of the bf16 weight
    and adds its ``<name>_s`` column scales."""
    for name in DENSE_FP8_FAMILIES:
        if name in arrays:
            arrays[name], arrays[name + "_s"] = quantize_e4m3_columns(arrays[name])
    return arrays


def load_moe_layer(
    ckpt: Checkpoint, layer: int, layout: MoeLayout, ranks, experts=None
) -> dict:
    """MoE block ``layer`` for ``ranks``; every array has a leading rank axis.

    ``experts`` (global ids) limits which routed experts are materialized;
    the others are zero, which is only valid if routing never selects them.
    ``experts=()`` leaves the routed-expert arrays out entirely. The expert
    arrays have ``layout.expert_slots`` slots (own experts, then replicas).
    """
    d, topo = layout.dims, layout.topo
    pre = f"layers.{layer}.block_sparse_moe."
    gate = ckpt.get(pre + "gate.weight")  # [E, H]
    bias = np.asarray(ckpt.get(pre + "gate.e_score_correction_bias"), np.float32)
    down = ckpt.get(pre + "routed_expert_down_proj.weight")  # [latent, H]
    up = ckpt.get(pre + "routed_expert_up_proj.weight")  # [H, latent]
    norm = np.asarray(ckpt.get(pre + "routed_expert_norm.weight"))
    sg = ckpt.get(pre + "shared_experts.gate_proj.weight")  # [6144, H]
    su = ckpt.get(pre + "shared_experts.up_proj.weight")
    sd = ckpt.get(pre + "shared_experts.down_proj.weight")  # [H, 6144]
    li, lc = layout.lane_inter, layout.latent_chunk
    sc = layout.shared_cols
    wanted = None if experts is None else set(int(e) for e in experts)

    out = {
        k: []
        for k in (
            "router",
            "router_bias",
            "latent_down",
            "latent_norm",
            "latent_up",
            "shared_gu",
            "shared_down",
            "expert_gu",
            "expert_gu_scale",
            "expert_down",
            "expert_down_scale",
        )
    }
    for r in ranks:
        host, lane, pair = r // topo.lanes, r % topo.lanes, r & 1
        rc, rp = layout.router_cols, layout.router_pad
        router = np.zeros((d.hidden, rp), ml_dtypes.bfloat16)
        router[:, :rc] = gate[pair * rc : (pair + 1) * rc].T
        out["router"].append(router)
        rb = np.full((1, 2 * rp), -np.inf, np.float32)  # padded columns never win
        rb[0, :rc], rb[0, rp : rp + rc] = bias[:rc], bias[rc:]
        out["router_bias"].append(rb)
        ld = np.zeros((d.hidden, lc), ml_dtypes.bfloat16)
        lo, hi = min(r * lc, d.latent), min((r + 1) * lc, d.latent)
        ld[:, : hi - lo] = down[lo:hi].T
        out["latent_down"].append(ld)
        out["latent_norm"].append(norm[None, :])
        uc = slice(lane * layout.latent_up_cols, (lane + 1) * layout.latent_up_cols)
        out["latent_up"].append(np.ascontiguousarray(up[uc].T))
        s = slice(r * sc, (r + 1) * sc)
        out["shared_gu"].append(np.concatenate([sg[s].T, su[s].T], axis=1))
        out["shared_down"].append(np.ascontiguousarray(sd[:, s].T))

        if experts is not None and len(experts) == 0:
            continue
        ns = layout.expert_slots
        gu = np.zeros((ns, d.latent, 2 * li), ml_dtypes.float8_e4m3fn)
        gus = np.zeros((ns, 1, 2 * li), np.float32)
        dn = np.zeros((ns, li // 2, d.latent), np.uint8)
        dns = np.zeros((ns, li // 32, d.latent), np.uint8)
        epre = pre + "experts."
        for j, e in enumerate(layout.slot_experts(host)):
            if wanted is not None and e not in wanted:
                continue
            gu[j], gus[j] = _expert_gate_up(ckpt, epre, e, lane, li)
            dn[j], dns[j] = _expert_down(ckpt, epre, e, lane, li)
        out["expert_gu"].append(gu)
        out["expert_gu_scale"].append(gus)
        out["expert_down"].append(dn)
        out["expert_down_scale"].append(dns)
    return {k: np.stack(v) for k, v in out.items() if v}


def load_dense_mlp(
    ckpt: Checkpoint, topo, channels: int, ranks, layer: int = 0
) -> dict:
    """Layer-0 dense MLP: gate|up ``[H, 2 * channels]`` and down ``[channels, H]``
    per rank; channels beyond the real split are zero."""
    pre = f"layers.{layer}.mlp."
    gate, up, down = (
        ckpt.get(pre + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj")
    )
    inter, hidden = gate.shape
    per = -(-inter // topo.world)
    out = {"dense_gu": [], "dense_down": []}
    for r in ranks:
        lo, hi = min(r * per, inter), min((r + 1) * per, inter)
        gu = np.zeros((hidden, 2 * channels), ml_dtypes.bfloat16)
        gu[:, : hi - lo] = gate[lo:hi].T
        gu[:, channels : channels + hi - lo] = up[lo:hi].T
        dn = np.zeros((channels, hidden), ml_dtypes.bfloat16)
        dn[: hi - lo] = down[:, lo:hi].T
        out["dense_gu"].append(gu)
        out["dense_down"].append(dn)
    return {k: np.stack(v) for k, v in out.items()}


def _out_proj(w_out_in, rank: int, k: int, pair: bool) -> np.ndarray:
    """Row-parallel output projection of ``rank`` from ``w_out_in`` ``[H, world * k]``:
    ``[k, H]`` (its input channels), or with ``pair`` (``stages.common.OutProjSpec``)
    ``[own k; peer k] x [H/2]`` over the column half ``rank & 1``."""
    if not pair:
        return np.ascontiguousarray(w_out_in[:, rank * k : (rank + 1) * k].T)
    half = w_out_in.shape[0] // 2
    rows = slice((rank & 1) * half, ((rank & 1) + 1) * half)
    parts = [w_out_in[rows, r * k : (r + 1) * k].T for r in (rank, rank ^ 1)]
    return np.ascontiguousarray(np.concatenate(parts, axis=0))


def load_kda_layer(
    ckpt: Checkpoint,
    layer: int,
    topo,
    ranks,
    heads: int = 96,
    head_dim: int = 128,
    pair_oproj: bool = False,
) -> dict:
    """KDA attention for ``ranks``; rank r owns heads ``[h*r, h*(r+1))`` (h = heads / world).

    proj  : [H, 4*hd + 256] bf16 = q|k|v|g (hd = h*128 each) | f_a (128) | b (h, padded to 128)
    f_b   : [128, hd] bf16;  conv: [4, 3*hd] f32 (tap 0 = oldest);  dt_bias, a_log: [1, hd] f32
    o_norm: [1, 128] f32 holding the bf16-rounded weight (sglang keeps it as a bf16 parameter)
    o_proj: [hd, H] bf16 (this rank's input channels; see ``_out_proj`` for ``pair_oproj``)
    """
    pre = f"layers.{layer}.self_attn."
    g = lambda n: ckpt.get(pre + n)
    h = heads // topo.world
    hd = h * head_dim
    fa = np.asarray(g("f_a_proj.weight"))
    a_log = np.asarray(g("A_log"), np.float32).reshape(-1)
    norm_w = (
        np.asarray(g("o_norm.weight"), np.float32)
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    out = {
        k: []
        for k in (
            "kda_proj",
            "kda_fb",
            "kda_conv",
            "kda_dt",
            "kda_alog",
            "kda_onorm",
            "kda_oproj",
        )
    }
    for r in ranks:
        cs = slice(r * hd, (r + 1) * hd)
        hs = slice(r * h, (r + 1) * h)
        proj = np.zeros((fa.shape[1], 4 * hd + 256), ml_dtypes.bfloat16)
        for i, n in enumerate(("q_proj", "k_proj", "v_proj", "g_proj")):
            proj[:, i * hd : (i + 1) * hd] = g(n + ".weight")[cs].T
        proj[:, 4 * hd : 4 * hd + 128] = fa.T
        proj[:, 4 * hd + 128 : 4 * hd + 128 + h] = g("b_proj.weight")[hs].T
        out["kda_proj"].append(proj)
        out["kda_fb"].append(np.ascontiguousarray(g("f_b_proj.weight")[cs].T))
        conv = np.concatenate(
            [
                np.asarray(g(n + ".weight"), np.float32)[cs, 0, :]
                for n in ("q_conv1d", "k_conv1d", "v_conv1d")
            ]
        )
        out["kda_conv"].append(np.ascontiguousarray(conv.T))
        out["kda_dt"].append(np.asarray(g("dt_bias"), np.float32)[cs][None])
        out["kda_alog"].append(np.repeat(a_log[hs], head_dim)[None])
        out["kda_onorm"].append(norm_w[None])
        out["kda_oproj"].append(_out_proj(g("o_proj.weight"), r, hd, pair_oproj))
    return {k: np.stack(v) for k, v in out.items()}


def load_mla_layer(
    ckpt: Checkpoint, layer: int, topo, ranks, heads: int = 96, pair_oproj: bool = False
) -> dict:
    """MLA attention for ``ranks``; rank r owns heads ``[h*r, h*(r+1))``.

    w1    : [H, 2176 + h*128] bf16 = q_a (1536) | kv_c (512) | k_pe (64) | 0 (64) | g_proj (this rank)
    qb    : [1536, 2*h*128] = q_nope (h*128) | q_pe (per head: 64 rope columns then 64 zeros)
    wkc   : [h, 128, 512]; wvc: [h, 512, 128]; oproj: [h*128, H]; norms f32 [1, 1536] / [1, 512]
    """
    pre = f"layers.{layer}.self_attn."
    g = lambda n: ckpt.get(pre + n)
    h = heads // topo.world
    hd = h * 128
    qa, kva = g("q_a_proj.weight"), g("kv_a_proj_with_mqa.weight")
    qb = np.asarray(g("q_b_proj.weight")).reshape(heads, 192, -1)
    kvb = np.asarray(g("kv_b_proj.weight")).reshape(heads, 256, -1)
    hidden = qa.shape[1]
    out = {
        k: []
        for k in (
            "mla_w1",
            "mla_qnorm",
            "mla_kvnorm",
            "mla_qb",
            "mla_wkc",
            "mla_wvc",
            "mla_oproj",
        )
    }
    for r in ranks:
        hs = slice(r * h, (r + 1) * h)
        w1 = np.zeros((hidden, 2176 + hd), ml_dtypes.bfloat16)
        w1[:, :1536] = qa[:].T
        w1[:, 1536:2112] = kva[:].T
        w1[:, 2176:] = g("g_proj.weight")[r * hd : (r + 1) * hd].T
        out["mla_w1"].append(w1)
        out["mla_qnorm"].append(np.asarray(g("q_a_layernorm.weight"), np.float32)[None])
        out["mla_kvnorm"].append(
            np.asarray(g("kv_a_layernorm.weight"), np.float32)[None]
        )
        pe = np.zeros((h, 128, qb.shape[2]), ml_dtypes.bfloat16)
        pe[:, :64] = qb[hs, 128:]
        out["mla_qb"].append(
            np.concatenate(
                [qb[hs, :128].reshape(hd, -1).T, pe.reshape(hd, -1).T], axis=1
            )
        )
        out["mla_wkc"].append(np.ascontiguousarray(kvb[hs, :128]))
        out["mla_wvc"].append(np.ascontiguousarray(kvb[hs, 128:].transpose(0, 2, 1)))
        out["mla_oproj"].append(_out_proj(g("o_proj.weight"), r, hd, pair_oproj))
    return {k: np.stack(v) for k, v in out.items()}


def fill_experts(
    ckpt: Checkpoint,
    layer: int,
    layout: MoeLayout,
    ranks,
    experts,
    out: dict,
    index: int,
) -> None:
    """Write routed experts ``experts`` (global ids) of ``layer`` into preallocated
    ``out[name][rank, index, expert_slot]`` arrays (zeros elsewhere); a replicated
    expert goes to every host holding it (``MoeLayout.expert_slot``)."""
    li = layout.lane_inter
    pre = f"layers.{layer}.block_sparse_moe.experts."
    for i, r in enumerate(ranks):
        host, lane = r // layout.topo.lanes, r % layout.topo.lanes
        for e in experts:
            e = int(e)
            j = layout.expert_slot(host, e)
            if j is None:
                continue
            gu, gus = _expert_gate_up(ckpt, pre, e, lane, li)
            dn, dns = _expert_down(ckpt, pre, e, lane, li)
            dst = out["expert_gu"]
            dst[i, index, j] = gu.view(np.uint8) if dst.dtype == np.uint8 else gu
            out["expert_gu_scale"][i, index, j] = gus
            out["expert_down"][i, index, j] = dn
            out["expert_down_scale"][i, index, j] = dns


def load_stack(
    ckpt: Checkpoint,
    spec,
    ranks,
    experts_by_layer=None,
    expert_cache: str = "",
    routed_experts: bool = True,
) -> dict:
    """All per-rank kernel inputs of ``stack.StackKernel`` except tables / meta / pages /
    x0 / states. ``experts_by_layer[layer]`` limits the routed experts loaded (tests);
    ``expert_cache`` (a directory) builds / reuses memmapped full expert families;
    ``routed_experts=False`` leaves the four expert families out (see ``stream_expert_layers``)."""
    from sglang_tpu.k3_megakernel.kernel.stack import NORMS, pack_inputs
    from sglang_tpu.k3_megakernel.kernel.stages.dense_mlp import dense_channels

    d, topo = spec.dims, spec.topo
    layout = MoeLayout(d, topo, replicas=spec.relax.expert_replicas)
    ranks = list(ranks)
    w = len(ranks)
    rep = lambda a: np.broadcast_to(a[None], (w,) + a.shape)
    out = {}
    out["norms"] = rep(
        np.stack(
            [
                np.stack(
                    [
                        np.asarray(
                            ckpt.get(f"layers.{i}.{n}.weight"), np.float32
                        ).reshape(1, -1)
                        for n in NORMS
                    ]
                )
                for i in range(spec.num_layers)
            ]
        )
    )
    out["final_w"] = rep(
        np.stack(
            [
                np.asarray(ckpt.get(n + ".weight"), np.float32).reshape(1, -1)
                for n in ("output_attn_res_proj", "output_attn_res_norm", "norm")
            ]
        )
    )
    pair = spec.relax.pair_oproj
    per = [
        load_kda_layer(ckpt, i, topo, ranks, pair_oproj=pair) for i in spec.kda_layers
    ]
    out.update({k: np.stack([p[k] for p in per], axis=1) for k in per[0]})
    per = [
        load_mla_layer(ckpt, i, topo, ranks, pair_oproj=pair) for i in spec.mla_layers
    ]
    out.update({k: np.stack([p[k] for p in per], axis=1) for k in per[0]})
    dn = load_dense_mlp(ckpt, topo, dense_channels(d, topo), ranks)
    out["dense_gu"], out["dense_down"] = dn["dense_gu"], dn["dense_down"]
    moe_layers = range(1, spec.num_layers)
    per = [load_moe_layer(ckpt, i, layout, ranks, experts=()) for i in moe_layers]
    for k in per[0]:
        out[k] = np.stack([p[k] for p in per], axis=1)
    out["router_bias"] = out["router_bias"].astype(np.float32)
    out["latent_norm"] = out["latent_norm"].astype(np.float32)
    if spec.relax.dense_fp8:
        apply_dense_fp8(out)
    n_moe, eh, li = len(moe_layers), layout.expert_slots, layout.lane_inter
    if not routed_experts:
        return pack_inputs(spec, out)
    if expert_cache:
        # The cache holds this host's ranks (the whole world when ``ranks`` spans it).
        lanes = topo.lanes
        host_ranks = list(
            range(ranks[0] // lanes * lanes, ranks[0] // lanes * lanes + lanes)
        )
        cache_ranks = list(range(topo.world)) if len(ranks) > lanes else host_ranks
        cache = build_expert_cache(
            ckpt._path, layout, moe_layers, expert_cache, ranks=cache_ranks
        )
        pick = [cache_ranks.index(r) for r in ranks]
        for n, a in cache.items():
            out[n] = a if pick == list(range(len(cache_ranks))) else a[pick]
        return pack_inputs(spec, out)
    out["expert_gu"] = np.zeros(
        (w, n_moe, eh, d.latent, 2 * li), ml_dtypes.float8_e4m3fn
    )
    out["expert_gu_scale"] = np.zeros((w, n_moe, eh, 1, 2 * li), np.float32)
    out["expert_down"] = np.zeros((w, n_moe, eh, li // 2, d.latent), np.uint8)
    out["expert_down_scale"] = np.zeros((w, n_moe, eh, li // 32, d.latent), np.uint8)
    for idx, layer in enumerate(moe_layers):
        experts = (
            range(d.num_experts)
            if experts_by_layer is None
            else experts_by_layer[layer]
        )
        fill_experts(ckpt, layer, layout, ranks, experts, out, idx)
    return pack_inputs(spec, out)


_EXPERT_FAMILIES = ("expert_gu", "expert_gu_scale", "expert_down", "expert_down_scale")


def _expert_family_shapes(layout: MoeLayout, world: int, n_layers: int) -> dict:
    d, eh, li = layout.dims, layout.expert_slots, layout.lane_inter
    return {
        "expert_gu": (
            (world, n_layers, eh, d.latent, 2 * li),
            np.uint8,
        ),  # e4m3 bytes (.npy has no fp8)
        "expert_gu_scale": ((world, n_layers, eh, 1, 2 * li), np.float32),
        "expert_down": ((world, n_layers, eh, li // 2, d.latent), np.uint8),
        "expert_down_scale": ((world, n_layers, eh, li // 32, d.latent), np.uint8),
    }


def _fill_expert_block(args):
    path, ckpt_path, layout, ranks, layers, index, layer, experts = args
    ckpt = Checkpoint(ckpt_path)
    shapes = _expert_family_shapes(layout, len(ranks), len(layers))
    out = {
        n: np.lib.format.open_memmap(os.path.join(path, n + ".npy"), mode="r+")
        for n in shapes
    }
    fill_experts(ckpt, layer, layout, ranks, experts, out, index)
    for a in out.values():
        a.flush()
    return len(experts)


def build_expert_cache(
    ckpt_path: str, layout: MoeLayout, layers, path: str, ranks=None, procs: int = 64
) -> dict:
    """Reblock every routed expert of ``layers`` for ``ranks`` (default: all ranks of the
    topology; serving passes this host's ranks) into ``.npy`` memmaps under ``path``
    (per-host /dev/shm). Concurrent callers on a host build once: the first takes a
    lock file, the others wait for ``DONE``. Returns ``{family: memmap [len(ranks), L, ...]}``."""
    import multiprocessing as mp
    import time

    ranks = list(range(layout.topo.world) if ranks is None else ranks)
    layers = list(layers)
    shapes = _expert_family_shapes(layout, len(ranks), len(layers))
    os.makedirs(path, exist_ok=True)
    done = os.path.join(path, "DONE")
    if not os.path.exists(done):
        try:
            fd = os.open(
                os.path.join(path, "LOCK"), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.close(fd)
            owner = True
        except FileExistsError:
            owner = False
        if owner:
            for n, (shape, dtype) in shapes.items():
                np.lib.format.open_memmap(
                    os.path.join(path, n + ".npy"), mode="w+", dtype=dtype, shape=shape
                )
            step = 32
            tasks = [
                (
                    path,
                    ckpt_path,
                    layout,
                    ranks,
                    layers,
                    i,
                    layer,
                    list(range(e, min(e + step, layout.dims.num_experts))),
                )
                for i, layer in enumerate(layers)
                for e in range(0, layout.dims.num_experts, step)
            ]
            with mp.get_context("fork").Pool(procs) as pool:
                for _ in pool.imap_unordered(_fill_expert_block, tasks):
                    pass
            open(done, "w").close()
        else:
            while not os.path.exists(done):
                time.sleep(5)
    out = {n: np.load(os.path.join(path, n + ".npy"), mmap_mode="r") for n in shapes}
    out["expert_gu"] = out["expert_gu"].view(ml_dtypes.float8_e4m3fn)
    return out


_WORKER_CKPTS: dict = {}


def _fill_layer_chunk(args):
    path, ckpt_path, layout, ranks, layer, experts = args
    if ckpt_path not in _WORKER_CKPTS:
        _WORKER_CKPTS[ckpt_path] = Checkpoint(ckpt_path)
    out = {
        n: np.lib.format.open_memmap(os.path.join(path, n + ".npy"), mode="r+")
        for n in _expert_family_shapes(layout, len(ranks), 1)
    }
    fill_experts(_WORKER_CKPTS[ckpt_path], layer, layout, ranks, experts, out, 0)
    for a in out.values():
        a.flush()


def stream_expert_layers(
    ckpt_path: str, layout: MoeLayout, layers, rank, procs: int = 8, chunk: int = 0
):
    """Yields ``(index, {family: [expert_slots, ...] array})`` for each MoE layer of one rank.
    A process pool reblocks the next layer into a one-layer /dev/shm scratch while the caller
    consumes the current one, so nothing persists (a 93-layer TP32 cache is ~570 GB per host).
    ``rank`` may be a list of ranks of one host: the arrays then get a leading ``len(rank)`` axis
    and each expert is read once for all of them. ``chunk`` experts per task (default: about four
    tasks per process)."""
    import multiprocessing as mp
    import shutil
    import tempfile

    layers = list(layers)
    ranks = [rank] if isinstance(rank, int) else list(rank)
    host = ranks[0] // layout.topo.lanes
    if any(r // layout.topo.lanes != host for r in ranks):
        raise ValueError(f"ranks {ranks} span more than one host")
    held = layout.slot_experts(host)
    chunk = chunk or -(-len(held) // (4 * procs))
    shapes = _expert_family_shapes(layout, len(ranks), 1)
    root = tempfile.mkdtemp(prefix=f"k3mk_stream_r{ranks[0]}_", dir="/dev/shm")

    def submit(pool, i):
        path = os.path.join(root, str(i % 2))
        os.makedirs(path, exist_ok=True)
        for n, (shape, dtype) in shapes.items():
            np.lib.format.open_memmap(
                os.path.join(path, n + ".npy"), mode="w+", dtype=dtype, shape=shape
            )
        tasks = [
            (path, ckpt_path, layout, ranks, layers[i], held[e : e + chunk])
            for e in range(0, len(held), chunk)
        ]
        return path, pool.map_async(_fill_layer_chunk, tasks)

    try:
        with mp.get_context("fork").Pool(procs) as pool:
            pending = submit(pool, 0)
            for i in range(len(layers)):
                path, result = pending
                result.get()
                pick = 0 if isinstance(rank, int) else slice(None)
                arrays = {
                    n: np.array(
                        np.load(os.path.join(path, n + ".npy"), mmap_mode="r")[pick, 0]
                    )
                    for n in shapes
                }
                pending = submit(pool, i + 1) if i + 1 < len(layers) else None
                arrays["expert_gu"] = arrays["expert_gu"].view(ml_dtypes.float8_e4m3fn)
                yield i, arrays
    finally:
        shutil.rmtree(root, ignore_errors=True)


def tensor_ranges(ckpt_path: str, num_layers: int) -> list[tuple[str, int, int]]:
    """(file, offset, size) of every text-model tensor of the first ``num_layers`` layers except
    the routed experts (those come from the expert cache), plus embedding, head and final norms."""
    with open(os.path.join(ckpt_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    keep = lambda n: (
        n.startswith("language_model.")
        and ".experts." not in n
        and (
            ".layers." not in n
            or int(n.split(".layers.")[1].split(".")[0]) < num_layers
        )
    )
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if keep(name):
            by_shard.setdefault(shard, []).append(name)
    out = []
    for shard, names in by_shard.items():
        path = os.path.join(ckpt_path, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name in names:
            start, end = header[name]["data_offsets"]
            out.append((path, 8 + n + start, end - start))
    return out
