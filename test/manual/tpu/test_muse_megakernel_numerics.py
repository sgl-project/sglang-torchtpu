# SPDX-License-Identifier: Apache-2.0
"""TP8 numerics of the Muse Spark megakernel against a CPU reference decoder.

Random, well-conditioned weights in the serving layouts at the real per-device geometry
(layout code follows benchmark/tpu/kernels/bench_muse_megakernel.py). Every weight is a pure
function of (layer, rank[, expert]) seeds, so any process can rebuild any rank's slice: the
reference runs teacher-forced, one layer per process, on the kernel's captured residual
streams. Run one process per device:

    eval $(python3 -m torch.tpu.distributed)
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
    torchrun --nproc_per_node="$WORLD_SIZE" test/manual/tpu/test_muse_megakernel_numerics.py \
        --cases fused_c1q1_s sal_c2q8_s --exact y_block_send=False

Rank 0 prints one MK_NUMERICS line per (case, layer) and MK_NUMERICS_PASS when every check
holds; every rank prints MK_NUMERICS_DONE.

SGLANG_TPU_MUSE_MK_QBLOCK=256 ... --qblock-ref --served loads the fp4 experts through the serving
path (checkpoint layout, prepare_tpu_nvfp4_moe_weights, requant_tpu_nvfp4_experts); the
reference requantizes the natural block-16 slabs independently.

--ep runs the EP megakernel (muse_megakernel/ep_mk): each rank gets its 32 whole experts and
the replicated pool / unpool assembled from every rank's slices, in ep_mk's layout (the pooled
width permuted as ep_mk.kernel.pooled_perm); fp8 experts then share one w2 column scale across
slices, as a checkpoint's whole expert does. Its arm is ep_mk.config.ep_arm's
(SGLANG_TPU_MUSE_MK_EP_ARM=a2a | gather forces one; a2a needs >= 64 rows).
"""

import argparse
import json
import math
import os
import shutil
import time
import zlib
from types import SimpleNamespace

import msgspec
import torch
import torch.distributed as dist

TP = 8
HIDDEN, HEAD_DIM, Q_HEADS, KV_HEADS = 8192, 64, 16, 2
N_EXPERTS, TOP_K, EXPERT_HIDDEN, INTER_LOCAL = 256, 8, 4096, 512
RMS_EPS, POST_EPS = 1e-5, 1e-8
SCALE_QUERY_BY, ATTN_SCALE = 5.473, 0.125
ROUTER_MULT, ROUTE_SCALE, ROUTER_EPS = 0.1768, 1.0, 1e-15
WINDOW = 2048
LOG2E = 1.4426950408889634
ROPE_BASE = 10000.0
# rows of the per-layer smalls slab replicated across ranks (the dense scales are per rank)
_RANK_SCALE_ROWS = ("wqkvg_s", "wo_s", "wpool_s", "wunpool_s")
_PAGES = {"fused": 128, "sal": 256, "sal_fp8": 256}
# SWA pages per request: the window plus a verify block, recycled modulo (as the bench)
_SWA_TOKENS = 3072

# Tolerances on a layer's output residual stream, live rows only, as (layer 0, later layers).
# A layer crosses ~12 bf16 rounding points (one bf16 ulp is 2^-8 relative); layer 0 sees the
# kernel's exact input (measured ~0.4% per row). Later layers are fed the bf16 capture of the
# f32 stream the kernel used, which moves ~1/3 of the rounded elements by one ulp (measured
# ~0.9%). A routing flip or a misindexed key moves a row by O(10%).
TOL_ROW_REL = (1e-2, 2e-2)  # per-row ||diff|| / ||ref||
TOL_MEAN_REL = (6e-3, 1.3e-2)  # mean of the per-row relative errors
# this step's K/V read back from the pool; fp8 pools flip a rounding when the input moves
TOL_KV_REL = {"fused": (1e-3, 1e-2), "sal": (1e-3, 1e-2), "sal_fp8": (1e-2, 6e-2)}
TOL_HID_REL = 1e-2  # final norm, computed from the captured last-layer stream
# biased router scores closer than this to the top-k boundary may legitimately flip
NEAR_TIE = 2e-3
# --ep: the EP megakernel and its weight layout (set once by main, before any weight is built)
_EP = False
# --qblock-ref: the reference reads the requantized fp4 experts the kernel reads
# (SGLANG_TPU_EP_QBLOCK with --ep, else SGLANG_TPU_MUSE_MK_QBLOCK)
_QBLOCK_REF = False
# --served: fp4 experts go through the serving weight load (checkpoint layout ->
# prepare_tpu_nvfp4_moe_weights -> requant_tpu_nvfp4_experts under SGLANG_TPU_MUSE_MK_QBLOCK)
_SERVED = False


class Case(msgspec.Struct, frozen=True):
    name: str
    kv: str  # fused | sal | sal_fp8
    n_req: int
    q: int
    ctx: tuple  # cached tokens per request
    expert_group: int = 0
    # not in DEFAULT_CASES (run with --cases)
    opt_in: bool = False


CASES = {
    c.name: c
    for c in (
        Case("fused_c1q1_s", "fused", 1, 1, (2500,)),
        Case("fused_c2q8_s", "fused", 2, 8, (3000, 1800)),
        Case("fused_c2q1_l", "fused", 2, 1, (20000, 17321), expert_group=2),
        Case("sal_c2q1_s", "sal", 2, 1, (2600, 1100), expert_group=2),
        Case("sal_c1q8_l", "sal", 1, 8, (20000,)),
        Case("sal_c2q8_s", "sal", 2, 8, (2900, 2047)),
        Case("salfp8_c1q1_s", "sal_fp8", 1, 1, (2100,)),
        Case("salfp8_c2q8_l", "sal_fp8", 2, 8, (20000, 13000), expert_group=2),
        Case("salfp8_c2q8_s", "sal_fp8", 2, 8, (1500, 3100)),
        Case("salfp8_c1q4_s", "sal_fp8", 1, 4, (2300,), opt_in=True),
        # 64 and 96 rows: the EP arms' crossover range
        Case(
            "salfp8_c8q8_s",
            "sal_fp8",
            8,
            8,
            tuple(900 + 97 * i for i in range(8)),
            opt_in=True,
        ),
        Case(
            "salfp8_c12q8_s",
            "sal_fp8",
            12,
            8,
            tuple(900 + 97 * i for i in range(12)),
            opt_in=True,
        ),
        # past the 32-chunk (32k-token) KV ring: the pipelined loop refills it
        Case("salfp8_c2q8_xl", "sal_fp8", 2, 8, (70000, 41000)),
        # 40 live rows padded to 48: the row-tiled program (MKConfig.row_tile)
        Case(
            "salfp8_c5q8_s",
            "sal_fp8",
            5,
            8,
            (1500, 3100, 900, 2200, 1700),
            expert_group=2,
            opt_in=True,
        ),
        # 128, 192 and 224 rows: the P/D decode host's row-tiled plans (VMEM-Y, its y_block_send
        # alternative, streamed-Y); pick one with SGLANG_TPU_MUSE_MK_DEV_* knobs
        Case(
            "salfp8_c16q8_s",
            "sal_fp8",
            16,
            8,
            tuple(900 + 97 * i for i in range(16)),
            opt_in=True,
        ),
        Case(
            "salfp8_c24q8_s",
            "sal_fp8",
            24,
            8,
            tuple(900 + 97 * i for i in range(24)),
            opt_in=True,
        ),
        Case(
            "salfp8_c32q8_s",
            "sal_fp8",
            32,
            8,
            tuple(900 + 97 * i for i in range(32)),
            opt_in=True,
        ),
        Case(
            "salfp8_c28q8_s",
            "sal_fp8",
            28,
            8,
            tuple(900 + 97 * i for i in range(28)),
            opt_in=True,
        ),
    )
}
DEFAULT_CASES = tuple(c for c in CASES if not CASES[c].opt_in)
# (case, flag=value): the kernel run with the flag toggled must be bit-identical to the
# default run; "expert_group=N" restacks the expert operands instead of an MKConfig field and
# "repeat" reruns the default (run-to-run determinism). vec_route=False is not listed: its
# outputs differ from vec_route=True by ~1 bf16 ulp in the expert branch (both match the
# reference equally well); pass --exact vec_route=False to see it. coll_order='ring' runs on a
# few-row step: the folds of 16+ row steps need xor peers (MKConfig.validate).
DEFAULT_EXACT = (
    ("fused_c1q1_s", "coll_order='ring'"),
    ("fused_c2q8_s", "repeat"),
    ("fused_c2q8_s", "y_block_send=False"),
    ("fused_c2q8_s", "o_rs_streamed=False"),
    ("fused_c2q8_s", "o_rs_rotate=False"),
    ("fused_c2q8_s", "expert_group=2"),
    ("salfp8_c2q8_s", "y_block_send=False"),
    ("salfp8_c2q8_s", "o_rs_streamed=False"),
)


# ----------------------------------------------------------------------------- seeded inputs
def _gen(*ids) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32(repr(ids).encode()))


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _fp8_bytes(shape, g) -> torch.Tensor:
    # sign kept, top exponent bit cleared: finite e4m3 of magnitude < 2, never NaN
    b = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g) & 0xBF
    return b.view(torch.float8_e4m3fn)


_FP8_RMS = float(
    (torch.arange(256, dtype=torch.int32).to(torch.uint8) & 0xBF)
    .view(torch.float8_e4m3fn)
    .float()
    .pow(2)
    .mean()
    .sqrt()
)
_E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
_E2M1_RMS = float(_E2M1.pow(2).mean().sqrt())


def _uniform(shape, g, lo, hi) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=g)


def _geometry(n_layers: int, global_layers: tuple):
    return SimpleNamespace(
        hidden=HIDDEN,
        head_dim=HEAD_DIM,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        n_layers=n_layers,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        expert_hidden=EXPERT_HIDDEN,
        inter_local=INTER_LOCAL,
        rms_eps=RMS_EPS,
        post_eps=POST_EPS,
        scale_query_by=SCALE_QUERY_BY,
        attn_scale=ATTN_SCALE,
        router_mult=ROUTER_MULT,
        route_scale=ROUTE_SCALE,
        router_eps=ROUTER_EPS,
        window=WINDOW,
        global_layers=global_layers,
    )


def _kinds(n_layers: int) -> tuple:
    return ("fp8",) + ("fp4",) * (n_layers - 2) + ("fp8",)


def replicated_layer(l: int) -> dict:
    """Per-layer values every rank holds identically (norm gains, residual gates, router)."""
    g = _gen("rep", l)
    out = {}
    for name in ("pre_attn_w", "pre_ffn_w", "post_ffn_w"):
        out[name] = _uniform(HIDDEN, g, 0.8, 1.2)
    for pre in ("attn", "ffn"):
        s = torch.randn(HIDDEN, generator=g)
        beta = torch.sigmoid(s)
        out[pre + "_beta"] = beta
        out[pre + "_alpha"] = torch.sqrt(
            torch.clamp(torch.sigmoid(-s) * (1 + beta), 1e-3)
        )
    for pre in os.environ.get("MK_NUMERICS_ZERO_BRANCH", "").split(","):
        if pre:  # debug: drop a branch from the residual stream ("attn" or "ffn")
            out[pre + "_alpha"], out[pre + "_beta"] = (
                torch.ones(HIDDEN),
                torch.zeros(HIDDEN),
            )
    out["pre_expert_w"] = _uniform(EXPERT_HIDDEN, g, 0.8, 1.2)
    out["post_expert_w"] = _uniform(EXPERT_HIDDEN, g, 0.8, 1.2)
    out["expert_bias"] = 0.02 * torch.randn(N_EXPERTS, generator=g)
    if os.environ.get("MK_NUMERICS_FIXED_ROUTE") == "1":
        # debug: every row picks experts 0..7, in that rank order
        out["expert_bias"][:TOP_K] += 20.0 - torch.arange(TOP_K)
    # logits std ~0.7 after router_mult: sigmoid scores spread, top-8 gaps ~1e-2
    out["wrouter"] = (
        torch.randn(HIDDEN, N_EXPERTS, generator=g) * (4 / HIDDEN**0.5)
    ).to(torch.bfloat16)
    return out


def dense_layer(l: int, rank: int) -> dict:
    """One rank's fp8 dense slices of layer l (K-major) and their per-output-channel scales."""
    g = _gen("dense", l, rank)
    qkvg = 2 * Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
    shapes = {
        "wqkvg": (HIDDEN, qkvg, 1.0),
        "wo": (Q_HEADS * HEAD_DIM, HIDDEN, 2.0 / TP**0.5),
        "wpool": (HIDDEN, EXPERT_HIDDEN // TP, 1.0),
        "wunpool": (EXPERT_HIDDEN, HIDDEN // TP, 3.0),
    }
    out = {}
    for name, (k, n, gain) in shapes.items():
        out[name] = _fp8_bytes((k, n), g)
        # outputs O(1): the fp8 bytes have rms _FP8_RMS, the inputs ~unit rms
        out[name + "_s"] = _uniform(n, g, 0.8, 1.2) * (gain / (_FP8_RMS * k**0.5))
    return out


def expert_slabs(l: int, rank: int, e: int, kind: str) -> list:
    """(w13, w13_scale, w2, w2_scale) of one expert's intermediate slice on one rank, as the
    kernel reads them: u32 words of 8 e2m1 codes / 4 e4m3 bytes along K."""
    g = _gen("expert", l, rank, e)
    He, Il = EXPERT_HIDDEN, INTER_LOCAL
    if kind == "fp4":
        b13 = 1.0 / (_E2M1_RMS * He**0.5)
        b2 = 1.0 / (_E2M1_RMS * (TP * Il) ** 0.5)
        return [
            torch.randint(
                -(2**31), 2**31 - 1, (He // 8, 2 * Il), dtype=torch.int32, generator=g
            ),
            (_uniform((He // 16, 2 * Il), g, 0.5, 1.5) * b13).to(torch.bfloat16),
            torch.randint(
                -(2**31), 2**31 - 1, (Il // 8, He), dtype=torch.int32, generator=g
            ),
            (_uniform((Il // 16, He), g, 0.5, 1.5) * b2).to(torch.bfloat16),
        ]
    s13 = (_uniform(2 * Il, g, 0.8, 1.2) / (_FP8_RMS * He**0.5)).to(torch.bfloat16)
    s2 = (_uniform(He, g, 0.8, 1.2) / (_FP8_RMS * (TP * Il) ** 0.5)).to(torch.bfloat16)
    if _EP:  # one column scale per whole expert
        s2 = (
            _uniform(He, _gen("expert_s2", l, e), 0.8, 1.2)
            / (_FP8_RMS * (TP * Il) ** 0.5)
        ).to(torch.bfloat16)
    return [
        _words8(_fp8_bytes((He, 2 * Il), g)),
        s13[None].expand(2, -1).contiguous(),
        _words8(_fp8_bytes((Il, He), g)),
        s2[None].expand(2, -1).contiguous(),
    ]


def _permute_packed_rows(
    words: torch.Tensor, bits: int, perm: torch.Tensor
) -> torch.Tensor:
    """int32 words [K / (32 / bits), N] of bits-wide codes along K, rows in perm order."""
    per = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)[None, :, None]
    codes = (words[:, None, :] >> shifts) & ((1 << bits) - 1)
    codes = codes.reshape(-1, words.shape[1])[perm].reshape(words.shape[0], per, -1)
    out = torch.zeros_like(words)
    for j in range(per):
        out |= codes[:, j] << (bits * j)
    return out


def _requant_kperm(
    words: torch.Tensor, scale: torch.Tensor, perm: torch.Tensor, block: int
) -> tuple:
    """ep_mk.experts.requant_fp4_kperm on natural int32 words [K/8, N]: (words in perm order,
    scales [K / block, N], the requantized weights as f32 in natural order)."""
    k, n = words.shape[0] * 8, words.shape[1]
    w = _dequant_fp4(words, scale)[perm].reshape(k // block, block, n)
    sb = (w.abs().amax(dim=1) / 6.0).to(torch.bfloat16)
    sbf = sb.float()
    q = (w / torch.where(sbf > 0, sbf, 1.0)[:, None]).clamp(-6.0, 6.0)
    d = (q.abs()[..., None] - _E2M1[:8]).abs()
    near = d == d.amin(dim=-1, keepdim=True)
    idx = torch.where(near, torch.arange(8) % 2, 2).argmin(dim=-1)  # ties to even
    deq = (torch.where(q < 0, -1.0, 1.0) * _E2M1[:8][idx] * sbf[:, None]).reshape(k, n)
    codes = (idx.to(torch.int32) | torch.where(q < 0, 8, 0).to(torch.int32)).reshape(
        k // 8, 8, n
    )
    out = torch.zeros(k // 8, n, dtype=torch.int32)
    for j in range(8):
        out |= codes[:, j] << (4 * j)
    return out, sb, deq[perm]


def tp_qblock() -> int:
    """SGLANG_TPU_MUSE_MK_QBLOCK (the TP kernel's requant block; 0 off)."""
    return int(os.environ.get("SGLANG_TPU_MUSE_MK_QBLOCK", "0"))


def tp_expert_slabs(l: int, rank: int, e: int, kind: str) -> list:
    """expert_slabs in the TP kernel's layout: under SGLANG_TPU_MUSE_MK_QBLOCK, fp4 rows
    requantized to its blocks of the natural order, scales [K / block, N]."""
    slabs = expert_slabs(l, rank, e, kind)
    b = tp_qblock()
    if not (b and kind == "fp4"):
        return slabs
    w13, s13, w2, s2 = slabs
    w13, s13, _ = _requant_kperm(w13, s13, torch.arange(EXPERT_HIDDEN), b)
    w2, s2, _ = _requant_kperm(w2, s2, torch.arange(INTER_LOCAL), b)
    return [w13, s13, w2, s2]


def _checkpoint_rows(words: torch.Tensor) -> torch.Tensor:
    """Natural int32 words [K/8, N] -> the checkpoint's [N, K/2] bytes (low nibble even K)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)[None, :, None]
    codes = ((words[:, None, :] >> shifts) & 0xF).reshape(-1, words.shape[1]).t()
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def served_expert_refs(l: int, rank: int, device: torch.device) -> list:
    """One fp4 layer's experts as serving loads them: checkpoint-layout codes with the bf16
    block-16 scales as checkpoint scales, prepare_tpu_nvfp4_moe_weights (fused-RS layout),
    then requant_tpu_nvfp4_experts to SGLANG_TPU_MUSE_MK_QBLOCK."""
    import types

    from sglang_tpu.fp4_moe import prepare_tpu_nvfp4_moe_weights
    from sglang_tpu.moe.pallas import fused_moe_rs_op
    from sglang_tpu.muse_megakernel.expert_requant import requant_tpu_nvfp4_experts

    slabs = [expert_slabs(l, rank, e, "fp4") for e in range(N_EXPERTS)]
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    for name, j, scale in (
        ("w13_weight", 0, False),
        ("w13_weight_scale", 1, True),
        ("w2_weight", 2, False),
        ("w2_weight_scale", 3, True),
    ):
        t = torch.stack(
            [s[j].t().contiguous() if scale else _checkpoint_rows(s[j]) for s in slabs]
        )
        layer.register_parameter(
            name, torch.nn.Parameter(t.to(device), requires_grad=False)
        )
    original_prepare = fused_moe_rs_op.prepare_fused_moe_rs
    fused_moe_rs_op.prepare_fused_moe_rs = lambda _: None
    try:
        prepare_tpu_nvfp4_moe_weights(
            layer, group_size=16, scale_dtype=torch.bfloat16, use_fused_rs=True
        )
    finally:
        fused_moe_rs_op.prepare_fused_moe_rs = original_prepare
    layer.__dict__["_tpu_fused_rs_expert_tp"] = True
    requant_tpu_nvfp4_experts(layer, tp_qblock())
    return [
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
    ]


def ref_expert(l: int, rank: int, e: int, kind: str) -> tuple:
    """decode_expert of one rank's slice; --qblock-ref: the requantized weights the kernel
    reads (SGLANG_TPU_EP_QBLOCK in kperm order with --ep, else SGLANG_TPU_MUSE_MK_QBLOCK in
    natural order)."""
    from sglang_tpu.muse_megakernel.ep_mk.experts import kperm, qblock

    slabs = expert_slabs(l, rank, e, kind)
    if not (_QBLOCK_REF and kind == "fp4"):
        return decode_expert(slabs, kind)
    w13, s13, w2, s2 = slabs
    if not _EP:
        nat_he, nat_il = torch.arange(EXPERT_HIDDEN), torch.arange(INTER_LOCAL)
        return (
            _requant_kperm(w13, s13, nat_he, tp_qblock())[2],
            _requant_kperm(w2, s2, nat_il, tp_qblock())[2],
        )
    kp_he = torch.from_numpy(kperm(EXPERT_HIDDEN))
    kp_il = torch.from_numpy(kperm(INTER_LOCAL))
    return (
        _requant_kperm(w13, s13, kp_he, qblock())[2],
        _requant_kperm(w2, s2, kp_il, qblock())[2],
    )


def ep_expert_slabs(l: int, rank: int, kind: str) -> list:
    """One rank's whole experts (w13, w13_scale, w2, w2_scale), stacked, in ep_mk's layout:
    every rank's intermediate slices of each expert side by side, the hidden rows in
    pooled_perm order and, for fp4, the gate / up columns and w2's rows in kperm order; then
    w13's columns in chunk_cols order."""
    from sglang_tpu.muse_megakernel.ep_mk.experts import chunk_cols, kperm, qblock

    el, il, he = N_EXPERTS // TP, INTER_LOCAL, EXPERT_HIDDEN
    cc = torch.from_numpy(chunk_cols(2 * TP * il, il))
    kp_he = torch.from_numpy(kperm(he))
    kp_i, kp_2i = torch.from_numpy(kperm(TP * il)), torch.from_numpy(kperm(2 * TP * il))
    out = [[], [], [], []]
    for e in range(rank * el, (rank + 1) * el):
        sl = [expert_slabs(l, d, e, kind) for d in range(TP)]
        w13 = torch.cat([s[0][:, :il] for s in sl] + [s[0][:, il:] for s in sl], dim=1)
        s13 = torch.cat([s[1][:, :il] for s in sl] + [s[1][:, il:] for s in sl], dim=1)
        w2 = torch.cat([s[2] for s in sl], dim=0)
        if kind == "fp4":
            s2 = torch.cat([s[3] for s in sl], dim=0)
            if qblock():
                w13, s13, _ = _requant_kperm(w13, s13, kp_he, qblock())
                w2, s2, _ = _requant_kperm(w2, s2, kp_i, qblock())
                w13, s13 = w13[:, kp_2i], s13[:, kp_2i]
            else:
                w13 = _permute_packed_rows(w13, 4, kp_he)[:, kp_2i]
                s13, w2 = s13[:, kp_2i], _permute_packed_rows(w2, 4, kp_i)
        else:
            s2 = sl[0][3]
            w13 = _permute_packed_rows(w13, 8, kp_he)
        w13, s13 = w13[:, cc], s13[:, cc]
        for j, t in enumerate((w13, s13, w2, s2)):
            out[j].append(t)
    return [torch.stack(t) for t in out]


def _words8(w: torch.Tensor) -> torch.Tensor:
    """fp8 [K, N] -> int32 words [K/4, N], byte i of word w = row 4w + i."""
    b = w.view(torch.uint8).to(torch.int32).reshape(w.shape[0] // 4, 4, w.shape[1])
    return b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)


def decode_expert(slabs: list, kind: str) -> tuple:
    """f32 (w13 [He, 2 Il], w2 [Il, He]) from expert_slabs."""
    w13, s13, w2, s2 = slabs
    if kind == "fp4":
        return _dequant_fp4(w13, s13), _dequant_fp4(w2, s2)
    return _dequant_fp8(w13) * s13[0].float(), _dequant_fp8(w2) * s2[0].float()


def _dequant_fp4(words: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)[None, :, None]
    codes = (words[:, None, :] >> shifts) & 0xF  # nibble j of word w is row 8w + j
    v = _E2M1[codes.long()].reshape(words.shape[0] * 8, words.shape[1])
    return v * scale.float().repeat_interleave(16, dim=0)


def _dequant_fp8(words: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 8, dtype=torch.int32)[None, :, None]
    b = ((words[:, None, :] >> shifts) & 0xFF).to(torch.uint8)
    return (
        b.view(torch.float8_e4m3fn).float().reshape(words.shape[0] * 4, words.shape[1])
    )


def rope_table(max_position: int) -> torch.Tensor:
    inv = 1.0 / (
        ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    )
    ang = torch.arange(max_position, dtype=torch.float64)[:, None] * inv
    return torch.cat(
        [ang.cos().repeat_interleave(2, 1), ang.sin().repeat_interleave(2, 1)], dim=1
    ).float()


def final_gain() -> torch.Tensor:
    return _uniform(HIDDEN, _gen("final"), 0.8, 1.2)


def kv_logical(case: Case, l: int, rank: int, pages: int) -> torch.Tensor:
    """Cached K/V [pages, page, kv_heads, 2, head_dim] as the pool stores them: K per-head
    RMS-normalized (as after the model's k norm), values rounded to the pool dtype."""
    g = _gen("kv", case.name, l, rank)
    x = torch.randn(pages, _PAGES[case.kv], KV_HEADS, 2, HEAD_DIM, generator=g)
    k = x[..., 0, :]
    x[..., 0, :] = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    dt = torch.float8_e4m3fn if case.kv == "sal_fp8" else torch.bfloat16
    return x.to(dt).float()


def pool_layout(x: torch.Tensor, kv: str) -> torch.Tensor:
    """Logical K/V -> the serving pool tensor (fused packed rows or SEQ_ALONG_LANE blocks)."""
    pages, ps = x.shape[:2]
    if (
        kv == "fused"
    ):  # [pages, page, 1, 2, kv_heads * head_dim]: K heads side by side, then V
        return (
            x.permute(0, 1, 3, 2, 4)
            .reshape(pages, ps, 1, 2, KV_HEADS * HEAD_DIM)
            .to(torch.bfloat16)
        )
    pk = 4 if kv == "sal_fp8" else 2
    dt = torch.float8_e4m3fn if kv == "sal_fp8" else torch.bfloat16
    # [pages, 2 * kv_heads (K_h0, V_h0, K_h1, ...), head_dim // pk, pk, page]
    return (
        x.permute(0, 2, 3, 4, 1)
        .reshape(pages, 2 * KV_HEADS, HEAD_DIM // pk, pk, ps)
        .to(dt)
    )


class StepInputs(msgspec.Struct):
    rows: int
    n_live: int
    pps: int  # page-table entries per request
    swa_pages: int
    pos: torch.Tensor  # [rows] int32
    pf: torch.Tensor  # [n_req * pps] int32, global layers
    psw: torch.Tensor  # [n_req * pps] int32, sliding-window layers
    emb: torch.Tensor  # [rows, H] bf16
    max_position: int


def step_inputs(case: Case) -> StepInputs:
    ps = _PAGES[case.kv]
    # ops.step_rows: the TP size, and past 32 rows (every EP step) the 16-row tile of the
    # row-tiled program
    rows = -(-case.n_req * case.q // TP) * TP
    if rows > 32 or _EP:
        rows = -(-case.n_req * case.q // math.lcm(16, TP)) * math.lcm(16, TP)
    n_live = case.n_req * case.q
    pps = -(-(max(case.ctx) + case.q) // ps)
    swa_pages = _SWA_TOKENS // ps
    g = _gen("step", case.name)
    pos = torch.zeros(rows, dtype=torch.int32)
    for j, c in enumerate(case.ctx):
        pos[j * case.q : (j + 1) * case.q] = torch.arange(
            c, c + case.q, dtype=torch.int32
        )
    # shuffled page ids, page 0 never used: a wrong table entry reads unrelated keys
    full = torch.randperm(case.n_req * pps, generator=g).to(torch.int32) + 1
    ring = torch.randperm(case.n_req * swa_pages, generator=g).to(torch.int32) + 1
    p = torch.arange(pps)
    swa = torch.cat([ring[j * swa_pages + p % swa_pages] for j in range(case.n_req)])
    emb = torch.randn(rows, HIDDEN, generator=g)
    emb[n_live:] = 0
    return StepInputs(
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


def n_pool_pages(case: Case, st: StepInputs, l: int, global_layers: tuple) -> int:
    return case.n_req * (st.pps if l in global_layers else st.swa_pages) + 1


# ----------------------------------------------------------------------------- kernel side
def _stack_slabs(slabs: list) -> torch.Tensor:
    """torch.stack; fp4 pairs go through their bytes (a device stack of float4_e2m1fn_x2
    mis-sizes the packed dim)."""
    if slabs[0].dtype != torch.float4_e2m1fn_x2:
        return torch.stack(slabs)
    return torch.stack([t.view(torch.uint8) for t in slabs]).view(slabs[0].dtype)


class _Runner:
    """Owns the fake MuseMegakernel (the attributes MuseMegakernel.config reads) and the device
    weights of this rank; runs one compiled step per call."""

    def __init__(
        self, K, ops, n_layers: int, global_layers: tuple, rank: int, max_position: int
    ):
        self.K, self.ops, self.rank = K, ops, rank
        self.n_layers, self.global_layers = n_layers, global_layers
        self.kinds = _kinds(n_layers)
        self.overrides: dict = {}
        self.last_cfg = None
        runner = self

        class _Fake:
            callable_for = ops.MuseMegakernel.callable_for

            def config(self, **kw):
                cfg = ops.MuseMegakernel.config(self, **kw)
                if runner.overrides:
                    cfg = msgspec.structs.replace(cfg, **runner.overrides)
                runner.last_cfg = cfg
                return cfg

        mk = _Fake()
        mk.geo = _geometry(n_layers, global_layers)
        mk.tp_size = TP
        mk.kinds = self.kinds
        mk.max_position = max_position
        mk.chunk_pages = ops._CHUNK_PAGES
        mk.kv_packed = True
        mk.expert_groups = ()
        mk.moe_mode = "ep" if _EP else "tp"
        mk._ops = {}
        self.mk = mk
        ops._INSTANCE[:] = [mk]
        ops._register_op()
        self.device = torch.device("tpu")
        self.weights = self._weights(max_position)
        self.per_layer_experts = [
            [t.to(self.device) for t in ep_expert_slabs(l, rank, k)]
            if _EP
            else served_expert_refs(l, rank, self.device)
            if _SERVED and k == "fp4"
            else [
                torch.stack(t).to(self.device)
                for t in zip(
                    *(tp_expert_slabs(l, rank, e, k) for e in range(N_EXPERTS))
                )
            ]
            for l, k in enumerate(self.kinds)
        ]

    def _weights(self, max_position: int) -> list:
        K, L = self.K, self.n_layers
        dense = [dense_layer(l, self.rank) for l in range(L)]
        if (
            _EP
        ):  # replicated pool / unpool: every rank's slices, the pooled width permuted
            from sglang_tpu.muse_megakernel.ep_mk.kernel import pooled_perm

            kp = torch.from_numpy(pooled_perm(EXPERT_HIDDEN))
            for l in range(L):
                ranks = [dense_layer(l, d) for d in range(TP)]
                for name in ("wpool", "wunpool"):
                    w = torch.cat([r[name] for r in ranks], dim=1)
                    sc = torch.cat([r[name + "_s"] for r in ranks])
                    if name == "wpool":
                        w, sc = w[:, kp], sc[kp]
                    dense[l][name], dense[l][name + "_s"] = w, sc
        smalls = torch.zeros(L, K.SMALL_ROWS, HIDDEN)
        for l in range(L):
            rep = replicated_layer(l)
            if _EP:
                rep["pre_expert_w"] = rep["pre_expert_w"][pooled_perm(EXPERT_HIDDEN)]
            for name in (
                "pre_attn_w",
                "attn_alpha",
                "attn_beta",
                "pre_ffn_w",
                "post_ffn_w",
                "ffn_alpha",
                "ffn_beta",
            ):
                smalls[l, K.SMALL[name]] = rep[name]
            for name in ("pre_expert_w", "post_expert_w"):
                smalls[l, K.SMALL[name], :EXPERT_HIDDEN] = rep[name]
            smalls[l, K.SMALL["expert_bias"]] = K.NEG
            smalls[l, K.SMALL["expert_bias"], :N_EXPERTS] = rep["expert_bias"]
            for name in _RANK_SCALE_ROWS:
                s = dense[l][name]
                smalls[l, K.SMALL[name], : s.numel()] = s
        final = torch.zeros(8, HIDDEN)
        final[0] = final_gain()
        by_name = {
            "rope": rope_table(max_position),
            "final_norm": final,
            "smalls": smalls,
            "wrouter": torch.stack([replicated_layer(l)["wrouter"] for l in range(L)]),
        }
        for name in ("wqkvg", "wo", "wpool", "wunpool"):
            by_name[name] = torch.stack([d[name] for d in dense])
        return [by_name[n].to(self.device) for n in K.WEIGHT_NAMES]

    def experts(self, expert_group: int) -> list:
        groups = self.ops.expert_groups_of(self.kinds, expert_group)
        self.mk.expert_groups = groups
        if not groups:
            return [t for slabs in self.per_layer_experts for t in slabs]
        cfg = SimpleNamespace(n_layers=self.n_layers, expert_groups=groups)
        refs = []
        for unit in self.K.expert_units(cfg):
            if unit[0] == "layer":
                refs += self.per_layer_experts[unit[1]]
            else:
                for j in range(4):
                    refs.append(
                        _stack_slabs(
                            [
                                self.per_layer_experts[li][j]
                                for li in range(unit[1], unit[2])
                            ]
                        )
                    )
        return refs

    def pools(self, case: Case, st: StepInputs) -> list:
        return [
            pool_layout(
                kv_logical(
                    case, l, self.rank, n_pool_pages(case, st, l, self.global_layers)
                ),
                case.kv,
            ).to(self.device)
            for l in range(self.n_layers)
        ]

    def step(
        self, case: Case, st: StepInputs, kv: list, expert_group: int, overrides: dict
    ):
        """(hidden [rows, H] bf16, aux [rows, L * H] bf16, the MKConfig used); kv updated in place."""
        self.overrides = dict(overrides)
        self.last_cfg = None
        experts = self.experts(expert_group)
        d = self.device
        meta = torch.tensor([st.n_live, self.rank], dtype=torch.int32, device=d)
        capture = list(range(self.n_layers))
        op = torch.ops.sglang_tpu_pallas.muse_megakernel

        def run(meta, pos, pf, psw, emb, weights, experts, kv):
            return op(
                meta,
                pos,
                pf,
                psw,
                emb,
                weights,
                experts,
                kv,
                case.q,
                case.n_req,
                capture,
            )

        torch._dynamo.reset()
        # torch_tpu keys its executable cache on the graph, not on the kernel config the op picks
        # while tracing: an override step would replay the previous step's executable
        torch.tpu._clear_cache()
        compiled = torch.compile(run, fullgraph=True, dynamic=False)
        hid, aux = compiled(
            meta,
            st.pos.to(d),
            st.pf.to(d),
            st.psw.to(d),
            st.emb.to(d),
            self.weights,
            experts,
            kv,
        )
        if self.last_cfg is None:
            raise RuntimeError(
                "the step ran a cached executable: the op was not traced"
            )
        return hid.cpu(), aux.cpu(), self.last_cfg

    def new_kv(self, case: Case, st: StepInputs, kv: list) -> torch.Tensor:
        """This step's K/V rows read back from the pools: [L, n_live, kv_heads, 2, head_dim]."""
        ps = _PAGES[case.kv]
        out = []
        for l in range(self.n_layers):
            tab = st.pf if l in self.global_layers else st.psw
            r = torch.arange(st.n_live)
            pos = st.pos[: st.n_live].long()
            page = tab[(r // case.q) * st.pps + pos // ps].long()
            pool = kv[l].cpu()
            if pool.dtype == torch.float8_e4m3fn:  # index the bytes
                pool = pool.view(torch.uint8)
            if case.kv == "fused":
                x = pool[page, pos % ps, 0].float()  # [n, 2, kv_heads * hd]
                x = x.reshape(-1, 2, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
            else:
                x = pool[page, :, :, :, pos % ps]  # [n, 2 * kv_heads, hd // pk, pk]
                if case.kv == "sal_fp8":
                    x = x.view(torch.float8_e4m3fn)
                x = x.float()
                x = x.reshape(-1, KV_HEADS, 2, HEAD_DIM)
            out.append(x)
        return torch.stack(out)


# ----------------------------------------------------------------------------- reference
def _rms(x: torch.Tensor, eps: float, w: torch.Tensor | None = None) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def _heads_rms(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    return _rms(x.reshape(x.shape[0], n_heads, HEAD_DIM), RMS_EPS).reshape(x.shape)


def _rope(x: torch.Tensor, n_heads: int, table: torch.Tensor) -> torch.Tensor:
    """Interleaved pairs; table rows [cos x2 | sin x2] per position."""
    xs = x.reshape(x.shape[0], n_heads, HEAD_DIM)
    c, s = table[:, None, :HEAD_DIM], table[:, None, HEAD_DIM:]
    partner = torch.stack([-xs[..., 1::2], xs[..., 0::2]], dim=-1).reshape(xs.shape)
    return (xs * c + partner * s).reshape(x.shape)


class _LayerRef:
    """Teacher-forced reference of one decoder layer on the gathered (all-rank) weights,
    mirroring the kernel's documented bf16 rounding points."""

    def __init__(
        self,
        l: int,
        case: Case,
        st: StepInputs,
        n_layers: int,
        global_layers: tuple,
        attn_mode: str,
    ):
        self.l, self.case, self.st, self.attn_mode = l, case, st, attn_mode
        self.is_global = l in global_layers
        self.kind = _kinds(n_layers)[l]
        self.rep = replicated_layer(l)
        self.dense = [dense_layer(l, d) for d in range(TP)]
        pages = n_pool_pages(case, st, l, global_layers)
        self.kv = [kv_logical(case, l, d, pages) for d in range(TP)]
        self.rope = rope_table(st.max_position)[st.pos[: st.n_live].long()]
        self.kv_dt = torch.float8_e4m3fn if case.kv == "sal_fp8" else torch.bfloat16
        self._y: dict = {}  # (row, expert) -> bf16-rounded cross-rank expert output

    def _mv(self, x: torch.Tensor, d: int, name: str) -> torch.Tensor:
        w = self.dense[d]
        return (_bf(x) @ w[name].float()) * w[name + "_s"]

    def attention_rank(self, x: torch.Tensor, d: int):
        qd, kd = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
        qkvg = self._mv(x, d, "wqkvg")
        q, k, v, og = (
            qkvg[:, :qd],
            qkvg[:, qd : qd + kd],
            qkvg[:, qd + kd : qd + 2 * kd],
            qkvg[:, qd + 2 * kd :],
        )
        qn = _heads_rms(q, Q_HEADS) * SCALE_QUERY_BY
        kn = _heads_rms(k, KV_HEADS)
        if not self.is_global:
            qn, kn = _rope(qn, Q_HEADS, self.rope), _rope(kn, KV_HEADS, self.rope)
        kb = kn.to(self.kv_dt).float().to(torch.bfloat16).float()
        vb = v.to(self.kv_dt).float().to(torch.bfloat16).float()
        out = torch.cat(
            [
                self._request_attention(j, qn, kb, vb, self.kv[d])
                for j in range(self.case.n_req)
            ]
        )
        attn = _bf(_heads_rms(out, Q_HEADS))
        gated = _bf(torch.sigmoid(_bf(og)) * attn)
        return self._mv(gated, d, "wo"), kb, vb

    def _request_attention(self, j, qn, kb, vb, kv):
        q, ps, st = self.case.q, _PAGES[self.case.kv], self.st
        rows = slice(j * q, (j + 1) * q)
        pos = st.pos[rows].long()
        p_lo = int(pos[0])
        first = 0 if self.is_global else max(p_lo - WINDOW + 1, 0)
        kpos = torch.arange(first, p_lo)
        tab = st.pf if self.is_global else st.psw
        page = tab[j * st.pps + kpos // ps].long()
        cached = kv[page, kpos % ps]  # [n, kv_heads, 2, hd]
        keys = torch.cat([cached[:, :, 0], kb[rows].reshape(q, KV_HEADS, HEAD_DIM)])
        vals = torch.cat([cached[:, :, 1], vb[rows].reshape(q, KV_HEADS, HEAD_DIM)])
        allpos = torch.cat([kpos, pos])
        live = allpos[None, :] <= pos[:, None]
        if not self.is_global:
            live &= pos[:, None] - allpos[None, :] < WINDOW
        g = Q_HEADS // KV_HEADS
        qh = qn[rows].reshape(q, KV_HEADS, g, HEAD_DIM)
        # the kernel's query rounding and exponent base (MKConfig.attn_exp2 / attn_unshifted)
        if self.attn_mode == "exp2":
            qs, sc, base = _bf(qh * (ATTN_SCALE * LOG2E)), 1.0, 2.0
        elif self.attn_mode == "unshifted":
            qs, sc, base = _bf(qh) * ATTN_SCALE, 1.0, math.e
        else:
            qs, sc, base = _bf(qh), ATTN_SCALE, math.e
        s = torch.einsum("thgd,nhd->thgn", qs.double(), keys.double()) * sc
        s = s.masked_fill(~live[:, None, None, :], -math.inf)
        if self.attn_mode == "shifted":
            s = s - s.amax(-1, keepdim=True)
        p = torch.pow(base, s)
        # probabilities enter P @ V as bf16; the row sums stay f32
        acc = torch.einsum("thgn,nhd->thgd", _bf(p.float()).double(), vals.double())
        return (acc / p.sum(-1, keepdim=True)).float().reshape(q, -1)

    def attention_block(self, stream_in: torch.Tensor):
        """stream after the attention residual, and every rank's new (K, V)."""
        rep = self.rep
        x = _bf(_rms(stream_in, RMS_EPS, rep["pre_attn_w"]))
        o_sum, new_kv = 0, []
        for d in range(TP):
            o, kb, vb = self.attention_rank(x, d)
            o_sum = o_sum + _bf(o)  # partials travel as bf16, summed in f32
            new_kv.append(
                torch.stack(
                    [
                        kb.reshape(-1, KV_HEADS, HEAD_DIM),
                        vb.reshape(-1, KV_HEADS, HEAD_DIM),
                    ],
                    2,
                )
            )
        a = _bf(o_sum)
        stream = rep["attn_alpha"] * stream_in + rep["attn_beta"] * _bf(
            _rms(a, POST_EPS)
        )
        return stream, torch.stack(new_kv)  # [TP, n, kv_heads, 2, hd]

    def route(self, stream: torch.Tensor):
        """(expert input [n, He] bf16, sigmoid scores [n, E], biased scores [n, E])."""
        ffn_bf = _bf(_rms(stream, RMS_EPS, self.rep["pre_ffn_w"]))
        pooled = torch.cat(
            [_bf(self._mv(ffn_bf, d, "wpool")) for d in range(TP)], dim=1
        )
        xe = _bf(_rms(pooled, RMS_EPS, self.rep["pre_expert_w"]))
        sc = torch.sigmoid((ffn_bf @ self.rep["wrouter"].float()) * ROUTER_MULT)
        return xe, sc, sc + self.rep["expert_bias"]

    def expert_rows(self, xe: torch.Tensor, pairs: set) -> None:
        """Fill the (row, expert) outputs: per-rank slices summed as the kernel's reduce."""
        todo = {}
        for r, e in pairs:
            if (r, e) not in self._y:
                todo.setdefault(e, []).append(r)
        for e, rs in todo.items():
            x = xe[rs]
            y = 0
            for d in range(TP):
                w13, w2 = ref_expert(self.l, d, e, self.kind)
                gu = x @ w13
                gate, up = gu[:, :INTER_LOCAL], gu[:, INTER_LOCAL:]
                h = _bf(gate * torch.sigmoid(gate) * up)
                y = y + _bf(h @ w2)
            y = _bf(y)
            for i, r in enumerate(rs):
                self._y[(r, e)] = y[i]

    def ffn_block(
        self,
        stream: torch.Tensor,
        xe: torch.Tensor,
        sc: torch.Tensor,
        ids: torch.Tensor,
    ):
        self.expert_rows(xe, {(r, int(e)) for r in range(ids.shape[0]) for e in ids[r]})
        ysum = torch.stack(
            [
                torch.stack([self._y[(r, int(e))] for e in ids[r]])
                for r in range(ids.shape[0])
            ]
        )  # [n, K, He]
        ynorm = _bf(_rms(ysum * self.rep["post_expert_w"], POST_EPS))
        w = torch.gather(sc, 1, ids)
        w = w / (w.sum(1, keepdim=True) + ROUTER_EPS) * ROUTE_SCALE
        u = _bf((ynorm * w[:, :, None]).sum(1))
        f = torch.cat([_bf(self._mv(u, d, "wunpool")) for d in range(TP)], dim=1)
        rep = self.rep
        return rep["ffn_alpha"] * stream + rep["ffn_beta"] * _bf(
            _rms(f * rep["post_ffn_w"], POST_EPS)
        )


def _row_rel(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)


def _near_tie_swaps(biased_row: torch.Tensor, ids_row: torch.Tensor):
    """Alternative top-k sets swapping one or two experts within NEAR_TIE of the boundary
    for as many just outside it."""
    order = torch.argsort(biased_row, descending=True)
    edge = 0.5 * (biased_row[order[TOP_K - 1]] + biased_row[order[TOP_K]])
    near = (biased_row - edge).abs() < NEAR_TIE
    inside = [int(e) for e in ids_row if near[e]]
    outside = [int(e) for e in order[TOP_K : 4 * TOP_K] if near[e]]
    for a in inside:
        for b in outside:
            yield torch.where(ids_row == a, torch.tensor(b), ids_row)
    for i, a1 in enumerate(inside):
        for a2 in inside[i + 1 :]:
            for j, b1 in enumerate(outside):
                for b2 in outside[j + 1 :]:
                    alt = torch.where(ids_row == a1, torch.tensor(b1), ids_row)
                    yield torch.where(alt == a2, torch.tensor(b2), alt)


def _debug_row(ref, mid, xe, sc, ids, out, r, biased) -> None:
    """Which slot of row r the kernel got wrong: the row's error with one slot's y zeroed or
    replaced by another slot's (debug)."""
    ref.expert_rows(xe, {(r, int(e)) for e in ids[r]})
    ys = torch.stack([ref._y[(r, int(e))] for e in ids[r]])  # [K, He]
    rep = ref.rep

    def row_out(y):
        ynorm = _bf(_rms(y[None] * rep["post_expert_w"], POST_EPS))
        w = torch.gather(sc[r : r + 1], 1, ids[r : r + 1])
        w = w / (w.sum(1, keepdim=True) + ROUTER_EPS) * ROUTE_SCALE
        u = _bf((ynorm * w[:, :, None]).sum(1))
        f = torch.cat([_bf(ref._mv(u, d, "wunpool")) for d in range(TP)], dim=1)
        return rep["ffn_alpha"] * mid[r : r + 1] + rep["ffn_beta"] * _bf(
            _rms(f * rep["post_ffn_w"], POST_EPS)
        )

    res = {
        "as_is": float(_row_rel(out[r : r + 1], row_out(ys))[0]),
        "post_attn": float(_row_rel(out[r : r + 1], mid[r : r + 1])[0]),
    }
    for k in range(TOP_K):
        for j in [None] + [j for j in range(TOP_K) if j != k]:
            y2 = ys.clone()
            y2[k] = 0 if j is None else ys[j]
            res[f"k{k}_" + ("zero" if j is None else f"as{j}")] = float(
                _row_rel(out[r : r + 1], row_out(y2))[0]
            )
    # a slot computed on another row's x (a wrong gather)
    k_bad = int(min((v, kk) for kk, v in res.items() if kk.endswith("_zero"))[1][1])
    e_bad = int(ids[r][k_bad])
    ref.expert_rows(xe, {(r2, e_bad) for r2 in range(ids.shape[0])})
    for r2 in range(ids.shape[0]):
        y2 = ys.clone()
        y2[k_bad] = ref._y[(r2, e_bad)]
        res[f"k{k_bad}_x{r2}"] = float(_row_rel(out[r : r + 1], row_out(y2))[0])
    # another expert of the same rank, or the right one missing / doubling one I-chunk (one
    # TP slice is one EP chunk)
    x = xe[r : r + 1]
    owner = e_bad // (N_EXPERTS // TP)
    for e2 in range(owner * (N_EXPERTS // TP), (owner + 1) * (N_EXPERTS // TP)):
        ref.expert_rows(xe, {(r, e2)})
        y2 = ys.clone()
        y2[k_bad] = ref._y[(r, e2)]
        res[f"k{k_bad}_e{e2}"] = float(_row_rel(out[r : r + 1], row_out(y2))[0])
    parts = []
    for d in range(TP):
        w13, w2 = ref_expert(ref.l, d, e_bad, ref.kind)
        gu = x @ w13
        h = _bf(
            gu[:, :INTER_LOCAL]
            * torch.sigmoid(gu[:, :INTER_LOCAL])
            * gu[:, INTER_LOCAL:]
        )
        parts.append(_bf(h @ w2)[0])
    for d in range(TP):
        for name, mult in (("drop", 0.0), ("twice", 2.0)):
            y2 = ys.clone()
            y2[k_bad] = _bf(sum(parts) + (mult - 1.0) * parts[d])
            res[f"k{k_bad}_chunk{d}_{name}"] = float(
                _row_rel(out[r : r + 1], row_out(y2))[0]
            )
    best = sorted(res.items(), key=lambda kv: kv[1])[:4]
    order = torch.argsort(biased[r], descending=True)
    edge = float(0.5 * (biased[r][order[TOP_K - 1]] + biased[r][order[TOP_K]]))
    top12 = [(int(e), round(float(biased[r][e]) - edge, 5)) for e in order[:12]]
    info = {
        "row": r,
        "ids": ids[r].tolist(),
        "owner": [int(e) // 32 for e in ids[r]],
        "score_minus_edge": top12,
    }
    print("MK_NUMERICS_DEBUG_ROW", json.dumps({**info, "best": best}), flush=True)


def check_layer(
    l, case, st, n_layers, global_layers, attn_mode, stream_in, out, new_kv
) -> dict:
    """Errors of one layer: kernel output stream vs the reference fed the kernel's input."""
    t0 = time.time()
    n = st.n_live
    ref = _LayerRef(l, case, st, n_layers, global_layers, attn_mode)
    mid, kv_ref = ref.attention_block(stream_in)
    xe, sc, biased = ref.route(mid)
    ids = torch.topk(biased, TOP_K, dim=1).indices
    want = ref.ffn_block(mid, xe, sc, ids)
    rel = _row_rel(out, want)
    flips, double_flips, unexplained = [], [], []
    forced = int(l > 0)
    tol_row = TOL_ROW_REL[forced]
    for r in torch.nonzero(rel > tol_row).flatten().tolist():
        best = None
        for alt in _near_tie_swaps(biased[r], ids[r]):
            ids2 = ids.clone()
            ids2[r] = alt
            e2 = float(
                _row_rel(out[r : r + 1], ref.ffn_block(mid, xe, sc, ids2)[r : r + 1])[0]
            )
            if best is None or e2 < best[0]:
                best = (e2, ids2)
        if best is not None and best[0] <= tol_row:
            flips.append(r)
            if int((best[1][r] != ids[r]).sum()) == 2:
                double_flips.append(r)
            rel[r] = best[0]
            want[r] = ref.ffn_block(mid, xe, sc, best[1])[r]
        else:
            unexplained.append(r)
            if os.environ.get("MK_NUMERICS_DEBUG_ROWS") == "1":
                _debug_row(ref, mid, xe, sc, ids, out, r, biased)
    kv_rel = _row_rel(new_kv.reshape(TP * n, -1), kv_ref.reshape(TP * n, -1))
    diff = (out - want).abs()
    res = {
        "case": case.name,
        "layer": l,
        "attn": "global" if l in global_layers else "swa",
        "experts": _kinds(n_layers)[l],
        "rows": n,
        "max_row_rel": float(rel.max()),
        "mean_row_rel": float(rel.mean()),
        "max_abs": float(diff.max()),
        "ref_rms": float(want.pow(2).mean().sqrt()),
        "kv_max_rel": float(kv_rel.max()),
        "route_flips": len(flips),
        "route_double_flips": double_flips,  # rows two near-tie swaps explain
        "route_unexplained": unexplained,
        "ref_s": round(time.time() - t0, 1),
    }
    res["ok"] = bool(
        not unexplained
        and res["max_row_rel"] <= tol_row
        and res["mean_row_rel"] <= TOL_MEAN_REL[forced]
        and res["kv_max_rel"] <= TOL_KV_REL[case.kv][forced]
        and torch.isfinite(out).all()
    )
    return res


def check_final(case, st, n_layers, hid, aux) -> dict:
    n = st.n_live
    last = aux[:n, (n_layers - 1) * HIDDEN :].float()
    want = _bf(_rms(last, RMS_EPS, final_gain()))
    rel = _row_rel(hid[:n].float(), want)
    dead_zero = bool((hid[n:] == 0).all())
    return {
        "case": case.name,
        "final_max_row_rel": float(rel.max()),
        "dead_rows_zero": dead_zero,
        "ok": bool(float(rel.max()) <= TOL_HID_REL and dead_zero),
    }


# ----------------------------------------------------------------------------- driver
def _parse_flag(s: str) -> tuple:
    import ast

    k, v = s.split("=", 1)
    return k, ast.literal_eval(v)


def run_case(
    runner: _Runner, case: Case, rank: int, out_dir: str, exact: list, base: dict
) -> None:
    st = step_inputs(case)
    if st.max_position > runner.mk.max_position:
        raise ValueError(f"{case.name} needs {st.max_position} rope rows")
    t0 = time.time()
    kv = runner.pools(case, st)
    hid, aux, cfg = runner.step(case, st, kv, case.expert_group, base)
    new_kv = runner.new_kv(case, st, kv)
    t_kernel = time.time() - t0
    cdir = os.path.join(out_dir, case.name)
    torch.save({"hid": hid, "aux": aux, "new_kv": new_kv}, f"{cdir}/rank{rank}.pt")
    exact_res = []
    for flag in exact:
        k, v = _parse_flag(flag) if flag != "repeat" else (flag, None)
        if k == "repeat":  # run-to-run determinism of the default variant
            hid2, aux2, cfg2 = runner.step(case, st, kv, case.expert_group, base)
            applied = cfg2 == cfg
        elif k == "expert_group":
            hid2, aux2, cfg2 = runner.step(case, st, kv, v, base)
            applied = cfg2.expert_groups == runner.ops.expert_groups_of(runner.kinds, v)
        else:
            hid2, aux2, cfg2 = runner.step(
                case, st, kv, case.expert_group, {**base, k: v}
            )
            applied = getattr(cfg2, k) == v and getattr(cfg, k) != v
        if os.environ.get("MK_NUMERICS_DUMP") == "1":
            torch.save(aux2, f"{cdir}/exact_{len(exact_res)}_rank{rank}.pt")
        exact_res.append(
            {
                "case": case.name,
                "flag": flag,
                "applied": bool(applied),
                "hid_identical": bool(torch.equal(hid, hid2)),
                "aux_identical": bool(torch.equal(aux[: st.n_live], aux2[: st.n_live])),
                "max_abs": float(
                    (aux[: st.n_live].float() - aux2[: st.n_live].float()).abs().max()
                ),
                "layer_max_abs": [
                    float(
                        (
                            aux[: st.n_live, l * HIDDEN : (l + 1) * HIDDEN].float()
                            - aux2[: st.n_live, l * HIDDEN : (l + 1) * HIDDEN].float()
                        )
                        .abs()
                        .max()
                    )
                    for l in range(runner.n_layers)
                ],
            }
        )
    with open(f"{cdir}/exact_rank{rank}.json", "w") as f:
        json.dump(exact_res, f)
    dist.barrier()
    # teacher-forced reference, one layer per rank
    data = [torch.load(f"{cdir}/rank{d}.pt") for d in range(TP)]
    n = st.n_live
    for l in range(runner.n_layers):
        if l % TP != rank:
            continue
        aux0 = data[0]["aux"].float()
        stream_in = (
            _bf(_rms(st.emb[:n].float(), RMS_EPS))
            if l == 0
            else aux0[:n, (l - 1) * HIDDEN : l * HIDDEN]
        )
        out = aux0[:n, l * HIDDEN : (l + 1) * HIDDEN]
        new_kv_l = torch.stack([data[d]["new_kv"][l] for d in range(TP)])
        attn_mode = (
            "exp2"
            if cfg.attn_exp2
            else ("unshifted" if cfg.attn_unshifted else "shifted")
        )
        res = check_layer(
            l,
            case,
            st,
            runner.n_layers,
            runner.global_layers,
            attn_mode,
            stream_in,
            out,
            new_kv_l,
        )
        res["attn_mode"] = attn_mode
        res["kernel_s"] = round(t_kernel, 1)
        with open(f"{cdir}/layer{l}.json", "w") as f:
            json.dump(res, f)
    if rank == 0:
        fin = check_final(case, st, runner.n_layers, data[0]["hid"], data[0]["aux"])
        fin["ranks_identical"] = all(
            torch.equal(data[0]["hid"], data[d]["hid"])
            and torch.equal(data[0]["aux"][:n], data[d]["aux"][:n])
            for d in range(1, TP)
        )
        fin["ok"] = fin["ok"] and fin["ranks_identical"]
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
            print("MK_NUMERICS " + json.dumps(r), flush=True)
        with open(f"{cdir}/final.json") as f:
            r = json.load(f)
        ok &= r["ok"]
        print("MK_NUMERICS_FINAL " + json.dumps(r), flush=True)
        for d in range(TP):
            with open(f"{cdir}/exact_rank{d}.json") as f:
                for r in json.load(f):
                    good = r["applied"] and r["hid_identical"] and r["aux_identical"]
                    ok &= good
                    if d == 0 or not good:
                        print(
                            "MK_NUMERICS_EXACT " + json.dumps({**r, "rank": d}),
                            flush=True,
                        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    # flag=value toggles applied to every selected case (bit-identity); default: DEFAULT_EXACT
    ap.add_argument("--exact", nargs="*", default=None)
    # MKConfig overrides for every run, e.g. vec_route=False (the reference check then
    # covers that variant)
    ap.add_argument("--base-cfg", nargs="*", default=[])
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--global-layers", default="1,4")
    ap.add_argument("--out-dir", default="/tmp/mk_numerics")
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--qblock-ref", action="store_true")
    ap.add_argument("--served", action="store_true")
    args = ap.parse_args()
    global _EP, _QBLOCK_REF, _SERVED
    _EP, _QBLOCK_REF, _SERVED = args.ep, args.qblock_ref, args.served
    if _SERVED and (_EP or not (_QBLOCK_REF and tp_qblock())):
        raise ValueError(
            "--served needs SGLANG_TPU_MUSE_MK_QBLOCK and --qblock-ref (TP)"
        )
    if _QBLOCK_REF and not _EP and not tp_qblock():
        raise ValueError("--qblock-ref without --ep needs SGLANG_TPU_MUSE_MK_QBLOCK")

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    if dist.get_world_size() != TP:
        raise RuntimeError(f"needs {TP} processes, got {dist.get_world_size()}")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // TP))
    torch._dynamo.config.cache_size_limit = 256
    from sglang_tpu.muse_megakernel import kernel as K
    from sglang_tpu.muse_megakernel import ops

    cases = [CASES[c] for c in args.cases]
    base = dict(_parse_flag(f) for f in args.base_cfg)
    global_layers = tuple(int(x) for x in args.global_layers.split(",") if x)
    if args.exact is None:
        exact = {c.name: [f for cn, f in DEFAULT_EXACT if cn == c.name] for c in cases}
    else:
        exact = {c.name: list(args.exact) for c in cases}
    if rank == 0:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        for c in cases:
            os.makedirs(os.path.join(args.out_dir, c.name))
    max_position = max(step_inputs(c).max_position for c in cases)
    t0 = time.time()
    runner = _Runner(K, ops, args.layers, global_layers, rank, max_position)
    if rank == 0:
        print(f"MK_NUMERICS_SETUP weights {time.time() - t0:.1f}s", flush=True)
    dist.barrier()
    for case in cases:
        t0 = time.time()
        run_case(runner, case, rank, args.out_dir, exact[case.name], base)
        if rank == 0:
            print(f"MK_NUMERICS_CASE {case.name} {time.time() - t0:.1f}s", flush=True)
    ok = True
    if rank == 0:
        ok = summarize(cases, args.layers, args.out_dir)
        print("MK_NUMERICS_PASS" if ok else "MK_NUMERICS_FAIL", flush=True)
    print(f"MK_NUMERICS_DONE rank={rank}", flush=True)
    dist.barrier()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
