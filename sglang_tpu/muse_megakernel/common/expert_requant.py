# SPDX-License-Identifier: Apache-2.0
"""SGLANG_TPU_MUSE_MK_QBLOCK: the served fp4 experts requantized once at load."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from sglang.srt.layers.utils import copy_or_rebind_param
from sglang_tpu.expert_loading import _wait_for
from sglang_tpu.fp4_moe import (
    _materialize_packed_fp4,
    mark_qblock_layout,
    qblock_layout,
)
from sglang_tpu.fp8_moe import _free_replaced_storage
from sglang_tpu.moe.nvfp4_requant import (
    NVFP4_SERVED_GROUP,
    QBLOCK_SIZES,
    probe_nvfp4_blocks,
    requant_nvfp4_blocks,
)

# Expert-TP slices per compiled call, bounding the f32 temporaries; whole experts
# (EP) are ep x larger, so a call takes ep x fewer.
_EXPERT_CHUNK = 32
_PROBE_EXPERT_CHUNK = 4

_Requant = Callable[
    [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
]


@functools.cache
def _compiled_requant() -> _Requant:
    # Eager, the ~30 ops per chunk add ~16 min to serving startup.
    return torch.compile(requant_nvfp4_blocks, fullgraph=True, dynamic=False)


@functools.cache
def _compiled_probe():
    return torch.compile(probe_nvfp4_blocks, fullgraph=True, dynamic=False)


def _check_slabs(w: torch.Tensor, s: torch.Tensor) -> None:
    if (
        w.dtype != torch.float4_e2m1fn_x2
        or s.dtype != torch.bfloat16
        or s.shape[1] * NVFP4_SERVED_GROUP != w.shape[1]
    ):
        raise NotImplementedError(
            "NVFP4 expert requant takes fp4 experts with bf16 block-16 scales, got "
            f"{w.dtype} {tuple(w.shape)} with {s.dtype} {tuple(s.shape)}"
        )


def _requant_slabs(
    w: torch.Tensor, s: torch.Tensor, block: int, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_slabs(w, s)
    u = w.view(torch.uint8)
    codes, scales = [], []
    for e0 in range(0, w.shape[0], chunk):
        # clones: a slice's storage offset is a guard, recompiling per chunk
        c, sb = _compiled_requant()(
            u[e0 : e0 + chunk].clone(),
            s[e0 : e0 + chunk].clone(),
            block,
        )
        _wait_for(c, sb)
        codes.append(c)
        scales.append(sb)
    out = _materialize_packed_fp4(torch.cat(codes)), torch.cat(scales)
    _wait_for(*out)
    return out


def requant_tpu_nvfp4_experts(
    layer: torch.nn.Module, block: int, *, ep_ranks: int = 0
) -> None:
    """Rewrite a fused-RS-prepared block-16 NVFP4 layer's experts in place to one bf16 scale
    per block K rows; the megakernel and the native fused-RS MoE then read only these.
    ep_ranks: under EP, whole experts per rank, converted next to the EP layout, which only
    the Muse EP programs read."""
    if block not in QBLOCK_SIZES:
        raise NotImplementedError(
            f"NVFP4 expert requant to block {block}: supported {QBLOCK_SIZES}"
        )
    if not ep_ranks and not layer.__dict__.get("_tpu_fused_rs_expert_tp", False):
        raise NotImplementedError(
            "NVFP4 expert requant needs the fused-RS expert-TP layout (the only native "
            "MoE with a block-scaled W4A16 arm)"
        )
    if qblock_layout(layer):
        raise NotImplementedError("NVFP4 experts already requantized")
    old = (
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
    )
    chunk = max(1, _EXPERT_CHUNK // max(ep_ranks, 1))
    # w13 rebound before w2 is requantized: one projection's temporaries at a time
    w13, s13 = _requant_slabs(old[0], old[1], block, chunk)
    copy_or_rebind_param(layer, "w13_weight", w13)
    copy_or_rebind_param(layer, "w13_weight_scale", s13)
    del w13, s13
    w2, s2 = _requant_slabs(old[2], old[3], block, chunk)
    copy_or_rebind_param(layer, "w2_weight", w2)
    copy_or_rebind_param(layer, "w2_weight_scale", s2)
    del w2, s2
    del old
    mark_qblock_layout(layer, block)


def probe_tpu_nvfp4_block_scales(layer: torch.nn.Module, block: int) -> list[float]:
    """SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES: requantize a fused-RS-prepared layer's fp4
    experts to one scale per block K rows but keep the block-16 layout, so every kernel
    serves the lossy weights unchanged. Returns each tensor's relative L2 error."""
    errs = []
    for name in ("w13_weight", "w2_weight"):
        w = getattr(layer, name).data
        s = getattr(layer, f"{name}_scale").data
        _check_slabs(w, s)
        u = w.view(torch.uint8)
        codes, scales, err2, ref2 = [], [], 0.0, 0.0
        # 32 experts needed 88 GiB of program memory on TPU7x; 4 fit next to the weights.
        for e0 in range(0, w.shape[0], _PROBE_EXPERT_CHUNK):
            c, s16, d2, r2 = _compiled_probe()(
                u[e0 : e0 + _PROBE_EXPERT_CHUNK].clone(),
                s[e0 : e0 + _PROBE_EXPERT_CHUNK].clone(),
                block,
            )
            codes.append(c)
            scales.append(s16)
            err2 += d2.item()
            ref2 += r2.item()
        new_w, new_s = _materialize_packed_fp4(torch.cat(codes)), torch.cat(scales)
        del codes, scales
        _wait_for(new_w, new_s)
        copy_or_rebind_param(layer, name, new_w)
        copy_or_rebind_param(layer, f"{name}_scale", new_s)
        _free_replaced_storage(w, getattr(layer, name).data)
        _free_replaced_storage(s, getattr(layer, f"{name}_scale").data)
        del w, s, u, new_w, new_s
        errs.append((err2 / ref2) ** 0.5 if ref2 else 0.0)
    return errs
