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
    requant_nvfp4_blocks,
)

# Experts per compiled call, bounding the f32 temporaries.
_EXPERT_CHUNK = 32

_Requant = Callable[
    [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
]


@functools.cache
def _compiled_requant() -> _Requant:
    # Eager, the ~30 ops per chunk add ~16 min to serving startup.
    return torch.compile(requant_nvfp4_blocks, fullgraph=True, dynamic=False)


def _requant_slabs(
    w: torch.Tensor, s: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        w.dtype != torch.float4_e2m1fn_x2
        or s.dtype != torch.bfloat16
        or s.shape[1] * NVFP4_SERVED_GROUP != w.shape[1]
    ):
        raise NotImplementedError(
            "NVFP4 expert requant takes fp4 experts with bf16 block-16 scales, got "
            f"{w.dtype} {tuple(w.shape)} with {s.dtype} {tuple(s.shape)}"
        )
    u = w.view(torch.uint8)
    codes, scales = [], []
    for e0 in range(0, w.shape[0], _EXPERT_CHUNK):
        # clones: a slice's storage offset is a guard, recompiling per chunk
        c, sb = _compiled_requant()(
            u[e0 : e0 + _EXPERT_CHUNK].clone(),
            s[e0 : e0 + _EXPERT_CHUNK].clone(),
            block,
        )
        _wait_for(c, sb)
        codes.append(c)
        scales.append(sb)
    out = _materialize_packed_fp4(torch.cat(codes)), torch.cat(scales)
    _wait_for(*out)
    return out


def requant_tpu_nvfp4_experts(layer: torch.nn.Module, block: int) -> None:
    """Rewrite a fused-RS-prepared block-16 NVFP4 layer's experts in place to one bf16 scale
    per block K rows; the megakernel and the native fused-RS MoE then read only these."""
    if block not in QBLOCK_SIZES:
        raise NotImplementedError(
            f"NVFP4 expert requant to block {block}: supported {QBLOCK_SIZES}"
        )
    if not layer.__dict__.get("_tpu_fused_rs_expert_tp", False):
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
    # w13 rebound before w2 is requantized: one projection's temporaries at a time
    w13, s13 = _requant_slabs(old[0], old[1], block)
    copy_or_rebind_param(layer, "w13_weight", w13)
    copy_or_rebind_param(layer, "w13_weight_scale", s13)
    del w13, s13
    w2, s2 = _requant_slabs(old[2], old[3], block)
    copy_or_rebind_param(layer, "w2_weight", w2)
    copy_or_rebind_param(layer, "w2_weight_scale", s2)
    del w2, s2
    new = (
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
    )
    for o, n in zip(old, new):
        _free_replaced_storage(o, n)
    mark_qblock_layout(layer, block)
