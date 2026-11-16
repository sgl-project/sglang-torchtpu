# SPDX-License-Identifier: Apache-2.0
"""Serving weights in the EP layout (common/ep_layout.py) from an EP-loaded model, in place.

A server at EP size == TP size loads 32 whole experts per rank in the fused-RS layout: fp4
w13 [E, H_e, 2I] (natural [gate | up] columns) and w2 [E, I, H_e] as e2m1 pairs along N,
fp8 w13 / w2 with (E, 1, 1, N) channel scales. This module rewrites them to what decode/spep
and prefill/spep read: hidden rows (and fp4 intermediate rows) in kperm order, fp4 gate / up
columns in kperm order, then w13 columns in chunk_cols order, bf16 scales. fp4 needs
SGLANG_TPU_MUSE_MK_QBLOCK=256 first: kperm permutes within 256-row blocks, so the block-256
requant of the natural order equals ep_layout.requant_fp4_kperm's of the kperm order.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from sglang.srt.layers.utils import copy_or_rebind_param
from sglang_tpu.expert_loading import _wait_for
from sglang_tpu.fp4_moe import _materialize_packed_fp4, qblock_layout
from sglang_tpu.muse_megakernel.common.ep_layout import (
    KPERM_BLOCK,
    chunk_cols,
    kperm,
)

# Experts per converted slab, bounding the unpacked fp4 code temporaries (8 x 32 MiB x 32).
_EXPERT_CHUNK = 8


def mark_ep_layout(experts: torch.nn.Module) -> None:
    experts.__dict__["_tpu_muse_ep_layout"] = True


def ep_layout(experts: torch.nn.Module) -> bool:
    """Whether the experts hold the EP megakernel layout, which no native MoE reads."""
    return experts.__dict__.get("_tpu_muse_ep_layout", False)


def _index(order: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(order, dtype=torch.int32, device=device)


@functools.cache
def _compiled_permute_fp4_cols():
    return torch.compile(_permute_fp4_cols, fullgraph=True, dynamic=False)


def _permute_fp4_cols(packed: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """[e, K, N/2] e2m1 pairs (low nibble = even column) with columns reordered: column j of
    the result is column order[j] of the input."""
    e, k, half = packed.shape
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(e, k, 2 * half)
    codes = codes.index_select(2, order).reshape(e, k, half, 2)
    return codes[..., 0] | (codes[..., 1] << 4)


def _rows(t: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return t.index_select(1, order).contiguous()


def _fp4_to_ep(experts: torch.nn.Module, chunk: int) -> None:
    w13 = experts.w13_weight.data
    _e, he, half = w13.shape
    two_i = 2 * half
    inter = experts.w2_weight.shape[1]
    if qblock_layout(experts) != KPERM_BLOCK:
        raise NotImplementedError(
            "EP fp4 experts need SGLANG_TPU_MUSE_MK_QBLOCK=256 (block scales that kperm keeps "
            f"whole), got the block-{qblock_layout(experts) or 16} layout"
        )
    dev = w13.device
    kp_he, kp_i = _index(kperm(he), dev), _index(kperm(inter), dev)
    # gate / up columns in kperm order, then chunk_cols: column j is gu[cc[j]]
    cols = _index(kperm(two_i)[chunk_cols(two_i, chunk)], dev)
    u13 = w13.view(torch.uint8)
    slabs = []
    for e0 in range(0, u13.shape[0], _EXPERT_CHUNK):
        slab = _rows(u13[e0 : e0 + _EXPERT_CHUNK], kp_he)
        slabs.append(_compiled_permute_fp4_cols()(slab, cols))
    new13 = _materialize_packed_fp4(torch.cat(slabs))
    s13 = experts.w13_weight_scale.data.to(torch.bfloat16).index_select(2, cols)
    _wait_for(new13, s13)
    copy_or_rebind_param(experts, "w13_weight", new13)
    copy_or_rebind_param(experts, "w13_weight_scale", s13.contiguous())
    del slabs, new13, s13, u13, w13
    w2 = _materialize_packed_fp4(_rows(experts.w2_weight.data.view(torch.uint8), kp_i))
    s2 = experts.w2_weight_scale.data.to(torch.bfloat16).contiguous()
    _wait_for(w2, s2)
    copy_or_rebind_param(experts, "w2_weight", w2)
    copy_or_rebind_param(experts, "w2_weight_scale", s2)


def _channel_scales(s: torch.Tensor) -> torch.Tensor:
    """(E, 1, 1, N) or (E, 1, N) channel scales -> the (E, 2, N) bf16 rows the kernels read
    (only row 0 is read; fused-RS expert TP keeps two)."""
    if s.dim() == 3 and s.shape[1] == 2:
        return s.to(torch.bfloat16).contiguous()
    s = s.reshape(s.shape[0], 1, s.shape[-1]).to(torch.bfloat16)
    return s.expand(-1, 2, -1).contiguous()


def _fp8_to_ep(experts: torch.nn.Module, chunk: int) -> None:
    w13 = experts.w13_weight.data
    _e, he, two_i = w13.shape
    dev = w13.device
    kp_he = _index(kperm(he), dev)
    cols = _index(chunk_cols(two_i, chunk), dev)
    u13 = w13.view(torch.uint8)
    new13 = torch.cat(
        [
            _rows(u13[e0 : e0 + _EXPERT_CHUNK], kp_he).index_select(2, cols)
            for e0 in range(0, u13.shape[0], _EXPERT_CHUNK)
        ]
    ).view(torch.float8_e4m3fn)
    s13 = _channel_scales(experts.w13_weight_scale.data).index_select(2, cols)
    s2 = _channel_scales(experts.w2_weight_scale.data)
    _wait_for(new13, s13, s2)
    copy_or_rebind_param(experts, "w13_weight", new13.contiguous())
    copy_or_rebind_param(experts, "w13_weight_scale", s13.contiguous())
    copy_or_rebind_param(experts, "w2_weight_scale", s2)


def convert_experts_to_ep(experts: torch.nn.Module, kind: str, chunk: int) -> None:
    """Rewrite one layer's local whole experts to the EP layout in place (chunk: the
    MKConfig's inter_local, I / tp)."""
    if ep_layout(experts):
        raise NotImplementedError("experts already in the EP layout")
    if kind == "fp4":
        _fp4_to_ep(experts, chunk)
    elif kind == "fp8":
        _fp8_to_ep(experts, chunk)
    else:
        raise NotImplementedError(f"EP expert layout for {kind} experts")
    mark_ep_layout(experts)


def ep_expert_refs(experts: torch.nn.Module) -> list[torch.Tensor]:
    """(w13, w13_scale, w2, w2_scale) of converted experts, as decode/spep reads them."""
    if not ep_layout(experts):
        raise NotImplementedError("experts not converted to the EP layout")
    return [
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
    ]
