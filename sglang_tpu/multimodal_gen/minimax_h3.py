# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 on TPU workers: one compiled shape per video size.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes, and a 50-block H3 stack takes
  minutes per new shape. H3 packs ``[text | cond | audio | video]`` into one
  attention segment padded to 64 rows, so every prompt length is a new
  shape. Separately, the breakable-CUDA-graph break point around H3's
  attention splits every compiled block into two graphs; BCG is CUDA-only.
- How: :func:`install` (1) wraps the packed-sequence builders so the padded
  length is a multiple of 512 (padding rows are tagged padding, carry zero
  RoPE positions and stay in ``cu_seqlens``' trailing segment) and (2) rebinds
  ``_minimax_h3_attention_core_bcg`` to the undecorated core. The attention
  backend reads the live length on device (``trailing_padding_on_device``),
  so the blocks take no host bounds and compile once per padded length.
- Upstream PR or issue: none.
- Remove when: the H3 packed builder takes a platform-provided alignment and
  its attention break point is skipped where BCG cannot run.
"""

from __future__ import annotations

import functools

import torch

# Most prompt lengths of one video size share a 512 bucket; the padding costs
# at most 511 extra rows next to ~28k video rows at 1344x768x4s.
TPU_PACKED_SEQUENCE_ALIGNMENT = 512


def pad_packed_sequence(packed: dict, alignment: int) -> dict:
    """Extend a packed layout's trailing padding to a multiple of ``alignment``."""
    seq_len = int(packed["seq_len"])
    padded = (seq_len + alignment - 1) // alignment * alignment
    if padded == seq_len:
        return packed
    extra = padded - seq_len
    out = dict(packed)
    out["seq_len"] = padded
    positions = packed["img_position_ids"]
    out["img_position_ids"] = torch.cat(
        (positions, positions.new_zeros(extra, *positions.shape[1:]))
    )
    tags = packed["token_tags"]
    out["token_tags"] = torch.cat((tags, tags.new_full((extra,), -1)))
    cu_seqlens = packed["cu_seqlens"].clone()
    cu_seqlens[-1] = padded
    out["cu_seqlens"] = cu_seqlens
    return out


def _padded(builder):
    @functools.wraps(builder)
    def build(*args, **kwargs):
        return pad_packed_sequence(
            builder(*args, **kwargs), TPU_PACKED_SEQUENCE_ALIGNMENT
        )

    build.__wrapped_for_tpu__ = True
    return build


def install() -> None:
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        packed_sequence,
    )

    for name in (
        "minimax_h3_packed_sequence",
        "minimax_h3_packed_sequence_ref2va_blocks",
    ):
        builder = getattr(packed_sequence, name)
        if not getattr(builder, "__wrapped_for_tpu__", False):
            setattr(packed_sequence, name, _padded(builder))
    dit._minimax_h3_attention_core_bcg = dit._minimax_h3_attention_core_impl
