# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 on TPU workers: one compiled shape per video size.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes, and a 50-block H3 stack takes
  minutes per new shape. H3 packs ``[text | cond | audio | video]`` into one
  attention segment padded to 64 rows, so every prompt length is a new
  shape. The token refiner's blocks also match the regional-compile
  condition and run at the prompt length. Separately, the breakable-CUDA-graph
  break point around H3's attention splits every compiled block into two
  graphs; BCG is CUDA-only. torch_tpu also splits a compiled graph at every
  collective (5 programs per block, one per side of its two Ulysses
  all-to-alls), and XLA turns an all-to-all of the flat 1-D buffer that
  ``usp._usp_all_to_all_single`` sends into a chain of relayout copies
  (~9 ms of a ~29 ms block at 32768 rows).
- How: :func:`install` (1) wraps the packed-sequence builders (and the
  ComfyUI step's references to them) so the padded length is a multiple of
  512 (padding rows are tagged padding, carry zero RoPE positions and stay in
  ``cu_seqlens``' trailing segment), (2) compiles only the 50 DiT blocks: the
  refiner runs once per request, eager, (3) rebinds
  ``_minimax_h3_attention_core_bcg`` to the undecorated core, (4) marks the
  blocks spmd-safe, so each block keeps its collectives in one program
  (every rank runs the same block graph), and (5) sends the Ulysses
  all-to-all as a ``[dim0, -1, last two dims]`` view: the same bytes and
  rank chunks as the flat buffer, lane-dense. The attention
  backend reads the live length on device (``trailing_padding_on_device``),
  so the blocks take no host bounds and compile once per padded length.
- Upstream PR or issue: none.
- Remove when: the H3 packed builder takes a platform-provided alignment, the
  refiner pads to a bucket, the attention break point is skipped where BCG
  cannot run, regional compile marks spmd-safe regions, and the flat
  all-to-all lowers without copies.
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


def _is_dit_block(name: str, module: object) -> bool:
    from sglang.multimodal_gen.configs.models.fsdp import is_block

    return is_block(name, module) and not name.startswith("token_refiner.")


# The annotation is not part of torch's compile cache key.
_SPMD_SAFE_CACHE_TAG = "minimax-h3-spmd-safe-blocks-v1"


def _mark_blocks_spmd_safe(block_cls) -> None:
    import torch.compiler.config as compiler_config

    try:
        from torch_tpu._internal.distributed.spmd_util import spmd_safe
    except ImportError:
        # A private torch_tpu API: without it, blocks keep split collectives.
        return
    if getattr(block_cls.forward, "__spmd_safe_for_tpu__", False):
        return
    forward = spmd_safe(block_cls.forward)
    forward.__spmd_safe_for_tpu__ = True
    block_cls.forward = forward
    tags = [t for t in str(compiler_config.cache_key_tag or "").split("|") if t]
    if _SPMD_SAFE_CACHE_TAG not in tags:
        compiler_config.cache_key_tag = "|".join([*tags, _SPMD_SAFE_CACHE_TAG])


def _lane_dense_all_to_all_single(flat_all_to_all):
    from sglang.multimodal_gen.runtime.layers import usp

    @functools.wraps(flat_all_to_all)
    def all_to_all(x: torch.Tensor, role: str | None = None) -> torch.Tensor:
        group = usp.get_sp_group().ulysses_group
        world = torch.distributed.get_world_size(group)
        if x.dim() < 3 or x.shape[0] % world:
            return flat_all_to_all(x, role)
        x_shape = x.shape
        x = x.contiguous().view(x_shape[0], -1, x_shape[-2] * x_shape[-1])
        if role is None:
            output = torch.empty_like(x)
        else:
            output = usp._a2a_staging_buffer(role, x.shape, x.dtype, x.device)
        torch.distributed.all_to_all_single(output, x, group=group)
        return output.reshape(x_shape)

    all_to_all.__wrapped_for_tpu__ = True
    return all_to_all


def install() -> None:
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        packed_sequence,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        comfyui_step,
    )

    for name in (
        "minimax_h3_packed_sequence",
        "minimax_h3_packed_sequence_ref2va_blocks",
    ):
        builder = getattr(packed_sequence, name)
        if not getattr(builder, "__wrapped_for_tpu__", False):
            builder = _padded(builder)
            setattr(packed_sequence, name, builder)
        setattr(comfyui_step, name, builder)
    if not getattr(comfyui_step.comfyui_layout_to_packed, "__wrapped_for_tpu__", False):
        comfyui_step.comfyui_layout_to_packed = _padded(
            comfyui_step.comfyui_layout_to_packed
        )
    dit.MiniMaxH3DiTModel._compile_conditions = [_is_dit_block]
    dit._minimax_h3_attention_core_bcg = dit._minimax_h3_attention_core_impl
    _mark_blocks_spmd_safe(dit.MiniMaxH3DiTBlock)
    if not getattr(usp._usp_all_to_all_single, "__wrapped_for_tpu__", False):
        usp._usp_all_to_all_single = _lane_dense_all_to_all_single(
            usp._usp_all_to_all_single
        )
