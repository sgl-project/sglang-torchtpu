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
  all-to-alls).
- How: :func:`install` (1) sets the alignment the packed-sequence builders
  and the ComfyUI layout read to 512 rows (padding rows are tagged padding,
  carry zero RoPE positions and stay in ``cu_seqlens``' trailing segment),
  (2) compiles only the 50 DiT blocks: the refiner runs once per request,
  eager, (3) rebinds ``_minimax_h3_attention_core_bcg`` to the undecorated
  core, and (4) marks each block spmd-safe when it compiles as its own region
  (``--regional-compile``), so each compiled block keeps its collectives in
  one program (every rank runs the same block graph); eager blocks and a
  whole-DiT compile run unmarked. The attention backend reads the live length
  on device (``trailing_padding_on_device``), so the blocks take no host
  bounds and compile once per padded length.
- Upstream PR or issue: none.
- Remove when: the H3 packed builder takes a platform-provided alignment, the
  refiner pads to a bucket, the attention break point is skipped where BCG
  cannot run, and regional compile marks spmd-safe regions.
"""

from __future__ import annotations

import functools

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Most prompt lengths of one video size share a 512 bucket; the padding costs
# at most 511 extra rows next to ~32k packed rows at 1344x768x4s.
TPU_PACKED_SEQUENCE_ALIGNMENT = 512


def _is_dit_block(name: str, module: object) -> bool:
    from sglang.multimodal_gen.configs.models.fsdp import is_block

    return is_block(name, module) and not name.startswith("token_refiner.")


def _mark_blocks_spmd_safe(block_cls) -> None:
    """Annotate a block's graph spmd-safe when the block compiles as its own
    region: regional compile sets the block's ``_compiled_call_impl``, a
    whole-DiT compile does not. Eager calls take the plain forward, since
    spmd_safe's eager branch syncs every block's inputs and outputs. The
    compile cache key already covers the annotation: the TPU kernel-inputs tag
    hashes this file and the torch_tpu version."""
    if getattr(block_cls.forward, "__spmd_safe_for_tpu__", False):
        return
    try:
        from torch_tpu._internal.distributed.spmd_util import spmd_safe
    except ImportError:
        # A private API: without it, compiled blocks keep split collectives.
        logger.warning_once(
            "torch_tpu has no spmd_safe; compiled MiniMax-H3 blocks keep "
            "their collectives split."
        )
        return
    forward = block_cls.forward
    marked = spmd_safe(forward)

    @functools.wraps(forward)
    def forward_spmd_safe_as_a_region(*args, **kwargs):
        # args[0] is the block; this signature keeps the graph's input names.
        if torch.compiler.is_compiling() and args[0]._compiled_call_impl is not None:
            return marked(*args, **kwargs)
        return forward(*args, **kwargs)

    forward_spmd_safe_as_a_region.__spmd_safe_for_tpu__ = True
    block_cls.forward = forward_spmd_safe_as_a_region


def install() -> None:
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        packed_sequence,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        comfyui_step,
    )

    # The builders read the alignment when called. Reading it first raises on
    # a rename; a multiple of it keeps the DiT's sequence-parallel admission,
    # which checks the core alignment, valid.
    for module in (packed_sequence, comfyui_step):
        if TPU_PACKED_SEQUENCE_ALIGNMENT % module.MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT:
            raise ValueError(
                f"{TPU_PACKED_SEQUENCE_ALIGNMENT} rows is not a multiple of "
                f"{module.__name__}.MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT"
            )
        module.MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = TPU_PACKED_SEQUENCE_ALIGNMENT
    dit.MiniMaxH3DiTModel._compile_conditions = [_is_dit_block]
    dit._minimax_h3_attention_core_bcg = dit._minimax_h3_attention_core_impl
    _mark_blocks_spmd_safe(dit.MiniMaxH3DiTBlock)
