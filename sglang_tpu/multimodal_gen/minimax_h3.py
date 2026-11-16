# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 on TPU workers: one compiled shape per video size.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes, and a 50-block H3 stack takes
  minutes per new shape. H3 packs ``[text | cond | audio | video]`` into one
  attention segment padded to 64 rows, so every prompt length is a new
  shape. The token refiner's blocks also match the regional-compile
  condition and run at the prompt length. Separately, the breakable-CUDA-graph
  break point around H3's attention splits every compiled block into two
  graphs; BCG is CUDA-only.
- How: :func:`install` (1) sets the alignment the packed-sequence builders
  and the ComfyUI layout read to 512 rows (padding rows are tagged padding,
  carry zero RoPE positions and stay in ``cu_seqlens``' trailing segment),
  (2) compiles only the 50 DiT blocks: the refiner runs once per request,
  eager, and (3) rebinds ``_minimax_h3_attention_core_bcg`` to the undecorated
  core. The attention backend reads the live length
  on device (``trailing_padding_on_device``), so the blocks take no host
  bounds and compile once per padded length.
- Upstream PR or issue: none.
- Remove when: the H3 packed builder takes a platform-provided alignment, the
  refiner pads to a bucket, and the attention break point is skipped where BCG
  cannot run.
"""

from __future__ import annotations

# Most prompt lengths of one video size share a 512 bucket; the padding costs
# at most 511 extra rows next to ~32k packed rows at 1344x768x4s.
TPU_PACKED_SEQUENCE_ALIGNMENT = 512


def _is_dit_block(name: str, module: object) -> bool:
    from sglang.multimodal_gen.configs.models.fsdp import is_block

    return is_block(name, module) and not name.startswith("token_refiner.")


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
