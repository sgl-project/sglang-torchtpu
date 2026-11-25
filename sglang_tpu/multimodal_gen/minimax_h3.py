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
- How: :func:`install` (1) sets the alignment that
  ``packed_sequence.minimax_h3_padded_seq_len``, the one padding function of
  the packed-sequence builders and the ComfyUI layout, rounds to: 512 rows
  (padding rows are tagged padding, carry zero RoPE positions and stay in
  ``cu_seqlens``' trailing segment),
  (2) compiles only the 50 DiT blocks: the refiner runs once per request,
  eager, and (3) rebinds ``_minimax_h3_attention_core_bcg`` to the undecorated
  core. The attention backend reads the live length
  on device (``trailing_padding_on_device``), so the blocks take no host
  bounds and compile once per padded length.
- Each SP rank embeds its own row shard, and how many video and audio rows
  fall in a shard moves with the prompt length. Those eager ops compiled again
  for every new prompt length: 6 s of a Pika-H3-Flex request. (4) The serving
  ``_embed`` projects every video and audio row into one padded-length buffer
  and returns the rank's slice, so only the text-row copy has a
  prompt-length shape. Each row's projection and placement are unchanged.
  ``SGLANG_TPU_OPT_H3_SHARD_EMBEDDING`` keeps the stock per-shard ``_embed``
  for deployments that compile their prompt lengths at startup.
- Upstream PR or issue: none.
- Remove when: the H3 packed builder takes a platform-provided alignment, the
  refiner pads to a bucket, the embedding takes shard-independent shapes, and
  the attention break point is skipped where BCG cannot run.
"""

from __future__ import annotations

import functools

import torch

from sglang_tpu.envs import envs as tpu_envs

# Most prompt lengths of one video size share a 512 bucket; the padding costs
# at most 511 extra rows next to ~32k packed rows at 1344x768x4s.
TPU_PACKED_SEQUENCE_ALIGNMENT = 512
_EMBED_CHUNK_ROWS = 8192


def _embed_full_rows(embed):
    """The serving ``_embed`` with shapes that do not move with the prompt."""

    @functools.wraps(embed)
    def embed_rows(self, **kwargs):
        text_len = kwargs.get("refined_prompt_embeds_length")
        if (
            kwargs.get("out") is not None
            or kwargs.get("local_embedding_layout") is None
            or not isinstance(text_len, int)
        ):
            return embed(self, **kwargs)
        x, audio_x = kwargs["x"], kwargs["audio_x"]
        img_pos, audio_pos = kwargs["img_pos"], kwargs["audio_pos"]
        text = kwargs["text_embeddings_selected"][:text_len]
        if int(text.shape[-1]) != self.hidden_size:
            return embed(self, **kwargs)
        rows = torch.zeros(
            (x.shape[1], self.hidden_size),
            device=kwargs["device"],
            dtype=torch.bfloat16,
        )
        # The packed sequence starts with the text rows.
        rows[:text_len].copy_(text.to(device=rows.device, dtype=torch.bfloat16))
        for pos, latents, project in (
            (img_pos, x, self.video_patch_proj),
            (audio_pos, audio_x, self.audio_patch_proj),
        ):
            latents = latents.view(-1, latents.shape[-1])
            # Chunks bound the float32 projection's scratch.
            for chunk in pos.split(_EMBED_CHUNK_ROWS) if pos.numel() else ():
                picked = latents.index_select(0, chunk).to(torch.float32)
                embedded, _ = project(picked)
                rows.index_copy_(0, chunk, embedded.to(torch.bfloat16))
        # Its own storage: the compiled blocks guard input offsets, and the
        # full buffer is not kept for the step.
        local = rows[kwargs["row_start"] : kwargs["row_stop"]].clone()
        return local, self._time_embedding(kwargs["unique_timesteps"])

    embed_rows.__wrapped_by_sglang_tpu__ = True
    return embed_rows


def _is_dit_block(name: str, module: object) -> bool:
    from sglang.multimodal_gen.configs.models.fsdp import is_block

    return is_block(name, module) and not name.startswith("token_refiner.")


def install() -> None:
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        packed_sequence,
    )

    # Every builder pads through packed_sequence.minimax_h3_padded_seq_len,
    # which reads the alignment when called. Reading it first raises on a
    # rename; a multiple of it keeps the DiT's sequence-parallel admission,
    # which checks the core alignment, valid.
    if (
        TPU_PACKED_SEQUENCE_ALIGNMENT
        % packed_sequence.MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    ):
        raise ValueError(
            f"{TPU_PACKED_SEQUENCE_ALIGNMENT} rows is not a multiple of "
            "packed_sequence.MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT"
        )
    packed_sequence.MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = TPU_PACKED_SEQUENCE_ALIGNMENT
    dit.MiniMaxH3DiTModel._compile_conditions = [_is_dit_block]
    dit._minimax_h3_attention_core_bcg = dit._minimax_h3_attention_core_impl
    model = dit.MiniMaxH3DiTModel
    if not tpu_envs.SGLANG_TPU_OPT_H3_SHARD_EMBEDDING.get() and not getattr(
        model._embed, "__wrapped_by_sglang_tpu__", False
    ):
        model._embed = _embed_full_rows(model._embed)
