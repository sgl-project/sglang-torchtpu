# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 on TPU workers: one compiled block program for every prompt.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The DiT takes the prompt at its exact
  row count: both text towers attend it and each of the 60 joint blocks
  cross-attends it from video and audio, so every prompt length compiles every
  block program again. Under Ulysses each rank's video shard and RoPE are
  views at a rank-dependent storage offset, which torch_tpu guards. The RoPE
  multiplies [..., D / 2, 2, 2] tables, a 2-wide minor dimension. The Reason1
  positions and kept prompt rows are an int64 cumsum, an int64 sum and a
  boolean index on the device, which torch_tpu aborts.
- How: :func:`install` replaces, for TPU tensors:
  (1) the DiT's ``forward`` and its text-encoder and joint block forwards:
      outside the compiled blocks the prompt pads with zeros to ``TEXT_ROWS``
      rows (the tokenizer keeps at most 512 user tokens) and a device boolean
      mask marks the real rows. Every attention over prompt keys is the
      float32 SDPA the stock TPU backend runs for these short key sets, with
      the padded keys masked out, so one program serves every prompt;
  (2) the video shard reaches the blocks at offset 0, and the blocks are
      spmd-safe: every rank runs one program;
  (3) RoPE as ``x * cos + (x @ R) * sin`` (R a signed pair swap), bitwise the
      stock rotation;
  (4) the Reason1 text positions, the kept rows and the prompt length from
      host copies of the mask.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prompt bucket with a key
  mask, SP shards reach the blocks at offset 0, and the text conditioning
  derives positions and rows on the host.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import residual_gate_fp32
from sglang.multimodal_gen.configs.pipeline_configs import kandinsky6 as config
from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as qwen_config
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.models.dits import kandinsky6 as dit
from sglang.multimodal_gen.runtime.models.encoders import qwen_vl
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6 import (
    denoising as stage,
)
from sglang_tpu.multimodal_gen.rope import _pair_swap_matrix

# The Reason1 tokenizer keeps at most 512 user tokens (the 129-token template
# is dropped), so one bucket holds every prompt.
TEXT_ROWS = 512

# Reason1 prompts (with the 129-token template) shorter than this pad to it: one
# eager encoder shape for most prompts, still on the float32 SDPA path.
PROMPT_TOKENS = 512

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_MODEL = dit.Kandinsky6Transformer3DModel
_STOCK_FORWARD = _MODEL.forward
_STOCK_ENCODER_BLOCK_FORWARD = dit.Kandinsky6TransformerEncoderBlock.forward
_STOCK_FUSED_BLOCK_FORWARD = dit.Kandinsky6FusedTransformerDecoderBlock.forward
_STOCK_TEXT_ROPE_INDEX = qwen_vl.QwenVLModelBase._get_text_rope_index
_STOCK_EXTRACT_MASKED_HIDDEN = qwen_config._extract_masked_hidden
_STOCK_TEXT_ROPE_POS = stage.Kandinsky6DenoisingStage._text_rope_pos
_STOCK_APPLY_ROTARY = dit._apply_rotary
_STOCK_TOKENIZE_PROMPT = config.Kandinsky6TI2VAPipelineConfig.tokenize_prompt


def _tokenize_prompt(self, prompt, tokenizer, tok_kwargs) -> dict:
    """Reason1 tokens right-padded to ``PROMPT_TOKENS``. The encoder is causal,
    so real rows never see the padding, and only real rows reach the DiT."""
    inputs = _STOCK_TOKENIZE_PROMPT(self, prompt, tokenizer, tok_kwargs)
    if (
        tok_kwargs.get("padding") is True
        and inputs["input_ids"].shape[1] < PROMPT_TOKENS
    ):
        padded = dict(tok_kwargs, padding="max_length", max_length=PROMPT_TOKENS)
        inputs = _STOCK_TOKENIZE_PROMPT(self, prompt, tokenizer, padded)
    return inputs


def rope_pair(rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``[..., D / 2, 2, 2]`` rotation table as full-width float32 (cos, sin)."""
    cos = rope[..., 0, 0].repeat_interleave(2, dim=-1)
    sin = rope[..., 1, 0].repeat_interleave(2, dim=-1)
    return cos, sin


def _apply_rotary(x, rope, dtype=None):
    """``_apply_rotary``; a :func:`rope_pair` rotates as ``x * cos + (x @ R) * sin``.

    R swaps each pair with a sign, so in bf16 every output of ``x @ R`` is one
    exact product, and each output is the stock pair of float32 products and
    one add: bitwise the stock rotation, without its 2-wide minor dimension.
    """
    if not isinstance(rope, tuple):
        return _STOCK_APPLY_ROTARY(x, rope, dtype)
    dtype = x.dtype if dtype is None else dtype
    cos, sin = rope
    x = x.to(dtype)
    swapped = torch.matmul(x, _pair_swap_matrix(x.shape[-1], dtype, x.device))
    return (x.float() * cos + swapped.float() * sin).to(dtype)


def _on_device(x: torch.Tensor | None) -> bool:
    return x is not None and x.device.type in _DEVICE_TYPES


def _text_rope_index(self, input_ids, attention_mask):
    """Reason1's text-only positions from a host copy of the mask: the stock
    int64 cumsum and max on the device abort torch_tpu."""
    if not _on_device(attention_mask):
        return _STOCK_TEXT_ROPE_INDEX(self, input_ids, attention_mask)
    position_ids, deltas = _STOCK_TEXT_ROPE_INDEX(self, None, attention_mask.cpu())
    device = attention_mask.device
    return position_ids.to(device), deltas.to(device)


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    """The kept prompt rows by host indices, not a device boolean mask."""
    if not _on_device(hidden_states):
        return _STOCK_EXTRACT_MASKED_HIDDEN(hidden_states, mask)
    bool_mask = mask.bool().cpu()
    rows = bool_mask.flatten().nonzero(as_tuple=True)[0]
    selected = hidden_states.flatten(0, 1).index_select(
        0, rows.to(hidden_states.device)
    )
    return torch.split(selected, bool_mask.sum(dim=1).tolist(), dim=0)


def _text_rope_pos(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The prompt's RoPE positions, counted on the host."""
    if _on_device(mask):
        mask = mask.cpu()
    return _STOCK_TEXT_ROPE_POS(mask, device)


def text_rows(num_rows: int) -> int:
    """Padded prompt row count: ``TEXT_ROWS``, or a multiple of it."""
    return max(TEXT_ROWS, -(-num_rows // TEXT_ROWS) * TEXT_ROWS)


def text_attention(
    attn: dit.Kandinsky6Attention,
    hidden_states: torch.Tensor,
    text: torch.Tensor | None,
    rope: torch.Tensor | None,
    text_mask: torch.Tensor,
) -> torch.Tensor:
    """``Kandinsky6Attention.forward`` over the real prompt keys only.

    ``text`` holds the keys (``None``: self-attention over ``hidden_states``,
    rotated by ``rope``); ``text_mask`` (bool ``[S_text]``) marks the real
    rows. The projections, norms and rotation are the stock ones; the
    attention is the float32 SDPA the stock TPU backend falls back to for
    fewer than 512 keys, with the padded keys masked.
    """
    query, _ = attn.to_query(hidden_states)
    kv_source = hidden_states if text is None else text
    key, _ = attn.to_key(kv_source)
    value, _ = attn.to_value(kv_source)

    shape, kv_shape = query.shape[:-1], key.shape[:-1]
    query = query.reshape(*shape, attn.local_num_heads, -1)
    key = key.reshape(*kv_shape, attn.local_num_heads, -1)
    value = value.reshape(*kv_shape, attn.local_num_heads, -1)

    query_dtype, key_dtype = query.dtype, key.dtype
    query = attn.query_norm(query.float())
    query = (
        dit._apply_rotary(query, rope, query_dtype)
        if rope is not None
        else query.to(query_dtype)
    )
    key = attn.key_norm(key.float())
    key = (
        dit._apply_rotary(key, rope, key_dtype)
        if rope is not None and text is None
        else key.to(key_dtype)
    )

    out = F.scaled_dot_product_attention(
        query.transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
        attn_mask=text_mask.view(1, 1, 1, -1),
        scale=attn.attention.softmax_scale,
    ).to(query_dtype)
    out, _ = attn.out_layer(out.transpose(1, 2).flatten(-2, -1))
    return out


def _encoder_block_forward(self, x, time_embed, rope, text_mask=None):
    """``Kandinsky6TransformerEncoderBlock.forward``; with ``text_mask`` the
    self-attention reads only the real prompt rows."""
    if text_mask is None:
        return _STOCK_ENCODER_BLOCK_FORWARD(self, x, time_embed, rope)
    self_attn_params, ff_params = torch.chunk(
        self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
    )
    shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
    out = dit._norm_scale_shift(self.self_attention_norm, x, shift, scale)
    out = text_attention(self.self_attention, out, None, rope, text_mask)
    x = (x.float() + gate.float() * out.float()).type_as(x)

    ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
    out = dit._norm_scale_shift(self.feed_forward_norm, x, ff_shift, ff_scale)
    out = self.feed_forward(out)
    return (x.float() + ff_gate.float() * out.float()).type_as(x)


def _fused_block_forward(
    self,
    vis,
    aud,
    text_v,
    text_a,
    time_embed,
    vis_rope,
    aud_rope,
    va_gate_scale=1.0,
    av_gate_scale=1.0,
    video_seq_len=None,
    video_attn_meta=None,
    text_mask=None,
):
    """``Kandinsky6FusedTransformerDecoderBlock.forward``; with ``text_mask``
    the video and audio cross-attention read only the real prompt rows."""
    if text_mask is None:
        return _STOCK_FUSED_BLOCK_FORWARD(
            self,
            vis,
            aud,
            text_v,
            text_a,
            time_embed,
            vis_rope,
            aud_rope,
            va_gate_scale,
            av_gate_scale,
            video_seq_len,
            video_attn_meta,
        )
    t_v, t_a = time_embed

    sa_p, ca_p, ff_p = torch.chunk(
        self.videoT.visual_modulation(t_v).unsqueeze(dim=1), 3, dim=-1
    )
    shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
    vis = residual_gate_fp32(
        vis,
        self.videoT.self_attention(
            dit._norm_scale_shift(self.videoT.self_attention_norm, vis, shift, scale),
            rotary_emb=vis_rope,
            attn_mask_meta=video_attn_meta,
        ),
        gate,
    )
    shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
    vis_pre_ca = dit._norm_scale_shift(
        self.videoT.cross_attention_norm, vis, shift, scale
    )
    vis_out_t = text_attention(
        self.videoT.cross_attention, vis_pre_ca, text_v, None, text_mask
    )

    sa_p, ca_p, ff_p_a = torch.chunk(
        self.audioT.visual_modulation(t_a).unsqueeze(dim=1), 3, dim=-1
    )
    shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
    aud = residual_gate_fp32(
        aud,
        self.audioT.self_attention(
            dit._norm_scale_shift(self.audioT.self_attention_norm, aud, shift, scale),
            rotary_emb=aud_rope,
            skip_sequence_parallel=True,
        ),
        gate,
    )
    shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
    aud_pre_ca = dit._norm_scale_shift(
        self.audioT.cross_attention_norm, aud, shift, scale
    )
    aud_out_t = text_attention(
        self.audioT.cross_attention, aud_pre_ca, text_a, None, text_mask
    )
    aud = residual_gate_fp32(aud, aud_out_t, gate_a)

    t_va_mod = t_a if not self.fix_modulation else t_v
    t_av_mod = t_v if not self.fix_modulation else t_a
    va_params = self.va_modulation(t_va_mod).unsqueeze(dim=1)
    av_params = self.av_modulation(t_av_mod).unsqueeze(dim=1)
    if self.cross_gates:
        va_shift, va_scale, va_gate = torch.split(
            va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
        )
        av_shift, av_scale, av_gate = torch.split(
            av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
        )
    else:
        va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
        av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

    vis = residual_gate_fp32(vis, vis_out_t, gate_v)
    vis_for_va = dit._apply_scale_shift(self.va_normalization, vis, va_scale, va_shift)
    aud_for_av = dit._apply_scale_shift(self.av_normalization, aud, av_scale, av_shift)
    rq_v = vis_rope if self.ca_rope else None
    rk_a = aud_rope if self.ca_rope else None
    vis_from_aud = self.va_cross_attention(
        vis_for_va,
        encoder_hidden_states=aud_pre_ca,
        rotary_emb=rq_v,
        rotary_emb_kv=rk_a,
    )
    aud_from_vis = self.av_cross_attention(
        aud_for_av,
        encoder_hidden_states=vis_pre_ca,
        rotary_emb=rk_a,
        rotary_emb_kv=rq_v,
        context_seq_len=video_seq_len,
    )
    vis = residual_gate_fp32(
        vis,
        vis_from_aud,
        (va_gate if not self.cross_gates else av_gate) * va_gate_scale,
    )
    aud = residual_gate_fp32(
        aud,
        aud_from_vis,
        (av_gate if not self.cross_gates else va_gate) * av_gate_scale,
    )

    shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
    vis = residual_gate_fp32(
        vis,
        self.videoT.feed_forward(
            dit._norm_scale_shift(self.videoT.feed_forward_norm, vis, shift, scale)
        ),
        gate,
    )
    shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
    aud = residual_gate_fp32(
        aud,
        self.audioT.feed_forward(
            dit._norm_scale_shift(self.audioT.feed_forward_norm, aud, shift, scale)
        ),
        gate,
    )
    return vis, aud


def _fresh(x: torch.Tensor) -> torch.Tensor:
    """``x`` in storage of its own at offset 0 (compiled regions guard offsets)."""
    return x.clone(memory_format=torch.contiguous_format)


def _encode_text(self, prefix, text_embed, pooled, time, text_rope_pos, text_mask):
    """``Kandinsky6Transformer3DModel._encode_text`` over the padded prompt."""
    prefix = f"{prefix}_" if prefix else ""
    te = self.get_submodule(f"{prefix}text_embeddings")(text_embed)
    tm = self.get_submodule(f"{prefix}time_embeddings")(time)
    tm = tm + self.get_submodule(f"{prefix}pooled_text_embeddings")(pooled)
    rope = self.get_submodule(f"{prefix}text_rope_embeddings")(text_rope_pos)
    rope = rope_pair(rope.unsqueeze(dim=0))
    for block in self.get_submodule(f"{prefix}text_transformer_blocks"):
        te = block(te, tm, rope, text_mask=text_mask)
    return te, tm


def _forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    pooled_projections: torch.Tensor | None = None,
    hidden_states_audio: torch.Tensor | None = None,
    audio_timestep: torch.Tensor | None = None,
    visual_rope_pos=None,
    audio_rope_pos: torch.Tensor | None = None,
    text_rope_pos: torch.Tensor | None = None,
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sparse_params: dict[str, Any] | None = None,
    visual_token_type_ids: torch.Tensor | None = None,
    va_gate_scale: float = 1.0,
    av_gate_scale: float = 1.0,
    return_dict: bool = False,
    **kwargs: Any,
):
    """``Kandinsky6Transformer3DModel.forward``; on TPU the prompt pads to a
    fixed row count with a key mask, and SP shards reach the blocks at offset 0."""
    if (
        hidden_states.device.type not in _DEVICE_TYPES
        or not self.is_multimodal
        or sparse_params is not None
        or text_rope_pos is None
        or text_rope_pos.shape[0] != encoder_hidden_states.shape[1]
        or torch.is_grad_enabled()
    ):
        return _STOCK_FORWARD(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            pooled_projections=pooled_projections,
            hidden_states_audio=hidden_states_audio,
            audio_timestep=audio_timestep,
            visual_rope_pos=visual_rope_pos,
            audio_rope_pos=audio_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=scale_factor,
            sparse_params=sparse_params,
            visual_token_type_ids=visual_token_type_ids,
            va_gate_scale=va_gate_scale,
            av_gate_scale=av_gate_scale,
            return_dict=return_dict,
            **kwargs,
        )
    if pooled_projections is None:
        if encoder_hidden_states_image is None:
            raise ValueError("pooled_projections must be provided for Kandinsky6.")
        pooled_projections = encoder_hidden_states_image
    if visual_rope_pos is None:
        raise ValueError("visual_rope_pos is required for Kandinsky6.")
    if visual_token_type_ids is not None and self.visual_token_type_num_embeddings == 0:
        raise ValueError(
            "visual_token_type_ids requires visual_token_type_num_embeddings > 0."
        )
    if hidden_states_audio is None:
        raise NotImplementedError(
            "Multimodal Kandinsky6 checkpoints require both video and audio latents."
        )

    # The prompt's rows are all real (the stock DiT attends every row it gets).
    num_text = encoder_hidden_states.shape[1]
    rows = text_rows(num_text)
    device = encoder_hidden_states.device
    text = F.pad(encoder_hidden_states, (0, 0, 0, rows - num_text))
    text_rope_pos = torch.cat(
        [
            text_rope_pos,
            torch.arange(num_text, rows, device=device, dtype=text_rope_pos.dtype),
        ]
    )
    text_mask = (torch.arange(rows) < num_text).to(device, non_blocking=True)

    video_te, video_tm = _encode_text(
        self, "video", text, pooled_projections, timestep, text_rope_pos, text_mask
    )
    audio_te, audio_tm = _encode_text(
        self,
        "audio",
        text,
        pooled_projections,
        audio_timestep if audio_timestep is not None else timestep,
        text_rope_pos,
        text_mask,
    )
    if audio_rope_pos is None:
        audio_rope_pos = torch.arange(
            hidden_states_audio.shape[1], device=hidden_states_audio.device
        )

    visual_embed = self.visual_embeddings(hidden_states)
    if (
        self.visual_token_type_embeddings is not None
        and visual_token_type_ids is not None
    ):
        type_embed = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_embed = visual_embed + type_embed[:, :, None, None, :]
    visual_shape = visual_embed.shape[:-1]
    visual_rope = self.visual_rope_embeddings(
        visual_shape, visual_rope_pos, scale_factor
    ).flatten(1, 3)
    visual_embed, shard = shard_seq(visual_embed.flatten(1, 3))
    visual_rope = rope_pair(shard_like(visual_rope, shard, pad_mode="repeat_last"))
    visual_embed = _fresh(visual_embed)
    attn_meta = tail_attn_meta(shard, visual_embed.shape[0], visual_embed.device)

    audio_embed = self.audio_embeddings(hidden_states_audio)
    audio_rope = rope_pair(self.audio_rope_embeddings(audio_rope_pos).unsqueeze(dim=0))
    for block in self.visual_transformer_blocks:
        visual_embed, audio_embed = block(
            visual_embed,
            audio_embed,
            video_te,
            audio_te,
            (video_tm, audio_tm),
            visual_rope,
            audio_rope,
            va_gate_scale,
            av_gate_scale,
            video_seq_len=shard.orig_len,
            video_attn_meta=attn_meta,
            text_mask=text_mask,
        )

    visual_embed = gather_seq(visual_embed, shard.orig_len).reshape(*visual_shape, -1)
    result = (
        self.out_layer(visual_embed, video_tm),
        self.audio_out_layer(audio_embed, audio_tm),
    )
    if return_dict:
        return dit.Kandinsky6TransformerOutput(sample=result)
    return result


def install() -> None:
    if getattr(_MODEL.forward, "__wrapped_for_tpu__", False):
        return
    forward = functools.wraps(_STOCK_FORWARD)(_forward)
    forward.__wrapped_for_tpu__ = True
    _MODEL.forward = forward
    dit.Kandinsky6TransformerEncoderBlock.forward = _encoder_block_forward
    qwen_vl.QwenVLModelBase._get_text_rope_index = _text_rope_index
    qwen_config._extract_masked_hidden = _extract_masked_hidden
    stage.Kandinsky6DenoisingStage._text_rope_pos = staticmethod(_text_rope_pos)
    dit._apply_rotary = _apply_rotary
    config.Kandinsky6TI2VAPipelineConfig.tokenize_prompt = _tokenize_prompt
    dit.Kandinsky6FusedTransformerDecoderBlock.forward = _fused_block_forward
    # Every rank runs the same block programs (offset-0 inputs, no
    # rank-dependent values).
    _MODEL._compile_regions_spmd_safe = True
