# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image and Qwen-Image-Edit on TPU workers: one compiled DiT per image size.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The DiT attends [prompt | image]
  jointly at the prompt's exact row count, so every prompt length compiles the
  whole DiT again (the TPU default compiles the whole module), and the eager
  text encoder compiles once per prompt length too. Its RoPE is a complex
  multiply on ``torch.polar`` tables, whose interleaved pairs torch_tpu re-lays
  out (see rope.py). Under Ulysses the stock attention also reads the SP
  padding rows of the image as keys.
- How: :func:`install`
  (1) wraps the DiT's ``__call__``, outside the compiled module: the prompt
      rows pad with zeros to a multiple of 512, then of the SP degree, and the
      attention gets the real prompt and image key counts as device int32
      tensors;
  (2) builds the RoPE on the host as real float32 cos/sin tables, padded
      prompt rows taking the identity rotation;
  (3) replaces ``QwenImageCrossAttention.forward``: queries attend the real
      prompt keys, then each image segment's real keys, as resumable kernel
      steps (each key run padded at its tail only), rotating pairs in the
      pair-swap form. These are the same keys, and padded keys are never read;
  (4) right-pads the prompt tokens to a multiple of 256 (the T2I tokenizer and
      the edit processor). The encoder is causal, so real tokens never see the
      padding, and only real rows reach the DiT;
  (5) marks the blocks spmd-safe, one program per block.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prompt bucket and key counts,
  RoPE goes through a platform rotation, and regional compile marks spmd-safe
  regions.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as config
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.layernorm import apply_qk_norm
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import qwen_image as dit
from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.rope import _pair_swap_matrix

# Prompts stay within one bucket; the kernel's prompt step needs >= 512 keys.
TEXT_ALIGNMENT = 512
PROMPT_TOKEN_ALIGNMENT = 256

_STOCK_GET_FREQS_CIS = config.QwenImagePipelineConfig.get_freqs_cis
_STOCK_ATTENTION_FORWARD = dit.QwenImageCrossAttention.forward


def _real_table(freqs: torch.Tensor) -> torch.Tensor:
    """Complex ``[N, D/2]`` rotations as float32 ``[N, 2D]``: cos, then sin, pair-repeated."""
    return torch.cat(
        (freqs.real.repeat_interleave(2, -1), freqs.imag.repeat_interleave(2, -1)), -1
    ).float()


def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
    """``QwenImagePipelineConfig.get_freqs_cis`` built on the host, as real tables.

    Both slots hold the ``[N, 2 * head_dim]`` tables, 2D as the SP shard
    helpers expect; only :func:`attention_forward` reads them.
    """
    if isinstance(rotary_emb, dit.QwenEmbedLayer3DRope):
        return _STOCK_GET_FREQS_CIS(img_shapes, txt_seq_lens, rotary_emb, device, dtype)
    _, (img, txt) = _STOCK_GET_FREQS_CIS(
        img_shapes, txt_seq_lens, rotary_emb, torch.device("cpu"), dtype
    )
    tables = (_real_table(img).to(device), _real_table(txt).to(device))
    return tables, tables


def apply_rope_pair_swap(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """The stock complex rotation of ``x`` ``[B, N, H, D]`` on a ``[N, 2D]`` table.

    ``x * cos + (x @ R) * sin`` in float32 forms (ac - bd, ad + bc) per pair,
    the complex product; ``x @ R`` is a signed pair swap, exact in bfloat16.
    """
    cos, sin = table[None, :, None].chunk(2, -1)
    swapped = torch.matmul(x, _pair_swap_matrix(x.shape[-1], x.dtype, x.device))
    return (x.float() * cos + swapped.float() * sin).to(x.dtype)


def _counts(real: int, device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


def _image_keys(img_shapes, sp: int, device) -> list:
    """``(rows per rank, [0, real keys])`` per image segment, as the config shards them.

    Under SP the noisy image and the condition images shard separately, each
    padded at its tail; one rank takes them as one segment.
    """
    sizes = [math.prod(shape) for shape in img_shapes[0]]
    counts = [sizes[0], sum(sizes[1:])] if sp > 1 else [sum(sizes)]
    return [(-(-n // sp), _counts(n, device)) for n in counts if n]


def _dit_call(self, *args, **kwargs):
    """Pads the prompt rows to the bucket before the compiled DiT runs.

    Padded rows are zeros with the identity rotation; the attention reads only
    the real prompt keys (``tpu_text_keys``), and their outputs are dropped.
    """
    states, freqs = kwargs.get("encoder_hidden_states"), kwargs.get("freqs_cis")
    if states is None or freqs is None or self.use_layer3d_rope:
        return torch.nn.Module.__call__(self, *args, **kwargs)
    packed = states[0] if isinstance(states, list) else states
    batch, text_len = packed.shape[:2]
    if kwargs.get("encoder_hidden_states_mask") is not None or any(
        n != text_len for n in kwargs.get("txt_seq_lens") or ()
    ):
        raise NotImplementedError(
            "Qwen-Image on TPU batches only prompts of the same length"
        )
    sp = get_sp_world_size()
    bucket = -(-max(text_len, 1) // TEXT_ALIGNMENT) * TEXT_ALIGNMENT
    bucket = -(-bucket // sp) * sp
    extra, half = bucket - text_len, freqs[1].shape[1] // 2
    identity = F.pad(freqs[1].new_ones(extra, half), (0, half))
    kwargs.update(
        encoder_hidden_states=F.pad(packed, (0, 0, 0, extra)),
        encoder_hidden_states_mask=None,
        txt_seq_lens=[bucket] * batch,
        freqs_cis=(freqs[0], torch.cat((freqs[1], identity))),
        freqs_complex=None,
        attention_kwargs=dict(
            kwargs.get("attention_kwargs") or {},
            tpu_text_keys=_counts(text_len, packed.device),
            tpu_image_keys=_image_keys(kwargs["img_shapes"], sp, packed.device),
        ),
    )
    return torch.nn.Module.__call__(self, *args, **kwargs)


def _uses_kernel(impl, q, keys) -> bool:
    if not getattr(impl, "use_splash", False) or q.device.type != "tpu":
        return False
    # Every step takes the kernel or none does: their carries differ.
    return all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def _joint_attention(usp, q, k, v, segments, text_sharded):
    """``[B, S, H, D]`` attention over each segment's real keys.

    ``segments`` lists ``(rows per rank, [0, real keys])`` in each rank's row
    order [prompt | image segments], every segment padded at its tail.
    """
    sp = get_sp_world_size()
    sharded = not usp.skip_sequence_parallel and sp > 1
    if sharded:
        if (
            not text_sharded
            or get_ring_parallel_world_size() > 1
            or usp.sp_attention_mode == "kv_gather"
        ):
            raise NotImplementedError(
                "Qwen-Image on TPU shards the sequence with Ulysses only"
            )
        q, k, v = (_usp_input_all_to_all(t, head_dim=2) for t in (q, k, v))
    ranks, rows = (sp if sharded else 1), [n for n, _ in segments]

    # Ulysses gathers [rank 0 rows | rank 1 rows | ...]: a segment's rows from
    # every rank form one run of keys, real keys first.
    def runs(t):
        return [
            part.flatten(1, 2).transpose(1, 2)
            for part in t.unflatten(1, (ranks, -1)).split(rows, 2)
        ]

    q, keys, values = q.transpose(1, 2), runs(k), runs(v)
    impl = usp.attn_impl
    if _uses_kernel(impl, q, keys):
        carry = None
        for key, value, (_, real) in zip(keys, values, segments, strict=True):
            out, carry = impl.forward_ring_resume(q, key, value, real, carry)
    else:
        valid = torch.cat(
            [
                torch.arange(key.shape[2], device=q.device) < real[1]
                for key, (_, real) in zip(keys, segments, strict=True)
            ]
        )
        # Zero padded keys and values: a masked score still adds q . k, and a
        # zero probability times a NaN value is NaN.
        out = F.scaled_dot_product_attention(
            q,
            torch.where(valid[:, None], torch.cat(keys, 2), 0),
            torch.where(valid[:, None], torch.cat(values, 2), 0),
            attn_mask=valid[None, None, None],
            scale=impl.softmax_scale,
        )
    out = out.transpose(1, 2)
    return _usp_output_all_to_all(out, head_dim=2) if sharded else out


def attention_forward(
    self, hidden_states, encoder_hidden_states, image_rotary_emb, **kwargs
):
    """``QwenImageCrossAttention.forward`` over a bucketed prompt (see :func:`_dit_call`)."""
    text_keys = kwargs.get("tpu_text_keys")
    if text_keys is None:
        return _STOCK_ATTENTION_FORWARD(
            self, hidden_states, encoder_hidden_states, image_rotary_emb, **kwargs
        )
    img_q, img_k, img_v, *_ = dit._get_qkv_projections(self, hidden_states)
    txt_q, txt_k, txt_v = self._get_added_qkv_projections(encoder_hidden_states)
    img_q, img_k, img_v, txt_q, txt_k, txt_v = (
        t.unflatten(-1, (self.local_num_heads, self.head_dim))
        for t in (img_q, img_k, img_v, txt_q, txt_k, txt_v)
    )
    img_q, img_k = apply_qk_norm(img_q, img_k, self.norm_q, self.norm_k, self.head_dim)
    txt_q, txt_k = apply_qk_norm(
        txt_q, txt_k, self.norm_added_q, self.norm_added_k, self.head_dim
    )
    img_table, txt_table = image_rotary_emb
    rope = apply_rope_pair_swap
    q = torch.cat((rope(txt_q, txt_table), rope(img_q, img_table)), 1)
    k = torch.cat((rope(txt_k, txt_table), rope(img_k, img_table)), 1)
    text_rows, image_rows = txt_q.shape[1], img_q.shape[1]
    out = _joint_attention(
        self.attn,
        q,
        k,
        torch.cat((txt_v, img_v), 1),
        [(text_rows, text_keys), *kwargs["tpu_image_keys"]],
        kwargs.get("sp_text_sharded", False),
    )
    txt_out, img_out = out.flatten(2).split((text_rows, image_rows), 1)
    img_out, img_bias = self.to_out[0](img_out)
    if len(self.to_out) > 1:
        (img_out,) = self.to_out[1](img_out)
    txt_out, txt_bias = self.to_add_out(txt_out)
    return img_out, txt_out, img_bias, txt_bias


def _pad_tokens(inputs, tokenizer):
    """``inputs`` with its token rows right-padded to a multiple of 256."""
    extra = -inputs["input_ids"].shape[1] % PROMPT_TOKEN_ALIGNMENT
    if extra:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, extra), value=pad_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra))
    return inputs


class _PromptPadding:
    """An image processor whose token rows are right-padded to a multiple of 256.

    It keeps the processor's named parameters: the stage passes only the
    keyword arguments ``__call__``'s signature names.
    """

    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        inputs = self._processor(images=images, text=text, videos=videos, **kwargs)
        return _pad_tokens(inputs, self._processor.tokenizer)


def _padded_tokenize(tokenize):
    @functools.wraps(tokenize)
    def wrapped(self, prompts, tokenizer, tok_kwargs):
        return _pad_tokens(tokenize(self, prompts, tokenizer, tok_kwargs), tokenizer)

    wrapped.__wrapped_for_tpu__ = True
    return wrapped


def _padded_processor_init(init):
    @functools.wraps(init)
    def wrapped(self, image_processor, image_encoder=None, text_encoder=None):
        from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import (
            Qwen2_5_VLForConditionalGeneration,
        )

        init(self, image_processor, image_encoder, text_encoder)
        # Qwen-Image-Edit: the stage keeps only the processor's real rows.
        if image_encoder is None and isinstance(
            text_encoder, Qwen2_5_VLForConditionalGeneration
        ):
            self.image_processor = _PromptPadding(image_processor)

    wrapped.__wrapped_for_tpu__ = True
    return wrapped


# The annotation is not part of torch's compile cache key.
_SPMD_SAFE_CACHE_TAG = "qwen-image-spmd-safe-blocks-v1"


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


def install() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
        ImageEncodingStage,
    )

    pipeline_config = config.QwenImagePipelineConfig
    if not getattr(pipeline_config.tokenize_prompt, "__wrapped_for_tpu__", False):
        pipeline_config.tokenize_prompt = _padded_tokenize(
            pipeline_config.tokenize_prompt
        )
    init = ImageEncodingStage.__init__
    if not getattr(init, "__wrapped_for_tpu__", False):
        ImageEncodingStage.__init__ = _padded_processor_init(init)
    pipeline_config.get_freqs_cis = staticmethod(get_freqs_cis)
    dit.QwenImageCrossAttention.forward = attention_forward
    dit.QwenImageTransformer2DModel.__call__ = _dit_call
    _mark_blocks_spmd_safe(dit.QwenImageTransformerBlock)
