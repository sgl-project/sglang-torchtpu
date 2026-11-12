# SPDX-License-Identifier: Apache-2.0
"""GLM-Image's DiT on TPU workers: one compiled program per glyph bucket.

Runtime patch, tracked debt:
- Why: the DiT's text rows are the ByT5 embeddings of the prompt's quoted
  text, packed by their mask, so their count changes per prompt (one row for
  the unconditional pass). torch_tpu compiles static shapes: every new count
  compiles the whole DiT again.
- How: :func:`install`
  (1) wraps the DiT's ``__call__``: the text rows right-pad with zeros to a
      power-of-two count (at least 16), and the real row count goes to every
      joint attention as a device int32 ``[0, n]`` in ``attention_kwargs``;
  (2) replaces ``GlmImageAttention.forward`` for calls that carry that count:
      queries attend the image keys and the first ``n`` text keys, the keys
      the stock attends; padded text keys are never read. One Pallas kernel
      step runs over ``[image, text]`` keys and reads only their first
      ``image + n``; shapes the kernel does not take run float32 attention
      with the padded keys masked. Ulysses splits the heads as the stock does.
  Every other op is per row, and the DiT returns the image rows only.
  Applies to text-to-image calls whose joint attention runs the Pallas
  backend; calls with reference-image K/V caches (image edits) and other
  attention backends run the stock forward.
  (3) On every TPU call, the wrapper copies RoPE tables that sit at a storage
      offset (an SP rank's slice of the full tables) to a storage of their
      own: torch_tpu specializes a compiled program on its inputs' storage
      offsets, so each rank would build a program of its own.
- Upstream PR or issue: none.
- Remove when: the GLM-Image pipeline pads its glyph rows to a bucket, the
  joint attention takes the real count as data, and SP RoPE shards start
  their storage.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_group,
    get_sp_world_size,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import glm_image as dit
from sglang_tpu.multimodal_gen import tpu_splash_attn

# Glyph texts share a program per power-of-two row count, at least this many.
# Every Ulysses rank runs all text rows, so the bucket stays within twice the
# real count.
MIN_TEXT_BUCKET = 16
# The ``attention_kwargs`` entry with the real text-row count.
TEXT_KEYS = "text_keys"
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_ATTENTION_FORWARD = dit.GlmImageAttention.forward


def text_bucket(rows: int) -> int:
    """The padded text-row count for ``rows`` real rows."""
    return max(MIN_TEXT_BUCKET, 1 << (max(rows, 1) - 1).bit_length())


@functools.lru_cache(maxsize=64)
def _counts(real: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


def _pads_text(transformer) -> bool:
    """Whether the joint attentions run the Pallas backend with QK norms, and
    locally or with heads split by Ulysses only."""
    attn = transformer.transformer_blocks[0].attn1
    if (
        not getattr(attn.attn.attn_impl, "use_splash", False)
        or attn.norm_q is None
        or attn.norm_k is None
    ):
        return False
    if get_sp_world_size() == 1:
        return True
    return (
        get_ring_parallel_world_size() == 1
        and attn.attn.sp_attention_mode == "ulysses"
        and attn.num_local_heads % get_ulysses_parallel_world_size() == 0
    )


def _at_storage_start(x):
    return x.clone(memory_format=torch.contiguous_format) if x.storage_offset() else x


def _transformer_call(self, *args, **kwargs):
    """Pads the text rows to the bucket and copies offset RoPE tables before
    the DiT runs."""
    latents = kwargs.get("hidden_states")
    text = kwargs.get("encoder_hidden_states")
    if isinstance(text, list):
        text = text[0]
    on_device = (
        isinstance(latents, torch.Tensor) and latents.device.type in _DEVICE_TYPES
    )
    if on_device and isinstance(kwargs.get("freqs_cis"), tuple):
        kwargs["freqs_cis"] = tuple(map(_at_storage_start, kwargs["freqs_cis"]))
    if (
        args
        or not on_device
        or not isinstance(text, torch.Tensor)
        # Reference-image K/V caches run the stock path.
        or kwargs.get("kv_caches") is not None
        or not _pads_text(self)
    ):
        return torch.nn.Module.__call__(self, *args, **kwargs)
    real = text.shape[1]
    bucket = text_bucket(real)
    attention_kwargs = dict(kwargs.get("attention_kwargs") or {})
    attention_kwargs[TEXT_KEYS] = _counts(real, latents.device)
    kwargs.update(
        encoder_hidden_states=F.pad(text, (0, 0, 0, bucket - real)),
        attention_kwargs=attention_kwargs,
    )
    return torch.nn.Module.__call__(self, **kwargs)


def joint_attention(attn, q, k, v, text, text_keys):
    """``attn`` over ``[B, text + S, H, D]`` rows, of the first ``text_keys[1]``
    text keys and every image key.

    The text rows are replicated over SP ranks, the image rows sharded.
    """
    sp = 1 if attn.skip_sequence_parallel else get_sp_world_size()
    parts = []
    for x in (q, k, v):
        txt, img = x[:, :text], x[:, text:]
        if sp > 1:
            # This rank's heads over the full sequence, as the stock splits.
            img = _usp_input_all_to_all(img, head_dim=2)
            heads = img.shape[2]
            txt = txt.narrow(2, get_ulysses_parallel_rank() * heads, heads)
        parts.append((txt.transpose(1, 2), img.transpose(1, 2)))
    (q_txt, q_img), (k_txt, k_img), (v_txt, v_img) = parts
    query = torch.cat([q_txt, q_img], dim=2)
    # Image keys first: the real keys are the first image + n.
    keys = torch.cat([k_img, k_txt], dim=2)
    values = torch.cat([v_img, v_txt], dim=2)
    kv_info = torch.stack((text_keys[0], text_keys[1] + k_img.shape[2]))
    out, _ = tpu_splash_attn.tpu_ring_attention_step(
        query, keys, values, kv_info, None, attn.attn_impl.softmax_scale
    )
    out = out.transpose(1, 2)
    out_txt, out_img = out[:, :text], out[:, text:]
    if sp > 1:
        out_img = _usp_output_all_to_all(out_img, head_dim=2)
        gathered = [
            torch.empty_like(out_txt) for _ in range(get_ulysses_parallel_world_size())
        ]
        torch.distributed.all_gather(
            gathered, out_txt.contiguous(), group=get_sp_group().ulysses_group
        )
        out_txt = torch.cat(gathered, dim=2)
    return torch.cat([out_txt, out_img], dim=1)


def attention_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    attention_mask=None,
    image_rotary_emb=None,
    kv_cache=None,
    text_keys=None,
):
    """``GlmImageAttention.forward``; with ``text_keys``, over the real text keys."""
    if text_keys is None:
        return _STOCK_ATTENTION_FORWARD(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            kv_cache,
        )
    dtype = encoder_hidden_states.dtype
    text = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    query, _ = self.to_q(hidden_states)
    key, _ = self.to_k(hidden_states)
    value, _ = self.to_v(hidden_states)
    query = query.unflatten(2, (self.num_local_heads, -1))
    key = key.unflatten(2, (self.num_local_kv_heads, -1))
    value = value.unflatten(2, (self.num_local_kv_heads, -1))
    query, key = dit._glm_qk_layernorm(self.norm_q, self.norm_k, query, key, dtype)
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        query[:, text:], key[:, text:] = self.rotary_emb(
            query=query[:, text:], key=key[:, text:], cos=cos, sin=sin
        )
    out = joint_attention(self.attn, query, key, value, text, text_keys)
    out, _ = self.to_out[0](out.flatten(2, 3).to(query.dtype))
    encoder_out, out = out.split([text, out.shape[1] - text], dim=1)
    return out, encoder_out


def install() -> None:
    dit.GlmImageTransformer2DModel.__call__ = _transformer_call
    dit.GlmImageAttention.forward = attention_forward
