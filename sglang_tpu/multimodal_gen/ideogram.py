# SPDX-License-Identifier: Apache-2.0
"""Ideogram 4 on TPU workers: one compiled DiT per image size and prompt bucket.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The single-stream DiT runs the packed
  sequence [prompt | image] at the prompt's exact length, so every prompt
  length compiles the whole DiT again (the TPU default compiles the whole
  module). Its masks and RoPE ids are int64 device tensors, and the text
  encoder picks the prompt's rows with device boolean masks. Under Ulysses the
  stage keeps the latents whole and the DiT does not shard, so every rank runs
  the full sequence. The FP8 weights dequantize on every forward: the
  first-use dequantization cache is CUDA-only.
- How: :func:`install`
  (1) wraps the DiT's ``__call__``, outside the compiled module: the packed
      rows pad at their tail to a multiple of 512 (then of the SP degree),
      and each Ulysses rank takes its share of the rows. The stock RoPE
      cos/sin and the prompt and image row masks are built once per request
      on the host; the real key count goes in as a device int32 tensor. The
      output rows are gathered after the call;
  (2) replaces the DiT's ``forward`` on those inputs: the same layers;
  (3) replaces ``Ideogram4Attention.forward``: queries attend the real keys
      through the resumable kernel step, with zero heads appended where
      Ulysses needs them (18 heads at 8). These are the keys the stock DiT
      attends, and padded keys are never read;
  (4) dequantizes each FP8 linear's weight once, at the DiT's first call, as
      the CUDA cache does (``_maybe_promote_fp8_weight``): the same values
      feed the same matmul;
  (5) runs the text encoder over the prompt's rows, found on the host and
      right-padded to a multiple of 256. The encoder is causal, so real
      tokens never see the padding;
  (6) marks the blocks spmd-safe, one program per block;
  (7) computes the logit-normal timestep schedule on the host: it runs in
      float64 with ``torch.special.ndtri``, which torch_tpu does not
      implement, and every request failed there.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prompt bucket and key count
  and shards its own sequence under SP, and the FP8 dequantization cache
  covers every device.
"""

from __future__ import annotations

import weakref
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import gather_seq
from sglang.multimodal_gen.runtime.layers.quantization import weight_only_fp8 as fp8
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ideogram as dit
from sglang.multimodal_gen.runtime.models.encoders import ideogram as encoder
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    ideogram as stage,
)
from sglang_tpu.compile_cache import append_compile_cache_key_tag

# Packed rows pad to this multiple: prompts of up to 512 tokens share a graph
# per image size, and the kernel step needs >= 512 keys.
SEQUENCE_ALIGNMENT = 512
PROMPT_TOKEN_ALIGNMENT = 256
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_FORWARD = dit.Ideogram4Transformer2DModel.forward
_STOCK_ATTENTION_FORWARD = dit.Ideogram4Attention.forward


class Layout(NamedTuple):
    """One request's DiT inputs for this rank, built on the host."""

    cos: torch.Tensor  # [B, rows, 1, head_dim]
    sin: torch.Tensor
    prompt_rows: torch.Tensor  # [B, rows, 1], the DiT dtype
    image_rows: torch.Tensor  # [B, rows, 1], bool
    keys: torch.Tensor  # int32 [0, real keys]


# Per-request host work, keyed by the source tensors (alive for the request).
_CACHE: dict[tuple, tuple[tuple[weakref.ref, ...], object]] = {}
_CACHE_SIZE = 8


def _cached(sources, key, build):
    key = tuple(id(t) for t in sources) + key
    hit = _CACHE.get(key)
    if hit is not None and all(r() is t for r, t in zip(hit[0], sources)):
        return hit[1]
    value = build()
    for stale in [
        k for k, (refs, _) in _CACHE.items() if any(r() is None for r in refs)
    ]:
        del _CACHE[stale]
    while len(_CACHE) >= _CACHE_SIZE:
        del _CACHE[next(iter(_CACHE))]
    _CACHE[key] = (tuple(weakref.ref(t) for t in sources), value)
    return value


def _padded_rows(rows: int, sp: int) -> int:
    padded = -(-rows // SEQUENCE_ALIGNMENT) * SEQUENCE_ALIGNMENT
    return -(-padded // sp) * sp


def _local(t: torch.Tensor, padded: int, sp: int, rank: int, value=0) -> torch.Tensor:
    """This rank's rows of ``t`` padded at its tail to ``padded`` rows (dim 1).

    A new contiguous tensor of this rank's rows only. Compiled programs guard
    input strides, and a copy of a full share made by ``F.pad`` keeps the
    source's batch stride, which spans the prompt's rows.
    """
    rows = padded // sp
    t = t[:, rank * rows : (rank + 1) * rows]
    local = t.new_full((t.shape[0], rows, *t.shape[2:]), value)
    local[:, : t.shape[1]] = t
    return local


def _layout(model, position_ids, indicator, segment_ids, dtype, device) -> Layout:
    sp = get_sp_world_size()
    rank = get_sp_parallel_rank() if sp > 1 else 0

    def build():
        segments = segment_ids.cpu()
        real = segments.shape[1]
        if not bool((segments > 0).all()):
            raise NotImplementedError(
                "Ideogram 4 on TPU batches only prompts of the same length"
            )
        padded = _padded_rows(real, sp)
        positions = position_ids.cpu()
        cos, sin = model.rotary_emb(torch.empty(0, dtype=dtype), positions)
        rows = indicator.cpu()
        tables = (
            cos.unsqueeze(2),
            sin.unsqueeze(2),
            (rows == dit.LLM_TOKEN_INDICATOR).to(dtype).unsqueeze(-1),
            (rows == dit.OUTPUT_IMAGE_INDICATOR).unsqueeze(-1),
        )
        keys = torch.tensor([0, real], dtype=torch.int32)
        return Layout(
            *(_local(t, padded, sp, rank).to(device) for t in tables),
            keys.to(device),
        )

    return _cached(
        (position_ids, indicator, segment_ids), (sp, rank, dtype, device), build
    )


def _dequantize_fp8_weights(model, dtype) -> None:
    """Each FP8 linear's weight, dequantized once (``_maybe_promote_fp8_weight``)."""
    if getattr(model, "_tpu_fp8_weights_decided", False):
        return
    model._tpu_fp8_weights_decided = True
    if not envs.SGLANG_DIFFUSION_FP8_WEIGHT_DEQUANT_CACHE:
        return
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if (
            not hasattr(module, "weight_scale")
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != fp8.FP8_WEIGHT_DTYPE
            or weight.device.type not in _DEVICE_TYPES
            or getattr(module, "enable_fused_w8a8", False)
        ):
            continue
        with torch.inference_mode(False), torch.no_grad():
            dequant = fp8.dequantize_rowwise_fp8_weight(
                weight, module.weight_scale, module.compute_dtype or dtype
            )
        module.weight = nn.Parameter(dequant, requires_grad=False)
        module._fp8_dequant_decided = True


def _dit_call(self, *args, **kwargs):
    """Pads and shards the packed rows before the compiled DiT runs.

    Padded rows are zeros, neither prompt nor image; the attention reads only
    the real keys (``tpu_layout.keys``), and padded rows' outputs are dropped.
    """
    x = kwargs.get("x")
    if (
        args
        or not isinstance(x, torch.Tensor)
        or x.device.type not in _DEVICE_TYPES
        or kwargs.get("position_ids") is None
    ):
        return torch.nn.Module.__call__(self, *args, **kwargs)
    dtype = self.embed_image_indicator.weight.dtype
    _dequantize_fp8_weights(self, dtype)
    layout = _layout(
        self,
        kwargs["position_ids"],
        kwargs["indicator"],
        kwargs["segment_ids"],
        dtype,
        x.device,
    )
    real = x.shape[1]
    sp = get_sp_world_size()
    rank = get_sp_parallel_rank() if sp > 1 else 0
    padded = layout.cos.shape[1] * sp
    llm_features = kwargs["llm_features"]
    # The prompt features are the same tensor at every step of a request.
    llm_features = _cached(
        (llm_features,),
        (sp, rank, padded, dtype),
        lambda: _local(llm_features.to(dtype), padded, sp, rank),
    )
    out = torch.nn.Module.__call__(
        self,
        llm_features=llm_features,
        x=_local(x.to(dtype), padded, sp, rank),
        # A fresh tensor: the stage's timestep is a view at the step's offset
        # into the schedule, and compiled programs specialize on the offset.
        t=kwargs["t"].clone(),
        position_ids=None,
        segment_ids=None,
        indicator=None,
        tpu_layout=layout,
    )
    out = gather_seq(out, real)
    return out[:, :real]


def forward(
    self,
    *,
    llm_features,
    x,
    t,
    position_ids,
    segment_ids,
    indicator,
    attn_mask=None,
    attn_mask_meta=None,
    tpu_layout: Layout | None = None,
    **kwargs,
):
    """``Ideogram4Transformer2DModel.forward`` over this rank's padded rows (see :func:`_dit_call`)."""
    if tpu_layout is None:
        return _STOCK_FORWARD(
            self,
            llm_features=llm_features,
            x=x,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
            **kwargs,
        )
    t = t.to(self.embed_image_indicator.weight.dtype)
    prompt_rows = tpu_layout.prompt_rows
    image_rows = tpu_layout.image_rows.to(x.dtype)
    llm_features = llm_features * prompt_rows
    x = x * image_rows
    x = self.input_proj(x) * image_rows
    t_cond = self.t_embedding(t)
    if t.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(self.adaln_proj(t_cond))
    llm_features = self.llm_cond_proj(self.llm_cond_norm(llm_features))
    llm_features = llm_features * prompt_rows
    h = x + llm_features
    # The indicator embedding: row 1 for image rows, row 0 otherwise.
    table = self.embed_image_indicator.weight
    h = h + torch.where(tpu_layout.image_rows, table[1], table[0])
    meta = {"tpu_keys": tpu_layout.keys}
    for layer in self.layers:
        h = layer(
            h,
            cos=tpu_layout.cos,
            sin=tpu_layout.sin,
            adaln_input=adaln_input,
            attn_mask=None,
            attn_mask_meta=meta,
        )
    return self.final_layer(h, c=adaln_input).to(torch.float32)


def _attention(usp, q, k, v, keys):
    """``[B, S, H, D]`` attention over the first ``keys[1]`` gathered keys."""
    sp = get_sp_world_size()
    heads = q.shape[2]
    if sp > 1:
        # An auto-picked K/V-gather mode falls back to Ulysses, as in USPAttention.
        if get_ring_parallel_world_size() > 1 or (
            usp.sp_attention_mode == "kv_gather" and not usp.sp_attention_mode_is_auto
        ):
            raise NotImplementedError("Ideogram 4 on TPU shards with Ulysses only")
        # Zero heads up to the SP multiple; their outputs are dropped.
        pad = (0, 0, 0, -heads % sp)
        q, k, v = (_usp_input_all_to_all(F.pad(t, pad), head_dim=2) for t in (q, k, v))
    # Ulysses gathers the ranks' rows in order: the padding stays at the tail.
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    impl = usp.attn_impl
    if getattr(impl, "use_splash", False) and q.device.type in _DEVICE_TYPES:
        # The kernel step; shapes it does not take run its float32 reference.
        out, _ = impl.forward_ring_resume(q, k, v, keys, None)
    else:
        valid = torch.arange(k.shape[2], device=q.device) < keys[1]
        # Zero padded keys and values: a zero probability times NaN is NaN.
        out = F.scaled_dot_product_attention(
            q,
            torch.where(valid[:, None], k, 0),
            torch.where(valid[:, None], v, 0),
            attn_mask=valid[None, None, None],
            scale=impl.softmax_scale,
        )
    out = out.transpose(1, 2)
    if sp > 1:
        out = _usp_output_all_to_all(out, head_dim=2)[:, :, :heads]
    return out


def attention_forward(self, x, cos, sin, attn_mask, attn_mask_meta):
    """``Ideogram4Attention.forward`` over padded rows (see :func:`forward`)."""
    keys = None
    if isinstance(attn_mask_meta, dict):
        keys = attn_mask_meta.get("tpu_keys")
    if keys is None:
        return _STOCK_ATTENTION_FORWARD(self, x, cos, sin, attn_mask, attn_mask_meta)
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x).view(batch_size, seq_len, 3, self.local_num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q, k = dit._ideogram_rope(self.norm_q(q), self.norm_k(k), cos, sin)
    out = _attention(self.attn, q, k, v, keys)
    return self.o(
        out.reshape(batch_size, seq_len, self.local_num_heads * self.head_dim)
    )


_STOCK_ENCODE = encoder.IdeogramQwen3VLTextEncoder.encode_ideogram_features


def encode_ideogram_features(
    self, token_ids, text_position_ids, indicator, llm_token_indicator
):
    """``encode_ideogram_features`` with the prompt's rows found on the host."""
    if token_ids.device.type not in _DEVICE_TYPES:
        return _STOCK_ENCODE(
            self, token_ids, text_position_ids, indicator, llm_token_indicator
        )
    batch_size, seq_len = token_ids.shape
    hidden_size = self.language_model.config.hidden_size
    features = torch.zeros(
        batch_size,
        seq_len,
        hidden_size * len(self._activation_layers),
        dtype=torch.float32,
        device=token_ids.device,
    )
    rows, ids, positions = indicator.cpu(), token_ids.cpu(), text_position_ids.cpu()
    for b in range(batch_size):
        (prompt,) = torch.nonzero(rows[b] == llm_token_indicator, as_tuple=True)
        count = prompt.numel()
        if count == 0:
            continue
        start = int(prompt[0])
        if int(prompt[-1]) - start + 1 != count:
            raise NotImplementedError("Ideogram 4 prompt rows must be contiguous")
        pad = -count % PROMPT_TOKEN_ALIGNMENT
        tokens = F.pad(ids[b, start : start + count], (0, pad))
        pos = F.pad(positions[b, start : start + count, 0], (0, pad))
        tokens = tokens.to(torch.int32)[None].to(token_ids.device)
        pos = pos.to(torch.int32)[None, None].expand(4, 1, -1).to(token_ids.device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self(
                input_ids=tokens,
                position_ids=pos,
                attention_mask=torch.ones_like(tokens),
                output_hidden_states=True,
            )
        selected = [
            outputs.hidden_states[i][:, :count] for i in self._activation_layers
        ]
        stacked = torch.stack(selected, dim=0).permute(1, 2, 3, 0)
        features[b, start : start + count] = stacked.reshape(count, -1).float()
    return features


_STOCK_SCHEDULE = stage.LogitNormalSchedule.__call__


def schedule_call(self, t: torch.Tensor) -> torch.Tensor:
    """``LogitNormalSchedule.__call__``, in float64 on the host."""
    if t.device.type not in _DEVICE_TYPES:
        return _STOCK_SCHEDULE(self, t)
    return _STOCK_SCHEDULE(self, t.cpu()).to(t.device)


# The annotation is not part of torch's compile cache key.
_SPMD_SAFE_CACHE_TAG = "ideogram4-spmd-safe-blocks-v1"


def _mark_blocks_spmd_safe(block_cls) -> None:
    # The whole DiT compiles as one module, outside the regional-compile path
    # that applies the platform's spmd-safe wrapper.
    try:
        from torch_tpu._internal.distributed.spmd_util import spmd_safe
    except ImportError:
        # A private torch_tpu API: without it, blocks keep split collectives.
        return
    if getattr(block_cls.forward, "__spmd_safe_for_tpu__", False):
        return
    wrapped = spmd_safe(block_cls.forward)
    wrapped.__spmd_safe_for_tpu__ = True
    block_cls.forward = wrapped
    append_compile_cache_key_tag(_SPMD_SAFE_CACHE_TAG)


def install() -> None:
    dit.Ideogram4Attention.forward = attention_forward
    dit.Ideogram4Transformer2DModel.forward = forward
    dit.Ideogram4Transformer2DModel.__call__ = _dit_call
    encoder.IdeogramQwen3VLTextEncoder.encode_ideogram_features = (
        encode_ideogram_features
    )
    stage.LogitNormalSchedule.__call__ = schedule_call
    _mark_blocks_spmd_safe(dit.Ideogram4TransformerBlock)
