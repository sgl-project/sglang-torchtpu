# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 Image Klein on TPU workers: compiled DiT blocks on the Pallas kernels.

Runtime patch, tracked debt:
- Why: the FLUX.3 image DiT runs its blocks through CUDA fused ops. The gated
  MLP imports ``sgl_kernel`` (absent on TPU), the QK-norm RoPE fallback rotates
  with bf16 tables and stride-2 lane slices, and packed attention runs bf16
  SDPA one segment at a time. The RoPE tables are built in float64 on the
  device per request. The DiT declares no compile regions, so
  ``--enable-torch-compile`` traces the whole packed forward, and the Qwen3-VL
  text encoder runs eagerly. Under Ulysses each rank's shard is a view at a
  rank-dependent offset, which compiled programs guard. The LayerNorm-modulation
  fallback rounds to bf16 twice, and a batch (``n``, request batching) decodes
  as one float32 batch, whose activations scale with it.
- How: :func:`install` (1) runs each ``Flux3ImageBlock`` as one program
  (compiled under ``--enable-torch-compile``, SPMD-safe, shared by every block):
  float32 LayerNorm modulation rounded once, QK RMSNorm, the float32 pair-swap
  rotation of ``rope.py``, the Pallas kernels per packed segment (under Ulysses,
  heads exchanged around the kernel, padded keys skipped by the live-length
  kernel), SwiGLU and the gated residual, on fresh offset-0 inputs (Ulysses
  shards cloned once per stream); (2) compiles the Qwen3-VL text encoder's
  decoder layers the same way, one program per prompt bucket; (3) builds latent
  ids and RoPE tables on the host; (4) runs the output head's LayerNorm and
  modulation in float32, as CUDA autocast does; (5) decodes a batch one image
  at a time.
- Upstream PR or issue: none.
- Remove when: the FLUX.3 blocks take platform kernels for SwiGLU, QK-norm RoPE
  and packed attention, declare SPMD-safe compile regions with offset-0 shards,
  and build their tables on the host.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import flux3
from sglang.multimodal_gen.runtime.models.dits import flux3_image as dit
from sglang.multimodal_gen.runtime.models.encoders import flux3_text_encoder
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    flux3_image as stages,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.rope import apply_rotary_embedding_pair_swap

# Devices whose tensors take the TPU paths (tests add "cpu").
_DEVICE_TYPES = ("tpu",)

_STOCK_BLOCK_FORWARD = dit.Flux3ImageBlock.forward
_STOCK_LAST_LAYER_FORWARD = flux3.Flux3LastLayer.forward
_STOCK_ROPE = dit.Flux3ImageDiT.rope
_STOCK_SHARD_LIKE = dit.shard_like
_STOCK_LATENT_IDS = stages.latent_ids
_STOCK_VAE_DECODE = stages.vae_decode
_STOCK_MAYBE_TORCH_COMPILE = stages.Flux3ImageDenoiseStage._maybe_torch_compile
_STOCK_ENCODE_BATCH = flux3_text_encoder.Flux3TextEncoder.encode_batch


def _on_tpu(tensor: torch.Tensor) -> bool:
    return tensor.device.type in _DEVICE_TYPES


def _offset0(x: torch.Tensor) -> torch.Tensor:
    # torch_tpu guards compiled inputs' storage offsets and strides.
    if x.storage_offset() == 0 and x.is_contiguous():
        return x
    return x.clone(memory_format=torch.contiguous_format)


@functools.lru_cache(maxsize=64)
def _int32(value: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int32).to(device)


def _attention(q, k, v, scale):
    """``[B, H, S, D]`` attention: the TPU kernels for TPU tensors, else the
    float32 SDPA of their fallback."""
    if q.device.type == "tpu":
        return tpu_splash_attn.tpu_attention(q, k, v, scale=scale)
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    return out.to(q.dtype)


def _segment_attention(q, k, v, lens, scale):
    """``(L, H, D)`` attention within each of the packed ``lens`` segments."""
    if not any(lens):
        # A content mode without tokens runs one placeholder row (as stock).
        return torch.zeros_like(v)
    outs, start = [], 0
    for length in lens:
        if length:
            q_s, k_s, v_s = (
                t[start : start + length].transpose(0, 1)[None] for t in (q, k, v)
            )
            out = _attention(q_s, k_s, v_s, scale)
            outs.append(out[0].transpose(0, 1))
        start += length
    return outs[0] if len(outs) == 1 else torch.cat(outs)


def _live_attention(q, k, v, live, scale):
    """``[1, S, H, D]`` attention over keys ``[0, live)`` (int32 ``[1]`` on
    device): one program for every count. Padded rows' output is dropped."""
    seq, heads, head_dim = q.shape[1:]
    bhsd = [t.transpose(1, 2) for t in (q, k, v)]
    if (
        q.device.type == "tpu"
        and seq >= 512
        and tpu_splash_attn._kernel_input_fallback_reason(*bhsd) is None
    ):
        tpu_splash_attn._register_ops()
        if tpu_splash_attn._FLASH_LIVE_LENGTH_OP is not None:
            rows = (t[0].reshape(seq, heads * head_dim) for t in (q, k, v))
            out = tpu_splash_attn._FLASH_LIVE_LENGTH_OP(*rows, live, scale, heads)
            return out.view(1, seq, heads, head_dim)
    valid = torch.arange(seq, device=q.device) < live
    # A masked score still adds q . k, and zero probability times NaN is NaN:
    # zero the padded keys and values. float32, as tpu_attention's SDPA.
    k, v = (torch.where(valid[:, None], t, 0) for t in (bhsd[1], bhsd[2]))
    out = F.scaled_dot_product_attention(
        bhsd[0].float(), k.float(), v.float(), attn_mask=valid[None], scale=scale
    )
    return out.to(q.dtype).transpose(1, 2)


def _sp_attention(q, k, v, live, scale):
    """Ulysses: ``(L_local, H, D)`` shards exchange heads around the kernel."""
    q, k, v = (_usp_input_all_to_all(t[None], head_dim=2) for t in (q, k, v))
    if live is None:
        bhsd = (t.transpose(1, 2) for t in (q, k, v))
        out = _attention(*bhsd, scale).transpose(1, 2)
    else:
        out = _live_attention(q, k, v, live, scale)
    return _usp_output_all_to_all(out, head_dim=2)[0]


def _block(self, x, cos_sin, shift, scale, gate, lens, live):
    """``Flux3ImageBlock.forward`` on ``(L, hidden)`` rows: ``lens`` packed
    segments, or (``lens=None``) this rank's Ulysses shard, whose gathered
    rows attend their first ``live`` keys (all of them for ``live=None``)."""
    dim = x.shape[-1]
    # As the CUDA kernel: LayerNorm and modulation in float32, rounded once.
    x_mod = (
        F.layer_norm(x.float(), (dim,), eps=self.norm_scale_shift.eps)
        * (1 + scale.float())
        + shift.float()
    ).to(x.dtype)
    qkv = self.to_qkv(x_mod)[0].view(x.shape[0], 3, self.local_heads, -1)
    cos, sin = cos_sin.chunk(2, dim=-1)
    q = apply_rotary_embedding_pair_swap(self.norm.query_norm(qkv[:, 0]), cos, sin)
    k = apply_rotary_embedding_pair_swap(self.norm.key_norm(qkv[:, 1]), cos, sin)
    v = qkv[:, 2]
    softmax_scale = self.attention._usp.softmax_scale
    if lens is None:
        attn = _sp_attention(q, k, v, live, softmax_scale)
    else:
        attn = _segment_attention(q, k, v, lens, softmax_scale)
    attn_out = self.attn_out(attn.flatten(1))[0]
    gate_value = self.mlp_in(x_mod)[0]
    half = gate_value.shape[-1] // 2
    mlp_out = self.mlp_out(F.silu(gate_value[:, :half]) * gate_value[:, half:])[0]
    # residual_gate_add2: the branch sum rounds, the gated product stays float32.
    update = attn_out + mlp_out
    return torch.addcmul(x.float(), gate.float(), update.float()).to(x.dtype)


_REGION: dict[str, object] = {}


def _compile_enabled() -> bool:
    try:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        return bool(get_global_server_args().enable_torch_compile)
    except Exception:  # noqa: BLE001 - no server (unit tests)
        return False


def _compile(fn, spmd=False):
    from sglang.multimodal_gen.runtime.platforms import current_platform

    wrap = current_platform.prepare_spmd_region_wrapper() if spmd else None
    return torch.compile(
        fn if wrap is None else wrap(fn),
        backend=current_platform.get_compile_backend(),
        dynamic=False,
    )


def _region():
    """:func:`_block`, compiled once under ``--enable-torch-compile``: every
    rank runs the same program (SPMD-safe), and blocks share its graphs."""
    if "block" not in _REGION:
        _REGION["block"] = _compile(_block, spmd=True) if _compile_enabled() else _block
    return _REGION["block"]


def block_forward(self, x, rope, mod, seq_lens=None, sp=None):
    """``Flux3ImageBlock.forward`` through :func:`_block` for TPU tensors."""
    if not _on_tpu(x) or self.tp_size != 1:
        return _STOCK_BLOCK_FORWARD(self, x, rope, mod, seq_lens=seq_lens, sp=sp)
    lens = tuple(seq_lens) if sp is None else None
    live = None
    if sp is not None and sp.shard.num_pad:
        live = _int32(sp.shard.orig_len, x.device)
    tpu_splash_attn._register_ops()
    return _region()(self, *map(_offset0, (x, rope[0], *mod)), lens, live)


def shard_like(x, shard, dim=1, pad_mode="zeros"):
    """``shard_like`` as a fresh offset-0 tensor for TPU tensors: once per
    Ulysses stream, not once per block call."""
    out = _STOCK_SHARD_LIKE(x, shard, dim=dim, pad_mode=pad_mode)
    return _offset0(out) if _on_tpu(out) else out


def last_layer_forward(self, x, vec):
    """``Flux3LastLayer.forward`` with CUDA autocast's float32 LayerNorm: the
    normalized, modulated rows stay float32 until the output projection."""
    projection = self.adaLN_modulation[1]
    if not _on_tpu(x) or not isinstance(projection, torch.nn.Linear):
        return _STOCK_LAST_LAYER_FORWARD(self, x, vec)
    if vec.ndim == 2:
        vec = vec[:, None, :]
    activated = self.adaLN_modulation[0](vec)
    shift_w, scale_w = projection.weight.chunk(2)
    scale = F.linear(activated, scale_w).add_(1)
    shift = F.linear(activated, shift_w)
    norm = self.norm_final
    x = F.layer_norm(x.float(), norm.normalized_shape, eps=norm.eps)
    return self.linear((x * scale.float() + shift.float()).to(projection.weight.dtype))


def rope(self, ids):
    """``Flux3Transformer.rope`` built on the host in float64, as the
    reference, then moved to the DiT's device."""
    device = self.txt_in.weight.device
    if device.type not in _DEVICE_TYPES:
        return _STOCK_ROPE(self, ids)
    return flux3.rope_cos_sin(ids.cpu(), self.axes_dim, self.theta).to(device)


def latent_ids(height, width, device="cpu"):
    """``latent_ids`` on the host, where the RoPE tables build from them."""
    return _STOCK_LATENT_IDS(height, width, "cpu")


def vae_decode(vae, tokens, ids, *, batched=False):
    """``vae_decode`` one image at a time for TPU tensors: decode memory stays
    at one image whatever the batch."""
    rows = ids.shape[0]
    if not _on_tpu(tokens) or tokens.shape[0] <= rows:
        return _STOCK_VAE_DECODE(vae, tokens, ids, batched=batched)
    decoded = torch.cat(
        [_STOCK_VAE_DECODE(vae, t, ids, batched=True) for t in tokens.split(rows)]
    )
    return decoded if batched else decoded[0]


def maybe_torch_compile(self, module):
    """``_maybe_torch_compile`` without the whole-DiT compile for a TPU DiT:
    its blocks compile themselves (see :func:`block_forward`)."""
    param = (
        next(module.parameters(), None) if isinstance(module, torch.nn.Module) else None
    )
    if param is None or param.device.type not in _DEVICE_TYPES:
        _STOCK_MAYBE_TORCH_COMPILE(self, module)


def _text_layer(layer, hidden, cos, sin, mask):
    """``Qwen3VLTextDecoderLayer.forward`` without the output-recording hook of
    its attention module, which breaks a compiled graph."""
    attn, _ = layer.self_attn.forward(
        hidden_states=layer.input_layernorm(hidden),
        position_embeddings=(cos, sin),
        attention_mask=mask,
    )
    hidden = hidden + attn
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def _text_layer_forward(
    self, hidden_states, position_embeddings, attention_mask=None, **kwargs
):
    if "text" not in _REGION:
        _REGION["text"] = _compile(_text_layer)
    return _REGION["text"](self, hidden_states, *position_embeddings, attention_mask)


def encode_batch(self, texts):
    """``Flux3TextEncoder.encode_batch`` with the Qwen3-VL decoder layers as one
    compiled program per prompt bucket, shared by every layer, for a TPU
    encoder under ``--enable-torch-compile``."""
    if self.device.type in _DEVICE_TYPES and _compile_enabled():
        for layer in self.model.language_model.layers:
            layer.forward = _text_layer_forward.__get__(layer)
    return _STOCK_ENCODE_BATCH(self, texts)


def patches():
    """``(owner, name, replacement)`` of every name :func:`install` binds."""
    yield dit.Flux3ImageBlock, "forward", block_forward
    yield flux3.Flux3LastLayer, "forward", last_layer_forward
    yield dit.Flux3ImageDiT, "rope", rope
    yield dit, "shard_like", shard_like
    yield stages, "latent_ids", latent_ids
    yield stages, "vae_decode", vae_decode
    yield stages.Flux3ImageDenoiseStage, "_maybe_torch_compile", maybe_torch_compile
    yield flux3_text_encoder.Flux3TextEncoder, "encode_batch", encode_batch


def install() -> None:
    for owner, name, value in patches():
        setattr(owner, name, value)
