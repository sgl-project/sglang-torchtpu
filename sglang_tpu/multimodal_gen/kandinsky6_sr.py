# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 video super-resolution on TPU workers.

Runtime patch, tracked debt:
- Why: the SR DiT attends every tile's tokens (31 x 32 x 48 = 47,616 for a
  768x512, 121-frame clip at x2) with 64-wide heads. The Pallas kernels take
  128-wide heads, so the stock TPU backend runs float32 SDPA over the whole
  tile. The KVAE encodes and decodes eagerly, segment by segment (8 s per
  decoded tile on one core). Under Ulysses each rank's video shard and RoPE
  are views at a rank-dependent storage offset, which torch_tpu guards: every
  rank compiles its own block program.
- How: :func:`install`
  (1) zero-pads the SR DiT's heads to 128 lanes for the attention kernels
      (Ulysses calls and ring steps) and cuts the output back: zero lanes add
      exact zeros to every query-key product, and the softmax scale stays the
      64-wide one;
  (2) runs each causal KVAE segment, encode and decode, as a compiled program
      under --enable-torch-compile (first, middle and last segment shapes),
      each segment in storage of its own;
  (3) hands the blocks the video shard and its RoPE as fresh tensors at
      offset 0, and marks the blocks spmd-safe: every rank runs one program.
- Upstream PR or issue: none.
- Remove when: the TPU attention kernels take 64-wide heads, the KVAE runs
  compiled segments, and SP shards reach the blocks at offset 0.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.models.dits import kandinsky6_sr as sr
from sglang.multimodal_gen.runtime.models.vaes import kandinsky6_sr_vae as kvae
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn

# Head width of the Pallas kernels.
KERNEL_HEAD_DIM = 128

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_MODEL = sr.Kandinsky6SRTransformer3DModel
_STOCK_INIT = _MODEL.__init__
_STOCK_SHARD_SEQ = sr.shard_seq
_STOCK_SHARD_LIKE = sr.shard_like
_VAE = kvae.Kandinsky6SRVAE
_STOCK_ENCODE = _VAE.encode
_STOCK_DECODE = _VAE._decode


def _fresh(x: torch.Tensor) -> torch.Tensor:
    """``x`` in storage of its own at offset 0 on TPU (compiled regions guard offsets)."""
    if x.device.type not in _DEVICE_TYPES:
        return x
    return x.clone(memory_format=torch.contiguous_format)


def _shard_seq(x, *args, **kwargs):
    out, shard = _STOCK_SHARD_SEQ(x, *args, **kwargs)
    return _fresh(out), shard


def _shard_like(x, *args, **kwargs):
    return _fresh(_STOCK_SHARD_LIKE(x, *args, **kwargs))


def padded_head_forward(impl):
    """``impl.forward`` with heads narrower than the kernels zero-padded to them."""
    stock = impl.forward

    @functools.wraps(stock)
    def forward(query, key, value, attn_metadata, return_softmax_lse=False):
        head_dim = query.shape[-1]
        if (
            return_softmax_lse
            or head_dim >= KERNEL_HEAD_DIM
            or query.device.type not in _DEVICE_TYPES
        ):
            return stock(query, key, value, attn_metadata, return_softmax_lse)
        pad = (0, KERNEL_HEAD_DIM - head_dim)
        out = tpu_splash_attn.tpu_attention(
            F.pad(query, pad).transpose(1, 2),
            F.pad(key, pad).transpose(1, 2),
            F.pad(value, pad).transpose(1, 2),
            scale=impl.softmax_scale,
            is_causal=impl.causal,
        )
        return out.transpose(1, 2)[..., :head_dim]

    forward.__padded_for_tpu__ = True
    return forward


def padded_head_ring_resume(impl):
    """``impl.forward_ring_resume`` with heads narrower than the kernels zero-padded
    to them; the float32 carry stays padded between steps."""
    stock = impl.forward_ring_resume

    @functools.wraps(stock)
    def forward_ring_resume(query, key, value, kv_info, carry=None):
        head_dim = query.shape[-1]
        if head_dim >= KERNEL_HEAD_DIM or query.device.type not in _DEVICE_TYPES:
            return stock(query, key, value, kv_info, carry)
        pad = (0, KERNEL_HEAD_DIM - head_dim)
        out, carry = stock(
            F.pad(query, pad), F.pad(key, pad), F.pad(value, pad), kv_info, carry
        )
        return out[..., :head_dim], carry

    return forward_ring_resume


def _segments(vae, name: str):
    """``vae.<name>.forward``, compiled once per segment shape under
    --enable-torch-compile (the first, causal segment is its own program)."""
    step = vae.__dict__.get(f"_tpu_{name}")
    if step is None:
        step = getattr(vae, name).forward
        if get_global_server_args().enable_torch_compile:
            step = torch.compile(step, **build_torch_compile_kwargs(mode=None))
        vae.__dict__[f"_tpu_{name}"] = step
    return step


def _encode(self, x: torch.Tensor):
    """``Kandinsky6SRVAE.encode``; on TPU each segment runs a compiled program."""
    if x.device.type not in _DEVICE_TYPES:
        return _STOCK_ENCODE(self, x)
    encoder = _segments(self, "encoder")
    split_list = kvae._segment_sizes(x.size(2), kvae._SEGMENT_FRAMES)
    cache = kvae._SegmentCache()
    latents = []
    for segment in torch.split(x, split_list, dim=2):
        latents.append(encoder(_fresh(segment), cache).chunk(2, dim=1)[0])
        cache.first = False
    return torch.cat(latents, dim=2), split_list


def _decode(self, z: torch.Tensor) -> torch.Tensor:
    """``Kandinsky6SRVAE._decode``; on TPU each segment runs a compiled program.
    Spatially sharded decodes keep the stock path."""
    if z.device.type not in _DEVICE_TYPES or self._spatial_parallel_decode_enabled:
        return _STOCK_DECODE(self, z)
    decoder = _segments(self, "decoder")
    split_list = kvae._segment_sizes(
        z.size(2), kvae._SEGMENT_FRAMES // self.temporal_compression
    )
    cache = kvae._SegmentCache()
    samples = []
    for chunk in torch.split(z, split_list, dim=2):
        samples.append(decoder(_fresh(chunk), cache))
        cache.first = False
    return torch.cat(samples, dim=2)


def _init(self, *args, **kwargs):
    _STOCK_INIT(self, *args, **kwargs)
    for module in self.modules():
        if not isinstance(module, USPAttention):
            continue
        impl = module.attn_impl
        if isinstance(impl, tpu_splash_attn.TpuSplashAttentionImpl) and not getattr(
            impl.forward, "__padded_for_tpu__", False
        ):
            impl.forward = padded_head_forward(impl)
            impl.forward_ring_resume = padded_head_ring_resume(impl)


def install() -> None:
    if getattr(_MODEL.__init__, "__wrapped_for_tpu__", False):
        return
    init = functools.wraps(_STOCK_INIT)(_init)
    init.__wrapped_for_tpu__ = True
    _MODEL.__init__ = init
    sr.shard_seq = _shard_seq
    _VAE.encode = _encode
    _VAE._decode = _decode
    sr.shard_like = _shard_like
    # Every rank runs the same block programs (offset-0 inputs, no
    # rank-dependent values).
    _MODEL._compile_regions_spmd_safe = True
