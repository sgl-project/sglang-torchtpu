# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D-2 volume decoding on TPU workers: the dense grid split over the ranks.

Runtime patch, tracked debt:
- Why: ``VanillaVolumeDecoder`` evaluates the occupancy logits on a dense grid
  (385^3 points at octree resolution 384) in 32000-point chunks sliced from one
  grid tensor, so every chunk starts at a new storage offset, which torch_tpu
  guards: a compiled decoder would compile once per chunk. Each chunk also
  recomputes the cross-attention keys and values of the same latents, and every
  rank decodes the whole grid.
- How: :func:`install` replaces ``VanillaVolumeDecoder.__call__`` for latents
  on TPU and a ``CrossAttentionDecoder`` geometry decoder with
  ``downsample_ratio == 1`` and no K/V cache (other calls run the stock
  decoder):
  (1) the stock grid is cut at the stock chunk boundaries into flat tensors of
      their own (storage offset 0), built once per grid and rank. Rank r of the
      W ranks of the VAE decode group (TP/SP/PP/CFG, never across DP replicas)
      takes chunks [r * P, (r + 1) * P), P = ceil(chunks / W), so only the last
      chunk (the stock tail) has a second shape;
  (2) the keys and values are computed once per request with the stock layers,
      and each chunk runs the decoder's layers in the stock order on them
      (compiled under --enable-torch-compile unless
      SGLANG_TPU_ENABLE_HUNYUAN3D_DECODE_COMPILE=0) and returns flat logits;
  (3) each rank pads its logits to P chunks, the decode group all-gathers them
      and the padding is trimmed: eager, the grid equals the stock grid bit for
      bit. Surface extraction stays stock, on every rank.
- Upstream PR or issue: none.
- Remove when: the volume decoders take a platform-provided chunk layout and
  split the grid over the ranks.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.models.vaes import hunyuan3d_vae as vae_module
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)
from sglang_tpu.envs import envs

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_CALL = vae_module.VanillaVolumeDecoder.__call__


def _decode_group():
    """The ranks that share this request's decode, or None for one rank."""
    if not parallel_state.model_parallel_is_initialized():
        return None
    group = parallel_state.get_decode_parallel_group_coordinator()
    return group if group.world_size > 1 else None


def _compile_enabled() -> bool:
    return bool(
        get_global_server_args().enable_torch_compile
        and envs.SGLANG_TPU_ENABLE_HUNYUAN3D_DECODE_COMPILE.get()
    )


def _takes_chunk_path(geo_decoder) -> bool:
    return (
        isinstance(geo_decoder, vae_module.CrossAttentionDecoder)
        and geo_decoder.downsample_ratio == 1
        and not geo_decoder.cross_attn_decoder.attn.kv_cache
    )


def _rank_chunks(
    geo_decoder, bounds, octree_resolution, num_chunks, latents, world, rank
):
    """This rank's grid chunks, each a flat tensor of its own, built once per grid.

    Returns (chunks, grid_size, points, rows): ``rows`` is the logit count every
    rank pads to before the all-gather.
    """
    batch, dtype, device = latents.shape[0], latents.dtype, latents.device
    key = (tuple(bounds), octree_resolution, num_chunks, batch, dtype, device)
    key += (world, rank)
    cached = geo_decoder.__dict__.get("_tpu_volume_chunks")
    if cached is not None and cached[0] == key:
        return cached[1]
    xyz, grid_size, _ = vae_module.generate_dense_grid_points(
        bbox_min=np.array(bounds[0:3]),
        bbox_max=np.array(bounds[3:6]),
        octree_resolution=octree_resolution,
        indexing="ij",
    )
    xyz = torch.from_numpy(xyz).reshape(-1, 3)
    points = xyz.shape[0]
    rows = math.ceil(math.ceil(points / num_chunks) / world) * num_chunks
    chunks = [
        xyz[start : start + num_chunks].to(dtype).repeat(batch, 1).flatten().to(device)
        for start in range(rank * rows, min((rank + 1) * rows, points), num_chunks)
    ]
    layout = (chunks, grid_size, points, rows)
    geo_decoder.__dict__["_tpu_volume_chunks"] = (key, layout)
    return layout


def _decode_chunk(geo_decoder, queries: torch.Tensor, kv: torch.Tensor):
    """``CrossAttentionDecoder.forward`` on keys and values computed once.

    ``queries`` is one flat chunk (batch x points x 3 values); returns the
    chunk's logits as [batch, points].
    """
    block = geo_decoder.cross_attn_decoder
    attn = block.attn
    queries = queries.view(kv.shape[0], -1, 3)
    x = geo_decoder.query_proj(geo_decoder.fourier_embedder(queries).to(kv.dtype))
    x = x + attn.c_proj(attn.attention(attn.c_q(block.ln_1(x)), kv))
    x = x + block.mlp(block.ln_3(x))
    if geo_decoder.enable_ln_post:
        x = geo_decoder.ln_post(x)
    return geo_decoder.output_proj(x).flatten(1)


def _chunk_step(geo_decoder):
    step = geo_decoder.__dict__.get("_tpu_chunk_step")
    if step is None:

        def step(queries, kv):
            return _decode_chunk(geo_decoder, queries, kv)

        if _compile_enabled():
            step = torch.compile(step, **build_torch_compile_kwargs(mode=None))
        geo_decoder.__dict__["_tpu_chunk_step"] = step
    return step


@torch.no_grad()
def volume_decoder_call(
    self,
    latents: torch.Tensor,
    geo_decoder,
    bounds=1.01,
    num_chunks: int = 10000,
    octree_resolution: int = None,
    enable_pbar: bool = True,
    **kwargs,
):
    """``VanillaVolumeDecoder.__call__`` with the grid split over the ranks."""
    if latents.device.type not in _DEVICE_TYPES or not _takes_chunk_path(geo_decoder):
        return _STOCK_CALL(
            self,
            latents,
            geo_decoder,
            bounds=bounds,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            enable_pbar=enable_pbar,
            **kwargs,
        )
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    group = _decode_group()
    world, rank = (group.world_size, group.rank_in_group) if group else (1, 0)
    chunks, grid_size, points, rows = _rank_chunks(
        geo_decoder, bounds, octree_resolution, num_chunks, latents, world, rank
    )
    block = geo_decoder.cross_attn_decoder
    kv = block.attn.c_kv(block.ln_2(latents))
    step = _chunk_step(geo_decoder)
    logits = [step(queries, kv) for queries in chunks]
    if group is None:
        grid_logits = torch.cat(logits, dim=1)
    else:
        local = torch.cat(logits or [latents.new_zeros(latents.shape[0], 0)], dim=1)
        grid_logits = group.all_gather(F.pad(local, (0, rows - local.shape[1])), dim=1)
    return grid_logits[:, :points].reshape(latents.shape[0], *grid_size).float()


def install() -> None:
    """Route ``VanillaVolumeDecoder`` calls through :func:`volume_decoder_call`."""
    vae_module.VanillaVolumeDecoder.__call__ = volume_decoder_call
