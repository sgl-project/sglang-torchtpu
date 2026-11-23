# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo VAE decoding on TPU workers: the stock tiles split over the ranks.

Runtime patch, tracked debt:
- Why: with SP > 1 the HunyuanVideo VAE decodes with
  ``ParallelTiledVAE.parallel_tiled_decode``, eager on TPU. Each tile is a
  latent slice at a new storage offset, which torch_tpu guards; the ranks
  exchange sizes with ``.item()`` and shapes with ``all_gather_object``; every
  seam blends row by row, about 50K eager ops at 960x544x61 (a 305 s decode).
- How: :func:`install` replaces ``AutoencoderKLHunyuanVideo.parallel_tiled_decode``
  for latents on TPU, and its blends for TPU tensors:
  (1) rank r of the W ranks of the VAE decode group takes tiles
      [r * N // W, (r + 1) * N // W) of the stock grid (less a last frame slab
      that decodes to no frames), each copied to storage of its own;
  (2) each tile runs the stock ``_decode``, compiled under
      --enable-torch-compile unless SGLANG_TPU_ENABLE_HUNYUAN_VIDEO_DECODE_COMPILE=0:
      one program per tile shape;
  (3) the decoded shapes follow from the latent shape, so every rank fills a
      buffer sized on the host and the decode group all-gathers it once;
  (4) the stock merge, each seam blended as one weighted add over the band,
      each product rounded to the tile dtype before the add as in the stock
      row loop: eager, the video equals the stock decode bit for bit.
- Upstream PR or issue: none.
- Remove when: ``parallel_tiled_decode`` sizes its gather on the host and
  blends whole bands.
"""

from __future__ import annotations

import functools
import math

import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.models.vaes import hunyuanvae
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)
from sglang_tpu.envs import envs

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_VAE = hunyuanvae.AutoencoderKLHunyuanVideo
_STOCK_TILED_DECODE = _VAE.parallel_tiled_decode
# The dim of [B, C, T, H, W] each stock blend fades along.
_BLEND_DIMS = {"blend_t": -3, "blend_v": -2, "blend_h": -1}
_STOCK_BLENDS = {name: getattr(_VAE, name) for name in _BLEND_DIMS}


def _tile_decoder(vae):
    step = vae.__dict__.get("_tpu_tile_decode")
    if step is None:
        step = vae._decode
        if (
            get_global_server_args().enable_torch_compile
            and envs.SGLANG_TPU_ENABLE_HUNYUAN_VIDEO_DECODE_COMPILE.get()
        ):
            step = torch.compile(step, **build_torch_compile_kwargs(mode=None))
        vae.__dict__["_tpu_tile_decode"] = step
    return step


@functools.lru_cache(maxsize=16)
def _weights(extent: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The stock row weights ``1 - y / extent`` and ``y / extent``, as float32."""
    fade_in = torch.arange(extent, dtype=torch.float64) / extent
    return (1 - fade_in).float().to(device), fade_in.float().to(device)


def _band_blend(name: str):
    stock, dim = _STOCK_BLENDS[name], _BLEND_DIMS[name]

    def blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """The stock blend: ``b``'s head band fades in from ``a``'s tail, in place."""
        if b.device.type not in _DEVICE_TYPES:
            return stock(self, a, b, blend_extent)
        extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if extent > 0:
            trailing = (1,) * (-dim - 1)
            out_w, in_w = (w.view(-1, *trailing) for w in _weights(extent, b.device))
            head = b.narrow(dim, 0, extent)
            tail = a.narrow(dim, a.shape[dim] - extent, extent)
            head.copy_((tail * out_w).to(b.dtype) + (head * in_w).to(b.dtype))
        return b

    return blend


def _tile_grid(vae, shape):
    """The stock tiles as [t][h][w] lists of (latent start, latent size, decoded shape)."""
    batch, _, *dims = shape
    sr, tr = vae.spatial_compression_ratio, vae.temporal_compression_ratio
    names, ratios = ("num_frames", "height", "width"), (tr, sr, sr)
    sizes = [getattr(vae, f"tile_sample_min_{n}") // r for n, r in zip(names, ratios)]
    sizes[0] += 1  # a tile also takes the next slab's first latent frame
    starts = [
        range(0, dim, getattr(vae, f"tile_sample_stride_{n}") // r)
        for dim, n, r in zip(dims, names, ratios)
    ]
    grid = []
    for t0 in starts[0]:
        nt = min(sizes[0], dims[0] - t0)
        # Tiles after the first drop their first decoded frame.
        frames = (nt - 1) * tr + 1 - bool(t0)
        if frames == 0:  # the last slab, one latent frame
            continue
        grid.append([])
        for h0 in starts[1]:
            nh = min(sizes[1], dims[1] - h0)
            grid[-1].append([])
            for w0 in starts[2]:
                nw = min(sizes[2], dims[2] - w0)
                out = (batch, vae.config.out_channels, frames, nh * sr, nw * sr)
                grid[-1][-1].append(((t0, h0, w0), (nt, nh, nw), out))
    return grid


@torch.no_grad()
def parallel_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
    """``ParallelTiledVAE.parallel_tiled_decode`` with one all-gather."""
    if z.device.type not in _DEVICE_TYPES:
        return _STOCK_TILED_DECODE(self, z)
    group = None
    if parallel_state.model_parallel_is_initialized():
        group = parallel_state.get_decode_parallel_group_coordinator()
    world, rank = (group.world_size, group.rank_in_group) if group else (1, 0)
    grid = _tile_grid(self, z.shape)
    tiles = [tile for slab in grid for row in slab for tile in row]
    bounds = [r * len(tiles) // world for r in range(world + 1)]
    numels = [[math.prod(t[2]) for t in tiles[a:b]] for a, b in zip(bounds, bounds[1:])]
    rows = max(map(sum, numels))
    # The stage runs the VAE under autocast or with the latents in its dtype:
    # one dtype on every rank, including ranks with no tile.
    autocast = torch.is_autocast_enabled(z.device.type)
    dtype = torch.get_autocast_dtype(z.device.type) if autocast else z.dtype
    local = torch.empty(rows, dtype=dtype, device=z.device)
    decode = _tile_decoder(self)
    own = tiles[bounds[rank] : bounds[rank + 1]]
    parts = local[: sum(numels[rank])].split(numels[rank])
    for part, ((t0, h0, w0), (nt, nh, nw), out) in zip(parts, own):
        latent = z[:, :, t0 : t0 + nt, h0 : h0 + nh, w0 : w0 + nw]
        decoded = decode(latent.clone(memory_format=torch.contiguous_format))
        decoded = decoded[:, :, 1:] if t0 else decoded
        if decoded.shape != out:
            raise ValueError(f"tile decoded to {tuple(decoded.shape)}, not {out}")
        part.view(out).copy_(decoded)
    gathered = group.all_gather(local, dim=0) if group else local

    chunks = zip(gathered.view(world, rows), numels)
    parts = [p for chunk, n in chunks for p in chunk[: sum(n)].split(n)]
    views = (part.view(tile[2]) for part, tile in zip(parts, tiles))
    slabs, previous = [], None
    for i, slab in enumerate(grid):
        merged = self._merge_spatial_tiles(
            [[next(views) for _ in row] for row in slab],
            self.tile_sample_min_height - self.tile_sample_stride_height,
            self.tile_sample_min_width - self.tile_sample_stride_width,
            self.tile_sample_stride_height,
            self.tile_sample_stride_width,
        )
        if i:
            merged = self.blend_t(previous, merged, self.blend_num_frames)
        slabs.append(merged[:, :, : self.tile_sample_stride_num_frames + (i == 0)])
        previous = merged
    return torch.cat(slabs, dim=2)


def install() -> None:
    _VAE.parallel_tiled_decode = parallel_tiled_decode
    for name in _BLEND_DIMS:
        setattr(_VAE, name, _band_blend(name))
