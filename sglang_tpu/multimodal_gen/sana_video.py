# SPDX-License-Identifier: Apache-2.0
"""SANA-Video on TPU workers: RoPE from float32 device tables, without float64.

Runtime patch, tracked debt:
- Why: SANA-Video's DiT keeps its RoPE tables in float64 on the host
  (``post_load_weights`` rebuilds them there). Every layer copies both tables
  to the device and rotates queries and keys in float64 through stride-2 lane
  slices and writes, which torch_tpu re-lays out (see
  :mod:`sglang_tpu.multimodal_gen.rope`). The split tables below stay on the
  device and give the same bfloat16 result with float32 device math.
- How: :func:`install` rebinds
  (1) ``SanaVideoTransformer3DModel.post_load_weights``: after the stock
      rebuild, each float64 table is split into float32 ``hi`` (16 significant
      bits) and ``lo`` (the remainder). The four tables stay on the worker's
      device as a plain attribute, each at storage offset 0; a module
      ``.to(dtype=...)`` casts only parameters and buffers;
  (2) ``SanaVideoRotaryPosEmbed.forward``: the stock token-order expansion,
      applied to the four tables inside the compiled graph;
  (3) ``apply_interleaved_rotary_emb_pair``: rotates adjacent pairs as
      ``x * cos + (x @ R) * sin``, with R the signed pair swap of
      :mod:`sglang_tpu.multimodal_gen.rope` (exact in bfloat16), in double
      float32. ``x * hi`` is exact for bfloat16 ``x``, a TwoSum keeps the
      rounding error of their sum, and the ``lo`` products add the rest. The
      bfloat16 result equals the stock float64 rotation, which also rounds
      through float32, on the 720p and 480p grids.
- Upstream PR or issue: none.
- Remove when: SANA-Video's RoPE takes a platform rotary implementation, or
  SANA-Video builds its tables on the device and torch_tpu lowers stride-2
  lane slices without a relayout.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits import sana_video as dit
from sglang_tpu.multimodal_gen.rope import _pair_swap_matrix

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_POST_LOAD = dit.SanaVideoTransformer3DModel.post_load_weights
_STOCK_ROPE_FORWARD = dit.SanaVideoRotaryPosEmbed.forward
_STOCK_ROTATE_PAIR = dit.apply_interleaved_rotary_emb_pair


def split_table(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 ``(hi, lo)`` of a float64 table: ``hi`` keeps 16 significant bits."""
    table = table.detach().to("cpu", torch.float64)
    hi = (table.float().view(torch.int32) & ~0xFF).view(torch.float32)
    # Exact in float64 (Sterbenz); lo rounds to float32.
    return hi, (table - hi.double()).float()


def post_load_weights(self) -> None:
    """The stock rebuild, then ``(cos_hi, cos_lo, sin_hi, sin_lo)`` on the device."""
    _STOCK_POST_LOAD(self)
    device = get_local_torch_device()
    if device.type not in _DEVICE_TYPES:
        return
    rope = self.rope
    tables = (*split_table(rope.freqs_cos), *split_table(rope.freqs_sin))
    rope.tpu_tables = tuple(table.contiguous().to(device) for table in tables)


def _grid_table(rope, table: torch.Tensor, shape) -> torch.Tensor:
    """``table`` in the stock token order, ``[1, frames * height * width, 1, D]``."""
    _, _, num_frames, height, width = shape
    patch_t, patch_h, patch_w = rope.patch_size
    grid = (num_frames // patch_t, height // patch_h, width // patch_w)
    parts = []
    for axis, part in enumerate(table.split(rope.split_sizes, dim=-1)):
        view = [1, 1, 1, -1]
        view[axis] = grid[axis]
        parts.append(part[: grid[axis]].view(view).expand(*grid, -1))
    return torch.cat(parts, dim=-1).reshape(1, -1, 1, table.shape[-1])


def rope_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    tables = getattr(self, "tpu_tables", None)
    if tables is None:
        return _STOCK_ROPE_FORWARD(self, hidden_states)
    return tuple(_grid_table(self, table, hidden_states.shape) for table in tables)


def _rotate(x, cos_hi, cos_lo, sin_hi, sin_lo) -> torch.Tensor:
    xf = x.float()
    xr = torch.matmul(x, _pair_swap_matrix(x.shape[-1], x.dtype, x.device)).float()
    # Exact: 8 by 16 significant bits.
    a = xf * cos_hi
    b = xr * sin_hi
    # TwoSum: s + err == a + b exactly.
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return (s + ((xf * cos_lo + xr * sin_lo) + err)).to(x.dtype)


def apply_interleaved_rotary_emb_pair(query, key, *freqs):
    """The stock pair rotation; four float32 tables take the double-float32 path."""
    if len(freqs) != 4:
        return _STOCK_ROTATE_PAIR(query, key, *freqs)
    if query.dtype is not torch.bfloat16:
        # x * hi is exact only for bfloat16: rotate with the tables hi + lo.
        cos_hi, cos_lo, sin_hi, sin_lo = (t.double() for t in freqs)
        return _STOCK_ROTATE_PAIR(query, key, cos_hi + cos_lo, sin_hi + sin_lo)
    return _rotate(query, *freqs), _rotate(key, *freqs)


def install() -> None:
    dit.SanaVideoTransformer3DModel.post_load_weights = post_load_weights
    dit.SanaVideoRotaryPosEmbed.forward = rope_forward
    dit.apply_interleaved_rotary_emb_pair = apply_interleaved_rotary_emb_pair
