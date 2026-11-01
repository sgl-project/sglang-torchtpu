# SPDX-License-Identifier: Apache-2.0
"""Lane-dense Ulysses all-to-all for TPU workers.

Runtime patch, tracked debt:
- Why: ``usp._usp_all_to_all_single`` flattens its input to 1-D before
  ``all_to_all_single``. XLA on TPU lowers an all-to-all of a rank-1 buffer
  through a chain of relayout copies around the collective (~9 ms of a ~29 ms
  MiniMax-H3 block at 32768 rows, against ~2.5 ms for the all-to-all itself).
- How: :func:`install` sends a ``[dim0, -1, last two dims]`` view instead.
  Every caller puts a world-divisible dimension first, so the rank chunks are
  the same bytes in the same order as the flat buffer's; other shapes keep
  the flat exchange.
- Upstream PR or issue: none.
- Remove when: the flat all-to-all lowers without copies on TPU.
"""

from __future__ import annotations

import functools

import torch


def lane_dense_all_to_all_single(flat_all_to_all):
    from sglang.multimodal_gen.runtime.layers import usp

    @functools.wraps(flat_all_to_all)
    def all_to_all(x: torch.Tensor, role: str | None = None) -> torch.Tensor:
        group = usp.get_sp_group().ulysses_group
        world = torch.distributed.get_world_size(group)
        if x.dim() < 3 or x.shape[0] % world:
            return flat_all_to_all(x, role)
        x_shape = x.shape
        x = x.contiguous().view(x_shape[0], -1, x_shape[-2] * x_shape[-1])
        if role is None:
            output = torch.empty_like(x)
        else:
            output = usp._a2a_staging_buffer(role, x.shape, x.dtype, x.device)
        torch.distributed.all_to_all_single(output, x, group=group)
        return output.reshape(x_shape)

    all_to_all.__wrapped_for_tpu__ = True
    return all_to_all


def install() -> None:
    from sglang.multimodal_gen.runtime.layers import usp

    if not getattr(usp._usp_all_to_all_single, "__wrapped_for_tpu__", False):
        usp._usp_all_to_all_single = lane_dense_all_to_all_single(
            usp._usp_all_to_all_single
        )
