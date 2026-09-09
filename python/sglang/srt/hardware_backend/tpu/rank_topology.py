# SPDX-License-Identifier: Apache-2.0
"""Logical/physical rank metadata shared by TPU Pallas collectives."""

from __future__ import annotations

import functools


def _get_global_device_id() -> int:
    """Return the physical TPU device ID, with a CPU-test fallback."""
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        return int(tpu_distributed.global_device_id())
    except Exception:
        return 0


@functools.cache
def get_tpu_physical_rank_metadata() -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Return local physical ID and TP physical/logical permutations.

    The tuple is ``(physical_id, physical_to_logical, logical_to_physical)``.
    Torch distributed ranks describe model ownership, while Pallas remote DMA
    addresses use physical TPU IDs. Keeping the translation in one place avoids
    accidentally treating process-group order as physical topology order.
    """
    import torch.distributed as dist

    physical_id = _get_global_device_id()
    if not dist.is_initialized():
        return physical_id, (0,), (physical_id,)

    try:
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        group = tp_group.cpu_group
        world_size = tp_group.world_size
        logical_rank = tp_group.rank_in_group
    except (AssertionError, RuntimeError):
        # Standalone distributed tests do not initialize SGLang parallel state.
        group = None
        world_size = dist.get_world_size()
        logical_rank = dist.get_rank()

    logical_to_physical = [None] * world_size
    dist.all_gather_object(
        logical_to_physical,
        physical_id,
        group=group,
    )
    if sorted(logical_to_physical) != list(range(world_size)):
        raise RuntimeError(f"Unexpected TPU physical device IDs: {logical_to_physical}")
    physical_to_logical = tuple(
        logical_to_physical.index(candidate) for candidate in range(world_size)
    )
    logical_to_physical = tuple(int(candidate) for candidate in logical_to_physical)
    if logical_to_physical[logical_rank] != physical_id:
        raise RuntimeError(
            "TPU local physical device ID does not match the gathered rank map: "
            f"logical rank {logical_rank} maps to "
            f"{logical_to_physical[logical_rank]}, got {physical_id}"
        )
    return physical_id, physical_to_logical, logical_to_physical


__all__ = ["get_tpu_physical_rank_metadata"]
