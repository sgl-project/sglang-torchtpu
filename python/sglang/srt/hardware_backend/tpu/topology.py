# SPDX-License-Identifier: Apache-2.0
"""Unified TPU topology and physical device mapping discovery for Cloud TPU v7x.

Centralizes physical device discovery, TP rank to device ID lookup, and physical
optical ring neighbor derivation for Pallas kernels and distributed operations.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence


def get_global_device_id() -> int:
    """Return the physical TPU device ID, defaulting to 0 if unavailable."""
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        return int(tpu_distributed.global_device_id())
    except Exception:
        return 0


import os

_CACHED_TP_RANK_TO_DEVICE_ID_MAPPING: tuple[int, tuple[int, ...]] | None = None


def get_tp_rank_to_device_id_mapping() -> tuple[int, tuple[int, ...]]:
    """Return (global_device_id, tp_rank_to_device_id) for the active TP communication group.

    Uses a 3-tier fallback hierarchy:
    1. Uninitialized distributed runtime -> returns (global_device_id, tuple(range(world_size_env))).
    2. SGLang TP group (get_tp_group()) if available.
    3. Default raw torch.distributed world_size fallback (for standalone unit tests).
    """
    global _CACHED_TP_RANK_TO_DEVICE_ID_MAPPING
    if _CACHED_TP_RANK_TO_DEVICE_ID_MAPPING is not None:
        return _CACHED_TP_RANK_TO_DEVICE_ID_MAPPING

    import torch.distributed as dist

    global_device_id = get_global_device_id()
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist.is_initialized():
        return global_device_id, tuple(range(world_size_env))

    try:
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        group, world_size = tp_group.cpu_group, tp_group.world_size
    except (AssertionError, RuntimeError, AttributeError):
        # Fallback for standalone unit tests running raw torch.distributed without SGLang parallel_state
        group, world_size = None, dist.get_world_size()

    physical_ids = [None] * world_size
    dist.all_gather_object(physical_ids, global_device_id, group=group)
    if sorted(physical_ids) != list(range(world_size)):
        if all(x == 0 for x in physical_ids):
            # Fallback for non-TPU or unassigned physical devices
            physical_ids = list(range(world_size))
        else:
            raise RuntimeError(f"Unexpected TPU physical device IDs: {physical_ids}")

    mapping = (global_device_id, tuple(int(x) for x in physical_ids))
    _CACHED_TP_RANK_TO_DEVICE_ID_MAPPING = mapping
    return mapping


def get_physical_device_id(
    tp_rank: int, tp_rank_to_device_id: Sequence[int]
) -> int:
    """Return the physical device ID for a given TP logical rank."""
    return int(tp_rank_to_device_id[tp_rank])


def get_cw_tp_rank(tp_rank: int, world_size: int) -> int:
    """Return the logical TP rank of the clockwise neighbor in a ring."""
    return (tp_rank + 1) % world_size


def get_ccw_tp_rank(tp_rank: int, world_size: int) -> int:
    """Return the logical TP rank of the counter-clockwise neighbor in a ring."""
    return (tp_rank - 1) % world_size


def get_cw_device_id(
    tp_rank: int, tp_rank_to_device_id: Sequence[int]
) -> int:
    """Return the physical device ID of the clockwise neighbor."""
    cw_rank = get_cw_tp_rank(tp_rank, len(tp_rank_to_device_id))
    return int(tp_rank_to_device_id[cw_rank])


def get_ccw_device_id(
    tp_rank: int, tp_rank_to_device_id: Sequence[int]
) -> int:
    """Return the physical device ID of the counter-clockwise neighbor."""
    ccw_rank = get_ccw_tp_rank(tp_rank, len(tp_rank_to_device_id))
    return int(tp_rank_to_device_id[ccw_rank])


def get_physical_ring_neighbors(
    tp_rank: int, tp_rank_to_device_id: Sequence[int]
) -> tuple[int, int]:
    """Return (cw_device_id, ccw_device_id) physical device IDs for a given TP logical rank."""
    return (
        get_cw_device_id(tp_rank, tp_rank_to_device_id),
        get_ccw_device_id(tp_rank, tp_rank_to_device_id),
    )
