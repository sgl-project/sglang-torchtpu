# SPDX-License-Identifier: Apache-2.0
"""Check that each multi-device pipeline stage fills whole chips, so no chip is
shared between stages and a pair's collectives stay on one chip, off ICI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import msgspec

from sglang_tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang.srt.runtime_context import ParallelContext

logger = get_logger(__name__)


class CoreLocation(msgspec.Struct, frozen=True):
    slice_index: int
    coords: tuple[int, ...]
    core_on_chip: int


def _partial_chip_problems(
    locations: Sequence[CoreLocation], *, devices_per_chip: int
) -> list[str]:
    """Problems with one stage's tensor-parallel group, locations[i] being the
    core of its rank i: each chip it uses must hold devices_per_chip of them."""
    cores_by_chip: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for loc in locations:
        cores_by_chip.setdefault((loc.slice_index, loc.coords), []).append(
            loc.core_on_chip
        )
    problems: list[str] = []
    for (_, coords), cores in sorted(cores_by_chip.items()):
        if len(set(cores)) != len(cores):
            problems.append(f"chip {coords} runs two ranks on one core: {cores}")
        elif len(cores) != devices_per_chip:
            problems.append(
                f"chip {coords} holds {len(cores)} of the group's devices, "
                f"not all {devices_per_chip}"
            )
    return problems


def _devices_per_chip() -> int | None:
    # Imported on call: the CPU unit lane imports this module without torch_tpu.
    from torch_tpu._internal.utils.hardware import get_devices_per_chip

    return get_devices_per_chip()


def _stage_core_locations(ranks: Sequence[int]) -> list[CoreLocation]:
    """The cores of world ranks ``ranks``, from the runtime's rank-to-device
    map and device list, so no collective runs."""
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.distributed.device_mesh import get_device_assignment

    rank_to_device = get_device_assignment()
    devices = {d.id: d for d in tpu_distributed.global_device_attributes()}
    locations = []
    for rank in ranks:
        device = (
            devices.get(rank_to_device[rank]) if rank < len(rank_to_device) else None
        )
        if device is None or not device.coords or device.core_on_chip is None:
            raise RuntimeError(
                f"The TPU runtime reports no chip coordinates or core for rank {rank} "
                f"(rank-to-device map {rank_to_device})."
            )
        locations.append(
            CoreLocation(
                # The runtime reports a slice only on multi-slice jobs.
                slice_index=device.slice_index or 0,
                coords=tuple(int(c) for c in device.coords),
                core_on_chip=int(device.core_on_chip),
            )
        )
    return locations


def validate_chips_for_pp(parallel: ParallelContext) -> None:
    """Raise on every rank of a multi-device pipeline stage that does not fill
    whole chips; a stage's devices are its tensor-parallel group."""
    if parallel.pp_size == 1 or parallel.tp_size == 1:
        return
    devices_per_chip = _devices_per_chip()
    if devices_per_chip is None:
        raise RuntimeError(
            "Multi-device pipeline stages need the TPU's devices per chip; "
            "torch_tpu does not recognize the attached TPU."
        )
    ranks = parallel.tp_group.ranks
    locations = _stage_core_locations(ranks)
    problems = _partial_chip_problems(locations, devices_per_chip=devices_per_chip)
    if problems:
        raise RuntimeError(
            f"Pipeline stage {parallel.pp_rank} (ranks {ranks}) does not fill "
            "whole chips: " + "; ".join(problems)
        )
    logger.info(
        "pipeline stage %d: tensor-parallel ranks %s on chips %s",
        parallel.pp_rank,
        ranks,
        sorted({loc.coords for loc in locations}),
    )
