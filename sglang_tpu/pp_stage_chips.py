# SPDX-License-Identifier: Apache-2.0
"""Place pipeline stages on the TPU slice and check where they sit.

Each multi-device stage must fill whole chips, so no chip is shared between
stages and a pair's collectives stay on one chip, off ICI. Consecutive stages
should sit on neighboring chips, since a hand-off between them may cross the
chips in between. Where the launch can order single-chip stages as a ring of
one-hop hand-offs (one host, or two hosts of a 2x2x2 slice), it does, and a
longer hand-off within the slice fails the launch; elsewhere it is warned
about."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs as core_envs
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs
from sglang_tpu.node_normalization import dimensions

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
    return torch.tpu.get_device_properties().devices_per_chip


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


def _hops(a: CoreLocation, b: CoreLocation) -> int | None:
    """ICI hops between the chips of two cores: 0 on one chip, else the
    Manhattan distance between the chips' coordinates, or None across slices.

    No wraparound links are assumed (a dimension of size 2 has none), so on a
    torus with longer dimensions this can overstate the distance."""
    if a.slice_index != b.slice_index:
        return None
    return sum(abs(x - y) for x, y in zip(a.coords, b.coords, strict=True))


def _boundary_hops(
    locations: Sequence[CoreLocation],
) -> list[tuple[int, int, int | None]]:
    """(stage, next stage, hops) for every hand-off, locations[k] being the
    core of stage k; the last stage's relay back to the first is included."""
    n = len(locations)
    return [
        (k, (k + 1) % n, _hops(locations[k], locations[(k + 1) % n])) for k in range(n)
    ]


def _fmt_hops(stage: int, next_stage: int, hops: int | None, *, n: int) -> str:
    relay = " (relay)" if next_stage == 0 and stage == n - 1 else ""
    where = "other slice" if hops is None else str(hops)
    return f"{stage}->{next_stage}{relay}: {where}"


def _fmt_far(
    far: Sequence[tuple[int, int, int | None]],
    locations: Sequence[CoreLocation],
) -> str:
    n = len(locations)
    return ", ".join(
        f"{_fmt_hops(k, nxt, h, n=n)} (chips {locations[k].coords} to "
        f"{locations[nxt].coords})"
        for k, nxt, h in far
    )


def _check_stage_adjacency(parallel: ParallelContext, *, ring_expected: bool) -> None:
    """Check the ICI hops of each stage hand-off in this rank's column of the
    pipeline (the ranks of its pipeline group, one per stage), the last
    stage's relay back to the first included.

    Where the launch ordered the stages as a ring of one-hop hand-offs
    (``ring_expected``), a longer hand-off within the slice means the devices
    are not in the order it assumed, so every rank raises unless
    SGLANG_TPU_PP_ALLOW_DISTANT_STAGES is set. Elsewhere longer hand-offs are
    warned about. Hand-offs to another slice are only logged. The column's
    first stage does the logging."""
    reporter = parallel.pp_rank == 0
    ranks = parallel.pp_group.ranks
    try:
        locations = _stage_core_locations(ranks)
    except Exception as exc:  # without coordinates there is nothing to check
        if reporter:
            logger.warning("pipeline stage adjacency not checked: %s", exc)
        return
    n = len(locations)
    hops = _boundary_hops(locations)
    if reporter:
        logger.info(
            "pipeline ranks %s: ICI hops per stage hand-off %s",
            ranks,
            ", ".join(_fmt_hops(k, nxt, h, n=n) for k, nxt, h in hops),
        )
    far = [(k, nxt, h) for k, nxt, h in hops if h is not None and h > 1]
    if not far:
        return
    allowed = envs.SGLANG_TPU_PP_ALLOW_DISTANT_STAGES.get()
    if ring_expected and not allowed:
        raise RuntimeError(
            f"Pipeline ranks {ranks}: stage hand-offs {_fmt_far(far, locations)} "
            "cross more than one ICI hop within the slice, but this launch "
            "ordered its single-chip stages as a ring of one-hop hand-offs, so "
            "the TPU devices are not in the order it assumes. Set "
            "SGLANG_TPU_PP_ALLOW_DISTANT_STAGES=1 to start anyway."
        )
    if reporter:
        logger.warning(
            "pipeline ranks %s: stage hand-offs %s are not between neighboring "
            "chips, so each may cross the chips in between (wraparound links "
            "are not counted); %s",
            ranks,
            _fmt_far(far, locations),
            (
                "allowed by SGLANG_TPU_PP_ALLOW_DISTANT_STAGES"
                if ring_expected
                else "the launch orders stages as a one-hop ring only when each "
                "fits on one chip on one host or two hosts of a 2x2x2 slice"
            ),
        )


def validate_chips_for_pp(parallel: ParallelContext, *, ring_expected: bool) -> None:
    """Raise on every rank of a multi-device pipeline stage that does not fill
    whole chips (a stage's devices are its tensor-parallel group), then check
    how far apart consecutive stages sit. ``ring_expected`` is
    PipelinePlacement.ring_expected for this worker."""
    if parallel.pp_size == 1:
        return
    devices_per_chip = None
    if parallel.tp_size > 1 or ring_expected:
        devices_per_chip = _devices_per_chip()
    if parallel.tp_size > 1:
        if devices_per_chip is None:
            raise RuntimeError(
                "Multi-device pipeline stages need the TPU's devices per chip; "
                "torch_tpu does not recognize the attached TPU."
            )
        _validate_whole_chips(parallel, devices_per_chip=devices_per_chip)
    # The native ring order is known only for hosts of four two-core chips.
    ring_expected = (
        ring_expected
        and devices_per_chip == _RING_DEVICES_PER_CHIP
        and parallel.tp_size <= devices_per_chip
    )
    _check_stage_adjacency(parallel, ring_expected=ring_expected)


def _validate_whole_chips(parallel: ParallelContext, *, devices_per_chip: int) -> None:
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


class PipelinePlacement(msgspec.Struct, frozen=True):
    # The local device this worker binds in place of its native one, or None.
    local_device: int | None
    # Whether the stages form a ring of one-hop hand-offs, so validate_chips_for_pp
    # refuses a longer one.
    ring_expected: bool


_NATIVE = PipelinePlacement(local_device=None, ring_expected=False)
# The two-host slice the mirrored order is verified on: four chips per host,
# the hosts stacked along z.
_TWO_HOST_RING_SLICE = (2, 2, 2)
_RING_DEVICES_PER_CHIP = 2
# A host of four two-core chips.
_HOST_RING_DEVICES = 4 * _RING_DEVICES_PER_CHIP


def pipeline_placement(cfg, *, tp_rank: int, pp_rank: int) -> PipelinePlacement:
    """Where this scheduler worker's stage goes, from the rank layout the
    worker bootstrap gives torch_tpu: pp-major ranks filling hosts in order.

    Each host's native device order walks its chips as a ring, so single-chip
    stages filling one host already form a one-hop ring. On two hosts of a 2x2x2
    slice, the second host walks its chips backwards: its first stage then
    sits next to the first host's last, and its last stage next to the first
    host's first. Cores keep their place on the chip, so a stage's
    tensor-parallel group stays on one chip. validate_chips_for_pp checks the
    result once the runtime is up."""
    from sglang_tpu import shared_slice

    if (
        cfg.device != "tpu"
        or cfg.pp_size == 1
        or core_envs.SGLANG_TPU_PD_COLOCATED.get()  # binds its devices itself
        or shared_slice.spec() is not None  # rebinds LOCAL_RANK for the slice
    ):
        return _NATIVE
    if cfg.nnodes == 1:
        # Only stages filling all four chips close the ring; three chips of it
        # leave the relay two hops long.
        fills_host = cfg.tp_size * cfg.pp_size == _HOST_RING_DEVICES
        single_chip = _RING_DEVICES_PER_CHIP % cfg.tp_size == 0
        return PipelinePlacement(
            local_device=None, ring_expected=fills_host and single_chip
        )
    topology = (
        cfg.torch_tpu_topology
        or os.environ.get("TORCH_TPU_TOPOLOGY")
        or os.environ.get("TPU_TOPOLOGY")
    )
    if cfg.nnodes != 2 or not topology:
        return _NATIVE
    if dimensions(topology)[:3] != _TWO_HOST_RING_SLICE:
        return _NATIVE
    chips_per_host = math.prod(_TWO_HOST_RING_SLICE) // 2
    local_world_size = cfg.tp_size * cfg.pp_size // 2
    devices_per_chip, rest = divmod(local_world_size, chips_per_host)
    if rest or devices_per_chip != _RING_DEVICES_PER_CHIP:
        return _NATIVE  # the native ring order is known only for two-core chips
    if devices_per_chip % cfg.tp_size:
        return _NATIVE  # multi-chip stages have no such ring
    node_index, local_rank = divmod(pp_rank * cfg.tp_size + tp_rank, local_world_size)
    if node_index == 0:
        return PipelinePlacement(local_device=None, ring_expected=True)
    chip, core = divmod(local_rank, devices_per_chip)
    if local_rank == 0:
        logger.info(
            "pipeline stages on this host bind its chips in reverse order, so "
            "stage hand-offs stay between neighboring chips"
        )
    return PipelinePlacement(
        local_device=(chips_per_host - 1 - chip) * devices_per_chip + core,
        ring_expected=True,
    )
