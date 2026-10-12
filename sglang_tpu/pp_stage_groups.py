# SPDX-License-Identifier: Apache-2.0
"""Process groups of a pipeline stage that spans several devices.

By default torch_tpu compiles the collectives of a process subgroup into
programs for every device of the slice, which all subgroups launch together.
Pipeline stages run their forwards at different times, so the collectives
inside one stage (the context-parallel K/V all-gather and the last-token
reduce) would wait for the devices of the other stages. A standalone subgroup
compiles them for its own devices, which launch them alone. The stage hand-off
stays one collective over the whole slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.distributed import parallel_state
from sglang_tpu._logging import get_logger

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.runtime_context import ParallelContext

logger = get_logger(__name__)


def _stage_groups(stage_ranks: list[int]) -> list[GroupCoordinator]:
    """Every built group of more than one device inside the stage of
    ``stage_ranks``, in creation order, which is the same on every rank."""
    stage = set(stage_ranks)
    groups = []
    for ref in parallel_state._groups.values():
        group = ref()
        if group is not None and group.world_size > 1 and set(group.ranks) <= stage:
            groups.append(group)
    return groups


def _load_make_standalone() -> Callable[[ProcessGroup], None]:
    try:
        from torch_tpu._internal.distributed.spmd_util import make_standalone
    except ImportError as e:
        raise RuntimeError(
            "Pipeline stages of more than one device (--pp-size > 1 with "
            "--tp-size > 1) need a torch_tpu with standalone process subgroups; "
            "the installed torch_tpu has no "
            "torch_tpu._internal.distributed.spmd_util.make_standalone."
        ) from e
    return make_standalone


def make_stage_groups_standalone(parallel: ParallelContext) -> None:
    """Let every pipeline stage of several devices launch its collectives
    without the other stages; must run before the first collective of the
    groups. Nothing to do for one-device stages or a single stage."""
    if parallel.pp_size == 1 or parallel.tp_size == 1:
        return
    make_standalone = _load_make_standalone()
    groups = _stage_groups(parallel.tp_group.ranks)
    for group in groups:
        # Returns once every member of the group has called it.
        make_standalone(group.device_group)
    logger.info(
        "pipeline stage %d: standalone process groups %s",
        parallel.pp_rank,
        [(group.unique_name, group.ranks) for group in groups],
    )
