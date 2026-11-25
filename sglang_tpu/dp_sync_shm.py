# SPDX-License-Identifier: Apache-2.0
"""The DP MLP-sync row exchange through one host's shared memory instead of gloo.

Every scheduler step all-gathers a small int row per DP rank over the CPU
(gloo) group before the forward can be sized; on gloo that is loopback TCP
hops on every rank's serving thread. When every rank of the group runs on one
x86 host, each rank instead writes its row into a shared-memory block and spins
(yielding the GIL) until every peer has published the same step.

Publish protocol: a rank writes its row, then its sequence number. x86 keeps
stores in program order and loads in program order, so a reader that sees a
peer's sequence number also sees that peer's row. Rows alternate between two
slots: a rank can publish step s + 2 only after every rank published s + 1,
which each rank does only after it finished reading step s.
"""

from __future__ import annotations

import logging
import platform
import socket
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# torch.distributed's default gloo timeout; a dead peer fails like gloo would.
_TIMEOUT_S = 1800.0
# Polls before each GIL-releasing yield; arbitrary, the wait is the rank skew.
_SPINS_PER_YIELD = 64
_SEQ_STRIDE = 8  # one int64 sequence number per 64-byte cache line


class ShmRowExchange:
    """All-gather of one fixed-width CPU row per rank of ``group``, in group-rank order."""

    def __init__(
        self,
        *,
        group: torch.distributed.ProcessGroup,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        self._world = torch.distributed.get_world_size(group)
        self._rank = torch.distributed.get_rank(group)
        self._width = width
        np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        seq_bytes = 2 * self._world * _SEQ_STRIDE * 8
        data_bytes = 2 * self._world * width * np_dtype.itemsize
        self._shm = _open_shared_block(group=group, size=seq_bytes + data_bytes)
        buf = self._shm.buf
        self._seq = np.ndarray(
            (2, self._world, _SEQ_STRIDE), dtype=np.int64, buffer=buf
        )[:, :, 0]
        self._rows = np.ndarray(
            (2, self._world, width), dtype=np_dtype, buffer=buf, offset=seq_bytes
        )
        self._step = 0

    def exchange(self, global_flat: torch.Tensor, local_row: torch.Tensor) -> None:
        self._step += 1
        step, slot = self._step, self._step & 1
        self._rows[slot, self._rank] = local_row.reshape(-1).numpy()
        self._seq[slot, self._rank] = step
        self._wait_all(slot=slot, step=step)
        global_flat.view(self._world, self._width).copy_(
            torch.from_numpy(self._rows[slot].copy())
        )

    def _wait_all(self, *, slot: int, step: int) -> None:
        seq = self._seq[slot]
        spins = 0
        deadline = None
        while not (seq == step).all():
            spins += 1
            if spins % _SPINS_PER_YIELD:
                continue
            time.sleep(0)
            if deadline is None:
                deadline = time.monotonic() + _TIMEOUT_S
            elif time.monotonic() > deadline:
                missing = np.flatnonzero(seq != step).tolist()
                raise RuntimeError(
                    f"DP sync shared-memory exchange: ranks {missing} did not "
                    f"publish step {step} within {_TIMEOUT_S:.0f} s"
                )


def _open_shared_block(
    *, group: torch.distributed.ProcessGroup, size: int
) -> shared_memory.SharedMemory:
    """Group rank 0 creates the block and names it to the others; it is unlinked
    once all have mapped it, so no name outlives the processes."""
    name = [None]
    if torch.distributed.get_rank(group) == 0:
        shm = shared_memory.SharedMemory(create=True, size=size)
        np.ndarray((size,), dtype=np.uint8, buffer=shm.buf)[:] = 0
        name[0] = shm.name
    torch.distributed.broadcast_object_list(
        name, src=torch.distributed.get_global_rank(group, 0), group=group
    )
    if torch.distributed.get_rank(group) != 0:
        shm = _attach_untracked(name[0])
    torch.distributed.barrier(group=group)
    if torch.distributed.get_rank(group) == 0:
        shm.unlink()
    return shm


def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    """Map a block rank 0 owns without registering it with a resource tracker,
    which would unlink it again at exit (or double-unregister a shared tracker)."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    register = resource_tracker.register
    resource_tracker.register = lambda *args, **kwargs: None
    try:
        return shared_memory.SharedMemory(name=name)
    finally:
        resource_tracker.register = register


def _group_on_one_x86_host(group: torch.distributed.ProcessGroup) -> bool:
    """Collective: True on every rank iff all ranks share this host and it is x86."""
    local = (socket.gethostname(), platform.machine())
    hosts = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(hosts, local, group=group)
    return len(set(hosts)) == 1 and local[1] in ("x86_64", "AMD64")


class ShmExchangeRegistry:
    """One exchange per (group, width, dtype); None where shared memory can't serve."""

    def __init__(self) -> None:
        self._eligible: dict[int, bool] = {}
        self._exchanges: dict[tuple, ShmRowExchange] = {}

    def get(
        self, *, group: torch.distributed.ProcessGroup, local_row: torch.Tensor
    ) -> Optional[ShmRowExchange]:
        # Every input to this decision is uniform across the group, so all ranks
        # reach the collectives below together.
        if local_row.device.type != "cpu":
            return None
        if torch.distributed.get_world_size(group) < 2:
            return None
        group_key = id(group)
        if group_key not in self._eligible:
            self._eligible[group_key] = _group_on_one_x86_host(group)
            if not self._eligible[group_key]:
                logger.warning(
                    "DP sync shared-memory exchange needs every rank on one x86 "
                    "host; keeping the gloo exchange."
                )
        if not self._eligible[group_key]:
            return None
        key = (group_key, local_row.numel(), local_row.dtype)
        exchange = self._exchanges.get(key)
        if exchange is None:
            exchange = ShmRowExchange(
                group=group, width=local_row.numel(), dtype=local_row.dtype
            )
            self._exchanges[key] = exchange
        return exchange
