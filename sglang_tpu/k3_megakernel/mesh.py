# SPDX-License-Identifier: Apache-2.0
"""Process start-up and the rank-ordered device mesh.

Rank layout the kernel assumes (config.Topology): ``rank = 8 * host + lane``,
``rank ^ 1`` is the other TensorCore of the same chip, and ``rank ^ 2``,
``rank ^ 4`` are other chips of the same host. Lane ``l`` sits on the same
(x, y) chip position on every host, so cross-host peers are z-neighbours.
"""

import jax
import numpy as np
from jax.sharding import Mesh

AXIS = "x"


def init_distributed(*, coordinator: str, num_processes: int, process_id: int) -> None:
    """One process per host; a no-op for the single-host (one process) mode."""
    if num_processes <= 1:
        return
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=num_processes,
        process_id=process_id,
        initialization_timeout=300,
    )


def sort_key(dev):
    x, y, z = (tuple(dev.coords) + (0, 0, 0))[:3]
    return (dev.process_index, z, y, x, dev.core_on_chip)


def ordered_devices(lanes: int = 8) -> list:
    """All TPU devices in rank order; raises if the pairing invariants do not hold."""
    devs = sorted(jax.devices(), key=sort_key)
    world = len(devs)
    if world % lanes:
        raise ValueError(f"{world} devices is not a multiple of {lanes} lanes")
    procs = sorted({d.process_index for d in devs})
    for r, d in enumerate(devs):
        if d.process_index != procs[r // lanes]:
            raise ValueError(f"rank {r} is not host-major")
        pair = devs[r ^ 1]
        if tuple(pair.coords) != tuple(d.coords) or pair.core_on_chip == d.core_on_chip:
            raise ValueError(
                f"rank {r} ({d}) and rank {r ^ 1} ({pair}) are not one chip"
            )
        for off in range(2, lanes):
            peer = devs[r ^ off]
            if peer.process_index != d.process_index:
                raise ValueError(f"rank {r} and rank {r ^ off} are on different hosts")
        lane_ref = devs[r % lanes]
        if (
            tuple(d.coords)[:2] != tuple(lane_ref.coords)[:2]
            or d.core_on_chip != lane_ref.core_on_chip
        ):
            raise ValueError(f"rank {r} is not at the (x, y, core) of lane {r % lanes}")
    return devs


def make_mesh(lanes: int = 8) -> Mesh:
    return Mesh(np.array(ordered_devices(lanes)), (AXIS,))


def describe(devs) -> list[str]:
    return [
        f"rank {r:2d}: id {d.id:3d} proc {d.process_index} coords {tuple(d.coords)} core {d.core_on_chip}"
        for r, d in enumerate(devs)
    ]


def local_ranks(mesh: Mesh) -> list[int]:
    """Mesh positions of this process's devices (a contiguous host block)."""
    me = jax.process_index()
    return [r for r, d in enumerate(mesh.devices.flat) if d.process_index == me]
