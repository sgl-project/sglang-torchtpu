# SPDX-License-Identifier: Apache-2.0
"""Independent TP engines inside one TorchTPU session.

All workers join the native session during startup. TP1 engines retain their
independent singleton Gloo worlds; larger engines precreate TPU device groups
and schedule independently using only their own groups. On several hosts each
engine owns whole hosts, listed engine by engine (``node_normalization``), so
engine k is session ranks ``k * tp_size`` to ``(k + 1) * tp_size - 1``.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist

from sglang.utils import LazyImport

logger = logging.getLogger(__name__)
_active_session: ReplicaSession | None = None
# A multi-host session store listens on --dist-init-addr's port plus this, past the
# ZMQ and handshake ports that SGLang derives from it.
_SESSION_PORT_OFFSET = 20

# Platform discovery can reach this module while envs or srt.utils is importing.
# patches also imports envs, so resolve these helpers only during worker setup.
envs = LazyImport("sglang_tpu.envs", "envs")
get_free_port = LazyImport("sglang.srt.utils.network", "get_free_port")
_resolve_single_host_tpu_topology = LazyImport(
    "sglang_tpu.patches", "_resolve_single_host_tpu_topology"
)
compute_single_host_tpu_env = LazyImport(
    "sglang_tpu.patches", "compute_single_host_tpu_env"
)


def get_replica_session() -> ReplicaSession | None:
    return _active_session


def clear_replica_session(session: ReplicaSession) -> None:
    global _active_session
    if _active_session is session:
        _active_session = None


def prepare_host_session(
    dp_size: int, tp_size: int, dist_init_addr: str | None = None
) -> None:
    """Pick the session's ports in the parent, before workers spawn. On a multi-host
    launch (``dist_init_addr`` set), every host derives the same port from it."""
    if dp_size <= 1:
        return
    if dist_init_addr is not None:
        port = int(dist_init_addr.rpartition(":")[2]) + _SESSION_PORT_OFFSET
        os.environ.setdefault("SGLANG_TPU_DP_SESSION_PORT", str(port))
        # Loading an executable from the host's shared tier-2 cache can crash libtpu
        # (IdForLogicalDevice) in this session. Remove when TorchTPU fixes that.
        os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"
        os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"
        return
    world_size = dp_size * tp_size
    addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
    if addresses is None:
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = compute_single_host_tpu_env(
            world_size
        )["TORCH_TPU_SLICEBUILDER_ADDRESSES"]
    elif len(addresses.split(",")) != world_size:
        raise ValueError(
            f"TPU replicas need a session of {world_size} workers, but "
            f"TORCH_TPU_SLICEBUILDER_ADDRESSES lists {len(addresses.split(','))}; "
            "unset it to pick ports"
        )
    # Keep libtpu's global barrier at program start: with
    # --xla_jf_emit_global_barrier_at_start=false, a replica's collective that overlaps
    # another group's collective can halt the SparseCore or return wrong values.
    # Pick these once in the parent; all spawned workers inherit the same ports.
    for name in ("TORCH_TPU_INTERNAL_HANDSHAKE_PORT", "SGLANG_TPU_DP_SESSION_PORT"):
        if name not in os.environ:
            os.environ[name] = str(get_free_port())


def validate_dp_replicas(cfg) -> None:
    requirements = {
        "no Ray": not cfg.use_ray,
        "SGLANG_TPU_DP_REPLICA_SHAPE on several nodes": (
            cfg.nnodes == 1 or envs.SGLANG_TPU_DP_REPLICA_SHAPE.get() is not None
        ),
        "PP1": cfg.pp_size == 1,
        "devices from 0 (--base-gpu-id 0, --gpu-id-step 1)": (
            cfg.base_gpu_id == 0 and cfg.gpu_id_step == 1
        ),
        "EP contained within TP": cfg.ep_size > 0 and cfg.tp_size % cfg.ep_size == 0,
        "no attention DP/CP or MoE DP": (
            cfg.attn_dp_size == cfg.attn_cp_size == cfg.moe_dp_size == 1
        ),
        "no shared-expert TP override": cfg.shared_experts_tp_size is None,
        "TP1 on the single-host test platform": (
            os.environ.get("SGLANG_PLATFORM") != "tpu_single_host" or cfg.tp_size == 1
        ),
    }
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    if failed:
        raise ValueError(
            "TPU --dp-size replicas have an unsupported configuration: "
            + ", ".join(failed)
        )


def order_hosts_by_replica(
    order: list[int], topology, chips_per_host, dp_size: int
) -> list[int]:
    """List hosts (given in (z, y, x) order) replica by replica.

    Boxes of ``SGLANG_TPU_DP_REPLICA_SHAPE`` chips tile the slice in (z, y, x) order,
    and replica k gets the hosts of box k in (z, y, x) order, numbered like a
    standalone slice.
    """
    shape = tuple(int(d) for d in envs.SGLANG_TPU_DP_REPLICA_SHAPE.get().split("x"))
    grid = [t // c for t, c in zip(topology[:3], chips_per_host)]
    box = [s // c for s, c in zip(shape, chips_per_host)]
    if (
        len(shape) != 3
        or any(s % c for s, c in zip(shape, chips_per_host))
        or any(g % b for g, b in zip(grid, box))
        or math.prod(grid) != dp_size * math.prod(box)
    ):
        raise ValueError(
            f"SGLANG_TPU_DP_REPLICA_SHAPE={'x'.join(map(str, shape))} must tile the "
            f"{'x'.join(map(str, topology[:3]))} slice with {dp_size} boxes of whole hosts"
        )

    def key(rank: int) -> tuple[int, ...]:
        # Ranks follow (z, y, x) host order, so within a box they already do.
        z, rest = divmod(rank, grid[0] * grid[1])
        y, x = divmod(rest, grid[0])
        return (z // box[2], y // box[1], x // box[0], rank)

    return [order[rank] for rank in sorted(range(len(order)), key=key)]


def device_group_ranks(
    dp_size: int, tp_size: int, ep_size: int, dcp_size: int = 1
) -> list[tuple[int, ...]]:
    """TP, singleton, EP, expert-TP and DCP groups, all confined to their engine."""
    if dp_size < 1 or tp_size < 1 or ep_size < 1 or tp_size % ep_size:
        raise ValueError("DP/TP/EP sizes must be positive and EP must divide TP")
    world_size = dp_size * tp_size
    etp_size = tp_size // ep_size
    groups = [
        tuple(range(base, base + tp_size)) for base in range(0, world_size, tp_size)
    ]
    groups.extend((rank,) for rank in range(world_size))
    for base in range(0, world_size, tp_size):
        groups.extend(
            tuple(range(base + rank, base + tp_size, etp_size))
            for rank in range(etp_size)
        )
        groups.extend(
            tuple(range(start, start + size))
            for size in (etp_size, dcp_size)
            for start in range(base, base + tp_size, size)
        )
    return list(dict.fromkeys(groups))


@dataclass
class ReplicaSession:
    dp_rank: int
    dp_size: int
    tp_size: int
    ep_size: int
    device: int
    timeout_seconds: int = 7200
    dcp_size: int = 1
    device_groups: dict[tuple[int, ...], dist.ProcessGroup] = field(
        init=False, default_factory=dict
    )

    @contextmanager
    def initialize(self):
        global _active_session
        if _active_session is not None:
            raise RuntimeError("A TPU replica session is already active")
        rank_base = self.dp_rank * self.tp_size
        rank = rank_base + int(os.environ["RANK"])
        world_size = self.dp_size * self.tp_size
        if rank != self.device:
            raise ValueError(f"Session rank {rank} must bind device {self.device}")
        hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",")
        local_world_size = world_size
        if len(hosts) > 1:
            local_world_size = int(os.environ["TORCH_TPU_DEVICES_PER_HOST"])
        values = dict(
            RANK=str(rank),
            LOCAL_RANK=str(rank % local_world_size),
            WORLD_SIZE=str(world_size),
            LOCAL_WORLD_SIZE=str(local_world_size),
        )
        saved = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        if self.tp_size == 1:
            os.environ["TPU_VISIBLE_DEVICES"] = str(self.device)
        try:
            yield
            # On several hosts the launcher already set the whole slice's topology.
            if len(hosts) == 1:
                topology = _resolve_single_host_tpu_topology(world_size)
                if topology is None:
                    raise RuntimeError(
                        f"No TPU topology for {world_size} session workers"
                    )
                preset = os.environ.get("TORCH_TPU_TOPOLOGY")
                if preset not in (None, topology):
                    raise ValueError(
                        f"TORCH_TPU_TOPOLOGY={preset!r} is not the topology "
                        f"{topology!r} of the {world_size} session workers; unset it"
                    )
                os.environ["TORCH_TPU_TOPOLOGY"] = topology
            # Materialize PJRT while its environment still describes the host.
            torch.ones(1, device="tpu").item()
            if self.tp_size == 1:
                # No TPU collectives: SGLang creates a separate singleton Gloo
                # world per engine after restoring its local rank below.
                return
            if dist.is_initialized():
                raise RuntimeError(
                    "The TPU replica session must own distributed initialization"
                )
            store_host = hosts[0] if len(hosts) > 1 else "127.0.0.1"
            dist.init_process_group(
                "tpu_dist",
                init_method=f"tcp://{store_host}:{envs.SGLANG_TPU_DP_SESSION_PORT.get()}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=self.timeout_seconds),
            )
            groups = {}
            for ranks in device_group_ranks(
                self.dp_size, self.tp_size, self.ep_size, self.dcp_size
            ):
                group = dist.new_group(
                    list(ranks),
                    backend="tpu_dist",
                    timeout=timedelta(seconds=self.timeout_seconds),
                )
                if rank in ranks:
                    groups[ranks] = group
            self.device_groups = groups
            _active_session = self
            if len(hosts) > 1:
                _default_pallas_mesh_to_engine(
                    groups[tuple(range(rank_base, rank_base + self.tp_size))],
                    self.tp_size,
                )
            logger.info(
                "TPU DP%d session rank=%d, engine groups=%s",
                self.dp_rank,
                rank,
                sorted(groups),
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _default_pallas_mesh_to_engine(tp_group, tp_size: int) -> None:
    """Build a Pallas mesh requested without a process group over the engine.

    Why: kernels build meshes of their TP size with no process group, and
    TorchTPU resolves that to the world group, here the whole session.
    How: such a request, sized for the engine or unsized, gets the engine's TP
    device group.
    Remove when: those kernels pass their TP group.
    """
    from torch_tpu._internal import pallas

    original = pallas.get_pallas_mesh

    @functools.wraps(original)
    def get_pallas_mesh(axis_names, mesh_shape=None, process_group=None):
        if process_group is None and (
            mesh_shape is None or math.prod(mesh_shape) == tp_size
        ):
            process_group = tp_group
        return original(axis_names, mesh_shape, process_group)

    pallas.get_pallas_mesh = get_pallas_mesh
