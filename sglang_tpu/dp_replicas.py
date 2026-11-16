# SPDX-License-Identifier: Apache-2.0
"""Independent TP engines inside one host-wide TorchTPU session.

All workers join the native session during startup. TP1 engines retain their
independent singleton Gloo worlds; larger engines precreate TPU device groups
and schedule independently using only their own groups.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist

from sglang.utils import LazyImport

logger = logging.getLogger(__name__)
_active_session: ReplicaSession | None = None

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


def prepare_host_session(dp_size: int, tp_size: int) -> None:
    if dp_size <= 1:
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
        "one node without Ray": cfg.nnodes == 1 and not cfg.use_ray,
        "PP1": cfg.pp_size == 1,
        "devices from 0 (--base-gpu-id 0, --gpu-id-step 1)": (
            cfg.base_gpu_id == 0 and cfg.gpu_id_step == 1
        ),
        "EP contained within TP": cfg.ep_size > 0 and cfg.tp_size % cfg.ep_size == 0,
        "no attention DP/CP/DCP or MoE DP": (
            cfg.attn_dp_size == cfg.attn_cp_size == cfg.dcp_size == cfg.moe_dp_size == 1
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


def device_group_ranks(
    dp_size: int, tp_size: int, ep_size: int
) -> list[tuple[int, ...]]:
    """TP, singleton, EP and expert-TP groups, all confined to their engine."""
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
            tuple(range(start, start + etp_size))
            for start in range(base, base + tp_size, etp_size)
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
        values = dict(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_WORLD_SIZE=str(world_size),
        )
        saved = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        if self.tp_size == 1:
            os.environ["TPU_VISIBLE_DEVICES"] = str(self.device)
        try:
            yield
            topology = _resolve_single_host_tpu_topology(world_size)
            if topology is None:
                raise RuntimeError(f"No TPU topology for {world_size} session workers")
            preset = os.environ.get("TORCH_TPU_TOPOLOGY")
            if preset not in (None, topology):
                raise ValueError(
                    f"TORCH_TPU_TOPOLOGY={preset!r} is not the topology {topology!r} "
                    f"of the {world_size} session workers; unset it"
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
            dist.init_process_group(
                "tpu_dist",
                init_method=f"tcp://127.0.0.1:{envs.SGLANG_TPU_DP_SESSION_PORT.get()}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=self.timeout_seconds),
            )
            groups = {}
            for ranks in device_group_ranks(self.dp_size, self.tp_size, self.ep_size):
                group = dist.new_group(
                    list(ranks),
                    backend="tpu_dist",
                    timeout=timedelta(seconds=self.timeout_seconds),
                )
                if rank in ranks:
                    groups[ranks] = group
            self.device_groups = groups
            _active_session = self
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
