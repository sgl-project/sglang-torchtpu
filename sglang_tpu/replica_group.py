# SPDX-License-Identifier: Apache-2.0
"""Engine-local coordinators backed by a shared TorchTPU session."""

import torch.distributed as dist

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang_tpu.dp_replicas import clear_replica_session, get_replica_session


class ReplicaGroupCoordinator(GroupCoordinator):
    @staticmethod
    def get_distributed_world_size() -> int:
        session = get_replica_session()
        if session is None:
            raise RuntimeError("No active TPU replica session")
        return session.tp_size

    def __init__(self, group_ranks, *args, **kwargs):
        session = get_replica_session()
        if session is None:
            raise RuntimeError("No active TPU replica session")
        if kwargs.get("rank_offset", 0) or kwargs.get("recovered_rank", False):
            raise ValueError("TPU engine replicas do not support elastic group joins")
        self._replica_session = session
        self._owns_session = kwargs.get("group_name") == "world"
        rank_base = session.dp_rank * session.tp_size
        engine_rank = dist.get_rank() - rank_base
        group_ranks = [ranks for ranks in group_ranks if engine_rank in ranks]
        if not group_ranks:
            raise ValueError(f"No replica group contains engine rank {engine_rank}")
        kwargs["rank_offset"] = rank_base
        super().__init__(group_ranks, *args, **kwargs)

    def _new_group(self, ranks, **kwargs):
        if kwargs["backend"] == "tpu_dist":
            return self._replica_session.device_groups[tuple(ranks)]
        if kwargs["backend"] != "gloo":
            raise ValueError(f"Unsupported TPU replica backend: {kwargs['backend']}")
        # Only members of this engine create its CPU groups, including DP0.
        kwargs["use_local_synchronization"] = True
        return dist.new_group(ranks, **kwargs)

    def destroy(self):
        # Device groups belong to the default distributed world. The base
        # coordinator still releases its independently created CPU group.
        self.device_group = None
        super().destroy()
        if self._owns_session:
            clear_replica_session(self._replica_session)
        self._replica_session = None
