"""Sizing and admission control for raiden's host staging pool.

A raiden engine reserves ``num_slots * max_blocks`` pages of pinned host memory
at startup and borrows one slot per in-flight transfer. Deriving the slot count
from a memory budget keeps a large context length from turning into a host
allocation that fails at startup; admission keeps a consumer from issuing more
concurrent pulls than the pool holds.
"""

from __future__ import annotations

from typing import Any, List, Set


class SlotAdmission:
    """Admission control over raiden's host staging slots.

    Raiden borrows one slot per in-flight transfer and rejects a read it cannot
    seat, so a consumer must not issue more concurrent reads than the pool
    holds. Reads that are not admitted stay pending and are retried. Keys are
    whatever identifies one read -- a room, or a room's chunk.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._in_flight: Set[Any] = set()

    def __len__(self) -> int:
        return len(self._in_flight)

    def __contains__(self, key: Any) -> bool:
        return key in self._in_flight

    def admit(self, key: Any) -> bool:
        """Reserves a slot for ``key``. False means try again later."""
        if key in self._in_flight:
            return True
        if len(self._in_flight) >= self.capacity:
            return False
        self._in_flight.add(key)
        return True

    def release(self, key: Any) -> None:
        self._in_flight.discard(key)

    def rooms(self) -> List[Any]:
        return list(self._in_flight)


def slot_host_bytes(kv_caches: List[Any], max_blocks: int) -> int:
    """Host staging bytes one transfer slot reserves.

    Raiden allocates ``max_blocks`` pages of host memory for every layer, so a
    slot costs the widest layer's page times the layer count times the per-slot
    block budget.
    """
    page_bytes = 0
    for buf in kv_caches:
        layer_bytes = buf.element_size()
        for dim in buf.shape[1:]:
            layer_bytes *= int(dim)
        page_bytes = max(page_bytes, layer_bytes)
    return page_bytes * len(kv_caches) * max_blocks


def ranks_per_node(
    attn_tp_size: int,
    attn_dp_size: int,
    system_dp_size: int,
    nnodes: int,
    attn_cp_size: int = 1,
) -> int:
    """Attention ranks resident on one host.

    A job's ranks -- every attention-TP rank of every context-parallel,
    attention-DP, and system-DP replica -- are spread evenly over its nodes,
    and each of them reserves its own host staging, so a per-node budget is
    shared by this many.
    """
    total = (
        max(1, attn_tp_size)
        * max(1, attn_dp_size)
        * max(1, system_dp_size)
        * max(1, attn_cp_size)
    )
    return max(1, total // max(1, nnodes))


def node_worker_rank(
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_dp_rank: int,
    attn_dp_size: int,
    system_dp_rank: int,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
) -> int:
    """Distinct rank of one worker among every worker on its node.

    Counts the attention-TP ranks of every context-parallel, attention-DP,
    and system-DP replica, so no two workers on a host share the ports,
    raiden engine id, or uuid namespace derived from it.
    """
    replica = system_dp_rank * max(1, attn_dp_size) + attn_dp_rank
    replica = replica * max(1, attn_cp_size) + attn_cp_rank
    return replica * max(1, attn_tp_size) + attn_tp_rank


def slots_within_budget(budget_bytes: float, slot_bytes: int, ceiling: int) -> int:
    """Slots that fit ``budget_bytes``, at most ``ceiling``.

    Raises ``ValueError`` when not even one slot fits: the pool cannot be
    allocated within the budget, so the configuration has to change.
    """
    if slot_bytes > budget_bytes:
        raise ValueError(
            "raiden host staging: one transfer slot needs "
            f"{slot_bytes / 1024**3:.2f} GiB but this engine's share of "
            f"SGLANG_RAIDEN_HOST_POOL_GB is {budget_bytes / 1024**3:.2f} GiB; "
            "raise SGLANG_RAIDEN_HOST_POOL_GB or lower the context length"
        )
    return max(1, min(ceiling, int(budget_bytes // max(1, slot_bytes))))
