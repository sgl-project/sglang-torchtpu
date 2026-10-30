"""Sizing and admission control for raiden's host staging pool.

A raiden engine reserves ``num_slots * max_blocks`` pages of pinned host memory
at startup and seats each in-flight transfer in as many of those pages as the
transfer needs (in a whole slot when fixed slots are configured). Deriving the
slot count from a memory budget keeps a large context length from turning into
a host allocation that fails at startup; admission keeps a consumer from
issuing pulls the pool cannot seat.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class StagingAdmission:
    """Admission control over raiden's host staging, in pages.

    Raiden seats each in-flight read in host pages of the pool it allocated at
    startup and rejects a read it cannot seat, so a consumer must not issue
    reads that together need more pages than the pool holds. Reads that are
    not admitted stay pending and are retried. A hybrid model's sliding-window
    pages and GDN state rows are seated by their own engines with their own
    pools, so a read that carries them is charged against each of those pools.
    Keys are whatever identifies one read -- a room, or a room's chunk.

    Reads are admitted in the order they ask: a read that did not fit holds
    the head of the line until it is admitted or forgotten, so a stream of
    smaller reads cannot keep a larger one waiting indefinitely.
    """

    def __init__(
        self,
        capacity: int,
        swa_capacity: Optional[int] = None,
        state_capacity: Optional[int] = None,
    ):
        self.capacity = max(1, int(capacity))
        self.swa_capacity = None if swa_capacity is None else max(1, int(swa_capacity))
        self.state_capacity = (
            None if state_capacity is None else max(1, int(state_capacity))
        )
        self._in_flight: Dict[Any, Tuple[int, int, int]] = {}
        self._waiting: Optional[Any] = None

    def __len__(self) -> int:
        return len(self._in_flight)

    def __contains__(self, key: Any) -> bool:
        return key in self._in_flight

    @property
    def waiting(self) -> Optional[Any]:
        """The read holding the head of the line, if any."""
        return self._waiting

    def pages(self) -> int:
        return sum(pages for pages, _, _ in self._in_flight.values())

    def swa_pages(self) -> int:
        return sum(swa_pages for _, swa_pages, _ in self._in_flight.values())

    def state_pages(self) -> int:
        return sum(state_pages for _, _, state_pages in self._in_flight.values())

    def admit(
        self, key: Any, pages: int, swa_pages: int = 0, state_pages: int = 0
    ) -> bool:
        """Reserves ``pages`` of the pool (``swa_pages`` of the sliding-window
        pool, ``state_pages`` of the GDN state pool) for ``key``. False means
        try again later."""
        if key in self._in_flight:
            return True
        pages, swa_pages, state_pages = int(pages), int(swa_pages), int(state_pages)
        if (
            pages > self.capacity
            or _exceeds(swa_pages, self.swa_capacity)
            or _exceeds(state_pages, self.state_capacity)
        ):
            raise ValueError(
                f"read {key!r} needs {pages} pages, {swa_pages} sliding-window "
                f"pages and {state_pages} state rows, more than the pool holds "
                f"({self.capacity}, {self.swa_capacity} and {self.state_capacity})"
            )
        if self._waiting is not None and self._waiting != key:
            return False
        if (
            self.pages() + pages > self.capacity
            or (swa_pages > 0 and self.swa_pages() + swa_pages > self.swa_capacity)
            or (
                state_pages > 0
                and self.state_pages() + state_pages > self.state_capacity
            )
        ):
            self._waiting = key
            return False
        self._waiting = None
        self._in_flight[key] = (pages, swa_pages, state_pages)
        return True

    def release(self, key: Any) -> None:
        self._in_flight.pop(key, None)

    def forget(self, key: Any) -> None:
        """Gives up ``key``'s place in line: its read will not ask again."""
        if self._waiting == key:
            self._waiting = None

    def rooms(self) -> List[Any]:
        return list(self._in_flight)


def _exceeds(pages: int, capacity: Optional[int]) -> bool:
    return pages > 0 and (capacity is None or pages > capacity)


def slot_host_bytes(
    kv_caches: List[Any], max_blocks: int, layers: Optional[int] = None
) -> int:
    """Host staging bytes one transfer slot reserves.

    Raiden allocates ``max_blocks`` pages of host memory for each of ``layers``
    host buffers, every layer by default, so a slot costs the widest layer's
    page times the buffer count times the per-slot block budget.
    """
    page_bytes = 0
    for buf in kv_caches:
        layer_bytes = buf.element_size()
        for dim in buf.shape[1:]:
            layer_bytes *= int(dim)
        page_bytes = max(page_bytes, layer_bytes)
    return page_bytes * (len(kv_caches) if layers is None else layers) * max_blocks


# SWA and state slots per full-attention slot, for dynamic staging only (fixed
# slots use one). Shorter transfers share full-attention pages, but each fills
# a whole window. Picked by hand, not measured. Ideally about the typical
# prefill length over the sliding window, so SWA slots don't cap concurrent
# transfers.
WINDOW_SLOTS_PER_FULL_SLOT = 8


def split_staging_layers(layers: Sequence[int], total: int) -> List[int]:
    """Host layer buffers per engine, ``total`` split roughly by layer count. 0
    keeps every layer.

    For example, Muse Spark has 16 full-attention and 46 SWA layers, so
    ``split_staging_layers([16, 46], 8)`` returns ``[2, 6]``, and a total of 0
    returns ``[16, 46]``. One buffer runs each layer's copy and send back to
    back, two overlap them.
    """
    if total <= 0:
        return list(layers)
    return [min(n, max(1, round(total * n / sum(layers)))) for n in layers]


def ranks_per_node(
    attn_tp_size: int,
    attn_dp_size: int,
    system_dp_size: int,
    nnodes: int,
    attn_cp_size: int = 1,
    pp_size: int = 1,
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
        * max(1, pp_size)
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
    pp_rank: int = 0,
    pp_size: int = 1,
) -> int:
    """Distinct rank of one worker among every worker on its node.

    Counts the attention-TP ranks of every context-parallel, attention-DP,
    and system-DP replica, so no two workers on a host share the ports,
    raiden engine id, or uuid namespace derived from it.
    """
    replica = system_dp_rank * max(1, attn_dp_size) + attn_dp_rank
    replica = replica * max(1, pp_size) + pp_rank
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
