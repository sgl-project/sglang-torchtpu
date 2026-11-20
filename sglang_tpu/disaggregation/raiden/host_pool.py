"""Sizing and admission control for raiden's host staging pool.

A raiden engine reserves ``num_slots * max_blocks`` pages of pinned host memory
at startup and seats each in-flight transfer in as many of those pages as the
transfer needs (in a whole slot when fixed slots are configured). Deriving the
slot count from a memory budget keeps a large context length from turning into
a host allocation that fails at startup; admission keeps a consumer from
issuing pulls the pool cannot seat.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


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
        draft_capacity: Optional[int] = None,
    ):
        self.capacity = max(1, int(capacity))
        self.swa_capacity = None if swa_capacity is None else max(1, int(swa_capacity))
        self.state_capacity = (
            None if state_capacity is None else max(1, int(state_capacity))
        )
        self.draft_capacity = (
            None if draft_capacity is None else max(1, int(draft_capacity))
        )
        self._in_flight: Dict[Any, Tuple[int, int, int, int]] = {}
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
        return sum(pages for pages, _, _, _ in self._in_flight.values())

    def swa_pages(self) -> int:
        return sum(swa_pages for _, swa_pages, _, _ in self._in_flight.values())

    def state_pages(self) -> int:
        return sum(state_pages for _, _, state_pages, _ in self._in_flight.values())

    def draft_pages(self) -> int:
        return sum(draft_pages for _, _, _, draft_pages in self._in_flight.values())

    def admit(
        self,
        key: Any,
        pages: int,
        swa_pages: int = 0,
        state_pages: int = 0,
        draft_pages: int = 0,
    ) -> bool:
        """Reserves ``pages`` of the pool (``swa_pages`` of the sliding-window
        pool, ``state_pages`` of the GDN state pool, plus any draft pool) for
        ``key``. False means try again later."""
        if key in self._in_flight:
            return True
        pages = int(pages)
        swa_pages = int(swa_pages)
        state_pages = int(state_pages)
        draft_pages = int(draft_pages)
        if (
            pages > self.capacity
            or _exceeds(swa_pages, self.swa_capacity)
            or _exceeds(state_pages, self.state_capacity)
            or _exceeds(draft_pages, self.draft_capacity)
        ):
            raise ValueError(
                f"read {key!r} needs {pages} pages, {swa_pages} sliding-window "
                f"pages, {state_pages} state rows and {draft_pages} draft pages, "
                f"more than the pool holds ({self.capacity}, {self.swa_capacity}, "
                f"{self.state_capacity} and {self.draft_capacity})"
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
            or (
                draft_pages > 0
                and self.draft_pages() + draft_pages > self.draft_capacity
            )
        ):
            self._waiting = key
            return False
        self._waiting = None
        self._in_flight[key] = (pages, swa_pages, state_pages, draft_pages)
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


def _buffer_page_bytes(buf: Any) -> int:
    page_bytes = buf.element_size()
    for dim in buf.shape[1:]:
        page_bytes *= int(dim)
    return page_bytes


def has_heterogeneous_page_layouts(kv_caches: Sequence[Any]) -> bool:
    """Whether ``kv_caches`` mix buffers with different per-page shapes or element sizes.

    ``tpu_sync``'s bounded shared-layer staging (``num_staging_layers <
    len(kv_caches)``) uses a per-session ``StagingLayerManager`` while striding
    each shared host buffer by ``host_block_id * layer_block_byte_size(layer)``.
    Two concurrent sessions sharing the same host buffer on layers with
    different page byte sizes (or different XLA-tiled physical sizes) overwrite
    each other even when their ``host_block_id`` ranges are disjoint. Engines
    with heterogeneous per-page layouts must therefore keep full per-layer host
    mirrors.
    """
    if len(kv_caches) <= 1:
        return False
    layouts = set()
    for buf in kv_caches:
        shape = getattr(buf, "shape", None)
        element_size = getattr(buf, "element_size", None)
        if shape is None or not callable(element_size):
            return False
        layouts.add((tuple(int(dim) for dim in shape[1:]), int(element_size())))
    return len(layouts) > 1


def slot_host_bytes(
    kv_caches: List[Any], max_blocks: int, layers: Optional[int] = None
) -> int:
    """Host staging bytes one transfer slot reserves.

    When Raiden allocates fewer host buffers than device layers
    (``layers < len(kv_caches)``), each shared buffer is sized for the widest
    layer's page. When every layer has its own host buffer (``layers is None``
    or ``layers >= len(kv_caches)``), Raiden sizes each host buffer to that
    layer's own page size.
    """
    page_sizes = [_buffer_page_bytes(buf) for buf in kv_caches]
    if not page_sizes:
        return 0
    if layers is None or layers >= len(kv_caches):
        return sum(page_sizes) * max_blocks
    return max(page_sizes) * layers * max_blocks


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
    """Slots that fit ``budget_bytes``, at most ``ceiling`` and at least one.

    One slot that does not fit is still allocated, over the budget, so a
    max-length transfer can always run.
    """
    if slot_bytes > budget_bytes:
        logger.warning(
            "raiden host staging: one transfer slot needs %.2f GiB, over this "
            "engine's %.2f GiB share of SGLANG_RAIDEN_HOST_POOL_GB; allocating "
            "one slot anyway",
            slot_bytes / 1024**3,
            budget_bytes / 1024**3,
        )
    return max(1, min(ceiling, int(budget_bytes // max(1, slot_bytes))))
