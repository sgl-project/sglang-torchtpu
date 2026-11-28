"""Decode side of the push: land the branches prefills offload to this decode.

The decode does not cache the requests it serves; its radix tree holds only
the sessions prefills push to it. Every ``_CADENCE`` scheduler iterations its
ranks agree on the offers all of them received, allocate host pages and start
host reads. Once every rank's read settled, the session is inserted as a
host-only path and the prefill is told whether it was stored (VICTIM_ACK).
"""

from __future__ import annotations

import logging
from array import array
from typing import Any, Dict, List, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    MAX_INFLIGHT,
    log_offload_stats,
    victim_transfer_id,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    host_backed_prefix,
    host_page_ids,
    insert_host_prefix,
    lock_host_path,
    unlock_host_path,
    window_host_pages,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    alloc_host_pages_agreed,
    push_manifest,
)

logger = logging.getLogger(__name__)

# Arbitrary: decode iterations between agreement rounds; a push not every
# rank received in 8 rounds is dropped.
_CADENCE = 16
_MAX_UNAGREED_ROUNDS = 8


def _slice_full_metadata(push: Any, attr: str, first_page: int) -> List[bytes]:
    rows = list(getattr(push, attr, ()) or ())
    if not rows:
        return []
    if len(rows) != len(push.blocks):
        raise ValueError("host page metadata count does not match the transfer")
    return rows[first_page:]


class _DramLanding(msgspec.Struct):
    push: Any
    indices: Any
    have_indices: Any = None
    first_page: int = 0
    locks: List[Any] = msgspec.field(default_factory=list)
    # Host pages reserved for the window of a hybrid model.
    swa_indices: Any = None
    # The native reads that started, into ``indices`` and ``swa_indices``.
    read_ids: List[str] = msgspec.field(default_factory=list)
    # A rank's read did not start: the pages wait for the reads that did,
    # then are freed.
    failed: bool = False


class DecodeDramVictimReceiver:
    """Lands the sessions prefills push into this decode's host pool."""

    def __init__(self, ctx: DramOffloadWorkerContext) -> None:
        self._channel = ctx.channel
        self._cache = ctx.cache
        self._page_size = ctx.page_size
        self._host_pool = ctx.host_pool
        self._swa_host_pool = ctx.swa_host_pool
        self._window_tokens = ctx.window_tokens
        self._lockstep = ctx.lockstep
        self._iterations = 0
        self._rounds = 0
        # Round a received push was first seen in, by (prefill, seq).
        self._first_seen: Dict[Tuple[str, int], int] = {}
        self._landings: Dict[Tuple[str, int], _DramLanding] = {}
        self.stats = dict.fromkeys(
            (
                "received",
                "granted",
                "no_room",
                "landed",
                "landed_tokens",
                "failed",
                "dropped",
            ),
            0,
        )
        self._last_log = [0.0]

    def has_host_landings(self) -> bool:
        return bool(self._landings)

    def release_all(self) -> None:
        """Before a cache reset, which core issues only on an idle engine: no
        landing may be in flight."""
        if self._landings:
            raise RuntimeError("victim pushes are landing; the engine is not idle")
        self._channel.clear_host_metadata()

    def advance(self) -> None:
        """Every decode iteration, every rank; acts once per ``_CADENCE``."""
        self._iterations += 1
        if self._iterations % _CADENCE:
            return
        self._rounds += 1
        self._admit_agreed_pushes()
        self.finish_agreed_landings()
        log_offload_stats(self.stats, "landing", self._last_log)

    def _admit_agreed_pushes(self) -> None:
        received = {
            key: push
            for key, push in self._channel.received_pushes().items()
            if key not in self._landings
        }
        for key in received:
            if self._first_seen.setdefault(key, self._rounds) == self._rounds:
                self.stats["received"] += 1
        manifests = {
            (
                *key,
                *push_manifest(
                    push.tokens,
                    push.extra_key,
                    push.cache_salt,
                    len(push.blocks),
                    len(push.swa_blocks),
                ),
            ): key
            for key, push in received.items()
        }
        for manifest in self._lockstep.common_to_all_ranks(list(manifests)):
            key = manifests[manifest]
            push = self._channel.take_push(key)
            self._first_seen.pop(key, None)
            inflight = sum(
                landing.indices is not None or landing.swa_indices is not None
                for landing in self._landings.values()
            )
            if inflight >= MAX_INFLIGHT or not self._reserve_host_pages_and_read(
                key, push
            ):
                self.stats["no_room"] += 1
                self._ack_prefill(push, stored=False)
        for key, first in list(self._first_seen.items()):
            if self._rounds - first > _MAX_UNAGREED_ROUNDS:
                self._first_seen.pop(key)
                if self._channel.take_push(key) is not None:
                    self.stats["dropped"] += 1

    def _ack_prefill(self, push: Any, *, stored: bool) -> None:
        self._channel.send_victim_ack(
            endpoint=push.ack_endpoint, seq=push.seq, stored=stored
        )

    def _reserve_host_pages_and_read(self, key: Tuple[str, int], push: Any) -> bool:
        """Reserve host pages for the push's missing full-KV suffix and, on a
        hybrid model, its window, then start the native reads into them."""
        num_tokens = len(push.tokens)
        if num_tokens <= 0 or num_tokens % self._page_size != 0:
            return False
        if push.blocks and len(push.blocks) * self._page_size != num_tokens:
            return False
        radix_key = RadixKey(
            array("q", push.tokens),
            extra_key=push.extra_key,
            cache_salt=push.cache_salt,
        )
        self._cache.match_prefix(MatchPrefixParams(key=radix_key))
        prefix_node, have_tokens, have_chunks = host_backed_prefix(
            self._cache, radix_key, num_tokens, self._page_size
        )
        raw_swa_tokens = len(push.swa_blocks) * self._page_size
        local_window_held = int(
            have_tokens >= num_tokens
            and (
                not raw_swa_tokens
                or window_host_pages(
                    self._cache,
                    self._swa_host_pool,
                    prefix_node,
                    tokens=num_tokens,
                    window_tokens=self._window_tokens,
                )
                is not None
            )
        )
        agreed_have_tokens, agreed_window_held = self._lockstep.reduce(
            [[have_tokens, local_window_held]], dist.ReduceOp.MIN
        )[0]
        while (
            prefix_node is not self._cache.tree_core.root_node
            and have_tokens - len(prefix_node.key) >= agreed_have_tokens
        ):
            have_tokens -= len(prefix_node.key)
            prefix_node = prefix_node.parent
        have_indices = (
            torch.cat(have_chunks)[:agreed_have_tokens]
            if have_chunks and agreed_have_tokens
            else torch.empty(0, dtype=torch.int64)
        )
        first_page = agreed_have_tokens // self._page_size
        missing_tokens = num_tokens - agreed_have_tokens
        swa_tokens = (
            0 if (missing_tokens == 0 and agreed_window_held) else raw_swa_tokens
        )
        locks = lock_host_path(self._cache, prefix_node)
        if missing_tokens or swa_tokens:
            self._make_host_room(missing_tokens, swa_tokens)
        allocated = alloc_host_pages_agreed(
            lockstep=self._lockstep,
            pools=[
                (self._host_pool, missing_tokens),
                (self._swa_host_pool, swa_tokens),
            ],
        )
        if allocated is None:
            unlock_host_path(self._cache, locks)
            return False
        indices, swa_indices = allocated
        landing = _DramLanding(
            push=push,
            indices=indices,
            have_indices=have_indices,
            first_page=first_page,
            locks=locks,
            swa_indices=swa_indices,
        )
        need_full = bool(push.blocks) and indices is not None
        need_swa = bool(push.swa_blocks) and swa_indices is not None
        started = (not need_full or self._start_full_read(landing)) and (
            not need_swa or self._start_window_read(landing)
        )
        # A rank whose read started cannot release its live destination, so
        # all ranks keep their allocations until the started reads settle.
        landing.failed = not self._lockstep.all(started)
        self._landings[key] = landing
        self.stats["granted"] += 1
        return True

    def _make_host_room(self, num_tokens: int, swa_tokens: int) -> None:
        """Evict until every rank has room for the push in each host pool. The
        manifest agreement made the push's tokens and blocks the same on every
        rank; collective maxima find the room any rank is short of."""
        pool, swa_pool = self._host_pool, self._swa_host_pool
        short, swa_short = self._lockstep.reduce(
            [
                [
                    max(0, num_tokens - pool.available_size()),
                    max(0, swa_tokens - swa_pool.available_size()) if swa_tokens else 0,
                ]
            ],
            dist.ReduceOp.MAX,
        )[0]
        if short:
            self._cache.evict_host(short, ComponentType.FULL)
            if swa_short:
                # Removing a full host leaf also frees its SWA pages. Recheck
                # every rank before evicting additional windows.
                swa_short = self._lockstep.reduce(
                    [[max(0, swa_tokens - swa_pool.available_size())]],
                    dist.ReduceOp.MAX,
                )[0][0]
        if swa_short:
            self._cache.evict_host(swa_short, ComponentType.SWA)

    def _start_full_read(self, landing: _DramLanding) -> bool:
        """Read the pushed full suffix pages into the landing's full pages."""
        push = landing.push
        read_id = victim_transfer_id(push.prefill, push.seq)
        started = self._channel.start_host_read(
            read_id,
            push.uuid,
            push.address,
            list(push.blocks[landing.first_page :]),
            host_page_ids(self._host_pool, landing.indices),
        )
        if started:
            landing.read_ids.append(read_id)
        return started

    def _start_window_read(self, landing: _DramLanding) -> bool:
        """Read the pushed window pages into the landing's SWA pages; the
        prefill registered them under the push's UUID on its SWA pool."""
        push = landing.push
        read_id = victim_transfer_id(push.prefill, push.seq, swa=True)
        started = self._channel.start_host_read(
            read_id,
            push.uuid,
            push.swa_address,
            list(push.swa_blocks),
            host_page_ids(self._swa_host_pool, landing.swa_indices),
            swa=True,
        )
        if started:
            landing.read_ids.append(read_id)
        return started

    def finish_agreed_landings(self) -> None:
        """Insert, or free, the landings whose read settled on every rank, and
        tell the prefill; a paused engine's tick calls this alone."""
        if not self._landings:
            return
        self._channel.drain_host_reads()
        keys = sorted(self._landings)
        rows = []
        for key in keys:
            landing = self._landings[key]
            received = self._channel.poll_host_reads(landing.read_ids)
            rows.append(
                [
                    int(received is not None),
                    int(received is True and not landing.failed),
                ]
            )
        agreed = self._lockstep.reduce(rows, dist.ReduceOp.MIN)
        for key, (settled, received) in zip(keys, agreed):
            if not settled:
                continue
            landing = self._landings.pop(key)
            self._channel.forget_host_reads(landing.read_ids)
            if received:
                # KV has settled; metadata is committed before the tree makes
                # these pages available to a subsequent prefix offer.
                restored = False
                try:
                    if landing.indices is not None:
                        self._channel.restore_host_metadata(
                            host_page_ids(self._host_pool, landing.indices),
                            _slice_full_metadata(
                                landing.push, "page_metadata", landing.first_page
                            ),
                        )
                    elif not landing.push.blocks and landing.push.page_metadata:
                        self._channel.restore_host_metadata(
                            [], landing.push.page_metadata
                        )
                    if landing.swa_indices is not None:
                        self._channel.restore_host_metadata(
                            host_page_ids(self._swa_host_pool, landing.swa_indices),
                            landing.push.swa_page_metadata,
                            swa=True,
                        )
                    elif not landing.push.swa_blocks and landing.push.swa_page_metadata:
                        self._channel.restore_host_metadata(
                            [], landing.push.swa_page_metadata, swa=True
                        )
                    restored = True
                except Exception:  # noqa: BLE001 - decline on every rank
                    logger.exception("Cannot restore victim push host metadata")
                received = self._lockstep.all(restored)
            if received:
                stored = self._insert_host_only_branch(landing)
            else:
                self._free_landing(landing)
                self.stats["failed"] += 1
                stored = False
            self._ack_prefill(landing.push, stored=stored)

    def _free_landing(self, landing: _DramLanding) -> None:
        unlock_host_path(self._cache, landing.locks)
        landing.locks = []
        if landing.indices is not None:
            self._host_pool.free(landing.indices)
        if landing.swa_indices is not None:
            self._swa_host_pool.free(landing.swa_indices)

    def _insert_host_only_branch(self, landing: _DramLanding) -> bool:
        push = landing.push
        reused_tokens = 0 if landing.have_indices is None else len(landing.have_indices)
        if reused_tokens and landing.indices is not None:
            full_indices = torch.cat([landing.have_indices, landing.indices])
        elif reused_tokens:
            full_indices = landing.have_indices
        else:
            full_indices = landing.indices
        try:
            inserted, prefix_len = insert_host_prefix(
                self._cache,
                RadixKey(
                    array("q", push.tokens),
                    extra_key=push.extra_key,
                    cache_salt=push.cache_salt,
                ),
                full_indices,
                self._page_size,
                swa_indices=landing.swa_indices,
                reused_tokens=reused_tokens,
            )
        finally:
            unlock_host_path(self._cache, landing.locks)
            landing.locks = []
        if inserted is None:
            self.stats["dropped"] += 1
            return False
        self.stats["landed"] += 1
        self.stats["landed_tokens"] += len(push.tokens) - max(prefix_len, reused_tokens)
        return True
