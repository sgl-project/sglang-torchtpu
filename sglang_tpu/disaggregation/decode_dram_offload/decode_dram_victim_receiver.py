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
from typing import Any, Dict, Optional, Tuple

import msgspec
import torch.distributed as dist

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    DRAIN_TIMEOUT_S,
    MAX_INFLIGHT,
    log_offload_stats,
    victim_transfer_id,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    insert_host_prefix,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    push_manifest,
)

logger = logging.getLogger(__name__)

# Arbitrary: decode iterations between agreement rounds; a push not every
# rank received in 8 rounds is dropped.
_CADENCE = 16
_MAX_UNAGREED_ROUNDS = 8


class _DramLanding(msgspec.Struct):
    push: Any
    indices: Any
    # The native read into ``indices``; None where it could not start.
    read_id: Optional[str] = None
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

    def drain_host_landings(self) -> None:
        """Settle every granted landing before a cache reset: a prefill may
        still be pushing into the pages."""
        self._lockstep.drain_until(
            done=lambda: not self._landings,
            step=self._finish_agreed_landings,
            timeout_s=DRAIN_TIMEOUT_S,
            what="victim pushes are landing",
        )

    def advance(self) -> None:
        """Every decode iteration, every rank; acts once per ``_CADENCE``."""
        self._iterations += 1
        if self._iterations % _CADENCE:
            return
        self._rounds += 1
        self._admit_agreed_pushes()
        self._finish_agreed_landings()
        log_offload_stats(self.stats, "decode", self._last_log)

    def finish_settled_host_landings(self) -> None:
        """A tick of a paused engine: finish the landings in flight, admit
        nothing new."""
        self._finish_agreed_landings()

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
                    push.tokens, push.extra_key, push.cache_salt, len(push.blocks)
                ),
            ): key
            for key, push in received.items()
        }
        for manifest in self._lockstep.common_to_all_ranks(list(manifests)):
            key = manifests[manifest]
            push = self._channel.take_push(key)
            self._first_seen.pop(key, None)
            if len(
                self._landings
            ) >= MAX_INFLIGHT or not self._reserve_host_pages_and_read(key, push):
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
        num_tokens = len(push.tokens)
        pool = self._host_pool
        valid = (
            num_tokens > 0
            and num_tokens == len(push.blocks) * self._page_size
            and push.uuid > 0
            and push.address
        )
        # One collective: every rank finds the push valid, and the most host
        # room any rank is short of.
        valid, short = self._lockstep.reduce(
            [[int(bool(valid)), -max(0, num_tokens - pool.available_size())]],
            dist.ReduceOp.MIN,
        )[0]
        if not valid:
            return False
        if short:
            self._cache.evict_host(-short, ComponentType.FULL)
        indices = pool.alloc(num_tokens)
        if not self._lockstep.all(indices is not None):
            if indices is not None:
                pool.free(indices)
            return False
        landing = _DramLanding(push=push, indices=indices)
        read_id = victim_transfer_id(push.prefill, push.seq)
        if self._channel.start_host_read(
            read_id,
            push.uuid,
            push.address,
            list(push.blocks),
            [int(p) for p in self._host_pool._to_page_ids(indices)],
        ):
            landing.read_id = read_id
        # A rank whose read started cannot release its live destination, so
        # all ranks keep their allocations until the started reads settle.
        landing.failed = not self._lockstep.all(landing.read_id is not None)
        self._landings[key] = landing
        self.stats["granted"] += 1
        return True

    def _finish_agreed_landings(self) -> None:
        """Insert, or free, the landings whose read settled on every rank, and
        tell the prefill."""
        if not self._landings:
            return
        self._channel.drain_host_reads()
        keys = sorted(self._landings)
        rows = []
        for key in keys:
            landing = self._landings[key]
            received = (
                False
                if landing.read_id is None
                else self._channel.poll_host_read(landing.read_id)
            )
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
            if landing.read_id is not None:
                self._channel.forget_host_read(landing.read_id)
            if received:
                stored = self._insert_host_only_branch(landing)
            else:
                self._host_pool.free(landing.indices)
                self.stats["failed"] += 1
                stored = False
            self._ack_prefill(landing.push, stored=stored)

    def _insert_host_only_branch(self, landing: _DramLanding) -> bool:
        push = landing.push
        result = insert_host_prefix(
            self._cache,
            RadixKey(
                array("q", push.tokens),
                extra_key=push.extra_key,
                cache_salt=push.cache_salt,
            ),
            landing.indices,
            self._page_size,
        )
        if result.host_insert_dropped or result.inserted_host_node is None:
            self.stats["dropped"] += 1
            return False
        self.stats["landed"] += 1
        self.stats["landed_tokens"] += len(push.tokens) - result.prefix_len
        return True
