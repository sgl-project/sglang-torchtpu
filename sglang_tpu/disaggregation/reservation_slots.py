# SPDX-License-Identifier: Apache-2.0
"""Count-only credits for selected requests that have not started prefill.

READY awaiting metadata still owns a credit. Silence never expires a credit:
normal request/transport cleanup resolves ownership. BUSY is agreed across all
attention ranks before a retry can reuse the credit.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_STATS_INTERVAL_S = 60.0
_MIN_BUSY_COOLDOWN_S = 0.010
_MAX_BUSY_COOLDOWN_S = 0.100


def effective_slot_limit(configured: int, capacity: int) -> int:
    if configured < 0:
        raise ValueError(
            "prefill reservation slots must be nonnegative (0 is automatic)"
        )
    return min(configured or capacity, max(0, capacity))


@dataclass
class _Outstanding:
    requested_at: float
    response_recorded: bool = False


class SlotReservations:
    def __init__(self, *, limit: int, rank_groups: Sequence, log_stats=False):
        if limit < 1:
            raise ValueError("prefill reservation capacity must be positive")
        self.limit = limit
        self.rank_groups = rank_groups
        self._next_id = 0
        self._ids = {}
        self._owners = {}
        self._outstanding = {}
        self._ready = set()
        self._retry_at = {}
        self._rtt = OrderedDict()
        self._candidates = []
        self._capacity = limit
        self._now = 0.0
        self._log_stats = log_stats
        self._asked = self._busy = self._max_ask_batch = 0
        self._rtt_count = 0
        self._rtt_sum = self._cooldown_sum = 0.0
        self._last_log_at = None
        self._last_logged = None
        self._pass_started_at = None
        self._passes = 0
        self._pass_total_s = self._max_pass_s = 0.0
        self._max_queue_depth = 0

    def is_idle(self):
        # Include tracked requests that still need cancellation/prune cleanup.
        return not self._owners and not self._outstanding

    def track(self, req):
        if self._owners.get(req.rid) is not req:
            self._owners[req.rid] = req
            self._ids[req.rid] = self._next_id
            self._next_id += 1
            self._outstanding.pop(req.rid, None)
            self._retry_at.pop(req.rid, None)
            self._ready.discard(req.rid)
        return self._ids[req.rid]

    def canonical_order(self, reqs):
        for req in reqs:
            self.track(req)
        return sorted(reqs, key=lambda req: self._ids[req.rid])

    def mark_ready(self, req):
        # Core's all-reduced transport poll calls this before metadata allocation.
        if self._owners.get(req.rid) is req:
            self._ready.add(req.rid)

    def _broadcast(self, message):
        for group in self.rank_groups:
            if dist.get_world_size(group) > 1:
                dist.broadcast(message, src=dist.get_global_rank(group, 0), group=group)

    def agree_order(self, reqs):
        if not reqs:
            return []
        by_id = {self.track(req): req for req in reqs}
        message = torch.tensor(list(by_id), dtype=torch.int64)
        self._broadcast(message)
        return [by_id[index] for index in message.tolist()]

    def _prune(self, reqs):
        live = {req.rid: req for req in reqs}
        self._ready.intersection_update(live)
        self._ids = {rid: value for rid, value in self._ids.items() if rid in live}
        self._owners = {rid: req for rid, req in self._owners.items() if rid in live}
        self._outstanding = {
            rid: value for rid, value in self._outstanding.items() if rid in live
        }
        self._retry_at = {
            rid: value for rid, value in self._retry_at.items() if rid in live
        }
        return live

    def _record_rtt(self, req, state):
        elapsed = max(0.0, self._now - state.requested_at)
        endpoint = req.disagg_kv_sender.decode_endpoint()
        previous = self._rtt.pop(endpoint, elapsed)
        self._rtt[endpoint] = 0.75 * previous + 0.25 * elapsed
        if len(self._rtt) > self.limit:
            self._rtt.popitem(last=False)
        self._rtt_count += 1
        self._rtt_sum += elapsed
        state.response_recorded = True
        return min(
            _MAX_BUSY_COOLDOWN_S, max(_MIN_BUSY_COOLDOWN_S, 2 * self._rtt[endpoint])
        )

    def begin(self, reqs, *, capacity=None, now=None):
        self._pass_started_at = time.perf_counter()
        self._max_queue_depth = max(self._max_queue_depth, len(reqs))
        self._now = time.monotonic() if now is None else now
        for req in reqs:
            self.track(req)
        live = self._prune(reqs)
        self._capacity = min(
            self.limit, max(0, self.limit if capacity is None else capacity)
        )
        ordered = sorted(self._outstanding, key=self._ids.__getitem__)
        busy = torch.zeros(len(ordered), dtype=torch.int64)
        for index, rid in enumerate(ordered):
            req = live[rid]
            if rid not in self._ready and req.pending_bootstrap:
                busy[index] = int(req.disagg_kv_sender.reservation_busy())
        # Raiden latches BUSY until the next ask. Waiting for every rank's latch
        # avoids clearing/retrying a room while another rank still owns its ask.
        if ordered:
            for group in self.rank_groups:
                if dist.get_world_size(group) > 1:
                    dist.all_reduce(busy, op=dist.ReduceOp.MIN, group=group)
        for rid, answered_busy in zip(ordered, busy.tolist()):
            state = self._outstanding[rid]
            req = live[rid]
            if rid in self._ready or not req.pending_bootstrap:
                if not state.response_recorded:
                    self._record_rtt(req, state)
            elif answered_busy:
                cooldown = self._record_rtt(req, state)
                del self._outstanding[rid]
                self._retry_at[rid] = self._now + cooldown
                self._busy += 1
                self._cooldown_sum += cooldown
        self._candidates = []

    def consider(self, req):
        if req.pending_bootstrap and req.rid not in self._outstanding:
            self._candidates.append(req)

    def finish(self, unstarted):
        live = self._prune(unstarted)
        eligible = [
            req
            for req in self._candidates
            if live.get(req.rid) is req
            and self._retry_at.get(req.rid, 0) <= self._now
            and req.disagg_kv_sender.decode_endpoint() is not None
        ]
        self._candidates = []
        per_decode = Counter(
            live[rid].disagg_kv_sender.decode_endpoint() for rid in self._outstanding
        )
        destinations = set(per_decode) | {
            req.disagg_kv_sender.decode_endpoint() for req in eligible
        }
        share = math.ceil(self._capacity / len(destinations)) if destinations else 0
        selected = []
        selected_ids = set()
        for req in eligible:
            if len(self._outstanding) + len(selected) >= self._capacity:
                break
            endpoint = req.disagg_kv_sender.decode_endpoint()
            if req.rid in selected_ids or per_decode[endpoint] >= share:
                continue
            selected.append(req)
            selected_ids.add(req.rid)
            per_decode[endpoint] += 1

        agreed = []
        if unstarted:
            # Keep a fixed ask shape: capacity/endpoint readiness can differ
            # locally, while the root's selected IDs must reach every rank.
            message = torch.full((self.limit,), -1, dtype=torch.int64)
            for index, req in enumerate(selected):
                message[index] = self.track(req)
            self._broadcast(message)
            by_id = {self.track(req): req for req in unstarted}
            for index in message.tolist():
                if index < 0:
                    continue
                req = by_id[index]
                self._outstanding[req.rid] = _Outstanding(self._now)
                self._retry_at.pop(req.rid, None)
                req.disagg_kv_sender.request_decode_reservation()
                agreed.append(req)
        self._asked += len(agreed)
        self._max_ask_batch = max(self._max_ask_batch, len(agreed))
        elapsed = time.perf_counter() - self._pass_started_at
        self._passes += 1
        self._pass_total_s += elapsed
        self._max_pass_s = max(self._max_pass_s, elapsed)
        self._maybe_log(live)
        return agreed

    def _maybe_log(self, live):
        if not self._log_stats:
            return
        if self._last_log_at is None:
            self._last_log_at = self._now
        ready = sum(
            rid in self._ready or not live[rid].pending_bootstrap
            for rid in self._outstanding
        )
        counts = (self._asked, self._busy, len(self._outstanding), ready, len(live))
        drained = not live and self._asked > 0 and counts != self._last_logged
        if (not live and counts == self._last_logged) or (
            self._now - self._last_log_at < _STATS_INTERVAL_S and not drained
        ):
            return
        self._last_log_at, self._last_logged = self._now, counts
        logger.info(
            "prefill_driven slots: limit=%d effective=%d unstarted=%d pending=%d ready=%d "
            "asked_total=%d busy_total=%d max_ask_batch=%d rtt_ms=%.2f cooldown_ms=%.2f "
            "passes_total=%d pass_ms=%.2f max_pass_ms=%.2f max_queue_depth=%d",
            self.limit,
            self._capacity,
            len(live),
            len(self._outstanding),
            ready,
            self._asked,
            self._busy,
            self._max_ask_batch,
            1000 * self._rtt_sum / max(1, self._rtt_count),
            1000 * self._cooldown_sum / max(1, self._busy),
            self._passes,
            1000 * self._pass_total_s / max(1, self._passes),
            1000 * self._max_pass_s,
            self._max_queue_depth,
        )
