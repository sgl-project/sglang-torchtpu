# SPDX-License-Identifier: Apache-2.0
"""DP rank choice by pending uncached prompt tokens (opt-in, TPU plugin).

Installed on the DataParallelController by sglang_tpu/hooks/dp_prompt_backlog.py.

The load-based DP methods count requests, or raw prompt tokens, per rank. A
rank prefilling a 300k-token prompt counts as one request, and a routed
(affinity-pinned) request goes to its rank however much prefill is queued
there. Under DP attention every rank steps in lockstep, so prefill concentrated
on one rank costs steps the other ranks spend waiting.

Backlog per rank: ``num_waiting_uncached_tokens`` from each scheduler's load
snapshot (refreshed by the controller at most every 20 ms), plus this
controller's own estimate of the uncached tokens of the requests it has sent
since.

Unrouted requests (``SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN = m > 0``): the controller
takes the configured method's choice (the stock rank) unless that rank's backlog
exceeds the least backlogged active rank's by more than ``m`` times the prompt's
length; then the request goes to the least backlogged rank (ties: fewer
requests, then lower index). Port of vllm-torchtpu's VLLM_TPU_DP_PROMPT_BACKLOG.

Routed requests (``routed_dp_rank``, e.g. the ``X-Data-Parallel-Rank`` header a
session-affinity front end sets) keep their rank unless
``SGLANG_TPU_DP_PROMPT_BACKLOG_ROUTED=1`` makes it an affinity hint. Moving a
conversation's turn away from the rank caching it makes another rank recompute
the whole prompt, and under DP lockstep every rank pays for those steps (a
screen with continuations movable at m = 1 lost 4% throughput and added 50% more
prefill). So the hint is only overridden without losing cache:

* the controller indexes the routed prompts each rank received (chained block
  hashes, an LRU per rank sized like its KV pool);
* a prompt that some rank holds at least ``CONTINUATION_FRACTION`` of, or that
  extends a whole prompt routed there, is a continuation: it goes to the hinted
  rank if that is a holder, else to the holder with most of it (an earlier turn
  was moved), so a moved conversation follows its cache;
* short prompts (under ``MIN_MOVED_PROMPT_TOKENS``) and requests in another
  cache namespace (LoRA, ``extra_key``, ``cache_salt``; the index hashes token
  ids only) keep
  their hint;
* any other prompt (a new conversation, or one whose cache is gone) costs a full
  prefill wherever it runs, so it goes to the least backlogged rank when the
  hinted rank's backlog exceeds it by more than ``m`` times the prompt.

``SGLANG_TPU_DP_PROMPT_BACKLOG_CONTINUATION_MARGIN = c > 0`` (default off) also
moves continuations when the backlog gap exceeds ``c`` times the prompt.

Known limits: the index records dispatches, not cache contents. An eviction on a
holder is not seen, so such a prompt is pinned like a continuation (the
conservative error). Subagents and branches sharing at least half of a parent's
prompt stay on the parent's rank, as the session-affinity proxy already places
them.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from sglang.srt.managers.dp_prefix_router import PrefixAffinityIndex

logger = logging.getLogger(__name__)

# Coarse blocks keep the per-request hashing cheap (~100 hashes per 100k-token
# prompt); a conversation's next turn extends its previous prompt by far more.
BLOCK_TOKENS = 1024
# A rank holding at least this fraction of a prompt keeps it (a system prompt
# shared by every session stays far below it in agentic workloads).
CONTINUATION_FRACTION = 0.5
DEFAULT_CAPACITY_BLOCKS = 1 << 16
# Routed prompts shorter than this keep their hint: the index can't see
# sub-block prompts, and moving a short prompt barely changes any backlog.
MIN_MOVED_PROMPT_TOKENS = 4 * BLOCK_TOKENS


def backlog_routing_enabled(
    margin: float,
    load_balance_method: str,
    disaggregation_mode: Optional[str],
    elastic: bool = False,
) -> bool:
    """Whether the DP controller routes by prompt backlog (and so reads load
    snapshots on every dispatch). Off under PD disaggregation and
    follow_bootstrap_room (bootstrap rooms pin ranks) and with elastic DP (load
    snapshots cover the launch ranks only)."""
    return (
        margin > 0
        and disaggregation_mode not in ("prefill", "decode")
        and load_balance_method.lower() != "follow_bootstrap_room"
        and not elastic
    )


@dataclass
class BacklogStats:
    requests: int = 0
    moved: int = 0
    moved_new: int = 0
    moved_continuations: int = 0
    followed: int = 0
    tokens_skipped: int = 0

    def line(self) -> str:
        return (
            f"dp prompt backlog: {self.requests} requests, {self.moved} moved to a "
            f"less backlogged rank ({self.moved_new} new prompts of routed "
            f"requests, {self.moved_continuations} continuations), "
            f"{self.followed} continuations followed a moved conversation, "
            f"{self.tokens_skipped} pending prompt tokens skipped"
        )


class PromptBacklogRouter:
    """Overrides a stock DP rank choice when its uncached prompt backlog is
    too far above the least backlogged rank's (see the module docstring)."""

    def __init__(
        self,
        num_ranks: int,
        margin: float,
        override_routed: bool,
        continuation_margin: float = 0.0,
        block_tokens: int = BLOCK_TOKENS,
        continuation_fraction: float = CONTINUATION_FRACTION,
        min_moved_tokens: int = MIN_MOVED_PROMPT_TOKENS,
        capacity_blocks: int = DEFAULT_CAPACITY_BLOCKS,
        log_interval: int = 100,
    ):
        if margin <= 0:
            raise ValueError(f"margin must be positive, got {margin}")
        self.num_ranks = num_ranks
        self.margin = margin
        self.override_routed = override_routed
        self.continuation_margin = continuation_margin
        self.continuation_fraction = continuation_fraction
        self.min_moved_tokens = min_moved_tokens
        self.log_interval = max(1, log_interval)
        self.backlog = [0.0] * num_ranks
        self.requests = [0] * num_ranks
        self.hit_rate = [0.0] * num_ranks
        self._last_timestamp = [None] * num_ranks
        self.index = PrefixAffinityIndex(num_ranks, block_tokens, capacity_blocks)
        # Where each routed prompt ended exactly: (rank, its last full block key,
        # 0 if none) -> {(tail length, hash of the tokens after that block)}.
        # The block index drops a prompt's partial last block, so "extends a
        # whole earlier prompt" needs the tail too.
        self._ends: OrderedDict[tuple[int, int], set[tuple[int, int]]] = OrderedDict()
        self._ends_capacity = capacity_blocks
        self.stats = BacklogStats()

    def update(self, loads) -> None:
        """Take each rank's latest load snapshot (skipping unchanged ones)."""
        for load in loads:
            r = load.dp_rank
            if r >= self.num_ranks or load.timestamp == self._last_timestamp[r]:
                continue
            self._last_timestamp[r] = load.timestamp
            self.backlog[r] = float(load.num_waiting_uncached_tokens)
            self.requests[r] = load.num_running_reqs + load.num_waiting_reqs
            self.hit_rate[r] = min(max(load.cache_hit_rate, 0.0), 1.0)

    def _least(self, active: Sequence[int]) -> int:
        return min(active, key=lambda r: (self.backlog[r], self.requests[r], r))

    def route(
        self,
        input_ids: Sequence[int],
        stock: int,
        active: Sequence[int],
        routed: bool,
        prompt_tokens: Optional[int] = None,
        pinned: bool = False,
    ) -> int:
        """The rank to send a request to, given the stock choice.

        ``prompt_tokens`` defaults to ``len(input_ids)`` (pass it for requests
        given as embeddings, whose ``input_ids`` are empty). ``pinned``
        requests (stateful sessions, whose history lives on one rank) and
        routed requests without SGLANG_TPU_DP_PROMPT_BACKLOG_ROUTED keep their
        rank; their work is still counted.
        """
        if prompt_tokens is None:
            prompt_tokens = len(input_ids)
        if not active or pinned or (routed and not self.override_routed):
            if 0 <= stock < self.num_ranks:
                self._account(stock, prompt_tokens * (1.0 - self.hit_rate[stock]))
            return stock
        if prompt_tokens <= 0 and stock in active:
            self._account(stock, 0.0)
            return stock
        if routed:
            return self._route_hint(input_ids, stock, active, prompt_tokens)

        least = self._least(active)
        gap = self.backlog[stock] - self.backlog[least]
        moved = stock not in active or gap > self.margin * prompt_tokens
        chosen = least if moved else stock
        self._count(moved, gap, new=False, followed=False, continuation=False)
        self._account(
            chosen,
            prompt_tokens * (1.0 if moved else 1.0 - self.hit_rate[chosen]),
        )
        return chosen

    def _tail(self, input_ids: Sequence[int], blocks: int) -> tuple[int, int]:
        tail = tuple(input_ids[blocks * self.index.block_size :])
        return len(tail), hash(tail)

    def _record(self, rank: int, input_ids: Sequence[int], keys: list[int]) -> None:
        if keys:
            self.index.insert(rank, keys)
        end = (rank, keys[-1] if keys else 0)
        tails = self._ends.pop(end, set())
        tails.add(self._tail(input_ids, len(keys)))
        self._ends[end] = tails
        while len(self._ends) > self._ends_capacity:
            self._ends.popitem(last=False)

    def _extends(self, rank: int, input_ids, keys, held: int) -> bool:
        """Whether this prompt extends, as a whole, an earlier prompt routed to
        `rank` whose full blocks are the `held` matched ones."""
        tails = self._ends.get((rank, keys[held - 1] if held else 0))
        if not tails:
            return False
        start = held * self.index.block_size
        return any(
            start + length <= len(input_ids)
            and hash(tuple(input_ids[start : start + length])) == digest
            for length, digest in tails
        )

    def _holds(self, held_blocks, keys, input_ids, rank: int, n: int) -> bool:
        """Whether `rank` is known to cache this prompt as a continuation: at
        least CONTINUATION_FRACTION of it, or all of an earlier prompt routed
        there that this one extends (a turn that appends more than it had)."""
        held = held_blocks[rank]
        return (
            held > 0 and held * self.index.block_size >= self.continuation_fraction * n
        ) or self._extends(rank, input_ids, keys, held)

    def _route_hint(
        self, input_ids: Sequence[int], hint: int, active: Sequence[int], n: int
    ) -> int:
        keys = self.index.block_keys(input_ids) if input_ids else []
        if n < self.min_moved_tokens and hint in active:
            # Recorded, so the conversation keeps this rank once it grows.
            if input_ids:
                self._record(hint, input_ids, keys)
            self._count(False, 0, new=False, followed=False, continuation=False)
            self._account(hint, n * (1.0 - self.hit_rate[hint]))
            return hint
        held_blocks = self.index.match(keys) if keys else [0] * self.num_ranks
        if hint in active and self._holds(held_blocks, keys, input_ids, hint, n):
            # The hinted rank is a holder: never trade it for an index entry
            # elsewhere that may be stale.
            holder = hint
        else:
            holders = [
                r for r in active if self._holds(held_blocks, keys, input_ids, r, n)
            ]
            holder = (
                max(holders, key=lambda r: (held_blocks[r], -self.backlog[r]))
                if holders
                else None
            )
        continuation = holder is not None
        least = self._least(active)
        if continuation:
            stock = holder
            gap = self.backlog[stock] - self.backlog[least]
            moved = self.continuation_margin > 0 and gap > self.continuation_margin * n
            followed = stock != hint and not moved
            held = held_blocks[holder] * self.index.block_size
        else:
            stock = hint
            gap = self.backlog[stock] - self.backlog[least]
            moved = stock not in active or gap > self.margin * n
            followed = False
            held = 0
        chosen = least if moved else stock
        self._count(
            moved,
            gap,
            new=not continuation,
            followed=followed,
            continuation=continuation,
        )
        if input_ids:
            self._record(chosen, input_ids, keys)
        uncached = n if moved or not continuation else max(0, n - held)
        self._account(chosen, float(uncached))
        return chosen

    def _count(self, moved, gap, new, followed, continuation) -> None:
        s = self.stats
        s.requests += 1
        if moved:
            s.moved += 1
            s.moved_new += int(new)
            s.moved_continuations += int(continuation)
            s.tokens_skipped += int(max(gap, 0))
        elif followed:
            s.followed += 1
        if s.requests % self.log_interval == 0:
            logger.info(s.line())

    def _account(self, rank: int, estimate: float) -> None:
        if 0 <= rank < self.num_ranks:
            self.backlog[rank] += estimate
            self.requests[rank] += 1
