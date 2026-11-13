# SPDX-License-Identifier: Apache-2.0
"""Prefix tails: prefix-cache hits inside a logical block.

sglang's radix cache matches whole KV pages, and a K3 program's page is a
logical block (16384 tokens in fp8). An AgentX turn extends the session's
earlier prompt, so a page-aligned match leaves the earlier prompt's last
partial page to compute again: about half a page a turn. On the AgentX corpus
that is 7.5% of the input against 2.6% for an exact match (vLLM's prefix
cache matches within a block: a copy-on-write copy of the partial block, and
the KDA state checkpointed every ``tokens_per_block`` tokens).

A tail is the rest of a prompt beyond its last full page, cut to the
programs' hit alignment (``tokens_per_block``: 256 tokens, 512 in fp8): its
partial page's KV (copied into a tail KV block) and the KDA state at its end
(a tail state slot). Both live in a region of the unified pools sglang does
not allocate (``UnifiedPoolPlan.tail_slots``). A later prompt whose radix match
ends where a tail begins and continues with the tail's tokens starts
computing at the tail's end: its new page takes a copy of the tail's block,
and the program reads the KDA state there from the tail's slot.

:class:`TailIndex` is the host index (no device work); the prefill worker
does the copies and points the programs' tables at the tail blocks.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib

import numpy as np


def _digest(tokens: np.ndarray) -> bytes:
    return hashlib.blake2b(
        np.ascontiguousarray(tokens, dtype=np.int32).tobytes(), digest_size=16
    ).digest()


def _page_digests(tokens: np.ndarray, page: int, upto: int) -> dict[int, bytes]:
    """``_digest(tokens[:f])`` for every page boundary ``f <= upto``, in one
    pass over the tokens."""
    data = np.ascontiguousarray(tokens[:upto], dtype=np.int32)
    hasher = hashlib.blake2b(digest_size=16)
    digests = {0: hasher.copy().digest()}
    for floor in range(page, upto + 1, page):
        hasher.update(data[floor - page : floor].tobytes())
        digests[floor] = hasher.copy().digest()
    return digests


@dataclasses.dataclass
class Tail:
    floor: int  # the page boundary the tail starts at
    end: int  # where the tail ends (a multiple of the hit alignment)
    tokens: np.ndarray  # the prompt's tokens [floor, end)
    key: tuple[bytes, int]  # (digest of the tokens before floor, floor)


class TailIndex:
    """Up to ``slots`` tails. A new tail takes the slot of a tail its prompt
    extends (the same session's earlier turn, which its next turn no longer
    needs), else the least recently used: with LRU alone, sessions' turns
    evicted each other's live tails while superseded ones lingered (124 of
    1137 turns an hour recomputed up to a page at AgentX concurrency 8)."""

    def __init__(self, slots: int, page: int, align: int) -> None:
        if slots < 1 or page % align:
            raise ValueError(f"{slots} tails, page {page} and alignment {align}")
        self.slots, self.page, self.align = slots, page, align
        self._tails: list[Tail | None] = [None] * slots
        self._by_key: dict[tuple[bytes, int], set[int]] = collections.defaultdict(set)
        self._lru: collections.OrderedDict[int, None] = collections.OrderedDict(
            (t, None) for t in range(slots)
        )
        # Tails the batch reads or writes, with how many of its requests do.
        self._pinned: collections.Counter[int] = collections.Counter()

    def begin_step(self) -> None:
        """A new batch: tails it reads are pinned until the next one."""
        self._pinned = collections.Counter()

    def lookup(
        self, prompt: np.ndarray, hit: int, limit: int
    ) -> tuple[int, int] | None:
        """The tail extending a page-aligned radix ``hit`` of ``prompt`` the
        furthest, ending at most at ``limit``: ``(tail, end)`` or None."""
        if hit % self.page or limit <= hit:
            return None
        bucket = self._by_key.get((_digest(prompt[:hit]), hit))
        best = None
        for t in bucket or ():
            tail = self._tails[t]
            n = tail.end - tail.floor
            if tail.end <= limit and (best is None or tail.end > best[1]):
                if np.array_equal(prompt[hit : hit + n], tail.tokens):
                    best = (t, tail.end)
        if best is not None:
            self._pinned[best[0]] += 1
            self._lru.move_to_end(best[0])
        return best

    def record(
        self, prompt: np.ndarray, end: int, own: int | None = None
    ) -> int | None:
        """A slot for the tail of ``prompt`` ending at ``end`` (a multiple of
        the alignment past its last page boundary), or None when there is
        nothing to keep: no tokens past the boundary, or that tail already
        kept (refreshed). ``own``: the tail this request's lookup hit in the
        same batch, which its new tail may replace (the program reads the hit
        state inside the step, before any checkpoint write, and the tail
        copies run after it) unless another request of the batch reads it."""
        if end % self.align:
            raise ValueError(
                f"a tail ends on the {self.align}-token alignment, not {end}"
            )
        floor = end // self.page * self.page
        if end == floor:
            return None
        digests = _page_digests(prompt, self.page, floor)
        key = (digests[floor], floor)
        tokens = np.array(prompt[floor:end], dtype=np.int32)
        for t in self._by_key.get(key, ()):
            tail = self._tails[t]
            if tail.end == end and np.array_equal(tail.tokens, tokens):
                self._lru.move_to_end(t)
                return None
        slot = self._superseded(prompt, digests, end, own)
        if slot is None:
            slot = next((t for t in self._lru if t not in self._pinned), None)
        if slot is None:
            return None
        old = self._tails[slot]
        if old is not None:
            self._by_key[old.key].discard(slot)
            if not self._by_key[old.key]:
                del self._by_key[old.key]
        self._tails[slot] = Tail(floor, end, tokens, key)
        self._by_key[key].add(slot)
        self._lru.move_to_end(slot)
        self._pinned[slot] += 1
        return slot

    def _superseded(
        self,
        prompt: np.ndarray,
        digests: dict[int, bytes],
        end: int,
        own: int | None,
    ) -> int | None:
        """The tail ``prompt`` extends that ends the furthest (before
        ``end``), unpinned or pinned only by ``own``'s lookup, or None."""
        best = None
        for t, tail in enumerate(self._tails):
            pinned = self._pinned[t] - (1 if t == own else 0)
            if (
                tail is None
                or pinned > 0
                or tail.end >= end
                or digests.get(tail.floor) != tail.key[0]
                or (best is not None and tail.end <= self._tails[best].end)
            ):
                continue
            if np.array_equal(prompt[tail.floor : tail.end], tail.tokens):
                best = t
        return best

    def drop(self, slot: int) -> None:
        """Forget a tail whose blocks were not written."""
        tail = self._tails[slot]
        if tail is None:
            return
        self._by_key[tail.key].discard(slot)
        if not self._by_key[tail.key]:
            del self._by_key[tail.key]
        self._tails[slot] = None
        self._lru.move_to_end(slot, last=False)
