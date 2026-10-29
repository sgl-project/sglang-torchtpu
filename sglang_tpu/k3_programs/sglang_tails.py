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


@dataclasses.dataclass
class Tail:
    floor: int  # the page boundary the tail starts at
    end: int  # where the tail ends (a multiple of the hit alignment)
    tokens: np.ndarray  # the prompt's tokens [floor, end)
    key: tuple[bytes, int]  # (digest of the tokens before floor, floor)


class TailIndex:
    """Up to ``slots`` tails, least recently used out first."""

    def __init__(self, slots: int, page: int, align: int) -> None:
        if slots < 1 or page % align:
            raise ValueError(f"{slots} tails, page {page} and alignment {align}")
        self.slots, self.page, self.align = slots, page, align
        self._tails: list[Tail | None] = [None] * slots
        self._by_key: dict[tuple[bytes, int], set[int]] = collections.defaultdict(set)
        self._lru: collections.OrderedDict[int, None] = collections.OrderedDict(
            (t, None) for t in range(slots)
        )
        self._pinned: set[int] = set()

    def begin_step(self) -> None:
        """A new batch: tails it reads are pinned until the next one."""
        self._pinned = set()

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
            self._pinned.add(best[0])
            self._lru.move_to_end(best[0])
        return best

    def record(self, prompt: np.ndarray, end: int) -> int | None:
        """A slot for the tail of ``prompt`` ending at ``end`` (a multiple of
        the alignment past its last page boundary), or None when there is
        nothing to keep: no tokens past the boundary, or that tail already
        kept (refreshed)."""
        if end % self.align:
            raise ValueError(
                f"a tail ends on the {self.align}-token alignment, not {end}"
            )
        floor = end // self.page * self.page
        if end == floor:
            return None
        key = (_digest(prompt[:floor]), floor)
        tokens = np.array(prompt[floor:end], dtype=np.int32)
        for t in self._by_key.get(key, ()):
            tail = self._tails[t]
            if tail.end == end and np.array_equal(tail.tokens, tokens):
                self._lru.move_to_end(t)
                return None
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
        self._pinned.add(slot)
        return slot

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
