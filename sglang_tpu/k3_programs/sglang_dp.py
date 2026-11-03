# SPDX-License-Identifier: Apache-2.0
"""Attention DP2 on a P/D decode: two DCP16 groups over the unified pools.

Under ``attention_dp=2`` the decode program's 32 ranks form two groups of 16
(ranks ``16 g .. 16 g + 15``) that attend their own requests' pages with
16-way decode context parallelism: a sequence's 512-token chunk ``k`` lives on
group rank ``k % 16`` in logical block ``k // 16`` (8192 tokens a block), and a
block-table id ``g N + row`` names pool row ``row`` on group ``g``'s ranks
(``KimiDecoder._mla_table``; ``N``: the pools' rows).

sglang keeps one KV page size on both P/D roles (16384 tokens: a DCP32 logical
block of the prefill), so a decode page holds two DCP16 blocks. The pages split
into one region per group: page ``p`` belongs to group ``(p - 1) // H``
(``H``: pages a group) at index ``i = (p - 1) % H`` and owns rows ``2i + 1``
and ``2i + 2`` on that group's ranks. The pool rows are as without DP (row 0
the null block, state and tail rows past the KV rows, shared by every rank).

The transfer keeps Raiden's rank pairing (decode rank ``d`` pulls prefill rank
``d``). Prefill rank ``16h + q``'s row for the request's ``j``-th page holds
chunks ``32 j' + 16 h + q``: DCP16 block ``2j + h`` of group rank ``q``. When
``h`` is the request's group the row lands in place; otherwise it lands in a
landing row of a page from group ``h``'s region, and a swap op moves it to
group rank ``q`` of the request's group before its first decode step.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

GROUPS = 2
GROUP_RANKS = 16
CHUNK_TOKENS = 512  # tokens a pool row holds on one rank (fp8)
GROUP_BLOCK_TOKENS = GROUP_RANKS * CHUNK_TOKENS  # 8192: a DCP16 logical block
PAGE_TOKENS = GROUPS * GROUP_BLOCK_TOKENS  # 16384: sglang's page (a DCP32 block)


@dataclasses.dataclass(frozen=True)
class DpPages:
    """The decode's KV pages ``1 .. pages`` split into one region per group."""

    pages: int  # sglang KV pages (page 0, the null page, not counted)
    num_blocks: int  # pool rows N (block-table ids are g N + row)

    def __post_init__(self) -> None:
        if self.pages < GROUPS or self.pages % GROUPS:
            raise ValueError(f"DP2 needs an even number of KV pages, not {self.pages}")
        if self.pages >= self.num_blocks:
            raise ValueError(f"{self.pages} KV pages do not fit {self.num_blocks} rows")

    @property
    def per_group(self) -> int:
        return self.pages // GROUPS

    def group(self, page: int) -> int:
        """The group whose region holds sglang page ``page``."""
        self._check(page)
        return (page - 1) // self.per_group

    def index(self, page: int) -> int:
        self._check(page)
        return (page - 1) % self.per_group

    def rows(self, page: int) -> tuple[int, int]:
        """The two pool rows page ``page`` owns on its group's ranks."""
        i = self.index(page)
        return 2 * i + 1, 2 * i + 2

    def region(self, group: int) -> range:
        """Group ``group``'s pages."""
        return range(1 + group * self.per_group, 1 + (group + 1) * self.per_group)

    def table_ids(self, page: int) -> tuple[int, int]:
        """Page ``page``'s two block-table ids (``g N + row``)."""
        base = self.group(page) * self.num_blocks
        first, second = self.rows(page)
        return base + first, base + second

    def table_row(self, pages: Sequence[int], seq_len: int, width: int) -> np.ndarray:
        """A request's MLA block-table row: two ids for each of its sglang
        ``pages``, cut to the DCP16 blocks its ``seq_len`` tokens touch,
        0-padded to ``width``."""
        used = -(-seq_len // GROUP_BLOCK_TOKENS)
        if used > width:
            raise ValueError(f"{seq_len} tokens need {used} blocks; the table holds {width}")
        if -(-used // GROUPS) > len(pages):
            raise ValueError(f"{len(pages)} pages do not cover {seq_len} tokens")
        groups = {self.group(int(p)) for p in pages[: -(-used // GROUPS)]}
        if len(groups) > 1:
            raise ValueError(f"pages {list(pages)} span DP groups {sorted(groups)}")
        ids = [i for p in pages[: -(-used // GROUPS)] for i in self.table_ids(int(p))]
        row = np.zeros(width, np.int32)
        row[:used] = ids[:used]
        return row

    def transfer_rows(
        self, rank: int, pages: Sequence[int], landing: Sequence[int]
    ) -> list[int]:
        """Decode rank ``rank``'s destination row for each of the request's
        ``pages`` (prefill rank ``rank``'s rows, in page order): in place when
        ``rank`` is in the pages' group, else row ``j % 2`` of landing page
        ``landing[j // 2]`` (pages of the other group's region)."""
        g = self.group(int(pages[0])) if len(pages) else 0
        h = rank // GROUP_RANKS
        if h == g:
            return [self.rows(int(p))[h] for p in pages]
        need = -(-len(pages) // 2)
        if len(landing) < need or any(self.group(int(p)) != h for p in landing[:need]):
            raise ValueError(
                f"{len(pages)} pages need {need} landing pages of group {h}, "
                f"got {list(landing)}"
            )
        return [self.rows(int(landing[j // 2]))[j % 2] for j in range(len(pages))]

    def swap_rows(
        self, pages: Sequence[int], landing: Sequence[int]
    ) -> tuple[list[int], list[int]]:
        """The swap after a transfer: ``(source rows on the other group's
        ranks, destination rows on the pages' group's ranks)``, group rank
        ``q`` to group rank ``q``."""
        g = self.group(int(pages[0]))
        h = 1 - g
        sources = self.transfer_rows(h * GROUP_RANKS, pages, landing)
        destinations = [self.rows(int(p))[h] for p in pages]
        return sources, destinations

    def _check(self, page: int) -> None:
        if not 1 <= page <= self.pages:
            raise ValueError(f"KV page {page} is outside 1 .. {self.pages}")


def landing_pages_needed(pages: int) -> int:
    """Landing pages a request of ``pages`` pages takes in the other group."""
    return -(-pages // 2)


# ------------------------------------------------------------ the swap op
SWAP_BUCKETS = (16, 64, 256, 1024)  # swap entries a call, padded


def swap_bucket(entries: int) -> int:
    for bucket in SWAP_BUCKETS:
        if entries <= bucket:
            return bucket
    raise ValueError(f"{entries} swap rows exceed {SWAP_BUCKETS[-1]}")


def swap_operands(
    jobs: Sequence[tuple[Sequence[int], Sequence[int], int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(source rows, destination rows, destination group)``, padded to a
    bucket with entries that write nowhere (group -1), from ``(sources,
    destinations, group)`` per request (:meth:`DpPages.swap_rows`)."""
    sources = [s for job in jobs for s in job[0]]
    destinations = [d for job in jobs for d in job[1]]
    groups = [job[2] for job in jobs for _ in job[0]]
    size = swap_bucket(max(len(sources), 1))
    pad = size - len(sources)
    return (
        np.asarray(sources + [0] * pad, np.int32),
        np.asarray(destinations + [0] * pad, np.int32),
        np.asarray(groups + [-1] * pad, np.int32),
    )


def swap_body(axis: str = "tp"):
    """The per-rank body (inside a ``shard_map`` over ``axis``, 32 ranks):
    each rank sends its pools' source rows to its partner in the other group
    (rank ``r`` to ``r + 16 mod 32``) and writes what it receives at the
    destination rows of the entries whose destination group is its own."""
    import jax
    import jax.numpy as jnp

    perm = [(r, (r + GROUP_RANKS) % (GROUPS * GROUP_RANKS)) for r in range(GROUPS * GROUP_RANKS)]

    def body(sources, destinations, groups, *pools):
        mine = jax.lax.axis_index(axis) // GROUP_RANKS
        write = groups == mine
        out = []
        for pool in pools:
            words = pool.view(jnp.uint8) if pool.dtype.itemsize == 1 else pool
            sent = words[sources]
            got = jax.lax.ppermute(sent, axis, perm)
            mask = write.reshape((-1,) + (1,) * (sent.ndim - 1))
            words = words.at[destinations].set(jnp.where(mask, got, words[destinations]))
            out.append(words.view(pool.dtype) if words.dtype != pool.dtype else words)
        return tuple(out)

    return body
