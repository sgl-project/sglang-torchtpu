"""Cross-rank agreement for the offload to decode DRAM.

Every attention-TP rank holds its own slice of a branch and its own host pool.
A tree changes only in a step where every rank made the same decision, so the
ranks reduce their local view first.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist

# Arbitrary: pushes a rank offers per agreement round.
_AGREE_SLOTS = 8


class RankLockstep:
    """All-reduces over the CPU group whose ranks must make the same tree
    decisions. Every rank must call the same method in the same step."""

    def __init__(self, group: Any = None) -> None:
        self._group = group

    @classmethod
    def of(cls, scheduler: Any) -> RankLockstep:
        # Context parallelism is refused at startup (startup_validation.py),
        # so the attention-TP group is the only one whose ranks must agree.
        group = scheduler.attn_tp_cpu_group
        if group is None or dist.get_world_size(group) == 1:
            group = None
        return cls(group)

    def reduce(self, rows: List[List[int]], op: Any) -> List[List[int]]:
        if self._group is None or not rows:
            return [list(row) for row in rows]
        values = torch.tensor(rows, dtype=torch.int64)
        dist.all_reduce(values, op=op, group=self._group)
        return values.tolist()

    def all(self, flag: Any) -> bool:
        return bool(self.reduce([[int(bool(flag))]], dist.ReduceOp.MIN)[0][0])

    def max(self, value: int) -> int:
        return self.reduce([[int(value)]], dist.ReduceOp.MAX)[0][0]

    def min_max(self, row: Sequence[int]) -> Tuple[List[int], List[int]]:
        """The minimum and the maximum of each value of ``row`` over the
        ranks, in one collective."""
        # MIN of a value and of its negation gives its minimum and maximum.
        low = self.reduce([[*row, *(-v for v in row)]], dist.ReduceOp.MIN)[0]
        return low[: len(row)], [-v for v in low[len(row) :]]

    def agree(self, row: Sequence[int]) -> bool:
        """Whether every rank holds the same ``row``."""
        low, high = self.min_max(row)
        return low == high

    def common_to_all_ranks(self, keys: list) -> list:
        """The bounded set of ``keys`` every rank holds."""
        common = sorted(keys)[:_AGREE_SLOTS]
        if self._group is None:
            return common
        gathered = [None] * dist.get_world_size(self._group)
        dist.all_gather_object(gathered, common, group=self._group)
        return sorted(set(common).intersection(*(set(row) for row in gathered)))


def alloc_host_pages_agreed(
    *, lockstep: RankLockstep, pools: Sequence[Tuple[Any, int]]
) -> Optional[List[Any]]:
    """For each (pool, num_tokens): its host pages, None for zero tokens; on
    every rank, or None on every rank once any rank missed an allocation, in
    which case a rank that got pages frees them."""
    indices = [pool.alloc(tokens) if tokens else None for pool, tokens in pools]
    got = all(i is not None for (_, tokens), i in zip(pools, indices) if tokens)
    if lockstep.all(got):
        return indices
    for (pool, _), i in zip(pools, indices):
        if i is not None:
            pool.free(i)
    return None


class DramOffloadWorkerContext(msgspec.Struct, frozen=True):
    """What every worker of one engine's extension shares."""

    # The radix cache whose L2 tier is extended, and its full-KV host pool.
    cache: Any
    host_pool: Any
    page_size: int
    # The raiden channel to the peer role (raiden/dram_offload_channel.py).
    channel: Any
    lockstep: RankLockstep
    # Hybrid sliding-window models: the host pool of SWA windows and the
    # window in tokens, rounded up to whole pages. None and 0 on other models.
    swa_host_pool: Any = None
    window_tokens: int = 0


def push_manifest(tokens, extra_key, cache_salt, *geometry):
    """Two non-negative int64 values that identify a branch's content across
    ranks; the ranks agree on them before any tree decision about the branch."""
    payload = msgspec.msgpack.encode((tokens, extra_key, cache_salt, geometry))
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little") >> 1 for i in (0, 8))
