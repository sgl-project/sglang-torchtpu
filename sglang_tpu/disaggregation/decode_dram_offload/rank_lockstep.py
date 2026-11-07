"""Cross-rank agreement for the offload to decode DRAM.

Every attention-TP rank holds its own slice of a branch and its own host pool.
A tree changes only in a step where every rank made the same decision, so the
ranks reduce their local view first.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, List, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist

# Arbitrary: pushes a rank offers per agreement round.
_AGREE_SLOTS = 8


class RankLockstep:
    """All-reduces over the CPU groups whose ranks must make the same tree
    decisions. Every rank must call the same method in the same step."""

    def __init__(self, groups: Sequence[Any] = ()) -> None:
        self._groups = list(groups)

    @classmethod
    def of(cls, scheduler: Any) -> RankLockstep:
        return cls(
            group
            for group in (scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group)
            if group is not None and dist.get_world_size(group) > 1
        )

    def reduce(self, rows: List[List[int]], op: Any) -> List[List[int]]:
        if not self._groups or not rows:
            return [list(row) for row in rows]
        values = torch.tensor(rows, dtype=torch.int64)
        for group in self._groups:
            dist.all_reduce(values, op=op, group=group)
        return values.tolist()

    def all(self, flag: Any) -> bool:
        return bool(self.reduce([[int(bool(flag))]], dist.ReduceOp.MIN)[0][0])

    def any(self, flag: Any) -> bool:
        return bool(self.reduce([[int(bool(flag))]], dist.ReduceOp.MAX)[0][0])

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

    def drain_until(
        self,
        *,
        done: Callable[[], bool],
        step: Callable[[], None],
        timeout_s: float,
        what: str,
    ) -> None:
        """Run ``step`` until ``done`` holds on every rank; raise on all of them
        once any rank's deadline passed, so that the caller can retry."""
        deadline = time.monotonic() + timeout_s
        while True:
            empty, expired = self.reduce(
                [[int(done()), int(time.monotonic() > deadline)]], dist.ReduceOp.MIN
            )[0]
            if empty:
                return
            if expired:
                raise RuntimeError(
                    f"Cannot release KV memory while {what}; retry after it drains"
                )
            step()
            time.sleep(0.05)

    def common_to_all_ranks(self, keys: list) -> list:
        """The bounded set of ``keys`` every rank holds, over the full TP/CP
        mesh: each group's intersection feeds the next group."""
        common = set(sorted(keys)[:_AGREE_SLOTS])
        for group in self._groups:
            gathered = [None] * dist.get_world_size(group)
            dist.all_gather_object(gathered, sorted(common), group=group)
            common.intersection_update(*(set(row) for row in gathered))
        return sorted(common)


class DramOffloadWorkerContext(msgspec.Struct, frozen=True):
    """What every worker of one engine's extension shares."""

    # The radix cache whose L2 tier is extended, and its full-KV host pool.
    cache: Any
    host_pool: Any
    page_size: int
    # The raiden channel to the peer role (raiden/dram_offload_channel.py).
    channel: Any
    lockstep: RankLockstep


def push_manifest(tokens, extra_key=None, cache_salt=None, *geometry):
    payload = msgspec.msgpack.encode((tokens, extra_key, cache_salt, geometry))
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little") >> 1 for i in (0, 8))
