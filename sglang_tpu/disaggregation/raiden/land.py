"""Reshard landing arena for fp8 token-major push resharding (#829).

A decode rank whose heads came from several one-head prefill ranks cannot take
every source straight into its final page: an fp8 32-bit granule holds two
heads, and a one-head source owns only half of it. Such a source lands in a
scratch block instead, and the decode merges the scratch into the final page
once the plan settles.

The scratch blocks are ``A`` extra KV pool blocks past the allocator's pages,
so raiden can address them like any other block and the allocator never hands
them out: the pool has ``size // page_size + 1 + A`` blocks, block 0 is the
padding page, pages ``1 .. size // page_size`` belong to the allocator
(``TpuPagedTokenToKVPoolAllocator.clear``), and the last ``A`` are the arena.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

from sglang.srt.environ import envs
from sglang_tpu.disaggregation.raiden.common import CHUNK_ANNOUNCE_TOKENS

# --kv-cache-dtype values that give the target an fp8 store dtype.
_FP8_KV_CACHE_DTYPES = frozenset({"fp8_e4m3", "fp8_e5m2"})


def land_arena_blocks(*, swa: bool = False) -> int:
    """Arena blocks every KV pool of a raiden group reserves past its pages.

    One decision per group, taken from the target's deployment (published
    config), never from the pool asking: raiden requires equal block counts
    within a group (``get_raiden_kv_caches``), and a draft pool may differ
    from the target in store dtype (--speculative-draft-kv-cache-dtype) or
    page size (DCP widens draft pages). The SWA group never lands: 0.

    Non-zero only when SGLANG_RAIDEN_RESHARD_LAND_ARENA is set on a raiden
    decode whose target pool is token-major fused with an fp8 store dtype, the
    only layout whose push reshard needs landing. A target resolved to fp8 from
    ``--kv-cache-dtype auto`` (a checkpoint's kv quant) is not seen here and
    gets no arena, so its LAND schedules stay refused.

    The size is two announcement chunks of schedule pages,
    ``2 * ceil(8192 / page_size)``.
    """
    if not envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.get() or swa:
        return 0
    from sglang.srt.runtime_context import (
        get_disagg,
        get_model,
        get_schedule,
        uses_mla_backend,
    )

    disagg = get_disagg()
    if (
        disagg.disaggregation_mode != "decode"
        or disagg.disaggregation_transfer_backend != "raiden"
        or get_model().kv_cache_dtype not in _FP8_KV_CACHE_DTYPES
        # The budget sizer charges nothing for an MLA target
        # (pool_configurator._land_arena_sizer); an MHA draft beside it must
        # not reserve either, or the full group's block counts differ.
        or uses_mla_backend()
    ):
        return 0
    from sglang.srt.platforms import current_platform
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    pool_cls = current_platform.get_mha_kv_pool_cls()
    if (
        not issubclass(pool_cls, TpuFusedMHATokenToKVPool)
        # Pools without a disagg_kv_layout are token-major.
        or getattr(pool_cls, "disagg_kv_layout", None) is not None
    ):
        return 0
    return 2 * -(-CHUNK_ANNOUNCE_TOKENS // get_schedule().page_size)


class ScratchArena:
    """Hands out a pool's arena block ids, each held by at most one owner.

    Not thread-safe: the decode's ``RaidenKVManager`` serializes every call
    under its ``_state_lock``.
    """

    def __init__(self, first_block: int, num_blocks: int):
        if first_block < 1 or num_blocks < 1:
            raise ValueError(
                f"arena needs blocks past the padding page; got first_block="
                f"{first_block}, num_blocks={num_blocks}"
            )
        self.first_block = first_block
        self.capacity = num_blocks
        self._free: deque[int] = deque(range(first_block, first_block + num_blocks))
        self._held: set[int] = set()

    @classmethod
    def from_pool(cls, pool: Any) -> Optional[ScratchArena]:
        """The arena of a KV pool (the full sub-pool of an SWA pool), or None
        when it reserved none."""
        pool = getattr(pool, "full_kv_pool", pool)
        count = int(getattr(pool, "land_arena_blocks", 0) or 0)
        if count == 0:
            return None
        return cls(int(pool.num_blocks) - count, count)

    @property
    def held(self) -> int:
        return len(self._held)

    @property
    def available(self) -> int:
        return len(self._free)

    def take(self, n: int) -> Optional[list[int]]:
        """``n`` block ids, or None (taking nothing) when fewer are free."""
        if n < 0:
            raise ValueError(f"cannot take {n} arena blocks")
        if n > len(self._free):
            return None
        ids = [self._free.popleft() for _ in range(n)]
        self._held.update(ids)
        return ids

    def free(self, ids: Iterable[int]) -> None:
        """Return taken ids. Every id must be held, and listed once."""
        ids = list(ids)
        unique = set(ids)
        # Not an assert: a duplicated free id would let two plans share one
        # scratch block, and -O must not drop the check.
        if len(unique) != len(ids) or not unique <= self._held:
            raise RuntimeError(
                f"arena double free or foreign block: {sorted(ids)} "
                f"(held {sorted(self._held)})"
            )
        self._held -= unique
        self._free.extend(ids)
