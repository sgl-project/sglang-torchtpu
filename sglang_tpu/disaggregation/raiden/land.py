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
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

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


# Carried by RESHARD_LAND_READY / RESHARD_LAND_CHUNK_READY. A prefill that
# meets another version fails the room instead of pushing a schedule its
# decode may place differently.
LAND_VERSION = 1

# Merge batch sizes (pairs), so a round compiles one of a few shapes; the
# arena capacity tops the list.
MERGE_BUCKETS = (8, 16, 32, 64)


@dataclass
class LandPlan:
    """Scratch blocks one receive plan holds, from take to free."""

    room: int
    scratch: List[int]
    # The final pages the scratch pairs with, set when the plan is published.
    final: List[int] = field(default_factory=list)
    # READY sent: the engine will report the plan done or failed.
    published: bool = False
    # Reported done while its room was live: waiting for the merge.
    settled: bool = False


@dataclass
class LandHold:
    """A take the arena could not serve yet, retried in FIFO order."""

    key: str
    room: int
    pages: int
    deadline: float
    # Re-runs the registration from the take on; called without the lock.
    retry: Callable[[], None]
    # Fails the room once the hold expires; called without the lock.
    expire: Callable[[], None]


class LandState:
    """The decode's scratch plans and holds over one ``ScratchArena``.

    Not thread-safe: every call runs under ``RaidenKVManager._state_lock``.
    A plan is keyed by its receive plan's req_id; ``free`` pops it, so each
    plan returns its ids exactly once whichever terminal path gets there
    first.
    """

    def __init__(self, arena: ScratchArena):
        self.arena = arena
        self.plans: Dict[str, LandPlan] = {}
        self.holds: Deque[LandHold] = deque()
        # When each room registered, for the hold deadline.
        self.started: Dict[int, float] = {}

    def take(self, key: str, room: int, n: int) -> Optional[List[int]]:
        """``n`` scratch ids for plan ``key``, or None when it must wait.

        A take never jumps the hold queue: only the head hold's own retry
        can take while holds are queued. Raises ValueError for a plan the
        arena can never serve.
        """
        if n > self.arena.capacity:
            raise ValueError(
                f"a landing plan of {n} pages exceeds the {self.arena.capacity}"
                "-block reshard landing arena"
            )
        if key in self.plans:
            raise ValueError(f"landing plan {key} already holds scratch")
        if self.holds and self.holds[0].key != key:
            return None
        ids = self.arena.take(n)
        if ids is None:
            return None
        if self.holds:
            self.holds.popleft()
        self.plans[key] = LandPlan(room=room, scratch=ids)
        return ids

    def hold(self, hold: LandHold) -> None:
        if all(h.key != hold.key for h in self.holds):
            self.holds.append(hold)

    def publish(self, key: str, final: List[int]) -> None:
        plan = self.plans[key]
        plan.final = list(final)
        plan.published = True

    def free(self, key: str) -> None:
        plan = self.plans.pop(key, None)
        if plan is not None:
            self.arena.free(plan.scratch)

    def settle(
        self,
        done: Iterable[str],
        failed: Iterable[str],
        live: Callable[[int], bool],
    ) -> None:
        """Record engine reports. A failed plan may have H2D'd some layers
        only, so it is never merged; the report fires once every H2D and
        accepted push has ended, so its scratch is free at once. A done plan
        merges if its room is still live, else frees."""
        for key in failed:
            self.free(key)
        for key in done:
            plan = self.plans.get(key)
            if plan is None or not plan.published:
                continue
            if live(plan.room):
                plan.settled = True
            else:
                self.free(key)

    def drop_room(self, room: int) -> None:
        """Forget a cleared room: its holds, and its settled plans (no merge,
        the pages are gone). Unsettled plans keep their scratch until the
        engine reports them."""
        self.holds = deque(h for h in self.holds if h.room != room)
        for key in [k for k, p in self.plans.items() if p.room == room and p.settled]:
            self.free(key)
        self.started.pop(room, None)

    def merged(self, room: int) -> bool:
        """No plan of ``room`` holds scratch: every landed page is merged."""
        return all(p.room != room for p in self.plans.values())


def merge_bucket(n: int, capacity: int) -> int:
    """The batch size ``n`` merge pairs pad to."""
    for b in (*MERGE_BUCKETS, capacity):
        if n <= b:
            return b
    raise ValueError(f"{n} merge pairs exceed the {capacity}-block arena")


def merge_pages(
    kv_caches: Sequence[Any],
    pairs: Sequence[Tuple[int, int]],
    capacity: int,
    layout_of: Callable[[Any], Any],
) -> None:
    """Merge every (final, scratch) block pair into the final page, in place
    on each KV tensor: gather both, move scratch slots 0-1 into final slots
    2-3 on a uint8 view of the gathered copy (never of the pool), and write
    the final pages back. Pads to ``merge_bucket`` by repeating the last pair:
    rewriting a page with its own merge is idempotent, a block-0 dummy is
    not."""
    import torch

    from sglang_tpu.disaggregation.raiden.reshard_schedule import land_merge
    from sglang_tpu.kv_page_write import write_pages_torch

    if not pairs:
        return
    pairs = list(pairs)
    pairs += [pairs[-1]] * (merge_bucket(len(pairs), capacity) - len(pairs))
    final = torch.tensor([f for f, _ in pairs], dtype=torch.long)
    scratch = torch.tensor([s for _, s in pairs], dtype=torch.long)
    # One H2D of the ids per device, not per layer.
    ids_on: dict = {}
    for kv in kv_caches:
        ptr = kv.data_ptr()
        if kv.device not in ids_on:
            ids_on[kv.device] = (final.to(kv.device), scratch.to(kv.device))
        f, s = ids_on[kv.device]
        pages = kv[f]
        land_merge(pages.view(torch.uint8), kv[s].view(torch.uint8), layout_of(kv))
        write_pages_torch(f, pages, kv)
        # raiden pushes into this buffer by address: the merge must write in
        # place, never rebind the tensor.
        assert kv.data_ptr() == ptr, "landing merge moved a KV buffer"


def device_sync(kv_caches: Sequence[Any]) -> None:
    """``torch.accelerator.synchronize()``; a no-op for CPU tensors."""
    if kv_caches and kv_caches[0].device.type != "cpu":
        import torch

        torch.accelerator.synchronize()
