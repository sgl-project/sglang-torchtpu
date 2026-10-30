"""Unit tests for write_back copy-ahead (sglang_tpu/hicache_write_back_copy_ahead.py).

A real ``UnifiedRadixCache`` on CPU (FULL-only, FULL+SWA where SWA matters) with
a fake raiden controller whose D2H futures complete only when the test says so.
Covers the contracts the TPU e2e gate cannot isolate: which victims the window
copies and how many it keeps in flight, that an eviction waits for the window's
copies only when it would otherwise come up short, and that a victim past the
window still takes the synchronous backup without waiting for the window.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import hicache_write_back_copy_ahead as copy_ahead_module
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hicache_write_back_copy_ahead import (
    WriteBackCopyAhead,
    attach_copy_ahead,
    resolve_copy_ahead_tokens,
)
from sglang_tpu.raiden_transfer import NOOP_EVENT
from sglang_tpu.tpu_dma_hybrid_cache_controller import (
    TpuDMAHybridCacheController,
)

FULL = ComponentType.FULL
SWA = ComponentType.SWA


class _FakeFuture:
    """A raiden D2H future: ready when the test says, and a blocking wait is counted."""

    def __init__(self, waits: list):
        self.ready = False
        self._waits = waits

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        if not self.ready:
            self._waits.append(self)
            self.ready = True


class _FakeHostPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0

    def available_size(self) -> int:
        return self.capacity - self.used

    def alloc(self, num_tokens: int):
        if num_tokens > self.available_size():
            return None
        indices = torch.arange(self.used, self.used + num_tokens, dtype=torch.int64)
        self.used += num_tokens
        return indices

    def free(self, indices) -> None:
        self.used -= len(indices)

    def clear(self) -> None:
        self.used = 0


class _FakeRaidenController(TpuDMAHybridCacheController):
    """The controller surface the cache's backup and ack paths touch."""

    def __init__(self, *, host_capacity: int, write_policy: str = "write_back"):
        self.write_policy = write_policy
        self.host = _FakeHostPool(host_capacity)
        self.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(
                host_pool=SimpleNamespace(shared_allocation_domain=None),
                host_evict_fn=None,
            ),
            available_size=self.host.available_size,
            clear=self.host.clear,
        )
        self.write_queue: list = []
        self.ack_write_queue: list = []
        self.ack_load_queue: list = []
        self.extra_host_mem_release_queues: dict = {}
        self.waits: list = []

    def write(
        self, device_indices, priority=None, node_id=-1, extra_pools=None, flush=True
    ):
        host_indices = self.host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append((host_indices, device_indices, node_id))
        if flush:
            self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        node_ids = [node_id for _, _, node_id in self.write_queue]
        num_tokens = sum(len(d) for _, d, _ in self.write_queue)
        self.write_queue.clear()
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=_FakeFuture(self.waits),
                node_ids=node_ids,
                num_tokens=num_tokens,
            )
        )

    def land_all(self) -> None:
        for ack in self.ack_write_queue:
            ack.finish_event.ready = True

    def reset(self) -> None:
        self.write_queue.clear()
        self.ack_write_queue.clear()


def _build_cache(
    *, kv_size: int = 64
) -> tuple[UnifiedRadixCache, TokenToKVPoolAllocator]:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=kv_size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        tree_components=(FULL,),
        tree_core_backend="python",
    )
    return UnifiedRadixCache(params), allocator


def _build_swa_cache(
    *, kv_size: int = 64, sliding_window_size: int = 16
) -> tuple[UnifiedRadixCache, SWATokenToKVPoolAllocator]:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size,
        page_size=1,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        sliding_window_size=sliding_window_size,
        tree_components=(FULL, SWA),
        tree_core_backend="python",
    )
    return UnifiedRadixCache(params), allocator


def _attach_write_back(
    cache: UnifiedRadixCache, *, host_capacity: int, copy_ahead_tokens: int
) -> _FakeRaidenController:
    controller = _FakeRaidenController(host_capacity=host_capacity)
    cache.cache_controller = controller
    # Host frees reach the fake pool, so host eviction makes room again.
    cache.components[FULL]._full_kv_pool_host = controller.host
    cache.host_pool_group = SimpleNamespace(
        free=lambda indices, pool=None: controller.host.free(indices)
    )
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.is_write_back = True
    cache.is_write_back = True
    cache._write_back_copy_ahead = (
        WriteBackCopyAhead(cache=cache, copy_ahead_tokens=copy_ahead_tokens)
        if copy_ahead_tokens
        else None
    )
    return controller


def _insert(cache: UnifiedRadixCache, allocator, tokens: list[int]) -> int:
    value = allocator.alloc(len(tokens))
    assert value is not None
    cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=value))
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    return match.last_device_node


def _node(cache: UnifiedRadixCache, node_id: int):
    return cache.tree_core.node_by_id(node_id)


class TestWriteBackCopyAhead(CustomTestCase):
    def setUp(self):
        originals = {
            name: getattr(UnifiedRadixCache, name)
            for name in (
                "init_hicache",
                "_evict",
                "check_hicache_events",
                "_finish_write_through_ack",
                "writing_check",
                "backup_node_for_write_back",
            )
        }
        had_copy_ahead_attr = "_write_back_copy_ahead" in vars(UnifiedRadixCache)

        def restore():
            for name, value in originals.items():
                setattr(UnifiedRadixCache, name, value)
            if not had_copy_ahead_attr:
                del UnifiedRadixCache._write_back_copy_ahead

        self.addCleanup(restore)
        copy_ahead_module.install(cache_cls=UnifiedRadixCache)

    def test_window_copies_the_eviction_order_and_bounds_what_is_in_flight(self):
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        c = _insert(cache, allocator, [3] * 8)

        cache.check_hicache_events()

        # The two oldest leaves fill the window; the third stays device-only.
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a, b})
        self.assertEqual(controller.host.used, 16)
        self.assertFalse(cache.tree_core.is_backuped(c))
        for node_id in (a, b):
            self.assertTrue(cache.tree_core.is_backuped(node_id))
            self.assertGreater(_node(cache, node_id).component_data[FULL].lock_ref, 0)
        # In-flight copies count against the window: nothing more is issued.
        cache.check_hicache_events()
        self.assertEqual(controller.host.used, 16)
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a, b})

        controller.land_all()
        cache.check_hicache_events()

        for node_id in (a, b):
            self.assertEqual(_node(cache, node_id).component_data[FULL].lock_ref, 0)
            self.assertIn(
                _node(cache, node_id), cache.tree_core.evictable_device_leaves
            )
        # Backed-up leaves still cover the window, so the third is left alone.
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, set())
        self.assertFalse(cache.tree_core.is_backuped(c))

    def test_eviction_demotes_acked_victims_and_waits_for_window_copies_only_when_short(
        self,
    ):
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        c = _insert(cache, allocator, [3] * 8)
        cache.check_hicache_events()
        controller.land_all()
        cache.check_hicache_events()

        result = cache.evict(EvictParams(num_tokens=16))

        self.assertEqual(result.num_tokens_evicted, 16)
        self.assertEqual(controller.waits, [])
        for node_id in (a, b):
            self.assertTrue(_node(cache, node_id).evicted)
            self.assertTrue(cache.tree_core.is_backuped(node_id))

        # The third leaf is now the frontier and its copy is in flight.
        cache.check_hicache_events()
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {c})
        # Nothing else is evictable; returning short would retract a request,
        # so the eviction waits for that copy once and demotes the victim.
        result = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertEqual(len(controller.waits), 1)
        self.assertTrue(_node(cache, c).evicted)
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, set())

    def test_short_eviction_finishes_landed_window_copies_without_a_collective(self):
        """A copy that lands after the step's poll still locks its victim; an
        eviction that needs the victim finishes the landed ack itself, without
        a collective or a blocking wait."""
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        a = _insert(cache, allocator, [1] * 8)
        _insert(cache, allocator, [2] * 8)
        cache.check_hicache_events()
        # The forward in flight finishes during scheduling: both copies land.
        controller.land_all()

        with patch.object(cache, "_all_reduce", wraps=cache._all_reduce) as polls:
            result = cache.evict(EvictParams(num_tokens=8))

        self.assertEqual(polls.call_count, 0)
        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertEqual(controller.waits, [])
        self.assertTrue(_node(cache, a).evicted)

    def test_a_victim_past_the_window_is_backed_up_synchronously(self):
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=8)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        cache.check_hicache_events()
        controller.land_all()
        cache.check_hicache_events()

        result = cache.evict(EvictParams(num_tokens=16))

        self.assertEqual(result.num_tokens_evicted, 16)
        # One blocking wait, for the copy of the leaf the window never reached.
        self.assertEqual(len(controller.waits), 1)
        for node_id in (a, b):
            self.assertTrue(_node(cache, node_id).evicted)
            self.assertTrue(cache.tree_core.is_backuped(node_id))
        self.assertEqual(controller.host.used, 16)

    def test_parent_joins_the_window_once_its_child_is_covered(self):
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        parent = _insert(cache, allocator, [1] * 8)
        child = _insert(cache, allocator, [1] * 8 + [2] * 8)
        self.assertEqual(_node(cache, child).parent.id, parent)

        cache.check_hicache_events()
        # The child's lock covers its path, so the parent waits for the ack.
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {child})
        controller.land_all()
        cache.check_hicache_events()

        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {parent})
        self.assertTrue(cache.tree_core.is_backuped(parent))

    def test_window_copy_ack_leaves_the_write_horizon_alone(self):
        """A request's lock release marks its path as just written; a window copy's
        release must not, or the demoted pages wait behind the in-flight
        forward and the next host reload syncs the device."""
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        parent = _insert(cache, allocator, [1] * 8)
        child = _insert(cache, allocator, [1] * 8 + [2] * 8)
        horizons = {n: _node(cache, n).write_horizon for n in (parent, child)}

        cache.check_hicache_events()
        controller.land_all()
        cache.check_hicache_events()

        self.assertTrue(cache.tree_core.is_backuped(child))
        self.assertEqual(_node(cache, child).component_data[FULL].lock_ref, 0)
        for node_id, horizon in horizons.items():
            self.assertEqual(_node(cache, node_id).write_horizon, horizon)

    def test_host_pressure_does_not_reclaim_the_window_copies(self):
        """With the host full, a new window copy must not make room by reclaiming
        an earlier window copy; otherwise the two victims trade places every step
        and the same KV is copied again and again."""
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=16, copy_ahead_tokens=24)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        c = _insert(cache, allocator, [3] * 8)
        cache.check_hicache_events()
        controller.land_all()
        cache.check_hicache_events()

        self.assertTrue(cache.tree_core.is_backuped(a))
        self.assertTrue(cache.tree_core.is_backuped(b))
        self.assertFalse(cache.tree_core.is_backuped(c))
        self.assertEqual(controller.host.used, 16)
        for node_id in (a, b):
            self.assertGreater(
                _node(cache, node_id).component_data[FULL].host_lock_ref, 0
            )

        # Once a victim leaves the device its copy is ordinary host content:
        # the next window copy takes its room.
        cache.evict(EvictParams(num_tokens=8))
        self.assertTrue(_node(cache, a).evicted)
        cache.check_hicache_events()
        self.assertNotIn(a, cache._write_back_copy_ahead._host_locks)
        self.assertTrue(cache.tree_core.is_backuped(c))
        self.assertEqual(controller.host.used, 16)

    def test_window_pins_exactly_the_copies_inside_it(self):
        """A copied node that is reused leaves the window while still on device;
        its host copy must become an ordinary duplicate again, or hot prefixes
        pin host capacity forever. Once the node ages back into the window its
        copy is pinned again, or host pressure reclaims it and it is recopied."""
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=8)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        cache.check_hicache_events()
        controller.land_all()
        cache.check_hicache_events()
        self.assertGreater(_node(cache, a).component_data[FULL].host_lock_ref, 0)

        # A request reuses a: its insert moves a behind b in eviction order.
        _insert(cache, allocator, [1] * 8)
        cache.check_hicache_events()

        self.assertFalse(_node(cache, a).evicted)
        self.assertTrue(cache.tree_core.is_backuped(a))
        self.assertEqual(_node(cache, a).component_data[FULL].host_lock_ref, 0)
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {b})

        controller.land_all()
        cache.check_hicache_events()
        cache.evict(EvictParams(num_tokens=8))
        self.assertTrue(_node(cache, b).evicted)
        cache.check_hicache_events()

        self.assertGreater(_node(cache, a).component_data[FULL].host_lock_ref, 0)
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, set())
        self.assertEqual(cache._write_back_copy_ahead.num_backups, 2)

    def test_host_pressure_skips_the_copy_and_leaves_the_victim_evictable(self):
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=8, copy_ahead_tokens=16)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)

        cache.check_hicache_events()

        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a})
        self.assertEqual(controller.host.used, 8)
        self.assertFalse(cache.tree_core.is_backuped(b))
        self.assertEqual(_node(cache, b).component_data[FULL].lock_ref, 0)
        self.assertIn(_node(cache, b), cache.tree_core.evictable_device_leaves)

    def test_synchronous_backup_waits_for_its_own_ack_not_the_window(self):
        """A victim past the window is backed up synchronously; that wait must
        not also block on window copies still in flight."""
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=8)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        cache.check_hicache_events()
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a})

        result = cache.evict(EvictParams(num_tokens=8))

        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertTrue(_node(cache, b).evicted)
        self.assertEqual(len(controller.waits), 1)
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a})
        self.assertEqual(len(controller.ack_write_queue), 1)
        self.assertEqual(controller.ack_write_queue[0].node_ids, [a])

    def test_barrier_backup_leaves_unrelated_window_copies_queued(self):
        """The barrier backup run before an internal-state tombstone waits for
        its own copy only; window copies elsewhere stay with the step's poll."""
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=8)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        cache.check_hicache_events()
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a})

        self.assertTrue(cache.backup_node_for_write_back(b))

        self.assertTrue(cache.tree_core.is_backuped(b))
        self.assertEqual(len(controller.waits), 1)
        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a})
        self.assertEqual([ack.node_ids for ack in controller.ack_write_queue], [[a]])

    def test_each_order_keeps_its_copies_in_flight_within_its_share(self):
        """Copies in flight are locked, so neither walk sees them again; each
        order must subtract its own, or every step before the acks land adds
        another share and the window overflows."""
        cache, allocator = _build_swa_cache(sliding_window_size=4)
        _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        leaves = [_insert(cache, allocator, [k] * 4) for k in range(1, 7)]
        cache.tree_core.has_swa_host_pool = True
        copy_ahead = cache._write_back_copy_ahead

        # Full-only copies: the fake controller has no SWA host pool.
        with (
            patch.object(
                cache.components[SWA], "build_hicache_transfers", return_value=None
            ),
            patch.object(copy_ahead, "_reserve_swa_host"),
        ):
            copy_ahead.refill()
            copy_ahead.refill()

        self.assertEqual(copy_ahead.pending_node_ids, set(leaves[:4]))
        self.assertEqual(copy_ahead._inflight(swa=False), 12)
        self.assertEqual(copy_ahead._inflight(swa=True), 4)

    def test_victim_fitting_beside_copies_in_flight_is_copied(self):
        """A copy still in flight takes its share of the window once, not twice:
        a victim that fits in what is left is copied."""
        cache, allocator = _build_cache()
        _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=8)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)
        cache.check_hicache_events()
        copy_ahead = cache._write_back_copy_ahead
        self.assertEqual(copy_ahead.pending_node_ids, {a})

        copy_ahead.copy_ahead_tokens = 16
        cache.check_hicache_events()

        self.assertEqual(copy_ahead.pending_node_ids, {a, b})

    def test_victim_crossing_the_window_end_is_not_copied(self):
        cache, allocator = _build_cache()
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=12)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [2] * 8)

        cache.check_hicache_events()

        self.assertEqual(cache._write_back_copy_ahead.pending_node_ids, {a})
        self.assertEqual(controller.host.used, 8)
        self.assertFalse(cache.tree_core.is_backuped(b))

    def test_swa_order_is_copied_ahead_even_when_full_victims_fill_the_window(self):
        """SWA-driven evictions get their own share of the window, and only an
        internal node whose tombstone would need a synchronous backup (SWA and
        Full KV both device-only) is copied for it."""
        cache, allocator = _build_swa_cache(sliding_window_size=4)
        _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        parent = _insert(cache, allocator, [1] * 8)
        _insert(cache, allocator, [1] * 8 + [2] * 8)
        _insert(cache, allocator, [3] * 8)
        cache.tree_core.has_swa_host_pool = True
        copy_ahead = cache._write_back_copy_ahead

        def copied_for_swa():
            # Only the SWA order's choices are recorded.
            with (
                patch.object(copy_ahead, "_cover_full"),
                patch.object(copy_ahead, "_keep_copied") as keep,
                patch.object(copy_ahead_module, "_SWA_WINDOW_DIVISOR", 1),
            ):
                copy_ahead.refill()
            return {call.args[0].id for call in keep.call_args_list}

        self.assertIn(parent, copied_for_swa())

        # The Full order leaves the SWA share free for copies in flight.
        with (
            patch.object(copy_ahead, "_cover_full") as cover_full,
            patch.object(copy_ahead, "_keep_copied"),
        ):
            copy_ahead.refill()
        self.assertEqual(cover_full.call_args.kwargs["budget"], 12)

        # Its SWA already on host: the tombstone needs no backup, so no copy.
        _node(cache, parent).component_data[SWA].host_value = torch.arange(8)
        self.assertNotIn(parent, copied_for_swa())

    def test_swa_order_copy_keeps_its_pin_after_it_lands(self):
        """Once the SWA order's copy of an internal node lands, the node is
        backed up but still inside the window; dropping its pin would let host
        pressure reclaim the copy and the node would be copied again."""
        cache, allocator = _build_swa_cache(sliding_window_size=4)
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=16)
        parent = _insert(cache, allocator, [1] * 8)
        _insert(cache, allocator, [1] * 8 + [2] * 8)
        cache.tree_core.has_swa_host_pool = True
        copy_ahead = cache._write_back_copy_ahead

        with (
            patch.object(copy_ahead, "_cover_full"),
            patch.object(copy_ahead_module, "_SWA_WINDOW_DIVISOR", 1),
            patch.object(
                cache.components[SWA], "build_hicache_transfers", return_value=None
            ),
            patch.object(copy_ahead, "_reserve_swa_host"),
        ):
            cache.check_hicache_events()
            self.assertIn(parent, copy_ahead.pending_node_ids)
            copied = copy_ahead.num_backups
            controller.land_all()
            cache.check_hicache_events()

        self.assertTrue(cache.tree_core.is_backuped(parent))
        self.assertGreater(_node(cache, parent).component_data[FULL].host_lock_ref, 0)
        self.assertEqual(copy_ahead.num_backups, copied)

    def test_window_pin_takes_no_swa_host_lock_on_the_path(self):
        """A window pin holds only the victim's Full host copy; its SWA window
        stays on the SWA host LRU."""
        cache, allocator = _build_swa_cache(sliding_window_size=16)
        controller = _attach_write_back(cache, host_capacity=64, copy_ahead_tokens=4)
        a = _insert(cache, allocator, [1] * 8)
        b = _insert(cache, allocator, [1] * 8 + [2] * 4)
        leaf = _node(cache, b)
        # Already on host, so the window pins it without a copy.
        leaf.component_data[FULL].host_value = controller.host.alloc(4)
        cache.tree_core._update_duplicate_tracking(leaf)
        copy_ahead = cache._write_back_copy_ahead

        copy_ahead.refill()

        self.assertEqual(copy_ahead.pending_node_ids, set())
        self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)
        for node_id in (a, b):
            self.assertEqual(_node(cache, node_id).component_data[SWA].host_lock_ref, 0)

        # Leaving the window releases the receipt it took.
        copy_ahead.copy_ahead_tokens = 0
        copy_ahead.refill()
        self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)
        cache.sanity_check()


class TestCopyAheadSelection(CustomTestCase):
    def test_copy_ahead_tokens_follow_the_env_var_then_four_chunks(self):
        with tpu_envs.SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.override(None):
            self.assertEqual(
                resolve_copy_ahead_tokens(chunked_prefill_size=2048, page_size=64), 8192
            )
            # Chunked prefill off: the stand-in chunk sizes the window.
            self.assertEqual(
                resolve_copy_ahead_tokens(chunked_prefill_size=-1, page_size=64), 16384
            )
        with tpu_envs.SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.override(10):
            self.assertEqual(
                resolve_copy_ahead_tokens(chunked_prefill_size=2048, page_size=4), 12
            )
        with tpu_envs.SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.override(0):
            self.assertEqual(
                resolve_copy_ahead_tokens(chunked_prefill_size=2048, page_size=4), 0
            )

    def _cache_with(self, controller, *, pp_size: int = 1) -> UnifiedRadixCache:
        cache, _ = _build_cache()
        cache.cache_controller = controller
        cache.pp_size = pp_size
        return cache

    def test_only_raiden_write_back_single_pipeline_caches_copy_ahead(self):
        with tpu_envs.SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.override(None):
            cache = self._cache_with(_FakeRaidenController(host_capacity=8))
            attach_copy_ahead(cache=cache, chunked_prefill_size=2048)
            self.assertIsInstance(cache._write_back_copy_ahead, WriteBackCopyAhead)
            self.assertEqual(cache._write_back_copy_ahead.copy_ahead_tokens, 8192)

            cache = self._cache_with(
                _FakeRaidenController(host_capacity=8, write_policy="write_through")
            )
            attach_copy_ahead(cache=cache, chunked_prefill_size=2048)
            self.assertIsNone(cache._write_back_copy_ahead)

            generic = object.__new__(HybridCacheController)
            generic.write_policy = "write_back"
            cache = self._cache_with(generic)
            attach_copy_ahead(cache=cache, chunked_prefill_size=2048)
            self.assertIsNone(cache._write_back_copy_ahead)

            cache = self._cache_with(_FakeRaidenController(host_capacity=8), pp_size=2)
            attach_copy_ahead(cache=cache, chunked_prefill_size=2048)
            self.assertIsNone(cache._write_back_copy_ahead)

        with tpu_envs.SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.override(0):
            cache = self._cache_with(_FakeRaidenController(host_capacity=8))
            attach_copy_ahead(cache=cache, chunked_prefill_size=2048)
            self.assertIsNone(cache._write_back_copy_ahead)


if __name__ == "__main__":
    unittest.main()
