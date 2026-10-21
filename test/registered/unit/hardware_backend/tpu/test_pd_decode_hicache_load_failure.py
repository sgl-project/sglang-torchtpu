"""A failed PD decode HiCache load-back aborts its requests and rolls back its nodes.

Covers the restore poll marking FAILED, holding a failed load until only it
locks its nodes, the deepest-first rollback (and its refusal for a node still
in use), and the loading_check flow around upstream's ack. Runs on CPU with no
TPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_decode_hicache_load_failure as lf
from sglang_tpu.raiden_hybrid_cache_controller import TpuRaidenHybridCacheController
from sglang_tpu.raiden_transfer import LoadBackEvent


class _Done:
    def __init__(self, error=None):
        self.error = error

    def query(self):
        if self.error is not None:
            raise self.error
        return True

    def synchronize(self):
        if self.error is not None:
            raise self.error


def _event(failed: bool) -> LoadBackEvent:
    event = LoadBackEvent(
        [_Done(RuntimeError("raiden DMA failed: boom") if failed else None)]
    )
    event.landed()
    return event


class _Binding:
    """The Rust core's inspection API over a scripted tree."""

    def __init__(self, lock_refs, evictable_leaves):
        self.lock_refs = lock_refs
        self.evictable_leaves = set(evictable_leaves)

    def inspect_get_component_device_lock_ref(self, node_id, component_type):
        return self.lock_refs[node_id]

    def inspect_is_device_evictable_leaf(self, node_id):
        return node_id in self.evictable_leaves


def _controller(acks):
    controller = object.__new__(TpuRaidenHybridCacheController)
    controller.ack_load_queue = list(acks)
    controller.handles_load_back_failures = True
    controller.layer_done_counter = SimpleNamespace(events=[None, None])
    return controller


class _Cache:
    def __init__(self, acks, lock_refs=None, evictable=(), loaded=None):
        self.cache_controller = _controller(acks)
        self.tree_core = SimpleNamespace(_binding=_Binding(lock_refs or {}, evictable))
        if loaded:
            lf._loaded_nodes(self.tree_core).update(loaded)
        self.ongoing_load_back = {}
        self.demoted = []

    @staticmethod
    def _all_reduce(tensor, op):
        pass

    def _demote(self, node_id, tracker):
        self.demoted.append(node_id)
        self.tree_core._binding.evictable_leaves.add(node_id - 1)  # parent next


def _ack(anchor, failed):
    return SimpleNamespace(node_ids=[anchor], finish_event=_event(failed))


def _dr(index, status=HiCacheRestoreResult.READY):
    return SimpleNamespace(
        hicache_load_consumer_index=index, hicache_restore_status=status
    )


class TestFailRestores(CustomTestCase):
    def test_restore_of_a_failed_load_is_failed_not_ready(self):
        failed, ok = _event(True), _event(False)
        failed.failed, ok.failed = True, False
        cache = _Cache([])
        cache.cache_controller.layer_done_counter.events[:] = [failed, ok]
        dr_failed, dr_ok = _dr(0), _dr(1)
        lf._fail_restores_of_failed_loads(
            SimpleNamespace(tree_cache=cache), [dr_failed, dr_ok]
        )
        self.assertEqual(dr_failed.hicache_restore_status, HiCacheRestoreResult.FAILED)
        self.assertEqual(dr_ok.hicache_restore_status, HiCacheRestoreResult.READY)

    def test_leaves_restores_without_a_load_and_pending_ones(self):
        failed = _event(True)
        failed.failed = True
        cache = _Cache([])
        cache.cache_controller.layer_done_counter.events[:] = [failed, None]
        no_load = _dr(-1)
        pending = _dr(0, status=HiCacheRestoreResult.PENDING)
        lf._fail_restores_of_failed_loads(
            SimpleNamespace(tree_cache=cache), [no_load, pending]
        )
        self.assertEqual(no_load.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(pending.hicache_restore_status, HiCacheRestoreResult.PENDING)


class TestReleaseFailedLoads(CustomTestCase):
    def test_failed_load_is_released_once_only_it_holds_its_nodes(self):
        ack = _ack(anchor=8, failed=True)
        cache = _Cache([ack], lock_refs={7: 1, 8: 1}, loaded={8: [7, 8]})
        lf._release_failed_loads(cache, cache.cache_controller)
        self.assertTrue(ack.finish_event.failed)
        self.assertTrue(ack.finish_event.released)

    def test_failed_load_waits_while_a_request_holds_a_node(self):
        failed_ack, next_ack = _ack(anchor=8, failed=True), _ack(anchor=9, failed=False)
        cache = _Cache(
            [failed_ack, next_ack], lock_refs={7: 1, 8: 2}, loaded={8: [7, 8]}
        )
        lf._release_failed_loads(cache, cache.cache_controller)
        self.assertFalse(failed_ack.finish_event.released)
        self.assertFalse(failed_ack.finish_event.query())
        # Acks are acked in order, so the walk stops at the held one.
        self.assertIsNone(next_ack.finish_event.failed)


class TestRollBack(CustomTestCase):
    def test_demotes_deepest_first(self):
        cache = _Cache([], evictable={8})
        lf._roll_back(cache, anchor=8, nodes=[7, 8])
        self.assertEqual(cache.demoted, [8, 7])

    def test_refuses_a_node_still_in_use(self):
        cache = _Cache([], evictable=set())
        with self.assertRaisesRegex(RuntimeError, "still in use"):
            lf._roll_back(cache, anchor=8, nodes=[7, 8])
        self.assertEqual(cache.demoted, [])


class TestLoadingCheckFlow(CustomTestCase):
    def test_failed_load_is_acked_then_rolled_back(self):
        failed_ack, ok_ack = _ack(anchor=8, failed=True), _ack(anchor=20, failed=False)
        cache = _Cache(
            [failed_ack, ok_ack],
            lock_refs={7: 1, 8: 1},
            evictable={8},
            loaded={8: [7, 8], 20: [20], 99: [99]},
        )

        def upstream(c, finish_count):
            # Upstream acks every load it counts ready, from the head.
            while c.cache_controller.ack_load_queue:
                if not c.cache_controller.ack_load_queue[0].finish_event.query():
                    break
                c.cache_controller.ack_load_queue.pop(0)

        lf._loading_check_with_cleanup(upstream, cache, None)
        self.assertEqual(cache.cache_controller.ack_load_queue, [])
        self.assertEqual(cache.demoted, [8, 7])
        # Records of acked loads, and of loads that never started, are dropped.
        self.assertEqual(lf._loaded_nodes(cache.tree_core), {})


class TestInstall(CustomTestCase):
    def test_install_patches_once(self):
        saved = (
            DecodeHiCacheTransferMixin._process_hicache_local_restores,
            UnifiedRadixCache.loading_check,
        )
        try:
            # The Rust tree core's recording hook is optional (its extension may
            # be unbuilt); the restore poll and loading_check are always patched.
            with (
                mock.patch.object(lf, "_is_installed", False),
                mock.patch.object(lf, "_record_load_back_nodes") as record,
            ):
                lf.install()
                patched = UnifiedRadixCache.loading_check
                self.assertIsNot(patched, saved[1])
                self.assertIsNot(
                    DecodeHiCacheTransferMixin._process_hicache_local_restores,
                    saved[0],
                )
                lf.install()
                self.assertIs(UnifiedRadixCache.loading_check, patched)
                record.assert_called_once()
        finally:
            (
                DecodeHiCacheTransferMixin._process_hicache_local_restores,
                UnifiedRadixCache.loading_check,
            ) = saved


if __name__ == "__main__":
    unittest.main()
