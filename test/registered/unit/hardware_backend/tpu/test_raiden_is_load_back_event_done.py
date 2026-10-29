"""Unit tests for raiden ``is_load_back_event_done`` in the PD decode restore loop.

Covers the hardware-agnostic contracts with fake futures (no TPU, no raiden):
a ``BatchDoneCounter`` slot holds the load's future-event itself (or ``None``
before first use), every rank agrees on completion before the cache runs its
collective ``loading_check``, and the TPU patch routes only raiden-backed
unified caches through that gate.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.tpu_dma_hybrid_cache_controller import (
    TpuDMAHybridCacheController,
)


class _FakeFuture:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self) -> bool:
        return self.ready


def _all_reduce_as(remote_done: int):
    """An all_reduce stub where the other ranks report ``remote_done``."""

    def all_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        tensor.fill_(min(int(tensor.item()), remote_done))

    return MagicMock(side_effect=all_reduce)


def _make_controller() -> TpuDMAHybridCacheController:
    controller = object.__new__(TpuDMAHybridCacheController)
    controller.layer_done_counter = BatchDoneCounter()
    return controller


class TestRaidenControllerLoadBackEventDone(CustomTestCase):
    def test_unused_slot_is_ready(self):
        controller = _make_controller()
        all_reduce = _all_reduce_as(1)
        self.assertTrue(controller.is_load_back_event_done(0, all_reduce))
        all_reduce.assert_called_once()

    def test_no_load_skips_the_collective(self):
        controller = _make_controller()
        all_reduce = _all_reduce_as(1)
        self.assertTrue(controller.is_load_back_event_done(-1, all_reduce))
        all_reduce.assert_not_called()

    def test_local_completion_waits_for_other_ranks(self):
        controller = _make_controller()
        controller.layer_done_counter.set_event(1, _FakeFuture(ready=True))
        self.assertFalse(controller.is_load_back_event_done(1, _all_reduce_as(0)))
        self.assertTrue(controller.is_load_back_event_done(1, _all_reduce_as(1)))

    def test_pending_load_is_not_ready(self):
        controller = _make_controller()
        controller.layer_done_counter.set_event(0, _FakeFuture(ready=False))
        self.assertFalse(controller.is_load_back_event_done(0, _all_reduce_as(1)))


class TestRaidenCacheLoadBackEventDonePatch(CustomTestCase):
    def setUp(self):
        original = UnifiedRadixCache.is_load_back_event_done
        self.original = original

        def restore():
            UnifiedRadixCache.is_load_back_event_done = original
            patches._raiden_is_load_back_event_done_patched = False

        self.addCleanup(restore)
        patches._raiden_is_load_back_event_done_patched = False
        patches._patch_raiden_is_load_back_event_done()

    def _make_cache(self, controller) -> UnifiedRadixCache:
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        cache.loading_check = MagicMock()
        return cache

    def test_the_answer_waits_for_every_rank_and_runs_no_bookkeeping(self):
        controller = _make_controller()
        future = _FakeFuture(ready=False)
        controller.layer_done_counter.set_event(0, future)
        cache = self._make_cache(controller)

        cache._all_reduce = _all_reduce_as(1)
        self.assertFalse(cache.is_load_back_event_done(0))
        future.ready = True
        cache._all_reduce = _all_reduce_as(0)
        self.assertFalse(cache.is_load_back_event_done(0))

        cache._all_reduce = _all_reduce_as(1)
        self.assertTrue(cache.is_load_back_event_done(0))
        # The poll only answers; loading_check runs from check_hicache_events.
        cache.loading_check.assert_not_called()

    def test_unused_slot_skips_bookkeeping_for_no_load(self):
        cache = self._make_cache(_make_controller())
        cache._all_reduce = _all_reduce_as(1)
        self.assertTrue(cache.is_load_back_event_done(-1))
        cache._all_reduce.assert_not_called()
        cache.loading_check.assert_not_called()

    def test_other_controllers_keep_the_original_method(self):
        finish_event = _FakeFuture(ready=True)
        counter = SimpleNamespace(events=[SimpleNamespace(finish_event=finish_event)])
        cache = self._make_cache(SimpleNamespace(layer_done_counter=counter))
        cache._all_reduce = MagicMock()
        self.assertTrue(cache.is_load_back_event_done(0))
        cache._all_reduce.assert_not_called()
        cache.loading_check.assert_not_called()

    def test_patch_is_installed_once(self):
        patched = UnifiedRadixCache.is_load_back_event_done
        patches._patch_raiden_is_load_back_event_done()
        self.assertIs(UnifiedRadixCache.is_load_back_event_done, patched)
        self.assertIsNot(patched, self.original)


if __name__ == "__main__":
    unittest.main()
