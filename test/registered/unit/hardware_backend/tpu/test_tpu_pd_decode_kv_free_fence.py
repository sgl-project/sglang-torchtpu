"""On a PD decode server, KV arriving from prefill is written into
preallocated pages by a DMA that is not ordered against the in-flight forward,
so preallocation must not hand out pages that forward may still write."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode import alloc_for_decode_prealloc
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.hardware_backend.tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang.srt.hardware_backend.tpu.kv_free_fence import install_kv_free_fence
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4


def _tpu_allocator(num_pages=4):
    return TpuPagedTokenToKVPoolAllocator(
        size=num_pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


def _generic_allocator():
    return PagedTokenToKVPoolAllocator(
        size=4 * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


def _pages(locs):
    return set((locs[::PAGE] // PAGE).tolist())


class TestPreallocUnderFence(CustomTestCase):
    def setUp(self):
        self.log = []
        self.allocator = _tpu_allocator()
        self.fence = install_kv_free_fence(
            self.allocator,
            synchronize_device=lambda: self.log.append("device"),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            copies_draft_tail=False,
        )

    def _launch(self):
        self.fence.on_batch_formed()
        self.fence.on_forward_launched(SimpleNamespace(copy_done=None))

    def _prealloc(self, num_pages):
        return alloc_for_decode_prealloc(
            self.allocator,
            req=SimpleNamespace(kv=SimpleNamespace()),
            fill_len=num_pages * PAGE,
            delta_len=num_pages * PAGE,
            prefix_len=0,
            total_prefix_len=0,
            prefix_indices=None,
            uses_swa_tail=False,
            swa_tail_len=0,
        )

    def test_prealloc_waits_for_the_in_flight_forward_before_taking_its_pages(self):
        """A request that finished at N-1 is freed while forward N still writes
        its last slot; prefill's KV pulled into that page could be overwritten."""
        busy = self.allocator.alloc(2 * PAGE)
        self._launch()
        self.allocator.free(busy)

        free_first = self._prealloc(2)
        self.assertEqual(self.log, [])
        self.assertFalse(_pages(free_first) & _pages(busy))

        after_wait = self._prealloc(1)
        self.assertEqual(self.log, ["device"])
        self.assertLessEqual(_pages(after_wait), _pages(busy))


def _scheduler(**overrides):
    attrs = dict(
        device="tpu",
        enable_overlap=True,
        enable_hierarchical_cache=False,
        disaggregation_mode=DisaggregationMode.DECODE,
        kv_free_fence=None,
        device_module=SimpleNamespace(synchronize=lambda: None),
        token_to_kv_pool_allocator=_tpu_allocator(),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        tpu_async_copy_worker=SimpleNamespace(copy_draft_tail=True),
    )
    attrs.update(overrides)
    scheduler = SimpleNamespace(**attrs)
    for name in (
        "_uses_raiden_hicache",
        "_needs_kv_free_fence",
        "maybe_init_kv_free_fence",
    ):
        setattr(scheduler, name, MethodType(getattr(Scheduler, name), scheduler))
    return scheduler


class TestPdDecodeFenceInstall(CustomTestCase):
    def test_pd_decode_installs_the_fence_without_hicache(self):
        scheduler = _scheduler()
        scheduler.maybe_init_kv_free_fence()
        self.assertIsNotNone(scheduler.kv_free_fence)
        self.assertIs(
            scheduler.token_to_kv_pool_allocator.kv_free_fence, scheduler.kv_free_fence
        )

    def test_pd_decode_with_eagle_installs_the_fence_when_the_draft_tail_is_copied(
        self,
    ):
        scheduler = _scheduler(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            tpu_async_copy_worker=SimpleNamespace(copy_draft_tail=True),
        )
        scheduler.maybe_init_kv_free_fence()
        self.assertIsNotNone(scheduler.kv_free_fence)

    def test_no_fence_without_overlap_or_off_decode(self):
        for overrides in (
            dict(enable_overlap=False),
            dict(device="cuda"),
            dict(disaggregation_mode=DisaggregationMode.PREFILL),
            dict(disaggregation_mode=DisaggregationMode.NULL),
        ):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                scheduler = _scheduler(**overrides)
                scheduler.maybe_init_kv_free_fence()
                self.assertIsNone(scheduler.kv_free_fence)

    def test_unfenceable_pd_decode_warns_instead_of_failing_startup(self):
        """A PD decode server the fence cannot cover keeps serving, unfenced."""
        generic = _generic_allocator()
        for overrides, expected in (
            (dict(token_to_kv_pool_allocator=generic), "PagedTokenToKVPoolAllocator"),
            (dict(spec_algorithm=SpeculativeAlgorithm.DFLASH), "DFLASH"),
        ):
            with self.subTest(expected=expected):
                scheduler = _scheduler(**overrides)
                with self.assertLogs(
                    "sglang.srt.managers.scheduler", level="WARNING"
                ) as logs:
                    scheduler.maybe_init_kv_free_fence()
                self.assertIsNone(scheduler.kv_free_fence)
                self.assertIn(expected, logs.output[0])

    def test_unfenceable_raiden_hicache_still_fails_startup(self):
        """raiden HiCache must not fall back to unfenced serving as PD decode does."""
        generic = _generic_allocator()
        for overrides, mode in (
            (dict(token_to_kv_pool_allocator=generic), DisaggregationMode.NULL),
            (dict(spec_algorithm=SpeculativeAlgorithm.DFLASH), DisaggregationMode.NULL),
            (dict(token_to_kv_pool_allocator=generic), DisaggregationMode.DECODE),
            (
                dict(
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    tpu_async_copy_worker=None,
                ),
                DisaggregationMode.DECODE,
            ),
        ):
            with self.subTest(
                mode=str(mode), **{k: str(v) for k, v in overrides.items()}
            ):
                scheduler = _scheduler(disaggregation_mode=mode, **overrides)
                scheduler._uses_raiden_hicache = lambda: True
                with self.assertRaisesRegex(ValueError, "raiden HiCache"):
                    scheduler.maybe_init_kv_free_fence()
                self.assertIsNone(scheduler.kv_free_fence)


class TestHicacheBackupFence(CustomTestCase):
    def test_backups_are_fenced_on_servers_that_decode(self):
        controller = object()
        for mode, fenced in (
            (DisaggregationMode.NULL, True),
            (DisaggregationMode.DECODE, True),
            (DisaggregationMode.PREFILL, False),
        ):
            with self.subTest(mode=str(mode)):
                scheduler = _scheduler(
                    disaggregation_mode=mode,
                    tree_cache=SimpleNamespace(cache_controller=controller),
                )
                scheduler._uses_raiden_hicache = lambda: True
                with patch(
                    "sglang.srt.hardware_backend.tpu.kv_free_fence.fence_hicache_backups"
                ) as fence_backups:
                    scheduler.maybe_init_kv_free_fence()
                self.assertIsNotNone(scheduler.kv_free_fence)
                if fenced:
                    fence_backups.assert_called_once_with(
                        controller, scheduler.kv_free_fence
                    )
                else:
                    fence_backups.assert_not_called()


if __name__ == "__main__":
    unittest.main()
