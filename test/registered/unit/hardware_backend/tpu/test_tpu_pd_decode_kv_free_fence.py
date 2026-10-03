"""On a PD decode server, KV arriving from prefill is written into
preallocated pages by a DMA that is not ordered against the in-flight forward,
so preallocation must not hand out pages that forward may still write."""

import types
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
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
        return DecodePreallocQueue._alloc_for_decode_prealloc(
            SimpleNamespace(),
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
        busy = self.allocator.alloc(4 * PAGE)
        self._launch()
        self.allocator.free(busy)

        kv_loc = self._prealloc(2)
        self.assertEqual(self.log, ["device"])
        self.assertLessEqual(_pages(kv_loc), _pages(busy))


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
        setattr(scheduler, name, types.MethodType(getattr(Scheduler, name), scheduler))
    return scheduler


class TestPdDecodeFenceInstall(CustomTestCase):
    def test_pd_decode_installs_the_fence_without_hicache(self):
        scheduler = _scheduler()
        scheduler.maybe_init_kv_free_fence()
        self.assertIsNotNone(scheduler.kv_free_fence)
        self.assertIs(
            scheduler.token_to_kv_pool_allocator.kv_free_fence, scheduler.kv_free_fence
        )

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
        generic = PagedTokenToKVPoolAllocator(
            size=4 * PAGE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
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


if __name__ == "__main__":
    unittest.main()
