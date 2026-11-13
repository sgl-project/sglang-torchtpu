"""Decode-side P/D state on TPU that lives on the host.

Three behaviours the TPU decode engine relies on, none of them tied to a
transfer backend:

1. ``ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt`` pads the
   prebuilt batch on the host (``out_cache_loc`` stays on CPU) when running on
   TPU, and still builds it on the generic path.
2. ``TpuPagedTokenToKVPoolAllocator.free`` accepts CPU index tensors and moves
   anything else to the host first.
3. ``DecodeReqToTokenPool`` keeps ``req_to_token`` host resident for a TPU
   device.

These were the connector-independent classes of the retired
``test_disagg_tpu_transfer.py`` and moved here when the ``tpu`` (ZMQ) transfer
backend was removed. Pure CPU tensors; no TPU needed.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation import decode_schedule_batch_mixin
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.managers.schedule_batch import Range
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase


class TestPrepareForPrebuiltTpuPadding(CustomTestCase):
    def test_prepare_for_prebuilt_padding(self):
        class MockScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
            def metadata_device(self):
                return self.device

            def __init__(self):
                self.device = torch.device("cpu")
                self.forward_mode = None
                self.return_logprob = False
                self.model_config = SimpleNamespace(vocab_size=32000)

                req = SimpleNamespace()
                # prepare_for_prebuilt reads the pool row off the request's KV state.
                req.kv = SimpleNamespace(req_pool_idx=1)
                req.prefix_indices = list(range(10))
                req.get_fill_ids = lambda: array("q", range(110))
                req.extend_range = Range(10, 110)  # length = 100
                req.origin_input_ids = list(range(110))
                req.output_ids = []
                req.retracted_stain = False
                req.cached_tokens = 0
                req.cached_tokens_device = 0
                req.already_computed = 0
                req.is_retracted = False
                req.multimodal_inputs = None
                req.sampling_params = SamplingParams()
                # Read by SamplingBatchInfo.from_schedule_batch (Req default).
                req.return_sampling_mask = False

                self.reqs = [req]
                self.req_to_token_pool = SimpleNamespace(
                    req_to_token=torch.zeros((10, 500), dtype=torch.int32)
                )
                self.tree_cache = None

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        batch = MockScheduleBatch()
        orig_is_tpu = decode_schedule_batch_mixin._is_tpu
        try:
            decode_schedule_batch_mixin._is_tpu = True
            batch.prepare_for_prebuilt()
            self.assertIsNotNone(batch.out_cache_loc)
            self.assertEqual(batch.out_cache_loc.shape[0], 100)
            self.assertEqual(batch.out_cache_loc.device.type, "cpu")

            # Test non-TPU path
            decode_schedule_batch_mixin._is_tpu = False
            batch.prepare_for_prebuilt()
            self.assertIsNotNone(batch.out_cache_loc)
            self.assertEqual(batch.out_cache_loc.shape[0], 100)
        finally:
            decode_schedule_batch_mixin._is_tpu = orig_is_tpu


class TestTpuPagedTokenToKVPoolAllocatorCpuAssertion(CustomTestCase):
    def test_free_cpu_tensor_succeeds(self):
        from sglang_tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        allocator = TpuPagedTokenToKVPoolAllocator(
            size=128,
            page_size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator.clear()
        # Hand out pages 1 and 2, which the frees below release.
        allocator.alloc(2 * 16)

        # Freeing CPU tensor should succeed
        cpu_index = torch.tensor([16, 17, 18], dtype=torch.int32, device="cpu")
        allocator.free(cpu_index)

        # Free group mode on CPU
        allocator.free_group_begin()
        allocator.free(torch.tensor([32, 33], dtype=torch.int32, device="cpu"))
        allocator.free_group_end()

    def test_free_non_cpu_tensor_converts_to_cpu(self):
        from sglang_tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        allocator = TpuPagedTokenToKVPoolAllocator(
            size=128,
            page_size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator.clear()
        # Hand out page 1, which the free below releases.
        allocator.alloc(16)

        class FakeDeviceTensor:
            device = SimpleNamespace(type="cuda")

            def cpu(self):
                return torch.tensor([16, 17, 18], dtype=torch.int32, device="cpu")

        fake_dev_tensor = FakeDeviceTensor()
        allocator.free(fake_dev_tensor)


class TestDecodeReqToTokenPoolTpuHostResident(CustomTestCase):
    def test_tpu_host_resident(self):
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        pool = DecodeReqToTokenPool(
            size=16,
            max_context_len=512,
            device="tpu:0",
            enable_memory_saver=False,
            pre_alloc_size=16,
        )
        self.assertTrue(pool.host_resident)
        self.assertEqual(pool.req_to_token.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
