"""CPU coverage for rebuilding TPU batch rows from their host copies."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hardware_backend.tpu import schedule_batch_tpu  # noqa: E402
from sglang.srt.managers import schedule_batch  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid):
    return SimpleNamespace(
        rid=rid,
        return_logprob=False,
        grammar=None,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
    )


def _batch(rids, seq_lens, *, device_seq_lens=None, spec=SpeculativeAlgorithm.NONE):
    """A batch whose device seq_lens may differ from its host copy."""
    n = len(rids)
    batch = ScheduleBatch(reqs=[_req(rid) for rid in rids])
    batch.spec_algorithm = spec
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.sampling_info = MagicMock()
    batch.req_pool_indices_cpu = torch.arange(n, dtype=torch.int64) + 100
    batch.req_pool_indices = batch.req_pool_indices_cpu.clone()
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    batch.seq_lens = torch.tensor(
        device_seq_lens if device_seq_lens is not None else seq_lens,
        dtype=torch.int64,
    )
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    return batch


def _tpu_rows(*, tpu_device=True):
    """Route ScheduleBatch through the TPU helpers on a CPU host."""
    stack = ExitStack()
    stack.enter_context(
        patch.multiple(
            schedule_batch,
            _is_tpu=True,
            rows_on_host_tpu=schedule_batch_tpu.rows_on_host_tpu,
            upload_rows_tpu=schedule_batch_tpu.upload_rows_tpu,
            create=True,
        )
    )
    if tpu_device:
        stack.enter_context(
            patch.object(schedule_batch_tpu, "_is_tpu_device", lambda device: True)
        )
    return stack


def _on_device(batch, device):
    batch.device = device
    return batch


class TestRowsOnHost(CustomTestCase):
    def test_requires_tpu_device_non_spec_and_host_seq_lens(self):
        rows_on_host = schedule_batch_tpu.rows_on_host_tpu
        self.assertTrue(rows_on_host(_on_device(_batch(["a"], [5]), "tpu")))
        self.assertTrue(rows_on_host(_on_device(_batch(["a"], [5]), "tpu:0")))
        self.assertFalse(rows_on_host(_batch(["a"], [5])))
        eagle = _batch(["a"], [5], spec=SpeculativeAlgorithm.EAGLE)
        self.assertFalse(rows_on_host(_on_device(eagle, "tpu")))
        no_host = _on_device(_batch(["a"], [5]), "tpu")
        no_host.seq_lens_cpu = None
        self.assertFalse(rows_on_host(no_host))
        self.assertFalse(rows_on_host(_on_device(_batch(["a"], [5]), "tpu"), no_host))

    def test_upload_rows(self):
        req_pool_indices, seq_lens, orig_seq_lens = schedule_batch_tpu.upload_rows_tpu(
            req_pool_indices_cpu=torch.tensor([3, 1]),
            seq_lens_cpu=torch.tensor([10, 20]),
            device="cpu",
        )
        self.assertEqual(req_pool_indices.tolist(), [3, 1])
        self.assertEqual(seq_lens.tolist(), [10, 20])
        self.assertEqual(orig_seq_lens.dtype, torch.int32)
        self.assertEqual(orig_seq_lens.tolist(), [10, 20])


class TestFilterBatchTpu(CustomTestCase):
    def test_host_rows_and_sampling_reqs(self):
        batch = _batch(["a", "b", "c"], [10, 20, 30], device_seq_lens=[11, 21, 31])
        with _tpu_rows():
            batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.req_pool_indices.tolist(), [100, 102])
        self.assertEqual(batch.seq_lens.tolist(), [10, 30])
        self.assertEqual(batch.orig_seq_lens.tolist(), [10, 30])
        kwargs = batch.sampling_info.filter_batch.call_args.kwargs
        self.assertEqual([r.rid for r in kwargs["reqs"]], ["a", "c"])

    def test_non_tpu_device_keeps_device_rows(self):
        batch = _batch(["a", "b", "c"], [10, 20, 30], device_seq_lens=[11, 21, 31])
        with _tpu_rows(tpu_device=False):
            batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.seq_lens.tolist(), [11, 31])
        kwargs = batch.sampling_info.filter_batch.call_args.kwargs
        self.assertIsNone(kwargs["reqs"])

    def test_spec_keeps_device_rows(self):
        batch = _batch(
            ["a", "b", "c"],
            [10, 20, 30],
            device_seq_lens=[11, 21, 31],
            spec=SpeculativeAlgorithm.EAGLE,
        )
        with _tpu_rows():
            batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.seq_lens.tolist(), [11, 31])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [10, 30])
        kwargs = batch.sampling_info.filter_batch.call_args.kwargs
        self.assertIsNone(kwargs["reqs"])


class TestMergeBatchTpu(CustomTestCase):
    def test_host_rows_and_sampling_reqs(self):
        batch = _batch(["a", "b"], [10, 20], device_seq_lens=[11, 21])
        other = _batch(["c"], [30], device_seq_lens=[31])
        with _tpu_rows():
            batch.merge_batch(other)
        self.assertEqual(batch.req_pool_indices.tolist(), [100, 101, 100])
        self.assertEqual(batch.seq_lens.tolist(), [10, 20, 30])
        self.assertEqual(batch.orig_seq_lens.tolist(), [10, 20, 30])
        kwargs = batch.sampling_info.merge_batch.call_args.kwargs
        self.assertEqual([r.rid for r in kwargs["reqs"]], ["a", "b", "c"])

    def test_missing_host_copy_keeps_device_rows(self):
        batch = _batch(["a", "b"], [10, 20], device_seq_lens=[11, 21])
        other = _batch(["c"], [30], device_seq_lens=[31])
        other.seq_lens_cpu = None
        with _tpu_rows():
            batch.merge_batch(other)
        self.assertEqual(batch.seq_lens.tolist(), [11, 21, 31])
        self.assertIsNone(batch.seq_lens_cpu)
        kwargs = batch.sampling_info.merge_batch.call_args.kwargs
        self.assertIsNone(kwargs["reqs"])


if __name__ == "__main__":
    unittest.main()
