"""Host staging for TPU pipeline expert loading."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu import expert_loading
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _param():
    return torch.nn.Parameter(torch.zeros(3, 4, 2), requires_grad=False)


class TestExpertWriteTracker(CustomTestCase):
    def test_interleaved_shards_complete_in_any_expert_order(self):
        tracker = expert_loading.ExpertWriteTracker()
        param = _param()
        writes = [(2, "w3"), (0, "w1"), (1, "w3"), (2, "w1"), (0, "w3")]
        for expert, shard in writes:
            self.assertFalse(tracker.record(param, expert, shard))
        self.assertTrue(tracker.record(param, 1, "w1"))

    def test_repeated_shards_do_not_complete_early(self):
        tracker = expert_loading.ExpertWriteTracker()
        param = _param()
        for _ in range(4):
            self.assertFalse(tracker.record(param, 0, "w2"))
        self.assertFalse(tracker.record(param, 2, "w2"))
        self.assertTrue(tracker.record(param, 1, "w2"))

    def test_unknown_and_whole_expert_writes_do_not_complete(self):
        tracker = expert_loading.ExpertWriteTracker()
        param = _param()
        for expert, shard in [(0, None), (None, "w2"), (0, "other")]:
            self.assertFalse(tracker.record(param, expert, shard))

    def test_parameters_track_independently(self):
        tracker = expert_loading.ExpertWriteTracker()
        first, second = _param(), _param()
        for expert in (0, 1):
            self.assertFalse(tracker.record(first, expert, "w2"))
            self.assertFalse(tracker.record(second, expert, "w2"))
        self.assertTrue(tracker.record(second, 2, "w2"))
        self.assertTrue(tracker.record(first, 2, "w2"))


class TestExpertParamStager(CustomTestCase):
    def test_upload_on_last_expert_preserves_parameter_and_loader_metadata(self):
        param = _param()
        loader = object()
        param.weight_loader = loader
        stager = expert_loading.ExpertParamStager()
        host = stager.host_param(param)
        self.assertIs(type(host), type(param))
        self.assertIs(host.weight_loader, loader)
        for expert in (2, 0, 1):
            host.data[expert].fill_(expert + 1)
            stager.written(param, expert, "w2")
        self.assertIs(param.weight_loader, loader)
        for expert in range(3):
            self.assertTrue(torch.all(param[expert] == expert + 1))
        self.assertFalse(stager.staging(param))

    def test_reusing_host_buffer_does_not_overwrite_uploaded_parameter(self):
        stager = expert_loading.ExpertParamStager()
        first, second = _param(), _param()
        host = stager.host_param(first)
        host.data.fill_(7)
        stager.flush(first)
        self.assertEqual(stager.host_param(second).data_ptr(), host.data_ptr())
        stager.host_param(second).data.fill_(9)
        stager.flush(second)
        self.assertTrue(torch.all(first == 7))
        self.assertTrue(torch.all(second == 9))

    def test_upload_chunks_preserve_expert_order_and_uneven_last_chunk(self):
        stager = expert_loading.ExpertParamStager()
        param = _param()
        expected = torch.arange(param.numel()).reshape(param.shape).float()
        stager.host_param(param).data.copy_(expected)
        with patch.object(expert_loading, "_UPLOAD_CHUNK_BYTES", 64):
            stager.flush(param)
        torch.testing.assert_close(param, expected)

    def test_early_upload_keeps_later_direct_writes(self):
        stager = expert_loading.ExpertParamStager()
        first, second = _param(), _param()
        with patch.object(expert_loading, "_MAX_STAGED", 1):
            stager.host_param(first).data.fill_(2)
            stager.host_param(second).data.fill_(3)
        self.assertFalse(stager.staging(first))
        first.data[2].fill_(5)
        stager.flush_all()
        self.assertTrue(torch.all(first[2] == 5))
        self.assertTrue(torch.all(second == 3))

    def test_context_flushes_partial_parameters_and_restores_outer_context(self):
        self.assertIsNone(expert_loading.get_expert_stager())
        param = _param()
        with expert_loading.expert_loading_context() as outer:
            self.assertIs(expert_loading.get_expert_stager(), outer)
            with expert_loading.expert_loading_context() as inner:
                inner.host_param(param).data.fill_(6)
            self.assertIs(expert_loading.get_expert_stager(), outer)
            self.assertTrue(torch.all(param == 6))
        self.assertIsNone(expert_loading.get_expert_stager())

    def test_exception_clears_context_without_uploading_partial_weights(self):
        param = _param()
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with expert_loading.expert_loading_context() as stager:
                stager.host_param(param).data.fill_(8)
                raise RuntimeError("load failed")
        self.assertIsNone(expert_loading.get_expert_stager())
        self.assertTrue(torch.all(param == 0))


if __name__ == "__main__":
    unittest.main()
