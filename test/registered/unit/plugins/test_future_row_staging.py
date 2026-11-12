"""CPU contracts for registered sampling placement and published future rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.host_padding import HostSamplingAlgorithm
from sglang_tpu.overlap_utils_tpu import ReindexableFutureRows

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _PublishedRows(ReindexableFutureRows):
    def __init__(self, published=True):
        self.published = published
        self.future_indices = None

    def can_reindex_future_rows(self):
        return self.published

    def reindex_future_rows(self, future_indices):
        self.future_indices = future_indices


class TestFutureRowComposition(CustomTestCase):
    def _batch(self, rows):
        return SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            spec_info=_PublishedRows(),
            enable_overlap=True,
            host_pad_bs=None,
            input_ids=None,
            is_extend_in_batch=False,
            req_pool_indices_cpu=torch.tensor(rows),
            req_pool_indices=torch.tensor([-1] * len(rows)),
            device="cpu",
            seq_lens=torch.tensor([10] * len(rows)),
            orig_seq_lens=torch.tensor([10] * len(rows)),
        )

    def test_filter_uses_published_request_rows_without_device_indexing(self):
        from sglang_tpu.hooks.experimental_spec import (
            filter_model_inputs,
            filter_spec_input,
        )

        # ScheduleBatch has already filtered its request-owned host rows.
        batch = self._batch([5, 1])
        original = MagicMock(side_effect=AssertionError("device filter called"))
        device_indices = object()
        filter_model_inputs(original, batch, [2, 0], device_indices)
        filter_spec_input(original, batch, [2, 0], device_indices)
        self.assertEqual(batch.req_pool_indices.tolist(), [5, 1])
        self.assertIs(batch.spec_info.future_indices, batch.req_pool_indices)
        self.assertIsNone(batch.seq_lens)
        self.assertIsNone(batch.orig_seq_lens)
        original.assert_not_called()

    def test_merge_reindexes_only_when_both_inputs_are_published(self):
        from sglang_tpu.hooks.experimental_spec import (
            merge_model_inputs,
            merge_spec_input,
        )

        for published in (False, True):
            with self.subTest(incoming_published=published):
                batch, incoming = self._batch([5, 1, 3]), self._batch([3])
                incoming.spec_info.published = published
                original = MagicMock()
                merge_model_inputs(original, batch, incoming)
                merge_spec_input(original, batch, incoming)
                if published:
                    original.assert_not_called()
                    self.assertEqual(batch.req_pool_indices.tolist(), [5, 1, 3])
                    self.assertIs(
                        batch.spec_info.future_indices, batch.req_pool_indices
                    )
                    self.assertIsNone(batch.seq_lens)
                else:
                    self.assertEqual(original.call_count, 2)
                    self.assertIsNone(batch.spec_info.future_indices)
                    self.assertIsNotNone(batch.seq_lens)

    def test_unpublished_extend_and_non_overlap_inputs_keep_core_composition(self):
        from sglang_tpu.hooks.experimental_spec import (
            filter_model_inputs,
            filter_spec_input,
        )

        for case in ("unpublished", "extend", "non_overlap", "direct_tokens"):
            with self.subTest(case=case):
                batch = self._batch([5, 1])
                if case == "unpublished":
                    batch.spec_info.published = False
                elif case == "extend":
                    batch.is_extend_in_batch = True
                elif case == "non_overlap":
                    batch.enable_overlap = False
                else:
                    batch.input_ids = torch.tensor([9, 11])
                original = MagicMock()
                device_indices = object()
                filter_model_inputs(original, batch, [2, 0], device_indices)
                filter_spec_input(original, batch, [2, 0], device_indices)
                self.assertEqual(original.call_count, 2)
                original.assert_called_with(batch, [2, 0], device_indices)
                self.assertIsNone(batch.spec_info.future_indices)
                self.assertIsNotNone(batch.seq_lens)

    def test_host_sampling_requires_a_registered_algorithm_and_tpu_sampler(self):
        from sglang_tpu.hooks.host_padding import sampling_from_batch

        class RegisteredAlgorithm:
            def is_none(self):
                return False

        HostSamplingAlgorithm.register(RegisteredAlgorithm)
        for algorithm, registered in (
            (RegisteredAlgorithm(), True),
            (SpeculativeAlgorithm.EAGLE3, False),
        ):
            for backend in ("tpu", "pytorch"):
                for explicit_host in (False, True):
                    with self.subTest(
                        registered=registered,
                        backend=backend,
                        explicit_host=explicit_host,
                    ):
                        batch = SimpleNamespace(spec_algorithm=algorithm)
                        execution = SimpleNamespace(
                            kernel=SimpleNamespace(sampling_backend=backend)
                        )
                        with patch(
                            "sglang_tpu.hooks.host_padding.get_exec",
                            return_value=execution,
                        ):
                            original = MagicMock()
                            sampling_from_batch(
                                original,
                                SamplingBatchInfo,
                                batch,
                                32,
                                sampling_params_on_cpu=explicit_host,
                            )
                        original.assert_called_once_with(
                            SamplingBatchInfo,
                            batch,
                            32,
                            sampling_params_on_cpu=explicit_host
                            or (registered and backend == "tpu"),
                        )


if __name__ == "__main__":
    unittest.main()
