"""Mixed prefill + verify steps select a fixed-size logits row set (CPU)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prune(metadata, hidden, aux=None):
    processor = object.__new__(LogitsProcessor)
    return LogitsProcessor._get_pruned_states(processor, hidden, None, aux, metadata)


class TestMixedLogitRows(CustomTestCase):
    def test_mixed_selects_the_given_rows_and_keeps_aux_aligned(self):
        """Two prompt requests (rows 0-4, 5-6) and one verify request (rows 7-12):
        each prompt request's last row, then the six verify rows, padded to L=10."""
        hidden = torch.arange(16, dtype=torch.float32)[:, None].repeat(1, 3)
        index = torch.tensor([4, 6, 7, 8, 9, 10, 11, 12, 12, 12])
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.MIXED,
            extend_seq_lens=torch.tensor([5, 2, 6]),
            draft_extend_select_index=index,
        )
        aux = [hidden * 2, hidden * 3]
        pruned, _, aux_pruned, sample, _, _ = _prune(metadata, hidden, aux)
        self.assertEqual(pruned[:, 0].tolist(), index.float().tolist())
        self.assertEqual(aux_pruned[0][:, 0].tolist(), (index * 2).float().tolist())
        self.assertIsNone(sample)

    def test_mixed_without_an_index_keeps_one_row_per_request(self):
        hidden = torch.arange(16, dtype=torch.float32)[:, None]
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.MIXED, extend_seq_lens=torch.tensor([5, 2, 6])
        )
        pruned, *_ = _prune(metadata, hidden)
        self.assertEqual(pruned[:, 0].tolist(), [4.0, 6.0, 12.0])

    def test_mixed_select_index_prefers_the_batch_field(self):
        from sglang.srt.layers.logits_processor import mixed_logits_select_index

        index = torch.tensor([3, 4])
        # Inside the traced forward spec_info carries no index: the field decides.
        traced = SimpleNamespace(logit_rows=index, spec_info=None)
        self.assertIs(mixed_logits_select_index(traced), index)
        eager = SimpleNamespace(
            logit_rows=None, spec_info=SimpleNamespace(logits_select_index=index)
        )
        self.assertIs(mixed_logits_select_index(eager), index)
        legacy = SimpleNamespace(logit_rows=None, spec_info=None)
        self.assertIsNone(mixed_logits_select_index(legacy))


if __name__ == "__main__":
    unittest.main()
