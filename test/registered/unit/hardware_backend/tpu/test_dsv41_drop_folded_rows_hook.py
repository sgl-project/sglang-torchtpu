"""The V4.1 TPU drop_folded_rows hook: bucket-row gather, and decoder-SWA tail rows
(``tail_lens``) passed to core's drop_folded_rows only when given. CPU only."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.hooks import drop_tpu_dsv41_folded_rows

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDropFoldedRowsHook(CustomTestCase):
    def test_tail_rows_take_the_original_with_tail_lens(self):
        calls = []
        out = SimpleNamespace(
            hidden_states=torch.zeros(4, 2),
            hidden_states_token_indices=torch.arange(4),
        )
        folded = SimpleNamespace(keep_rows=torch.arange(2), num_rows=4)
        drop_tpu_dsv41_folded_rows(
            lambda **kw: calls.append(kw),
            logits_output=out,
            folded=folded,
            tail_lens=[2, 2],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tail_lens"], [2, 2])

    def test_without_tail_lens_keeps_the_row_count(self):
        out = SimpleNamespace(
            hidden_states=torch.arange(8.0).reshape(4, 2),
            hidden_states_token_indices=None,
        )
        folded = SimpleNamespace(keep_rows=torch.tensor([0, 2]), num_rows=4)
        drop_tpu_dsv41_folded_rows(
            lambda **kw: self.fail("original called"), logits_output=out, folded=folded
        )
        self.assertEqual(tuple(out.hidden_states.shape), (4, 2))
        self.assertEqual(out.hidden_states[:2, 0].tolist(), [0.0, 4.0])

    def test_last_hidden_states_take_the_same_rows(self):
        hidden = torch.arange(8.0).reshape(4, 2)
        out = SimpleNamespace(
            hidden_states=hidden,
            last_hidden_states=hidden + 100,
            hidden_states_token_indices=None,
        )
        folded = SimpleNamespace(keep_rows=torch.tensor([0, 2]), num_rows=4)
        drop_tpu_dsv41_folded_rows(
            lambda **kw: self.fail("original called"), logits_output=out, folded=folded
        )
        index = torch.tensor([0, 2, 3, 3])
        self.assertTrue(torch.equal(out.hidden_states, hidden[index]))
        self.assertTrue(torch.equal(out.last_hidden_states, hidden[index] + 100))

    def test_bucket_padded_last_hidden_states_take_the_same_rows(self):
        hidden = torch.arange(8.0).reshape(4, 2)
        last = torch.arange(16.0).reshape(8, 2) + 100
        out = SimpleNamespace(
            hidden_states=hidden,
            last_hidden_states=last,
            hidden_states_token_indices=None,
        )
        folded = SimpleNamespace(keep_rows=torch.tensor([2, 3]), num_rows=4)
        drop_tpu_dsv41_folded_rows(
            lambda **kw: self.fail("original called"), logits_output=out, folded=folded
        )
        index = torch.tensor([2, 3, 3, 3])
        self.assertTrue(torch.equal(out.last_hidden_states, last[index]))

    def test_fallback_without_tail_lens_does_not_pass_it(self):
        calls = []
        out = SimpleNamespace(
            hidden_states=torch.zeros(4, 2),
            hidden_states_token_indices=torch.arange(4),
        )
        folded = SimpleNamespace(keep_rows=torch.arange(2), num_rows=4)
        drop_tpu_dsv41_folded_rows(
            lambda **kw: calls.append(kw), logits_output=out, folded=folded
        )
        self.assertEqual([sorted(c) for c in calls], [["folded", "logits_output"]])

    def test_tail_lens_without_token_indices_take_the_original(self):
        calls = []
        out = SimpleNamespace(
            hidden_states=torch.zeros(4, 2), hidden_states_token_indices=None
        )
        folded = SimpleNamespace(keep_rows=torch.tensor([0, 2]), num_rows=4)
        drop_tpu_dsv41_folded_rows(
            lambda **kw: calls.append(kw),
            logits_output=out,
            folded=folded,
            tail_lens=[1, 1],
        )
        self.assertEqual([c["tail_lens"] for c in calls], [[1, 1]])


if __name__ == "__main__":
    unittest.main()
