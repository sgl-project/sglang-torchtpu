"""Optional input-logprob chunk cuts (LogitsMetadata.input_logprob_chunk_cuts_cpu).

Core never sets the cuts; a platform hook may (the TPU plugin's
SGLANG_TPU_LOGPROB_GRID_ROWS, see
test/registered/unit/hardware_backend/tpu/test_logprob_grid.py). With no cuts the
input-logprob chunking is the uniform SGLANG_LOGPROB_CHUNK_SIZE split.
"""

import os
import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.logprob_processor import input_logprob_chunk_bounds
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.logprob_test_utils import CpuDefaultDeviceMixin, score_input_logprobs
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 2048  # SGLANG_LOGPROB_CHUNK_SIZE
VOCAB = 13


class TestLogprobChunkBounds(CpuDefaultDeviceMixin, CustomTestCase):
    def test_without_cuts_is_the_uniform_split(self):
        # no cuts: exactly the chunking before the cuts existed
        for total in range(0, 70):
            for chunk_size in range(1, 21):
                n = (total + chunk_size - 1) // chunk_size
                uniform = [
                    (i * chunk_size, min((i + 1) * chunk_size, total)) for i in range(n)
                ]
                self.assertEqual(
                    input_logprob_chunk_bounds(total, chunk_size, None), uniform
                )
                self.assertEqual(
                    input_logprob_chunk_bounds(total, chunk_size, []), uniform
                )

    def test_bounds(self):
        self.assertEqual(input_logprob_chunk_bounds(95, 95, [6]), [(0, 6), (6, 95)])
        self.assertEqual(
            input_logprob_chunk_bounds(5000, 2048, None),
            [(0, 2048), (2048, 4096), (4096, 5000)],
        )
        self.assertEqual(
            input_logprob_chunk_bounds(5000, 2048, [100]),
            [(0, 100), (100, 2148), (2148, 4196), (4196, 5000)],
        )
        self.assertEqual(input_logprob_chunk_bounds(0, 1, None), [])

    def test_cut_order_duplicates_and_out_of_range(self):
        expected = [(0, 100), (100, 2148), (2148, 4096), (4096, 5000)]
        self.assertEqual(input_logprob_chunk_bounds(5000, 2048, [100, 4096]), expected)
        self.assertEqual(
            input_logprob_chunk_bounds(5000, 2048, [4096, 100, 100, 0, 5000, 7000]),
            expected,
        )

    def test_properties(self):
        rng = random.Random(5)
        for _ in range(3000):
            total = rng.randrange(0, 300)
            chunk_size = rng.randrange(1, 64)
            cuts = sorted(rng.sample(range(1, 300), rng.randrange(0, 6)))
            bounds = input_logprob_chunk_bounds(total, chunk_size, cuts)
            inside = [c for c in cuts if 0 < c < total]
            edges = [0] + inside + [total]
            # the chunks tile [0, total)
            self.assertEqual(
                [lo for lo, _ in bounds],
                [0] + [hi for _, hi in bounds[:-1]] if bounds else [],
            )
            self.assertEqual(bounds[-1][1] if bounds else 0, total)
            for lo, hi in bounds:
                self.assertLess(lo, hi)
                self.assertLessEqual(hi - lo, chunk_size)
                # no chunk crosses a cut, and pieces start chunk_size apart from the segment start
                seg = max(e for e in edges if e <= lo)
                self.assertTrue(
                    all(not (lo < c < hi) for c in inside), (total, chunk_size, cuts)
                )
                self.assertEqual((lo - seg) % chunk_size, 0, (total, chunk_size, cuts))


class TestInputLogprobChunkCuts(CpuDefaultDeviceMixin, CustomTestCase):
    """The real InputLogprobProcessor with a row-count-dependent stub epilogue."""

    def _score(self, states, ids, extend_lens, start_lens, cuts, dp_attn=False):
        return score_input_logprobs(
            states,
            ids,
            extend_lens,
            start_lens,
            cuts,
            vocab_size=VOCAB,
            chunk_size=CHUNK,
            skip_chunking_for_dp_attn=dp_attn,
        )

    def test_cuts_split_the_epilogue_calls(self):
        torch.manual_seed(0)
        rows = torch.randn(95, VOCAB)
        ids = torch.randint(0, VOCAB, (95,))
        # the same 95 rows as two separate calls of 6 and 89 rows
        a, calls_a = self._score(rows[:6], ids[:6], [2048], [2042], None)
        b, calls_b = self._score(rows[6:], ids[6:], [89], [0], None)
        apart = torch.cat([a, b])
        on, calls_on = self._score(rows, ids, [2137], [2042], [6])
        off, calls_off = self._score(rows, ids, [2137], [2042], None)
        self.assertEqual(calls_a + calls_b, [6, 89])
        self.assertEqual(calls_on, [6, 89])
        self.assertEqual(calls_off, [95])
        self.assertTrue(torch.equal(on, apart))
        self.assertFalse(torch.equal(off, apart))

    def test_dp_attention_ignores_cuts(self):
        # DP-attention keeps one chunk (the collective schedule cannot follow per-rank rows)
        torch.manual_seed(2)
        rows = torch.randn(95, VOCAB)
        ids = torch.randint(0, VOCAB, (95,))
        _, calls = self._score(rows, ids, [2137], [2042], [6], dp_attn=True)
        self.assertEqual(calls, [95])
        _, calls = self._score(rows, ids, [2137], [2042], [6])
        self.assertEqual(calls, [6, 89])

    def test_no_cuts_keep_the_uniform_chunking(self):
        torch.manual_seed(1)
        for extend_lens, start_lens, n, expected in [
            ([700, 2048, 1300], [0, 1990, 1300], 700 + 58 + 1, [759]),
            ([5000], [0], 5000, [2048, 2048, 904]),
        ]:
            with self.subTest(extend_lens=extend_lens):
                rows = torch.randn(n, VOCAB)
                ids = torch.randint(0, VOCAB, (n,))
                unset, calls_unset = self._score(
                    rows, ids, extend_lens, start_lens, None
                )
                empty, calls_empty = self._score(rows, ids, extend_lens, start_lens, [])
                self.assertEqual(calls_unset, expected)
                self.assertEqual(calls_empty, expected)
                self.assertTrue(torch.equal(unset, empty))


class TestCoreSetsNoCuts(CpuDefaultDeviceMixin, CustomTestCase):
    """Core only honours the cuts; it reads no TPU variable and never sets them."""

    def test_pruned_states_leave_the_cuts_unset(self):
        md = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            extend_return_logprob=True,
            extend_logprob_start_lens_cpu=[2042],
            extend_seq_lens_cpu=[8192],
        )
        self.assertIsNone(md.input_logprob_chunk_cuts_cpu)
        with mock.patch.dict(os.environ, {"SGLANG_TPU_LOGPROB_GRID_ROWS": "2048"}):
            pruned, *_ = LogitsProcessor._get_pruned_states(
                SimpleNamespace(), torch.randn(8192, 4), None, None, md
            )
        self.assertEqual(pruned.shape[0], 8192 - 2042)
        self.assertIsNone(md.input_logprob_chunk_cuts_cpu)


if __name__ == "__main__":
    unittest.main()
