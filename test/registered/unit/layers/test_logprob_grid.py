"""SGLANG_TPU_LOGPROB_GRID_ROWS: input-logprob chunks of a long forward follow the 2048-row forward grid.

The eager input-logprob epilogue (lm_head on power-of-two row buckets, the logsumexp
normalizer) rounds differently for different row counts. A scoring request whose scored
rows cross a 2048-row boundary is split there by 2048-row forwards (Q2048) but not by
4096-row forwards (Q4096), so Q4096 changed those rows' logprobs by an ulp (the Q4096
vision gate: prompts with n_prompt 2042, L 2137 scored as 6 | 89 rows vs 95 rows).
With the grid on, Q4096 hands the epilogue the same row sets as Q2048; under Q2048
(no request has more than 2048 rows in one forward) the grid cuts nothing.
"""

import importlib.util
import os
import random
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import logits_processor as lp_mod
from sglang.srt.layers import logprob_processor as lpp_mod
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    input_logprob_chunk_bounds,
    logprob_grid_cuts,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _CpuDefaultDevice:
    """Build this module's tensors on the CPU even if an earlier test in the same
    process left another default device (torch.set_default_device) behind."""

    def setUp(self):
        super().setUp()
        device = torch.device("cpu")
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)


GRID = 2048
CHUNK = 2048  # SGLANG_LOGPROB_CHUNK_SIZE
VOCAB = 13


def _forwards(prefix, length, chunk):
    """Forward row ranges of one request's chunked prefill from ``prefix`` to ``length``."""
    out, start = [], prefix
    while start < length:
        end = min(start + chunk, length)
        out.append((start, end))
        start = end
    return out


def _chunk_size(n):
    # InputLogprobProcessor.forward with SGLANG_ENABLE_LOGPROB_CHUNK
    return max(n, 1) if n <= CHUNK else CHUNK


def _plan(prefix, length, logprob_start, prefill_chunk, grid):
    """The epilogue calls of a lone scoring request: every chunk holding a logprob row,
    as its absolute rows (the call's row set fixes its shapes and so its rounding)."""
    calls = []
    for fs, fe in _forwards(prefix, length, prefill_chunk):
        extend = fe - fs
        start = min(max(logprob_start - fs, 0), extend)
        eff = start - 1 if start == extend else start
        rows = list(range(fs + eff, fe))
        lp_rows = set(range(fs + start, fe))
        cuts = logprob_grid_cuts([eff], [extend], grid) if grid else []
        for lo, hi in input_logprob_chunk_bounds(
            len(rows), _chunk_size(len(rows)), cuts
        ):
            chunk = tuple(rows[lo:hi])
            if lp_rows.intersection(chunk):
                calls.append(chunk)
    return calls


CASES = [
    # (radix prefix, prompt + target length, logprob_start_len)
    (0, 2137, 2042),  # vision gate prompts 30-32: 6 | 89 under Q2048
    (0, 4152, 4057),  # vision gate prompts 36-38: 39 | 56 in both
    (0, 2729, 2634),  # vision gate videos
    (3584, 9000, 5000),  # starts mid-grid (radix hit), unaligned logprob start
    (1536, 7000, 2100),
    (512, 12800, 3000),  # three Q4096 forwards
    (0, 11000, 777),  # three Q4096 forwards, scored from inside the first
    (0, 6000, 0),  # scored from row 0 (the text TF gates)
    (0, 5000, 2048),  # logprob start on a grid boundary
    (0, 8192, 6143),
    (2048, 10240, 9000),
    (0, 4096, 4095),  # one scored row, last row of a 4096 forward
]


class TestLogprobGridPlan(_CpuDefaultDevice, CustomTestCase):
    def test_q4096_with_grid_matches_q2048(self):
        for prefix, length, start in CASES:
            with self.subTest(prefix=prefix, length=length, start=start):
                self.assertEqual(
                    _plan(prefix, length, start, 4096, GRID),
                    _plan(prefix, length, start, 2048, 0),
                )

    def test_q4096_with_grid_matches_q2048_random(self):
        rng = random.Random(7)
        for _ in range(2000):
            prefix = 512 * rng.randrange(0, 17)
            length = prefix + rng.randrange(1, 20000)
            start = rng.randrange(prefix, length)
            self.assertEqual(
                _plan(prefix, length, start, 4096, GRID),
                _plan(prefix, length, start, 2048, 0),
                (prefix, length, start),
            )

    def test_q4096_without_grid_differs_on_a_straddle(self):
        # the R7Q failure the grid fixes (sanity: the plan comparison is sensitive)
        self.assertEqual(
            _plan(0, 2137, 2042, 2048, 0),
            [tuple(range(2042, 2048)), tuple(range(2048, 2137))],
        )
        self.assertEqual(_plan(0, 2137, 2042, 4096, 0), [tuple(range(2042, 2137))])

    def test_grid_is_a_noop_under_q2048(self):
        # every Q2048 forward: no request has more than 2048 rows, so nothing is cut and
        # the chunking is the uniform one -- with the env ON the plan equals env OFF.
        rng = random.Random(11)
        for _ in range(2000):
            n = rng.randrange(1, 9)
            extends = [rng.randrange(1, GRID + 1) for _ in range(n)]
            starts = [rng.randrange(0, e + 1) for e in extends]
            effs = [s - 1 if s == e else s for s, e in zip(starts, extends)]
            cuts = logprob_grid_cuts(effs, extends, GRID)
            self.assertEqual(cuts, [])
            total = sum(e - s for s, e in zip(effs, extends))
            for chunk_size in (1, 3, 512, CHUNK, _chunk_size(total)):
                self.assertEqual(
                    input_logprob_chunk_bounds(total, chunk_size, cuts),
                    input_logprob_chunk_bounds(total, chunk_size, None),
                )
        for prefix, length, start in CASES:
            self.assertEqual(
                _plan(prefix, length, start, 2048, GRID),
                _plan(prefix, length, start, 2048, 0),
            )

    def test_cuts_are_relative_to_each_request(self):
        # request A: 1000 scored rows, then B starting mid-forward with rows [100, 4096)
        # of its own extend: B's cut sits at its own offset 2048 (pruned offset 1000 + 1948)
        self.assertEqual(logprob_grid_cuts([0, 100], [1000, 4096], GRID), [1948 + 1000])
        self.assertEqual(logprob_grid_cuts([0], [8192], GRID), [2048, 4096, 6144])
        self.assertEqual(logprob_grid_cuts([3000], [8192], GRID), [1096, 3144])
        self.assertEqual(logprob_grid_cuts([2048], [4096], GRID), [])

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


def _bruteforce_cuts(start_lens, extend_lens, grid):
    cuts, base = [], 0
    for start, extend in zip(start_lens, extend_lens):
        cuts += [
            base + pos - start for pos in range(start + 1, extend) if pos % grid == 0
        ]
        base += extend - start
    return cuts


class TestLogprobGridCuts(_CpuDefaultDevice, CustomTestCase):
    def test_boundaries(self):
        for start, extend, cuts in [
            (0, 2047, []),
            (0, 2048, []),
            (0, 2049, [2048]),
            (0, 4096, [2048]),
            (0, 4097, [2048, 4096]),
            (2047, 2049, [1]),
            (2047, 4096, [1]),
            (2048, 4096, []),
            (2048, 4097, [2048]),
            (2049, 4097, [2047]),
            (4095, 4097, [1]),
            (4096, 4097, []),
        ]:
            with self.subTest(start=start, extend=extend):
                self.assertEqual(logprob_grid_cuts([start], [extend], GRID), cuts)

    def test_request_boundaries_are_not_cuts(self):
        # a request ending on a grid point, or the next one starting on it, adds no cut there
        self.assertEqual(logprob_grid_cuts([0, 0], [4096, 100], GRID), [2048])
        self.assertEqual(logprob_grid_cuts([0, 0], [1000, 3000], GRID), [3048])
        self.assertEqual(logprob_grid_cuts([0, 2048], [100, 4097], GRID), [2148])
        self.assertEqual(logprob_grid_cuts([5, 0], [6, 2049], GRID), [2049])

    def test_matches_bruteforce(self):
        grid = 8
        for extend in range(1, 3 * grid + 2):
            for start in range(extend):
                self.assertEqual(
                    logprob_grid_cuts([start], [extend], grid),
                    _bruteforce_cuts([start], [extend], grid),
                    (start, extend),
                )
        rng = random.Random(3)
        for _ in range(3000):
            extends = [
                rng.randrange(1, 4 * grid + 2) for _ in range(rng.randrange(1, 5))
            ]
            starts = [rng.randrange(0, e) for e in extends]
            self.assertEqual(
                logprob_grid_cuts(starts, extends, grid),
                _bruteforce_cuts(starts, extends, grid),
                (starts, extends),
            )


class TestLogprobChunkBounds(_CpuDefaultDevice, CustomTestCase):
    def test_without_cuts_is_the_uniform_split(self):
        # the env-off path: exactly the chunking before this change
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


class TestLogprobGridEnv(_CpuDefaultDevice, CustomTestCase):
    """SGLANG_TPU_LOGPROB_GRID_ROWS is read at import; default (unset or empty) is off."""

    def _load(self, value):
        env = {
            k: v for k, v in os.environ.items() if k != "SGLANG_TPU_LOGPROB_GRID_ROWS"
        }
        if value is not None:
            env["SGLANG_TPU_LOGPROB_GRID_ROWS"] = value
        name = "_logprob_grid_env_probe"
        spec = importlib.util.spec_from_file_location(name, lpp_mod.__file__)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules):
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.LOGPROB_GRID_ROWS

    def test_env(self):
        self.assertEqual(self._load(None), 0)
        self.assertEqual(self._load(""), 0)
        self.assertEqual(self._load("0"), 0)
        self.assertEqual(self._load("2048"), 2048)


def _metadata(extend_lens, start_lens, ids):
    sample, lp_idx, pruned_lens, t2s = [], [], [], []
    sample_pt, lp_pt = -1, 0
    for i, (extend, start) in enumerate(zip(extend_lens, start_lens)):
        eff = start - 1 if extend == start else start
        rows = extend - eff
        t2s += [i] * rows
        sample_pt += rows
        sample.append(sample_pt)
        lp_idx += [lp_pt + j for j in range(extend - start)]
        lp_pt += rows
        pruned_lens.append(extend - start)
    md = SimpleNamespace(
        sample_indices_cpu=sample,
        input_logprob_indices_cpu=lp_idx,
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        top_logprobs_nums=[0] * len(extend_lens),
        extend_logprob_pruned_lens_cpu=pruned_lens,
        extend_input_logprob_token_ids_gpu=ids[: len(lp_idx)],
        token_ids_logprobs=[None] * len(extend_lens),
    )
    return md, torch.tensor(sample), torch.tensor(lp_idx), t2s


def _row_count_dependent_logits(calls):
    def get_logits_fn(states, lm_head, logits_metadata, **kwargs):
        calls.append(states.shape[0])
        # emulate an epilogue whose rounding depends on the row count of the call
        return states.float() * (1.0 + 1e-6 * states.shape[0])

    return get_logits_fn


class TestLogprobGridProcessor(_CpuDefaultDevice, CustomTestCase):
    """The real InputLogprobProcessor with a row-count-dependent stub epilogue."""

    def _score(self, states, ids, extend_lens, start_lens, cuts, dp_attn=False):
        proc = InputLogprobProcessor(vocab_size=VOCAB)
        proc.enable_logprobs_chunk = True
        proc.logprobs_chunk_size = CHUNK
        proc.enable_fast_input_logprobs = True
        md, sample, lp_idx, t2s = _metadata(extend_lens, start_lens, ids)
        md.input_logprob_grid_cuts_cpu = cuts
        calls = []
        res, _ = proc.forward(
            pruned_states=states,
            sample_indices=sample,
            input_logprob_indices=lp_idx,
            token_to_seq_idx=t2s,
            lm_head=None,
            get_logits_fn=_row_count_dependent_logits(calls),
            logits_metadata=md,
            skip_chunking_for_dp_attn=dp_attn,
        )
        return res.token_logprobs, calls

    def _serve(self, hidden, ids, prefix, length, logprob_start, prefill_chunk, grid):
        """A lone scoring request's prompt logprobs over its chunked-prefill forwards."""
        out, calls = [], []
        for fs, fe in _forwards(prefix, length, prefill_chunk):
            extend = fe - fs
            start = min(max(logprob_start - fs, 0), extend)
            if start == extend:
                continue  # no logprob row in this forward
            cuts = logprob_grid_cuts([start], [extend], grid) if grid else None
            lps, c = self._score(
                hidden[fs + start : fe], ids[fs + start : fe], [extend], [start], cuts
            )
            out.append(lps)
            calls += c
        return torch.cat(out), calls

    def test_q4096_with_grid_is_bitwise_q2048(self):
        for prefix, length, logprob_start, straddles in [
            (0, 2137, 2042, True),  # vision gate prompts 30-32
            (3584, 7680, 5000, True),  # radix hit mid-grid
            (512, 12800, 2000, True),  # three Q4096 forwards
            (2048, 6200, 4095, True),
            (0, 2049, 0, False),  # one row past the boundary, scored from row 0
            (512, 12800, 3000, False),  # scored rows already on the 2048-row chunks
        ]:
            with self.subTest(prefix=prefix, length=length, start=logprob_start):
                g = torch.Generator().manual_seed(length)
                hidden = torch.randn(length, VOCAB, generator=g)
                ids = torch.randint(0, VOCAB, (length,), generator=g)
                args = (hidden, ids, prefix, length, logprob_start)
                q2048, calls_2048 = self._serve(*args, 2048, 0)
                on, calls_on = self._serve(*args, 4096, GRID)
                off, calls_off = self._serve(*args, 4096, 0)
                self.assertEqual(calls_on, calls_2048)
                self.assertTrue(torch.equal(on, q2048))
                # without the grid, Q4096 groups a straddling range differently
                self.assertEqual(calls_off != calls_2048, straddles)
                self.assertEqual(not torch.equal(off, q2048), straddles)

    def test_dp_attention_stays_single_chunk(self):
        # DP-attention keeps one chunk (the collective schedule cannot follow per-rank rows)
        torch.manual_seed(2)
        rows = torch.randn(95, VOCAB)
        ids = torch.randint(0, VOCAB, (95,))
        _, calls = self._score(rows, ids, [2137], [2042], [6], dp_attn=True)
        self.assertEqual(calls, [95])
        _, calls = self._score(rows, ids, [2137], [2042], [6])
        self.assertEqual(calls, [6, 89])

    def test_vision_gate_straddle(self):
        torch.manual_seed(0)
        rows = torch.randn(
            95, VOCAB
        )  # scored rows 2042..2136 (same hidden in both arms)
        ids = torch.randint(0, VOCAB, (95,))
        # Q2048: forward [0, 2048) scores rows 2042..2047, forward [2048, 2137) rows 2048..2136
        a, calls_a = self._score(rows[:6], ids[:6], [2048], [2042], None)
        b, calls_b = self._score(rows[6:], ids[6:], [89], [0], None)
        q2048 = torch.cat([a, b])
        # Q4096: one forward [0, 2137) scoring rows 2042..2136
        cuts = logprob_grid_cuts([2042], [2137], GRID)
        on, calls_on = self._score(rows, ids, [2137], [2042], cuts)
        off, calls_off = self._score(rows, ids, [2137], [2042], None)
        self.assertEqual(calls_a + calls_b, [6, 89])
        self.assertEqual(calls_on, [6, 89])
        self.assertEqual(calls_off, [95])
        self.assertTrue(torch.equal(on, q2048))
        self.assertFalse(torch.equal(off, q2048))

    def test_noop_under_q2048_multi_request(self):
        torch.manual_seed(1)
        extend_lens, start_lens = [700, 2048, 1300], [0, 1990, 1300]
        cuts = logprob_grid_cuts([0, 1990, 1299], extend_lens, GRID)
        self.assertEqual(cuts, [])
        n = 700 + 58 + 1
        rows = torch.randn(n, VOCAB)
        ids = torch.randint(0, VOCAB, (n,))
        on, calls_on = self._score(rows, ids, extend_lens, start_lens, cuts)
        off, calls_off = self._score(rows, ids, extend_lens, start_lens, None)
        self.assertEqual(calls_on, calls_off)
        self.assertTrue(torch.equal(on, off))


class TestPrunedStatesSetsCuts(_CpuDefaultDevice, CustomTestCase):
    """LogitsProcessor._get_pruned_states records the cuts only with the env on."""

    def _pruned(self, grid, start_lens=(2042, 5), extend_lens=(2137, 5)):
        md = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_return_logprob=True,
            extend_logprob_start_lens_cpu=list(start_lens),
            extend_seq_lens_cpu=list(extend_lens),
            draft_extend_select_index=None,
        )
        hidden = torch.randn(sum(extend_lens), 4)
        with mock.patch.object(lp_mod, "LOGPROB_GRID_ROWS", grid):
            out = lp_mod.LogitsProcessor._get_pruned_states(
                SimpleNamespace(), hidden, None, None, md
            )
        return out, md

    def test_env_on(self):
        (pruned, _, _, sample, lp_idx, _), md = self._pruned(GRID)
        self.assertEqual(md.input_logprob_grid_cuts_cpu, [6])
        self.assertEqual(pruned.shape[0], 95 + 1)

    def test_env_on_after_a_zero_logprob_row_request(self):
        # a request with no logprob row still contributes its last row to the pruned rows
        (pruned, *_), md = self._pruned(GRID, (5, 2042), (5, 2137))
        self.assertEqual(md.input_logprob_grid_cuts_cpu, [1 + 6])
        self.assertEqual(pruned.shape[0], 1 + 95)

    def test_env_off(self):
        _, md = self._pruned(0)
        self.assertIsNone(getattr(md, "input_logprob_grid_cuts_cpu", None))


if __name__ == "__main__":
    unittest.main()
