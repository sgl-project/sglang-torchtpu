"""SGLANG_TPU_LOGPROB_GRID_ROWS (sglang_tpu/hooks/logprob_grid.py): input-logprob chunks
of a long forward follow the 2048-row forward grid.

The eager input-logprob epilogue (lm_head on power-of-two row buckets, the logsumexp
normalizer) rounds differently for different row counts. A lone scoring request whose
scored rows cross a 2048-row boundary is split there by 2048-row forwards (Q2048) but
not by 4096-row forwards (Q4096), so Q4096 changed those rows' logprobs by an ulp (the
Q4096 vision gate: prompts with n_prompt 2042, L 2137 scored as 6 | 89 rows vs 95 rows).
With the grid on, a Q4096 forward holding one request hands the epilogue the same row
sets as Q2048; under Q2048 (no request has more than 2048 rows in one forward) the grid
cuts nothing. A forward holding several requests is not cut: the scheduler splits one
prefill budget between them, so their Q2048 forwards do not start on each request's own
grid multiples.
"""

import os
import pkgutil
import random
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.logprob_processor import input_logprob_chunk_bounds
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.plugins.hook_registry import HookType, _wrap_fn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.logprob_test_utils import (
    CpuDefaultDeviceMixin,
    chunked_input_logprob_processor,
    row_count_dependent_logits,
    score_input_logprobs,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.hooks import logprob_grid as hooks
from sglang_tpu.hooks.logprob_grid import logprob_grid_cuts

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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


class TestLogprobGridPlan(CpuDefaultDeviceMixin, CustomTestCase):
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

    def test_cuts_of_a_lone_request(self):
        # offsets into the request's pruned rows, counted from its first row
        self.assertEqual(logprob_grid_cuts([100], [4096], GRID), [1948])
        self.assertEqual(logprob_grid_cuts([0], [8192], GRID), [2048, 4096, 6144])
        self.assertEqual(logprob_grid_cuts([3000], [8192], GRID), [1096, 3144])
        self.assertEqual(logprob_grid_cuts([2048], [4096], GRID), [])


def _shared_budget_forwards(lengths, chunk):
    """Chunked prefill with one shared budget: every request queued at once, in order;
    each forward takes the next ``chunk`` extend rows, splitting a request at the end
    of the budget. Each forward is a list of ``(request, first row, end row)``."""
    out, cur, budget = [], [], chunk
    for r, n in enumerate(lengths):
        done = 0
        while done < n:
            take = min(budget, n - done)
            cur.append((r, done, done + take))
            done, budget = done + take, budget - take
            if budget == 0:
                out.append(cur)
                cur, budget = [], chunk
    return out + [cur] if cur else out


def _batch_plan(reqs, prefill_chunk, grid):
    """The epilogue calls of requests ``[(length, logprob_start)]`` that share the
    prefill budget, each call as its ``(request, row)`` pairs."""
    calls = []
    for forward in _shared_budget_forwards([n for n, _ in reqs], prefill_chunk):
        rows, lp_rows, starts, extends = [], set(), [], []
        for r, fs, fe in forward:
            extend = fe - fs
            start = min(max(reqs[r][1] - fs, 0), extend)
            eff = start - 1 if start == extend else start
            starts.append(start)
            extends.append(extend)
            rows += [(r, fs + i) for i in range(eff, extend)]
            lp_rows.update((r, fs + i) for i in range(start, extend))
        cuts = logprob_grid_cuts(starts, extends, grid) if grid else []
        for lo, hi in input_logprob_chunk_bounds(
            len(rows), _chunk_size(len(rows)), cuts
        ):
            chunk = tuple(rows[lo:hi])
            if lp_rows.intersection(chunk):
                calls.append(chunk)
    return calls


class TestLogprobGridSharedBudget(CpuDefaultDeviceMixin, CustomTestCase):
    def test_a_forward_with_several_requests_is_not_cut(self):
        # Review case, scoring lengths [1024, 3072]: Q2048 runs [A, B[0:1024)] then
        # [B[1024:3072)]; Q4096 runs both requests in one forward. A cut on B's own
        # grid (B row 2048) would split B[1024:3072) in two. The forward is not cut,
        # so it is chunked exactly as with the env off.
        reqs = [(1024, 0), (3072, 0)]
        self.assertEqual(logprob_grid_cuts([0, 0], [1024, 3072], GRID), [])
        q4096 = _batch_plan(reqs, 4096, GRID)
        self.assertEqual(q4096, _batch_plan(reqs, 4096, 0))
        self.assertEqual(q4096, _batch_plan(reqs, 2048, 0))
        self.assertEqual([len(call) for call in q4096], [2048, 2048])

    def test_several_requests_keep_the_env_off_chunking(self):
        # every Q4096 forward here holds two requests, so the grid changes nothing
        for reqs in (
            [(1024, 0), (3072, 2000)],  # B scored past the Q2048 split
            [(1024, 1024), (3072, 0)],  # A asks for no input logprob
            [(2000, 100), (2096, 1500)],
            [(3000, 0), (1096, 500)],
        ):
            with self.subTest(reqs=reqs):
                forwards = _shared_budget_forwards([n for n, _ in reqs], 4096)
                self.assertTrue(all(len(forward) > 1 for forward in forwards))
                self.assertEqual(
                    _batch_plan(reqs, 4096, GRID), _batch_plan(reqs, 4096, 0)
                )

    def test_a_lone_request_still_follows_the_grid(self):
        for reqs in ([(2137, 2042)], [(12800, 3000)], [(9000, 0)]):
            with self.subTest(reqs=reqs):
                self.assertEqual(
                    _batch_plan(reqs, 4096, GRID), _batch_plan(reqs, 2048, 0)
                )


def _bruteforce_cuts(start, extend, grid):
    return [pos - start for pos in range(start + 1, extend) if pos % grid == 0]


class TestLogprobGridCuts(CpuDefaultDeviceMixin, CustomTestCase):
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

    def test_a_forward_with_several_requests_is_not_cut(self):
        for starts, extends in [
            ([0, 100], [1000, 4096]),
            ([0, 0], [4096, 100]),
            ([0, 0], [1000, 3000]),
            ([0, 2048], [100, 4097]),
            ([5, 0], [6, 2049]),
            ([0, 0], [1024, 3072]),
            ([2042, 5], [2137, 5]),
        ]:
            with self.subTest(starts=starts, extends=extends):
                self.assertEqual(logprob_grid_cuts(starts, extends, GRID), [])

    def test_no_grid_or_no_rows(self):
        self.assertEqual(logprob_grid_cuts([0], [8192], 0), [])
        self.assertEqual(logprob_grid_cuts(None, None, GRID), [])
        self.assertEqual(logprob_grid_cuts([], [], GRID), [])

    def test_a_request_without_logprob_rows_keeps_its_last_row(self):
        # start == extend: the pruned rows are the last row only, so nothing to cut
        self.assertEqual(logprob_grid_cuts([4097], [4097], GRID), [])
        self.assertEqual(
            logprob_grid_cuts([4097], [4097], GRID),
            logprob_grid_cuts([4096], [4097], GRID),
        )

    def test_matches_bruteforce(self):
        grid = 8
        for extend in range(1, 3 * grid + 2):
            for start in range(extend):
                self.assertEqual(
                    logprob_grid_cuts([start], [extend], grid),
                    _bruteforce_cuts(start, extend, grid),
                    (start, extend),
                )
        rng = random.Random(3)
        for _ in range(3000):
            extends = [
                rng.randrange(1, 4 * grid + 2) for _ in range(rng.randrange(1, 5))
            ]
            starts = [rng.randrange(0, e) for e in extends]
            expected = (
                _bruteforce_cuts(starts[0], extends[0], grid)
                if len(extends) == 1
                else []
            )
            self.assertEqual(
                logprob_grid_cuts(starts, extends, grid), expected, (starts, extends)
            )


class TestLogprobGridEnv(CustomTestCase):
    """The variable is declared in sglang_tpu.envs; unset, empty or 0 registers no hook."""

    NAME = "SGLANG_TPU_LOGPROB_GRID_ROWS"

    def _read(self, value):
        env = {k: v for k, v in os.environ.items() if k != self.NAME}
        if value is not None:
            env[self.NAME] = value
        with mock.patch.dict(os.environ, env, clear=True), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "" is not an integer: default, warned
            return envs.SGLANG_TPU_LOGPROB_GRID_ROWS.get(), hooks.hook_specs()

    def test_off(self):
        self.assertIn(self.NAME, envs.names())
        for value in (None, "", "0"):
            with self.subTest(value=value):
                self.assertEqual(self._read(value), (0, []))

    def test_on(self):
        grid, specs = self._read("2048")
        self.assertEqual(grid, 2048)
        ((target, hook, hook_type),) = specs
        self.assertIs(hook, hooks.record_logprob_grid_cuts)
        self.assertIs(hook_type, HookType.AFTER)
        self.assertEqual(
            pkgutil.resolve_name(target), LogitsMetadata.from_forward_batch
        )


class TestLogprobGridProcessor(CpuDefaultDeviceMixin, CustomTestCase):
    """The real InputLogprobProcessor with a row-count-dependent stub epilogue."""

    def _score(self, states, ids, extend_lens, start_lens, cuts):
        return score_input_logprobs(
            states,
            ids,
            extend_lens,
            start_lens,
            cuts,
            vocab_size=VOCAB,
            chunk_size=CHUNK,
        )

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


def _forward_batch(extend_lens, start_lens, return_logprob=True, mode=None):
    n = len(extend_lens)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if mode is None else mode,
        _original_forward_mode=None,
        return_logprob=return_logprob,
        top_logprobs_nums=[0] * n,
        token_ids_logprobs=[None] * n,
        extend_seq_lens_cpu=list(extend_lens),
        extend_logprob_start_lens_cpu=list(start_lens),
        extend_input_logprob_token_ids_gpu=torch.randint(0, VOCAB, (sum(extend_lens),)),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        next_token_logits_buffer=None,
        extend_seq_lens=torch.tensor(extend_lens),
        is_prefill_only=False,
        attn_tp_sequence_sharded=False,
        global_num_tokens_gpu=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_for_logprob_gpu=None,
        mm_input_embeds=None,
        spec_info=None,
    )


class TestLogprobGridHook(CpuDefaultDeviceMixin, CustomTestCase):
    """The AFTER hook on LogitsMetadata.from_forward_batch, wrapped the way
    HookRegistry wraps a classmethod, then core's pruning and chunking."""

    def _metadata(self, extend_lens, start_lens, grid=GRID, **kwargs):
        from_forward_batch = _wrap_fn(
            LogitsMetadata.__dict__["from_forward_batch"].__func__,
            hooks.record_logprob_grid_cuts,
            HookType.AFTER,
        )
        forward_batch = _forward_batch(extend_lens, start_lens, **kwargs)
        with envs.SGLANG_TPU_LOGPROB_GRID_ROWS.override(grid):
            return from_forward_batch(LogitsMetadata, forward_batch)

    def _epilogue_calls(self, md):
        hidden = torch.randn(sum(md.extend_seq_lens_cpu), VOCAB)
        pruned, _, _, sample, lp_idx, t2s = LogitsProcessor._get_pruned_states(
            SimpleNamespace(), hidden, None, None, md
        )
        calls = []
        chunked_input_logprob_processor(VOCAB, CHUNK).forward(
            pruned_states=pruned,
            sample_indices=sample,
            input_logprob_indices=lp_idx,
            token_to_seq_idx=t2s,
            lm_head=None,
            get_logits_fn=row_count_dependent_logits(calls),
            logits_metadata=md,
        )
        return calls

    def test_a_lone_request_is_cut_on_its_grid(self):
        md = self._metadata([2137], [2042])
        self.assertEqual(md.input_logprob_chunk_cuts_cpu, [6])
        self.assertEqual(self._epilogue_calls(md), [6, 89])
        md = self._metadata([8192], [3000])
        self.assertEqual(md.input_logprob_chunk_cuts_cpu, [1096, 3144])
        self.assertEqual(self._epilogue_calls(md), [1096, 2048, 2048])

    def test_a_forward_with_several_requests_is_not_cut(self):
        for extend_lens, start_lens, calls in [
            ([2137, 5], [2042, 5], [96]),
            ([5, 2137], [5, 2042], [96]),
            ([1024, 3072], [0, 0], [2048, 2048]),
        ]:
            with self.subTest(extend_lens=extend_lens, start_lens=start_lens):
                md = self._metadata(extend_lens, start_lens)
                self.assertIsNone(md.input_logprob_chunk_cuts_cpu)
                self.assertEqual(self._epilogue_calls(md), calls)

    def test_other_forwards_keep_no_cuts(self):
        cases = {
            "rows fit the grid": dict(extend_lens=[2048], start_lens=[100]),
            "no logprob request": dict(
                extend_lens=[4096], start_lens=[0], return_logprob=False
            ),
            "no logprob row": dict(extend_lens=[4096], start_lens=[4096]),
            "decode": dict(extend_lens=[1], start_lens=[0], mode=ForwardMode.DECODE),
            "grid off": dict(extend_lens=[2137], start_lens=[2042], grid=0),
        }
        for name, case in cases.items():
            with self.subTest(name):
                self.assertIsNone(self._metadata(**case).input_logprob_chunk_cuts_cpu)


if __name__ == "__main__":
    unittest.main()
