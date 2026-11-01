"""Bucketed repetition-penalty rows (SGLANG_BUCKETED_PENALTY_ROWS).

The bucketed state must give the same penalties, bit for bit, as the exact-row
state through any sequence of decode steps, finishes (filter) and joins (merge).
"""

import os
import random
import unittest
from unittest.mock import patch

import torch

import sglang.srt.sampling.penaltylib.repetition_penalty as rp
from sglang.srt.environ import envs
from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

VOCAB_SIZE = 97
REPETITION_ONLY = {rp.BatchedRepetitionPenalizer}
MIXED = {
    rp.BatchedRepetitionPenalizer,
    BatchedFrequencyPenalizer,
    BatchedPresencePenalizer,
}


def _eager_apply_scaling_penalties(logits, scaling_penalties):
    # Same math as apply_scaling_penalties, without torch.compile.
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


class _SamplingParams:
    def __init__(self, repetition=1.0, frequency=0.0, presence=0.0):
        self.repetition_penalty = repetition
        self.frequency_penalty = frequency
        self.presence_penalty = presence


class _Req:
    def __init__(self, **sampling_params):
        self.sampling_params = _SamplingParams(**sampling_params)


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.device = "cpu"


def _random_req(rnd, mixed):
    if not mixed:
        return _Req(repetition=rnd.choice([1.0, 1.1, 1.3]))
    return _Req(
        repetition=rnd.choice([1.0, 1.0, 1.1, 1.3]),
        frequency=rnd.choice([0.0, 0.0, 0.4, 0.8]),
        presence=rnd.choice([0.0, 0.3, 0.6]),
    )


def _run(bucketed, seed, penalizers, mixed, steps, max_batch):
    """Random decode / finish / join trace; returns the penalized logits per step.

    batch.reqs is updated before filter and after merge, as ScheduleBatch does,
    so penalizers tear down and re-prepare as in serving.
    """
    rnd = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    outputs = []
    with (
        patch.object(rp, "_BUCKETED_ROWS", bucketed),
        patch.object(rp, "apply_scaling_penalties", _eager_apply_scaling_penalties),
    ):
        batch = _Batch([_random_req(rnd, mixed) for _ in range(rnd.randint(1, 6))])
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, penalizers)
        for _ in range(steps):
            n = len(batch.reqs)
            ids = torch.tensor([rnd.randrange(VOCAB_SIZE) for _ in range(n)])
            orch.cumulate_output_tokens(ids)
            logits = torch.randn(n, VOCAB_SIZE, generator=gen)
            # Speculative layout: two draft positions per request.
            draft_logits = torch.randn(2 * n, VOCAB_SIZE, generator=gen)
            if orch.is_required:
                scaling = orch.accumulate_scaling_penalties()
                if scaling is not None:
                    assert scaling.shape == (n, VOCAB_SIZE), (scaling.shape, n)
                orch.apply(logits)
                orch.apply(draft_logits, repeat=2)
            outputs.append((logits, draft_logits))

            r = rnd.random()
            if r < 0.35 and n > 1:
                # Some requests finish.
                k = rnd.randint(1, min(4, n - 1))
                keep = sorted(rnd.sample(range(n), n - k))
                batch.reqs = [batch.reqs[i] for i in keep]
                orch.filter(torch.tensor(keep, dtype=torch.int64))
            elif r < 0.7 and n < max_batch:
                # A prefilled batch joins.
                other = _Batch(
                    [_random_req(rnd, mixed) for _ in range(rnd.randint(1, 5))]
                )
                other_orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, other, penalizers)
                other_orch.cumulate_output_tokens(
                    torch.tensor([rnd.randrange(VOCAB_SIZE) for _ in other.reqs])
                )
                orch.merge(other_orch)
                batch.reqs = batch.reqs + other.reqs
    return outputs


class TestBucketedRepetitionPenaltyRows(CustomTestCase):
    def test_flag_defaults_off(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_BUCKETED_PENALTY_ROWS", None)
            self.assertFalse(envs.SGLANG_BUCKETED_PENALTY_ROWS.get())

    def test_row_bucket(self):
        expected = {0: 1, 1: 1, 2: 2, 3: 4, 5: 8, 32: 32, 33: 64, 64: 64, 65: 128}
        for rows, bucket in expected.items():
            with self.subTest(rows=rows):
                self.assertEqual(rp._row_bucket(rows), bucket)

    def test_state_keeps_live_rows_first_in_a_row_bucket(self):
        with patch.object(rp, "_BUCKETED_ROWS", True):
            batch = _Batch([_Req(repetition=p) for p in (1.1, 1.2, 1.3)])
            orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, REPETITION_ONLY)
            pen = orch.penalizers[rp.BatchedRepetitionPenalizer]
            self.assertEqual(pen.cumulated_repetition_penalties.shape[0], 4)
            self.assertEqual(
                pen.repetition_penalties.squeeze(1).tolist(),
                torch.tensor([1.1, 1.2, 1.3, 1.0]).tolist(),
            )
            self.assertEqual(pen.get_scaling_penalties().shape, (3, VOCAB_SIZE))

            orch.cumulate_output_tokens(torch.tensor([5, 6, 7]))
            batch.reqs = [batch.reqs[0], batch.reqs[2]]
            orch.filter(torch.tensor([0, 2]))
            self.assertEqual(pen.cumulated_repetition_penalties.shape[0], 2)
            scaling = pen.get_scaling_penalties()
            self.assertEqual(scaling.shape, (2, VOCAB_SIZE))
            self.assertAlmostEqual(scaling[0, 5].item(), 1.1, places=6)
            self.assertAlmostEqual(scaling[1, 7].item(), 1.3, places=6)

            other = _Batch([_Req(repetition=p) for p in (1.4, 1.5, 1.6)])
            other_orch = BatchedPenalizerOrchestrator(
                VOCAB_SIZE, other, REPETITION_ONLY
            )
            orch.merge(other_orch)
            batch.reqs = batch.reqs + other.reqs
            self.assertEqual(pen.cumulated_repetition_penalties.shape[0], 8)
            self.assertEqual(pen.get_scaling_penalties().shape, (5, VOCAB_SIZE))
            self.assertEqual(
                pen.repetition_penalties[:5].squeeze(1).tolist(),
                torch.tensor([1.1, 1.3, 1.4, 1.5, 1.6]).tolist(),
            )

    def test_bucketed_rows_match_exact_rows(self):
        for seed in range(40):
            exact = _run(False, seed, REPETITION_ONLY, False, 60, 40)
            bucketed = _run(True, seed, REPETITION_ONLY, False, 60, 40)
            self._assert_bitwise_equal(exact, bucketed, seed)

    def test_bucketed_rows_match_exact_rows_with_mixed_penalizers(self):
        max_batch = 0
        for seed in range(60):
            exact = _run(False, seed, MIXED, True, 80, 70)
            bucketed = _run(True, seed, MIXED, True, 80, 70)
            self._assert_bitwise_equal(exact, bucketed, seed)
            max_batch = max(max_batch, max(x.shape[0] for x, _ in exact))
        # The traces cross the 32 / 64 row buckets.
        self.assertGreater(max_batch, 64)

    def _assert_bitwise_equal(self, exact, bucketed, seed):
        self.assertEqual(len(exact), len(bucketed))
        for step, (a, b) in enumerate(zip(exact, bucketed)):
            for x, y in zip(a, b):
                self.assertEqual(x.shape, y.shape, (seed, step))
                self.assertTrue(torch.equal(x, y), (seed, step))


if __name__ == "__main__":
    unittest.main()
