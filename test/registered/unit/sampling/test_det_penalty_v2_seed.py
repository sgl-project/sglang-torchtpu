"""SGLANG_TPU_DETERMINISTIC_PENALTY_TOKENS v2 (penaltylib seed_history): a request
prefilled again after a retraction gets a fresh penalizer row. seed_history must
leave that row with the bits the step-by-step path gives (cumulate_output_tokens
once per decode step), for every penalizer, and leave other rows untouched, so
the request's later tokens do not depend on when it was retracted. No server."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch
from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer
from sglang.test.test_utils import CustomTestCase

VOCAB = 97
PENALIZERS = {
    BatchedRepetitionPenalizer,
    BatchedFrequencyPenalizer,
    BatchedPresencePenalizer,
    BatchedMinNewTokensPenalizer,
}
STATE = (
    "cumulated_repetition_penalties",
    "cumulated_frequency_penalties",
    "cumulated_presence_penalties",
    "len_output_tokens",
)


def _req(rep, freq, pres, min_new):
    req = MagicMock()
    req.sampling_params.repetition_penalty = rep
    req.sampling_params.frequency_penalty = freq
    req.sampling_params.presence_penalty = pres
    req.sampling_params.min_new_tokens = min_new
    req.sampling_params.stop_token_ids = {3}
    req.eos_token_ids = {5}
    req.tokenizer.additional_stop_token_ids = None
    req.tokenizer.eos_token_id = 7
    return req


class TestSeedHistory(CustomTestCase):
    def setUp(self):
        self._batches = []  # the orchestrator holds its batch weakly

    def _orch(self, reqs):
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = "cpu"
        self._batches.append(batch)
        return BatchedPenalizerOrchestrator(VOCAB, batch, PENALIZERS)

    def _row(self, orch, row):
        out = {}
        for pen in orch.penalizers.values():
            for name in STATE:
                if hasattr(pen, name):
                    value = getattr(pen, name)[row]
                    out[name] = (
                        value.view(torch.int32)
                        if value.dtype == torch.float32
                        else value
                    ).clone()
        return out

    def _reference(self, req, tokens):
        # The step-by-step path for one request. A filler row with every
        # penalty set keeps the same penalizers prepared as in the batch
        # under test; only row 0 is compared.
        o = self._orch([req, _req(1.4, 0.9, 0.6, 50)])
        for t in tokens:
            o.cumulate_output_tokens(torch.tensor([t, 0]))
        return self._row(o, 0)

    def _assert_rows_equal(self, got, want, what):
        self.assertEqual(sorted(got), sorted(want), what)
        for name in got:
            self.assertTrue(torch.equal(got[name], want[name]), f"{what}: {name}")

    def test_seed_equals_step_by_step(self):
        reqs = [
            _req(1.1, 0.3, 0.2, 4),
            _req(1.3, 0.7, 0.0, 0),
            _req(1.05, 0.1, 0.5, 30),
            _req(1.2, 0.33, 0.25, 12),
        ]
        g = torch.Generator().manual_seed(0)
        # row 0 fresh; rows 1-3 retracted, histories with repeated tokens
        hist = [
            torch.randint(0, VOCAB, (n,), generator=g).tolist() for n in (0, 1, 9, 23)
        ]
        hist[3] += hist[3][:5]  # force repeats (frequency counts > 1)
        orch = self._orch(reqs)
        rows = [i for i, h in enumerate(hist) if h]
        orch.seed_history(rows, [hist[i] for i in rows])
        for i, req in enumerate(reqs):
            self._assert_rows_equal(
                self._row(orch, i), self._reference(req, hist[i]), f"row {i}"
            )

    def test_seeded_rows_continue_like_uninterrupted(self):
        reqs = [_req(1.1, 0.3, 0.2, 4), _req(1.2, 0.4, 0.1, 8)]
        hist = [[1, 2, 2, 9], [4, 4, 4]]
        more = [[9, 11, 2], [4, 13, 13]]
        orch = self._orch(reqs)
        orch.seed_history([0, 1], hist)
        for k in range(3):
            orch.cumulate_output_tokens(torch.tensor([more[0][k], more[1][k]]))
        for i, req in enumerate(reqs):
            self._assert_rows_equal(
                self._row(orch, i),
                self._reference(req, hist[i] + more[i]),
                f"row {i}",
            )

    def test_unprepared_penalizers_are_skipped(self):
        # no penalty set: nothing is prepared and seeding is a no-op
        orch = self._orch([_req(1.0, 0.0, 0.0, 0)])
        orch.seed_history([0], [[1, 2, 3]])
        for pen in orch.penalizers.values():
            self.assertFalse(pen.is_prepared())


if __name__ == "__main__":
    unittest.main()
