import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch as schedule_batch_module  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT = [11, 12, 13]


def _req(output_ids, decode_batch_idx, base=0, prefill_last=13):
    req = types.SimpleNamespace(
        origin_input_ids=list(PROMPT),
        output_ids=list(output_ids),
        decode_batch_idx=decode_batch_idx,
    )
    if base is not None:
        req.pen_output_base = base
        req.pen_prefill_last_token = prefill_last
    return req


class TestDeterministicPenaltyToken(CustomTestCase):
    """SGLANG_TPU_DETERMINISTIC_PENALTY_TOKENS: the token penalized at decode step d
    is what a request decoding alone under overlap scheduling gets, whether or not
    the scheduler has already processed its newest sampled token."""

    def test_first_decode_penalizes_last_prefill_input_token(self):
        # Prefill result not processed yet (alone) or already processed (co-batched).
        for outputs in ([], [101]):
            self.assertEqual(ScheduleBatch._deterministic_penalty_token(_req(outputs, 0)), 13)

    def test_later_steps_penalize_the_token_fed_one_step_earlier(self):
        # Decode step d=2: the token fed at d=1 is output_ids[1]; output_ids may lag
        # the device token by one step (len 2) or not (len 3).
        for outputs in ([101, 102], [101, 102, 103]):
            self.assertEqual(ScheduleBatch._deterministic_penalty_token(_req(outputs, 2)), 102)

    def test_after_retraction_counts_from_the_reprefill(self):
        # Re-prefilled with 2 earlier outputs: its last input token is output 102,
        # its first new output has index 2.
        req = _req([101, 102], 0, base=2, prefill_last=102)
        self.assertEqual(ScheduleBatch._deterministic_penalty_token(req), 102)
        req = _req([101, 102, 201, 202], 2, base=2, prefill_last=102)
        self.assertEqual(ScheduleBatch._deterministic_penalty_token(req), 202)

    def test_without_prefill_record_falls_back_to_the_parent_rule(self):
        self.assertEqual(ScheduleBatch._deterministic_penalty_token(_req([], 0, base=None)), 13)
        self.assertEqual(
            ScheduleBatch._deterministic_penalty_token(_req([101, 102], 3, base=None)), 102
        )

    def test_cumulate_uses_the_deterministic_token_only_under_overlap(self):
        seen = []
        batch = ScheduleBatch(reqs=[_req([], 0), _req([101, 102], 2)])
        batch.device = "cpu"
        batch.sampling_info = types.SimpleNamespace(
            penalizer_orchestrator=types.SimpleNamespace(
                cumulate_output_tokens=lambda ids: seen.append(ids.tolist())
            )
        )
        with patch.object(schedule_batch_module, "_TPU_DETERMINISTIC_PENALTY_TOKENS", True):
            batch.enable_overlap = True
            batch.cumulate_penalty_output_tokens()
            batch.enable_overlap = False
            batch.cumulate_penalty_output_tokens()
        with patch.object(schedule_batch_module, "_TPU_DETERMINISTIC_PENALTY_TOKENS", False):
            batch.enable_overlap = True
            batch.cumulate_penalty_output_tokens()
        self.assertEqual(seen[0], [13, 102])  # flag on, overlap: deterministic
        self.assertEqual(seen[1], [13, 102])  # non-overlap: parent rule (exact there)
        self.assertEqual(seen[2], [13, 102])  # flag off: parent rule

    def test_cumulate_differs_from_the_parent_when_the_result_is_already_processed(self):
        seen = []
        # Co-batched first decode whose prefill result was already processed.
        batch = ScheduleBatch(reqs=[_req([101], 0)])
        batch.device = "cpu"
        batch.enable_overlap = True
        batch.sampling_info = types.SimpleNamespace(
            penalizer_orchestrator=types.SimpleNamespace(
                cumulate_output_tokens=lambda ids: seen.append(ids.tolist())
            )
        )
        with patch.object(schedule_batch_module, "_TPU_DETERMINISTIC_PENALTY_TOKENS", False):
            batch.cumulate_penalty_output_tokens()
        with patch.object(schedule_batch_module, "_TPU_DETERMINISTIC_PENALTY_TOKENS", True):
            batch.cumulate_penalty_output_tokens()
        self.assertEqual(seen, [[101], [13]])


if __name__ == "__main__":
    unittest.main()
