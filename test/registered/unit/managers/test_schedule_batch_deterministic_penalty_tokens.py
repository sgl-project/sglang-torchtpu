import types
import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import schedule_batch as schedule_batch_module  # noqa: E402
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: E402
from sglang.srt.platforms.interface import SRTPlatform  # noqa: E402
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT = [11, 12, 13]
FLAG = envs.SGLANG_DETERMINISTIC_PENALTY_TOKENS


def _req(output_ids, decode_batch_idx, base=0, prefill_last=13):
    req = types.SimpleNamespace(
        rid="req",
        origin_input_ids=list(PROMPT),
        output_ids=list(output_ids),
        decode_batch_idx=decode_batch_idx,
    )
    if base is not None:
        req.pen_output_base = base
        req.pen_prefill_last_token = prefill_last
    return req


def _batch(reqs, seen, overlap=True, spec=SpeculativeAlgorithm.NONE):
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    batch.enable_overlap = overlap
    batch.spec_algorithm = spec
    batch.sampling_info = types.SimpleNamespace(
        penalizer_orchestrator=types.SimpleNamespace(
            is_required=True,
            cumulate_output_tokens=lambda ids: seen.append(ids.tolist()),
            seed_history=lambda rows, tokens: seen.append(("seed", rows, tokens)),
        )
    )
    return batch


class TestDeterministicPenaltyToken(CustomTestCase):
    """SGLANG_DETERMINISTIC_PENALTY_TOKENS: the token penalized at decode step d
    is what a request decoding alone under overlap scheduling gets, whether or not
    the scheduler has already processed its newest sampled token."""

    def test_first_decode_penalizes_last_prefill_input_token(self):
        # Prefill result not processed yet (alone) or already processed (co-batched).
        for outputs in ([], [101]):
            self.assertEqual(
                ScheduleBatch._deterministic_penalty_token(_req(outputs, 0)), 13
            )

    def test_later_steps_penalize_the_token_fed_one_step_earlier(self):
        # Decode step d=2: the token fed at d=1 is output_ids[1]; output_ids may lag
        # the device token by one step (len 2) or not (len 3).
        for outputs in ([101, 102], [101, 102, 103]):
            self.assertEqual(
                ScheduleBatch._deterministic_penalty_token(_req(outputs, 2)), 102
            )

    def test_after_retraction_counts_from_the_reprefill(self):
        # Re-prefilled with 2 earlier outputs: its last input token is output 102,
        # its first new output has index 2.
        req = _req([101, 102], 0, base=2, prefill_last=102)
        self.assertEqual(ScheduleBatch._deterministic_penalty_token(req), 102)
        req = _req([101, 102, 201, 202], 2, base=2, prefill_last=102)
        self.assertEqual(ScheduleBatch._deterministic_penalty_token(req), 202)

    def test_without_prefill_record_falls_back_to_the_parent_rule(self):
        self.assertEqual(
            ScheduleBatch._deterministic_penalty_token(_req([], 0, base=None)), 13
        )
        self.assertEqual(
            ScheduleBatch._deterministic_penalty_token(_req([101, 102], 3, base=None)),
            102,
        )

    def test_cumulate_uses_the_deterministic_token_only_under_overlap(self):
        seen = []
        batch = _batch([_req([], 0), _req([101, 102], 2)], seen)
        with FLAG.override(True):
            batch.enable_overlap = True
            batch.cumulate_penalty_output_tokens()
            batch.enable_overlap = False
            batch.cumulate_penalty_output_tokens()
        with FLAG.override(False):
            batch.enable_overlap = True
            batch.cumulate_penalty_output_tokens()
        self.assertEqual(seen[0], [13, 102])  # flag on, overlap: deterministic
        self.assertEqual(seen[1], [13, 102])  # non-overlap: parent rule (exact there)
        self.assertEqual(seen[2], [13, 102])  # flag off: parent rule

    def test_cumulate_differs_from_the_parent_when_the_result_is_already_processed(
        self,
    ):
        seen = []
        # Co-batched first decode whose prefill result was already processed.
        batch = _batch([_req([101], 0)], seen)
        with FLAG.override(False):
            batch.cumulate_penalty_output_tokens()
        with FLAG.override(True):
            batch.cumulate_penalty_output_tokens()
        self.assertEqual(seen, [[101], [13]])


class TestNotOnHostFallback(CustomTestCase):
    """The token fed one step earlier (output_ids[base + d - 1]) is not on the
    host yet: use the request's prefill token, warn (rate-limited) and count;
    never output_ids[-1], and no IndexError on an empty output_ids."""

    def setUp(self):
        counter = patch.object(schedule_batch_module, "_DET_PENALTY_MISSING", [0])
        counter.start()
        self.addCleanup(counter.stop)

    def _token_and_logs(self, req):
        with self.assertLogs(schedule_batch_module.logger, level="WARNING") as logs:
            token = ScheduleBatch._deterministic_penalty_token(req)
        return token, logs.output

    def test_empty_output_ids(self):
        # d=1 wants output_ids[0]; nothing is on the host yet.
        token, logs = self._token_and_logs(_req([], 1))
        self.assertEqual(token, 13)
        self.assertEqual(schedule_batch_module._DET_PENALTY_MISSING[0], 1)
        self.assertEqual(len(logs), 1)
        self.assertIn("token 0 of request req not on the host", logs[0])
        self.assertIn("(output_ids 0, base 0, step 1)", logs[0])

    def test_too_short_output_ids(self):
        # d=2 wants output_ids[1] (index == len) and d=3 wants output_ids[2]
        # (index > len), with one output on the host; output_ids[-1] would be 101.
        for d in (2, 3):
            token, logs = self._token_and_logs(_req([101], d))
            self.assertEqual(token, 13)
            self.assertIn(f"token {d - 1} of request req not on the host", logs[0])
        self.assertEqual(schedule_batch_module._DET_PENALTY_MISSING[0], 2)

    def test_too_short_after_retraction(self):
        # Re-prefilled with 2 outputs: d=1 wants output_ids[2], not on the host.
        token, logs = self._token_and_logs(
            _req([101, 102], 1, base=2, prefill_last=102)
        )
        self.assertEqual(token, 102)
        self.assertEqual(schedule_batch_module._DET_PENALTY_MISSING[0], 1)
        self.assertIn("(output_ids 2, base 2, step 1)", logs[0])

    def test_warning_is_rate_limited_but_every_miss_is_counted(self):
        req = _req([], 1)
        with self.assertLogs(schedule_batch_module.logger, level="WARNING") as logs:
            tokens = [
                ScheduleBatch._deterministic_penalty_token(req) for _ in range(25)
            ]
        self.assertEqual(tokens, [13] * 25)
        self.assertEqual(len(logs.output), 20)
        for line in logs.output:
            self.assertIn("token 0 of request req not on the host", line)
        self.assertEqual(schedule_batch_module._DET_PENALTY_MISSING[0], 25)

    def test_cumulate_with_a_token_not_on_the_host(self):
        seen = []
        batch = _batch([_req([], 1), _req([101, 102], 2)], seen)
        with (
            FLAG.override(True),
            self.assertLogs(schedule_batch_module.logger, level="WARNING") as logs,
        ):
            batch.cumulate_penalty_output_tokens()
        self.assertEqual(seen, [[13, 102]])
        self.assertEqual(schedule_batch_module._DET_PENALTY_MISSING[0], 1)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("token 0 of request req not on the host", logs.output[0])


class TestSupportedModes(CustomTestCase):
    """The option models overlap decode with one token per step. Without overlap
    or with speculative decoding every new path stays off (the default rule)."""

    def test_enabled_only_for_overlap_without_speculative_decoding(self):
        for flag in (False, True):
            for overlap in (False, True):
                for spec in (SpeculativeAlgorithm.NONE, SpeculativeAlgorithm.EAGLE):
                    batch = _batch([], [], overlap=overlap, spec=spec)
                    with FLAG.override(flag):
                        got = bool(batch._uses_deterministic_penalty_tokens())
                    want = flag and overlap and spec.is_none()
                    self.assertEqual(got, want, (flag, overlap, spec))

    def test_cumulate_keeps_the_parent_rule_for_speculative_decoding(self):
        # Two verify steps that committed several tokens each: the deterministic
        # rule would pick output_ids[1] (102); the parent rule picks 105.
        seen = []
        req = _req([101, 102, 103, 104, 105], 2)
        batch = _batch([req], seen, spec=SpeculativeAlgorithm.EAGLE)
        for flag in (False, True):
            with FLAG.override(flag):
                batch.cumulate_penalty_output_tokens()
        self.assertEqual(seen, [[105], [105]])


class TestSeedRetractedPenaltyHistory(CustomTestCase):
    def test_seeds_the_overlap_history_of_retracted_requests_only(self):
        # Outputs [101, 102]: overlap decode cumulated the last prompt token and
        # 101 before sampling the next token; the next decode step will cumulate
        # 102 as the last input token of the re-prefill. A fresh request is not
        # seeded.
        seen = []
        batch = _batch([], seen)
        reqs = [_req([], 0), _req([101, 102], 0)]
        batch._seed_retracted_penalty_history(reqs)
        self.assertEqual(seen, [("seed", [1], [[13, 101]])])

    def test_no_seed_without_penalties(self):
        seen = []
        batch = _batch([], seen)
        batch.sampling_info.penalizer_orchestrator.is_required = False
        batch._seed_retracted_penalty_history([_req([101, 102], 0)])
        self.assertEqual(seen, [])


class _HostInputPlatform(SRTPlatform):
    def get_batch_input_device(self, batch):
        return "cpu"


def _prefill_again_after_retraction(overlap, spec):
    """Run the real prepare_for_extend for a request retracted after outputs
    [101, 102]; return the seed_history calls and the prefill record."""
    req = Req("req", "", array("q", PROMPT), SamplingParams(), False)
    req.output_ids.extend([101, 102])
    req.refresh_fill_ids()
    req.prefix_indices = torch.arange(0)
    n = len(req.full_untruncated_fill_ids)
    req.set_extend_range(0, n)
    batch = ScheduleBatch(
        reqs=[req],
        device="meta",
        model_config=types.SimpleNamespace(is_encoder_decoder=False, vocab_size=256),
        spec_algorithm=spec,
        enable_overlap=overlap,
    )
    seeds = []
    info = types.SimpleNamespace(
        penalizer_orchestrator=types.SimpleNamespace(
            is_required=True,
            seed_history=lambda rows, tokens: seeds.append((rows, tokens)),
        )
    )

    def allocate(batch):
        return (
            torch.arange(n, dtype=torch.int32),
            torch.empty(1, dtype=torch.int64, device="meta"),
            torch.tensor([0]),
        )

    with (
        get_context().override_server_args(),
        get_parallel().override(attn_dcp_size=1),
        patch.object(
            schedule_batch_module.SamplingBatchInfo,
            "from_schedule_batch",
            return_value=info,
        ),
        patch.object(schedule_batch_module, "current_platform", _HostInputPlatform()),
        patch.object(schedule_batch_module, "alloc_for_extend", allocate),
    ):
        batch.prepare_for_extend()
    record = (
        getattr(req, "pen_output_base", None),
        getattr(req, "pen_prefill_last_token", None),
    )
    return seeds, record


class TestPrepareForExtend(CustomTestCase):
    def test_records_and_seeds_only_in_the_supported_mode(self):
        # The re-prefill input ends with output 102 and its first new output has
        # index 2; the seed is the history of TestSeedRetractedPenaltyHistory.
        for flag in (False, True):
            for overlap in (False, True):
                for spec in (SpeculativeAlgorithm.NONE, SpeculativeAlgorithm.EAGLE):
                    with self.subTest(flag=flag, overlap=overlap, spec=spec.name):
                        with FLAG.override(flag):
                            seeds, record = _prefill_again_after_retraction(
                                overlap, spec
                            )
                        if flag and overlap and spec.is_none():
                            self.assertEqual(seeds, [([0], [[13, 101]])])
                            self.assertEqual(record, (2, 102))
                        else:
                            self.assertEqual(seeds, [])
                            self.assertEqual(record, (None, None))


if __name__ == "__main__":
    unittest.main()
