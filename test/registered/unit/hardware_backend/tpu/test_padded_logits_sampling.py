"""CPU checks of greedy decode on the padded logits rows
(SGLANG_TPU_PADDED_LOGITS_SAMPLING)."""

import os
import pkgutil
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling import penaltylib
from sglang.srt.sampling.penaltylib import repetition_penalty
from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import padded_logits_sampling as hooks
from sglang_tpu.patches import _unwrap_dynamic_compile_fns
from sglang_tpu.sampling.padded_logits import (
    PaddedRepetitionPenalizer,
    padded_greedy_sample,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

VOCAB_SIZE = 32

# As the TPU platform does at startup (sglang_tpu/patches.py); also lets this
# file run on a TPU host, where the compiled wrapper would pick the tpu backend.
_unwrap_dynamic_compile_fns()


def apply_scaling_penalties(logits, scaling):
    repetition_penalty.apply_scaling_penalties(logits, scaling)


def _make_req(repetition=1.0, freq=0.0, last=5):
    req = MagicMock()
    req.sampling_params.repetition_penalty = repetition
    req.sampling_params.frequency_penalty = freq
    req.output_ids = [last]
    return req


def _make_batch(reqs):
    batch = MagicMock()
    batch.reqs = reqs
    batch.device = "cpu"
    return batch


def _orchestrator(values, penalizers=(PaddedRepetitionPenalizer,)):
    reqs = [_make_req(v) for v in values]
    batch = _make_batch(reqs)
    orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, set(penalizers))
    # The orchestrator holds the batch weakly; keep it alive with the test.
    orch._test_batch = batch
    return orch


class TestHookSpecs(CustomTestCase):
    def test_no_hooks_when_off(self):
        with mock.patch.dict(os.environ, {"SGLANG_TPU_PADDED_LOGITS_SAMPLING": "0"}):
            self.assertEqual(hooks.hook_specs(), [])

    def test_hooks_when_on(self):
        with mock.patch.dict(os.environ, {"SGLANG_TPU_PADDED_LOGITS_SAMPLING": "1"}):
            targets = [target for target, _, _ in hooks.hook_specs()]
        self.assertEqual(len(targets), 5)
        self.assertTrue(targets[0].endswith(".BatchedRepetitionPenalizer"))
        self.assertTrue(targets[-1].endswith("ModelRunner.sample"))
        for target in targets:
            self.assertIsNotNone(pkgutil.resolve_name(target), target)
        # SamplingBatchInfo reads the class through the package at call time,
        # which the class REPLACE rebinds along with the defining module.
        self.assertIs(
            pkgutil.resolve_name(targets[0]), penaltylib.BatchedRepetitionPenalizer
        )


class TestPaddedRepetitionPenalizer(CustomTestCase):
    def test_state_is_padded_with_ones_to_the_bucket(self):
        orch = _orchestrator([1.1, 1.2, 1.3])
        pen = orch.penalizers[PaddedRepetitionPenalizer]
        self.assertEqual(pen.cumulated_repetition_penalties.shape[0], 4)
        self.assertEqual(pen.repetition_penalties[3, 0].item(), 1.0)
        # Padded output ids (the pad row points at column 0) scatter 1.0 there.
        pen.cumulate_output_tokens(torch.tensor([5, 6, 7, 0]))
        cum = pen.cumulated_repetition_penalties
        self.assertAlmostEqual(cum[0, 5].item(), 1.1, places=5)
        self.assertAlmostEqual(cum[2, 7].item(), 1.3, places=5)
        self.assertEqual(cum[3].tolist(), [1.0] * VOCAB_SIZE)

    def test_apply_on_unpadded_logits_matches_stock(self):
        orch = _orchestrator([1.1, 1.2, 1.3])
        pen = orch.penalizers[PaddedRepetitionPenalizer]
        pen.cumulate_output_tokens(torch.tensor([5, 6, 7, 0]))
        logits = torch.randn(3, VOCAB_SIZE)
        expected = logits.clone()
        apply_scaling_penalties(expected, pen.cumulated_repetition_penalties[:3])
        pen.apply(logits)
        self.assertTrue(torch.equal(logits, expected))

    def test_apply_uses_the_module_function_at_call_time(self):
        orch = _orchestrator([1.1, 1.2, 1.3])
        pen = orch.penalizers[PaddedRepetitionPenalizer]
        with mock.patch.object(repetition_penalty, "apply_scaling_penalties") as fn:
            pen.apply(torch.randn(3, VOCAB_SIZE))
        fn.assert_called_once()
        self.assertEqual(fn.call_args.args[1].shape[0], 3)

    def test_filter_and_merge_repad(self):
        orch = _orchestrator([1.1, 1.2, 1.3])
        other = _orchestrator([1.4, 1.5])
        pen = orch.penalizers[PaddedRepetitionPenalizer]
        pen.filter(torch.tensor([0, 2]))
        self.assertEqual(pen.cumulated_repetition_penalties.shape[0], 2)
        pen.merge(other.penalizers[PaddedRepetitionPenalizer])
        # 2 + 2 requests -> bucket 4, real rows in order, no pad rows.
        self.assertEqual(pen.cumulated_repetition_penalties.shape[0], 4)
        self.assertEqual(
            [round(v, 5) for v in pen.repetition_penalties[:, 0].tolist()],
            [1.1, 1.3, 1.4, 1.5],
        )


class TestBatchHooks(CustomTestCase):
    def test_cumulate_pads_last_tokens(self):
        orch = _orchestrator([1.1, 1.2, 1.3])
        batch = orch._test_batch
        batch.sampling_info.penalizer_orchestrator = orch
        original = MagicMock()
        with mock.patch.object(orch, "cumulate_output_tokens") as cumulate:
            hooks.cumulate_penalty_output_tokens(original, batch)
        original.assert_not_called()
        self.assertEqual(cumulate.call_args.args[0].tolist(), [5, 5, 5, 0])

    def test_cumulate_keeps_original_with_another_prepared_penalizer(self):
        reqs = [_make_req(1.1, freq=0.5) for _ in range(3)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {PaddedRepetitionPenalizer, BatchedFrequencyPenalizer}
        )
        batch.sampling_info.penalizer_orchestrator = orch
        original = MagicMock()
        hooks.cumulate_penalty_output_tokens(original, batch)
        original.assert_called_once_with(batch)

    def test_update_penalties_skips_the_additive_accumulator(self):
        orch = _orchestrator([1.1, 1.2, 1.3])
        info = SimpleNamespace(
            penalizer_orchestrator=orch,
            temperatures=torch.ones(3, 1),
            acc_additive_penalties="stale",
            acc_scaling_penalties=None,
        )
        original = MagicMock()
        hooks.update_penalties(original, info)
        original.assert_not_called()
        self.assertIsNone(info.acc_additive_penalties)
        self.assertEqual(info.acc_scaling_penalties.shape[0], 4)

    def test_scaling_is_applied_to_the_logits_rows(self):
        scaling = 1.0 + torch.rand(4, VOCAB_SIZE)
        info = SimpleNamespace(acc_scaling_penalties=scaling)
        logits = torch.randn(3, VOCAB_SIZE)
        seen = []

        def original(info_, logits_):
            seen.append(info_.acc_scaling_penalties.shape[0])

        hooks.apply_pre_grammar_logits_transforms(original, info, logits)
        self.assertEqual(seen, [3])
        self.assertIs(info.acc_scaling_penalties, scaling)


def _sampling_info(scaling):
    return SimpleNamespace(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        sampling_mask_batch_indices=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=scaling,
        penalizer_orchestrator=None,
    )


class TestPaddedGreedySample(CustomTestCase):
    def _run(self, info, logits):
        runner = MagicMock()
        runner.sampling_observer = None
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            return_logprob=False,
            sampling_info=info,
        )
        output = SimpleNamespace(
            next_token_logits=logits,
            has_next_token_global_top_k=False,
            auxiliary_device_output="stale",
        )
        ids = padded_greedy_sample(runner, output, forward_batch)
        return ids, runner, output

    def test_same_tokens_as_the_stock_path(self):
        torch.manual_seed(0)
        base = torch.randn(4, VOCAB_SIZE)
        scaling = torch.ones(4, VOCAB_SIZE)
        scaling[:3] = 1.0 + torch.rand(3, VOCAB_SIZE)
        logits = base[:3]
        expected_logits = logits.clone()
        apply_scaling_penalties(expected_logits, scaling[:3])
        expected = torch.argmax(expected_logits, -1)

        ids, runner, output = self._run(_sampling_info(scaling), logits)

        self.assertTrue(torch.equal(ids, expected))
        self.assertEqual(ids.dtype, torch.int64)
        self.assertIsNone(output.auxiliary_device_output)
        runner.sampler._sync_token_ids_across_tp.assert_called_once()
        runner.ngram_embedding_manager.update_after_decode.assert_called_once()

    def test_declines_non_greedy(self):
        info = _sampling_info(None)
        info.is_all_greedy = False
        ids, _, _ = self._run(info, torch.randn(4, VOCAB_SIZE)[:3])
        self.assertIsNone(ids)

    def test_declines_mismatched_scaling_rows(self):
        info = _sampling_info(torch.ones(3, VOCAB_SIZE))
        ids, _, _ = self._run(info, torch.randn(4, VOCAB_SIZE)[:3])
        self.assertIsNone(ids)


if __name__ == "__main__":
    unittest.main()
