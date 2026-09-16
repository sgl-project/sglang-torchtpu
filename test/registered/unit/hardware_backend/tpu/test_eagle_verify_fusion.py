"""Acceptance, padding and output ownership across fused TPU verification."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.eagle_verify import (
    try_verify_and_bookkeep_tpu,
    verify_and_bookkeep_tpu,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _inputs(bs, width):
    winners = torch.arange(bs * width).reshape(bs, width) % 7
    logits = torch.full((bs * width, 11), -10.0)
    logits.scatter_(1, winners.flatten()[:, None], 10.0)
    candidates = torch.cat((torch.zeros(bs, 1, dtype=torch.int64), winners[:, :-1]), 1)
    for row in range(bs):
        if width > 1:
            candidates[row, row % (width - 1) + 1] = 10
    indices = torch.arange(bs * width, dtype=torch.int64).reshape(bs, width)
    seq_lens = torch.arange(bs, dtype=torch.int64) + 10
    return logits, candidates.flatten(), indices, seq_lens


def _oracle(logits, candidates, indices, seq_lens, forced):
    bs, width = indices.shape
    winners = logits.argmax(-1).reshape(bs, width).tolist()
    candidates = candidates.reshape(bs, width).tolist()
    predict = [token for row in winners for token in row]
    lengths, paths = [], []
    for row in range(bs):
        length = 1
        while length < width and winners[row][length - 1] == candidates[row][length]:
            length += 1
        for col in range(length):
            if indices[row, col] == -1:
                length = col
                break
        if forced:
            length = abs(forced)
            for col in range(length - 1):
                predict[row * width + col] = candidates[row][col + 1]
        lengths.append(length)
        paths.append(
            [int(indices[row, col]) if col < length else -1 for col in range(width)]
        )
    if forced < 0:
        predict = [100] * len(predict)
    gathered = [predict[index] for path in paths for index in path]
    bonus = [gathered[row * width + length - 1] for row, length in enumerate(lengths)]
    return (
        torch.tensor(predict, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        torch.tensor(paths, dtype=torch.int32),
        seq_lens + torch.tensor(lengths),
        torch.tensor(bonus, dtype=torch.int64),
    )


class TestEagleVerifyFusion(CustomTestCase):
    def test_natural_and_forced_acceptance_preserve_final_state(self):
        for bs, width in ((1, 1), (1, 4), (4, 4), (16, 4)):
            inputs = _inputs(bs, width)
            saved = tuple(t.clone() for t in inputs)
            for count in (0, *range(1, width + 1), -width):
                with self.subTest(bs=bs, width=width, count=count):
                    result = verify_and_bookkeep_tpu(
                        *inputs, torch.tensor(count, dtype=torch.int32)
                    )
                    expected = _oracle(*inputs, count)
                    for actual, reference in zip(result, expected):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    for original, before in zip(inputs, saved):
                        torch.testing.assert_close(original, before, rtol=0, atol=0)

    def test_invalid_indices_stop_natural_acceptance_without_host_reads(self):
        inputs = _inputs(4, 4)
        inputs[2][1, 1:] = -1
        inputs[2][2, :] = -1
        compiled = torch.compile(
            verify_and_bookkeep_tpu.__wrapped__, backend="aot_eager", fullgraph=True
        )
        result = compiled(*inputs, torch.tensor(0, dtype=torch.int32))
        for actual, reference in zip(result, _oracle(*inputs, 0)):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_padded_simulation_uses_fallback_without_consuming_rng(self):
        sampling = SimpleNamespace(
            is_all_greedy=True,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            sampling_info=sampling,
            return_logprob=False,
        )
        verify = SimpleNamespace(
            tree_topk=1,
            padding_state=SimpleNamespace(logical_batch_size=3, physical_batch_size=4),
        )
        with patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ), patch("sglang.srt.speculative.spec_utils.SIMULATE_ACC_LEN", 2.5), patch(
            "sglang.srt.speculative.spec_utils.sample_simulated_acc_len"
        ) as sample:
            self.assertIsNone(
                try_verify_and_bookkeep_tpu(
                    batch=batch,
                    verify_input=verify,
                    logits_output=None,
                    grammar_mask=None,
                )
            )
            sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
