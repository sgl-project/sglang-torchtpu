"""Output TopK backends preserve scoring and leave prompt scoring unchanged."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.kernels.fused_op as fused_op
from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.sampling import OutputTopK, output_topk
from sglang.srt.layers import logprob_processor
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOutputTopkBackend(CustomTestCase):
    def setUp(self):
        patches = [
            patch.dict(BaseFusedOp._oot_forward_registry, {}, clear=True),
            patch.object(output_topk, "_forward_method", None),
            patch.object(fused_op, "_forced_backend", None),
            patch.object(fused_op, "_platform_key", lambda: "cpu"),
            patch.object(fused_op, "_oot_dispatch_key", lambda: None),
        ]
        for patched in patches:
            patched.start()
            self.addCleanup(patched.stop)
        self.logits = torch.arange(34, dtype=torch.float32).reshape(2, 17)
        self.topk = Mock(side_effect=lambda values, k: values.topk(k, dim=-1))

    def register_topk(self, platform_key="cpu"):
        BaseFusedOp.register_oot_forward(
            OutputTopK, lambda op, values, k: self.topk(values, k), platform_key
        )

    def test_unregistered_device_keeps_tensor_topk(self):
        self.register_topk("other_device")
        values, indices = logprob_processor.get_top_logprobs(
            self.logits, [3, 0], no_copy_to_cpu=True
        )
        expected = self.logits.topk(3, dim=-1)
        self.assertTrue(torch.equal(values[0], expected.values[0]))
        self.assertTrue(torch.equal(indices[0], expected.indices[0]))
        self.assertEqual(indices[1].numel(), 0)
        self.topk.assert_not_called()

    def test_registered_output_processors(self):
        self.register_topk()
        processor = logprob_processor.OutputLogprobProcessor()
        scores = self.logits.log_softmax(-1)
        expected = scores.topk(5, dim=-1)
        for scoring_only in (False, True):
            with self.subTest(scoring_only=scoring_only):
                self.topk.reset_mock()
                if scoring_only:
                    result = processor.compute_logprobs_only(
                        self.logits, [5, 0], [[1, 4], None], preprocess_fn=lambda x: x
                    )
                else:
                    result = processor.compute_logprobs(
                        scores, [5, 0], [[1, 4], None], torch.tensor([2, 3])
                    )
                    self.assertTrue(
                        torch.equal(result.token_logprobs, scores[[0, 1], [2, 3]])
                    )
                self.topk.assert_called_once()
                self.assertEqual(self.topk.call_args.args[1], 5)
                self.assertTrue(
                    torch.equal(result.top_logprobs_val[0], expected.values[0])
                )
                self.assertTrue(
                    torch.equal(result.top_logprobs_idx[0], expected.indices[0])
                )
                self.assertEqual(result.top_logprobs_idx[0].dtype, torch.int64)
                self.assertEqual(result.top_logprobs_val[1].numel(), 0)
                self.assertTrue(
                    torch.equal(result.token_ids_logprobs_val[0], scores[0, [1, 4]])
                )

    def test_registered_spec_scoring(self):
        self.register_topk()
        logits = self.logits.repeat(2, 1)
        output = LogitsProcessorOutput(next_token_logits=logits)
        batch = SimpleNamespace(
            seq_lens=[1, 1],
            sampling_info=SimpleNamespace(is_all_greedy=True),
            top_logprobs_nums=[3, 0],
            token_ids_logprobs=[[1, 4], None],
        )
        token_ids = torch.tensor([2, 3, 4, 5])
        logprob_processor.compute_spec_logprobs(
            batch, output, token_ids, chain_stride=2
        )
        self.topk.assert_called_once()
        self.assertEqual(self.topk.call_args.args[1], 3)
        scores = logits.log_softmax(-1)
        expected = scores.topk(3, dim=-1)
        for row, k in enumerate([3, 3, 0, 0]):
            self.assertTrue(
                torch.equal(
                    output.next_token_top_logprobs_val[row], expected.values[row, :k]
                )
            )
            self.assertTrue(
                torch.equal(
                    output.next_token_top_logprobs_idx[row], expected.indices[row, :k]
                )
            )
        self.assertTrue(
            torch.equal(
                output.next_token_logprobs,
                scores[torch.arange(4), token_ids].reshape(2, 2),
            )
        )
        self.assertTrue(
            torch.equal(output.next_token_token_ids_logprobs_val[1], scores[1, [1, 4]])
        )

    def test_prompt_and_zero_k_bypass_registered_backend(self):
        self.register_topk()
        values, indices = logprob_processor.get_top_logprobs_raw(
            self.logits, [3, 0], logprob_processor.LogprobStage.PREFILL, [1, 1]
        )
        expected = self.logits.topk(3, dim=-1)
        self.assertEqual(values, [[expected.values[0].tolist()], [[]]])
        self.assertEqual(indices, [[expected.indices[0].tolist()], [[]]])
        self.assertEqual(
            logprob_processor.get_top_logprobs(self.logits, [0, 0]),
            ([[], []], [[], []]),
        )
        self.topk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
