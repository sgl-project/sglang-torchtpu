"""True on-policy mode on Qwen3-0.6B: decode log-probs equal the prefill re-score of the same tokens, bit for bit."""

import importlib.util
import unittest

from true_on_policy_base import TrueOnPolicyTestCase

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=300,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "accuracy", "logprob", "seq-along-lane"],
)


@unittest.skipIf(
    importlib.util.find_spec("torch_tpu") is None, "torch_tpu not available"
)
class TestTpuQwen306bSinglehostTrueOnPolicy(TrueOnPolicyTestCase):
    model_path = "Qwen/Qwen3-0.6B"

    def test_decode_log_probs_equal_the_prefill_rescore(self):
        self.assert_decode_equals_prefill_rescore(temperature=1.0)

    def test_greedy_log_probs_equal_the_prefill_rescore(self):
        self.assert_decode_equals_prefill_rescore(temperature=0.0)

    def test_top_p_sampling_is_refused(self):
        self.assert_top_p_sampling_is_refused()


if __name__ == "__main__":
    unittest.main()
