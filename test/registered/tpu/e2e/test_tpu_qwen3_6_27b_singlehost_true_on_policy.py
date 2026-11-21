"""True on-policy mode on four layers of Qwen3.6-27B: a decode step equals a prefill of the same prefix."""

import importlib.util
import unittest

from true_on_policy_base import TrueOnPolicyTestCase

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3.6-27b", "accuracy", "logprob", "seq-along-lane"],
    disabled="Pending Qwen3.6-27B checkpoint staging on shared CI PVC",
)


@unittest.skipIf(
    importlib.util.find_spec("torch_tpu") is None, "torch_tpu not available"
)
class TestTpuQwen3627bSinglehostTrueOnPolicy(TrueOnPolicyTestCase):
    model_path = resolve_tpu_model_path("Qwen/Qwen3.6-27B")
    vision_language = True
    model_args = (
        "--json-model-override-args",
        '{"language_model_only": true, "text_config": {"num_hidden_layers": 4}}',
    )

    def test_decode_log_probs_equal_the_prefix_prefill(self):
        self.assert_decode_equals_prefix_prefill()


if __name__ == "__main__":
    unittest.main()
