"""True on-policy mode on four layers of Qwen3.5-35B-A3B: a decode step equals a prefill of the same prefix."""

import importlib.util
import unittest

from true_on_policy_base import TrueOnPolicyTestCase

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3.5-35b-a3b", "accuracy", "logprob", "seq-along-lane"],
)


@unittest.skipIf(
    importlib.util.find_spec("torch_tpu") is None, "torch_tpu not available"
)
class TestTpuQwen3535bA3bSinglehostTrueOnPolicy(TrueOnPolicyTestCase):
    model_path = resolve_tpu_model_path("Qwen/Qwen3.5-35B-A3B")
    vision_language = True
    model_args = (
        "--json-model-override-args",
        '{"language_model_only": true, "text_config": {"num_hidden_layers": 4}}',
    )

    def test_decode_log_probs_equal_the_prefix_prefill(self):
        self.assert_decode_equals_prefix_prefill()

    def test_single_request_decode_equals_the_prefix_prefill(self):
        # a one-row decode step runs the router and the shared-expert gate on a single row
        self.assert_decode_equals_prefix_prefill(concurrency=1)


if __name__ == "__main__":
    unittest.main()
