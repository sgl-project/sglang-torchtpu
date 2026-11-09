"""Unit tests for the one-batch KV-capacity skip with ragged --input-len-list batches."""

import unittest
from unittest.mock import Mock, patch

from sglang.benchmark import one_batch_server
from sglang.benchmark.one_batch_server import should_skip_due_to_token_capacity
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRaggedTokenCapacity(CustomTestCase):
    def test_uniform_batch_counts_the_total(self):
        self.assertFalse(should_skip_due_to_token_capacity(4, 100, 10, 440))
        self.assertTrue(should_skip_due_to_token_capacity(4, 100, 10, 439))

    def test_ragged_batch_sums_its_lengths(self):
        self.assertFalse(should_skip_due_to_token_capacity(4, 0, 10, 440, "50,150"))
        self.assertTrue(should_skip_due_to_token_capacity(4, 0, 10, 439, "50,150"))

    def test_heaviest_dp_rank_must_fit_its_share(self):
        # Round-robin over 2 ranks: rank 0 holds 300 + 300, rank 1 holds 100 + 100.
        lens = "300,100"
        self.assertFalse(should_skip_due_to_token_capacity(4, 0, 0, 800, lens))
        self.assertTrue(
            should_skip_due_to_token_capacity(4, 0, 0, 800, lens, num_dp_ranks=2)
        )
        self.assertFalse(
            should_skip_due_to_token_capacity(4, 0, 0, 1200, lens, num_dp_ranks=2)
        )

    def test_uniform_batch_ignores_dp_ranks(self):
        self.assertFalse(
            should_skip_due_to_token_capacity(3, 100, 0, 300, num_dp_ranks=2)
        )

    def test_warmup_only_runs_batches_with_a_measurable_shape(self):
        cases = (
            ({}, 200, 8, 1, {1}),
            ({"input_len": (1, 100)}, 200, 8, 1, {1, 4}),
            ({}, 1000, 2, 1, {1}),
            ({"enable_multi_batch": True}, 250, 2, 1, {1, 4}),
            ({"input_len_list": "150,10"}, 500, 8, 2, {1}),
        )
        for overrides, capacity, running_cap, dp, expected in cases:
            with self.subTest(overrides=overrides, capacity=capacity, cap=running_cap):
                bench = one_batch_server.BenchArgs(
                    batch_size=(1, 4),
                    **{"input_len": (100,), "output_len": (1,), **overrides},
                )
                response = Mock()
                response.json.return_value = {
                    "tokenizer_path": "unused",
                    "internal_states": [
                        {
                            "dp_size": dp,
                            "effective_max_running_requests_per_dp": running_cap,
                            "memory_usage": {"token_capacity": capacity},
                        }
                    ],
                }
                with (
                    patch.object(one_batch_server, "acquire_endpoint"),
                    patch.object(one_batch_server, "get_tokenizer"),
                    patch.object(
                        one_batch_server.requests, "get", return_value=response
                    ),
                    patch.object(one_batch_server, "run_one_case") as run_case,
                ):
                    one_batch_server.run_benchmark_internal(None, bench)
                warmup_batches = {
                    call.kwargs["batch_size"]
                    for call in run_case.call_args_list
                    if call.kwargs["run_name"] == ""
                }
                self.assertEqual(warmup_batches, expected)


if __name__ == "__main__":
    unittest.main()
