"""Unit tests for the one-batch KV-capacity skip with ragged --input-len-list batches."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.benchmark import one_batch_server as bench
from sglang.benchmark.one_batch_server import should_skip_due_to_token_capacity
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestRaggedBatch(CustomTestCase):
    def test_lengths_cycle_and_reject_invalid_entries(self):
        self.assertEqual(bench._batch_input_lens("3, 6", 3), [3, 6, 3])
        for lengths in ("0,3", "3,-1", "3,", "2.5"):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                bench._parse_input_len_list(lengths)

    def test_request_payload_and_exact_input_accounting(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            (
                "data: "
                + json.dumps(
                    {
                        "index": index,
                        "meta_info": {
                            "completion_tokens": count,
                            "finish_reason": {"type": "length"} if count == 2 else None,
                        },
                    }
                )
            ).encode()
            for count in (1, 2)
            for index in (0, 1)
        ]
        rows = [SimpleNamespace(prompt=[1, 2, 3, 4]), SimpleNamespace(prompt=[4, 5, 6])]
        with (
            patch.object(bench, "get_dataset", return_value=rows),
            patch.object(bench, "_flush_cache_with_retry"),
            patch.object(bench, "get_cache_tokens_from_metrics", return_value=None),
            patch.object(bench.requests, "post", return_value=response) as post,
            patch.object(bench.requests, "get") as get,
            patch.object(bench.time, "perf_counter", side_effect=range(6)),
        ):
            get.return_value.json.return_value = {"internal_states": []}
            result = bench.run_one_case(
                "http://unused",
                2,
                4,
                2,
                temperature=0,
                ignore_eos=True,
                return_logprob=False,
                stream_interval=1,
                input_len_step_percentage=0,
                run_name="ragged",
                result_filename="",
                tokenizer=SimpleNamespace(),
                dataset_name="random-ids",
                input_len_list="3,6",
                fake_prefill=True,
            )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["input_ids"], [[1, 2, 3], [4, 5, 6, 4, 5, 6]])
        self.assertEqual(payload["bootstrap_host"], bench.FAKE_BOOTSTRAP_HOST)
        self.assertEqual(result.input_len, 4)
        self.assertEqual(result.total_output_tokens, 4)
        self.assertAlmostEqual(result.input_throughput * result.last_ttft, 9)
        self.assertAlmostEqual(result.overall_throughput * result.latency, 13)


if __name__ == "__main__":
    unittest.main()
