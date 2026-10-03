"""Hardware-free tests for the KV-offload probe's speculative gate."""

import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.kv_offload_probe import (
    CacheObservation,
    KvOffloadProbe,
    generate_with_cache_report,
    mean_accept_length,
    spec_failures,
    validate_probe,
)

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


def _read(*, completion_tokens: int = 8, spec_verify_ct: int = 4):
    return CacheObservation(
        text="10 11 12 13",
        device_tokens=0,
        host_tokens=0,
        completion_tokens=completion_tokens,
        spec_verify_ct=spec_verify_ct,
    )


def _probe(*, cold=None, warm=None, reload=None):
    return KvOffloadProbe(
        cold=cold or _read(),
        warm=warm or _read(),
        reload=reload or _read(),
    )


class TestSpeculativeKvOffloadProbe(unittest.TestCase):
    def test_mean_accept_length_aggregates_all_reads(self) -> None:
        probe = _probe(
            cold=_read(completion_tokens=8, spec_verify_ct=4),
            warm=_read(completion_tokens=9, spec_verify_ct=3),
            reload=_read(completion_tokens=7, spec_verify_ct=1),
        )
        self.assertEqual(mean_accept_length(probe=probe), 3.0)

    def test_mean_accept_length_is_zero_without_verify_forwards(self) -> None:
        probe = _probe(
            cold=_read(spec_verify_ct=0),
            warm=_read(spec_verify_ct=0),
            reload=_read(spec_verify_ct=0),
        )
        self.assertEqual(mean_accept_length(probe=probe), 0.0)

    def test_each_read_must_run_speculative_decoding(self) -> None:
        failures = spec_failures(probe=_probe(reload=_read(spec_verify_ct=0)))
        self.assertEqual(len(failures), 1)
        self.assertIn("host-reload", failures[0])

    def test_host_reload_must_cost_the_same_verify_forwards(self) -> None:
        failures = spec_failures(
            probe=_probe(
                warm=_read(spec_verify_ct=5),
                reload=_read(spec_verify_ct=13),
            )
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("the host reload changed speculative decoding", failures[0])

    def test_cold_read_may_cost_a_different_number_of_forwards(self) -> None:
        probe = _probe(
            cold=_read(spec_verify_ct=12),
            warm=_read(spec_verify_ct=5),
            reload=_read(spec_verify_ct=5),
        )
        self.assertEqual(spec_failures(probe=probe), [])

    def test_matching_cached_reads_pass(self) -> None:
        self.assertEqual(spec_failures(probe=_probe()), [])


class TestKvOffloadOutputOracle(unittest.TestCase):
    def _tiered_probe(self, *, warm_text: str, reload_text: str) -> KvOffloadProbe:
        return KvOffloadProbe(
            cold=_read(),
            warm=CacheObservation(warm_text, 960, 0, 8, 0),
            reload=CacheObservation(reload_text, 0, 960, 8, 0),
        )

    def test_equivalent_cache_reads_pass(self) -> None:
        failures = validate_probe(
            probe=self._tiered_probe(
                warm_text="not a count", reload_text="not a count"
            ),
            page_size=64,
        )
        self.assertEqual(failures, [])

    def test_changed_host_reload_fails(self) -> None:
        failures = validate_probe(
            probe=self._tiered_probe(warm_text="device", reload_text="host"),
            page_size=64,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("host reload changed", failures[0])

    def test_two_empty_reads_are_not_a_match(self) -> None:
        failures = validate_probe(
            probe=self._tiered_probe(warm_text="", reload_text=""),
            page_size=64,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("must be non-empty", failures[0])


class TestDpRouting(unittest.TestCase):
    @patch("sglang.test.tpu.utils.kv_offload_probe.requests.post")
    def test_generate_routes_to_requested_dp_rank(self, post: Mock) -> None:
        response = post.return_value
        response.json.return_value = {
            "text": "ok",
            "meta_info": {
                "dp_rank": 3,
                "cached_tokens": 64,
                "cached_tokens_details": {"device": 64, "host": 0},
                "completion_tokens": 1,
            },
        }

        observation = generate_with_cache_report(
            base_url="http://localhost:30000",
            prompt="prompt",
            max_new_tokens=1,
            routed_dp_rank=3,
        )

        self.assertEqual(observation.device_tokens, 64)
        self.assertEqual(post.call_args.kwargs["json"]["routed_dp_rank"], 3)

    @patch("sglang.test.tpu.utils.kv_offload_probe.requests.post")
    def test_generate_rejects_wrong_dp_rank(self, post: Mock) -> None:
        post.return_value.json.return_value = {
            "text": "ok",
            "meta_info": {
                "dp_rank": 2,
                "cached_tokens": 0,
                "completion_tokens": 1,
            },
        }

        with self.assertRaisesRegex(RuntimeError, "response reports dp_rank=2"):
            generate_with_cache_report(
                base_url="http://localhost:30000",
                prompt="prompt",
                max_new_tokens=1,
                routed_dp_rank=3,
            )


if __name__ == "__main__":
    unittest.main()
