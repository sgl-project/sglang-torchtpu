"""Hardware-free tests for the KV-offload probe."""

import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.kv_offload_probe import (
    CacheObservation,
    KvOffloadProbe,
    TailReplay,
    generate_with_cache_report,
    kv_offload_runner,
    mean_accept_length,
    plan_eviction,
    run_probe,
    spec_failures,
    tail_failures,
    tail_shapes,
    validate_probe,
)

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


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


class TestDecodeTails(unittest.TestCase):
    def _replay(self, *, device: int, host: int, logprobs=(-0.72, -1.5)):
        return CacheObservation(
            "x", device, host, len(logprobs), 0, (11, 12), tuple(logprobs)
        )

    def _tail(self, *, device=None, host=None) -> TailReplay:
        return TailReplay(
            committed_tokens=128,
            device=device or self._replay(device=128, host=0),
            host=host or self._replay(device=0, host=128),
        )

    def test_every_shape_ends_on_a_page_boundary_at_its_own_step(self) -> None:
        shapes = tail_shapes(page_size=64)
        self.assertTrue(all((p + n) % 64 == 0 for p, n in shapes))
        self.assertEqual(len({n for _, n in shapes}), len(shapes))

    def test_replays_must_not_complete_a_page(self) -> None:
        with self.assertRaises(ValueError):
            tail_shapes(page_size=6)
        tail_shapes(page_size=7)

    def test_spec_and_dp_are_not_qualified(self) -> None:
        kwargs = dict(
            test_name="t",
            device_pool_tokens=8192,
            page_size=64,
            probe_tokens=1024,
            hicache_ratio=3.0,
            decode_tails=True,
        )
        kv_offload_runner(**kwargs)
        with self.assertRaises(ValueError):
            kv_offload_runner(**kwargs, spec_decoding=True)
        with self.assertRaises(ValueError):
            kv_offload_runner(**kwargs, dp_size=2)

    def test_tails_count_against_the_host_pool(self) -> None:
        kwargs = dict(device_pool_tokens=8192, probe_tokens=1024, hicache_ratio=1.7)
        plan_eviction(**kwargs)
        with self.assertRaisesRegex(ValueError, "decode tails"):
            plan_eviction(**kwargs, tail_tokens=1024)

    def test_swa_host_pool_bounds_the_fillers(self) -> None:
        kwargs = dict(device_pool_tokens=8192, probe_tokens=1024, hicache_ratio=3.0)
        self.assertEqual(plan_eviction(**kwargs).host_pool_tokens, 24576)
        # The SWA host pool is the ratio times the smaller SWA device pool.
        self.assertEqual(
            plan_eviction(**kwargs, swa_full_tokens_ratio=0.8).host_pool_tokens,
            19660,
        )
        with self.assertRaisesRegex(ValueError, "evicted from host"):
            plan_eviction(**kwargs, swa_full_tokens_ratio=0.5)

    def test_matching_replays_pass(self) -> None:
        self.assertEqual(tail_failures(tails=[self._tail()]), [])

    def test_changed_logprob_fails(self) -> None:
        tail = self._tail(host=self._replay(device=0, host=128, logprobs=(-0.69, -1.5)))
        failures = tail_failures(tails=[tail])
        self.assertEqual(len(failures), 1)
        self.assertIn("changed the replay", failures[0])

    def test_empty_replays_are_not_a_match(self) -> None:
        empty = CacheObservation("", 128, 0, 0, 0)
        tail = TailReplay(
            committed_tokens=128,
            device=empty,
            host=CacheObservation("", 0, 128, 0, 0),
        )
        failures = tail_failures(tails=[tail])
        self.assertEqual(len(failures), 1)
        self.assertIn("changed the replay", failures[0])

    def test_device_replay_must_cover_the_tail_page(self) -> None:
        failures = tail_failures(
            tails=[self._tail(device=self._replay(device=64, host=0))]
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("nothing was checked", failures[0])

    def test_host_replay_must_come_from_host(self) -> None:
        failures = tail_failures(
            tails=[self._tail(host=self._replay(device=64, host=64))]
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("from host", failures[0])

    @patch("sglang.test.tpu.utils.kv_offload_probe.requests.post")
    def test_token_ids_and_logprobs_round_trip(self, post: Mock) -> None:
        post.return_value.json.return_value = {
            "text": "ab",
            "meta_info": {
                "cached_tokens": 0,
                "completion_tokens": 2,
                "output_token_logprobs": [[-0.5, 7, None], [-1.25, 9, None]],
            },
        }

        observation = generate_with_cache_report(
            base_url="http://localhost:30000",
            prompt=[1, 2, 3],
            max_new_tokens=2,
            ignore_eos=True,
            return_logprob=True,
        )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["input_ids"], [1, 2, 3])
        self.assertTrue(payload["sampling_params"]["ignore_eos"])
        self.assertTrue(payload["return_logprob"])
        self.assertEqual(observation.output_ids, (7, 9))
        self.assertEqual(observation.output_logprobs, (-0.5, -1.25))


class TestDecodeTailSequence(unittest.TestCase):
    """``run_probe`` against a fake server that records every request."""

    def setUp(self) -> None:
        self.calls = []
        self.fillers_done = False

    def _post(self, url, json=None, timeout=None):
        response = Mock()
        if url.endswith("/flush_cache"):
            return response
        if url.endswith("/tokenize"):
            response.json.return_value = {"tokens": list(range(1000, 1600))}
            return response
        ids = json.get("input_ids")
        new = json["sampling_params"]["max_new_tokens"]
        self.calls.append((ids, new, self.fillers_done))
        if json.get("text") and new == 1:
            self.fillers_done = True
        cached = 0
        if ids is not None and new == 4:
            cached = len(ids) - 2
        tier = "host" if self.fillers_done else "device"
        response.json.return_value = {
            "text": "x",
            "meta_info": {
                "cached_tokens": cached,
                "cached_tokens_details": {tier: cached} if cached else None,
                "completion_tokens": new,
                "output_token_logprobs": [[-0.5, 7, None]] * new,
            },
        }
        return response

    def test_device_replays_precede_fillers_and_host_replays_follow(self) -> None:
        sizes = [(96, 32), (95, 33)]
        plan = plan_eviction(
            device_pool_tokens=8192, probe_tokens=1024, hicache_ratio=3.0
        )
        with patch(
            "sglang.test.tpu.utils.kv_offload_probe.requests.post",
            side_effect=self._post,
        ):
            probe = run_probe(
                base_url="http://localhost:30000",
                test_name="t",
                probe_tokens=1024,
                plan=plan,
                tail_sizes=sizes,
            )

        replays = [c for c in self.calls if c[0] is not None and c[1] == 4]
        self.assertEqual([after for _, _, after in replays], [False] * 2 + [True] * 2)
        for (ids, _, _), (prompt_tokens, max_new) in zip(replays, sizes * 2):
            committed = prompt_tokens + max_new
            self.assertEqual(len(ids), committed + 2)
            self.assertEqual(ids[committed:], ids[:2])
        self.assertEqual([t.committed_tokens for t in probe.tails], [128, 128])
        self.assertEqual(tail_failures(tails=probe.tails), [])

    def test_a_short_first_turn_is_an_error(self) -> None:
        def short(url, json=None, timeout=None):
            response = self._post(url, json, timeout)
            if (
                json
                and json.get("input_ids")
                and json["sampling_params"]["max_new_tokens"] != 4
            ):
                response.json.return_value["meta_info"]["output_token_logprobs"] = []
            return response

        plan = plan_eviction(
            device_pool_tokens=8192, probe_tokens=1024, hicache_ratio=3.0
        )
        with patch(
            "sglang.test.tpu.utils.kv_offload_probe.requests.post", side_effect=short
        ):
            with self.assertRaisesRegex(RuntimeError, "generated 0 tokens"):
                run_probe(
                    base_url="http://localhost:30000",
                    test_name="t",
                    probe_tokens=1024,
                    plan=plan,
                    tail_sizes=[(96, 32)],
                )


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
