"""Hardware-free tests for the device-cache and KV-offload probes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.kv_offload_probe import (
    CacheObservation,
    DeviceCacheProbe,
    KvOffloadProbe,
    mean_accept_length,
    spec_failures,
    validate_device_cache_probe,
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


def _device_probe(*, cold=None, warm=None):
    return DeviceCacheProbe(
        cold=cold or _read(),
        warm=warm
        or CacheObservation(
            text="10 11 12 13",
            device_tokens=2048,
            host_tokens=0,
            completion_tokens=8,
            spec_verify_ct=4,
        ),
    )


class TestDeviceCacheProbe(unittest.TestCase):
    def test_cold_and_device_hit_pass(self) -> None:
        self.assertEqual(
            validate_device_cache_probe(probe=_device_probe(), page_size=64), []
        )

    def test_cold_read_must_miss_both_cache_tiers(self) -> None:
        failures = validate_device_cache_probe(
            probe=_device_probe(
                cold=CacheObservation(
                    text="10 11 12 13",
                    device_tokens=64,
                    host_tokens=0,
                    completion_tokens=8,
                    spec_verify_ct=4,
                )
            ),
            page_size=64,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("cold probe hit the cache", failures[0])

    def test_immediate_read_must_hit_device(self) -> None:
        failures = validate_device_cache_probe(
            probe=_device_probe(warm=_read()), page_size=64
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("no device cache hit", failures[0])

    def test_immediate_read_must_not_hit_host(self) -> None:
        failures = validate_device_cache_probe(
            probe=_device_probe(
                warm=CacheObservation(
                    text="10 11 12 13",
                    device_tokens=2048,
                    host_tokens=64,
                    completion_tokens=8,
                    spec_verify_ct=4,
                )
            ),
            page_size=64,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("reloaded 64 tokens from host", failures[0])

    def test_each_read_must_continue_the_forced_count(self) -> None:
        failures = validate_device_cache_probe(
            probe=_device_probe(
                warm=CacheObservation(
                    text="garbled output",
                    device_tokens=2048,
                    host_tokens=0,
                    completion_tokens=8,
                    spec_verify_ct=4,
                )
            ),
            page_size=64,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("device-hit", failures[0])


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
        failures = spec_failures(
            probe=_probe(reload=_read(spec_verify_ct=0)), min_accept_length=1.0
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("host-reload", failures[0])

    def test_accept_length_must_be_strictly_above_floor(self) -> None:
        failures = spec_failures(
            probe=_probe(
                cold=_read(completion_tokens=4, spec_verify_ct=4),
                warm=_read(completion_tokens=4, spec_verify_ct=4),
                reload=_read(completion_tokens=4, spec_verify_ct=4),
            ),
            min_accept_length=1.0,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("not above 1.00", failures[0])

    def test_productive_drafts_pass(self) -> None:
        self.assertEqual(spec_failures(probe=_probe(), min_accept_length=1.0), [])


if __name__ == "__main__":
    unittest.main()
