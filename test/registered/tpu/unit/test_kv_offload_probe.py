"""Hardware-free tests for the KV-offload probe's speculative gate."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.kv_offload_probe import (
    CacheObservation,
    KvOffloadProbe,
    mean_accept_length,
    spec_failures,
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
