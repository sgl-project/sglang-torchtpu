import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import update_tpu_est_time as u  # noqa: E402

_SUITE_LOG = """\
2026-10-24T19:00:00.1Z [pod/a-slice-0-0/test] ========== TIMINGS BEGIN ==========
2026-10-24T19:00:00.2Z [pod/a-slice-0-0/test] {"file": "test/x/a.py", "passed": true, "elapsed": 5422}
2026-10-24T19:00:00.3Z [pod/a-slice-0-0/test] {"file": "test/x/b.py", "passed": false, "elapsed": 99}
2026-10-24T19:00:00.4Z [pod/a-slice-0-0/test] ========== TIMINGS END ==========
"""

_LIVE_LOG = """\
2026-10-24T20:09:11.0Z [pod/k-slice-0-0/test] booting
2026-10-24T20:37:16.0Z == live-server test: test/x/k0.py ==
2026-10-24T21:37:16.0Z == live-server test: test/x/k1.py ==
2026-10-24T23:00:33.0Z [pod/k-slice-0-0/test] Ran 2 tests in 8589.5s
2026-10-24T23:01:00.0Z Post job cleanup.
"""

_SRC = """\
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=14400,
    suite="nightly-tpu-multihost-accuracy",
    labels=["multihost", "kimi-k2.6"],
)
MULTIHOST_LAUNCH = {"timeout_per_file": 14400, "weights": "w"}
"""


class TestUpdateTpuEstTime(unittest.TestCase):
    def test_timings_keep_only_passed_files(self):
        self.assertEqual(u.timings_samples(_SUITE_LOG), [("test/x/a.py", 5422)])

    def test_live_server_segments_exclude_bringup(self):
        self.assertEqual(
            u.live_server_samples(_LIVE_LOG),
            [("test/x/k0.py", 3600), ("test/x/k1.py", 4997)],
        )

    def test_propose_needs_samples_and_a_real_move(self):
        self.assertIsNone(u.propose(10800, [5000, 5400]))  # too few samples
        self.assertIsNone(u.propose(6780, [5400, 5300, 5420]))  # within 10%
        self.assertEqual(u.propose(10800, [4800, 5400, 6120]), 7650)
        self.assertEqual(
            u.propose(25, [60, 70, 80]), 100
        )  # 10s rounding, not whole minutes

    def test_rewrite_moves_mirrored_timeout_per_file(self):
        out = u.rewrite(_SRC, 10800)
        self.assertIn("est_time=10800", out)
        self.assertIn('"timeout_per_file": 10800', out)

    def test_rewrite_leaves_independent_timeout_alone(self):
        src = _SRC.replace('"timeout_per_file": 14400', '"timeout_per_file": 8100')
        out = u.rewrite(src, 10800)
        self.assertIn("est_time=10800", out)
        self.assertIn('"timeout_per_file": 8100', out)

    def test_two_registrations_are_skipped(self):
        src = _SRC + 'register_tpu_ci(est_time=60, suite="other")\n'
        self.assertIsNone(u.current_est(src))
        self.assertEqual(u.rewrite(src, 10800), src)


if __name__ == "__main__":
    unittest.main()
