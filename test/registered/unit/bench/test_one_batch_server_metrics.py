"""Server throughput readback must account for every DP scheduler."""

import unittest

from sglang.benchmark.one_batch_server import get_last_gen_throughput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestServerThroughput(unittest.TestCase):
    def test_idle_rank_cannot_overwrite_active_rank(self):
        states = [{"last_gen_throughput": 110.45}, {"last_gen_throughput": 0.0}]
        self.assertEqual(get_last_gen_throughput(states), 110.45)
        self.assertEqual(get_last_gen_throughput(states[::-1]), 110.45)

    def test_sums_active_ranks(self):
        states = [{"last_gen_throughput": 100.0}, {"last_gen_throughput": 120.0}]
        self.assertEqual(get_last_gen_throughput(states), 220.0)

    def test_idle_server_reports_zero(self):
        states = [{"last_gen_throughput": 0.0}, {"last_gen_throughput": 0.0}]
        self.assertEqual(get_last_gen_throughput(states), 0.0)

    def test_missing_metrics_do_not_report_partial_throughput(self):
        for states in ([], [{}], [{"last_gen_throughput": 100.0}, {}]):
            with self.subTest(states=states):
                self.assertEqual(get_last_gen_throughput(states), -1)


if __name__ == "__main__":
    unittest.main()
