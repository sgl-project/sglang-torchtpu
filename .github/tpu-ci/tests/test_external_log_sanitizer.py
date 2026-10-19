"""Exercise the log stream that is displayed and persisted by external TPU CI."""

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "external" / "sanitize_logs.py"
SPEC = importlib.util.spec_from_file_location("sanitize_logs", SCRIPT)
SANITIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SANITIZER)


class TestExternalLogSanitizer(unittest.TestCase):
    def test_cluster_context_and_overlapping_identifiers(self):
        values = SANITIZER.sensitive_values(
            {
                "EXTERNAL_CLUSTER": "example-cluster",
                "EXTERNAL_PROJECT": "example",
                "EXTERNAL_NODEPOOL_2X2X2_0": "example-pool",
                "EXTERNAL_NODEPOOL_2X2X2_1": "",
                "EXTERNAL_NAMESPACE": "default",
            }
        )
        result = SANITIZER.sanitize(
            "gke_example_us-central1_example-cluster example-pool default", values
        )
        self.assertNotIn("example", result)
        self.assertTrue(result.endswith("default"))

    def test_node_names_and_ip_addresses(self):
        line = (
            "gke-example-cluster-pool-abcd-1234.c.internal "
            "10.20.30.40:30000 [fd00:1234::1]:30000 "
            "::ffff:192.0.2.10 fe80::1%eth0"
        )
        self.assertEqual(
            SANITIZER.sanitize(line, []),
            "[REDACTED_NODE] [REDACTED_IP]:30000 [[REDACTED_IP]]:30000 "
            "[REDACTED_IP] [REDACTED_IP]",
        )

    def test_preserves_benchmark_results_and_timestamps(self):
        line = '2026-10-17T10:30:45Z {"throughput": 123.45, "accuracy": 0.98}\n'
        self.assertEqual(SANITIZER.sanitize(line, []), line)

    def test_persisted_stream_is_redacted_including_final_partial_line(self):
        env = dict(os.environ, EXTERNAL_PROJECT="example-private-project")
        result = subprocess.run(
            [sys.executable, "-u", str(SCRIPT)],
            input="example-private-project 10.20.30.40\nthroughput=123.45",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout, "[REDACTED] [REDACTED_IP]\nthroughput=123.45")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
