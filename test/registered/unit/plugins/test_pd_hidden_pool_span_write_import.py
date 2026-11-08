"""PdHiddenPool's TPU row write imports sglang_tpu.kv_span_write lazily; keep the module present."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPdHiddenPoolSpanWriteImport(unittest.TestCase):
    def test_kv_span_write_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("sglang_tpu.kv_span_write"))


if __name__ == "__main__":
    unittest.main()
