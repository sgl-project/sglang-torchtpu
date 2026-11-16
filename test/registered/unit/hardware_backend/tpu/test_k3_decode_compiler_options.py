"""Per-executable XLA options of the K3 decode steps (DecodeOps._compiler_options)."""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from sglang_tpu.k3_programs.kimi_k3 import config as kernel_config  # noqa: E402
from sglang_tpu.k3_programs.kimi_k3.decode_ops import DecodeOps  # noqa: E402

ROWS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 128)
TABLE = "xla_tpu_overlay_allocation_table_size"


def _options(**knobs):
    config = kernel_config.KernelConfig(**knobs)
    with mock.patch.object(kernel_config, "active", return_value=config):
        return {rows: DecodeOps._compiler_options(rows) for rows in ROWS}


class TestDecodeCompilerOptions(CustomTestCase):
    def test_overlay_table_is_off_by_default(self):
        for rows, options in _options().items():
            self.assertNotIn(TABLE, options, rows)

    def test_overlay_table_only_adds_its_option_at_16_rows(self):
        base = _options()
        knob = _options(decode_b16_overlay_table=2)
        for rows in ROWS:
            want = dict(base[rows], **({TABLE: "2"} if rows == 16 else {}))
            self.assertEqual(knob[rows], want, rows)


if __name__ == "__main__":
    unittest.main()
