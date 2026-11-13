"""K3 prefill step planning: which steps fold into the next, and how a folded
request's tokens split into windows."""

import sys
import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# The prefill module imports the bridge, which imports torch_tpu's Pallas op
# support at load; the planner under test is plain Python.
if "torch_tpu" not in sys.modules:
    for name in (
        "torch_tpu",
        "torch_tpu._internal",
        "torch_tpu._internal.pallas",
        "torch_tpu._internal.pallas.pallas",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch_tpu._internal.pallas.pallas"].__getattr__ = lambda attr: (
        mock.MagicMock(name=attr)
    )

from sglang_tpu.k3_programs.kimi_k3.prefill import plan_pieces, skips_step  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ALIGN = 512
BLOCK = 32 * ALIGN  # a logical block (fp8)
LARGEST = 8192


class TestSkipsStep(CustomTestCase):
    def test_a_step_inside_a_block_that_the_rest_fits_folds(self):
        self.assertTrue(
            skips_step(BLOCK, BLOCK + 3000, BLOCK + 3100, BLOCK, LARGEST, BLOCK)
        )

    def test_the_final_step_and_a_completed_block_never_fold(self):
        self.assertFalse(
            skips_step(BLOCK, BLOCK + 3100, BLOCK + 3100, BLOCK, LARGEST, BLOCK)
        )
        self.assertFalse(skips_step(0, BLOCK, BLOCK + 300, 0, LARGEST, BLOCK, ALIGN))

    def test_a_tail_cut_folds_even_when_the_rest_exceeds_a_window(self):
        start, prompt = BLOCK, BLOCK + 10_100
        cut = prompt // ALIGN * ALIGN  # the tail's end, inside the last 512 tokens
        # Without the alignment the rest (10100 rows) does not fit one window.
        self.assertFalse(skips_step(start, cut, prompt, start, LARGEST, BLOCK))
        self.assertTrue(skips_step(start, cut, prompt, start, LARGEST, BLOCK, ALIGN))
        # An earlier stop is not a tail cut: still no fold past one window.
        self.assertFalse(
            skips_step(start, cut - ALIGN, prompt, start, LARGEST, BLOCK, ALIGN)
        )

    def test_the_folded_rest_keeps_its_one_checkpoint(self):
        start, prompt = BLOCK, BLOCK + 10_100
        cut = prompt // ALIGN * ALIGN
        pieces = plan_pieces(0, start, prompt, [cut], LARGEST, ALIGN)
        self.assertEqual(
            [(p.start, p.end) for p in pieces],
            [(start, start + LARGEST), (start + LARGEST, prompt)],
        )
        self.assertEqual([p.stop for p in pieces], [None, cut])


if __name__ == "__main__":
    unittest.main()
