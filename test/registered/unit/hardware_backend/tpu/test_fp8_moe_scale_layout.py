"""Unit tests for the TPU FP8 MoE scale buffer layout.

The bug these guard: `scale[:, None, None, :]` on a contiguous `[E, N]` tensor is
*already* contiguous, so the trailing `.contiguous()` returns the view unchanged
and the parameter declares `[E, 1, 1, N]` while its device buffer is still the
2-D one it aliases. torch_tpu then reconciles the two on every use -- one
`tt_jit_as_strided` program per decode step copying all 120 scale buffers, for
weights that are constant after load.

`_base is None` is the load-bearing assertion. Shape and value assertions passed
before the fix too, which is exactly why this went unnoticed.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.fp8_moe import (
    _own_storage,
    _to_tpu_fp8_moe_block_scale_layout,
    _to_tpu_fp8_moe_scale_layout,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS = 4
OUT_SIZE = 8


class TestOwnStorage(CustomTestCase):
    def test_a_contiguous_view_is_copied_into_its_own_buffer(self):
        # The trap: unsqueezing trailing singleton dims leaves the tensor
        # contiguous, so `.contiguous()` is a no-op and the view survives.
        base = torch.arange(NUM_EXPERTS * OUT_SIZE, dtype=torch.float32).reshape(
            NUM_EXPERTS, OUT_SIZE
        )
        view = base[:, None, None, :].contiguous()
        self.assertIsNotNone(view._base)  # precondition: still a view

        owned = _own_storage(view)

        self.assertIsNone(owned._base)
        self.assertEqual(owned.shape, view.shape)
        torch.testing.assert_close(owned, view)

    def test_the_copy_does_not_alias_the_source(self):
        base = torch.zeros(NUM_EXPERTS, OUT_SIZE)
        owned = _own_storage(base[:, None, None, :].contiguous())

        base.fill_(7.0)

        self.assertTrue(torch.all(owned == 0.0))

    def test_a_tensor_that_already_owns_its_buffer_is_returned_unchanged(self):
        # No wasted 46 MB copy when the input is already a real buffer.
        owned = torch.zeros(NUM_EXPERTS, 1, 1, OUT_SIZE)

        self.assertIs(_own_storage(owned), owned)


class TestScaleLayoutOwnsItsBuffer(CustomTestCase):
    """Every accepted input shape must come out owning its declared buffer."""

    def _channelwise_inputs(self):
        return {
            "per_expert_1d": torch.rand(NUM_EXPERTS),
            "per_expert_2d": torch.rand(NUM_EXPERTS, 1),
            "two_halves": torch.rand(NUM_EXPERTS, 2),
            "full_width": torch.rand(NUM_EXPERTS, OUT_SIZE),
            "already_4d": torch.rand(NUM_EXPERTS, 1, 1, OUT_SIZE),
        }

    def test_channelwise_layout_owns_its_buffer(self):
        for name, scale in self._channelwise_inputs().items():
            with self.subTest(case=name):
                out = _to_tpu_fp8_moe_scale_layout(scale, out_size=OUT_SIZE)

                self.assertEqual(tuple(out.shape), (NUM_EXPERTS, 1, 1, OUT_SIZE))
                self.assertIsNone(
                    out._base, f"{name}: scale buffer is still a view of its source"
                )
                self.assertTrue(out.is_contiguous())

    def test_channelwise_layout_broadcasts_the_scale_value(self):
        # Guards the layout itself, so a future rewrite cannot satisfy the
        # ownership assertion by returning a differently-filled buffer.
        scale = torch.arange(NUM_EXPERTS, dtype=torch.float32)[:, None]

        out = _to_tpu_fp8_moe_scale_layout(scale, out_size=OUT_SIZE)

        for expert in range(NUM_EXPERTS):
            self.assertTrue(torch.all(out[expert] == float(expert)))

    def test_block_layout_owns_its_buffer(self):
        # Blockwise scales arrive as [E, n_blocks, k_blocks]; the layout
        # transposes, repeat_interleaves out to out_size, then unsqueezes -- the
        # same trailing-singleton pattern that leaves a view behind.
        num_out_blocks = 2
        num_k_blocks = 3
        for name, scale in {
            "3d_blocks": torch.rand(NUM_EXPERTS, num_out_blocks, num_k_blocks),
            "4d_full_width": torch.rand(NUM_EXPERTS, num_k_blocks, 1, OUT_SIZE),
        }.items():
            with self.subTest(case=name):
                out = _to_tpu_fp8_moe_block_scale_layout(scale, out_size=OUT_SIZE)

                self.assertIsNone(
                    out._base, f"{name}: block scale buffer is still a view"
                )
                self.assertTrue(out.is_contiguous())
                self.assertEqual(out.shape[-1], OUT_SIZE)

    def test_rejects_an_unsupported_scale_shape(self):
        with self.assertRaisesRegex(ValueError, "Unsupported TPU FP8 MoE scale"):
            _to_tpu_fp8_moe_scale_layout(torch.rand(NUM_EXPERTS, 3), out_size=OUT_SIZE)


if __name__ == "__main__":
    unittest.main()
