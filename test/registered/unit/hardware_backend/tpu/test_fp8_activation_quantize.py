"""Unit tests for the TPU FP8 per-row activation quantizer.

`_quantize_activation_per_row` divides by a per-row scale and casts to fp8
*without* clamping the quotient, which is only safe because the scale is derived
from that row's own amax. These tests pin that invariant. They matter because the
fp8 cast does not saturate: `torch.float8_e4m3fn` has no infinity, so a quotient
past the rounding window becomes NaN rather than a clipped value, and a future
change to how the scale is computed (per-tensor, calibrated, chunked amax) would
turn activations into NaN silently.

Device equivalence of the unclamped form is covered on TPU by
test/registered/tpu/unit/test_fp8_linear.py; these run on CPU and cover the
arithmetic.
"""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.fp8_linear import _quantize_activation_per_row
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TARGET_DTYPE = torch.float8_e4m3fn
DTYPE_MAX = torch.finfo(TARGET_DTYPE).max  # 448.0
# The scale floor in the quantizer. It binds only for rows whose amax is below
# DTYPE_MAX * SCALE_FLOOR, and at exactly that crossover the quotient reaches
# DTYPE_MAX -- see test_scale_floor_cannot_push_quotient_past_dtype_max.
SCALE_FLOOR = 1e-12
# torch.clamp narrows the floor to fp32 before applying it, and fp32 cannot hold
# 1e-12 exactly, so comparisons against a clamped scale use the narrowed value.
SCALE_FLOOR_FP32 = torch.tensor(SCALE_FLOOR, dtype=torch.float32).item()


def _quantize_with_clamp(
    x: torch.Tensor, target_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-change implementation, kept as the reference to diff against."""
    dtype_max = torch.finfo(target_dtype).max
    x_scale = torch.clamp(
        x.abs().amax(dim=1, keepdim=True).float() / dtype_max, min=SCALE_FLOOR
    )
    x_q = torch.clamp(x.float() / x_scale, min=-dtype_max, max=dtype_max).to(
        target_dtype
    )
    return x_q, x_scale


def _cases() -> dict[str, torch.Tensor]:
    """Rows chosen for where they land relative to the scale floor and fp8 max."""
    generator = torch.Generator().manual_seed(0)
    floor_crossover = DTYPE_MAX * SCALE_FLOOR  # quotient lands exactly on DTYPE_MAX
    return {
        # amax == 0, so the floor binds and the numerator is zero anyway.
        "all_zeros": torch.zeros(2, 8),
        # One outlier sets the scale; every other lane quantizes to ~0.
        "single_outlier": torch.tensor([[1e30, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0]]),
        # Whole row already at fp8 max: quotient is exactly DTYPE_MAX everywhere.
        "at_dtype_max": torch.full((2, 8), DTYPE_MAX),
        # fp32 extremes in one row, so the scale spans ~35 orders of magnitude.
        "fp32_extremes": torch.tensor(
            [[3.4e38, -3.4e38, 1e-30, 0.0, 1.0, -1.0, 2.0, -2.0]]
        ),
        # Below the floor crossover: the floor binds, quotient stays under max.
        "under_scale_floor": torch.full((1, 8), floor_crossover / 2),
        # Exactly at the crossover: quotient == DTYPE_MAX, the tightest real case.
        "at_scale_floor": torch.full((1, 8), floor_crossover),
        # Subnormal input, far below the floor.
        "subnormals": torch.full((1, 8), 1e-40),
        # Mixed signs and magnitudes, the ordinary case.
        "randn_scaled": torch.randn(16, 64, generator=generator) * 37,
        # Rows differing wildly in scale, so a per-tensor scale would fail here.
        "per_row_spread": torch.tensor([[1e-8, -2e-8], [1e8, -2e8], [0.0, 0.0]]),
    }


class TestQuantizeActivationPerRow(CustomTestCase):
    def test_matches_clamped_reference_bitwise(self):
        # Dropping the clamp must be a no-op, not an approximation: the fp8
        # bit patterns have to be identical, not merely close.
        for name, x in _cases().items():
            with self.subTest(case=name):
                q, scale = _quantize_activation_per_row(x, TARGET_DTYPE)
                ref_q, ref_scale = _quantize_with_clamp(x, TARGET_DTYPE)
                self.assertTrue(torch.equal(scale, ref_scale))
                self.assertTrue(
                    torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)),
                    f"{name}: unclamped quantization diverged from the clamped form",
                )

    def test_output_is_finite(self):
        # The guard that catches a broken scale: fp8_e4m3fn cannot represent
        # infinity, so an over-range quotient lands on NaN instead of saturating.
        for name, x in _cases().items():
            with self.subTest(case=name):
                q, _ = _quantize_activation_per_row(x, TARGET_DTYPE)
                self.assertTrue(
                    q.float().isfinite().all(),
                    f"{name}: quantized activations contain NaN or inf",
                )

    def test_quotient_never_exceeds_dtype_max(self):
        # The invariant the clamp removal rests on. Scale rounding can overshoot
        # by ~1 ulp of fp32, which is ~7 orders below the fp8 spacing at
        # DTYPE_MAX, so allow a hair over 1.0 and nothing more.
        for name, x in _cases().items():
            with self.subTest(case=name):
                _, scale = _quantize_activation_per_row(x, TARGET_DTYPE)
                ratio = (x.float() / scale).abs().max().item() / DTYPE_MAX
                self.assertLessEqual(ratio, 1.0 + 1e-6, f"{name}: quotient overshot")

    def test_scale_floor_cannot_push_quotient_past_dtype_max(self):
        # The floor is tuned so the crossover lands exactly on DTYPE_MAX: a row
        # whose amax is DTYPE_MAX * SCALE_FLOOR divides to DTYPE_MAX, and any
        # smaller amax divides to less. Lowering the floor would break this.
        for factor in (1.0, 0.999, 0.5, 1e-3, 1e-10):
            amax = DTYPE_MAX * SCALE_FLOOR * factor
            with self.subTest(factor=factor):
                x = torch.tensor([[amax, -amax / 3, 0.0, amax / 7]])
                _, scale = _quantize_activation_per_row(x, TARGET_DTYPE)
                self.assertEqual(scale.item(), SCALE_FLOOR_FP32)
                quotient_max = (x / scale).abs().max().item()
                self.assertLessEqual(quotient_max, DTYPE_MAX)

    def test_dequantization_recovers_input_within_fp8_resolution(self):
        # Pins the scale's magnitude, not just its effect on the quotient: a
        # scale off by a factor would still pass the bound checks above.
        x = torch.randn(8, 32, generator=torch.Generator().manual_seed(1)) * 12.5
        q, scale = _quantize_activation_per_row(x, TARGET_DTYPE)
        dequantized = q.float() * scale
        # fp8_e4m3fn carries 3 mantissa bits, so ~6% relative error per element.
        torch.testing.assert_close(dequantized, x, rtol=0.07, atol=0.0)

    def test_scale_is_per_row(self):
        # Row independence is what makes the unclamped divide safe; a shared
        # scale would let a large row overflow a small one.
        x = torch.tensor([[1.0, -0.5], [1e6, -1e6]])
        _, scale = _quantize_activation_per_row(x, TARGET_DTYPE)
        self.assertEqual(scale.shape, (2, 1))
        self.assertAlmostEqual(scale[0].item(), 1.0 / DTYPE_MAX, places=9)
        self.assertAlmostEqual(scale[1].item() / (1e6 / DTYPE_MAX), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
