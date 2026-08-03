"""Numerics check for Kimi-K2.5's TPU bicubic pos-embed interpolation.

`_bicubic_pos_embed_interpolate` reformulates the vision position-embedding
interpolation as host-computed taps/weights + a device gather (Qwen3-VL's
structure), so it carries no dynamic-shape `F.interpolate` and is compile-safe
on the TPU backend. This test pins that reformulation to the *exact* math it
replaces: `F.interpolate(mode="bicubic", align_corners=False)`.

CPU-only — no TPU / libtpu required. Run:
    python test/manual/tpu/test_kimi_pos_embed_interp.py
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.models.kimi_k25 import _bicubic_pos_embed_interpolate


def _reference(weight: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Exactly the pre-existing `get_rope_shape` body (bicubic, align_corners
    defaulting to False)."""
    return (
        F.interpolate(
            weight.permute((2, 0, 1)).unsqueeze(0),
            size=(h, w),
            mode="bicubic",
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


# (src_h, src_w) grid is 14x14 in the real checkpoint; targets cover
# upsample / downsample / non-square / equal-one-axis cases.
_SRC = (14, 14)
_TARGETS = [(27, 37), (37, 27), (8, 8), (32, 32), (14, 20), (20, 14), (5, 60)]


class TestKimiBicubicPosEmbedInterp(unittest.TestCase):
    def test_matches_f_interpolate_float32(self):
        torch.manual_seed(0)
        weight = torch.randn(*_SRC, 96, dtype=torch.float32)
        for h, w in _TARGETS:
            ref = _reference(weight, h, w)
            got = _bicubic_pos_embed_interpolate(weight, h, w)
            self.assertEqual(got.shape, ref.shape, msg=f"shape @ {(h, w)}")
            torch.testing.assert_close(
                got, ref, rtol=1e-4, atol=1e-4, msg=f"values @ {(h, w)}"
            )

    def test_matches_f_interpolate_bfloat16(self):
        torch.manual_seed(0)
        weight = torch.randn(*_SRC, 96, dtype=torch.bfloat16)
        for h, w in _TARGETS:
            ref = _reference(weight, h, w)
            got = _bicubic_pos_embed_interpolate(weight, h, w)
            self.assertEqual(got.dtype, torch.bfloat16)
            # bf16 has ~2-3 significant digits; compare on a generous tolerance.
            torch.testing.assert_close(
                got.float(), ref.float(), rtol=3e-2, atol=3e-2, msg=f"bf16 @ {(h, w)}"
            )


if __name__ == "__main__":
    unittest.main()
