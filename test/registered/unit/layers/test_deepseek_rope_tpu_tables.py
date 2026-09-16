import itertools
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding import base as rope_base
from sglang.srt.layers.rotary_embedding import rope_variant
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make(rotary_dim, head_size, neox, dtype):
    # Build with the TPU flag on so the tables are registered, on CPU tensors.
    # The base constructor must also select TPU to avoid optional GPU kernels.
    with (
        patch.object(rope_base, "_is_tpu", True),
        patch.object(rope_variant, "_is_tpu", True),
    ):
        return DeepseekScalingRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=neox,
            scaling_factor=4.0,
            dtype=dtype,
            device="cpu",
        )


def _reference(rope, positions, q, k, offsets):
    """The native rope math, written out for both styles.

    gptj uses the real ``forward_native``. Upstream's neox branch there does
    ``cos.repeat(1, 1, 2)`` on the 2-D ``[T, rd/2]`` slice (it assumes 2-D
    positions), which DeepSeek/Kimi never hit since both are gptj style; spell
    out the same formula with a correct broadcast instead.
    """
    if not rope.is_neox_style:
        with patch.object(rope_variant, "_is_tpu", False):
            return rope.forward_native(positions, q, k, offsets)
    rd = rope.rotary_dim
    idx = positions + offsets if offsets is not None else positions
    cos, sin = rope.cos_sin_cache[idx].chunk(2, dim=-1)
    cos = cos.repeat(1, 2).unsqueeze(-2)
    sin = sin.repeat(1, 2).unsqueeze(-2)

    def apply(x):
        x_rot = x[..., :rd]
        x_rot = x_rot * cos + rope_variant.rotate_neox(x_rot) * sin
        return torch.cat((x_rot, x[..., rd:]), dim=-1).to(x.dtype)

    return apply(q), apply(k)


class TestDeepseekRopeTpuTables(CustomTestCase):
    CASES = tuple(
        itertools.product(
            (torch.float32, torch.bfloat16),  # dtype
            (False, True),  # neox style
            ((64, 64), (64, 192)),  # (rotary_dim, head_size)
            (False, True),  # offsets
        )
    )

    def test_tpu_path_is_bit_identical_to_native(self):
        torch.manual_seed(0)
        for dtype, neox, (rotary_dim, head_size), use_offsets in self.CASES:
            with self.subTest(
                dtype=dtype,
                neox=neox,
                rotary_dim=rotary_dim,
                head_size=head_size,
                offsets=use_offsets,
            ):
                rope = _make(rotary_dim, head_size, neox, dtype)
                t = 7
                positions = torch.randint(0, 900, (t,))
                offsets = torch.randint(0, 100, (t,)) if use_offsets else None
                q = torch.randn(t, 16, head_size, dtype=dtype)
                k = torch.randn(t, 1, head_size, dtype=dtype)
                q_ref, k_ref = _reference(rope, positions, q, k, offsets)
                q_tpu, k_tpu = rope._forward_tpu(positions, q, k, offsets)
                self.assertEqual(q_tpu.dtype, dtype)
                torch.testing.assert_close(q_tpu, q_ref, atol=0, rtol=0)
                torch.testing.assert_close(k_tpu, k_ref, atol=0, rtol=0)

    def test_tables_are_non_persistent_buffers(self):
        rope = _make(64, 64, True, torch.bfloat16)
        names = dict(rope.named_buffers())
        self.assertIn("_tpu_cos_sin", names)
        self.assertIn("_tpu_rot", names)
        self.assertNotIn("_tpu_cos_sin", rope.state_dict())
        self.assertEqual(tuple(names["_tpu_rot"].shape), (64, 64))
        self.assertEqual(names["_tpu_cos_sin"].shape[-1], 2 * 64)


if __name__ == "__main__":
    unittest.main()
