# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 megakernel's weight-only fixed-max proof
(sglang_tpu.multimodal_gen.h3_megakernel.bounds): it is the kernel's own
``fixed_max_layers``, on the bf16 weights the kernel packs. The comparisons
with the kernel need JAX; the rest runs on NumPy (and torch where installed).
"""

import math
import os
import sys
import unittest

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.multimodal_gen.h3_megakernel import bounds  # noqa: E402

try:
    import jax
    import jax.numpy as jnp

    from sglang_tpu.multimodal_gen.h3_megakernel import kernel as mk
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

    _HAS_JAX = True
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_SKIP = "Requires JAX"


class TestBounds(CustomTestCase):
    """``bounds.fixed_max_layers`` is the kernel's weight-only proof."""

    def test_the_bound_pairs_the_dims_rope_rotates_together(self):
        # A score sees q's and k's weights only where they share a RoPE pair
        # (d, d + 48) or an unrotated dim (96 and up).
        for q_dim, k_dim, fixed in ((0, 1, True), (0, 48, False), (100, 100, False)):
            with self.subTest(q_dim=q_dim, k_dim=k_dim):
                q, k = np.ones((1, 128), np.float32), np.ones((1, 128), np.float32)
                q[0, q_dim], k[0, k_dim] = 6.0, 6.0
                got = bounds.fixed_max_layers(q, k)
                self.assertEqual(got.layers.tolist(), [fixed])
                # 6 x sqrt(128) x log2(e) x 1.025 = 100.4 on a shared pair.
                want = (6.0 if fixed else 36.0) * math.sqrt(128) * math.log2(math.e)
                self.assertAlmostEqual(float(got.bounds[0]), want * 1.025, places=3)

    def test_rounds_the_weights_to_bf16_as_pack_weights_does(self):
        # A q weight of 6.5154 bounds the scores at 109.005, over 109; the
        # kernel's packed weights hold its bf16 rounding, 6.5 (108.75), and run
        # the layer at a fixed max. 6.53125 is a bf16 value (109.27).
        scale = math.sqrt(128) * math.log2(math.e) * bounds.BOUND_SLACK
        q = np.full((3, 128), 1.0 / 16, np.float32)
        k = q.copy()
        q[:, 3], k[:, 3] = (6.5154, 6.53125, 6.5), 1.0
        self.assertGreater(6.5154 * scale, bounds.FIXED_BOUND)
        got = bounds.fixed_max_layers(q, k)
        self.assertEqual(got.layers.tolist(), [True, False, True])
        self.assertEqual(got.bounds[0], got.bounds[2])
        if _HAS_JAX or _HAS_TORCH:
            np.testing.assert_array_equal(
                got.bounds, bounds.fixed_max_layers(_round_bf16(q), k).bounds
            )
        # [L, 1, D] arrays, and per-layer torch tensors in bf16 or float32.
        for name, (q_in, k_in) in (
            ("[L, 1, D]", (q[:, None], k[:, None])),
            *(
                (
                    f"torch {dtype}",
                    (
                        [torch.tensor(row).to(dtype) for row in q],
                        [torch.tensor(row).to(dtype) for row in k],
                    ),
                )
                for dtype in ((torch.bfloat16, torch.float32) if _HAS_TORCH else ())
            ),
        ):
            with self.subTest(name):
                again = bounds.fixed_max_layers(q_in, k_in)
                np.testing.assert_array_equal(again.layers, got.layers)
                np.testing.assert_array_equal(again.bounds, got.bounds)
        with self.assertRaisesRegex(ValueError, "not \\[layers, 128\\]"):
            bounds.fixed_max_layers(q[:, :96], k[:, :96])

    def test_summary_lists_the_online_layers(self):
        q = np.full((50, 128), 1.0 / 16, np.float32)
        k = q.copy()
        for layer in (36, 37, 43, 44, 45, 46, 47, 48, 49):
            q[layer, 7] = k[layer, 7] = 8.0
        got = bounds.fixed_max_layers(q, k)
        self.assertEqual(got.online, [36, 37, 43, 44, 45, 46, 47, 48, 49])
        self.assertEqual(
            got.summary(), "41/50 layers at a fixed max (online: 36, 37, 43-49)"
        )
        heads = got.heads(56)
        self.assertEqual(heads.shape, (50, 56))
        np.testing.assert_array_equal(heads, np.repeat(got.layers[:, None], 56, 1))

    @unittest.skipUnless(_HAS_JAX, _SKIP)
    def test_constants_are_the_kernels(self):
        self.assertEqual(bounds.FIXED_BOUND, mk._FIXED_BOUND)
        self.assertEqual(bounds.BOUND_SLACK, mk._BOUND_SLACK)
        config = ref.Config()
        self.assertEqual(
            (bounds.HEAD_DIM, bounds.ROPE_DIMS), (config.head_dim, config.rope_dims)
        )

    @unittest.skipUnless(_HAS_JAX, _SKIP)
    def test_matches_the_kernels_fixed_max_layers(self):
        config = ref.Config()
        # Random log-normal weights at several spreads, then one layer per
        # pair of bf16 mantissas whose product lands near the threshold.
        rng = np.random.default_rng(0)
        layers = [
            np.exp(rng.standard_normal((400, 2, 128)) * scale) * base
            for scale, base in ((0.3, 1.0), (0.6, 0.8), (0.9, 0.5), (1.2, 0.3))
        ]
        mantissas = 1 + np.arange(128) / 128
        m1, m2 = np.meshgrid(mantissas, mantissas, indexing="ij")
        near = np.full((m1.size * 2, 2, 128), 1.0 / 128)
        near[: m1.size, 0, 5], near[: m1.size, 1, 5] = 4 * m1.ravel(), m2.ravel()
        near[m1.size :, 0, 5], near[m1.size :, 1, 5] = 2 * m1.ravel(), m2.ravel()
        q, k = np.moveaxis(np.concatenate(layers + [near]).astype(np.float32), 1, 0)
        weights = {
            "q_norm": jnp.asarray(q[:, None], jnp.bfloat16),
            "k_norm": jnp.asarray(k[:, None], jnp.bfloat16),
        }
        want = np.asarray(jax.jit(lambda w: mk.fixed_max_layers(w, config))(weights))
        got = bounds.fixed_max_layers(q, k)
        np.testing.assert_array_equal(got.layers, want)
        self.assertGreater(want.sum(), 1000)
        self.assertGreater((~want).sum(), 1000)


def _round_bf16(x):
    """bf16 round-to-nearest-even of float32 ``x``, as float32 (via JAX if
    present, else torch)."""
    if _HAS_JAX:
        return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))
    return torch.tensor(x).to(torch.bfloat16).float().numpy()


if __name__ == "__main__":
    unittest.main()
