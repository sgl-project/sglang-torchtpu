"""Fused Pallas router top-k vs topk.biased_topk_impl's torch math."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

EPS = 1e-20


def _reference(logits, bias, k, renormalize, scale):
    """biased_topk_impl: sigmoid in the logits dtype, f32 biased choice, gather,
    f32 renorm."""
    s = 1.0 / (1.0 + np.exp(-np.asarray(logits.astype(jnp.float32), np.float32)))
    s = np.asarray(jnp.asarray(s).astype(logits.dtype).astype(jnp.float32))
    choice = s + bias[None]
    ids = np.argsort(-choice, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(s, ids, -1)
    if renormalize:
        w = w / (w.sum(-1, keepdims=True) + EPS) * scale
    return w, ids


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestRouterTopk(CustomTestCase):
    def test_matches_reference(self):
        from sglang_tpu.kernels.router_topk import router_topk_jax

        rng = np.random.default_rng(0)
        for tokens, experts, k, renorm, scale, dtype in (
            (1, 896, 16, True, 2.5, jnp.bfloat16),
            (4, 896, 16, True, 1.0, jnp.bfloat16),
            (16, 256, 8, False, 1.0, jnp.bfloat16),
            # f32 logits: the sigmoid stays in f32, as it would with an f32 router.
            (1, 896, 16, True, 2.5, jnp.float32),
            (2, 896, 16, True, 1.0, jnp.float32),
            (64, 896, 16, True, 2.5, jnp.float32),
        ):
            logits = jnp.asarray(rng.standard_normal((tokens, experts)), dtype)
            bias = rng.standard_normal(experts).astype(np.float32) * 0.1
            w, ids = router_topk_jax(logits, jnp.asarray(bias), k, renorm, EPS, scale)
            want_w, want_ids = _reference(logits, bias, k, renorm, scale)
            # Same expert set and the same weight per expert (order is free).
            got = {
                (t, int(i)): float(x)
                for t in range(tokens)
                for i, x in zip(np.asarray(ids)[t], np.asarray(w)[t])
            }
            want = {
                (t, int(i)): float(x)
                for t in range(tokens)
                for i, x in zip(want_ids[t], want_w[t])
            }
            self.assertEqual(set(got), set(want), f"{tokens=} {experts=} {k=}")
            for key, x in want.items():
                self.assertAlmostEqual(got[key], x, delta=1e-6 * max(1.0, abs(x)))

    def test_f32_ties_pick_the_lowest_ids(self):
        """f32 logits quantized to a few values, so many scores tie exactly: the
        kernel picks the lowest expert ids among ties, as a stable sort does."""
        from sglang_tpu.kernels.router_topk import router_topk_jax

        rng = np.random.default_rng(1)
        for tokens in (1, 8):
            logits = np.round(rng.standard_normal((tokens, 896)) * 2) / 2
            logits = jnp.asarray(logits, jnp.float32)
            bias = np.zeros(896, np.float32)  # keep the ties in the biased choice
            w, ids = router_topk_jax(logits, jnp.asarray(bias), 16, True, EPS, 2.5)
            want_w, want_ids = _reference(logits, bias, 16, True, 2.5)
            np.testing.assert_array_equal(
                np.sort(np.asarray(ids), -1), np.sort(want_ids, -1)
            )
            np.testing.assert_allclose(
                np.sort(np.asarray(w), -1), np.sort(want_w, -1), rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
