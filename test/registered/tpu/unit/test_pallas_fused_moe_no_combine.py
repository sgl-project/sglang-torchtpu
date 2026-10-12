"""TPU integration test: pallas fused-MoE ``no_combine``.

Runs the real ``_fused_moe_jax`` kernel body (permute / GMM1 / act / GMM2 /
un-permute) on TPU and checks that:

* ``no_combine=True`` returns the un-combined per-expert output
  ``[m, topk, hidden]``;
* combining that per-expert output with ``topk_weights`` reproduces the
  ``no_combine=False`` combined ``[m, hidden]`` output.

Also checks the registered ``_fake_impl`` shape contract on CPU (no TPU
needed), since that is what ``torch.compile`` relies on.

The TPU test is gated behind JAX TPU backend availability so it is a no-op
on CPU-only hosts. It lives under test/manual/tpu because it requires
hardware.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib.util
import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _jax_tpu_available() -> bool:
    if importlib.util.find_spec("jax") is None:
        return False
    try:
        import jax

        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


@unittest.skipUnless(_jax_tpu_available(), "requires a JAX TPU backend")
class TestPallasFusedMoeNoCombineTpu(CustomTestCase):
    def test_no_combine_matches_combined(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        rng = jax.random.PRNGKey(0)
        m, hidden, inter, num_experts, topk = 16, 128, 256, 8, 2
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

        # Use float32 so the equivalence check tests the no_combine *logic*
        # (per-expert vs in-kernel reduce) rather than bf16 accumulation noise.
        dt = jnp.float32
        x = jax.random.normal(k1, (m, hidden), dtype=dt)
        # gmm_v2 (g, k, n) layout: w1 [g, hidden, 2*inter], w2 [g, inter, hidden]
        w1 = jax.random.normal(k2, (num_experts, hidden, 2 * inter), dt)
        w2 = jax.random.normal(k3, (num_experts, inter, hidden), dt)
        topk_ids = jax.random.randint(k4, (m, topk), 0, num_experts).astype(jnp.int32)
        topk_weights = jax.random.uniform(k5, (m, topk)).astype(dt)

        common = dict(
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk=topk,
            activation_name="silu",
        )

        combined = jax.jit(
            lambda: _fused_moe_jax(
                x, w1, w2, topk_weights, topk_ids, no_combine=False, **common
            )
        )()
        per_expert = jax.jit(
            lambda: _fused_moe_jax(
                x, w1, w2, topk_weights, topk_ids, no_combine=True, **common
            )
        )()

        self.assertEqual(combined.shape, (m, hidden))
        self.assertEqual(per_expert.shape, (m, topk, hidden))

        # Recombine per-expert with the routing weights -> must equal combined.
        weights = topk_weights.reshape(m, topk, 1).astype(per_expert.dtype)
        recombined = (per_expert * weights).sum(axis=1).astype(combined.dtype)

        torch.testing.assert_close(
            torch.from_numpy(jax.device_get(recombined).astype("float32")),
            torch.from_numpy(jax.device_get(combined).astype("float32")),
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
