"""Production ``_fused_moe_jax`` with the destination-major SparseCore combine
vs a pure-JAX reference MoE.

Shapes whose dense gather-reduce would fall back to XLA (top-k 10; top-k 4
with rows not in whole 16384-row waves) must reach the destination-major kernel
and match the dense reference; shapes the dense SparseCore kernel handles
(top-k 8 and aligned top-k 4) and small batches must keep it.

Run as:
    SGLANG_PLATFORM=tpu python \\
      test/registered/tpu/unit/test_pallas_fused_moe_local_combine.py
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU backend visible to jax.devices()"


def _jax_reference_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """Pure-JAX SwiGLU MoE reference: every expert applied densely to every
    token, weighted by the router weight of the slot that selected it."""
    g = w1.shape[0]
    n = w1.shape[-1] // 2
    hs = hidden_states.astype(jnp.float32)
    w1f = w1.astype(jnp.float32)
    w2f = w2.astype(jnp.float32)
    tw = topk_weights.astype(jnp.float32)
    onehot = jax.nn.one_hot(topk_ids, g, dtype=jnp.float32)  # [m, topk, g]
    expert_weight = jnp.einsum("mj,mjg->mg", tw, onehot)  # [m, g]
    out = jnp.zeros(hs.shape, jnp.float32)
    for e in range(g):
        gmm1 = hs @ w1f[e]
        act = jax.nn.silu(gmm1[:, :n]) * gmm1[:, n:]
        out = out + expert_weight[:, e][:, None] * (act @ w2f[e])
    return out


def _make_inputs(m, k, n, g, topk, seed):
    key = jax.random.PRNGKey(seed)
    kh, kw1, kw2, ktw, kids = jax.random.split(key, 5)
    hidden = (jax.random.normal(kh, (m, k), dtype=jnp.float32) * 0.5).astype(
        jnp.bfloat16
    )
    w1 = (jax.random.normal(kw1, (g, k, 2 * n), dtype=jnp.float32) * 0.05).astype(
        jnp.bfloat16
    )
    w2 = (jax.random.normal(kw2, (g, n, k), dtype=jnp.float32) * 0.05).astype(
        jnp.bfloat16
    )
    # Distinct experts per token, as a router produces.
    keys = jax.random.split(kids, m)
    topk_ids = jax.vmap(lambda key: jax.random.permutation(key, g)[:topk])(keys)
    topk_ids = topk_ids.astype(jnp.int32)
    topk_weights = jax.nn.softmax(
        jax.random.normal(ktw, (m, topk), dtype=jnp.float32), axis=-1
    ).astype(jnp.bfloat16)
    return hidden, w1, w2, topk_weights, topk_ids


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoELocalCombine(unittest.TestCase):
    HIDDEN = 4096
    INTERMEDIATE = 256
    EXPERTS = 16

    def _run(self, *, m, topk, seed, expect_destination_major):
        from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax
        from sglang.srt.layers.moe.pallas.sparse_core import (
            dense_gather_reduce,
            ragged_gather_reduce_v3,
        )

        hidden, w1, w2, w_tk, ids = _make_inputs(
            m, self.HIDDEN, self.INTERMEDIATE, self.EXPERTS, topk, seed
        )
        calls = []

        def spy(name, fn):
            def wrapped(*args, **kwargs):
                calls.append(name)
                return fn(*args, **kwargs)

            return wrapped

        with mock.patch.object(
            ragged_gather_reduce_v3,
            "ragged_gather_reduce",
            spy("destination-major", ragged_gather_reduce_v3.ragged_gather_reduce),
        ), mock.patch.object(
            dense_gather_reduce,
            "dense_gather_reduce",
            spy("dense", dense_gather_reduce.dense_gather_reduce),
        ):
            out = jax.block_until_ready(
                _fused_moe_jax(
                    hidden,
                    w1,
                    w2,
                    w_tk,
                    ids,
                    None,
                    None,
                    None,
                    None,
                    topk,
                    "silu",
                    use_sc_gather=1,
                )
            )
        expected = "destination-major" if expect_destination_major else "dense"
        self.assertEqual(calls, [expected], (m, topk))

        ref = _jax_reference_moe(hidden, w1, w2, w_tk, ids)
        out32 = jnp.asarray(out).astype(jnp.float32)
        self.assertEqual(out32.shape, ref.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out32))), "non-finite output")
        max_diff = float(jnp.max(jnp.abs(out32 - ref)))
        self.assertLess(max_diff, 5e-2, f"max diff {max_diff} (m={m}, topk={topk})")

    def test_topk10_2048_tokens_uses_destination_major(self):
        self._run(m=2048, topk=10, seed=0, expect_destination_major=True)

    def test_topk10_512_tokens_uses_destination_major(self):
        self._run(m=512, topk=10, seed=1, expect_destination_major=True)

    def test_topk10_odd_token_count_pads_before_kernel(self):
        self._run(m=1025, topk=10, seed=2, expect_destination_major=True)

    def test_topk10_small_batch_keeps_dense(self):
        self._run(m=256, topk=10, seed=3, expect_destination_major=False)

    def test_topk8_keeps_dense_kernel(self):
        self._run(m=2048, topk=8, seed=4, expect_destination_major=False)

    def test_topk4_aligned_keeps_dense_kernel(self):
        self._run(m=4096, topk=4, seed=5, expect_destination_major=False)

    def test_topk4_unaligned_uses_destination_major(self):
        self._run(m=2048, topk=4, seed=6, expect_destination_major=True)


if __name__ == "__main__":
    unittest.main()
