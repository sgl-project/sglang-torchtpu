# SPDX-License-Identifier: Apache-2.0
"""The MoE ring kernel must match the gmm_v2 fused-MoE path per (token, expert)
bit for bit, and up to the f32 combine order with real routing weights."""

import contextlib
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sglang_tpu.moe.pallas import moe_ring  # noqa: F401  (needs torch_tpu)

    _HAS_JAX = True
    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - host without the TPU stack
    _HAS_JAX = _HAS_TPU = False

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit", chips=1)

# 288 tensor-parallel experts of width 256 (one TP8 rank of a 2048-wide expert).
HIDDEN, INTER, EXPERTS, TOPK, LIMIT = 4096, 256, 288, 8, 10.0


@unittest.skipUnless(_HAS_JAX, "requires JAX and torch_tpu")
class TestMoeRingRoute(CustomTestCase):
    def test_pairs_are_expert_major_and_complete(self):
        from sglang_tpu.moe.pallas.moe_ring import _route

        rng = np.random.default_rng(0)
        tokens, experts, k = 24, 40, 4
        ids = np.stack([rng.choice(experts, k, replace=False) for _ in range(tokens)])
        ids[3, 1] = ids[3, 2]  # duplicate top-k
        ids[7] = -1  # padding row
        ids[9, 0] = experts  # out of range
        wts = rng.random((tokens, k)).astype(np.float32)
        counts, offsets, tok, wt = (
            np.asarray(v) for v in _route(jnp.asarray(ids), jnp.asarray(wts), experts)
        )
        for e in range(experts):
            want = sorted(
                (t, j) for t in range(tokens) for j in range(k) if ids[t, j] == e
            )
            self.assertEqual(counts[e], len(want))
            got = slice(offsets[e], offsets[e] + counts[e])
            np.testing.assert_array_equal(tok[got], [t for t, _ in want])
            np.testing.assert_array_equal(wt[got], [wts[t, j] for t, j in want])


@unittest.skipUnless(_HAS_JAX, "requires JAX and torch_tpu")
class TestMoeRingCostModel(CustomTestCase):
    """The trace-time choice between the ring and gmm_v2 (static shapes only)."""

    def test_measured_points(self):
        from sglang_tpu.moe.pallas.moe_ring import gmm_cost_cycles, ring_cost_cycles

        # (tokens, experts, hidden, width) -> measured k cycles (ring, gmm_v2),
        # routing spread over all experts.
        measured = {
            (2, 288, 4096, 256): (46.7, 89.3),
            (64, 288, 4096, 256): (448.2, 856.9),
            (256, 288, 4096, 256): (616.6, 1161.1),
            (64, 288, 4096, 512): (861.8, 1054.3),
            (64, 288, 4096, 128): (353.8, 747.1),
            (64, 128, 2048, 256): (181.5, 376.4),
            (64, 96, 8192, 128): (203.9, 498.8),
        }
        for (tokens, experts, hidden, n), (ring, gmm) in measured.items():
            with self.subTest(tokens=tokens, experts=experts, hidden=hidden, n=n):
                self.assertAlmostEqual(
                    ring_cost_cycles(tokens, TOPK, experts, hidden, n) / 1e3 / ring,
                    1,
                    delta=0.25,
                )
                self.assertAlmostEqual(
                    gmm_cost_cycles(tokens, TOPK, experts, hidden, n) / 1e3 / gmm,
                    1,
                    delta=0.25,
                )

    def test_prefers_ring_at_decode_sizes(self):
        from sglang_tpu.moe.pallas.moe_ring import prefers_ring

        for tokens in (1, 2, 8, 32, 64, 128, 256, 512):
            for hidden, n in ((4096, 256), (4096, 512), (4096, 128), (2048, 256)):
                with self.subTest(tokens=tokens, hidden=hidden, n=n):
                    self.assertTrue(prefers_ring(tokens, TOPK, EXPERTS, hidden, n))

    def test_few_experts_many_tokens_prefers_gmm(self):
        from sglang_tpu.moe.pallas.moe_ring import prefers_ring

        # Every expert gets hundreds of rows: row tiles, not weight bytes, set the
        # time, and gmm_v2's larger tiles win (measured from about 128 tokens on
        # 16 experts).
        self.assertFalse(prefers_ring(512, TOPK, 16, HIDDEN, INTER))
        self.assertTrue(prefers_ring(16, TOPK, 16, HIDDEN, INTER))


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestMoeRing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.envs import envs as tpu_envs

        stack = contextlib.ExitStack()
        cls.addClassCleanup(stack.close)
        stack.enter_context(
            tpu_envs.SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD.override(512)
        )
        keys = iter(jax.random.split(jax.random.PRNGKey(0), 8))
        f8 = jnp.float8_e4m3fn
        cls.w1 = (
            jax.random.normal(next(keys), (EXPERTS, HIDDEN, 2 * INTER)) * 0.5
        ).astype(f8)
        cls.w2 = (jax.random.normal(next(keys), (EXPERTS, INTER, HIDDEN)) * 0.5).astype(
            f8
        )
        cls.s1 = jax.random.uniform(
            next(keys), (EXPERTS, 1, 1, 2 * INTER), minval=1e-3, maxval=1e-2
        )
        cls.s2 = jax.random.uniform(
            next(keys), (EXPERTS, 1, 1, HIDDEN), minval=1e-3, maxval=1e-2
        )

    def routed(self, x, ids, wts):
        """The gmm_v2 path, with both streaming dispatches disabled."""
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        with (
            mock.patch(
                "sglang_tpu.moe.pallas.moe_stream.maybe_moe_stream", return_value=None
            ),
            mock.patch(
                "sglang_tpu.moe.pallas.moe_ring.maybe_moe_ring", return_value=None
            ),
        ):
            return _fused_moe_jax(
                x,
                self.w1,
                self.w2,
                wts,
                ids,
                self.s1,
                self.s2,
                None,
                None,
                topk=TOPK,
                activation_name=f"silu_and_mul_clamp:{LIMIT}",
                permute_dense_threshold=512,
            )

    def ring(self, x, ids, wts, **kw):
        from sglang_tpu.moe.pallas.moe_ring import moe_ring

        return moe_ring(x, ids, wts, self.w1, self.s1, self.w2, self.s2, LIMIT, **kw)

    def batch(self, tokens, seed, pool=EXPERTS):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        x = (jax.random.normal(k1, (tokens, HIDDEN)) * 3).astype(jnp.bfloat16)
        ids = jnp.stack(
            [
                jax.random.permutation(k, pool)[:TOPK]
                for k in jax.random.split(k2, tokens)
            ]
        ).astype(jnp.int32)
        wts = jax.nn.softmax(jax.random.normal(k3, (tokens, TOPK))).astype(jnp.bfloat16)
        return x, ids, wts

    @staticmethod
    def edge_ids(ids):
        ids = ids.at[:, 0].set(5)  # one expert in every row: several row tiles
        ids = ids.at[1, 1].set(ids[1, 2])  # duplicate top-k
        return ids.at[-1].set(-1)  # padding row

    def test_one_hot_weights_bit_identical(self):
        """Exact sums: every routed (token, expert) output equals gmm_v2's."""
        for tokens, seed, pool, kw in (
            (16, 1, EXPERTS, {}),
            (64, 2, EXPERTS, {}),
            (192, 3, EXPERTS, {}),
            (32, 4, 12, {}),  # few experts: every count class, multi-tile experts
            (24, 5, 40, dict(tile_rows=(4, 16), slots=3, w1_chunks=4, w2_chunks=2)),
            (16, 6, 40, dict(tile_rows=(8,), slots=2)),
        ):
            with self.subTest(tokens=tokens, pool=pool, **kw):
                x, ids, _ = self.batch(tokens, seed, pool)
                ids = self.edge_ids(ids)
                hot = jax.random.randint(jax.random.PRNGKey(seed), (tokens,), 0, TOPK)
                wts = jax.nn.one_hot(hot, TOPK, dtype=jnp.bfloat16)
                np.testing.assert_array_equal(
                    np.asarray(self.ring(x, ids, wts, **kw), np.float32),
                    np.asarray(self.routed(x, ids, wts), np.float32),
                )

    def test_routing_weights_match_up_to_combine_order(self):
        for tokens, seed in ((16, 11), (64, 12), (128, 13)):
            with self.subTest(tokens=tokens):
                x, ids, wts = self.batch(tokens, seed)
                ids = self.edge_ids(ids)
                got = np.asarray(self.ring(x, ids, wts), np.float32)
                want = np.asarray(self.routed(x, ids, wts), np.float32)
                # f32 sums in expert vs top-k order: at most one bf16 rounding apart.
                np.testing.assert_allclose(got, want, rtol=2**-7, atol=1e-6)
                np.testing.assert_array_equal(got[-1], 0)

    def test_all_padding(self):
        x, ids, wts = self.batch(16, 21)
        out = self.ring(x, jnp.full_like(ids, -1), wts)
        np.testing.assert_array_equal(np.asarray(out, np.float32), 0)

    def test_dispatch(self):
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        x, ids, wts = self.batch(32, 31)
        call = dict(
            topk=TOPK,
            activation_name=f"silu_and_mul_clamp:{LIMIT}",
            permute_dense_threshold=512,
        )
        args = (x, self.w1, self.w2, wts, ids, self.s1, self.s2, None, None)
        with mock.patch(
            "sglang_tpu.moe.pallas.moe_ring.moe_ring", wraps=lambda *a, **k: x
        ) as ring:
            # No flag: the kernel takes every call it covers and its model prefers.
            out = _fused_moe_jax(*args, **call)
            self.assertTrue(ring.called)
            self.assertIs(out, x)
            ring.reset_mock()
            with mock.patch(
                "sglang_tpu.moe.pallas.moe_ring.prefers_ring", return_value=False
            ):
                _fused_moe_jax(*args, **call)
            self.assertFalse(ring.called)
            # Expert parallelism keeps the gmm path.
            _fused_moe_jax(*args, **call, global_num_experts=2 * EXPERTS)
            self.assertFalse(ring.called)
            # One token is not the ring's case.
            ring.reset_mock()
            one = (
                x[:1],
                self.w1,
                self.w2,
                wts[:1],
                ids[:1],
                self.s1,
                self.s2,
                None,
                None,
            )
            _fused_moe_jax(*one, **call)
            self.assertFalse(ring.called)

    def test_fit_moe_ring(self):
        from sglang_tpu.moe.pallas.moe_ring import fit_moe_ring

        self.assertTrue(fit_moe_ring(2, HIDDEN, 2 * INTER))
        self.assertTrue(fit_moe_ring(256, HIDDEN, 2 * INTER))
        # The row buffers grow with tokens * hidden until they pass the VMEM limit.
        self.assertFalse(fit_moe_ring(2048, HIDDEN, 2 * INTER))
        # Wider than one quantization block: not gmm_v2's down projection.
        self.assertTrue(fit_moe_ring(16, HIDDEN, 2 * 512))
        self.assertFalse(fit_moe_ring(16, HIDDEN, 2 * 640))
        # One EP8 rank's full-width experts are refused.
        self.assertFalse(fit_moe_ring(16, HIDDEN, 2 * 2048))


if __name__ == "__main__":
    unittest.main()
