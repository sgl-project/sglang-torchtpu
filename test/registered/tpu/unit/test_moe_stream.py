# SPDX-License-Identifier: Apache-2.0
"""The streaming MoE kernels must match the route-meta + gmm_v2 fused-MoE path."""

import contextlib
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

# 288 tensor-parallel experts of width 256.
HIDDEN, INTER, EXPERTS, TOPK, LIMIT = 4096, 256, 288, 8, 10.0


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestMoeStreamBs1(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.envs import envs as tpu_envs

        stack = contextlib.ExitStack()
        cls.addClassCleanup(stack.close)
        stack.enter_context(
            tpu_envs.SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL.override(True)
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
        cls.keys = keys

    def inputs(self, seed):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        x = (jax.random.normal(k1, (1, HIDDEN)) * 3).astype(jnp.bfloat16)
        ids = jax.random.permutation(k2, EXPERTS)[:TOPK].astype(jnp.int32)
        wts = jax.nn.softmax(jax.random.normal(k3, (1, TOPK))).astype(jnp.bfloat16)
        return x, ids.reshape(1, TOPK), wts

    def routed(self, x, ids, wts):
        """The route-meta + gmm_v2 path, with the streaming dispatch disabled."""
        from sglang_tpu.layers.moe.pallas.fused_moe_op import _fused_moe_jax

        with mock.patch(
            "sglang_tpu.layers.moe.pallas.moe_stream.maybe_moe_stream",
            return_value=None,
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

    def test_bit_identical_to_route_meta_path(self):
        from sglang_tpu.layers.moe.pallas.moe_stream import moe_stream_bs1

        for seed in range(5):
            with self.subTest(seed=seed):
                x, ids, wts = self.inputs(seed)
                expected = self.routed(x, ids, wts)
                actual = moe_stream_bs1(
                    x, ids, wts, self.w1, self.s1, self.w2, self.s2, LIMIT
                )
                np.testing.assert_array_equal(
                    np.asarray(actual, np.float32), np.asarray(expected, np.float32)
                )

    def test_padded_slots_contribute_nothing(self):
        from sglang_tpu.layers.moe.pallas.moe_stream import moe_stream_bs1

        x, ids, wts = self.inputs(1)
        padded = ids.at[0, TOPK // 2 :].set(-1)
        kept = wts.at[0, TOPK // 2 :].set(0)
        np.testing.assert_array_equal(
            np.asarray(
                moe_stream_bs1(
                    x, padded, wts, self.w1, self.s1, self.w2, self.s2, LIMIT
                ),
                np.float32,
            ),
            np.asarray(
                moe_stream_bs1(x, ids, kept, self.w1, self.s1, self.w2, self.s2, LIMIT),
                np.float32,
            ),
        )

    def test_dispatch_runs_stream_kernel(self):
        from sglang_tpu.layers.moe.pallas.fused_moe_op import _fused_moe_jax

        x, ids, wts = self.inputs(2)
        out = _fused_moe_jax(
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
        np.testing.assert_array_equal(
            np.asarray(out, np.float32),
            np.asarray(self.routed(x, ids, wts), np.float32),
        )

    def test_dispatch_declines_outside_its_case(self):
        from sglang_tpu.layers.moe.pallas.moe_stream import maybe_moe_stream

        x, ids, wts = self.inputs(0)
        common = dict(activation_name=f"silu_and_mul_clamp:{LIMIT}", w_quant_bits=0)
        args = (self.w1, self.w2, wts, ids, self.s1, self.s2, None, None)
        self.assertIsNotNone(
            maybe_moe_stream(
                x, *args, no_combine=False, use_group_offset=False, **common
            )
        )
        # Expert parallelism and no_combine keep the gmm path.
        self.assertIsNone(
            maybe_moe_stream(
                x, *args, no_combine=False, use_group_offset=True, **common
            )
        )
        self.assertIsNone(
            maybe_moe_stream(
                x, *args, no_combine=True, use_group_offset=False, **common
            )
        )
        # Plain SiLU fuses the activation into gmm_v2 and keeps that path.
        self.assertIsNone(
            maybe_moe_stream(
                x,
                *args,
                no_combine=False,
                use_group_offset=False,
                activation_name="silu",
                w_quant_bits=0,
            )
        )
        # Block-scaled and unquantized experts keep the gmm path.
        block_s1 = jnp.ones((EXPERTS, HIDDEN // 128, 1, 2 * INTER), jnp.float32)
        self.assertIsNone(
            maybe_moe_stream(
                x,
                self.w1,
                self.w2,
                wts,
                ids,
                block_s1,
                self.s2,
                None,
                None,
                no_combine=False,
                use_group_offset=False,
                **common,
            )
        )
        self.assertIsNone(
            maybe_moe_stream(
                x,
                self.w1.astype(jnp.bfloat16),
                self.w2.astype(jnp.bfloat16),
                wts,
                ids,
                None,
                None,
                None,
                None,
                no_combine=False,
                use_group_offset=False,
                **common,
            )
        )
        # A verify window takes the few-token kernel; wider batches keep gmm.
        for tokens, expect in ((2, True), (8, True), (9, False)):
            out = maybe_moe_stream(
                jnp.tile(x, (tokens, 1)),
                self.w1,
                self.w2,
                jnp.tile(wts, (tokens, 1)),
                jnp.tile(ids, (tokens, 1)),
                self.s1,
                self.s2,
                None,
                None,
                no_combine=False,
                use_group_offset=False,
                **common,
            )
            self.assertEqual(out is not None, expect, tokens)

    def test_few_tokens_match_route_meta_path(self):
        """Verify windows (2..8 tokens) match the routed path up to the f32
        combine order, including shared experts and padding rows."""
        from sglang_tpu.layers.moe.pallas.moe_stream import moe_stream_few_tokens

        for tokens, seed, pad in ((2, 10, False), (6, 11, False), (8, 12, True)):
            with self.subTest(tokens=tokens, pad=pad):
                k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
                x = (jax.random.normal(k1, (tokens, HIDDEN)) * 3).astype(jnp.bfloat16)
                # A narrow expert range so tokens share experts.
                ids = jnp.stack(
                    [
                        jax.random.permutation(k, 24)[:TOPK]
                        for k in jax.random.split(k2, tokens)
                    ]
                ).astype(jnp.int32)
                wts = jax.nn.softmax(jax.random.normal(k3, (tokens, TOPK))).astype(
                    jnp.bfloat16
                )
                if pad:
                    # A padding row routes to an invalid id and must come out zero.
                    ids = ids.at[-1].set(-1)
                expected = self.routed(x, ids, wts)
                actual = moe_stream_few_tokens(
                    x, ids, wts, self.w1, self.s1, self.w2, self.s2, LIMIT
                )
                rows = tokens - 1 if pad else tokens
                np.testing.assert_allclose(
                    np.asarray(actual[:rows], np.float32),
                    np.asarray(expected[:rows], np.float32),
                    rtol=2e-2,
                    atol=2e-2,
                )
                if pad:
                    np.testing.assert_array_equal(np.asarray(actual[-1], np.float32), 0)


if __name__ == "__main__":
    unittest.main()
