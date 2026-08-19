"""Numerics and API-contract tests for fused-RS post-expert processing."""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - JAX is absent on some CPU CI agents
    _HAS_JAX = False

if _HAS_JAX:
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import (
        _post_expert_rms_norm,
        _synchronize_routing_from_source,
        _validate_postprocess_contract,
        _weighted_topk_reduce,
        moe_gmm_local_rs_nodedup,
    )


@unittest.skipUnless(_HAS_JAX, "Requires JAX")
class TestFusedMoeRsPostprocess(CustomTestCase):
    def test_fp32_reduce_rounds_once_at_output_boundary(self):
        expert_outputs = jnp.asarray(
            [154.0, -29.375, -218.0, 56.75, -108.5, -140.0, 40.25, 84.0],
            dtype=jnp.bfloat16,
        ).reshape(1, 8, 1)
        router_weights = jnp.asarray(
            [
                0.06237677112221718,
                0.08555051684379578,
                0.07051427662372589,
                0.15362077951431274,
                0.05436801165342331,
                0.3849273920059204,
                0.043871425092220306,
                0.1447708010673523,
            ],
            dtype=jnp.float32,
        ).reshape(1, 8)

        actual = _weighted_topk_reduce(
            expert_outputs,
            router_weights,
            output_dtype=jnp.bfloat16,
        )
        legacy_weights = router_weights.astype(jnp.bfloat16)
        legacy_actual = jnp.sum(expert_outputs * legacy_weights[:, :, None], axis=1)
        reference = np.sum(
            np.asarray(expert_outputs, dtype=np.float64)
            * np.asarray(router_weights[:, :, None], dtype=np.float64),
            axis=1,
        )

        self.assertEqual(actual.dtype, jnp.bfloat16)
        self.assertEqual(actual.item(), -45.5)
        self.assertLess(
            np.abs(np.asarray(actual, dtype=np.float64) - reference).item(),
            np.abs(np.asarray(legacy_actual, dtype=np.float64) - reference).item(),
        )

    def test_bfloat16_postprocess_matches_legacy_exactly(self):
        expert_outputs = jnp.asarray(
            np.random.default_rng(123).normal(size=(3, 8, 17)),
            dtype=jnp.bfloat16,
        )
        router_weights = jnp.asarray(
            np.random.default_rng(456).uniform(size=(3, 8)),
            dtype=jnp.bfloat16,
        )
        norm_weight = jnp.asarray(
            np.random.default_rng(789).normal(scale=0.05, size=(17,)),
            dtype=jnp.bfloat16,
        )

        normalized = _post_expert_rms_norm(expert_outputs, norm_weight).astype(
            jnp.bfloat16
        )
        actual = jnp.sum(normalized * router_weights[:, :, None], axis=1)

        pnw = norm_weight.astype(jnp.float32) + 1.0
        legacy_f32 = expert_outputs.astype(jnp.float32) * pnw[None, None, :]
        variance = jnp.sum(legacy_f32**2, axis=-1, keepdims=True) / 17
        legacy_normalized = (legacy_f32 * jax.lax.rsqrt(variance + 1e-8)).astype(
            jnp.bfloat16
        )
        expected = jnp.sum(legacy_normalized * router_weights[:, :, None], axis=1)

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_post_expert_norm_ignores_nan_padding(self):
        expert_outputs = jnp.arange(2 * 2 * 256, dtype=jnp.float32).reshape(2, 2, 256)
        expert_outputs = (expert_outputs / 31.0).astype(jnp.bfloat16)
        expert_outputs = expert_outputs.at[:, :, 192:].set(jnp.bfloat16(jnp.nan))
        weight = jnp.linspace(-0.25, 0.25, 192, dtype=jnp.float32).astype(jnp.bfloat16)

        actual = _post_expert_rms_norm(expert_outputs, weight)

        valid = (
            expert_outputs[:, :, :192].astype(jnp.float32)
            * (weight.astype(jnp.float32) + 1.0)[None, None, :]
        )
        variance = jnp.sum(valid**2, axis=-1, keepdims=True) / 192
        expected_valid = valid * jax.lax.rsqrt(variance + 1e-8)
        expected = jnp.pad(expected_valid, ((0, 0), (0, 0), (0, 64)))
        self.assertEqual(actual.dtype, jnp.float32)
        self.assertTrue(np.isfinite(np.asarray(actual)).all())
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_fp32_source_sync_preserves_low_bits_and_source_semantics(self):
        source_weights = jnp.asarray(
            [[0.10012345, 0.20023456, 0.30034567, 0.39945678]],
            dtype=jnp.float32,
        )
        weights = jnp.concatenate(
            [source_weights, source_weights * 2.0, source_weights * 3.0], axis=0
        )
        source_ids = jnp.asarray([[0, 31, 128, 255]], dtype=jnp.int32)
        ids = jnp.concatenate(
            [source_ids, (source_ids + 1) % 256, (source_ids + 2) % 256], axis=0
        )
        is_source = jnp.asarray([True, False, False])

        def synchronize(rank_ids, rank_weights, rank_is_source):
            return _synchronize_routing_from_source(
                rank_ids,
                rank_weights,
                route_source=rank_is_source,
                axis_name="model",
                postprocess_accum_dtype="float32",
            )

        actual_ids, actual_weights = jax.vmap(synchronize, axis_name="model")(
            ids, weights, is_source
        )
        expected_ids = jnp.broadcast_to(source_ids, actual_ids.shape)
        expected_weights = jnp.broadcast_to(source_weights, actual_weights.shape)

        self.assertTrue(
            np.any(
                np.asarray(source_weights)
                != np.asarray(source_weights.astype(jnp.bfloat16), dtype=np.float32)
            )
        )
        np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
        np.testing.assert_array_equal(
            np.asarray(actual_weights), np.asarray(expected_weights)
        )

        def synchronize_bfloat16(rank_ids, rank_weights, rank_is_source):
            return _synchronize_routing_from_source(
                rank_ids,
                rank_weights,
                route_source=rank_is_source,
                axis_name="model",
                postprocess_accum_dtype="bfloat16",
            )

        bf16_weights = weights.astype(jnp.bfloat16)
        bf16_ids, bf16_actual = jax.vmap(synchronize_bfloat16, axis_name="model")(
            ids, bf16_weights, is_source
        )
        bf16_expected = jnp.broadcast_to(
            source_weights.astype(jnp.bfloat16), bf16_actual.shape
        )
        np.testing.assert_array_equal(np.asarray(bf16_ids), np.asarray(expected_ids))
        np.testing.assert_array_equal(
            np.asarray(bf16_actual), np.asarray(bf16_expected)
        )

    def test_precision_contract_rejects_implicit_or_unsupported_dtypes(self):
        _validate_postprocess_contract(jnp.ones((1, 1), dtype=jnp.bfloat16), "bfloat16")
        _validate_postprocess_contract(jnp.ones((1, 1), dtype=jnp.float32), "float32")
        with self.assertRaisesRegex(TypeError, "requires router weights"):
            _validate_postprocess_contract(
                jnp.ones((1, 1), dtype=jnp.float32), "bfloat16"
            )
        with self.assertRaisesRegex(TypeError, "must be bfloat16 or float32"):
            _validate_postprocess_contract(
                jnp.ones((1, 1), dtype=jnp.float16), "bfloat16"
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _validate_postprocess_contract(
                jnp.ones((1, 1), dtype=jnp.float32), "float64"
            )

    def test_non_sp_routing_rejects_more_than_256_experts(self):
        with self.assertRaisesRegex(ValueError, "at most 256 experts"):
            moe_gmm_local_rs_nodedup(
                jnp.zeros((16, 1), dtype=jnp.bfloat16),
                jnp.zeros((257, 1, 1), dtype=jnp.bfloat16),
                None,
                None,
                jnp.zeros((257, 1, 1), dtype=jnp.bfloat16),
                None,
                None,
                None,
                None,
                jnp.asarray([0, 0], dtype=jnp.int32),
                jnp.ones((16, 1), dtype=jnp.bfloat16),
                jnp.zeros((16, 1), dtype=jnp.int32),
                activation="silu",
                topk=1,
                ep_size=1,
                sp_enabled=False,
            )


if __name__ == "__main__":
    unittest.main()
