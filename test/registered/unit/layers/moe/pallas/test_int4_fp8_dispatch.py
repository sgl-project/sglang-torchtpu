"""Validate one-token vmem-fp8 EP routing and the explicit channel-scale arguments."""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# The CPU lane has no JAX and skips; the TPU lane runs it.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestInt4Fp8Dispatch(CustomTestCase):
    def test_bs1_ep_window_and_padding(self):
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        w1 = jax.ShapeDtypeStruct((48, 896, 4096), jnp.int32)
        w2 = jax.ShapeDtypeStruct((48, 256, 7168), jnp.int32)
        s1 = jax.ShapeDtypeStruct((48, 224, 1, 4096), jnp.float32)
        s2 = jax.ShapeDtypeStruct((48, 64, 1, 7168), jnp.float32)
        c1 = jax.ShapeDtypeStruct((48, 1, 4096), jnp.float32)
        c2 = jax.ShapeDtypeStruct((48, 1, 7168), jnp.float32)
        for ids in (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [49, 2, 51, 4, 53, 6, 55, 8],
            [55, 54, 53, 52, 51, 50, 49, 48],
        ):
            calls = []

            def fake_gmm(lhs, rhs, group_sizes, **kwargs):
                calls.append(kwargs)
                sizes = np.asarray(group_sizes)
                # 8 real routes; the sublane-pad token's 8 slots carry -1 ids
                # and are dropped, so no GMM row goes to them.
                self.assertEqual(int(sizes.sum()), 8)
                self.assertEqual(int(sizes[0]), ids.count(0))
                width = rhs.shape[-1] // (2 if kwargs.get("fuse_act") else 1)
                out = jnp.zeros((lhs.shape[0], width), jnp.bfloat16)
                start = int(sizes[:48].sum())
                for e, count in enumerate(sizes[48:96]):
                    count = int(count)
                    out = out.at[start : start + count].set(e + 1)
                    start += count
                return out

            with (
                self.subTest(ids=ids),
                mock.patch("sglang_tpu.moe.pallas.gmm_v2.gmm_v2", side_effect=fake_gmm),
                # bf16 sublane tiling on TPU (m pads from 1 to 2 at top-8), here
                # without a TPU backend for JAX.
                mock.patch(
                    "sglang_tpu.moe.pallas.gmm_v2.lhs_sublane_tiling",
                    return_value=16,
                ),
            ):
                out = _fused_moe_jax(
                    jnp.ones((1, 7168), jnp.bfloat16),
                    w1,
                    w2,
                    jnp.full((1, 8), 0.125, jnp.bfloat16),
                    jnp.asarray([ids], jnp.int32),
                    s1,
                    s2,
                    None,
                    None,
                    8,
                    "silu",
                    global_num_experts=384,
                    expert_group_offset=48,
                    w_quant_bits=4,
                    w1_channel_scale=c1,
                    w2_channel_scale=c2,
                )
                expected = sum((i - 48 + 1) / 8 for i in ids if 48 <= i < 96)
                np.testing.assert_array_equal(
                    np.asarray(out, np.float32),
                    np.full((1, 7168), expected, np.float32),
                )
                self.assertIsNone(calls[0]["fuse_act"])  # W4A8 runs GMM1 unfused
                self.assertIs(calls[0]["rhs_channel_scale"], c1)
                self.assertIs(calls[1]["rhs_channel_scale"], c2)
                self.assertIsNone(calls[0]["rhs_bias"])
                self.assertFalse(calls[0]["dequantize_rhs"])


if __name__ == "__main__":
    unittest.main()
