"""CPU numerical checks of the actual scale helper; TPU lane supplies JAX.

Extract the pure helper to avoid initializing a device during numerical tests.
This does not claim to test compiled GMM performance or end-to-end accuracy.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import jax.numpy as jnp
import numpy as np
import torch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)
ROOT = Path(__file__).resolve().parents[4]


def extract(path, names, ns):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names
    ]
    exec(compile(tree, str(path), "exec"), ns)
    return ns


class TestFp32MoeScales(unittest.TestCase):
    def test_scale_math_and_production_parity(self):
        ns = extract(
            "sglang_tpu/layers/moe/pallas/gmm_v2.py",
            {"_scale_quantized_partial"},
            {"jnp": jnp},
        )
        scale = ns["_scale_quantized_partial"]
        rng = np.random.default_rng(7)
        partial = jnp.array(rng.normal(size=(16, 128)), jnp.bfloat16)
        lhs = jnp.array(rng.uniform(0.01, 1, size=(16, 1)), jnp.bfloat16)
        rhs = jnp.array(rng.uniform(0.01, 1, size=(1, 128)), jnp.float32)
        for weight_scale in [rhs, None]:
            for precise in [False, True]:
                dtype = jnp.float32 if precise else jnp.bfloat16
                expected = partial.astype(dtype) * lhs.astype(dtype)
                if weight_scale is not None:
                    expected = expected * weight_scale.astype(dtype)
                got = scale(partial, lhs, weight_scale, scales_dtype=dtype)
                self.assertEqual(got.dtype, partial.dtype)
                np.testing.assert_array_equal(got, expected.astype(partial.dtype))
        self.assertTrue(
            np.any(
                np.asarray(scale(partial, lhs, rhs, scales_dtype=jnp.float32))
                != np.asarray(scale(partial, lhs, rhs, scales_dtype=jnp.bfloat16))
            )
        )
        zeros = scale(
            jnp.zeros_like(partial), jnp.zeros_like(lhs), rhs, scales_dtype=jnp.float32
        )
        np.testing.assert_array_equal(zeros, np.zeros(partial.shape))

    def test_loader_preserves_fp32_scale_values(self):
        for enabled in [False, True]:
            ns = {
                "torch": torch,
                "envs": NS(SGLANG_TPU_FP8_MOE_FP32_SCALES=NS(get=lambda: enabled)),
            }
            extract(
                "sglang_tpu/layers/quantization/fp8_moe.py",
                {"_own_storage", "_to_tpu_fp8_moe_scale_layout"},
                ns,
            )
            original = torch.tensor(
                [[1.001, 0.5001, 0.2501, 0.1251]], dtype=torch.float32
            )
            result = ns["_to_tpu_fp8_moe_scale_layout"](original, 4)
            expected_dtype = torch.float32 if enabled else torch.bfloat16
            self.assertEqual(result.dtype, expected_dtype)
            torch.testing.assert_close(
                result.reshape(1, 4), original.to(expected_dtype), rtol=0, atol=0
            )
            self.assertEqual(original.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
