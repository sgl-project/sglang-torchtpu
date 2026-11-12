"""Shape/resource admission for block top-k, without accelerator execution."""

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.block_topk import exact_topk, published_blocks
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kernels.deepseek_v41.index_selection import block_topk as impl

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBlockMaximumTopKAdmission(CustomTestCase):
    def test_shapes_and_vmem_admission(self):
        info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
        with (
            patch.object(jax, "default_backend", return_value="tpu"),
            patch.object(pltpu, "get_tpu_info", return_value=info),
            patch.object(plsc, "get_sparse_core_info", return_value=info.sparse_core),
        ):
            self.assertTrue(impl.is_supported(16, 16384, 512, 8, publish=128))
            for args, publish in [
                ((0, 16384, 512, 8), 0),
                ((16, 16383, 512, 8), 0),
                ((16, 16384, 511, 8), 0),
                ((16, 16384, 0, 8), 0),
                ((16, 16384, 16384, 8), 0),
                ((16, 16384, 512, 3), 0),
                # HBM-aligned but final full-span maxima gather would overrun.
                ((1, 2176, 128, 16), 0),
                ((16, 16384, 512, 0), 0),
                ((16, 4096, 512, 8), 0),
                ((16, 16384, 512, 8), -128),
                ((16, 16384, 512, 8), 1),
                ((16, 16384, 512, 8), 4096),
                # The maxima alone exceed per-subcore VMEM.
                ((16, 1 << 24, 512, 8), 0),
            ]:
                with self.subTest(args=args, publish=publish):
                    self.assertFalse(impl.is_supported(*args, publish=publish))

    def test_plain_cpu_is_not_accelerator_admission(self):
        with patch.object(jax, "default_backend", return_value="cpu"):
            self.assertFalse(impl.is_supported(16, 16384, 512, 8))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                impl.block_topk(
                    jnp.zeros((1, 16384), jnp.float32),
                    jnp.array([16384], jnp.int32),
                    k=512,
                    block_size=8,
                )

    def test_integer_oracle_distinguishes_zero_and_publication(self):
        # Pin independent oracle behavior so float max/argsort cannot erase -0.
        scores = np.array(
            [[-0.0, -0.0, 0.0, -np.inf, 1.0, 1.0, -9.0, -9.0]], np.float32
        )
        lengths = np.array([8], np.int32)
        np.testing.assert_array_equal(exact_topk(scores, lengths, 4), [[0, 2, 4, 5]])
        np.testing.assert_array_equal(published_blocks(scores, lengths, 2, 2), [[2, 3]])
        # Length clipping includes no invalid suffix and pads all empty rows.
        np.testing.assert_array_equal(exact_topk(scores, [-1], 4), [[-1] * 4])
        np.testing.assert_array_equal(exact_topk(scores, [99], 4), [[0, 2, 4, 5]])


if __name__ == "__main__":
    unittest.main()
