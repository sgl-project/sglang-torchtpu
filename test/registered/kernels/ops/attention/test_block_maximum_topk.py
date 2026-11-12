"""Native SparseCore exact selection: ties, masks, publication and VMEM plans."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.kernels.block_topk import (
    adversarial_scores,
    exact_topk,
    published_blocks,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kernels.deepseek_v41.index_selection import block_topk as impl

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


class TestBlockMaximumTopK(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if jax.default_backend() != "tpu":
            raise unittest.SkipTest("native SparseCore execution requires TPU")
        cls.info = pltpu.get_tpu_info()
        if cls.info.sparse_core is None:
            raise unittest.SkipTest("target has no SparseCore")

    def test_exact_contract_resident_and_streamed(self):
        k, block_size, rows = 512, 8, 16
        plans = {}
        # Bound the matrix: one representative of each physical execution plan.
        for width in (8192, 16384, 32768, 65536, 131072, 262144, 524288):
            if not impl.is_supported(rows, width, k, block_size, publish=128):
                continue
            blocks = impl.default_blocks(k, self.info.sparse_core.num_lanes)
            block_width = (
                (width // block_size + self.info.num_lanes - 1)
                // self.info.num_lanes
                * self.info.num_lanes
            )
            plan = impl._plan(rows, width, block_width, k, blocks, block_size, 128)
            plans.setdefault(plan[0], width)
        self.assertEqual(set(plans), {True, False}, "must exercise both VMEM plans")
        for resident, width in plans.items():
            scores, lengths = adversarial_scores(width, k, block_size)
            expected = exact_topk(scores, lengths, k)
            for publish in (0, 128):
                with self.subTest(resident=resident, width=width, publish=publish):
                    run = jax.jit(
                        lambda s, n: impl.block_topk(
                            s, n, k=k, block_size=block_size, publish=publish
                        )
                    )
                    result = jax.device_get(
                        run(jnp.asarray(scores), jnp.asarray(lengths))
                    )
                    if publish:
                        selected, published = result
                        np.testing.assert_array_equal(
                            published,
                            published_blocks(scores, lengths, publish, block_size),
                        )
                    else:
                        selected = result
                    # Full array equality checks order, exact membership and -1 suffix.
                    np.testing.assert_array_equal(selected, expected)

    def test_buffer_reuse_and_full_span_boundary(self):
        # More than three rows per vector subcore wraps both score slots and
        # exercises previous-output waits, including empty/nonempty rows.
        rows = (
            self.info.sparse_core.num_cores * self.info.sparse_core.num_subcores * 3 + 1
        )
        for width, block_size, k in [
            (8192, 16, 128),
            (65536, 8, 512),
            (131072, 8, 512),
        ]:
            with self.subTest(width=width, block_size=block_size):
                self.assertTrue(
                    impl.is_supported(rows, width, k, block_size, publish=128)
                )
                scores, lengths = adversarial_scores(width, k, block_size)
                owners = np.arange(rows) % len(scores)
                scores, lengths = scores[owners], lengths[owners]
                selected, published = jax.jit(
                    lambda s, n: impl.block_topk(
                        s, n, k=k, block_size=block_size, publish=128
                    )
                )(jnp.asarray(scores), jnp.asarray(lengths))
                np.testing.assert_array_equal(
                    np.asarray(selected), exact_topk(scores, lengths, k)
                )
                np.testing.assert_array_equal(
                    np.asarray(published),
                    published_blocks(scores, lengths, 128, block_size),
                )
        self.assertFalse(impl.is_supported(1, 2176, 128, 16))

    def test_positive_infinity_token_selection(self):
        width, k, block_size = 16384, 512, 8
        self.assertTrue(impl.is_supported(1, width, k, block_size))
        scores = np.full((1, width), -np.inf, np.float32)
        scores[0, ::7] = np.inf
        lengths = np.array([width], np.int32)
        selected, published = jax.jit(
            lambda s, n: impl.block_topk(s, n, k=k, block_size=block_size, publish=128)
        )(jnp.asarray(scores), jnp.asarray(lengths))
        np.testing.assert_array_equal(
            np.asarray(selected), exact_topk(scores, lengths, k)
        )
        np.testing.assert_array_equal(
            np.asarray(published), published_blocks(scores, lengths, 128, block_size)
        )


if __name__ == "__main__":
    unittest.main()
