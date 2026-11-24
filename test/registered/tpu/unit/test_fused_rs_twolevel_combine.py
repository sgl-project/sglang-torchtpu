# SPDX-License-Identifier: Apache-2.0
"""Two-level fused-RS post-norm combine: the VMEM scatter-add kernel against the 0/1-matmul
fallback, in CPU interpret mode with the DMA race detector."""

import contextlib
import os
import sys
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
# TPU stays the default backend (it sizes the kernel's VMEM); the kernels run on CPU.
os.environ["JAX_PLATFORMS"] = "tpu,cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.moe.pallas.fused_rs import fused_moe

TOPK = 8
BLOCKS = 2  # receive blocks summed per row
H1, H2 = 8, 128


def _combine(chunk: int, *, scatter: bool) -> tuple[np.ndarray, bool]:
    """(_twolevel_post_combine's [chunk, H] output, whether a race was found)."""
    rows = chunk * TOPK
    rng = np.random.default_rng(chunk)
    cpu = jax.devices("cpu")[0]

    # Committed to CPU, so the kernels run there; the default device stays the TPU,
    # whose kind get_tpu_info reads.
    def put(a, dtype):
        return jax.device_put(jnp.asarray(a, dtype), cpu)

    recv = put(rng.normal(size=(BLOCKS * rows, H1, H2)), jnp.bfloat16)
    norm_weight = put(rng.normal(size=H1 * H2) * 0.1, jnp.bfloat16)
    topk_indices = put(rng.integers(0, 16, size=(chunk, TOPK)), jnp.int32)
    topk_weights = put(rng.uniform(0.25, 1.0, size=(chunk, TOPK)), jnp.bfloat16)
    fallback = mock.patch.object(
        fused_moe, "supports_twolevel_post_combine_scatter", return_value=False
    )
    interpreter = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
    # The detector gives each DMA a random virtual clock slot; fix the draw.
    np.random.seed(0)
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            contextlib.nullcontext() if scatter else fallback,
            pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)),
        ):
            out = fused_moe._twolevel_post_combine(
                recv,
                topk_indices,
                topk_weights,
                norm_weight,
                my_id=jnp.int32(0),
                chunk_size=chunk,
                topk=TOPK,
                hidden_size=H1 * H2,
            )
            out = np.asarray(out.astype(jnp.float32))
        return out, interpreter.races.races_found
    finally:
        pltpu.reset_tpu_interpret_mode_state()


class TestTwolevelCombine(CustomTestCase):
    def test_scatter_matches_matmul_fallback(self):
        """Every (token, slot) row is added to its token exactly once, with the matmul
        path's bf16 casts, at chunks whose last row step and token store overlap the
        previous ones (37, 513 tokens) and whose last 256-row matmul-path block is
        partial."""
        for chunk in (37, 513):
            with self.subTest(chunk=chunk):
                self.assertTrue(
                    fused_moe.supports_twolevel_post_combine_scatter(
                        jax.ShapeDtypeStruct(
                            (BLOCKS * chunk * TOPK, H1, H2), jnp.bfloat16
                        ),
                        rows=chunk * TOPK,
                        chunk=chunk,
                        out_dtype=jnp.bfloat16,
                    )
                )
                got, got_race = _combine(chunk, scatter=True)
                want, want_race = _combine(chunk, scatter=False)
                self.assertFalse(got_race, "DMA race in the scatter combine")
                self.assertFalse(want_race, "DMA race in the matmul-path combine")
                np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
