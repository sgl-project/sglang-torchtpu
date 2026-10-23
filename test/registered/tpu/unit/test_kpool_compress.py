# SPDX-License-Identifier: Apache-2.0
"""Exact cache-byte tests for the complete raw-to-pooled Pallas pipeline."""

import os
import unittest
from functools import partial

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas import dsa_kpool_topk as kp
from sglang_tpu.attention.pallas.kpool_compress import compress_and_store


def reference(raw, old, ape, lens, pages, cu, distribution, tokens):
    """Original vectorized gather/compress/scatter, before the fused writer."""
    req, pos, valid = kp._token_rows(lens, cu, distribution, tokens)
    closing = valid & (pos % 4 == 3)
    member = jnp.maximum(pos[:, None] - 3 + jnp.arange(4)[None, :], 0)
    loc = {
        "page_indices": pages,
        "num_seqs": lens.shape[0],
        "page_size": 128,
        "num_pages": raw.shape[0],
    }
    page, slot = kp._page_slot(
        jnp.broadcast_to(req[:, None], member.shape),
        member,
        valid=jnp.broadcast_to(closing[:, None], member.shape),
        **loc,
    )
    rows = raw[jnp.minimum(page, raw.shape[0] - 1), slot]
    pooled = kp.compress_pools(rows[..., :128], rows[..., 128:], ape)
    records = kp._pool_records(pooled, 256)
    page, slot = kp._page_slot(req, pos // 4, valid=closing, **loc)
    return old.at[page, slot // 4, slot % 4].set(records, mode="drop")


class TestKpoolCompress(unittest.TestCase):
    def test_exact_cache_bytes(self):
        ref = jax.jit(reference, static_argnums=(7,))
        fused = jax.jit(compress_and_store, static_argnums=(7,))
        specs = [
            ("full_pages", [0, 0], [128, 128], 256, "random"),
            ("raw_page_boundary", [124, 508], [12, 8], 32, "random"),
            ("pooled_page_boundary", [504, 508], [20, 12], 32, "random"),
            ("decode_close", [3, 131], [1, 1], 32, "random"),
            ("decode_no_close", [4, 130], [1, 1], 32, "random"),
            ("partial_quad", [4, 132], [8, 8], 32, "random"),
            ("inactive_request", [3, 0], [1, 0], 32, "random"),
            ("all_padding", [0, 0], [0, 0], 32, "random"),
            ("zeros", [0, 0], [16, 16], 32, "zero"),
            ("tiny", [0, 0], [16, 16], 32, "tiny"),
            ("nonfinite", [0, 0], [16, 16], 32, "nonfinite"),
            ("shared_quad", [0, 4], [4, 4], 32, "shared"),
            ("decode_shared_quad", [3, 7], [1, 1], 32, "shared"),
            ("verify_q6", [131, 130], [6, 6], 32, "random"),
            ("verify_tile_boundary", [15, 31], [6, 6], 32, "random"),
            ("verify_page_boundary", [127, 511], [6, 6], 32, "random"),
            ("verify_mixed_q", [125, 511], [3, 9], 32, "random"),
            ("verify_shared_quad", [1, 8], [6, 8], 32, "shared"),
        ]
        for i, (name, start, new, tokens, mode) in enumerate(specs):
            with self.subTest(name=name):
                rng = np.random.default_rng(1024 + i)
                raw = rng.normal(size=(17, 128, 256)).astype(np.float32)
                old = rng.integers(0, 256, (17, 32, 4, 256), dtype=np.uint8)
                pages = rng.permutation(np.arange(1, 17)).astype(np.int32)
                if mode == "shared":
                    pages[8] = pages[0]
                elif mode == "zero":
                    raw[..., :128] = 0
                elif mode == "tiny":
                    raw[..., :128] *= 1e-7
                elif mode == "nonfinite":
                    raw[pages[0], 0, 0] = np.nan
                    raw[pages[0], 4, 0] = np.inf
                    raw[pages[0], 8, 0] = -np.inf
                active = 0 if sum(new) == 0 else 1 if new[1] == 0 else 2
                args = (
                    jnp.asarray(raw, jnp.bfloat16),
                    jnp.asarray(old),
                    jnp.asarray(rng.normal(size=(4, 128)), jnp.float32),
                    jnp.asarray(np.array(start) + np.array(new), jnp.int32),
                    jnp.asarray(pages),
                    jnp.asarray([0, new[0], sum(new)], jnp.int32),
                    jnp.asarray([0, 0, active], jnp.int32),
                    tokens,
                )
                expected = np.asarray(ref(*args))
                actual = np.asarray(fused(*args))
                # Includes all inactive records, record padding, FP8 keys and scales.
                np.testing.assert_array_equal(actual, expected)
                bounded = jax.jit(
                    partial(compress_and_store, max_query_length=max(1, max(new))),
                    static_argnums=(7,),
                )
                np.testing.assert_array_equal(np.asarray(bounded(*args)), expected)


if __name__ == "__main__":
    unittest.main()
