# SPDX-License-Identifier: Apache-2.0
"""Direct TPU correctness coverage for the batched-RPA kernel."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# SGLang platform discovery selects CPU-only JAX for export, so select the
# physical backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaKernel(unittest.TestCase):
    """Compare the kernel against pure-JAX attention on the token-major layout.

    16 q heads over 8 kv heads with kv_len 2049 -- a partial last page and an
    idle lane -- plus a check that the new K/V landed in the right cache slot.
    """

    PAGE_SIZE = 256
    PAGES_PER_SEQ = 9
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    TOLERANCE = 0.05

    def test_idle_lane_multihead_decode_and_writeback_match_reference(self):
        import jax
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.batched_rpa.utils import (
            get_dtype_packing,
        )
        from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
            ragged_paged_attention,
        )
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        kv_dtype = jnp.bfloat16
        kv_len = 2049
        total_pages = self.PAGES_PER_SEQ
        padded_q_tokens = 8

        key = jax.random.PRNGKey(17)
        q_key, new_k_key, new_v_key, cache_k_key, cache_v_key = jax.random.split(key, 5)
        q = jax.random.normal(
            q_key,
            (padded_q_tokens, self.NUM_Q_HEADS, self.HEAD_DIM),
            dtype=jnp.bfloat16,
        )
        new_k = jax.random.normal(
            new_k_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        new_v = jax.random.normal(
            new_v_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        cache_k = jax.random.normal(
            cache_k_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        cache_v = jax.random.normal(
            cache_v_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )

        page_indices_host = np.arange(total_pages, dtype=np.int32).reshape(1, -1)
        new_token_pos = kv_len - 1
        rel_page, slot = divmod(new_token_pos, self.PAGE_SIZE)
        abs_page = int(page_indices_host[0, rel_page])
        ref_k = cache_k.at[abs_page, slot].set(new_k[0])
        ref_v = cache_v.at[abs_page, slot].set(new_v[0])

        packing = get_dtype_packing(kv_dtype)
        unpacked_kv_heads = self.NUM_KV_HEADS * 2
        aligned_kv_heads = ((unpacked_kv_heads + packing - 1) // packing) * packing
        kv_cache_unpacked = jnp.stack([cache_k, cache_v], axis=3).reshape(
            total_pages,
            self.PAGE_SIZE,
            unpacked_kv_heads,
            self.HEAD_DIM,
        )
        kv_cache_unpacked = jnp.pad(
            kv_cache_unpacked,
            ((0, 0), (0, 0), (0, aligned_kv_heads - unpacked_kv_heads), (0, 0)),
        )
        kv_cache = kv_cache_unpacked.reshape(
            total_pages,
            self.PAGE_SIZE,
            aligned_kv_heads // packing,
            packing,
            self.HEAD_DIM,
        )

        kv_lens = jnp.asarray([kv_len], dtype=jnp.int32)
        cu_q_lens = jnp.asarray([0, 1], dtype=jnp.int32)
        page_indices = jnp.asarray(page_indices_host.reshape(-1))
        distribution = jnp.ones((3,), dtype=jnp.int32)
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)

        expected = ref_ragged_paged_attention(
            q[:1],
            ref_k,
            ref_v,
            kv_lens,
            jnp.asarray(page_indices_host),
            cu_q_lens,
            jnp.asarray([1], dtype=jnp.int32),
            sm_scale=sm_scale,
        )
        actual, updated_cache = ragged_paged_attention(
            q,
            new_k,
            new_v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
        )

        max_diff = jnp.max(
            jnp.abs(actual[:1].astype(jnp.float32) - expected.astype(jnp.float32))
        )
        self.assertLess(float(max_diff), self.TOLERANCE)

        unpacked_cache = updated_cache.reshape(
            total_pages, self.PAGE_SIZE, aligned_kv_heads, self.HEAD_DIM
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_cache[abs_page, slot, 0:unpacked_kv_heads:2]),
            np.asarray(new_k[0]),
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_cache[abs_page, slot, 1:unpacked_kv_heads:2]),
            np.asarray(new_v[0]),
        )


if __name__ == "__main__":
    unittest.main()
