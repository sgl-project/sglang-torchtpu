# SPDX-License-Identifier: Apache-2.0
"""Batched-RPA kernel vs ``ref_ragged_paged_attention`` with an fp8 KV cache.

fp8 e4m3 is what the RPA backends resolve ``--kv-cache-dtype auto`` to, yet the
unit file checks the batched kernel against a reference only for one bf16
decode, and the layout-equivalence test compares the two layouts with each
other in bf16. Here both layouts run the production sequence -- an extend
batch that fills an empty cache, then a step that mixes decodes with an
extend on the cached prefix -- each checked against the reference.

The cache starts zeroed and the kernel writes every token itself, so the test
never needs to know how either layout packs head_dim; the second step reads
only what the first one wrote.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-kernel",
    labels=["kernel", "attention", "rpa-batched"],
)

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


# Qwen3-Coder-30B-A3B at tp8 per rank: 4 q heads over one replicated KV head.
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
# The batched backend's page-size floor (SEQ_ALONG_LANE needs a 128 multiple).
PAGE_SIZE = 256
PAGES_PER_SEQ = 3
# Pages outside every sequence's table; they must stay zero.
SPARE_PAGES = 2
# Distinct powers of two: a k/v swap is caught and the reference's bf16
# dequant stays exact.
K_SCALE, V_SCALE = 0.5, 0.25
# Kernel feeds fp8 K/V to the MXU where the reference upcasts first.
FP8_TOLERANCE = 0.1

# Step 1, extend into an empty cache: (q_len, kv_len) per row; prompts end
# mid-page and row 1 spans two pages.
EXTEND_STEP = ((300, 300), (517, 517), (64, 64), (129, 129))
# Step 2: rows 0-1 decode (DECODE range), rows 2-3 run MIXED with one
# single-token and one 37-token extend on their cached prefixes.
MIXED_STEP = ((1, 301), (1, 518), (1, 65), (37, 166))
MIXED_DISTRIBUTION = (2, 2, 4)


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaReferenceModes(unittest.TestCase):
    def _quantize(self, t, scale):
        """Mirror of the backend's ``_quantize_kv_to_fp8``."""
        import jax.numpy as jnp

        finfo = jnp.finfo(jnp.float8_e4m3fn)
        s = jnp.clip(t.astype(jnp.float32) / scale, float(finfo.min), float(finfo.max))
        return s.astype(jnp.float8_e4m3fn)

    def _step(self, *, layout, cache, page_table, rows, distribution, history, seed):
        """Run one kernel call; compare with the reference over ``history``.

        ``history[i]`` holds row i's logical fp8 (K, V) and is extended here.
        Returns the updated cache.
        """
        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
            ragged_paged_attention,
        )
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        num_seqs = len(rows)
        total_q = sum(q for q, _ in rows)
        keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        q = jax.random.normal(keys[0], (total_q, NUM_Q_HEADS, HEAD_DIM), jnp.bfloat16)
        kv_shape = (total_q, NUM_KV_HEADS, HEAD_DIM)
        new_k = self._quantize(
            jax.random.normal(keys[1], kv_shape, jnp.bfloat16), K_SCALE
        )
        new_v = self._quantize(
            jax.random.normal(keys[2], kv_shape, jnp.bfloat16), V_SCALE
        )

        cu_q_lens = np.cumsum([0] + [q_len for q_len, _ in rows]).astype(np.int32)
        new_k_np, new_v_np = np.asarray(new_k), np.asarray(new_v)
        for i, (q_len, kv_len) in enumerate(rows):
            k_i, v_i = history[i]
            history[i] = (
                np.concatenate([k_i, new_k_np[cu_q_lens[i] : cu_q_lens[i + 1]]]),
                np.concatenate([v_i, new_v_np[cu_q_lens[i] : cu_q_lens[i + 1]]]),
            )
            self.assertEqual(len(history[i][0]), kv_len)

        page_shape = (cache.shape[0], PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
        ref_k = np.zeros(page_shape, dtype=new_k_np.dtype)
        ref_v = np.zeros(page_shape, dtype=new_v_np.dtype)
        for i, (k_i, v_i) in enumerate(history):
            pos = np.arange(len(k_i))
            pages = page_table[i, pos // PAGE_SIZE]
            ref_k[pages, pos % PAGE_SIZE] = k_i
            ref_v[pages, pos % PAGE_SIZE] = v_i

        kv_lens = jnp.asarray([kv_len for _, kv_len in rows], jnp.int32)
        sm_scale = HEAD_DIM**-0.5
        expected = ref_ragged_paged_attention(
            q,
            jnp.asarray(ref_k),
            jnp.asarray(ref_v),
            kv_lens,
            jnp.asarray(page_table),
            jnp.asarray(cu_q_lens),
            jnp.asarray([num_seqs], jnp.int32),
            sm_scale=sm_scale,
            k_scale=K_SCALE,
            v_scale=V_SCALE,
        )
        expected = np.asarray(expected.astype(jnp.float32))

        # q/k/v/cache are donated.
        out, cache = ragged_paged_attention(
            q,
            new_k,
            new_v,
            cache,
            kv_lens,
            jnp.asarray(page_table.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution, jnp.int32),
            sm_scale=sm_scale,
            k_scale=K_SCALE,
            v_scale=V_SCALE,
            kv_layout=layout,
        )
        got = np.asarray(out.astype(jnp.float32))
        diff = float(np.abs(got - expected).max())
        self.assertLess(
            diff, FP8_TOLERANCE, f"distribution={distribution}: max abs diff {diff}"
        )
        return cache

    def test_fp8_extend_then_mixed(self):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

        num_seqs = len(EXTEND_STEP)
        total_pages = num_seqs * PAGES_PER_SEQ + SPARE_PAGES
        rng = np.random.default_rng(0)
        page_table = (
            rng.permutation(total_pages)[: num_seqs * PAGES_PER_SEQ]
            .astype(np.int32)
            .reshape(num_seqs, PAGES_PER_SEQ)
        )
        spare = sorted(set(range(total_pages)) - set(page_table.ravel()))

        for layout in (
            configs.KVLayout.HEAD_ALONG_SUBLANE,
            configs.KVLayout.SEQ_ALONG_LANE,
        ):
            with self.subTest(layout=layout):
                cache = jnp.zeros(
                    wrapper.get_kv_cache_shape(
                        total_num_pages=total_pages,
                        page_size=PAGE_SIZE,
                        actual_num_kv_heads=NUM_KV_HEADS,
                        actual_head_dim=HEAD_DIM,
                        kv_dtype=jnp.float8_e4m3fn,
                        kv_layout=layout,
                    ),
                    jnp.float8_e4m3fn,
                )
                empty = np.zeros((0, NUM_KV_HEADS, HEAD_DIM), jnp.float8_e4m3fn)
                history = [(empty, empty) for _ in range(num_seqs)]
                cache = self._step(
                    layout=layout,
                    cache=cache,
                    page_table=page_table,
                    rows=EXTEND_STEP,
                    distribution=(0, 0, num_seqs),
                    history=history,
                    seed=1,
                )
                cache = self._step(
                    layout=layout,
                    cache=cache,
                    page_table=page_table,
                    rows=MIXED_STEP,
                    distribution=MIXED_DISTRIBUTION,
                    history=history,
                    seed=2,
                )
                # Page is the leading axis in both layouts.
                spare_bytes = np.asarray(cache[np.asarray(spare)].astype(jnp.float32))
                self.assertFalse(spare_bytes.any(), "write landed outside the table")


if __name__ == "__main__":
    unittest.main()
