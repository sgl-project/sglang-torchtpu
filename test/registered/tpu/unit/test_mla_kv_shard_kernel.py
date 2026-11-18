# SPDX-License-Identifier: Apache-2.0
"""A latent cache sharded by page: the ranks' partial sparse-MLA steps, merged,
must match the unsharded step, and each rank's cache must hold its pages only.

One TPU plays every rank of a group: each attends the group's heads over its
own candidates from its own cache, as serving does after the query all-gather.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_sparse import kv_shard
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    sparse_mla_attention_fused,
    sparse_mla_attention_kv_shard,
)

PAGE, PPS, HEADS, KPOOL, TOPK = 128, 4, 16, 4, 64
NUM_PAGES = 1 + 2 * PPS
SM_SCALE = float(256**-0.5)
# Merging two bf16-rounded partials adds about one bf16 step (the measured gap
# between the sparse and dense kernels in test_mla_sparse_fused is 0.016).
MAX_ABS, MEAN_ABS = 0.05, 0.003


def _rows(rng, lens, cu, width):
    """Pooled-indexer rows: up to TOPK // KPOOL closed pools, then the open pool."""
    rows = np.full((cu[-1], width), -1, np.int32)
    for r, n in enumerate(lens):
        for i in range(n):
            pos, t = i, cu[r] + i
            closed = (pos + 1) // KPOOL
            pick = rng.choice(closed, size=min(closed, TOPK // KPOOL), replace=False)
            toks = [p * KPOOL + j for p in sorted(pick) for j in range(KPOOL)]
            toks += list(range(closed * KPOOL, pos + 1))
            rows[t, : len(toks)] = toks
    return rows


class TestMlaKvShardKernel(CustomTestCase):
    def test_merged_shards_match_unsharded(self):
        rng = np.random.default_rng(0)
        lens = [300, 213]
        cu = np.array([0, lens[0], sum(lens)], np.int32)
        tokens = int(cu[-1])
        # Page ids interleave owners irregularly, as the allocator's do.
        page_indices = jnp.asarray(rng.permutation(np.arange(1, NUM_PAGES)), jnp.int32)
        args = (
            jnp.asarray(rng.normal(size=(tokens, HEADS, 512)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(tokens, HEADS, 64)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(tokens, 512)).astype(ml_dtypes.float8_e4m3fn)),
            jnp.asarray(rng.normal(size=(tokens, 64)).astype(ml_dtypes.float8_e4m3fn)),
        )
        meta = (jnp.asarray(lens, jnp.int32), page_indices, jnp.asarray(cu))
        dist = jnp.asarray([0, 0, 2], jnp.int32)
        rows = jnp.asarray(_rows(rng, lens, cu, TOPK + KPOOL - 1))

        def caches(pages):
            return (
                jnp.zeros((pages, PAGE, 4, 128), jnp.uint8),
                jnp.zeros((pages, PAGE // 4, 4, 128), jnp.uint8),
            )

        want, full_nope, _ = jax.jit(
            lambda *a: sparse_mla_attention_fused(*a, sm_scale=SM_SCALE, k_scale=1.0)
        )(*args, *caches(NUM_PAGES), *meta, dist, rows)

        shards = 2
        outs, lses = [], []
        for role in range(shards):
            out, lse, nope, _ = jax.jit(
                lambda *a, role=role: sparse_mla_attention_kv_shard(
                    *a, sm_scale=SM_SCALE, k_scale=1.0, shards=shards, role=role,
                    kpool=KPOOL,
                )
            )(*args, *caches(kv_shard.mla_shard_pages(NUM_PAGES, shards)), *meta, dist, rows)  # fmt: skip
            outs.append(out.astype(jnp.float32))
            lses.append(lse)
            # This rank's cache holds exactly its own pages, at page id // shards.
            for g in range(1, NUM_PAGES):
                if g % shards == role:
                    np.testing.assert_array_equal(
                        np.asarray(nope[g // shards]), np.asarray(full_nope[g])
                    )
        got = kv_shard.merge_partials(jnp.stack(outs), jnp.stack(lses))
        diff = np.abs(np.asarray(got) - np.asarray(want, np.float32))
        self.assertLess(float(diff.max()), MAX_ABS)
        self.assertLess(float(diff.mean()), MEAN_ABS)


if __name__ == "__main__":
    unittest.main()
