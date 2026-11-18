# SPDX-License-Identifier: Apache-2.0
"""A latent cache split inside every page: the ranks' partial sparse-MLA steps,
merged, must match the unsharded step, and each rank's cache must hold exactly
its offsets of every page.

One TPU plays every rank of a group: each attends the group's heads over its
own candidates from its own cache, as serving does after the query all-gather.
A prefill step is followed by a decode step on the caches it wrote.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

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
# Merging bf16-rounded partials adds about one bf16 step (the measured gap
# between the sparse and dense kernels in test_mla_sparse_fused is 0.016).
MAX_ABS, MEAN_ABS = 0.05, 0.003


def _rows(rng, starts, lens, width):
    """Pooled-indexer rows for positions starts[r] .. starts[r] + lens[r] - 1:
    up to TOPK // KPOOL closed pools, then the open pool."""
    rows = np.full((sum(lens), width), -1, np.int32)
    t = 0
    for start, n in zip(starts, lens):
        for pos in range(start, start + n):
            closed = (pos + 1) // KPOOL
            pick = rng.choice(closed, size=min(closed, TOPK // KPOOL), replace=False)
            toks = [p * KPOOL + j for p in sorted(pick) for j in range(KPOOL)]
            toks += list(range(closed * KPOOL, pos + 1))
            rows[t, : len(toks)] = toks
            t += 1
    return rows


def _step_inputs(rng, tokens):
    return (
        jnp.asarray(rng.normal(size=(tokens, HEADS, 512)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(tokens, HEADS, 64)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(tokens, 512)).astype(ml_dtypes.float8_e4m3fn)),
        jnp.asarray(rng.normal(size=(tokens, 64)).astype(ml_dtypes.float8_e4m3fn)),
    )


def _caches(page):
    return (
        jnp.zeros((NUM_PAGES, page, 4, 128), jnp.uint8),
        jnp.zeros((NUM_PAGES, page // 4, 4, 128), jnp.uint8),
    )


class TestMlaKvShardKernel(CustomTestCase):
    def _check(self, shards):
        rng = np.random.default_rng(shards)
        sub = PAGE // shards
        # Page ids in an irregular order, as the allocator's are.
        page_indices = jnp.asarray(rng.permutation(np.arange(1, NUM_PAGES)), jnp.int32)
        prefill = [300, 213]
        steps = [  # (kv_lens, q_lens, distribution, first position)
            (prefill, prefill, [0, 0, 2], [0, 0]),
            ([n + 1 for n in prefill], [1, 1], [2, 2, 2], prefill),
        ]
        fused = jax.jit(
            lambda *a: sparse_mla_attention_fused(*a, sm_scale=SM_SCALE, k_scale=1.0)
        )
        shard_ops = [
            jax.jit(
                lambda *a, role=role: sparse_mla_attention_kv_shard(
                    *a, sm_scale=SM_SCALE, k_scale=1.0, shards=shards, role=role,
                    kpool=KPOOL,
                )
            )
            for role in range(shards)
        ]  # fmt: skip
        full = _caches(PAGE)
        own = [_caches(sub) for _ in range(shards)]
        for kv_lens, q_lens, dist, starts in steps:
            cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
            args = _step_inputs(rng, int(cu[-1]))
            meta = (jnp.asarray(kv_lens, jnp.int32), page_indices, jnp.asarray(cu))
            rows = jnp.asarray(_rows(rng, starts, q_lens, TOPK + KPOOL - 1))
            dist = jnp.asarray(dist, jnp.int32)
            want, *full = fused(*args, *full, *meta, dist, rows)
            outs, lses = [], []
            for role in range(shards):
                out, lse, *own[role] = shard_ops[role](
                    *args, *own[role], *meta, dist, rows
                )
                outs.append(out.astype(jnp.float32))
                lses.append(lse)
                # This rank's cache holds exactly its offsets of every page.
                np.testing.assert_array_equal(
                    np.asarray(own[role][0]),
                    np.asarray(full[0])[:, role * sub : (role + 1) * sub],
                )
            got = kv_shard.merge_partials(jnp.stack(outs), jnp.stack(lses))
            diff = np.abs(np.asarray(got) - np.asarray(want, np.float32))
            self.assertLess(float(diff.max()), MAX_ABS, f"q_lens {q_lens}")
            self.assertLess(float(diff.mean()), MEAN_ABS, f"q_lens {q_lens}")

    def test_merged_shards_match_unsharded(self):
        for shards in (2, 8):
            with self.subTest(shards=shards):
                self._check(shards)


if __name__ == "__main__":
    unittest.main()
