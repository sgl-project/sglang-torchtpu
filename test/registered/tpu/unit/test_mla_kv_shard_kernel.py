# SPDX-License-Identifier: Apache-2.0
"""A latent cache split inside every page: the ranks' partial sparse-MLA steps,
merged, must match the unsharded step, and each rank's cache must hold exactly
its offsets of every page.

One TPU plays every rank of a group: each attends the group's heads over its
own candidates from its own cache, as serving does after the query all-gather.
A prefill step is followed by a decode step on the caches it wrote.

The token split's owned-row prefill (owned_width; groups of 2, 4 and 8) runs a 1024-token chunk at a
20k context and a 256-token chunk whose picks all sit in one rank's offsets
(that rank runs every further slice), with partial pages and padding tokens,
against the unsharded step and the whole-row split step.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

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
    QBLOCK_GATHER_CHUNK_TOKENS,
    _store_step,
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


def _step_inputs(rng, tokens, zero_rope=False):
    rope = 0.0 if zero_rope else 1.0
    return (
        jnp.asarray(rng.normal(size=(tokens, HEADS, 512)), jnp.bfloat16),
        jnp.asarray(rope * rng.normal(size=(tokens, HEADS, 64)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(tokens, 512)).astype(ml_dtypes.float8_e4m3fn)),
        jnp.asarray(
            (rope * rng.normal(size=(tokens, 64))).astype(ml_dtypes.float8_e4m3fn)
        ),
    )


def _caches(page, num_pages=NUM_PAGES):
    return (
        jnp.zeros((num_pages, page, 4, 128), jnp.uint8),
        jnp.zeros((num_pages, page // 4, 4, 128), jnp.uint8),
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
        for shards in (2, 4, 8):
            with self.subTest(shards=shards):
                self._check(shards)

    def _check_owned(self, zero_rope, shards):
        topk, width, pps = 512, 640, 168
        sub = PAGE // shards
        rng = np.random.default_rng(11 + zero_rope)
        num_pages = 2 * pps + 1
        page_indices = jnp.asarray(rng.permutation(np.arange(1, num_pages)), jnp.int32)
        prefix, new = [20000, 4096], [1024, 256]
        kv_lens = jnp.asarray([p + n for p, n in zip(prefix, new)], jnp.int32)
        cu = jnp.asarray([0, new[0], sum(new)], jnp.int32)
        dist = jnp.asarray([0, 2, 2], jnp.int32)
        tokens = sum(new) + 256  # 256 padding tokens
        caches = [_caches(PAGE, num_pages)] + [
            _caches(sub, num_pages) for _ in range(shards)
        ]
        # The prefixes, stored by every cache's writer.
        pre_cu = jnp.asarray([0, prefix[0], sum(prefix)], jnp.int32)
        kv_c = jnp.asarray(
            rng.normal(size=(sum(prefix), 512)).astype(ml_dtypes.float8_e4m3fn)
        )
        k_pe = jnp.zeros((sum(prefix), 64), jnp.float8_e4m3fn)
        if not zero_rope:
            k_pe = jnp.asarray(
                rng.normal(size=(sum(prefix), 64)).astype(ml_dtypes.float8_e4m3fn)
            )
        for i, cache in enumerate(caches):
            shard = None if i == 0 else (shards, i - 1)
            caches[i] = _store_step(
                *cache, kv_c, k_pe, jnp.asarray(prefix, jnp.int32), page_indices,
                pre_cu, dist, write_rope=not zero_rope, kv_shards=shard,
            )  # fmt: skip
        # Seq 0 picks pools anywhere; seq 1 only pools at rank 0's offsets.
        rows = np.full((tokens, width), -1, np.int32)
        for t in range(sum(new)):
            seq = int(t >= new[0])
            pos = prefix[seq] + t - seq * new[0]
            closed = (pos + 1) // KPOOL
            pools = np.arange(closed)
            if seq:
                pools = pools[(pools * KPOOL % PAGE) // sub == 0]
            pick = rng.choice(pools, size=min(len(pools), topk // KPOOL), replace=False)
            toks = [p * KPOOL + j for p in pick for j in range(KPOOL)]
            toks += list(range(closed * KPOOL, pos + 1))
            rows[t, : len(toks)] = toks
        rows = jnp.asarray(rows)
        args = _step_inputs(rng, tokens, zero_rope)
        meta = (kv_lens, page_indices, cu, dist, rows)
        want, *_ = jax.jit(
            lambda *a: sparse_mla_attention_fused(
                *a, sm_scale=SM_SCALE, k_scale=1.0, zero_rope=zero_rope,
                gather_chunk_tokens=QBLOCK_GATHER_CHUNK_TOKENS,
            )
        )(*args, *caches[0], *meta)  # fmt: skip
        want = np.asarray(want, np.float32)[: sum(new)]
        merged = {}
        for owned_width in (0, 64, 128):
            outs, lses = [], []
            for role in range(shards):
                out, lse, *_ = jax.jit(
                    lambda *a, role=role: sparse_mla_attention_kv_shard(
                        *a, sm_scale=SM_SCALE, k_scale=1.0, shards=shards,
                        role=role, kpool=KPOOL, zero_rope=zero_rope,
                        gather_chunk_tokens=QBLOCK_GATHER_CHUNK_TOKENS,
                        owned_width=owned_width,
                    )
                )(*args, *caches[role + 1], *meta)  # fmt: skip
                outs.append(out.astype(jnp.float32))
                lses.append(lse)
            got = kv_shard.merge_partials(jnp.stack(outs), jnp.stack(lses))
            merged[owned_width] = np.asarray(got)[: sum(new)]
        for owned_width, got in merged.items():
            for name, ref in (("unsharded", want), ("whole-row split", merged[0])):
                diff = np.abs(got - ref)
                msg = f"owned_width {owned_width} vs {name}"
                self.assertLess(float(diff.max()), MAX_ABS, msg)
                self.assertLess(float(diff.mean()), MEAN_ABS, msg)

    def test_owned_rows_match_whole_rows(self):
        for shards in (2, 4, 8):
            for zero_rope in (False, True):
                with self.subTest(shards=shards, zero_rope=zero_rope):
                    self._check_owned(zero_rope, shards)

    def test_owned_rows_kernel_equals_xla(self):
        """owned_rows_kernel (one VMEM pass) returns kv_shard.owned_rows' rows exactly."""
        from sglang_tpu.attention.pallas.mla_sparse.owned_rows_kernel import (
            owned_rows_kernel,
        )

        rng = np.random.default_rng(3)
        for tokens, width, shards, owned in ((1024, 2048, 2, 1408), (256, 2048, 8, 512)):
            pools = rng.integers(0, 20000, size=(tokens, width // 4))
            rows = (pools[..., None] * 4 + np.arange(4)).reshape(tokens, width)
            valid = rng.integers(0, width // 4 + 1, size=tokens) * 4
            rows = np.where(np.arange(width) < valid[:, None], rows, -1)
            rows[::5, -3:] = -1  # a partial last pool
            rows = jnp.asarray(rows, jnp.int32)
            out_width = -(-width // owned) * owned
            for role in range(shards):
                with self.subTest(shards=shards, role=role):
                    kw = dict(page_size=128, shards=shards, role=role, group=4, out_width=out_width)
                    ref = jax.jit(lambda r: kv_shard.owned_rows(r, **kw))(rows)
                    got = owned_rows_kernel(rows, **kw)
                    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


if __name__ == "__main__":
    unittest.main()
