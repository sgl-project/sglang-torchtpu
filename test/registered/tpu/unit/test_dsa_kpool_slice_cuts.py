# SPDX-License-Identifier: Apache-2.0
"""The prefill exchange of the split pooled indexer must keep what the merge keeps.

Every rank scores all rows against its own pools (local top-k); the owner of a
row slice gets every rank's local scores of its rows, finds each row's global
top-k cut (kpool_slice_cuts) and sends each rank its cut, and the rank keeps its
candidates above it (kpool_cut_pools). Each rank must keep exactly the pools the
all-gather merge keeps (global top-k over rank-major candidates, ties to the
lower rank, then the lower slot), in the same order once compacted, and the
union must be the unsplit top-k. Runs on CPU with the SparseCore top-k replaced
by a reference of its contract; test_dsa_kpool_index_shard.py runs the kernels.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_kpool_topk as kp
from sglang_tpu.attention.pallas.mla_sparse import kv_shard
from sglang_tpu.dsv4 import sparsecore_topk as sc
from sglang_tpu.dsv4.streamindex_topk.metadata import (
    cp_global_to_local,
    cp_owner_rank,
)

KPOOL, PAGE, SHARDS = 4, 128, 8
OWN_POOLS = PAGE // KPOOL // SHARDS


def _key(bits):
    """The SparseCore top-k's order of f32 score bits: -0.0 below +0.0."""
    return bits ^ ((bits >> 31) & sc.MONO_MASK)


def _ref_topk(scores, k, row_lengths=None, *, return_scores=False, **_):
    """The SparseCore top-k at p = 1: the exact top k of each row by score key,
    lowest column first on ties, -1 / -inf suffix, winners in column order."""
    key = _key(scores)
    vals, idx = jax.lax.top_k(key, k)
    idx = jnp.sort(jnp.where(vals > sc.KEY_NEG_INF, idx, scores.shape[1]), axis=1)
    live = idx < scores.shape[1]
    idx = jnp.where(live, idx, -1)
    bits = jnp.take_along_axis(scores, jnp.maximum(idx, 0), axis=1)
    bits = jnp.where(live, bits, sc.BITS_NEG_INF)
    return (idx, bits) if return_scores else idx


def _keys(bits: np.ndarray) -> np.ndarray:
    return _key(np.asarray(bits, np.int32)).astype(np.int64)


def _batch(rng, reqs, tied):
    """Rows of requests (context, new tokens): positions, every row's score of
    every pool it sees (f32 bits, -inf past its closed pools)."""
    pos = np.concatenate([np.arange(c, c + n) for c, n in reqs]).astype(np.int32)
    seen = (pos + 1) // KPOOL
    width = -(-int(seen.max()) // (PAGE // KPOOL)) * (PAGE // KPOOL)
    if tied:  # few distinct values: the cut falls on 1.0 or on +0.0 over -0.0
        values = np.array([2.0, 1.0, 0.0, -0.0, -1.0], np.float32)
        p = [0.003, 0.01, 0.3, 0.3, 0.387]
        scores = rng.choice(values, size=(len(pos), width), p=p)
    else:
        scores = rng.normal(size=(len(pos), width)).astype(np.float32)
    scores[np.arange(width)[None, :] >= seen[:, None]] = -np.inf
    return pos, scores.view(np.int32)


def _local_top(rng, bits, n_pick, role):
    """A rank's local candidates: its pools' top n_pick (local ids, score bits),
    valid ones first in a shuffled order, then -1 / -inf."""
    num_rows, width = bits.shape
    pools = np.arange(width)
    mine = pools[np.asarray(cp_owner_rank(pools, SHARDS, OWN_POOLS)) == role]
    local = np.asarray(cp_global_to_local(mine, SHARDS, OWN_POOLS))
    picked = np.full((num_rows, n_pick), -1, np.int32)
    scores = np.full((num_rows, n_pick), sc.BITS_NEG_INF, np.int32)
    keys = _keys(bits[:, mine])
    for t in range(num_rows):
        order = np.lexsort((local, -keys[t]))[:n_pick]
        order = order[keys[t, order] > sc.KEY_NEG_INF]
        order = rng.permutation(order)
        picked[t, : len(order)] = local[order]
        scores[t, : len(order)] = bits[t, mine[order]]
    return picked, scores


def _merge_ref(picked, scores, n_pick):
    """Per rank, the global pools the all-gather merge keeps, in slot order: the
    top n_pick of each row over rank-major columns, lowest column on ties."""
    num_rows = picked[0].shape[0]
    keys = np.concatenate([_keys(s) for s in scores], axis=1)  # [T, S * n_pick]
    kept = [np.full((num_rows, n_pick), -1, np.int32) for _ in range(SHARDS)]
    for t in range(num_rows):
        cols = np.lexsort((np.arange(keys.shape[1]), -keys[t]))[:n_pick]
        cols = np.sort(cols[keys[t, cols] > sc.KEY_NEG_INF])
        for s in range(SHARDS):
            slots = cols[cols // n_pick == s] % n_pick
            g = kp.cp_local_to_global(picked[s][t, slots], s, SHARDS, OWN_POOLS)
            kept[s][t, : len(slots)] = np.asarray(g)
    return kept


def _exchange(picked, scores, n_pick):
    """The prefill exchange, ranks one by one: each owner's cuts of its row
    slice, then each rank's kept pools."""
    num_rows = picked[0].shape[0]
    rows = -(-(-(-num_rows // SHARDS)) // 8) * 8
    pad = SHARDS * rows - num_rows
    padded = [
        np.pad(s, ((0, pad), (0, 0)), constant_values=sc.BITS_NEG_INF) for s in scores
    ]
    cuts = [
        np.asarray(
            kp.kpool_slice_cuts(
                jnp.asarray(np.stack([s[r * rows : (r + 1) * rows] for s in padded]))
            )
        )
        for r in range(SHARDS)
    ]  # cuts[r]: [SHARDS, rows, 2], owner r's cut for each rank
    return [
        np.asarray(
            kp.kpool_cut_pools(
                jnp.asarray(picked[s]),
                jnp.asarray(scores[s]),
                jnp.asarray(np.concatenate([c[s] for c in cuts])),
                shards=SHARDS,
                role=s,
                own_pools=OWN_POOLS,
            )
        )
        for s in range(SHARDS)
    ]


def _compact(rows):
    return [r[r >= 0].tolist() for r in rows]


class TestKpoolSliceCuts(CustomTestCase):
    def setUp(self):
        patch = mock.patch.object(kp, "sparsecore_topk", _ref_topk)
        patch.start()
        self.addCleanup(patch.stop)

    def _check(self, seed, reqs, n_pick, *, tied):
        rng = np.random.default_rng(seed)
        pos, bits = _batch(rng, reqs, tied)
        locals_ = [_local_top(rng, bits, n_pick, s) for s in range(SHARDS)]
        picked, scores = [p for p, _ in locals_], [s for _, s in locals_]
        got = _exchange(picked, scores, n_pick)
        want = _merge_ref(picked, scores, n_pick)
        for s in range(SHARDS):
            self.assertEqual(_compact(got[s]), _compact(want[s]), f"rank {s}")
        seen = (pos + 1) // KPOOL
        total = sum((g >= 0).sum(axis=1) for g in got)
        np.testing.assert_array_equal(total, np.minimum(seen, n_pick))
        if not tied:
            keys = _keys(bits)
            for t in range(len(pos)):
                unsplit = np.argsort(-keys[t], kind="stable")[: min(seen[t], n_pick)]
                union = {int(x) for g in got for x in g[t] if x >= 0}
                self.assertEqual(union, set(unsplit.tolist()), f"row {t}")
        return pos, picked, scores, got

    def test_random_scores_match_merge_and_unsplit(self):
        # Rows before any closed pool, under n_pick pools, and far past it.
        reqs = [(0, 40), (900, 72), (20000, 136)]
        self._check(0, reqs, 64, tied=False)

    def test_chunk_shape(self):
        # 512 pools per row, a slice of 32 rows per rank, padding rows.
        self._check(1, [(16000, 200), (3000, 50)], 512, tied=False)

    def test_tied_scores_split_like_the_merge(self):
        # Mostly ties at the cut, -0.0 below +0.0 as the SparseCore orders them.
        self._check(2, [(0, 40), (900, 72), (20000, 136)], 64, tied=True)

    def test_token_rows_attend_the_merge_candidates(self):
        """Expanded and packed for attention (owned_rows), each rank's rows equal
        the merge's, so its attention output is the merge path's."""
        reqs = [(12000, 64), (700, 64)]
        n_pick, topk = 64, 256
        pos, picked, scores, got = self._check(3, reqs, n_pick, tied=True)
        want = _merge_ref(picked, scores, n_pick)
        q_lens = [n for _, n in reqs]
        cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        kv = jnp.asarray([c + n for c, n in reqs], jnp.int32)
        dist = jnp.asarray([0, 0, len(reqs)], jnp.int32)
        _, row_pos, valid = kp._token_rows(kv, cu, dist, len(pos))
        np.testing.assert_array_equal(np.asarray(row_pos), pos)
        for s in range(SHARDS):
            packed = []
            for pools in (got[s], want[s]):
                rows = kp.expand_picked(
                    jnp.asarray(pools), row_pos, valid, topk=topk, kpool=KPOOL
                )
                width = -(-rows.shape[1] // KPOOL) * KPOOL
                packed.append(
                    np.asarray(
                        kv_shard.owned_rows(
                            rows, page_size=PAGE, shards=SHARDS, role=s,
                            group=KPOOL, out_width=width,
                        )
                    )
                )  # fmt: skip
            np.testing.assert_array_equal(packed[0], packed[1], f"rank {s}")


if __name__ == "__main__":
    unittest.main()
