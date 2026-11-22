# SPDX-License-Identifier: Apache-2.0
"""The pooled DSA indexer on a key cache split across ranks must pick the same pools.

Each rank holds the pools at its own offsets of every page; it scores all rows
against them, and the merge of every rank's local top-k scores keeps the global
winners it holds. The union over ranks must equal the unsplit selection, and a
rank's split cache must hold exactly its share of the unsplit cache's bytes.
StreamIndex and the SparseCore top-k need a TPU; the ranks run one by one here.
"""

import os
import sys
import unittest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"
# The serving setting; the split cache's small pages take the two-level DMA groups.
os.environ["SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_kpool_topk as kp

D, HEADS, KPOOL, PAGE, TOPK, SHARDS = 128, 32, 4, 128, 2048, 8
N_PICK = TOPK // KPOOL
POOLS_PER_PAGE = PAGE // KPOOL
OWN_POOLS = POOLS_PER_PAGE // SHARDS
RECORD = kp.pooled_record_bytes(D)
WEIGHT_SCALE = HEADS**-0.5 * D**-0.5


def _split(pooled: jax.Array, role: int) -> jax.Array:
    """Rank ``role``'s share of an unsplit pooled cache."""
    rows = pooled.shape[1] // SHARDS
    return pooled[:, role * rows : (role + 1) * rows]


def _page_table(rng, lens, pages_per_seq):
    """Pages in runs of random length (1..80), shuffled: both one-DMA groups and
    page-by-page fallbacks occur. Page 0 pads every row."""
    need = [-(-n // PAGE) for n in lens]
    runs, nxt = [], 1
    while nxt <= sum(need):
        n = int(rng.integers(1, 81))
        runs.append(list(range(nxt, nxt + n)))
        nxt += n
    order = [p for i in rng.permutation(len(runs)) for p in runs[i]]
    table = np.zeros((len(lens), pages_per_seq), np.int32)
    at = 0
    for s, n in enumerate(need):
        table[s, :n] = order[at : at + n]
        at += n
    return table, nxt


class TestKpoolIndexSplitCache(CustomTestCase):
    def _split_rows(self, q, w, pooled, table, *, cu, dist, kv, dqs):
        """Every rank's local (global pool -> score) candidates and its merged
        token rows."""
        locals_, cands = [], {}
        for role in range(SHARDS):
            picked, scores = kp._score_pools(
                q, w, _split(pooled, role), kv, table, cu, dist,
                weight_scale=WEIGHT_SCALE, topk=TOPK, kpool=KPOOL,
                decode_query_size=dqs, cp_size=SHARDS, cp_rank=role,
                interleave_size=OWN_POOLS * KPOOL, return_scores=True,
            )  # fmt: skip
            locals_.append((picked, scores))
            g = np.asarray(kp.cp_local_to_global(picked, role, SHARDS, OWN_POOLS))
            sc = np.asarray(scores).view(np.float32)
            ok = np.asarray(picked) >= 0
            for t, (gr, sr, okr) in enumerate(zip(g, sc, ok)):
                cands.setdefault(t, {}).update(zip(gr[okr].tolist(), sr[okr].tolist()))
        all_scores = jnp.concatenate([s for _, s in locals_], axis=0)
        rows = [
            np.asarray(
                kp.kpool_merge_rows(
                    all_scores, picked, kv, cu, dist, shards=SHARDS, role=role,
                    own_pools=OWN_POOLS, topk=TOPK, kpool=KPOOL,
                )
            )
            for role, (picked, _) in enumerate(locals_)
        ]  # fmt: skip
        return rows, cands

    def _check_batch(self, name, q, w, pooled, table, *, dqs, q_lens, kv_lens, dist):
        cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        dist = jnp.asarray(dist, jnp.int32)
        kv = jnp.asarray(kv_lens, jnp.int32)
        ref = kp._score_pools(
            q, w, pooled, kv, table, cu, dist,
            weight_scale=WEIGHT_SCALE, topk=TOPK, kpool=KPOOL, decode_query_size=dqs,
        )  # fmt: skip
        _, pos, valid = kp._token_rows(kv, cu, dist, q.shape[0])
        ref_rows = np.asarray(kp.expand_picked(ref, pos, valid, topk=TOPK, kpool=KPOOL))
        rows, cands = self._split_rows(
            q, w, pooled, table, cu=cu, dist=dist, kv=kv, dqs=dqs
        )
        identical = ties = other = 0
        for t in range(int(cu[-1])):
            want = {int(x) for x in ref_rows[t] if x >= 0}
            got = set().union(*({int(x) for x in r[t] if x >= 0} for r in rows))
            if got == want:
                identical += 1
                continue
            # A tie-order difference swaps only pools scored at the k-th score.
            scores = cands[t]
            kth = sorted(scores.values(), reverse=True)[N_PICK - 1]
            swapped = {x // KPOOL for x in got ^ want}
            if all(scores.get(p) == kth for p in swapped):
                ties += 1
            else:
                other += 1
        print(f"{name}: {identical} rows identical, {ties} tie-order, {other} other")
        self.assertEqual(other, 0, f"{name}: {other} rows differ beyond ties")
        return identical, ties

    def test_split_selection_matches_unsplit(self):
        rng = np.random.default_rng(0)
        # Contexts in tokens: one under N_PICK pools (all picked), the rest long.
        lens = [1500, 9001, 20003, 33000, 47111, 60000, 70002, 81920]
        pages_per_seq = 768
        table, num_pages = _page_table(rng, lens, pages_per_seq)
        keys = rng.normal(size=(num_pages * POOLS_PER_PAGE, D)).astype(np.float32)
        records = kp._pool_records(jnp.asarray(keys), RECORD)
        pooled = records.reshape(num_pages, PAGE // 16, 4, RECORD)
        flat_table = jnp.asarray(table.reshape(-1))
        # The prefill batch extends sequences 4 and 7 by 96 and 160 rows.
        pre_table = np.zeros_like(table)
        pre_table[:2] = table[[4, 7]]
        batches = (
            ("decode", 1, [1] * 8, lens, [8, 8, 8], flat_table, 128),
            ("verify", 4, [4] * 8, lens, [8, 8, 8], flat_table, 128),
            ("prefill", 1, [96, 160] + [0] * 6, [lens[4], lens[7]] + [0] * 6,
             [0, 0, 2], jnp.asarray(pre_table.reshape(-1)), 256),
        )  # fmt: skip
        for name, dqs, q_lens, kv, dist, tab, n_tok in batches:
            q = jnp.asarray(rng.normal(size=(n_tok, HEADS, D)), jnp.bfloat16)
            w = jnp.asarray(rng.normal(size=(n_tok, HEADS)), jnp.float32)
            identical, ties = self._check_batch(
                name, q, w, pooled, tab, dqs=dqs, q_lens=q_lens, kv_lens=kv, dist=dist
            )
            self.assertEqual(ties, 0, f"{name}: random keys gave exact score ties")

    def test_tied_scores_differ_only_in_tie_order(self):
        """Every pool equal: ranks break ties by rank, the unsplit top-k does not."""
        rng = np.random.default_rng(2)
        lens = [40960, 65536]
        table, num_pages = _page_table(rng, lens, 768)
        key = rng.normal(size=(1, D)).astype(np.float32)
        records = kp._pool_records(jnp.asarray(key), RECORD)
        pooled = jnp.broadcast_to(
            records.reshape(1, 1, 1, RECORD), (num_pages, PAGE // 16, 4, RECORD)
        )
        q = jnp.asarray(rng.normal(size=(128, HEADS, D)), jnp.bfloat16)
        w = jnp.asarray(rng.normal(size=(128, HEADS)), jnp.float32)
        self._check_batch(
            "ties", q, w, pooled, jnp.asarray(table.reshape(-1)), dqs=4,
            q_lens=[4, 4], kv_lens=lens, dist=[2, 2, 2],
        )  # fmt: skip

    def test_split_writer_stores_own_share(self):
        rng = np.random.default_rng(1)
        pages_per_seq, num_pages = 8, 2 * 8 + 1
        ape = jnp.asarray(rng.normal(size=(KPOOL, D)).astype(np.float32))
        table = jnp.asarray(
            [1 + r * pages_per_seq + i for r in range(2) for i in range(pages_per_seq)],
            jnp.int32,
        )
        raw = jnp.zeros((num_pages, PAGE, 2 * D), jnp.bfloat16)
        full = jnp.zeros((num_pages, PAGE // 16, 4, RECORD), jnp.uint8)
        split = [_split(full, r) for r in range(SHARDS)]
        store = jax.jit(
            kp.store_pooled_keys,
            static_argnames=("kpool", "max_query_length", "shards", "role"),
        )
        lens = [0, 0]
        # A long prefill (fused long path), decodes, and verify-width steps.
        for new in [(301, 214), (1, 1), (8, 8), (1, 1), (4, 4), (300, 2)]:
            n_tok = sum(new)
            pad = 1024 if n_tok > 32 else 32
            key = jnp.asarray(rng.normal(size=(pad, D)), jnp.bfloat16)
            gate = jnp.asarray(rng.normal(size=(pad, D)), jnp.bfloat16)
            lens = [lens[0] + new[0], lens[1] + new[1]]
            args = (
                key, gate, ape, raw, None,
                jnp.asarray(lens, jnp.int32), table,
                jnp.asarray([0, new[0], n_tok], jnp.int32),
                jnp.asarray([0, 0, 2], jnp.int32),
            )  # fmt: skip
            new_raw, full = store(*args[:4], full, *args[5:], kpool=KPOOL)
            for r in range(SHARDS):
                _, split[r] = store(
                    *args[:4], split[r], *args[5:], kpool=KPOOL, shards=SHARDS, role=r
                )
            raw = new_raw
            for r in range(SHARDS):
                np.testing.assert_array_equal(
                    np.asarray(split[r]), np.asarray(_split(full, r)), f"rank {r} {new}"
                )
        self.assertGreater(int(jnp.sum(full != 0)), 0)


if __name__ == "__main__":
    unittest.main()
