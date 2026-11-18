# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA KV sharded by page (host): the shift-network compaction is a
stable partition, merged partials equal one unsharded attention (the JAX merge
and the serving merge, which hands each rank its own heads), and every page id
has exactly one owner and physical page."""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_sparse import kv_shard
from sglang_tpu.attention_pallas_mla_kv_shard import merge_kv_shards

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


def _attend(q, keys, rows):
    """Normalized output and natural-log sum over each row's -1-padded keys."""
    tokens, heads, dim = q.shape
    out = np.zeros((tokens, heads, dim), np.float32)
    lse = np.full((tokens, heads), -np.inf, np.float32)
    for t, row in enumerate(rows):
        live = [p for p in row if p >= 0]
        if not live:
            continue
        k = keys[t][live]
        s = q[t] @ k.T
        m = s.max(-1, keepdims=True)
        e = np.exp(s - m)
        out[t] = (e @ k) / e.sum(-1, keepdims=True)
        lse[t] = (m[:, 0] + np.log(e.sum(-1))).astype(np.float32)
    return out, lse


class _FakeGroup:
    """One rank of a merge group: all_gather stacks every rank's lse. The first
    pass over the ranks records each one's reduce-scatter input, the second
    returns this rank's chunk summed over them."""

    def __init__(self, rank: int, lses: list, fed: list):
        self.world_size, self.rank_in_group = len(lses), rank
        self.lses, self.fed = lses, fed

    def all_gather(self, x, dim):
        return torch.cat([s.unsqueeze(0) for s in self.lses], dim=dim)

    def reduce_scatter_tensor(self, out, inp):
        if len(self.fed) < self.world_size:
            self.fed.append(inp)
        else:
            out.copy_(sum(f[self.rank_in_group] for f in self.fed).unsqueeze(0))


class TestMlaKvShard(CustomTestCase):
    def test_compact_left_is_a_stable_partition(self):
        # Shifts applied bit by bit must never land two kept entries on one slot.
        rng = np.random.default_rng(1)
        for groups in (1, 2, 7, 64, 513, 700):
            for p in (0.0, 0.1, 0.5, 0.9, 1.0):
                keep = rng.random((3, groups)) < p
                vals = rng.integers(0, 1 << 20, (3, groups, 8)).astype(np.int32)
                vals = np.where(keep[..., None], vals, -1)
                got = np.asarray(
                    kv_shard.compact_left(jnp.asarray(vals), jnp.asarray(keep))
                )
                for t in range(3):
                    want = np.concatenate(
                        [vals[t][keep[t]], np.full((groups - keep[t].sum(), 8), -1)]
                    )
                    np.testing.assert_array_equal(got[t], want, f"{groups} {p}")

    def test_merged_partials_equal_unsharded_attention(self):
        rng = np.random.default_rng(0)
        tokens, heads, dim, width, n_keys = 9, 3, 8, 12, 40
        q = rng.normal(size=(tokens, heads, dim)).astype(np.float32)
        keys = rng.normal(size=(tokens, n_keys, dim)).astype(np.float32)
        rows = np.full((tokens, width), -1, np.int32)
        for t in range(tokens - 1):  # the last row is padding: no candidate
            n = rng.integers(0, width + 1)
            rows[t, :n] = rng.choice(n_keys, size=n, replace=False)
        owner = rng.integers(0, 2, n_keys)  # each key lives on one rank
        want, _ = _attend(q, keys, rows)
        outs, lses = [], []
        for role in range(2):
            mine = np.where((rows >= 0) & (owner[rows] == role), rows, -1)
            o, s = _attend(q, keys, mine)
            outs.append(o)
            lses.append(s)
        got = np.asarray(
            kv_shard.merge_partials(
                jnp.asarray(np.stack(outs)), jnp.asarray(np.stack(lses))
            )
        )
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_serving_merge_hands_each_rank_its_heads(self):
        # Every rank attends all group heads; rank r keeps heads [r * h, (r + 1) * h).
        rng = np.random.default_rng(2)
        shards, heads, tokens, dim, width, n_keys = 4, 2, 9, 8, 12, 40
        q = rng.normal(size=(tokens, shards * heads, dim)).astype(np.float32)
        keys = rng.normal(size=(tokens, n_keys, dim)).astype(np.float32)
        rows = np.full((tokens, width), -1, np.int32)
        for t in range(tokens - 1):  # the last row is padding: no candidate
            n = rng.integers(0, width + 1)
            rows[t, :n] = rng.choice(n_keys, size=n, replace=False)
        owner = rng.integers(0, shards, n_keys)
        want, _ = _attend(q, keys, rows)
        outs, lses = [], []
        for role in range(shards):
            mine = np.where((rows >= 0) & (owner[rows] == role), rows, -1)
            o, s = _attend(q, keys, mine)
            o[~np.isfinite(s)] = np.nan  # no candidate: the lse must mask it
            outs.append(torch.from_numpy(o))
            lses.append(torch.from_numpy(s))
        fed = []
        for _ in range(2):  # record every rank's partial, then reduce
            got = [
                merge_kv_shards(outs[r], lses[r], _FakeGroup(r, lses, fed))
                for r in range(shards)
            ]
        np.testing.assert_allclose(
            torch.cat(got, dim=1).numpy(), want, rtol=1e-5, atol=1e-6
        )

    def test_mla_page_indices_route_every_page_to_one_owner(self):
        pages = jnp.arange(0, 37, dtype=jnp.int32)
        hits = np.zeros(37, np.int32)
        seen = set()
        for role in range(4):
            phys = np.asarray(
                kv_shard.mla_page_indices(pages, shards=4, role=role, oob_page=10**6)
            )
            mine = phys < 10**6
            hits += mine
            self.assertTrue((phys[mine] < kv_shard.mla_shard_pages(37, 4)).all())
            seen |= {(role, int(p)) for p in phys[mine]}
        np.testing.assert_array_equal(hits, 1)
        self.assertEqual(len(seen), 37)  # no two page ids share a physical page


if __name__ == "__main__":
    unittest.main()
