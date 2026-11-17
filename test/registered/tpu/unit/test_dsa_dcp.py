# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA (DSA) under decode context parallelism, against the unsharded ops.

Simulates a DCP group of ``_DCP`` ranks on one chip: each rank gets its own
index-key, NoPE and RoPE shards and runs the DCP ops with its rank. Checks:

* every key and latent lands, byte for byte, in its owner's shard at the slot
  the unsharded cache holds it (page-striped, one page_size run per rank);
* the union of the ranks' merged top-k winners is the unsharded indexer's
  top-k, row by row;
* the ranks' partial attention outputs, merged by log-sum-exp, match the
  unsharded sparse attention.

Usage:
    python test/registered/tpu/unit/test_dsa_dcp.py
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: select the
# physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.dsa_index_topk import (
    dsa_indexer_step,
    dsa_indexer_step_dcp,
    index_key_cache_shape,
    merge_dcp_topk,
)
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    NOPE_DIM,
    ROPE_DIM,
    sparse_mla_attention_fused,
    sparse_mla_attention_fused_dcp,
)
from sglang_tpu.dsv4.streamindex_topk.metadata import cp_local_to_global

_DCP = 4
_PAGE_SIZE = 128
_TOPK = 2048
_INDEX_HEADS = 32
_INDEX_DIM = 128
_WEIGHT_SCALE = float(_INDEX_DIM**-0.5 * _INDEX_HEADS**-0.5)
_HEADS = 16  # the DCP group's query heads
# K3.1: (qk_nope_head_dim 128 + qk_rope_head_dim 64) ** -0.5.
_SM_SCALE = float((128 + ROPE_DIM) ** -0.5)
_FP8 = ml_dtypes.float8_e4m3fn
# Sequences longer than top-k, so the indexer selects, plus one shorter.
_SEQ_LENS = (2900, 2304, 700)
_TOKENS = 8192  # padded prefill bucket
_PAGES_PER_RANK = 24
# Merging partials reorders fp32 sums; see test_mla_sparse_fused.py for the
# kernel-vs-kernel scale of a bf16 step.
_MAX_ABS_DIFF = 0.05
_MEAN_ABS_DIFF = 0.003


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _Layout:
    """Page tables for the unsharded cache and for the DCP shards.

    The unsharded table maps each page_size run of a sequence to a page; the
    DCP table maps each page_size * dcp run (a widened page) to one page id,
    the same on every rank, as TpuPallasRpaV3AttnBackend builds it.
    """

    def __init__(self, seq_lens, rng):
        num_seqs = len(seq_lens)
        # Headroom for a second chunk (TestDsaDcpDensePrefill).
        max_len = max(seq_lens) + _PAGE_SIZE * _DCP
        self.ref_pages_per_seq = _cdiv(max_len, _PAGE_SIZE)
        self.dcp_pages_per_seq = _cdiv(max_len, _PAGE_SIZE * _DCP)
        ref_ids = rng.permutation(num_seqs * self.ref_pages_per_seq)
        dcp_ids = rng.permutation(_PAGES_PER_RANK)[: num_seqs * self.dcp_pages_per_seq]
        self.ref_pages = ref_ids.astype(np.int32)
        self.dcp_pages = dcp_ids.astype(np.int32)
        self.ref_num_pages = len(ref_ids)

    def ref_slot(self, seq: int, pos: int) -> tuple[int, int]:
        page = self.ref_pages[seq * self.ref_pages_per_seq + pos // _PAGE_SIZE]
        return int(page), pos % _PAGE_SIZE

    def dcp_slot(self, seq: int, pos: int) -> tuple[int, int, int]:
        widened = _PAGE_SIZE * _DCP
        owner = (pos % widened) // _PAGE_SIZE
        page = self.dcp_pages[seq * self.dcp_pages_per_seq + pos // widened]
        return owner, int(page), pos % _PAGE_SIZE


def _prefill_metadata(seq_lens):
    num_seqs = len(seq_lens)
    cu = np.zeros(num_seqs + 1, np.int32)
    cu[1:] = np.cumsum(seq_lens)
    return (
        jnp.asarray(np.asarray(seq_lens, np.int32)),
        jnp.asarray(cu),
        # MLA extend: everything in the mixed range.
        jnp.asarray(np.array([0, 0, num_seqs], np.int32)),
    )


def _random_inputs(rng, num_tokens):
    def f(*shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.standard_normal(shape, np.float32), dtype)

    return dict(
        query=f(num_tokens, _INDEX_HEADS, _INDEX_DIM),
        key=f(num_tokens, _INDEX_DIM),
        head_weights=f(num_tokens, _INDEX_HEADS),
        ql_nope=f(num_tokens, _HEADS, NOPE_DIM),
        q_pe=f(num_tokens, _HEADS, ROPE_DIM),
        new_kv_c=jnp.asarray(
            rng.standard_normal((num_tokens, NOPE_DIM), np.float32).astype(_FP8)
        ),
        new_k_pe=jnp.asarray(
            rng.standard_normal((num_tokens, ROPE_DIM), np.float32).astype(_FP8)
        ),
    )


def _empty_caches(num_pages):
    return (
        jnp.zeros(index_key_cache_shape(num_pages, _PAGE_SIZE, _INDEX_DIM), jnp.uint8),
        jnp.zeros((num_pages, _PAGE_SIZE, 4, 128), jnp.uint8),
        jnp.zeros((num_pages, _PAGE_SIZE // 4, 4, 128), jnp.uint8),
    )


def _token_rows(cache) -> np.ndarray:
    """``[num_pages, page_size, bytes]``: one row per token slot."""
    cache = np.asarray(cache)
    return cache.reshape(cache.shape[0], _PAGE_SIZE, -1)


class TestDsaDcp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.layout = _Layout(_SEQ_LENS, rng)
        cls.inputs = _random_inputs(rng, _TOKENS)
        seq_lens, cu_q_lens, distribution = _prefill_metadata(_SEQ_LENS)
        cls.meta = (seq_lens, cu_q_lens, distribution)
        cls._run_reference()
        cls._run_dcp()

    @classmethod
    def _run_reference(cls):
        x, (seq_lens, cu, dist) = cls.inputs, cls.meta
        pages = jnp.asarray(cls.layout.ref_pages)
        cache_k, nope, rope = _empty_caches(cls.layout.ref_num_pages)
        cls.ref_topk, cls.ref_cache_k = dsa_indexer_step(
            x["query"], x["key"], x["head_weights"], cache_k,
            seq_lens, pages, cu, dist,
            weight_scale=_WEIGHT_SCALE, topk=_TOPK, segments="mixed",
        )  # fmt: skip
        cls.ref_out, cls.ref_nope, cls.ref_rope = sparse_mla_attention_fused(
            x["ql_nope"], x["q_pe"], x["new_kv_c"], x["new_k_pe"], nope, rope,
            seq_lens, pages, cu, dist, cls.ref_topk,
            sm_scale=_SM_SCALE, k_scale=1.0,
        )  # fmt: skip

    @classmethod
    def _run_dcp(cls):
        x, (seq_lens, cu, dist) = cls.inputs, cls.meta
        pages = jnp.asarray(cls.layout.dcp_pages)
        shards, cands = [], []
        for r in range(_DCP):
            rank = jnp.asarray([r], jnp.int32)
            cache_k, nope, rope = _empty_caches(_PAGES_PER_RANK)
            idx, scores, cache_k = dsa_indexer_step_dcp(
                x["query"], x["key"], x["head_weights"], cache_k,
                seq_lens, pages, cu, dist, rank,
                weight_scale=_WEIGHT_SCALE, topk=_TOPK, dcp_size=_DCP,
                interleave_size=_PAGE_SIZE, segments="mixed",
            )  # fmt: skip
            shards.append([cache_k, nope, rope])
            cands.append((idx, scores))
        cand_idx = jnp.stack([c[0] for c in cands])
        cand_scores = jnp.stack([c[1] for c in cands])
        cls.dcp_local_rows, outs, lses = [], [], []
        for r in range(_DCP):
            rank = jnp.asarray([r], jnp.int32)
            local = merge_dcp_topk(
                cand_idx, cand_scores, rank,
                topk=_TOPK, dcp_size=_DCP, interleave_size=_PAGE_SIZE,
            )  # fmt: skip
            cls.dcp_local_rows.append(local)
            out, lse, shards[r][1], shards[r][2] = sparse_mla_attention_fused_dcp(
                x["ql_nope"], x["q_pe"], x["new_kv_c"], x["new_k_pe"],
                shards[r][1], shards[r][2], seq_lens, pages, cu, dist, rank, local,
                sm_scale=_SM_SCALE, k_scale=1.0, dcp_size=_DCP,
                interleave_size=_PAGE_SIZE,
            )  # fmt: skip
            outs.append(np.asarray(out, np.float32))
            lses.append(np.asarray(lse, np.float32))
        cls.shards = shards
        cls.dcp_out = _merge_partials(outs, lses)

    def _real_tokens(self):
        start = 0
        for seq, n in enumerate(_SEQ_LENS):
            for pos in range(n):
                yield start + pos, seq, pos
            start += n

    def test_caches_are_owner_striped(self):
        ref_k, ref_nope = _token_rows(self.ref_cache_k), _token_rows(self.ref_nope)
        ref_rope = np.asarray(self.ref_rope)
        shards = [
            (_token_rows(k), _token_rows(nope), np.asarray(rope))
            for k, nope, rope in self.shards
        ]
        for _, seq, pos in self._real_tokens():
            ref_page, ref_slot = self.layout.ref_slot(seq, pos)
            owner, page, slot = self.layout.dcp_slot(seq, pos)
            k, nope, rope = shards[owner]
            msg = f"seq {seq} pos {pos}"
            np.testing.assert_array_equal(k[page, slot], ref_k[ref_page, ref_slot], msg)
            np.testing.assert_array_equal(
                nope[page, slot], ref_nope[ref_page, ref_slot], msg
            )
            # RoPE packs 4 consecutive tokens per row; an interleave of
            # page_size keeps a row on one owner, so whole rows must match.
            np.testing.assert_array_equal(
                rope[page, slot // 4], ref_rope[ref_page, ref_slot // 4], msg
            )

    def test_merged_topk_equals_unsharded(self):
        ref = np.asarray(self.ref_topk)
        winners = [set() for _ in range(ref.shape[0])]
        for r, local in enumerate(self.dcp_local_rows):
            local = np.asarray(local)
            glob = np.asarray(
                cp_local_to_global(jnp.asarray(local), r, _DCP, _PAGE_SIZE)
            )
            for t in range(local.shape[0]):
                winners[t].update(glob[t][local[t] >= 0].tolist())
        mismatched = [
            t for t, _, _ in self._real_tokens()
            if winners[t] != set(ref[t][ref[t] >= 0].tolist())
        ]  # fmt: skip
        self.assertFalse(
            mismatched,
            f"{len(mismatched)} rows differ from the unsharded top-k, first "
            f"{mismatched[:8]}",
        )

    def test_merged_attention_matches_unsharded(self):
        num_real = sum(_SEQ_LENS)
        want = np.asarray(self.ref_out, np.float32)[:num_real]
        got = self.dcp_out[:num_real]
        diff = np.abs(got - want)
        self.assertLess(float(diff.max()), _MAX_ABS_DIFF)
        self.assertLess(float(diff.mean()), _MEAN_ABS_DIFF)


class TestDsaDcpDensePrefill(CustomTestCase):
    """Prefill under DCP gathers the shards into an unsharded cache
    (TpuPallasSparseMlaAttnBackend._dcp_dense_cache) and runs the unsharded
    ops, so its rows and outputs must equal an unsharded server's bit for bit."""

    def test_dense_gather_reproduces_unsharded_step(self):
        base = TestDsaDcp
        base.setUpClass()
        rng = np.random.default_rng(2)
        num_seqs = len(_SEQ_LENS)
        # A second chunk of 256 tokens per sequence on top of the prefill.
        chunk = 256
        x = _random_inputs(rng, chunk * num_seqs)
        seq_lens = jnp.asarray(np.asarray(_SEQ_LENS, np.int32) + chunk)
        cu = jnp.asarray(np.arange(num_seqs + 1, dtype=np.int32) * chunk)
        dist = jnp.asarray(np.array([0, 0, num_seqs], np.int32))

        ref_pages = jnp.asarray(base.layout.ref_pages)
        ref_rows, _ = dsa_indexer_step(
            x["query"], x["key"], x["head_weights"], base.ref_cache_k,
            seq_lens, ref_pages, cu, dist,
            weight_scale=_WEIGHT_SCALE, topk=_TOPK, segments="mixed",
        )  # fmt: skip
        ref_out, _, _ = sparse_mla_attention_fused(
            x["ql_nope"], x["q_pe"], x["new_kv_c"], x["new_k_pe"],
            base.ref_nope, base.ref_rope, seq_lens, ref_pages, cu, dist, ref_rows,
            sm_scale=_SM_SCALE, k_scale=1.0,
        )  # fmt: skip

        dense = [_dense_cache([s[i] for s in base.shards], base.layout.dcp_pages) for i in range(3)]
        dense_pages = jnp.arange(len(base.layout.dcp_pages) * _DCP, dtype=jnp.int32)
        rows, _ = dsa_indexer_step(
            x["query"], x["key"], x["head_weights"], dense[0],
            seq_lens, dense_pages, cu, dist,
            weight_scale=_WEIGHT_SCALE, topk=_TOPK, segments="mixed",
        )  # fmt: skip
        out, _, _ = sparse_mla_attention_fused(
            x["ql_nope"], x["q_pe"], x["new_kv_c"], x["new_k_pe"],
            dense[1], dense[2], seq_lens, dense_pages, cu, dist, rows,
            sm_scale=_SM_SCALE, k_scale=1.0,
        )  # fmt: skip
        rows, ref_rows = np.asarray(rows), np.asarray(ref_rows)
        for t in range(chunk * num_seqs):
            self.assertEqual(
                set(rows[t][rows[t] >= 0].tolist()),
                set(ref_rows[t][ref_rows[t] >= 0].tolist()),
            )
        np.testing.assert_array_equal(
            np.asarray(out, np.float32), np.asarray(ref_out, np.float32)
        )


def _dense_cache(shards, pages) -> jax.Array:
    """``_dcp_dense_cache`` for simulated ranks: each rank's pages of the page
    table, rank-major inside every widened page."""
    per_rank = [np.asarray(s)[pages] for s in shards]  # [ranks][N, ...]
    stacked = np.stack(per_rank, axis=1)  # [N, ranks, ...]
    return jnp.asarray(stacked.reshape(-1, *stacked.shape[2:]))


class TestDsaDcpDecode(CustomTestCase):
    """One decode step (padded batch) on top of a prefill, as the server runs it."""

    def test_decode_step_matches_unsharded(self):
        base = TestDsaDcp
        base.setUpClass()
        rng = np.random.default_rng(1)
        num_seqs, padded = len(_SEQ_LENS), 8
        x = _random_inputs(rng, padded)
        seq_lens = jnp.asarray(np.asarray(_SEQ_LENS, np.int32) + 1)
        cu = np.full(num_seqs + 1, num_seqs, np.int32)
        cu[: num_seqs + 1] = np.arange(num_seqs + 1)
        cu = jnp.asarray(cu)
        dist = jnp.asarray(np.array([num_seqs] * 3, np.int32))

        ref_pages = jnp.asarray(base.layout.ref_pages)
        ref_topk, _ = dsa_indexer_step(
            x["query"], x["key"], x["head_weights"], base.ref_cache_k,
            seq_lens, ref_pages, cu, dist,
            weight_scale=_WEIGHT_SCALE, topk=_TOPK, segments="decode",
        )  # fmt: skip
        ref_out, _, _ = sparse_mla_attention_fused(
            x["ql_nope"], x["q_pe"], x["new_kv_c"], x["new_k_pe"],
            base.ref_nope, base.ref_rope, seq_lens, ref_pages, cu, dist, ref_topk,
            sm_scale=_SM_SCALE, k_scale=1.0,
        )  # fmt: skip

        pages = jnp.asarray(base.layout.dcp_pages)
        cands = []
        for r, (cache_k, _, _) in enumerate(base.shards):
            idx, scores, _ = dsa_indexer_step_dcp(
                x["query"], x["key"], x["head_weights"], cache_k,
                seq_lens, pages, cu, dist, jnp.asarray([r], jnp.int32),
                weight_scale=_WEIGHT_SCALE, topk=_TOPK, dcp_size=_DCP,
                interleave_size=_PAGE_SIZE, segments="decode",
            )  # fmt: skip
            cands.append((idx, scores))
        cand_idx = jnp.stack([c[0] for c in cands])
        cand_scores = jnp.stack([c[1] for c in cands])
        outs, lses, winners = [], [], [set() for _ in range(num_seqs)]
        for r, (_, nope, rope) in enumerate(base.shards):
            rank = jnp.asarray([r], jnp.int32)
            local = merge_dcp_topk(
                cand_idx, cand_scores, rank,
                topk=_TOPK, dcp_size=_DCP, interleave_size=_PAGE_SIZE,
            )  # fmt: skip
            glob = np.asarray(cp_local_to_global(local, r, _DCP, _PAGE_SIZE))
            local = np.asarray(local)
            for t in range(num_seqs):
                winners[t].update(glob[t][local[t] >= 0].tolist())
            out, lse, _, _ = sparse_mla_attention_fused_dcp(
                x["ql_nope"], x["q_pe"], x["new_kv_c"], x["new_k_pe"],
                nope, rope, seq_lens, pages, cu, dist, rank, jnp.asarray(local),
                sm_scale=_SM_SCALE, k_scale=1.0, dcp_size=_DCP,
                interleave_size=_PAGE_SIZE,
            )  # fmt: skip
            outs.append(np.asarray(out, np.float32))
            lses.append(np.asarray(lse, np.float32))

        ref_topk = np.asarray(ref_topk)
        for t in range(num_seqs):
            self.assertEqual(winners[t], set(ref_topk[t][ref_topk[t] >= 0].tolist()))
        got = _merge_partials(outs, lses)[:num_seqs]
        diff = np.abs(got - np.asarray(ref_out, np.float32)[:num_seqs])
        self.assertLess(float(diff.max()), _MAX_ABS_DIFF)
        self.assertLess(float(diff.mean()), _MEAN_ABS_DIFF)


class TestGatherPages(CustomTestCase):
    """The DCP prefill gathers pages with a DMA kernel instead of an XLA gather
    (which laid out the whole cache again); it must return cache[ids] exactly."""

    def test_gather_pages_equals_take(self):
        from sglang_tpu.attention.pallas.page_gather import gather_pages

        rng = np.random.default_rng(3)
        ids = rng.integers(0, 48, size=37).astype(np.int32)
        ids[5] = ids[6]  # repeated pages
        for shape in ((48, _PAGE_SIZE, 4, 128), (48, _PAGE_SIZE // 4, 4, 256)):
            cache = rng.integers(0, 256, size=shape, dtype=np.uint8)
            got = np.asarray(gather_pages(jnp.asarray(cache), jnp.asarray(ids)))
            np.testing.assert_array_equal(got, cache[ids])


def _merge_partials(outs, lses) -> np.ndarray:
    """Log-sum-exp merge of per-rank ``out [T, H, D]``, ``lse [T, H]``."""
    lse = np.stack(lses)
    lse = np.where(np.isfinite(lse), lse, -np.inf)
    m = lse.max(axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.exp(lse - m)
    w = w / np.maximum(w.sum(axis=0), 1e-30)
    out = np.stack([np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0) for o in outs])
    return (out * w[..., None]).sum(axis=0)


if __name__ == "__main__":
    unittest.main()
