# SPDX-License-Identifier: Apache-2.0
"""The pooled (kpool) DSA indexer step must select what a from-scratch reference selects.

The reference keeps each request's raw keys and gates as plain arrays and, at
every step, recompresses all closed pools, scores them and expands the choice;
the step under test pages its caches and closes pools incrementally across
prefill, decode and speculative-verify steps. Scoring runs on the StreamIndex
Pallas kernel, so this needs a TPU.
"""

import os
import sys
import unittest
from functools import partial

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_kpool_topk as kp
from sglang_tpu.attention.pallas.kpool_compress import compress_and_store

# The kernel needs whole 128-token pages; TOPK keeps 16 of up to 128 pools.
D, HEADS, KPOOL, PAGE, PAGES_PER_SEQ, TOPK = 128, 8, 4, 128, 4, 64
NUM_PAGES = 2 * PAGES_PER_SEQ + 1
# Pools may swap only when their scores are within this fraction of the largest
# sum of |head terms|: the step compresses pools inside its graph, so a pool's
# FP8 record can round differently from the reference's (~1e-2), and the kernel
# weights heads in bf16 (~5e-3); signed head weights cancel in the sum.
SCORE_RTOL = 4e-2  # measured: p99 1.0%, max 2.5% over 381 prefill rows


def _ref_scores(keys, gates, ape, q, w, pos, weight_scale):
    """Reference (score, sum of |head terms|) of every closed pool for one query
    at sequence position pos."""
    pool_len = (pos + 1) // KPOOL
    pooled = []
    for p in range(pool_len):
        sl = slice(p * KPOOL, (p + 1) * KPOOL)
        c = kp.compress_pools(jnp.asarray(keys[sl]), jnp.asarray(gates[sl]), ape)
        k8, ks, _ = kp.quantize_ue8m0(c[None])
        pooled.append((np.asarray(k8[0].astype(jnp.float32)), float(ks[0, 0])))
    q_rot = kp.hadamard_rotate(jnp.asarray(q, jnp.bfloat16)).astype(jnp.bfloat16)
    q8, qs, _ = kp.quantize_ue8m0(q_rot)
    q8 = np.asarray(q8.astype(jnp.float32))
    ww = np.asarray(w) * np.asarray(qs)[:, 0] * weight_scale
    terms = [ww * np.maximum(q8 @ k, 0.0) * s for k, s in pooled]
    return (
        np.array([t.sum() for t in terms], np.float32),
        np.array([np.abs(t).sum() for t in terms], np.float32),
    )


class TestKpoolIndexerStep(CustomTestCase):
    def _check_row(self, row, ref, pos):
        scores, mags = ref
        pool_len = (pos + 1) // KPOOL
        n_pick = min(TOPK // KPOOL, pool_len)
        n = int((row >= 0).sum())
        # Selected positions form a prefix of the row.
        self.assertTrue((row[:n] >= 0).all() and (row[n:] == -1).all())
        hist, tail = row[: n_pick * KPOOL], row[n_pick * KPOOL : n]
        # The open pool's tokens follow the history.
        want_tail = [pool_len * KPOOL + j for j in range((pos + 1) % KPOOL)]
        self.assertEqual(sorted(tail.tolist()), want_tail, f"pos {pos} tail")
        groups = hist.reshape(n_pick, KPOOL)
        self.assertTrue(
            (groups == groups[:, :1] + np.arange(KPOOL)).all(), f"pos {pos}"
        )
        picked = groups[:, 0] // KPOOL
        self.assertEqual(len(set(picked.tolist())), n_pick, f"pos {pos} dup")
        self.assertTrue((picked < pool_len).all(), f"pos {pos} future pool")
        if n_pick < pool_len:
            kth = np.sort(scores)[::-1][n_pick - 1]
            tol = SCORE_RTOL * max(mags.max(), 1e-6)
            self.assertGreaterEqual(scores[picked].min(), kth - tol, f"pos {pos}")

    def test_matches_reference_over_prefill_decode_verify(self):
        rng = np.random.default_rng(0)
        ape = jnp.asarray(rng.normal(size=(KPOOL, D)).astype(np.float32))
        weight_scale = HEADS**-0.5 * D**-0.5
        step = jax.jit(
            lambda *a: kp.kpool_indexer_step(
                *a, weight_scale=weight_scale, topk=TOPK, kpool=KPOOL
            )
        )
        raw = jnp.zeros((NUM_PAGES, PAGE, 2 * D), jnp.bfloat16)
        pooled = jnp.zeros(
            (NUM_PAGES, PAGE // 4, 4, kp.pooled_record_bytes(D)), jnp.uint8
        )
        # Request r owns pages 1 + r * PAGES_PER_SEQ ...
        page_indices = jnp.asarray(
            [1 + r * PAGES_PER_SEQ + i for r in range(2) for i in range(PAGES_PER_SEQ)],
            jnp.int32,
        )
        hist = {r: ([], []) for r in range(2)}  # committed raw keys / gates
        lens = [0, 0]
        # (tokens per request, committed per request) for each forward. The
        # first is a chunked prefill (4096 rows > the kernel's 2048-row chunk).
        plan = [((301, 214), (301, 214)), ((1, 1), (1, 1)), ((1, 1), (1, 1))]
        plan += [((8, 8), (3, 8)), ((1, 1), (1, 1)), ((8, 8), (8, 2)), ((1, 1), (1, 1))]
        plan += [((1, 1), (1, 1))] * 6
        for new, keep in plan:
            n_tok = sum(new)
            pad = 4096 if n_tok > 32 else 32
            keys = rng.normal(size=(pad, D)).astype(np.float32)
            gates = rng.normal(size=(pad, D)).astype(np.float32)
            query = rng.normal(size=(pad, HEADS, D)).astype(np.float32)
            weights = rng.normal(size=(pad, HEADS)).astype(np.float32)
            kb = np.asarray(jnp.asarray(keys, jnp.bfloat16).astype(jnp.float32))
            gb = np.asarray(jnp.asarray(gates, jnp.bfloat16).astype(jnp.float32))
            kv_lens = jnp.asarray([lens[0] + new[0], lens[1] + new[1]], jnp.int32)
            cu = jnp.asarray([0, new[0], n_tok], jnp.int32)
            out, raw, pooled = step(
                jnp.asarray(query, jnp.bfloat16),
                jnp.asarray(keys, jnp.bfloat16),
                jnp.asarray(gates, jnp.bfloat16),
                jnp.asarray(weights),
                ape,
                raw,
                pooled,
                kv_lens,
                page_indices,
                cu,
                jnp.asarray([0, 0, 2], jnp.int32),
            )
            out = np.asarray(out)
            t = 0
            for r in range(2):
                ks, gs = hist[r]
                win_k = ks + list(kb[t : t + new[r]])
                win_g = gs + list(gb[t : t + new[r]])
                for i in range(new[r]):
                    pos = lens[r] + i
                    ref = _ref_scores(
                        np.stack(win_k[: pos + 1]),
                        np.stack(win_g[: pos + 1]),
                        ape,
                        query[t + i],
                        weights[t + i],
                        pos,
                        weight_scale,
                    )
                    self._check_row(out[t + i], ref, pos)
                # Keep only the accepted tokens, as a verify step would.
                hist[r] = (win_k[: lens[r] + keep[r]], win_g[: lens[r] + keep[r]])
                lens[r] += keep[r]
                t += new[r]
            self.assertTrue((out[n_tok:] == -1).all())

    def test_row_chunked_prefill_matches_one_pass(self):
        # A long prefill row-chunks the scores (here forced at a small context);
        # chunking must not change the selection.
        tokens, context, topk = 2 * kp._CHUNK_TOKENS, 8192, 2048
        pages = context // PAGE
        rng = np.random.default_rng(1)

        def f(*shape, dtype=jnp.bfloat16):
            return jnp.asarray(rng.standard_normal(shape), jnp.float32).astype(dtype)

        args = (
            f(tokens, HEADS, D),
            f(tokens, D),
            f(tokens, D),
            f(tokens, HEADS, dtype=jnp.float32),
            f(KPOOL, D, dtype=jnp.float32),
            jnp.zeros((pages + 1, PAGE, 2 * D), jnp.bfloat16),
            jnp.zeros((pages + 1, PAGE // 4, 4, kp.pooled_record_bytes(D)), jnp.uint8),
            jnp.asarray([context], jnp.int32),
            jnp.asarray(1 + np.arange(pages), jnp.int32),
            jnp.asarray([0, tokens], jnp.int32),
            jnp.asarray([0, 0, 1], jnp.int32),
        )
        outs = []
        default = kp._MAX_UNCHUNKED_SCORE_BYTES
        for threshold in (1 << 62, 0):
            kp._MAX_UNCHUNKED_SCORE_BYTES = threshold
            try:
                step = jax.jit(
                    lambda *a: kp.kpool_indexer_step(
                        *a, weight_scale=1.0, topk=topk, kpool=KPOOL
                    )[0]
                )
                outs.append(np.asarray(step(*args)))
            finally:
                kp._MAX_UNCHUNKED_SCORE_BYTES = default
        np.testing.assert_array_equal(outs[1], outs[0])


def _scatter_pools(raw, old, ape, lens, pages, cu, distribution, tokens):
    """The generic gather/compress/scatter pooling that the fused writer replaces."""
    req, pos, valid = kp._token_rows(lens, cu, distribution, tokens)
    closing = valid & (pos % KPOOL == KPOOL - 1)
    member = jnp.maximum(pos[:, None] - 3 + jnp.arange(KPOOL)[None, :], 0)
    loc = {
        "page_indices": pages,
        "num_seqs": lens.shape[0],
        "page_size": raw.shape[1],
        "num_pages": raw.shape[0],
    }
    page, slot = kp._page_slot(
        jnp.broadcast_to(req[:, None], member.shape),
        member,
        valid=jnp.broadcast_to(closing[:, None], member.shape),
        **loc,
    )
    rows = raw[jnp.minimum(page, raw.shape[0] - 1), slot]
    pooled = kp.compress_pools(rows[..., :D], rows[..., D:], ape)
    records = kp._pool_records(pooled, kp.pooled_record_bytes(D))
    page, slot = kp._page_slot(req, pos // KPOOL, valid=closing, **loc)
    return old.at[page, slot // 4, slot % 4].set(records, mode="drop")


class TestFusedPooling(CustomTestCase):
    def test_writes_the_same_cache_bytes_as_the_generic_scatter(self):
        # (name, start, new tokens per request, token bucket, mode); page edges
        # sit at p-1/p and pooled-page edges at 4p-1/4p for page size p.
        def cases(p):
            return [
                # 128 new tokens per request: the streaming schedule.
                ("full_pages", [0, 0], [p, p], 2 * p, "random"),
                ("raw_page_boundary", [p - 4, 4 * p - 4], [12, 8], 32, "random"),
                (
                    "pooled_page_boundary",
                    [4 * p - 8, 4 * p - 4],
                    [20, 12],
                    32,
                    "random",
                ),
                ("decode_close", [3, p + 3], [1, 1], 32, "random"),
                ("partial_quad", [4, p + 4], [8, 8], 32, "random"),
                ("inactive_request", [3, 0], [1, 0], 32, "random"),
                ("nonfinite", [0, 0], [16, 16], 32, "nonfinite"),
                # Two requests write distinct records of one physical quad.
                ("shared_quad", [0, 4], [4, 4], 32, "shared"),
                ("decode_shared_quad", [3, 7], [1, 1], 32, "shared"),
                ("verify_page_boundary", [p - 1, 4 * p - 1], [6, 6], 32, "random"),
                ("verify_mixed_q", [p - 3, 4 * p - 1], [3, 9], 32, "random"),
            ]

        ref = jax.jit(_scatter_pools, static_argnums=(7,))
        for page_size in (64, 128, 256):
            for i, (name, start, new, tokens, mode) in enumerate(cases(page_size)):
                with self.subTest(name=name, page_size=page_size):
                    rng = np.random.default_rng(1024 + i)
                    raw = rng.normal(size=(17, page_size, 2 * D)).astype(np.float32)
                    old = rng.integers(
                        0, 256, (17, page_size // 4, 4, 2 * D), dtype=np.uint8
                    )
                    pages = rng.permutation(np.arange(1, 17)).astype(np.int32)
                    if mode == "shared":
                        pages[8] = pages[0]
                    elif mode == "nonfinite":
                        raw[pages[0], 0, 0] = np.nan
                        raw[pages[0], 4, 0] = np.inf
                        raw[pages[0], 8, 0] = -np.inf
                    active = 1 if new[1] == 0 else 2
                    args = (
                        jnp.asarray(raw, jnp.bfloat16),
                        jnp.asarray(old),
                        jnp.asarray(rng.normal(size=(KPOOL, D)), jnp.float32),
                        jnp.asarray(np.array(start) + np.array(new), jnp.int32),
                        jnp.asarray(pages),
                        jnp.asarray([0, new[0], sum(new)], jnp.int32),
                        jnp.asarray([0, 0, active], jnp.int32),
                        tokens,
                    )
                    expected = np.asarray(ref(*args))
                    # The token-bucket bound and the exact per-request bound can
                    # pick different schedules; both must write the same bytes,
                    # including untouched records and record padding.
                    for bound in (None, max(new)):
                        fused = jax.jit(
                            partial(compress_and_store, max_query_length=bound),
                            static_argnums=(7,),
                        )
                        np.testing.assert_array_equal(
                            np.asarray(fused(*args)), expected
                        )


if __name__ == "__main__":
    unittest.main()
