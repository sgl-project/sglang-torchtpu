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

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_kpool_topk as kp

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


if __name__ == "__main__":
    unittest.main()
