# SPDX-License-Identifier: Apache-2.0
"""The DCP sparse MLA step, run once per rank on one chip, against the unsharded step.

``sparse_mla_attention_dcp`` stores the step's tokens a rank owns in its page
stripe and attends that rank's share of every token's top-k. Run for every rank
of the group and merged by log-sum-exp, the partial outputs must equal
``sparse_mla_attention_fused`` on the unsharded cache, and each rank's stripe
must hold exactly the unsharded cache's tokens of its pages. An extend from an
empty cache is followed by a decode step that reads what the extend wrote.

GLM-5.3-Flash sizes per DCP group (zero rope, 512 latent), page size 128.

Usage:
    python test/registered/tpu/unit/test_sparse_mla_dcp_kernel.py
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before JAX loads.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_sparse_dcp import sparse_mla_attention_dcp
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    NOPE_DIM,
    ROPE_DIM,
    sparse_mla_attention_fused,
)

_PAGE = 128
_DCP = 4
_HEADS = 16  # the DCP group's heads, as every rank attends them
_TOPK = 256
_SM_SCALE = float(256**-0.5)
_FP8 = ml_dtypes.float8_e4m3fn
# Same kernel on both sides; the merge adds one bf16 rounding of each partial.
# Not measured yet: borrowed from test_mla_sparse_fused's cross-kernel bound.
_MAX_ABS_DIFF = 0.05
_MEAN_ABS_DIFF = 0.003


def _cdiv(a, b):
    return -(-a // b)


def _placeholder_rope():
    return jnp.zeros((1, 1, 4, 128), jnp.uint8)


def _rows(rng, starts, q_lens, kv_lens, bucket):
    """Random top-k rows over each token's causal positions, -1 padded."""
    rows = np.full((bucket, _TOPK), -1, np.int32)
    for start, q_len, kv_len in zip(starts, q_lens, kv_lens):
        for i in range(q_len):
            causal = kv_len - q_len + i + 1
            picks = rng.permutation(causal)[:_TOPK]
            rows[start + i, : len(picks)] = picks
    return rows


class _Layout:
    """One sequence set's unsharded and striped page tables."""

    def __init__(self, max_lens):
        self.num_seqs = len(max_lens)
        pages = [_cdiv(n, _PAGE) for n in max_lens]
        stripes = [_cdiv(n, _PAGE * _DCP) for n in max_lens]
        self.pages_per_seq = max(pages)
        self.stripes_per_seq = max(stripes)
        full = np.zeros((self.num_seqs, self.pages_per_seq), np.int32)
        stripe = np.zeros((self.num_seqs, self.stripes_per_seq), np.int32)
        next_page, next_stripe = 1, 1  # page 0 is the padded-token slot
        for seq in range(self.num_seqs):
            full[seq, : pages[seq]] = np.arange(next_page, next_page + pages[seq])
            next_page += pages[seq]
            stripe[seq, : stripes[seq]] = np.arange(
                next_stripe, next_stripe + stripes[seq]
            )
            next_stripe += stripes[seq]
        self.full, self.stripe = full, stripe
        self.full_pages, self.stripe_pages = next_page, next_stripe

    def stripe_slot(self, seq, pos):
        """(rank, page, slot) of position ``pos`` of sequence ``seq``."""
        widened, offset = divmod(pos, _PAGE * _DCP)
        rank, slot = divmod(offset, _PAGE)
        return rank, int(self.stripe[seq, widened]), slot


def _merge(outs, lses):
    lse = np.stack(lses)  # [R, T, H]
    m = np.max(lse, axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.where(np.isfinite(lse), np.exp(lse - m), 0.0)
    num = sum(w[r][..., None] * np.nan_to_num(outs[r]) for r in range(len(outs)))
    return num / np.maximum(w.sum(axis=0), 1e-30)[..., None]


@unittest.skipUnless(
    any(d.platform == "tpu" for d in jax.devices()), "needs a TPU device"
)
class TestSparseMlaDcpKernel(CustomTestCase):
    def _step(self, layout, caches, q_lens, kv_lens, *, is_decode, seed):
        rng = np.random.default_rng(seed)
        bucket = sum(q_lens)
        ql_nope = jnp.asarray(
            rng.normal(size=(bucket, _HEADS, NOPE_DIM)).astype(ml_dtypes.bfloat16)
        )
        q_pe = jnp.zeros((bucket, _HEADS, ROPE_DIM), jnp.bfloat16)
        new_kv_c = jnp.asarray(rng.normal(size=(bucket, NOPE_DIM)).astype(_FP8))
        new_k_pe = jnp.zeros((bucket, ROPE_DIM), jnp.float8_e4m3fn)
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        rows = jnp.asarray(_rows(rng, cu_q_lens[:-1], q_lens, kv_lens, bucket))
        n = layout.num_seqs
        dist = jnp.asarray([n] * 3 if is_decode else [0, 0, n], jnp.int32)
        common = dict(
            kv_lens=jnp.asarray(kv_lens, jnp.int32),
            cu_q_lens=jnp.asarray(cu_q_lens),
            distribution=dist,
        )

        want, full_nope, _ = sparse_mla_attention_fused(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            caches["full"],
            _placeholder_rope(),
            common["kv_lens"],
            jnp.asarray(layout.full.reshape(-1)),
            common["cu_q_lens"],
            dist,
            rows,
            sm_scale=_SM_SCALE,
            k_scale=1.0,
            zero_rope=True,
        )
        caches["full"] = full_nope
        outs, lses = [], []
        for rank in range(_DCP):
            out, lse, nope, _ = sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                caches[rank],
                _placeholder_rope(),
                common["kv_lens"],
                jnp.asarray(layout.stripe.reshape(-1)),
                common["cu_q_lens"],
                dist,
                rows,
                jnp.asarray([rank], jnp.int32),
                sm_scale=_SM_SCALE,
                k_scale=1.0,
                dcp_size=_DCP,
                zero_rope=True,
            )
            caches[rank] = nope
            outs.append(np.asarray(out, np.float32))
            lses.append(np.asarray(lse, np.float32))
        got = _merge(outs, lses)
        want = np.asarray(want, np.float32)
        diff = np.abs(got - want)
        self.assertLess(diff.max(), _MAX_ABS_DIFF)
        self.assertLess(diff.mean(), _MEAN_ABS_DIFF)
        self._assert_stripes(layout, caches, kv_lens)

    def _assert_stripes(self, layout, caches, kv_lens):
        full = np.asarray(caches["full"]).reshape(layout.full_pages, _PAGE, -1)
        stripes = [
            np.asarray(caches[r]).reshape(layout.stripe_pages, _PAGE, -1)
            for r in range(_DCP)
        ]
        for seq, kv_len in enumerate(kv_lens):
            for pos in range(kv_len):
                rank, page, slot = layout.stripe_slot(seq, pos)
                full_page = int(layout.full[seq, pos // _PAGE])
                np.testing.assert_array_equal(
                    stripes[rank][page, slot], full[full_page, pos % _PAGE]
                )

    def test_extend_then_decode(self):
        # 3 widened pages for the first sequence; the second has 1, so ranks 1-3
        # own none of its first tokens' keys (the -inf / dummy-key rows).
        lens = [_PAGE * _DCP * 2 + 70, 200]
        layout = _Layout([n + 1 for n in lens])
        caches = {
            "full": jnp.zeros((layout.full_pages, _PAGE, 4, 128), jnp.uint8),
            **{
                r: jnp.zeros((layout.stripe_pages, _PAGE, 4, 128), jnp.uint8)
                for r in range(_DCP)
            },
        }
        self._step(layout, caches, lens, lens, is_decode=False, seed=0)
        decode_lens = [n + 1 for n in lens]
        ones = [1] * len(lens)
        self._step(layout, caches, ones, decode_lens, is_decode=True, seed=1)


if __name__ == "__main__":
    unittest.main()
