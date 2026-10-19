# SPDX-License-Identifier: Apache-2.0
"""Write-then-attend tests for the vendored sparse MLA kernels.

The tests vendored with the kernels (``test_mla_sparse_*.py``) compare each
piece against a sibling of itself: the Pallas writer against the XLA writer,
the SparseCore cache layout against the TensorCore one, on one sequence and a
pre-filled cache. That leaves two things unchecked, and an sglang attention
backend relies on both:

1. The attention math against an implementation that shares no code with the
   kernel. The reference here is NumPy over each sequence's logical keys.
2. The writer and the reader agreeing on where a token lives when the batch
   holds several sequences. One addresses through ``seq_lens``, the other
   through per-token ``topk_indices``.

Each case runs one serving step: scatter the step's new latents with
``update_sparse_mla_kv_cache``, then call ``sparse_ragged_paged_attention``.
The dense-top-k cases also pin the equivalence a backend can use to serve a
dense MLA model on this kernel: a row that lists every causal position gives
dense attention.

Metadata mirrors ``TpuPallasRpaV3AttnBackend._build_metadata``: a rectangular
``page_indices`` table whose unused entries are 0, page ids that never include
page 0 (the pool keeps it as the padded-token slot), ``kv_lens`` that include
the step's new tokens, and the MLA backend's ``distribution``. Batches are
padded to a token bucket and a sequence bucket, as compiled graphs are.

Usage:
    python test/registered/tpu/unit/test_mla_sparse_write_then_attend.py
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: select the
# physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import msgspec
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.layers.attention.pallas.mla_sparse import native_sc_cache_layout
from sglang_tpu.layers.attention.pallas.mla_sparse.kernel import (
    sparse_ragged_paged_attention,
)
from sglang_tpu.layers.attention.pallas.mla_sparse.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
    update_sparse_mla_kv_cache,
)

# The only geometry the kernel accepts; DeepSeek V3 and GLM-5.2 both have it.
_KV_LORA_RANK = 512
_ROPE_DIM = 64
# A rope token is stored zero-padded to one 128-byte lane row.
_ROPE_ROW_BYTES = 128
_TOPK = 256
# DeepSeek V3: (qk_nope_head_dim 128 + qk_rope_head_dim 64) ** -0.5.
_SM_SCALE = float((128 + _ROPE_DIM) ** -0.5)
_FP8 = ml_dtypes.float8_e4m3fn

# The tolerance the vendored attention tests use between kernel variants.
_RTOL, _ATOL = 2e-2, 2e-2


class _Step(msgspec.Struct, frozen=True):
    """One padded serving step and what it must produce."""

    page_size: int
    total_pages: int
    num_tokens: int  # real tokens; the arrays below are padded past it
    k_scale: float
    q: np.ndarray  # [token_bucket, heads, 576] bf16
    new_nope: np.ndarray  # [token_bucket, 512] fp8
    new_rope: np.ndarray  # [token_bucket, 64] fp8
    topk: np.ndarray  # [token_bucket, _TOPK] i32, sequence positions, -1 padded
    kv_lens: np.ndarray  # [seq_bucket] i32
    page_indices: np.ndarray  # [seq_bucket * pages_per_seq] i32
    cu_q_lens: np.ndarray  # [seq_bucket + 1] i32
    distribution: np.ndarray  # [3] i32
    # Cache contents as one byte row per token slot, [pages * page_size, bytes].
    nope_before: np.ndarray
    rope_before: np.ndarray
    nope_after: np.ndarray
    rope_after: np.ndarray
    expected: np.ndarray  # [num_tokens, heads, 512] f32


def _cdiv(a, b):
    return (a + b - 1) // b


def _random_fp8(rng, shape, k_scale):
    """Random latents, quantised per-tensor as the backend does before a write."""
    values = rng.normal(size=shape).astype(np.float32)
    return (values / k_scale).astype(_FP8)


def _stale_rows(rng, num_rows, k_scale):
    """Valid FP8 in every slot, as pages freed by earlier requests hold."""
    nope_rows = _random_fp8(rng, (num_rows, _KV_LORA_RANK), k_scale).view(np.uint8)
    rope_rows = np.zeros((num_rows, _ROPE_ROW_BYTES), np.uint8)
    rope_rows[:, :_ROPE_DIM] = _random_fp8(rng, (num_rows, _ROPE_DIM), k_scale).view(
        np.uint8
    )
    return nope_rows, rope_rows


def _topk_rows(positions, rng, num_selected):
    """Per-token rows of sequence positions, valid entries first, ``-1`` padded.

    ``num_selected=None`` lists every causal position; otherwise each row is an
    unsorted random subset of them, as an indexer would hand over.
    """
    rows = np.full((len(positions), _TOPK), -1, np.int32)
    for row, pos in zip(rows, positions):
        if num_selected is None:
            row[: pos + 1] = np.arange(pos + 1)
        else:
            n = min(pos + 1, num_selected)
            row[:n] = rng.choice(pos + 1, size=n, replace=False)
    return rows


def _reference(q, keys, topk):
    """Softmax over each token's selected keys; the value is the nope latent."""
    out = np.zeros((len(q), q.shape[1], _KV_LORA_RANK), np.float32)
    for t, row in enumerate(topk):
        selected = keys[row[row >= 0]]
        scores = q[t].astype(np.float32) @ selected.T * _SM_SCALE
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out[t] = probs @ selected[:, :_KV_LORA_RANK]
    return out


def _build_step(
    lens,
    *,
    page_size,
    num_heads,
    k_scale,
    token_bucket,
    seq_bucket,
    is_decode,
    num_selected,
    seed,
):
    """``lens`` is ``[(q_len, kv_len)]`` per real sequence; ``kv_len`` counts the
    ``q_len`` new tokens, so ``kv_len - q_len`` tokens are already cached."""
    rng = np.random.default_rng(seed)
    num_seqs = len(lens)
    num_tokens = sum(q_len for q_len, _ in lens)
    assert num_seqs <= seq_bucket and num_tokens <= token_bucket
    assert all(1 <= q_len <= kv_len <= _TOPK for q_len, kv_len in lens)

    needed = [_cdiv(kv_len, page_size) for _, kv_len in lens]
    # One column more than any sequence needs, so every row has an unused tail.
    page_table = np.zeros((seq_bucket, max(needed) + 1), np.int32)
    total_pages = 1 + sum(needed) + 8
    page_pool = 1 + rng.permutation(total_pages - 1)

    nope_before, rope_before = _stale_rows(rng, total_pages * page_size, k_scale)
    nope_after, rope_after = nope_before.copy(), rope_before.copy()

    head_dim = _KV_LORA_RANK + _ROPE_DIM
    q = np.zeros((token_bucket, num_heads, head_dim), ml_dtypes.bfloat16)
    new_nope = np.zeros((token_bucket, _KV_LORA_RANK), _FP8)
    new_rope = np.zeros((token_bucket, _ROPE_DIM), _FP8)
    topk = np.full((token_bucket, _TOPK), -1, np.int32)
    expected = np.zeros((num_tokens, num_heads, _KV_LORA_RANK), np.float32)

    pages_used = 0
    token = 0
    for i, (q_len, kv_len) in enumerate(lens):
        pages = page_pool[pages_used : pages_used + needed[i]]
        pages_used += needed[i]
        page_table[i, : needed[i]] = pages

        k_nope = _random_fp8(rng, (kv_len, _KV_LORA_RANK), k_scale)
        k_rope = _random_fp8(rng, (kv_len, _ROPE_DIM), k_scale)
        positions = np.arange(kv_len)
        rows = pages[positions // page_size] * page_size + positions % page_size
        prefix = kv_len - q_len
        for nope, rope, live in (
            (nope_before, rope_before, rows[:prefix]),
            (nope_after, rope_after, rows),
        ):
            nope[live] = k_nope[: len(live)].view(np.uint8)
            rope[live, :_ROPE_DIM] = k_rope[: len(live)].view(np.uint8)
            rope[live, _ROPE_DIM:] = 0

        step = slice(token, token + q_len)
        new_nope[step] = k_nope[prefix:]
        new_rope[step] = k_rope[prefix:]
        q[step] = rng.normal(size=(q_len, num_heads, head_dim)).astype(q.dtype)
        topk[step] = _topk_rows(positions[prefix:], rng, num_selected)

        keys = np.concatenate([k_nope, k_rope], axis=-1).astype(np.float32) * k_scale
        expected[step] = _reference(q[step], keys, topk[step])
        token += q_len

    q_lens = np.zeros(seq_bucket, np.int32)
    kv_lens = np.zeros(seq_bucket, np.int32)
    q_lens[:num_seqs] = [q_len for q_len, _ in lens]
    kv_lens[:num_seqs] = [kv_len for _, kv_len in lens]
    # TpuPallasMlaAttnBackend: decode is [N, N, N], extend is [0, 0, N].
    distribution = [num_seqs] * 3 if is_decode else [0, 0, num_seqs]

    return _Step(
        page_size=page_size,
        total_pages=total_pages,
        num_tokens=num_tokens,
        k_scale=k_scale,
        q=q,
        new_nope=new_nope,
        new_rope=new_rope,
        topk=topk,
        kv_lens=kv_lens,
        page_indices=page_table.reshape(-1),
        cu_q_lens=np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        distribution=np.array(distribution, np.int32),
        nope_before=nope_before,
        rope_before=rope_before,
        nope_after=nope_after,
        rope_after=rope_after,
        expected=expected,
    )


def _to_layout(*, nope_rows, rope_rows, step, layout):
    """Byte rows -> the physical cache arrays of ``layout``."""
    nope = nope_rows.reshape(step.total_pages, step.page_size, 4, 128)
    rope = rope_rows.reshape(step.total_pages, step.page_size // 4, 4, 128)
    if layout is KVCacheLayout.SPARSECORE:
        return (
            native_sc_cache_layout.pack_nope(nope),
            native_sc_cache_layout.pack_rope_banded(rope),
        )
    return nope, rope


# Fresh prefill, a chunk continuing a prefix across a page boundary, and a
# short extend.
_EXTEND_LENS = [(37, 37), (64, 200), (5, 133)]
# First token, a prefix ending one past a page boundary, and kv_len == topk,
# the row with no -1 padding.
_DECODE_LENS = [(1, 1), (1, 129), (1, _TOPK)]


@unittest.skipUnless(
    any(d.platform == "tpu" for d in jax.devices()), "needs a TPU device"
)
class TestSparseMlaWriteThenAttend(CustomTestCase):
    def _run(self, lens, *, layout, **step_kwargs):
        step = _build_step(lens, **step_kwargs)
        specs = [
            SparseMLAKVCacheSpec.create(
                cache_type, layout, step.total_pages, step.page_size, dim
            )
            for cache_type, dim in (
                (KVCacheType.NOPE, _KV_LORA_RANK),
                (KVCacheType.ROPE, _ROPE_DIM),
            )
        ]
        before = _to_layout(
            nope_rows=step.nope_before,
            rope_rows=step.rope_before,
            step=step,
            layout=layout,
        )
        self.assertEqual([c.shape for c in before], [s.shape for s in specs])

        page_indices = jnp.asarray(step.page_indices)
        cu_q_lens = jnp.asarray(step.cu_q_lens)
        nope, rope = update_sparse_mla_kv_cache(
            jnp.asarray(before[0]),
            jnp.asarray(before[1]),
            jnp.asarray(step.new_nope),
            jnp.asarray(step.new_rope),
            jnp.asarray(step.kv_lens),
            page_indices,
            cu_q_lens,
            nope_spec=specs[0],
            rope_spec=specs[1],
        )
        out = sparse_ragged_paged_attention(
            jnp.asarray(step.q),
            nope,
            rope,
            jnp.asarray(step.topk),
            page_indices,
            cu_q_lens,
            jnp.asarray(step.distribution),
            sm_scale=_SM_SCALE,
            k_scale=step.k_scale,
            cache_layout=layout.value,
        )
        nope, rope, out = jax.block_until_ready((nope, rope, out))

        # The write: new tokens land in their slots and nothing else moves.
        nope_after, rope_after = _to_layout(
            nope_rows=step.nope_after,
            rope_rows=step.rope_after,
            step=step,
            layout=layout,
        )
        np.testing.assert_array_equal(np.asarray(nope), nope_after)
        np.testing.assert_array_equal(np.asarray(rope), rope_after)

        # The read: the output is head_dim wide; its first 512 dims are the latent.
        got = out[: step.num_tokens, :, :_KV_LORA_RANK].astype(jnp.float32)
        np.testing.assert_allclose(
            np.asarray(got), step.expected, rtol=_RTOL, atol=_ATOL
        )

    def _extend(self, layout, **overrides):
        kwargs = dict(
            page_size=128,
            num_heads=16,
            k_scale=1.0,
            token_bucket=128,
            seq_bucket=4,
            is_decode=False,
            num_selected=None,
            seed=0,
        )
        kwargs.update(overrides)
        self._run(_EXTEND_LENS, layout=layout, **kwargs)

    def _decode(self, layout):
        # Small-chunk path. tokens * topk must be a multiple of the SparseCore
        # gather's owner batch (2,048 rows on v7x), hence 8 tokens at topk 256.
        self._run(
            _DECODE_LENS,
            layout=layout,
            page_size=128,
            num_heads=8,
            k_scale=1.0,
            token_bucket=8,
            seq_bucket=8,
            is_decode=True,
            num_selected=None,
            seed=1,
        )

    def test_extend_dense_topk_sparsecore(self):
        self._extend(KVCacheLayout.SPARSECORE)

    def test_extend_dense_topk_tensorcore(self):
        self._extend(KVCacheLayout.TENSORCORE)

    def test_decode_dense_topk_sparsecore(self):
        self._decode(KVCacheLayout.SPARSECORE)

    def test_decode_dense_topk_tensorcore(self):
        self._decode(KVCacheLayout.TENSORCORE)

    def test_extend_page_size_64_k_scale(self):
        # k_scale enters twice, in the scores and in the values.
        self._extend(KVCacheLayout.SPARSECORE, page_size=64, k_scale=0.5, seed=2)

    def test_extend_sparse_topk(self):
        self._extend(KVCacheLayout.SPARSECORE, num_selected=24, seed=3)


if __name__ == "__main__":
    unittest.main()
