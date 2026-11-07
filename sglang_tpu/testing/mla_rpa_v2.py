# SPDX-License-Identifier: Apache-2.0
"""Shared inputs, NumPy reference and cache-write check for the TPU MLA v2
ragged-paged-attention kernel tests.

Geometry is DeepSeek V2/V3 MLA: 512 latent + 64 rope, both padded to 128 lanes
by the kernel. Metadata mirrors what the MLA backend feeds the kernel in
production: real (unaligned) ``kv_lens``, a rectangular ``page_indices`` table of
``bs * pages_per_seq`` entries and ``cu_kv_lens[i] = i * pages_per_seq *
page_size``. Page ids come from a shuffled pool, so an addressing bug that a
contiguous layout would hide still shows up.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.mla_ragged_paged_attention_v2 import (
    align_to,
    get_dtype_packing,
    get_kv_cache_shape,
)

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
# DeepSeek-V3 at attention_tp=16 and Ling-2.6 at attention_tp=8.
NUM_Q_HEADS = 8
SM_SCALE = float((QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5)

# The re-vendored kernel dropped its module-level DEFAULT_MASK_VALUE (the
# default is now derived from s_dtype); the tests pin the mask explicitly so
# the fp32 numpy reference and the kernel always agree on it.
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)


def _cdiv(a, b):
    return (a + b - 1) // b


def cache_slots(seq_pages, positions, page_size, kv_packing):
    """Map a sequence's token positions to ``(page, row, col)`` cache slots."""
    return (
        seq_pages[positions // page_size],
        (positions % page_size) // kv_packing,
        (positions % page_size) % kv_packing,
    )


def build_case(lens, *, page_size, num_heads, dtype, seed, q_dtype=None):
    """Build kernel inputs + the expected post-call cache for one batch.

    Args:
      lens: ``[(q_len, kv_len)]`` per sequence. ``q_len`` is the number of new
        tokens (the kernel's Q rows); ``kv_len`` is the total KV length
        *including* those new tokens, i.e. prefix length is ``kv_len - q_len``.
      page_size: tokens per page.
      num_heads: per-shard q head count.
      dtype: latent KV dtype (and Q dtype unless ``q_dtype`` is given).
      seed: RNG seed.
      q_dtype: Q dtype when it differs from the KV cache dtype (fp8 KV).

    Eight pages past what the batch needs are left unallocated, and the unused
    tail of each sequence's ``page_indices`` row points back at a page that
    sequence already owns, so neither a stray write nor a stray fetch can
    involve another sequence's KV without the assertions noticing.

    Returns a SimpleNamespace holding the kernel arguments, the logical
    per-sequence tensors the reference consumes, ``cache_before`` (prefix tokens
    only) and ``cache_after`` — the cache the kernel is expected to produce.
    """
    np_dtype = jnp.dtype(dtype)
    q_np_dtype = np_dtype if q_dtype is None else jnp.dtype(q_dtype)
    kv_packing = get_dtype_packing(dtype)
    nope_dim = align_to(KV_LORA_RANK, 128)
    rope_dim = align_to(QK_ROPE_HEAD_DIM, 128)

    rng = np.random.default_rng(seed)
    bs = len(lens)
    q_lens = np.array([q for q, _ in lens], dtype=np.int32)
    kv_lens = np.array([kv for _, kv in lens], dtype=np.int32)
    assert np.all(q_lens >= 1) and np.all(q_lens <= kv_lens)

    pages_per_seq_list = [_cdiv(int(kv), page_size) for kv in kv_lens]
    pages_per_seq = max(pages_per_seq_list)
    total_num_pages = sum(pages_per_seq_list) + 8
    page_pool = rng.permutation(total_num_pages).astype(np.int32)

    cache_shape = get_kv_cache_shape(
        total_num_pages, page_size, nope_dim + rope_dim, dtype
    )
    cache_before = np.zeros(cache_shape, dtype=np_dtype)
    cache_after = np.zeros(cache_shape, dtype=np_dtype)

    page_indices = np.zeros((bs * pages_per_seq,), dtype=np.int32)
    per_seq = []
    seq_pages_list = []
    cursor = 0
    for i, (q_len, kv_len) in enumerate(lens):
        num_pages = pages_per_seq_list[i]
        seq_pages = page_pool[cursor : cursor + num_pages]
        cursor += num_pages
        seq_pages_list.append(seq_pages)

        row = page_indices[i * pages_per_seq : (i + 1) * pages_per_seq]
        row[:num_pages] = seq_pages
        row[num_pages:] = seq_pages[-1]

        kv_c = rng.standard_normal((kv_len, KV_LORA_RANK)).astype(np_dtype)
        k_pe = rng.standard_normal((kv_len, QK_ROPE_HEAD_DIM)).astype(np_dtype)
        ql_nope = rng.standard_normal((q_len, num_heads, KV_LORA_RANK)).astype(
            q_np_dtype
        )
        q_pe = rng.standard_normal((q_len, num_heads, QK_ROPE_HEAD_DIM)).astype(
            q_np_dtype
        )
        per_seq.append(
            SimpleNamespace(ql_nope=ql_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe)
        )

        for cache, num_tokens in (
            (cache_before, kv_len - q_len),
            (cache_after, kv_len),
        ):
            if num_tokens == 0:
                continue
            slots = cache_slots(seq_pages, np.arange(num_tokens), page_size, kv_packing)
            cache[(*slots, slice(None, KV_LORA_RANK))] = kv_c[:num_tokens]
            cache[(*slots, slice(nope_dim, nope_dim + QK_ROPE_HEAD_DIM))] = k_pe[
                :num_tokens
            ]

    cu_q_lens = np.zeros(bs + 1, dtype=np.int32)
    cu_q_lens[1:] = np.cumsum(q_lens)
    cu_kv_lens = np.arange(bs + 1, dtype=np.int32) * np.int32(pages_per_seq * page_size)

    return SimpleNamespace(
        lens=lens,
        per_seq=per_seq,
        ql_nope=np.concatenate([s.ql_nope for s in per_seq], axis=0),
        q_pe=np.concatenate([s.q_pe for s in per_seq], axis=0),
        new_kv_c=np.concatenate(
            [s.kv_c[kv - q :] for s, (q, kv) in zip(per_seq, lens)], axis=0
        ),
        new_k_pe=np.concatenate(
            [s.k_pe[kv - q :] for s, (q, kv) in zip(per_seq, lens)], axis=0
        ),
        cache_before=cache_before,
        cache_after=cache_after,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        nope_dim=nope_dim,
        np_dtype=np_dtype,
        page_size=page_size,
        kv_packing=kv_packing,
        seq_pages=seq_pages_list,
    )


def ref_attention(case, *, sm_scale, sliding_window, soft_cap, mask_value):
    """Reference MQA attention over the logical per-sequence latent K/V.

    Ported from sglang-jax's ``ref_mla_ragged_paged_attention`` (v1), keeping its
    operation order — scale, then soft-cap, then mask, then softmax, then a
    KV-dtype-rounded probability matmul — so the upstream tolerances carry over.
    Evaluated in float32 NumPy over contiguous per-sequence tensors: it never
    touches a page table, so it cannot repeat a cache-addressing mistake made by
    the kernel.

    The kernel pads the latent dims to 128 lanes with zeros on both Q and K, so
    contracting the unpadded 576 dims here is exactly equivalent.
    """
    outputs = []
    for (q_len, kv_len), seq in zip(case.lens, case.per_seq):
        q = np.concatenate(
            [seq.ql_nope.astype(np.float32), seq.q_pe.astype(np.float32)], axis=-1
        )
        k = np.concatenate(
            [seq.kv_c.astype(np.float32), seq.k_pe.astype(np.float32)], axis=-1
        )
        v = seq.kv_c.astype(np.float32)

        attn = np.einsum("qnh,kh->nqk", q, k, optimize=True) * sm_scale
        if soft_cap is not None:
            attn = soft_cap * np.tanh(attn / soft_cap)

        q_span = (kv_len - q_len) + np.arange(q_len, dtype=np.int32)[None, :, None]
        kv_span = np.arange(kv_len, dtype=np.int32)[None, None, :]
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = np.logical_or(mask, (q_span - sliding_window) >= kv_span)
        attn = np.where(mask, np.float32(mask_value), attn)

        attn = attn - attn.max(axis=-1, keepdims=True)
        probs = np.exp(attn)
        probs /= probs.sum(axis=-1, keepdims=True)
        probs = probs.astype(case.np_dtype).astype(np.float32)

        out = np.einsum("nqk,kl->qnl", probs, v, optimize=True)
        outputs.append(out.astype(case.np_dtype))
    return np.concatenate(outputs, axis=0)


def assert_cache_write(case, updated_cache):
    """The kernel owns the latent cache write; check every readable slot.

    Two slot classes are excluded from the comparison, both by design:

    * The rope segment is padded from ``qk_rope_head_dim`` (64) up to 128
      lanes; only the real lanes are compared.
    * The kernel merges new KV into the cache at 32-bit *word* granularity
      (``kv_packing`` tokens per word). For the word holding the last new
      token, lanes past ``kv_len`` are filled from the new-KV staging buffer
      instead of the cache, so they can hold the next sequence's latent row.
      Those positions are beyond every reader's ``kv_len`` and are re-merged
      correctly the next time that slot is written (the fetch then covers the
      word), so they are allowed to be dirty — but *only* they are: everything
      else, including other sequences' pages and unallocated pages, must match
      exactly.
    """
    got = np.asarray(updated_cache)
    readable = ~trailing_word_slots(case)
    nope, rope_end = KV_LORA_RANK, case.nope_dim + QK_ROPE_HEAD_DIM
    for name, sl in (
        ("kv_c (nope)", np.s_[..., :nope]),
        ("k_pe (rope)", np.s_[..., case.nope_dim : rope_end]),
    ):
        np.testing.assert_array_equal(
            got[sl][readable].astype(np.float32),
            case.cache_after[sl][readable].astype(np.float32),
            err_msg=f"kernel KV-cache write mismatch in the {name} segment",
        )


def trailing_word_slots(case):
    """Slots in ``[kv_len, align_to(kv_len, kv_packing))`` of each sequence."""
    mask = np.zeros(case.cache_after.shape[:3], dtype=bool)
    for seq_pages, kv_len in zip(case.seq_pages, case.kv_lens):
        positions = np.arange(int(kv_len), align_to(int(kv_len), case.kv_packing))
        if positions.size:
            mask[cache_slots(seq_pages, positions, case.page_size, case.kv_packing)] = (
                True
            )
    return mask
