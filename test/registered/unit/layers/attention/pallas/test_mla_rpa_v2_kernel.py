# SPDX-License-Identifier: Apache-2.0
"""Numerical-accuracy tests for the vendored MLA v2 ragged-paged-attention kernel.

Ported from sglang-jax ``python/sgl_jax/test/test_mla_attention.py``: the same
shape/flag matrix (page sizes, mixed-extend and decode length lists, sliding
window, logit soft-cap), re-pointed at ``mla_ragged_paged_attention`` itself
instead of sglang-jax's ``MLAAttentionBackend``, since only the kernel is
vendored here. The reference is ported from sglang-jax's v1
``ref_mla_ragged_paged_attention`` (same scale / soft-cap / mask / bf16-probs
ordering) but evaluated in NumPy over the *logical* per-sequence K/V, so it
shares no paging or cache-addressing code with the kernel under test.

What each case checks
---------------------
1. ``o_latent`` matches the reference MQA attention, and
2. the kernel's own KV-cache write (the cache is donated and the kernel owns the
   write — the pool rejects out-of-band writes) lands in exactly the expected
   slots, byte-for-byte, with no collateral damage to prefix tokens, to other
   sequences' pages, or to unallocated pages. The one documented exception is
   the sub-word tail past ``kv_len``; see ``_assert_cache_write``.
3. the ``segments`` the backend passes for this distribution (skipping the
   passes it leaves empty) is bit-identical to ``segments="all"``.

Metadata mirrors what ``TpuPallasRpaV3AttnBackend._build_metadata`` feeds the
kernel in production: real (unaligned) ``kv_lens``, a *rectangular*
``page_indices`` table of ``bs * pages_per_seq`` entries and
``cu_kv_lens[i] = i * pages_per_seq * page_size`` as the page-indices offset.
Page ids are assigned from a shuffled pool rather than sequentially, so an
addressing bug that a contiguous layout would hide still shows up.

``distribution`` follows the MLA backend's override, not RPA v3's: decode is
``[N, N, N]`` and extend is ``[0, 0, N]`` (the kernel's PREFILL slot is dead
code, so extend must route through MIXED). Block-size tuples use that same slot
order, ``(decode, prefill, mixed)``, and carry the kernel's hardcoded fallbacks
so a growing tuned table cannot quietly change what these tests measure.
``_DECODE_BATCH_SIZE`` is what ``TpuPallasMlaAttnBackend`` passes in production.

Geometry is DeepSeek V2/V3 MLA — 512 latent + 64 rope, both padded to 128 lanes
by the kernel — at 8 q heads per shard, which is DeepSeek-V3 at attention_tp=16
and Ling-2.6 at attention_tp=8, the shape the v7 tuned table targets.

Runtime is ~50 s for the 21 cases on one v7x host, dominated by pallas compiles.

Usage:
    python test/registered/unit/layers/attention/pallas/test_mla_rpa_v2_kernel.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

import itertools
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_ragged_paged_attention_v2 import (
    align_to,
    get_dtype_packing,
    get_kv_cache_shape,
    mla_ragged_paged_attention,
    prepare_kv_inputs,
    unsigned_align_to,
)

_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_NUM_Q_HEADS = 8
_SM_SCALE = float((_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM) ** -0.5)

_NUM_KV_PAGES_PER_BLOCK = (3, 1, 1)
_NUM_QUERIES_PER_BLOCK = (1, 1, 16)
_DECODE_BATCH_SIZE = 4

# atol must absorb a few bf16 ulps at the output scale (1 ulp in [2, 4) is
# 0.015625); observed kernel-vs-reference deviations are sparse (<0.005% of
# elements) and hardware-dependent (up to 0.031 on v7x). Upstream tpu-inference
# tests this kernel at atol=0.1/rtol=0.2.
_RTOL, _ATOL = 2e-2, 4e-2

# The re-vendored kernel dropped its module-level DEFAULT_MASK_VALUE (the
# default is now derived from s_dtype); the tests pin the mask explicitly so
# the fp32 numpy reference and the kernel always agree on it.
_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)


def _cdiv(a, b):
    return (a + b - 1) // b


def _cache_slots(seq_pages, positions, page_size, kv_packing):
    """Map a sequence's token positions to ``(page, row, col)`` cache slots."""
    return (
        seq_pages[positions // page_size],
        (positions % page_size) // kv_packing,
        (positions % page_size) % kv_packing,
    )


def _build_case(lens, *, page_size, num_heads, dtype, seed):
    """Build kernel inputs + the expected post-call cache for one batch.

    Args:
      lens: ``[(q_len, kv_len)]`` per sequence. ``q_len`` is the number of new
        tokens (the kernel's Q rows); ``kv_len`` is the total KV length
        *including* those new tokens, i.e. prefix length is ``kv_len - q_len``.
      page_size: tokens per page.
      num_heads: per-shard q head count.
      dtype: latent KV / Q dtype.
      seed: RNG seed.

    Eight pages past what the batch needs are left unallocated, and the unused
    tail of each sequence's ``page_indices`` row points back at a page that
    sequence already owns, so neither a stray write nor a stray fetch can
    involve another sequence's KV without the assertions noticing.

    Returns a SimpleNamespace holding the kernel arguments, the logical
    per-sequence tensors the reference consumes, ``cache_before`` (prefix tokens
    only) and ``cache_after`` — the cache the kernel is expected to produce.
    """
    np_dtype = jnp.dtype(dtype)
    kv_packing = get_dtype_packing(dtype)
    nope_dim = align_to(_KV_LORA_RANK, 128)
    rope_dim = align_to(_QK_ROPE_HEAD_DIM, 128)

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

        kv_c = rng.standard_normal((kv_len, _KV_LORA_RANK)).astype(np_dtype)
        k_pe = rng.standard_normal((kv_len, _QK_ROPE_HEAD_DIM)).astype(np_dtype)
        ql_nope = rng.standard_normal((q_len, num_heads, _KV_LORA_RANK)).astype(
            np_dtype
        )
        q_pe = rng.standard_normal((q_len, num_heads, _QK_ROPE_HEAD_DIM)).astype(
            np_dtype
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
            slots = _cache_slots(
                seq_pages, np.arange(num_tokens), page_size, kv_packing
            )
            cache[(*slots, slice(None, _KV_LORA_RANK))] = kv_c[:num_tokens]
            cache[(*slots, slice(nope_dim, nope_dim + _QK_ROPE_HEAD_DIM))] = k_pe[
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


def _ref_attention(case, *, sm_scale, sliding_window, soft_cap, mask_value):
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


@unittest.skipUnless(
    any(d.platform == "tpu" for d in jax.devices()), "needs a TPU device"
)
class TestMlaRaggedPagedAttentionV2(CustomTestCase):
    """Kernel-level accuracy for ``mla_ragged_paged_attention``.

    ``EXTEND_LENS`` mixes pure prefill (``q_len == kv_len``), extend with a
    prefix and single-token extend, at ``kv_len`` values that both do and do not
    fill their last page. ``DECODE_LENS`` is single-token decodes straddling page
    boundaries (127/128/129, 1023/1024/1025); 9 sequences at
    ``decode_batch_size=4`` exercises the batched pass (8 seqs) and the residual
    DECODE-tail pass (1 seq) in the same call.

    The ``SWA_*`` lists straddle a 128-token window, so both the in-window and
    the dropped ranges are covered: ``sliding_window`` bounds the causal span to
    ``(p - sliding_window, p]`` for a query at absolute position ``p``.
    ``soft_cap`` applies ``s <- soft_cap * tanh(s / soft_cap)`` to the logits
    before masking; 30.0 is the Gemma-2 / DeepSeek-style magnitude.
    """

    EXTEND_LENS = [(1, 128), (125, 125), (123, 522), (1, 511), (256, 512)]

    DECODE_LENS = [
        (1, 119),
        (1, 127),
        (1, 128),
        (1, 129),
        (1, 133),
        (1, 1001),
        (1, 1023),
        (1, 1024),
        (1, 1025),
    ]

    SWA_EXTEND_LENS = [(1, 128), (64, 64), (128, 256), (100, 300), (1, 400)]
    SWA_DECODE_LENS = [(1, 256), (1, 400), (1, 512), (1, 1024)]

    def _run(
        self,
        mode,
        lens,
        *,
        page_size,
        num_heads=_NUM_Q_HEADS,
        sliding_window=None,
        soft_cap=None,
        tuned_blocks=False,
        decode_batch_size=_DECODE_BATCH_SIZE,
        dtype=None,
        s_dtype=None,
        seed=0,
    ):
        assert mode in ("extend", "decode")
        dtype = jnp.bfloat16 if dtype is None else dtype
        case = _build_case(
            lens, page_size=page_size, num_heads=num_heads, dtype=dtype, seed=seed
        )
        bs = len(lens)

        if mode == "decode":
            assert all(q == 1 for q, _ in lens), "decode expects q_len == 1"
            distribution = np.array([bs, bs, bs], dtype=np.int32)
        else:
            distribution = np.array([0, 0, bs], dtype=np.int32)

        block_kwargs = {}
        if not tuned_blocks:
            block_kwargs = dict(
                num_kv_pages_per_block=_NUM_KV_PAGES_PER_BLOCK,
                num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
            )

        def call(segments):
            return mla_ragged_paged_attention(
                jnp.asarray(case.ql_nope),
                jnp.asarray(case.q_pe),
                jnp.asarray(case.new_kv_c),
                jnp.asarray(case.new_k_pe),
                jnp.asarray(case.cache_before),
                jnp.asarray(case.kv_lens),
                jnp.asarray(case.page_indices),
                jnp.asarray(case.cu_q_lens),
                jnp.asarray(case.cu_kv_lens),
                jnp.asarray(distribution),
                sm_scale=_SM_SCALE,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=_MASK_VALUE,
                decode_batch_size=decode_batch_size,
                segments=segments,
                **({} if s_dtype is None else {"s_dtype": s_dtype}),
                **block_kwargs,
            )

        out, updated_cache = call("all")
        jax.block_until_ready((out, updated_cache))
        # Skipping the passes this distribution leaves empty is bit-identical.
        out_seg, cache_seg = call("decode" if mode == "decode" else "mixed")
        np.testing.assert_array_equal(np.asarray(out_seg), np.asarray(out))
        np.testing.assert_array_equal(np.asarray(cache_seg), np.asarray(updated_cache))

        expected = _ref_attention(
            case,
            sm_scale=_SM_SCALE,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=_MASK_VALUE,
        )

        got = np.asarray(out).astype(np.float32)
        exp = expected.astype(np.float32)
        self.assertEqual(got.shape, exp.shape)
        max_diff = float(np.max(np.abs(got - exp))) if got.size else 0.0
        print(
            f"mode={mode} page_size={page_size} heads={num_heads} "
            f"seqs={len(lens)} shape={got.shape} max_diff={max_diff:.6f}"
        )
        np.testing.assert_allclose(
            got,
            exp,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=(
                f"o_latent differs from the reference (mode={mode}, "
                f"page_size={page_size}, max_diff={max_diff})"
            ),
        )

        self._assert_cache_write(case, updated_cache)

    def _assert_cache_write(self, case, updated_cache):
        """The kernel owns the latent cache write; check every readable slot.

        Two slot classes are excluded from the comparison, both by design:

        * The rope segment is padded from ``qk_rope_head_dim`` (64) up to 128
          lanes; only the real lanes are compared.
        * The kernel merges new KV into the cache at 32-bit *word* granularity
          (``kv_packing`` = 2 tokens per word for bf16). For the word holding
          the last new token, lanes past ``kv_len`` are filled from the new-KV
          staging buffer instead of the cache, so they can hold the next
          sequence's latent row. Those positions are beyond every reader's
          ``kv_len`` and are re-merged correctly the next time that slot is
          written (the fetch then covers the word), so they are allowed to be
          dirty — but *only* they are: everything else, including other
          sequences' pages and unallocated pages, must match exactly.
        """
        got = np.asarray(updated_cache)
        readable = ~self._trailing_word_slots(case)
        nope, rope_end = _KV_LORA_RANK, case.nope_dim + _QK_ROPE_HEAD_DIM
        for name, sl in (
            ("kv_c (nope)", np.s_[..., :nope]),
            ("k_pe (rope)", np.s_[..., case.nope_dim : rope_end]),
        ):
            np.testing.assert_array_equal(
                got[sl][readable].astype(np.float32),
                case.cache_after[sl][readable].astype(np.float32),
                err_msg=f"kernel KV-cache write mismatch in the {name} segment",
            )

    @staticmethod
    def _trailing_word_slots(case):
        """Slots in ``[kv_len, align_to(kv_len, kv_packing))`` of each sequence."""
        mask = np.zeros(case.cache_after.shape[:3], dtype=bool)
        for seq_pages, kv_len in zip(case.seq_pages, case.kv_lens):
            positions = np.arange(int(kv_len), align_to(int(kv_len), case.kv_packing))
            if positions.size:
                mask[
                    _cache_slots(seq_pages, positions, case.page_size, case.kv_packing)
                ] = True
        return mask

    def test_extend_page_size_8(self):
        """page_size 8 at bf16 packing 2 -> 4 rows per page, the tightest layout."""
        self._run("extend", self.EXTEND_LENS, page_size=8)

    def test_extend_page_size_16(self):
        self._run("extend", self.EXTEND_LENS, page_size=16)

    def test_extend_page_size_32(self):
        self._run("extend", self.EXTEND_LENS, page_size=32)

    def test_extend_page_size_64(self):
        """64 is the TPU backend's production page size."""
        self._run("extend", self.EXTEND_LENS, page_size=64)

    def test_extend_single_long_seq(self):
        """One pure-prefill sequence spanning many pages and many query blocks."""
        self._run("extend", [(1024, 1024)], page_size=64)

    def test_extend_odd_num_q_heads(self):
        """3 heads is not a multiple of the bf16 q packing (2).

        The kernel pads the head axis and slices the padding back off.
        """
        self._run("extend", self.EXTEND_LENS, page_size=64, num_heads=3)

    def test_extend_six_q_heads(self):
        """6 bf16 heads (Kimi-K3 at TP16) must compile in Mosaic.

        3 packed head rows do not fit the sublane tile; without q/o scratch
        padding the MIXED kernel fails Mosaic lowering.
        """
        self._run("extend", self.EXTEND_LENS, page_size=128, num_heads=6)

    def test_decode_six_q_heads(self):
        """6 bf16 heads through both the batched and the tail decode passes."""
        self._run(
            "decode",
            [(1, 64), (1, 130), (1, 257), (1, 300), (1, 511)],
            page_size=128,
            num_heads=6,
            decode_batch_size=4,
        )

    def test_extend_k3_tp32_tuned_block_sizes(self):
        """Kimi-K3 at TP32: 3 bf16 heads, page 128, tuned v7 MIXED entry."""
        self._run(
            "extend",
            self.EXTEND_LENS,
            page_size=128,
            num_heads=3,
            tuned_blocks=True,
        )

    def test_decode_k3_tp32_tuned_block_sizes(self):
        """Kimi-K3 at TP32 decode, batched + tail passes, tuned v7 entry."""
        self._run(
            "decode",
            self.DECODE_LENS,
            page_size=128,
            num_heads=3,
            tuned_blocks=True,
        )

    def test_decode_k3_tp32_single_seq(self):
        """bs=1 decode: segments="decode" skips the batched pass entirely."""
        self._run("decode", [(1, 8192)], page_size=128, num_heads=3, tuned_blocks=True)

    def test_decode_k3_tp32_tuned_batch_of_8(self):
        """bs 37 (bucket 64): tuned decode batch 8, so 32 batched + 5 tail rows."""
        lens = [(1, 100 + 37 * i) for i in range(37)]
        self._run("decode", lens, page_size=128, num_heads=3, tuned_blocks=True)

    def test_decode_page_size_8(self):
        self._run("decode", self.DECODE_LENS, page_size=8)

    def test_decode_page_size_16(self):
        self._run("decode", self.DECODE_LENS, page_size=16)

    def test_decode_page_size_32(self):
        self._run("decode", self.DECODE_LENS, page_size=32)

    def test_decode_page_size_64(self):
        self._run("decode", self.DECODE_LENS, page_size=64)

    def test_decode_batched_pass_only(self):
        """bs == decode_batch_size: the DECODE-tail pass gets an empty grid."""
        self._run(
            "decode",
            [(1, 64), (1, 128), (1, 192), (1, 256)],
            page_size=64,
            decode_batch_size=4,
        )

    def test_decode_tail_pass_only(self):
        """bs < decode_batch_size: the BATCHED_DECODE grid is empty instead."""
        self._run(
            "decode",
            [(1, 64), (1, 130), (1, 257)],
            page_size=64,
            decode_batch_size=4,
        )

    def test_decode_batch_size_one(self):
        """decode_batch_size=1 (the kernel default): tail pass for all."""
        self._run(
            "decode",
            [(1, 129), (1, 512)],
            page_size=64,
            decode_batch_size=1,
        )

    def test_extend_fp32_scores(self):
        """s_dtype=fp32 anchors correctness independent of the bf16-scores
        default the other tests exercise (upstream #2198 made bf16 the
        default; fp32 removes that rounding term from the comparison)."""
        self._run("extend", self.EXTEND_LENS, page_size=64, s_dtype=jnp.float32)

    def test_decode_fp32_scores(self):
        self._run("decode", self.DECODE_LENS, page_size=64, s_dtype=jnp.float32)

    def test_extend_sliding_window(self):
        self._run("extend", self.SWA_EXTEND_LENS, page_size=16, sliding_window=128)

    def test_decode_sliding_window(self):
        self._run("decode", self.SWA_DECODE_LENS, page_size=16, sliding_window=128)

    def test_extend_soft_cap(self):
        self._run("extend", self.SWA_EXTEND_LENS, page_size=16, soft_cap=30.0)

    def test_decode_soft_cap(self):
        self._run("decode", self.SWA_DECODE_LENS, page_size=16, soft_cap=30.0)

    def test_extend_sliding_window_and_soft_cap(self):
        self._run(
            "extend",
            self.SWA_EXTEND_LENS,
            page_size=16,
            sliding_window=128,
            soft_cap=30.0,
        )

    def test_decode_sliding_window_and_soft_cap(self):
        self._run(
            "decode",
            self.SWA_DECODE_LENS,
            page_size=16,
            sliding_window=128,
            soft_cap=30.0,
        )

    def test_extend_with_tuned_block_sizes(self):
        """Block sizes left unset -> the kernel consults the tuned table.

        ``get_tuned_block_sizes_mla`` falls back to the hardcoded defaults on a
        miss, so this guards the lookup wiring: a wrong slot order or a bad
        tuple unpack fails here, not in numerics.
        """
        self._run("extend", self.EXTEND_LENS, page_size=64, tuned_blocks=True)

    def test_decode_with_tuned_block_sizes(self):
        self._run("decode", self.DECODE_LENS, page_size=64, tuned_blocks=True)


class TestPrepareKvInputs(CustomTestCase):
    """The single-pad KV prep matches the former pad/reshape/pad sequence."""

    @staticmethod
    def _two_pad_reference(kv, kv_packing):
        max_num_tokens, actual_head_dim = kv.shape
        if max_num_tokens % kv_packing != 0:
            pad = kv_packing - (max_num_tokens % kv_packing)
            kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)
        head_dim = unsigned_align_to(actual_head_dim, 128)
        kv = kv.reshape(-1, kv_packing, actual_head_dim)
        return jnp.pad(
            kv, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)), constant_values=0
        )

    def test_bitwise_equal_to_two_pad_reference(self):
        rng = np.random.default_rng(0)
        for dtype, num_tokens, head_dim in itertools.product(
            (jnp.bfloat16, jnp.float8_e4m3fn), (1, 3, 4, 7, 64), (64, 128, 512)
        ):
            with self.subTest(dtype=dtype, tokens=num_tokens, dim=head_dim):
                kv_packing = get_dtype_packing(dtype)
                kv = jnp.asarray(
                    rng.standard_normal((num_tokens, head_dim)), dtype=dtype
                )
                got = prepare_kv_inputs(kv, kv_packing)
                ref = self._two_pad_reference(kv, kv_packing)
                self.assertEqual(got.shape, ref.shape)
                self.assertEqual(got.dtype, ref.dtype)
                np.testing.assert_array_equal(
                    np.asarray(got).view(np.uint8),
                    np.asarray(ref).view(np.uint8),
                )


if __name__ == "__main__":
    unittest.main()
