"""Query-blocked sparse MLA prefill vs the per-query sparse MLA kernel.

Selections come from the real pooled indexer (kpool_indexer_step) on random
keys; the reference is sparse_mla_attention_fused (cache write + per-query
kernel), and the q-block kernel reads the caches that step wrote.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_JAX = True
    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_JAX = False
    _HAS_TPU = False

D, KPOOL, PAGE, IH, TOPK, W, QH = 128, 4, 128, 32, 2048, 2176, 8
# bf16 softmax weights: observed max_abs / max|ref| <= 7e-3.
REL_TOL = 1e-2


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestSparseMlaPrefillQblock(CustomTestCase):
    @staticmethod
    def _glm_shape():
        # GLM-5.3-Flash's q-block prefill: NoPE + RoPE query lanes, the
        # recipe's 128-token pages, and the most pages per request the op
        # lets this kernel take.
        from sglang_tpu.attention.pallas import mla_sparse_fused as fused

        return dict(
            q_width=fused.NOPE_DIM + fused.ROPE_DIM,
            page_size=PAGE,
            pages_per_seq=fused.QBLOCK_MAX_PAGES_PER_SEQ,
            dtype=jnp.bfloat16,
        )

    @classmethod
    def setUpClass(cls):
        from sglang_tpu.attention.pallas import dsa_kpool_topk as kp
        from sglang_tpu.attention.pallas.mla_sparse_fused import (
            sparse_mla_attention_fused,
        )
        from sglang_tpu.attention.pallas.mla_sparse_qblock import (
            sparse_mla_prefill_qblock,
        )

        cls.kp = kp
        cls.indexer = staticmethod(
            jax.jit(
                lambda *a: kp.kpool_indexer_step(
                    *a,
                    weight_scale=IH**-0.5 * D**-0.5,
                    topk=TOPK,
                    kpool=KPOOL,
                    out_width=W,
                )
            )
        )
        cls.fused = staticmethod(sparse_mla_attention_fused)
        cls.qblock = staticmethod(sparse_mla_prefill_qblock)

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _f(self, *shape, scale=1.0):
        return jnp.asarray((self.rng.normal(size=shape) * scale).astype(np.float32))

    def _case(self, seqs, num_tokens, max_seqs, ctx, sm_scale, k_scale, heads=QH):
        """seqs: (prefix, q_len) per request, prefixed requests first. Returns the
        q-block arguments and the per-query kernel's output for the last step."""
        kp = self.kp
        pps = -(-ctx // PAGE)
        num_pages = max_seqs * pps + 1
        pi = jnp.asarray(1 + self.rng.permutation(max_seqs * pps), jnp.int32)
        raw = jnp.zeros((num_pages, PAGE, 2 * D), jnp.bfloat16)
        pooled = jnp.zeros(
            (num_pages, PAGE // KPOOL // 4, 4, kp.pooled_record_bytes(D)), jnp.uint8
        )
        cn = jnp.zeros((num_pages, PAGE, 4, 128), jnp.uint8)
        cr = jnp.zeros((num_pages, PAGE // 4, 4, 128), jnp.uint8)
        att = jax.jit(lambda *a: self.fused(*a, sm_scale=sm_scale, k_scale=k_scale))

        def run(T, q_lens, kv_lens, raw, pooled, cn, cr):
            cu = np.zeros(max_seqs + 1, np.int32)
            cu[1 : len(q_lens) + 1] = np.cumsum(q_lens)
            cu[len(q_lens) + 1 :] = cu[len(q_lens)]
            kv = np.zeros(max_seqs, np.int32)
            kv[: len(kv_lens)] = kv_lens
            kv, cu = jnp.asarray(kv), jnp.asarray(cu)
            dist = jnp.asarray([0, 0, len(q_lens)], jnp.int32)
            rows, raw, pooled = self.indexer(
                self._f(T, IH, D).astype(jnp.bfloat16),
                self._f(T, D).astype(jnp.bfloat16),
                self._f(T, D).astype(jnp.bfloat16),
                self._f(T, IH),
                self._f(KPOOL, D),
                raw,
                pooled,
                kv,
                pi,
                cu,
                dist,
            )
            ql = self._f(T, heads, 512).astype(jnp.bfloat16)
            qpe = self._f(T, heads, 64).astype(jnp.bfloat16)
            out, cn, cr = att(
                ql,
                qpe,
                self._f(T, 512).astype(jnp.float8_e4m3fn),
                self._f(T, 64).astype(jnp.float8_e4m3fn),
                cn,
                cr,
                kv,
                pi,
                cu,
                dist,
                rows,
            )
            return (ql, qpe, cn, cr, rows, kv, pi, cu, dist), out, raw, pooled

        prefixed = [p for p, _ in seqs if p > 0]
        if prefixed:
            # The prefixes first, as their own prefill (writes every cache).
            t0 = -(-sum(prefixed) // 128) * 128
            args0, _, raw, pooled = run(t0, prefixed, prefixed, raw, pooled, cn, cr)
            cn, cr = args0[2], args0[3]
        args, ref, _, _ = run(
            num_tokens,
            [q for _, q in seqs],
            [p + q for p, q in seqs],
            raw,
            pooled,
            cn,
            cr,
        )
        return args, ref

    def _check(self, args, ref, sm_scale, k_scale, query_block=256):
        out = self.qblock(
            *args,
            sm_scale=sm_scale,
            k_scale=k_scale,
            kpool=KPOOL,
            query_block=query_block,
        )
        cu, dist = args[7], args[8]
        n = int(cu[dist[2]])
        o = np.asarray(out[:n].astype(jnp.float32))
        r = np.asarray(ref[:n].astype(jnp.float32))
        self.assertFalse(np.isnan(o).any())
        self.assertLessEqual(np.abs(o - r).max() / np.abs(r).max(), REL_TOL)
        # Padding tokens come back as zeros.
        self.assertFalse(np.asarray(out[n:]).any())

    def test_padded_multi_request_batch_with_prefixes(self):
        # Prefixed and fresh requests, one ending past a 2048-token block
        # boundary, a one-token request, and padding rows.
        seqs = [(1024, 2500), (600, 1800), (0, 2049), (0, 1)]
        args, ref = self._case(seqs, 8192, 6, 4096, 0.08, 0.75)
        self._check(args, ref, 0.08, 0.75)

    def test_64_heads_128_query_blocks(self):
        # 64 heads per chip (GLM-5.3-Flash at attention TP1) take 128-query
        # blocks, since 64 x 256 score rows do not fit in VMEM. 8 and 32 heads
        # per chip (TP8, TP2) keep 256.
        from sglang_tpu.attention.pallas.mla_sparse_fused import qblock_query_block

        def query_block(heads):
            return qblock_query_block(heads, **self._glm_shape())

        self.assertEqual(query_block(8), 256)
        self.assertEqual(query_block(32), 256)
        self.assertEqual(query_block(64), 128)
        seqs = [(300, 700), (0, 300)]
        args, ref = self._case(seqs, 1024, 4, 1024, 0.08, 0.75, heads=64)
        self._check(args, ref, 0.08, 0.75, query_block=query_block(64))

    def test_largest_shape_within_vmem_estimate_compiles(self):
        # The op trusts the VMEM estimate, so the most heads it allows at the
        # default query block (35 at 256 queries on tpu7x) must compile.
        from sglang_tpu.attention.pallas import mla_sparse_qblock as mq

        shape = self._glm_shape()
        heads = max(
            h
            for h in range(1, 129)
            if mq.vmem_required_by_sparse_mla(h, mq.QUERY_BLOCK, **shape)
            <= mq.vmem_limit_bytes()
        )
        pps, seqs, tokens = shape["pages_per_seq"], 4, 1024
        sd = jax.ShapeDtypeStruct
        mq.sparse_mla_prefill_qblock.lower(
            sd((tokens, heads, 512), jnp.bfloat16),
            sd((tokens, heads, 64), jnp.bfloat16),
            sd((seqs * pps + 1, PAGE, 4, 128), jnp.uint8),
            sd((seqs * pps + 1, PAGE // 4, 4, 128), jnp.uint8),
            sd((tokens, W), jnp.int32),
            sd((seqs,), jnp.int32),
            sd((seqs * pps,), jnp.int32),
            sd((seqs + 1,), jnp.int32),
            sd((3,), jnp.int32),
            sm_scale=0.08,
            k_scale=0.75,
            kpool=KPOOL,
            query_block=mq.QUERY_BLOCK,
        ).compile()


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestQblockSelectionWords(CustomTestCase):
    """Per-token selection words equal ones built per block entry (runs on CPU)."""

    @staticmethod
    def _per_entry_words(topk_indices, tok, is_row, *, kpool, pages_pad):
        # Reference: the words built from each block entry's own row gather.
        from sglang_tpu.attention.pallas import mla_sparse_qblock as mq

        g = jnp.arange(pages_pad, dtype=jnp.int32)
        first_tok = topk_indices[:, ::kpool][tok]  # [nb, qb, width / kpool]
        sel = jnp.where(is_row[..., None] & (first_tok >= 0), first_tok // kpool, -1)
        page = jnp.where(sel >= 0, sel // mq._WORD_BITS, -1)
        bit = jnp.where(sel >= 0, jnp.left_shift(1, sel % mq._WORD_BITS), 0)
        return jax.lax.reduce(
            jnp.where(page[:, None] == g[None, :, None, None], bit[:, None], 0),
            np.dtype(mq._WORD_DTYPE).type(0),
            jax.lax.bitwise_or,
            (3,),
        )  # [nb, pages_pad, qb]

    def _rows(self, rng, pos, width):
        # A pooled-indexer row: selected pools' tokens (any order), the open
        # pool's tail, then -1.
        pool_len = (pos + 1) // KPOOL
        n_pick = TOPK // KPOOL
        picked = np.arange(pool_len)
        if pool_len > n_pick:
            picked = rng.choice(pool_len, n_pick, replace=False)
        toks = (picked[:, None] * KPOOL + np.arange(KPOOL)).reshape(-1)
        tail = pool_len * KPOOL + np.arange((pos + 1) % KPOOL)
        row = np.full(width, -1, np.int32)
        row[: toks.size] = toks
        row[toks.size : toks.size + tail.size] = tail
        return row

    def test_matches_per_entry_words(self):
        from sglang_tpu.attention.pallas import mla_sparse_qblock as mq

        rng = np.random.default_rng(0)
        # (prefix, q_len) per request: past 512 pools, odd tails, a request
        # straddling block boundaries, and unused request slots.
        seqs = [(3001, 700), (0, 130), (17, 61)]
        num_tokens, max_seqs, ctx, qb = 1024, 8, 10240, 128
        topk = np.full((num_tokens, W), -1, np.int32)
        cu, kv = [0], []
        for prefix, q_len in seqs:
            for i in range(q_len):
                topk[cu[-1] + i] = self._rows(rng, prefix + i, W)
            cu.append(cu[-1] + q_len)
            kv.append(prefix + q_len)
        cu += [cu[-1]] * (max_seqs - len(seqs))
        kv += [0] * (max_seqs - len(seqs))
        cu, kv = jnp.asarray(cu, jnp.int32), jnp.asarray(kv, jnp.int32)
        dist = jnp.asarray([0, 0, len(seqs)], jnp.int32)

        num_blocks = (num_tokens + max_seqs * (qb - 1)) // qb
        _, start, count, _, _, _ = mq._query_blocks(
            cu, kv, dist, query_block=qb, num_blocks=num_blocks, num_tokens=num_tokens
        )
        r = jnp.arange(qb, dtype=jnp.int32)
        is_row = r[None, :] < count[:, None]
        tok = jnp.where(is_row, start[:, None] + r[None, :], 0)
        pages_pad = -(-(-(-ctx // PAGE)) // 8) * 8
        words = [
            np.asarray(
                fn(jnp.asarray(topk), tok, is_row, kpool=KPOOL, pages_pad=pages_pad)
            )
            for fn in (self._per_entry_words, mq._selection_words)
        ]
        self.assertEqual(words[0].shape, (num_blocks, pages_pad, qb))
        self.assertTrue(words[0].any())
        np.testing.assert_array_equal(words[0], words[1])


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestQblockZeroRope(CustomTestCase):
    """read_rope=False matches read_rope=True on a zero rope lane, and never
    reads the rope cache (Pallas interpret mode off TPU, so it also runs on
    CPU)."""

    def test_skip_rope_reads(self):
        from sglang_tpu.attention.pallas.mla_sparse_qblock import (
            sparse_mla_prefill_qblock,
        )

        rng = np.random.default_rng(0)
        max_seqs, pps, heads, qb = 2, 8, 8, 128
        q_lens, prefix = [300, 129], [600, 0]
        num_tokens = 512
        topk = np.full((num_tokens, W), -1, np.int32)
        t = 0
        for p, n in zip(prefix, q_lens):
            for pos in range(p, p + n):
                # Short rows select every closed pool, then the open pool's
                # tail: every position up to the query's own.
                topk[t, : pos + 1] = np.arange(pos + 1)
                t += 1
        cu = jnp.asarray([0, q_lens[0], sum(q_lens)], jnp.int32)
        kv = jnp.asarray([p + n for p, n in zip(prefix, q_lens)], jnp.int32)
        dist = jnp.asarray([0, 0, 2], jnp.int32)
        pi = jnp.asarray(1 + rng.permutation(max_seqs * pps), jnp.int32)
        num_pages = max_seqs * pps + 1
        cn = jnp.asarray(rng.integers(0, 0x70, (num_pages, PAGE, 4, 128)), jnp.uint8)
        cr = jnp.zeros((num_pages, PAGE // 4, 4, 128), jnp.uint8)
        # The read_rope=False arm gets an all-NaN rope cache (0x7F is FP8
        # e4m3fn NaN): with zeros both arms would match even if it read it.
        cr_nan = jnp.full((num_pages, PAGE // 4, 4, 128), 0x7F, jnp.uint8)
        ql = jnp.asarray(rng.normal(size=(num_tokens, heads, 512)), jnp.bfloat16)
        qpe = jnp.zeros((num_tokens, heads, 64), jnp.bfloat16)
        outs = [
            np.asarray(
                sparse_mla_prefill_qblock(
                    ql,
                    qpe,
                    cn,
                    rope_cache,
                    jnp.asarray(topk),
                    kv,
                    pi,
                    cu,
                    dist,
                    sm_scale=0.05,
                    k_scale=0.5,
                    kpool=KPOOL,
                    query_block=qb,
                    interpret=not _HAS_TPU,
                    read_rope=read_rope,
                ).astype(jnp.float32)
            )
            for read_rope, rope_cache in ((True, cr), (False, cr_nan))
        ]
        self.assertFalse(np.isnan(outs[0]).any())
        self.assertTrue(outs[0][: sum(q_lens)].any())
        np.testing.assert_array_equal(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
