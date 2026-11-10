"""Query-blocked sparse MLA prefill for plain DSA vs the per-query kernel.

Selections come from the real DSA indexer step (dsa_indexer_step) on random
keys; the reference is sparse_mla_attention_fused (cache write + per-query
kernel), and the q-block kernel reads the caches that step wrote. Also checks
the SparseCore bitmap builder against its XLA reference.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

D, PAGE, IH, TOPK, QH = 128, 128, 32, 2048, 8
# bf16 softmax weights and summation order: observed max_abs / max|ref| <= 5e-3.
REL_TOL = 1e-2


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestSparseMlaPrefillQblockTokens(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.attention.pallas import dsa_index_topk as dit
        from sglang_tpu.attention.pallas import dsa_select_bits as sb
        from sglang_tpu.attention.pallas import mla_sparse_fused as msf

        cls.dit, cls.sb, cls.msf = dit, sb, msf

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _f(self, *shape, dtype=jnp.bfloat16):
        return jnp.asarray(self.rng.normal(size=shape).astype(np.float32)).astype(dtype)

    def _run(self, seqs, num_tokens, max_seqs, ctx, sm_scale=0.08, k_scale=0.75):
        """seqs: (prefix, q_len) per request. Fills the caches at random, runs
        the indexer for this step's rows, then compares the two kernels."""
        dit, msf = self.dit, self.msf
        pps = -(-ctx // PAGE)
        num_pages = max_seqs * pps + 1
        pi = jnp.asarray(1 + self.rng.permutation(max_seqs * pps), jnp.int32)
        ic = self.rng.integers(0, 256, dit.index_key_cache_shape(num_pages, PAGE, D))
        ic = jnp.asarray(ic.astype(np.uint8))
        ic = ic.at[..., :D].set(ic[..., :D] & 0x77).at[..., D].set(127)
        cn = self.rng.integers(0, 256, (num_pages, PAGE, 4, 128)) & 0x77
        cr = self.rng.integers(0, 256, (num_pages, PAGE // 4, 4, 128)) & 0x77
        cn, cr = jnp.asarray(cn.astype(np.uint8)), jnp.asarray(cr.astype(np.uint8))

        q_lens = [q for _, q in seqs]
        cu = np.zeros(max_seqs + 1, np.int32)
        cu[1 : len(seqs) + 1] = np.cumsum(q_lens)
        cu[len(seqs) + 1 :] = cu[len(seqs)]
        kv = np.zeros(max_seqs, np.int32)
        kv[: len(seqs)] = [p + q for p, q in seqs]
        kv, cu = jnp.asarray(kv), jnp.asarray(cu)
        dist = jnp.asarray([0, 0, len(seqs)], jnp.int32)
        T = num_tokens
        rows, _ = jax.jit(
            lambda c: dit.dsa_indexer_step(
                self._f(T, IH, D),
                self._f(T, D),
                self._f(T, IH),
                c,
                kv,
                pi,
                cu,
                dist,
                weight_scale=0.01,
                topk=TOPK,
            )
        )(ic)

        ql, qpe = self._f(T, QH, 512), self._f(T, QH, 64)
        nkv = self._f(T, 512, dtype=jnp.float8_e4m3fn)
        nkpe = self._f(T, 64, dtype=jnp.float8_e4m3fn)
        common = (nkv, nkpe, cn, cr, kv, pi, cu, dist, rows)
        kw = dict(sm_scale=sm_scale, k_scale=k_scale)
        ref, ref_n, ref_r = jax.jit(
            lambda *a: msf.sparse_mla_attention_fused(*a, **kw)
        )(ql, qpe, *common)
        out, out_n, out_r = jax.jit(
            lambda *a: msf.sparse_mla_prefill_fused_tokens(*a, **kw)
        )(ql.transpose(1, 0, 2), qpe.transpose(1, 0, 2), *common)

        # Same cache writes.
        self.assertTrue(bool(jnp.all(ref_n == out_n)) and bool(jnp.all(ref_r == out_r)))
        n = int(cu[len(seqs)])
        o = np.asarray(out.transpose(1, 0, 2)[:n].astype(jnp.float32))
        r = np.asarray(ref[:n].astype(jnp.float32))
        self.assertFalse(np.isnan(o).any())
        self.assertLessEqual(np.abs(o - r).max() / np.abs(r).max(), REL_TOL)
        # Padding tokens come back as zeros.
        self.assertFalse(np.asarray(out[:, n:]).any())

        # The bitmaps the kernel used, against the XLA scatter.
        nw = self.sb.select_bits_num_words(pps * PAGE, PAGE)
        got = self.sb.select_bits(rows, num_words=nw)
        want = self.sb.select_bits_reference(rows, num_words=nw)
        self.assertTrue(bool(jnp.all(got == want)))

    def test_single_long_request(self):
        self._run([(7168, 1024)], 1024, 4, 9216)

    def test_padded_multi_request_batch_with_prefixes(self):
        # Requests sharing tiles, one shorter than top-k, a one-token request,
        # and padding tiles at the end.
        self._run([(5000, 300), (77, 1), (0, 500), (3000, 200)], 1536, 4, 9216)


if __name__ == "__main__":
    unittest.main()
