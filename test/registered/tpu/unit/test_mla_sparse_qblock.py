"""Query-blocked sparse MLA prefill vs the per-query sparse MLA kernel.

Selections come from the real pooled indexer (kpool_indexer_step) on random
keys; the reference is sparse_mla_attention_fused (cache write + per-query
kernel), and the q-block kernel reads the caches that step wrote.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

D, KPOOL, PAGE, IH, TOPK, W, QH = 128, 4, 128, 32, 2048, 2176, 8
# bf16 softmax weights: observed max_abs / max|ref| <= 7e-3 across these cases.
REL_TOL = 1e-2


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestSparseMlaPrefillQblock(CustomTestCase):
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

    def _case(self, seqs, num_tokens, max_seqs, ctx, sm_scale, k_scale):
        """seqs: (prefix, q_len) per request, prefixed requests first. Returns the
        q-block arguments and the per-query kernel's output for the last step."""
        kp = self.kp
        pps = -(-ctx // PAGE)
        num_pages = max_seqs * pps + 1
        pi = jnp.asarray(1 + self.rng.permutation(max_seqs * pps), jnp.int32)
        raw = jnp.zeros((num_pages, PAGE, 2 * D), jnp.bfloat16)
        pooled = jnp.zeros(
            (num_pages, PAGE // 4, 4, kp.pooled_record_bytes(D)), jnp.uint8
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
            ql = self._f(T, QH, 512).astype(jnp.bfloat16)
            qpe = self._f(T, QH, 64).astype(jnp.bfloat16)
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

    def _check(self, args, ref, sm_scale, k_scale):
        out = self.qblock(*args, sm_scale=sm_scale, k_scale=k_scale, kpool=KPOOL)
        cu, dist = args[7], args[8]
        n = int(cu[dist[2]])
        o = np.asarray(out[:n].astype(jnp.float32))
        r = np.asarray(ref[:n].astype(jnp.float32))
        self.assertFalse(np.isnan(o).any())
        self.assertLessEqual(np.abs(o - r).max() / np.abs(r).max(), REL_TOL)
        # Padding tokens come back as zeros.
        self.assertFalse(np.asarray(out[n:]).any())

    def test_single_8k_prefill(self):
        args, ref = self._case([(0, 8192)], 8192, 1, 10240, 0.1, 1.0)
        self._check(args, ref, 0.1, 1.0)

    def test_padded_multi_request_batch_with_prefixes(self):
        seqs = [(1024, 2500), (600, 1800), (0, 2900), (0, 1)]
        args, ref = self._case(seqs, 8192, 6, 4096, 0.08, 0.75)
        self._check(args, ref, 0.08, 0.75)

    def test_block_boundary_lengths(self):
        for n in (2047, 2049):
            with self.subTest(n=n):
                t = -(-n // 128) * 128
                args, ref = self._case([(0, n)], t, 1, 10240, 0.1, 1.0)
                self._check(args, ref, 0.1, 1.0)


if __name__ == "__main__":
    unittest.main()
