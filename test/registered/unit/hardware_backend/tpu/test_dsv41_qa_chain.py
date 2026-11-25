"""The DSV4.1 qa_chain kernel (interpret mode) matches the XLA chain it replaces."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dsv4.rope import cos_sin_rows
from sglang_tpu.models.deepseek_v41.kernels.qa_chain import (
    qa_chain,
    qa_chain_reference,
    supports,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

K, Q_RANK, KV_DIM, ROPE = 5120, 1280, 512, 64
EPS = 1e-20


def _inputs(tokens, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    x = (jax.random.normal(keys[0], (tokens, K)) * 2).astype(jnp.bfloat16)
    w = (jax.random.normal(keys[1], (K, Q_RANK + KV_DIM)) * 100).astype(
        jnp.float8_e4m3fn
    )
    ws = jax.random.uniform(keys[2], (Q_RANK + KV_DIM,), minval=1e-4, maxval=1e-3)
    qn = (1 + 0.1 * jax.random.normal(keys[3], (Q_RANK,))).astype(jnp.bfloat16)
    kvn = (1 + 0.1 * jax.random.normal(keys[4], (KV_DIM,))).astype(jnp.bfloat16)
    max_pos = 4097
    freqs = jax.random.uniform(keys[5], (max_pos, ROPE), minval=-1, maxval=1)
    positions = jnp.asarray(
        np.random.RandomState(seed).randint(0, max_pos, tokens), jnp.int32
    )
    cs = cos_sin_rows(freqs, positions)
    return x, w, ws, qn, kvn, cs


class QaChainTest(unittest.TestCase):
    def test_supports_dsv41_shapes(self):
        for tokens in (6, 16, 48, 80, 128):
            self.assertTrue(supports(tokens, K, Q_RANK + KV_DIM, Q_RANK, ROPE))
        self.assertFalse(supports(129, K, Q_RANK + KV_DIM, Q_RANK, ROPE))

    def test_matches_xla_chain(self):
        for tokens in (6, 16, 48):
            with self.subTest(tokens=tokens):
                args = _inputs(tokens, seed=tokens)
                q, kv = qa_chain(*args, q_rank=Q_RANK, eps=EPS, interpret=True)
                q_ref, kv_ref = qa_chain_reference(*args, q_rank=Q_RANK, eps=EPS)
                self.assertEqual(q.shape, (tokens, Q_RANK))
                self.assertEqual(kv.shape, (tokens, KV_DIM))
                np.testing.assert_array_equal(
                    np.asarray(q, np.float32), np.asarray(q_ref, np.float32)
                )
                np.testing.assert_array_equal(
                    np.asarray(kv, np.float32), np.asarray(kv_ref, np.float32)
                )


if __name__ == "__main__":
    unittest.main()
