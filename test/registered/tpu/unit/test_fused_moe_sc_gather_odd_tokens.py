"""Pallas fused MoE: SparseCore permute gather at an odd token count (top-16)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=50, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

E, TOPK, N = 32, 16, 256


def _bf16_weights(hidden_size: int) -> tuple:
    rng = np.random.default_rng(0)
    w1 = jnp.asarray(rng.standard_normal((E, hidden_size, 2 * N)) / 16, jnp.bfloat16)
    w2 = jnp.asarray(rng.standard_normal((E, N, hidden_size)) / 11, jnp.bfloat16)
    return w1, w2, None, None, "silu"


def _mxfp4_weights(hidden_size: int) -> tuple:
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        process_mxfp4_moe_weights_native,
    )

    g = torch.Generator().manual_seed(0)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    e8 = lambda *s: torch.randint(122, 127, s, dtype=torch.uint8, generator=g)
    zeros = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
        u8(E, 2 * N, hidden_size // 2),
        e8(E, 2 * N, hidden_size // 32),
        zeros(E, 2 * N),
        u8(E, hidden_size, N // 2),
        e8(E, hidden_size, N // 32),
        zeros(E, hidden_size),
        w13_interleaved=False,
    )
    w1, w2, w1_s, w2_s = (jnp.asarray(t.numpy()) for t in (w13, w2, w13_s, w2_s))
    return w1, w2, w1_s, w2_s, "situ:4.0:25.0"


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestFusedMoESparseCoreGatherOddTokens(CustomTestCase):
    def test_odd_token_count_matches_tensorcore_gather(self):
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        # Top-16 needs no token padding for bf16 tiling, so m reaches the
        # gather unpadded.
        for name, make, hidden_size in (
            ("bf16", _bf16_weights, 3584),
            ("mxfp4", _mxfp4_weights, 512),
        ):
            w1, w2, w1_s, w2_s, act = make(hidden_size)
            for m in (1, 3, 5):
                with self.subTest(weights=name, m=m):
                    rng = np.random.default_rng(m)
                    hidden = jnp.asarray(
                        rng.standard_normal((m, hidden_size)), jnp.bfloat16
                    )
                    ids = np.stack(
                        [rng.choice(E, TOPK, replace=False) for _ in range(m)]
                    )
                    w = rng.random((m, TOPK)).astype(np.float32)
                    w = jnp.asarray(w / w.sum(-1, keepdims=True), jnp.bfloat16)
                    outs = []
                    for sc in (0, 1):
                        out = _fused_moe_jax(
                            hidden,
                            w1,
                            w2,
                            w,
                            jnp.asarray(ids, jnp.int32),
                            w1_s,
                            w2_s,
                            None,
                            None,
                            TOPK,
                            act,
                            use_sc_gather=sc,
                        )
                        out = jax.block_until_ready(out).astype(jnp.float32)
                        outs.append(np.asarray(out))
                    tensor_core, sparse_core = outs
                    self.assertTrue(np.all(np.isfinite(sparse_core)))
                    err = np.max(np.abs(sparse_core - tensor_core))
                    self.assertLess(float(err / np.max(np.abs(tensor_core))), 1e-2)


if __name__ == "__main__":
    unittest.main()
