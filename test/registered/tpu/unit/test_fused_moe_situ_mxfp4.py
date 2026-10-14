"""Pallas fused MoE on packed MXFP4 with bf16 activations and SiTU."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

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

_SKIP = "requires a TPU backend"

# Kimi-K3 activation_situ_beta / activation_situ_linear_beta.
_SITU_PARAMS = (4.0, 25.0)


def _situ_ref(gate, up, beta, linear_beta):
    gate_act = beta * np.tanh(gate / beta) / (1.0 + np.exp(-gate))
    return gate_act * linear_beta * np.tanh(up / linear_beta)


def _moe_ref(hidden, w1, w2, topk_weights, topk_ids):
    """Dense float32 SiTU MoE over [g, k, 2n] / [g, n, k] weights."""
    hs = np.asarray(hidden, np.float32)
    w1 = np.asarray(w1.astype(jnp.float32))
    w2 = np.asarray(w2.astype(jnp.float32))
    n = w1.shape[-1] // 2
    out = np.zeros((hs.shape[0], w2.shape[-1]), np.float32)
    for t in range(hs.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            gmm1 = hs[t, : w1.shape[1]] @ w1[e]
            gate, up = gmm1[:n], gmm1[n:]
            act = _situ_ref(gate, up, *_SITU_PARAMS)
            out[t] += float(topk_weights[t, j]) * (act @ w2[e])
    return out[:, : hs.shape[1]]


def _routing(m, g, topk, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(g, size=topk, replace=False) for _ in range(m)])
    w = rng.random((m, topk)).astype(np.float32)
    return ids.astype(np.int32), w / w.sum(-1, keepdims=True)


def _run_fused(hidden, w1, w2, w1_scale, w2_scale, ids, w, activation):
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    out = _fused_moe_jax(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weights=jnp.asarray(w, jnp.bfloat16),
        topk_ids=jnp.asarray(ids),
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        topk=ids.shape[1],
        activation_name=activation,
    )
    return np.asarray(jax.block_until_ready(out).astype(jnp.float32))


def _assert_close(test, out, ref, rtol):
    err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9)
    test.assertTrue(np.all(np.isfinite(out)))
    test.assertLess(float(err), rtol)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEMxfp4Situ(CustomTestCase):
    # INTER=96 pads each gate/up half to 128 and w2's K by one 32-block.
    E, HIDDEN, INTER, M, TOPK = 8, 256, 96, 32, 4

    def _hf_mxfp4(self, seed):
        g = torch.Generator().manual_seed(seed)
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
        # Scales up to 2^3 push gate/up past beta and linear_beta, so SiTU's
        # parameters change the output.
        e8 = lambda *s: torch.randint(124, 131, s, dtype=torch.uint8, generator=g)
        E, H, N = self.E, self.HIDDEN, self.INTER
        return (
            u8(E, 2 * N, H // 2),
            e8(E, 2 * N, H // 32),
            torch.zeros(E, 2 * N, dtype=torch.bfloat16),
            u8(E, H, N // 2),
            e8(E, H, N // 32),
            torch.zeros(E, H, dtype=torch.bfloat16),
        )

    def test_packed_mxfp4_bf16_act_matches_dequantized_reference(self):
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_bf16,
            process_mxfp4_moe_weights_native,
        )

        hf = self._hf_mxfp4(seed=0)
        w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
            *hf, w13_interleaved=False
        )
        w13_bf16, _, w2_bf16, _ = process_mxfp4_moe_weights_bf16(
            *hf, w13_interleaved=False
        )
        to_jax = lambda t: jnp.asarray(t.numpy())
        bf16_to_jax = lambda t: jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)

        rng = np.random.default_rng(1)
        hidden = rng.standard_normal((self.M, self.HIDDEN)).astype(np.float32)
        hidden = jnp.asarray(hidden).astype(jnp.bfloat16)
        ids, w = _routing(self.M, self.E, self.TOPK, seed=2)
        run = lambda activation: _run_fused(
            hidden,
            to_jax(w13),
            to_jax(w2),
            to_jax(w13_s),
            to_jax(w2_s),
            ids,
            w,
            activation,
        )
        out = run("situ:4.0:25.0")
        ref = _moe_ref(
            np.asarray(hidden.astype(jnp.float32)),
            bf16_to_jax(w13_bf16),
            bf16_to_jax(w2_bf16),
            w,
            ids,
        )
        _assert_close(self, out, ref, 0.03)
        # beta=1 without the linear clip must not also match: the params reach SiTU.
        wrong = run("situ:1.0:none")
        err = np.max(np.abs(wrong - ref)) / np.max(np.abs(ref))
        self.assertGreater(float(err), 0.1)


if __name__ == "__main__":
    unittest.main()
