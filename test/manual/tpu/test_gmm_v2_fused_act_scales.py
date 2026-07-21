"""Tests for gmm_v2."""

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a TPU backend"

_FMAX = 448.0  # float8_e4m3fn max


def _quantize(w_f32, quant_mode):
    """Quantize [g, k, 2n] weights. Returns (rhs, rhs_scale-or-None)."""
    g, k, n2 = w_f32.shape
    if quant_mode == "bf16":
        return w_f32.astype(jnp.bfloat16), None
    if quant_mode == "channelwise":
        s = jnp.clip(jnp.abs(w_f32).max(axis=1, keepdims=True) / _FMAX, 1e-12)
        wq = jnp.clip(w_f32 / s, -_FMAX, _FMAX).astype(jnp.float8_e4m3fn)
        return wq, s[:, :, None, :].astype(jnp.bfloat16)  # [g, 1, 1, 2n]
    if quant_mode == "block128":
        kb = k // 128
        s = jnp.clip(jnp.abs(w_f32.reshape(g, kb, 128, n2)).max(axis=2) / _FMAX, 1e-12)
        wq = jnp.clip(w_f32 / jnp.repeat(s, 128, axis=1), -_FMAX, _FMAX).astype(
            jnp.float8_e4m3fn
        )
        return wq, s[:, :, None, :].astype(jnp.bfloat16)  # [g, kb, 1, 2n]
    raise ValueError(quant_mode)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedActScales(CustomTestCase):
    # G: number of groups (experts), M: num_tokens * topk, K: hidden size,
    # N2: 2 * intermediate (gate || up).
    G, M, K, N2 = 4, 512, 256, 512

    def _fused_vs_unfused(self, quant_mode):
        from sglang.srt.layers.moe.pallas.gmm_v2 import apply_act_fn, gmm_v2

        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        lhs = (jax.random.normal(k1, (self.M, self.K), jnp.float32) * 0.3).astype(
            jnp.bfloat16
        )
        w_f32 = jax.random.normal(k2, (self.G, self.K, self.N2), jnp.float32) * 0.05
        group_sizes = jnp.full((self.G,), self.M // self.G, jnp.int32)
        rhs, scale = _quantize(w_f32, quant_mode)

        fused = gmm_v2(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            rhs_scale=scale,
            fuse_act="silu",
            preferred_element_type=jnp.bfloat16,
        )
        plain = gmm_v2(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            rhs_scale=scale,
            preferred_element_type=jnp.bfloat16,
        )
        ref = apply_act_fn(
            plain.astype(jnp.float32), plain.shape[-1] // 2, "silu"
        ).astype(jnp.bfloat16)

        diff = jnp.max(jnp.abs(fused.astype(jnp.float32) - ref.astype(jnp.float32)))
        scale_ = jnp.max(jnp.abs(ref.astype(jnp.float32))) + 1e-9
        self.assertLess(
            float(diff / scale_),
            0.02,
            f"fused act diverges from unfused reference for {quant_mode} scales",
        )

    def test_fused_act_bf16(self):
        self._fused_vs_unfused("bf16")

    def test_fused_act_block128_scales(self):
        self._fused_vs_unfused("block128")

    def test_fused_act_channelwise_scales(self):
        self._fused_vs_unfused("channelwise")

    def test_misaligned_gate_up_boundary_asserts(self):
        from sglang.srt.layers.moe.pallas.gmm_v2 import gmm_v2

        g, m, k = 2, 128, 256
        n2 = 384  # per-projection n = 192; 192 % 128 != 0
        lhs = jnp.ones((m, k), jnp.bfloat16)
        wq = jnp.ones((g, k, n2), jnp.float8_e4m3fn)
        scale = jnp.ones((g, 1, 1, n2 // 128), jnp.bfloat16)  # scale_block_n=128
        gs = jnp.full((g,), m // g, jnp.int32)
        with self.assertRaises(AssertionError):
            gmm_v2(
                lhs=lhs,
                rhs=wq,
                group_sizes=gs,
                rhs_scale=scale,
                fuse_act="silu",
                preferred_element_type=jnp.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
