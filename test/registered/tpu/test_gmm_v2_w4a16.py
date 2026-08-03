"""gmm_v2 with a checkpoint-packed int4 rhs (W4A16).

Covers the ``rhs_quant_dtype`` path: the rhs arrives already packed as int32
words ([G, K//8, N], the compressed-tensors layout) rather than being packed
in-kernel, and activations stay bf16. Compares against the same weights held
unpacked in bf16, so any drift in the unpack order, the nibble sign flip, or
the per-group scale application lands here.

Direct JAX like the sibling gmm_v2 tests -- no torch custom op and no
``current_platform.init_backend()``, which would put JAX on its CPU export
frontend and leave Pallas without a TPU device kind.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a TPU backend"

# compressed-tensors stores nibbles offset-binary (q+8); pltpu reads two's
# complement. Flipping each nibble's top bit converts: (q+8) ^ 0x8 == int4(q).
_INT4_SIGN_XOR = -2004318072  # 0x88888888 as int32


def _pack_int4_offset_binary(q_i8):
    """[G, K, N] int4 values (-8..7) -> [G, K//8, N] int32, checkpoint layout.

    Eight consecutive K values per word, LSB-first, stored offset-binary so the
    result matches what a compressed-tensors checkpoint ships.
    """
    g, k, n = q_i8.shape
    off = (q_i8 + 8).astype(jnp.uint32) & 0xF  # offset-binary nibbles
    off = off.reshape(g, k // 8, 8, n)
    shifts = (jnp.arange(8, dtype=jnp.uint32) * 4)[None, None, :, None]
    return jnp.bitwise_or.reduce(off << shifts, axis=2).astype(jnp.int32)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestGmmV2W4A16(CustomTestCase):
    # G: experts, M: num_tokens * topk, K: hidden, N: intermediate.
    # K must be a multiple of 8 (nibbles per int32) and of GROUP_SIZE.
    G, M, K, N = 4, 256, 256, 128
    GROUP_SIZE = 32

    def _run(self, dtype):
        from sglang.srt.layers.moe.pallas.gmm_v2 import gmm_v2

        k1, k2 = jax.random.split(jax.random.key(0))
        lhs = (jax.random.normal(k1, (self.M, self.K), jnp.float32) * 0.3).astype(dtype)

        # Quantized weights as the checkpoint holds them: int4 levels + one
        # bf16 scale per (K-group, N).
        q = jax.random.randint(k2, (self.G, self.K, self.N), -8, 8, dtype=jnp.int8)
        nb = self.K // self.GROUP_SIZE
        scale = (
            jnp.abs(jax.random.normal(k2, (self.G, nb, self.N), jnp.float32)) * 0.02
            + 0.01
        ).astype(jnp.bfloat16)

        packed = _pack_int4_offset_binary(q)
        # Load-time relayout, exactly as prepare_tpu_w4a16_moe_weights does.
        packed = jax.lax.bitwise_xor(packed, jnp.int32(_INT4_SIGN_XOR))

        got = gmm_v2(
            lhs=lhs,
            rhs=packed,
            group_sizes=jnp.full((self.G,), self.M // self.G, jnp.int32),
            rhs_scale=scale[:, :, None, :],  # [G, blocks, 1, N]
            rhs_quant_dtype=jnp.int4,
            maybe_quantize_lhs=False,
            preferred_element_type=dtype,
        )

        # Reference: the same weights dequantized to bf16, no packing involved.
        deq = (
            q.astype(jnp.float32)
            * jnp.repeat(scale.astype(jnp.float32), self.GROUP_SIZE, axis=1)
        ).astype(dtype)
        ref = gmm_v2(
            lhs=lhs,
            rhs=deq,
            group_sizes=jnp.full((self.G,), self.M // self.G, jnp.int32),
            preferred_element_type=dtype,
        )

        a = got.astype(jnp.float32)
        b = ref.astype(jnp.float32)
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-9))
        self.assertGreater(
            cos, 0.999, f"packed int4 rhs diverges from the bf16 reference ({dtype})"
        )

    def test_bfloat16(self):
        self._run(jnp.bfloat16)

    def test_float32(self):
        self._run(jnp.float32)


if __name__ == "__main__":
    unittest.main()
