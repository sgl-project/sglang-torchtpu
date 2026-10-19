"""gmm_v2 with a checkpoint-packed int4 rhs (W4A16), through the production prep.

End-to-end over the load path: weights start in the compressed-tensors
checkpoint layout ([G, K//8, N] int32 words, offset-binary nibbles),
``prepare_tpu_w4a16_moe_weights`` converts them in place (nibble sign flip,
scale relayout), and the result feeds ``gmm_v2`` via ``rhs_quant_dtype``.
Compares against the same weights dequantized to bf16 with no packing
involved, so any drift in the prep step, the in-kernel unpack order, or the
per-group scale application lands here.

Direct JAX like the sibling gmm_v2 tests -- no ``current_platform
.init_backend()``, which would put JAX on its CPU export frontend and leave
Pallas without a TPU device kind. The prep step itself is torch-on-CPU.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a TPU backend"


def _pack_int4_offset_binary(q_i8):
    """[G, K, N] int4 values (-8..7) -> [G, K//8, N] int32, checkpoint layout.

    Eight consecutive K values per word, LSB-first, stored offset-binary
    (q+8) so the result matches what a compressed-tensors checkpoint ships.
    The sign flip to two's complement is prepare_tpu_w4a16_moe_weights' job.
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

    def _checkpoint_weights(self, key):
        """One weight set as the checkpoint holds it: int4 levels + one bf16
        scale per (K-group, N), plus the packed int32 form."""
        kq, ks = jax.random.split(key)
        q = jax.random.randint(kq, (self.G, self.K, self.N), -8, 8, dtype=jnp.int8)
        nb = self.K // self.GROUP_SIZE
        scale = (
            jnp.abs(jax.random.normal(ks, (self.G, nb, self.N), jnp.float32)) * 0.02
            + 0.01
        ).astype(jnp.bfloat16)
        return q, scale, _pack_int4_offset_binary(q)

    def _prep_layer(self, w13, w2):
        """Run the production load-time prep on a stub layer holding both
        weight pairs exactly as the loader leaves them, and hand back the
        converted (packed, scale) jnp pairs."""
        import numpy as np
        import torch

        from sglang_tpu.layers.quantization.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        def to_torch(x, dtype=None):
            # copy: prep mutates in place, and jnp->np buffers can be read-only
            t = torch.from_numpy(np.array(x, dtype=np.float32 if dtype else None))
            return t.to(dtype) if dtype else t

        layer = torch.nn.Module()
        for name, (_, scale, packed) in (("w13", w13), ("w2", w2)):
            setattr(
                layer,
                f"{name}_weight_packed",
                torch.nn.Parameter(to_torch(packed), requires_grad=False),
            )
            setattr(
                layer,
                f"{name}_weight_scale",
                torch.nn.Parameter(
                    to_torch(scale, torch.bfloat16), requires_grad=False
                ),
            )
        checkpoint_scale_buffers = {
            name: getattr(layer, f"{name}_weight_scale").data_ptr()
            for name in ("w13", "w2")
        }
        prepare_tpu_w4a16_moe_weights(layer, self.GROUP_SIZE)

        for name in ("w13", "w2"):
            self.assertNotEqual(
                getattr(layer, f"{name}_weight_scale").data_ptr(),
                checkpoint_scale_buffers[name],
                f"{name} scale still aliases the checkpoint buffer",
            )

        out = []
        for name in ("w13", "w2"):
            packed = jnp.asarray(getattr(layer, f"{name}_weight_packed").numpy())
            scale = jnp.asarray(
                getattr(layer, f"{name}_weight_scale").float().numpy()
            ).astype(jnp.bfloat16)  # exact: bf16 -> f32 -> bf16
            out.append((packed, scale))
        return out

    def _run(self, dtype, dequantize_rhs=False, scale_dtype=None):
        from sglang_tpu.layers.moe.pallas.gmm_v2 import gmm_v2

        klhs, k13, k2 = jax.random.split(jax.random.key(0), 3)
        lhs = (jax.random.normal(klhs, (self.M, self.K), jnp.float32) * 0.3).astype(
            dtype
        )
        w13 = self._checkpoint_weights(k13)
        w2 = self._checkpoint_weights(k2)
        prepped = self._prep_layer(w13, w2)
        group_sizes = jnp.full((self.G,), self.M // self.G, jnp.int32)

        for (q, scale, _), (packed, prepped_scale) in zip((w13, w2), prepped):
            got = gmm_v2(
                lhs=lhs,
                rhs=packed,
                group_sizes=group_sizes,
                rhs_scale=prepped_scale.astype(scale_dtype or prepped_scale.dtype),
                rhs_quant_dtype=jnp.int4,
                maybe_quantize_lhs=False,
                dequantize_rhs=dequantize_rhs,
                preferred_element_type=dtype,
            )

            # Reference: the same weights dequantized to bf16, no packing.
            deq = (
                q.astype(jnp.float32)
                * jnp.repeat(scale.astype(jnp.float32), self.GROUP_SIZE, axis=1)
            ).astype(dtype)
            ref = gmm_v2(
                lhs=lhs,
                rhs=deq,
                group_sizes=group_sizes,
                preferred_element_type=dtype,
            )

            diff = jnp.max(jnp.abs(got.astype(jnp.float32) - ref.astype(jnp.float32)))
            scale_ = jnp.max(jnp.abs(ref.astype(jnp.float32))) + 1e-9
            self.assertLess(
                float(diff / scale_),
                0.02,
                f"prepped int4 rhs diverges from the bf16 reference ({dtype})",
            )

    def test_bfloat16(self):
        self._run(jnp.bfloat16)

    def test_bfloat16_vmem_dequantization(self):
        self._run(jnp.bfloat16, dequantize_rhs=True)

    def test_bfloat16_vmem_float32_scales(self):
        # The production loader stores FP32 scales for zero-stride VMEM loads.
        self._run(jnp.bfloat16, dequantize_rhs=True, scale_dtype=jnp.float32)

    def test_vmem_rejects_partial_scale_group(self):
        from sglang_tpu.layers.moe.pallas.gmm_v2 import TileSizes, gmm_v2

        with self.assertRaisesRegex(ValueError, "multiple of the INT4 group size"):
            gmm_v2(
                lhs=jnp.ones((16, 512), jnp.bfloat16),
                rhs=jnp.zeros((1, 64, 128), jnp.int32),
                group_sizes=jnp.array([16], jnp.int32),
                rhs_scale=jnp.ones((1, 1, 1, 128), jnp.float32),
                rhs_quant_dtype=jnp.int4,
                dequantize_rhs=True,
                tile_info=TileSizes(16, 128, 128),
            )

    def test_float32(self):
        self._run(jnp.float32)

    def test_kimi_gate_up_width(self):
        # Kimi prefill's tile and 32-wide groups; a packed-only VMEM estimate fails.
        self.G, self.M, self.K, self.N = 2, 128, 7168, 4096
        self._run(jnp.bfloat16)


if __name__ == "__main__":
    unittest.main()
