"""One routed expert (one lane's channels) on resident VMEM weights vs a numpy
dequant of the real Kimi-K3 checkpoint: gate/up must match bf16-rounded
references (rare 1-ulp ties allowed), the down partial an f64 reference."""

import os
import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

_CKPT = os.environ.get("K3_CKPT", "/weights/Kimi-K3")

try:
    import jax
    import jax.numpy as jnp

    _TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover
    _TPU = False


def _ulp_diff(a, b):
    ai = np.asarray(a, ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
    bi = np.asarray(b, ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
    return np.abs(ai - bi)


@unittest.skipUnless(_TPU and os.path.isdir(_CKPT), "needs a TPU and the K3 checkpoint")
class TestExpertMath(CustomTestCase):
    def test_one_expert_all_lanes(self):
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_megakernel.config import (
            K3Dims,
            MoeLayout,
            Topology,
        )
        from sglang_tpu.k3_megakernel.kernel.stages.expert import (
            down_dot,
            gate_up_dot,
            situ_and_mul,
        )
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            _unpack_fp4,
            load_moe_layer,
        )

        dims = K3Dims()
        layout = MoeLayout(dims, Topology(1, 8))
        ckpt = Checkpoint(_CKPT)
        expert, layer, rows = 37, 1, 1
        w = load_moe_layer(ckpt, layer, layout, ranks=range(8), experts=[expert])
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((rows, dims.latent)) * 0.5).astype(ml_dtypes.bfloat16)

        def kernel(x_ref, gu_ref, gus_ref, dn_ref, dns_ref, gu_out, a_out, dn_out):
            gu = gate_up_dot(x_ref[...], gu_ref, gus_ref)
            gu_out[...] = gu
            a = situ_and_mul(gu, beta=dims.situ_beta, linear_beta=dims.situ_linear_beta)
            a_out[...] = a
            dn_out[...] = down_dot(a, dn_ref, dns_ref)

        li = layout.lane_inter
        vm = pl.BlockSpec(memory_space=pltpu.VMEM)
        call = jax.jit(
            pl.pallas_call(
                kernel,
                out_shape=(
                    jax.ShapeDtypeStruct((rows, 2 * li), jnp.bfloat16),
                    jax.ShapeDtypeStruct((rows, li), jnp.bfloat16),
                    jax.ShapeDtypeStruct((rows, dims.latent), jnp.float32),
                ),
                in_specs=[vm] * 5,
                out_specs=(vm, vm, vm),
                compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
            )
        )

        # Reference: dequantized checkpoint in f64.
        pre = f"layers.{layer}.block_sparse_moe.experts.{expert}."

        def deq(name):
            p = np.asarray(ckpt.get(pre + name + ".weight_packed"))
            s = np.asarray(ckpt.get(pre + name + ".weight_scale")).astype(np.int32)
            v = _unpack_fp4(p).astype(np.float64)
            return (
                v.reshape(v.shape[0], -1, 32) * np.exp2(s - 127.0)[:, :, None]
            ).reshape(v.shape)

        w1, w3, w2 = deq("w1"), deq("w3"), deq("w2")  # [3072, 3584], [3584, 3072]
        xf = x.astype(np.float64)
        total = np.zeros((rows, dims.latent))
        for lane in range(8):
            j = slice(lane * li, (lane + 1) * li)
            gu, a, dn = call(
                jnp.asarray(x),
                jnp.asarray(w["expert_gu"][lane, expert]),
                jnp.asarray(w["expert_gu_scale"][lane, expert]),
                jnp.asarray(w["expert_down"][lane, expert]),
                jnp.asarray(w["expert_down_scale"][lane, expert]),
            )
            g_ref = (xf @ w1[j].T).astype(ml_dtypes.bfloat16)
            u_ref = (xf @ w3[j].T).astype(ml_dtypes.bfloat16)
            gu = np.asarray(gu)
            ulp = np.concatenate(
                [_ulp_diff(gu[:, :li], g_ref), _ulp_diff(gu[:, li:], u_ref)], 1
            )
            self.assertLessEqual(ulp.max(), 1, f"lane {lane}")
            self.assertLess((ulp > 0).mean(), 0.01, f"lane {lane}")
            # Down partial from the kernel's own activation: f32 accumulation only.
            a = np.asarray(a).astype(np.float64)
            dn_ref = a @ w2[:, j].T
            dn = np.asarray(dn)
            rel = np.abs(dn - dn_ref).max() / np.abs(dn_ref).max()
            self.assertLess(rel, 1e-5, f"lane {lane}")
            total += dn
            print(
                f"lane {lane}: gate/up ulp>0 frac {(ulp > 0).mean():.4f}, down rel {rel:.2e}",
                flush=True,
            )


@unittest.skipUnless(_TPU, "needs a TPU")
class TestPairedDown(CustomTestCase):
    def test_pair_equals_two_singles(self):
        """down_dot_pair must be bit-identical to two down_dot calls (the MoE's pair_down option)."""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_megakernel.kernel.stages.expert import (
            down_dot,
            down_dot_pair,
        )

        c, n = 384, 3584  # one lane's channels at 8 lanes, latent width
        rng = np.random.default_rng(1)
        packed = rng.integers(0, 256, (2, c // 2, n), dtype=np.uint8)
        exps = rng.integers(118, 128, (2, c // 32, n), dtype=np.uint8)
        a = (rng.standard_normal((2, 1, c))).astype(ml_dtypes.bfloat16)

        def kernel(a_ref, p_ref, e_ref, pair_out, single_out):
            p0, p1 = down_dot_pair(a_ref[0], a_ref[1], p_ref, e_ref)
            pair_out[0] = p0
            pair_out[1] = p1
            single_out[0] = down_dot(a_ref[0], p_ref.at[0], e_ref.at[0])
            single_out[1] = down_dot(a_ref[1], p_ref.at[1], e_ref.at[1])

        vm = pl.BlockSpec(memory_space=pltpu.VMEM)
        shape = jax.ShapeDtypeStruct((2, 1, n), jnp.float32)
        call = jax.jit(
            pl.pallas_call(
                kernel, out_shape=(shape, shape), in_specs=[vm] * 3, out_specs=(vm, vm)
            )
        )
        pair, single = call(jnp.asarray(a), jnp.asarray(packed), jnp.asarray(exps))
        np.testing.assert_array_equal(np.asarray(pair), np.asarray(single))


if __name__ == "__main__":
    unittest.main()
