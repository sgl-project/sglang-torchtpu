"""K3 attention-residual stage math vs the eager sglang goldens (all 13 layers):
attention input (input_layernorm(agg(h))) and MLP input
(post_attention_layernorm(agg(bf16(h + attn)))), bank written on i % 12 == 0."""

import os
import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

_CKPT = os.environ.get("K3_CKPT", "/weights/Kimi-K3")
_GOLDEN = os.environ.get("K3MK_GOLDEN", "")

try:
    import jax
    import jax.numpy as jnp

    _TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover
    _TPU = False

_NB = 8


@unittest.skipUnless(_TPU and os.path.isdir(_GOLDEN), "needs a TPU and K3MK_GOLDEN")
class TestAttnRes(CustomTestCase):
    def test_layers(self):
        import torch
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_megakernel.kernel.stages.attn_res import (
            aggregate,
            bf16,
            rmsnorm,
        )
        from sglang_tpu.k3_megakernel.weights import Checkpoint

        eps = 1e-5
        ckpt = Checkpoint(_CKPT)

        def kernel(
            nvb_ref,
            h_ref,
            attn_ref,
            bank_ref,
            ap,
            an,
            il,
            mp,
            mn,
            pl_ref,
            a_in,
            m_in,
            bank_out,
        ):
            # nvb_ref = [nvb, write]; the write happens after the attention-side mix (old nvb).
            nvb, write = nvb_ref[0], nvb_ref[1]
            h, bank = h_ref[...], bank_ref[...]
            a_in[...] = rmsnorm(
                aggregate(h, bank, nvb, ap[...], an[...], eps), il[...], eps
            )
            slot = jax.lax.broadcasted_iota(jnp.int32, (bank.shape[0], 1, 1), 0)
            bank = jnp.where((write == 1) & (slot == nvb), h[None], bank)
            nvb = nvb + write
            bank_out[...] = bank
            p2 = jnp.where(write == 1, attn_ref[...], bf16(h + attn_ref[...]))
            m_in[...] = rmsnorm(
                aggregate(p2, bank, nvb, mp[...], mn[...], eps), pl_ref[...], eps
            )

        vm = pl.BlockSpec(memory_space=pltpu.VMEM)
        call = jax.jit(
            pl.pallas_call(
                kernel,
                out_shape=(jax.ShapeDtypeStruct((1, 7168), jnp.float32),) * 2
                + (jax.ShapeDtypeStruct((_NB, 1, 7168), jnp.float32),),
                in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] + [vm] * 9,
                out_specs=(vm, vm, vm),
            )
        )
        step = os.path.join(_GOLDEN, "rank0", "step1")
        bank = np.zeros((_NB, 1, 7168), np.float32)
        nvb = 0
        worst = 0
        for i in range(13):
            ld = lambda n: torch.load(
                os.path.join(step, f"L{i:02d}.{n}.pt"), weights_only=False
            )
            h = ld("layer.in")["hidden_states"].float().numpy()
            attn = ld("attn.out")["out"].float().numpy()
            w = lambda n: np.asarray(
                ckpt.get(f"layers.{i}.{n}.weight"), np.float32
            ).reshape(1, -1)
            write = int(i % 12 == 0)
            a_in, m_in, bank = call(
                jnp.array([nvb, write], jnp.int32),
                jnp.asarray(h),
                jnp.asarray(attn),
                jnp.asarray(bank),
                jnp.asarray(w("self_attention_res_proj")),
                jnp.asarray(w("self_attention_res_norm")),
                jnp.asarray(w("input_layernorm")),
                jnp.asarray(w("mlp_res_proj")),
                jnp.asarray(w("mlp_res_norm")),
                jnp.asarray(w("post_attention_layernorm")),
            )
            bank = np.asarray(bank)
            nvb += write
            g_a = ld("attn.in")["hidden_states"].float().numpy()
            g_m = ld("mlp.in")["arg0"].float().numpy()
            for name, mine, gold in (("attn.in", a_in, g_a), ("mlp.in", m_in, g_m)):
                mine = np.asarray(mine).astype(ml_dtypes.bfloat16)
                gold = gold.astype(ml_dtypes.bfloat16)
                ulp = np.abs(
                    mine.view(np.int16).astype(np.int32)
                    - gold.view(np.int16).astype(np.int32)
                )
                print(
                    f"L{i:02d} {name}: bit-equal {np.mean(ulp == 0):.5f}, max ulp {ulp.max()}",
                    flush=True,
                )
                worst = max(worst, int(ulp.max()))
        self.assertLessEqual(worst, 1)


if __name__ == "__main__":
    unittest.main()
