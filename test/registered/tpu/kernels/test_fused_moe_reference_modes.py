"""Pallas fused MoE vs the dense float32 reference, per production weight mode.

Each case builds weights the way the checkpoint stores them, runs them through
the production load-time prep where one exists, and feeds the result to
``_fused_moe_jax`` with the SparseCore gather on (``fused_moe_call``'s default).
The reference (``sglang_tpu.testing.moe_reference``) dequantizes the checkpoint
encoding independently, so prep, unpack order, scale indexing, bias, and
activation drift all land here. Weights are exact in both (vmem-fp8 alone
rounds them to fp8 by design); the tolerances cover output and activation
rounding.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=240,
    suite="stage-a-test-tpu-kernel",
    labels=["kernel", "moe", "fused-moe"],
)

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import Parameter

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    from sglang_tpu.testing import moe_reference as ref

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a TPU backend"

# bf16 activations, f32 accumulation: bf16 rounding of the GMM outputs (~0.003).
_BF16_TOL = 0.02
# fp8 activations in both GMMs, bf16 accumulator: a CPU emulation peaks near 0.05.
_FP8_ACT_TOL = 0.1
# vmem-fp8 also rounds each weight tile to fp8: the emulation peaks near 0.085.
_W4A8_TOL = 0.15


def _to_jax(t: torch.Tensor):
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _routing(m, num_experts, topk, seed):
    """Distinct experts per token, as a top-k router emits; bf16 weights."""
    rng = np.random.default_rng(seed)
    ids = np.stack(
        [rng.choice(num_experts, size=topk, replace=False) for _ in range(m)]
    )
    w = rng.random((m, topk)).astype(np.float32)
    weights = jnp.asarray(w / w.sum(-1, keepdims=True)).astype(jnp.bfloat16)
    return jnp.asarray(ids, jnp.int32), weights


def _hidden(m, k, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((m, k)), jnp.float32).astype(jnp.bfloat16)


def _run(
    hidden,
    w1,
    w2,
    ids,
    weights,
    *,
    activation,
    w1_scale=None,
    w2_scale=None,
    w1_bias=None,
    w2_bias=None,
    use_sc_gather=1,
    **kwargs,
):
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    out = _fused_moe_jax(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk=int(ids.shape[1]),
        activation_name=activation,
        use_sc_gather=use_sc_gather,
        **kwargs,
    )
    return jax.block_until_ready(out)


# ---- GPT-OSS: interleaved gate/up MXFP4 with biases -------------------------


def _gpt_oss_checkpoint(num_experts, hidden, inter, seed):
    """HF GPT-OSS MoE tensors: w13 rows interleave gate_0, up_0, gate_1, ...;
    MXFP4 along K with e8m0 scales; bf16 biases."""
    g = torch.Generator().manual_seed(seed)

    def u8(lo, hi, *shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=g)

    E, H, INTER = num_experts, hidden, inter
    return dict(
        w13=u8(0, 256, E, 2 * INTER, H // 2),
        # e8m0 122..125 = 2^-5..2^-2: gate/up straddle the swiglu limit of 7.
        w13_scale=u8(122, 126, E, 2 * INTER, H // 32),
        w13_bias=torch.randn(E, 2 * INTER, generator=g).to(torch.bfloat16),
        w2=u8(0, 256, E, H, INTER // 2),
        w2_scale=u8(118, 122, E, H, INTER // 32),
        w2_bias=torch.randn(E, H, generator=g).to(torch.bfloat16),
    )


def _gpt_oss_reference_weights(ckpt):
    """Dequantize and de-interleave the checkpoint into moe_reference's layout."""
    w13 = ref.dequant_mxfp4_rows(ckpt["w13"].numpy(), ckpt["w13_scale"].numpy())
    b13 = jnp.asarray(ckpt["w13_bias"].float().numpy())
    w1 = jnp.concatenate([w13[:, 0::2], w13[:, 1::2]], axis=1).transpose(0, 2, 1)
    b1 = jnp.concatenate([b13[:, 0::2], b13[:, 1::2]], axis=1)
    w2 = ref.dequant_mxfp4_rows(ckpt["w2"].numpy(), ckpt["w2_scale"].numpy())
    b2 = jnp.asarray(ckpt["w2_bias"].float().numpy())
    return w1, b1, w2.transpose(0, 2, 1), b2


def _unclamped_swiglu_oai(gate, up):
    return ref.swiglu_oai(gate, up, limit=float("inf"))


# ---- INT4 (compressed-tensors pack-quantized) -------------------------------


def _int4_checkpoint(num_experts, hidden, inter, seed):
    """[E, K//8, N] int32 offset-binary words + bf16 group-32 scales [E, K//32, N]."""
    rng = np.random.default_rng(seed)

    def one(k, n):
        q = rng.integers(-8, 8, (num_experts, k, n))
        nib = ((q + 8) & 0xF).astype(np.uint32).reshape(num_experts, k // 8, 8, n)
        shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :, None]
        words = np.bitwise_or.reduce(nib << shifts, axis=2).view(np.int32)
        # 2^-8..2^-4 per group: a group-index slip moves outputs by up to 16x.
        scale = torch.from_numpy(
            np.exp2(rng.uniform(-8, -4, (num_experts, k // 32, n)))
        ).to(torch.bfloat16)
        return words, scale

    w13, w13_scale = one(hidden, 2 * inter)
    w2, w2_scale = one(inter, hidden)
    return dict(w13=w13, w13_scale=w13_scale, w2=w2, w2_scale=w2_scale)


def _prep_int4(ckpt):
    """Run prepare_tpu_w4a16_moe_weights on a stub layer; return its jnp tensors."""
    from sglang_tpu.w4a16_moe import (
        prepare_tpu_w4a16_moe_weights,
    )

    layer = torch.nn.Module()
    for name in ("w13", "w2"):
        # Copies: the prep flips nibble sign bits in place.
        packed = torch.from_numpy(ckpt[name].copy())
        layer.register_parameter(
            f"{name}_weight_packed", Parameter(packed, requires_grad=False)
        )
        layer.register_parameter(
            f"{name}_weight_scale",
            Parameter(ckpt[f"{name}_scale"].clone(), requires_grad=False),
        )
    prepare_tpu_w4a16_moe_weights(layer, 32)
    out = {}
    for name in ("w13", "w2"):
        out[name] = _to_jax(getattr(layer, f"{name}_weight_packed").data)
        out[f"{name}_scale"] = _to_jax(getattr(layer, f"{name}_weight_scale").data)
        channel = getattr(layer, f"{name}_vmem_channel_scale", None)
        if channel is not None:
            out[f"{name}_channel"] = _to_jax(channel.data)
    return out


def _int4_reference_weights(ckpt):
    def deq(words, scale):
        q = ref.unpack_int4(words, axis=1, offset_binary=True)
        return ref.apply_gmm_scale(q, jnp.asarray(scale.float().numpy())[:, :, None])

    return deq(ckpt["w13"], ckpt["w13_scale"]), deq(ckpt["w2"], ckpt["w2_scale"])


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEReferenceModes(CustomTestCase):
    def assert_matches(self, out, expected, tol, context=""):
        err = ref.max_rel_error(out, expected)
        self.assertLess(err, tol, f"{context}: max rel error {err:.4f}")

    def test_gpt_oss_bf16_swiglu_oai_with_bias(self):
        """GPT-OSS's TPU path: MXFP4 dequantized to bf16 at load, de-interleaved,
        gate/up halves padded; hidden is not lane-aligned (2880 in the model)
        so the kernel pads K of the activations and slices the output."""
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_bf16,
        )

        E, H, INTER, M, TOPK = 16, 320, 320, 64, 4
        ckpt = _gpt_oss_checkpoint(E, H, INTER, seed=0)
        w13, b13, w2, b2 = process_mxfp4_moe_weights_bf16(
            ckpt["w13"],
            ckpt["w13_scale"],
            ckpt["w13_bias"],
            ckpt["w2"],
            ckpt["w2_scale"],
            ckpt["w2_bias"],
            w13_interleaved=True,
        )
        hidden = _hidden(M, H, seed=1)
        ids, weights = _routing(M, E, TOPK, seed=2)
        out = _run(
            hidden,
            _to_jax(w13),
            _to_jax(w2),
            ids,
            weights,
            activation="swigluoai",
            w1_bias=_to_jax(b13),
            w2_bias=_to_jax(b2),
            # As GPT-OSS serves (SGLANG_TPU_FUSED_MOE_SC_GATHER=0 in its
            # recipes): SC gather does not lower when hidden is not a multiple
            # of 128 (320 here, 2880 in the model).
            use_sc_gather=0,
        )

        w1r, b1r, w2r, b2r = _gpt_oss_reference_weights(ckpt)

        def moe(act):
            return ref.moe_reference(
                hidden, w1r, w2r, weights, ids, activation=act, w1_bias=b1r, w2_bias=b2r
            )

        expected = moe("swigluoai")
        self.assert_matches(out, expected, _BF16_TOL)
        # The inputs must drive the clamp, or a wrong limit would pass.
        self.assertGreater(ref.max_rel_error(moe(_unclamped_swiglu_oai), expected), 0.1)

    def test_packed_mxfp4_swiglu_oai_with_bias(self):
        """GPT-OSS layout on the packed-FP4 path (fused GMM1 with per-32 scales
        and biases on both gate/up halves); hidden is lane-aligned, as the packed
        path requires."""
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_native,
        )

        E, H, INTER, M, TOPK = 16, 256, 320, 64, 4
        ckpt = _gpt_oss_checkpoint(E, H, INTER, seed=3)
        w13, w13_s, b13, w2, w2_s, b2 = process_mxfp4_moe_weights_native(
            ckpt["w13"],
            ckpt["w13_scale"],
            ckpt["w13_bias"],
            ckpt["w2"],
            ckpt["w2_scale"],
            ckpt["w2_bias"],
            w13_interleaved=True,
        )
        hidden = _hidden(M, H, seed=4)
        ids, weights = _routing(M, E, TOPK, seed=5)
        out = _run(
            hidden,
            _to_jax(w13),
            _to_jax(w2),
            ids,
            weights,
            activation="swigluoai",
            w1_scale=_to_jax(w13_s),
            w2_scale=_to_jax(w2_s),
            w1_bias=_to_jax(b13),
            w2_bias=_to_jax(b2),
        )

        w1r, b1r, w2r, b2r = _gpt_oss_reference_weights(ckpt)
        expected = ref.moe_reference(
            hidden,
            w1r,
            w2r,
            weights,
            ids,
            activation="swigluoai",
            w1_bias=b1r,
            w2_bias=b2r,
        )
        # MXFP4 blocks of 32 run W4A16 (dequantized in VMEM): bf16 activations.
        self.assert_matches(out, expected, _BF16_TOL)

    def test_w4a16_matches_dequantized_checkpoint(self):
        """Checkpoint int4 through prepare_tpu_w4a16_moe_weights and the unfused
        GMM1, in both W4A16 kernel modes."""
        from sglang.srt.environ import envs

        E, H, INTER, M, TOPK = 16, 512, 256, 64, 4
        ckpt = _int4_checkpoint(E, H, INTER, seed=6)
        hidden = _hidden(M, H, seed=7)
        ids, weights = _routing(M, E, TOPK, seed=8)
        w1r, w2r = _int4_reference_weights(ckpt)
        expected = ref.moe_reference(hidden, w1r, w2r, weights, ids, activation="silu")

        # (SGLANG_TPU_INT4_MOE_MODE, SGLANG_TPU_INT4_MOE_FP32_SCALES)
        for mode, fp32_scales in ((None, False), ("vmem-bf16", True)):
            with (
                self.subTest(mode=mode),
                envs.SGLANG_TPU_INT4_MOE_MODE.override(mode),
                envs.SGLANG_TPU_INT4_MOE_FP32_SCALES.override(fp32_scales),
            ):
                w = _prep_int4(ckpt)
                out = _run(
                    hidden,
                    w["w13"],
                    w["w2"],
                    ids,
                    weights,
                    activation="silu",
                    w1_scale=w["w13_scale"],
                    w2_scale=w["w2_scale"],
                    w_quant_bits=4,
                )
                self.assert_matches(out, expected, _BF16_TOL, f"mode={mode}")

    def test_w4a8_vmem_fp8_matches_dequantized_checkpoint(self):
        """vmem-fp8: normalized group scales + per-channel scales from the prep,
        int4 requantized to fp8 in VMEM, fp8 activations."""
        from sglang.srt.environ import envs

        E, H, INTER, M, TOPK = 16, 512, 256, 64, 4
        ckpt = _int4_checkpoint(E, H, INTER, seed=9)
        with envs.SGLANG_TPU_INT4_MOE_MODE.override("vmem-fp8"):
            w = _prep_int4(ckpt)
        hidden = _hidden(M, H, seed=10)
        ids, weights = _routing(M, E, TOPK, seed=11)
        out = _run(
            hidden,
            w["w13"],
            w["w2"],
            ids,
            weights,
            activation="silu",
            w1_scale=w["w13_scale"],
            w2_scale=w["w2_scale"],
            w_quant_bits=4,
            w1_channel_scale=w["w13_channel"],
            w2_channel_scale=w["w2_channel"],
        )
        w1r, w2r = _int4_reference_weights(ckpt)
        expected = ref.moe_reference(hidden, w1r, w2r, weights, ids, activation="silu")
        self.assert_matches(out, expected, _W4A8_TOL)

    def test_block_fp8_scales_in_both_runtime_layouts(self):
        """Block-128 FP8 checkpoint scales in the default per-column layout
        (fused GMM1) and the compact 128-slot layout (unfused GMM1)."""
        from sglang.srt.environ import envs
        from sglang_tpu.fp8_moe import (
            _to_tpu_fp8_moe_block_scale_layout,
        )

        # out sizes 2 * INTER and H above 512 so compact scales span several records.
        E, H, INTER, M, TOPK, B = 8, 1024, 512, 64, 4, 128
        g = torch.Generator().manual_seed(12)

        def checkpoint(n, k):
            # [E, N, K] fp8 + scale_inv [E, N/128, K/128], as block-FP8 ships.
            q = (torch.randn(E, n, k, generator=g) * 64).clamp(-448, 448)
            q = q.to(torch.float8_e4m3fn)
            # Powers of two: exact in the bf16 runtime scale, 16x spread.
            scale = torch.exp2(
                torch.randint(-13, -8, (E, n // B, k // B), generator=g).float()
            )
            return q, scale

        def dequant(q, scale):
            s = scale.repeat_interleave(B, dim=1).repeat_interleave(B, dim=2)
            return jnp.asarray((q.float() * s).transpose(1, 2).numpy())

        w13, w13_s = checkpoint(2 * INTER, H)
        w2, w2_s = checkpoint(H, INTER)
        hidden = _hidden(M, H, seed=13)
        ids, weights = _routing(M, E, TOPK, seed=14)
        expected = ref.moe_reference(
            hidden,
            dequant(w13, w13_s),
            dequant(w2, w2_s),
            weights,
            ids,
            activation="silu",
        )
        # fp8_moe.prepare_tpu_fp8_moe_weights transposes weights to [E, K, N].
        w1 = jnp.asarray(w13.transpose(1, 2).float().numpy()).astype(jnp.float8_e4m3fn)
        w2_k = jnp.asarray(w2.transpose(1, 2).float().numpy()).astype(jnp.float8_e4m3fn)
        for compact in (False, True):
            with (
                self.subTest(compact=compact),
                envs.SGLANG_TPU_FP8_MOE_COMPACT_SCALES.override(compact),
            ):
                s1 = _to_tpu_fp8_moe_block_scale_layout(w13_s, 2 * INTER)
                s2 = _to_tpu_fp8_moe_block_scale_layout(w2_s, H)
                out = _run(
                    hidden,
                    w1,
                    w2_k,
                    ids,
                    weights,
                    activation="silu",
                    w1_scale=_to_jax(s1),
                    w2_scale=_to_jax(s2),
                )
                self.assert_matches(out, expected, _FP8_ACT_TOL, f"compact={compact}")

    def test_nvfp4_block512_with_split_global_scales(self):
        """NVFP4 through prepare_tpu_nvfp4_moe_weights: separate gate/up global
        scales folded into the block-512 scales, packed W4A8 GMMs."""
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        E, H, INTER, M, TOPK, B = 8, 1024, 512, 64, 4, 512
        g = torch.Generator().manual_seed(15)

        def u8(*shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

        def block_scale(*shape):
            # fp8 block scales in {0.5, 1, 2, 4}: exact in e4m3, 8x spread.
            e = torch.randint(-1, 3, shape, generator=g).float()
            return torch.exp2(e).to(torch.float8_e4m3fn)

        ckpt = dict(
            w13=u8(E, 2 * INTER, H // 2),
            w13_scale=block_scale(E, 2 * INTER, H // B),
            # Distinct gate / up global scales per expert.
            w13_global=torch.exp2(-6.0 - torch.randint(0, 2, (E, 2), generator=g)),
            w2=u8(E, H, INTER // 2),
            w2_scale=block_scale(E, H, INTER // B),
            w2_global=torch.exp2(-6.0 - torch.randint(0, 2, (E,), generator=g)),
        )
        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        for name, key in (
            ("w13_weight", "w13"),
            ("w13_weight_scale", "w13_scale"),
            ("w13_weight_scale_2", "w13_global"),
            ("w2_weight", "w2"),
            ("w2_weight_scale", "w2_scale"),
            ("w2_weight_scale_2", "w2_global"),
        ):
            layer.register_parameter(
                name, Parameter(ckpt[key].clone(), requires_grad=False)
            )
        prepare_tpu_nvfp4_moe_weights(layer)

        hidden = _hidden(M, H, seed=16)
        ids, weights = _routing(M, E, TOPK, seed=17)
        out = _run(
            hidden,
            _to_jax(layer.w13_weight.data),
            _to_jax(layer.w2_weight.data),
            ids,
            weights,
            activation="silu",
            w1_scale=_to_jax(layer.w13_weight_scale.data),
            w2_scale=_to_jax(layer.w2_weight_scale.data),
        )

        def dequant(packed, scale, global_scale):
            # [E, N, K]: codes along K, one scale per 512-K block.
            s = jnp.asarray(scale.float().numpy()) * global_scale
            values = ref.unpack_fp4(packed.numpy(), axis=-1)
            return (values * jnp.repeat(s, B, axis=-1)).transpose(0, 2, 1)

        gate_up_global = jnp.repeat(
            jnp.asarray(ckpt["w13_global"].numpy()), INTER, axis=1
        )
        w1r = dequant(ckpt["w13"], ckpt["w13_scale"], gate_up_global[:, :, None])
        w2_global = jnp.asarray(ckpt["w2_global"].numpy())[:, None, None]
        w2r = dequant(ckpt["w2"], ckpt["w2_scale"], w2_global)
        expected = ref.moe_reference(hidden, w1r, w2r, weights, ids, activation="silu")
        self.assert_matches(out, expected, _FP8_ACT_TOL)

    def test_expert_parallel_rank_sum_matches_reference(self):
        """EP4 over W4A16: each rank holds a quarter of the experts and gets the
        global ids plus its group offset; the rank outputs must sum to the
        reference over all experts, not just to the single-rank kernel."""
        E, EP, H, INTER, M, TOPK = 16, 4, 512, 256, 64, 4
        local = E // EP
        ckpt = _int4_checkpoint(E, H, INTER, seed=18)
        w = _prep_int4(ckpt)
        hidden = _hidden(M, H, seed=19)
        ids, weights = _routing(M, E, TOPK, seed=20)

        partials = []
        for rank in range(EP):
            sl = slice(rank * local, (rank + 1) * local)
            partials.append(
                _run(
                    hidden,
                    w["w13"][sl],
                    w["w2"][sl],
                    ids,
                    weights,
                    activation="silu",
                    w1_scale=w["w13_scale"][sl],
                    w2_scale=w["w2_scale"][sl],
                    w_quant_bits=4,
                    global_num_experts=E,
                    expert_group_offset=rank * local,
                ).astype(jnp.float32)
            )
        w1r, w2r = _int4_reference_weights(ckpt)
        expected = ref.moe_reference(hidden, w1r, w2r, weights, ids, activation="silu")
        # Each rank rounds its partial to bf16 once more than a single kernel.
        self.assert_matches(sum(partials), expected, _BF16_TOL)


if __name__ == "__main__":
    unittest.main()
