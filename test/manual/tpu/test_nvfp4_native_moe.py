"""On-device test for the TPU NVFP4 (``modelopt_fp4``) fused-MoE path.

Requires torch_tpu + a TPU device. Quantizes fp32 weights to NVFP4, runs them
through ``prepare_tpu_nvfp4_moe_weights`` + ``fused_moe_call``, and compares
against a bf16-dequant-of-the-same-codes reference. The CPU-only
layout check runs in per-commit CI as
``test/registered/unit/layers/quantization/test_nvfp4_tpu_moe_fold.py``.
"""

from __future__ import annotations

import types
import unittest

import torch

try:
    import torch_tpu  # noqa: F401

    _HAS_TPU = True
except Exception:  # pragma: no cover
    _HAS_TPU = False


def _has_tpu_device() -> bool:
    if not _HAS_TPU:
        return False
    try:
        torch.empty(1, device="tpu")
        return True
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu + a TPU device"

NVFP4_BLOCK = 512
E2M1_MAX = 6.0
E2M1_ABS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


# NVFP4 quantize/dequant helpers for the stock row-major block-512 path.


def _quantize_rows_nvfp4(x, g, block_size=NVFP4_BLOCK):
    """Quantize ``(E, R, K)`` fp32 to NVFP4 with a per-expert global ``g`` (E,).

    Returns ``(packed uint8 (E, R, K/2), block_scale fp8-e4m3 (E, R, K/16))``.
    The stored fp8 block scale is ``(amax/6) / global`` so that
    ``code * fp8_scale.float() * global`` reconstructs the value.
    """
    E, R, K = x.shape
    nb = K // block_size
    blocks = x.float().reshape(E, R, nb, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    real = (amax / E2M1_MAX).clamp(min=1e-12)  # real per-block scale
    g4 = g.float()[:, None, None, None]
    fp8 = (real / g4).to(torch.float8_e4m3fn)  # stored fp8 block scale
    eff = (fp8.float() * g4).clamp(min=1e-12)  # effective scale used to code
    q = (blocks / eff).clamp(-E2M1_MAX, E2M1_MAX)
    sign = torch.sign(q)
    sign_bit = ((2 - sign) // 2).to(torch.uint8)  # 0 for +, 1 for -
    ord_ = torch.bucketize(q.abs(), E2M1_BOUNDS).to(torch.uint8)  # e2m1 magnitude
    codes = sign_bit * 0b1000 + ord_  # (E, R, nb, block_size)
    low = codes[..., 0::2]
    high = codes[..., 1::2] << 4
    packed = (low + high).to(torch.uint8).reshape(E, R, nb * (block_size // 2))
    return packed, fp8.squeeze(-1)


def _dequant_rows_nvfp4(packed, fp8_scale, g, dtype=torch.bfloat16):
    """Inverse of :func:`_quantize_rows_nvfp4` (independent reference)."""
    E, R, K_packed = packed.shape
    nb = fp8_scale.shape[-1]
    K = K_packed * 2
    block_size = K // nb
    view = packed.reshape(E, R, nb, block_size // 2)
    low = view & 0x0F
    high = (view >> 4) & 0x0F
    codes = torch.zeros(E, R, nb, block_size, dtype=torch.uint8)
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    sign = 1.0 - 2.0 * ((codes & 0b1000) >> 3).to(torch.float32)
    mag = (codes & 0b0111).long()
    absv = E2M1_ABS[mag.reshape(-1)].reshape(mag.shape)
    eff = (fp8_scale.float() * g.float()[:, None, None]).unsqueeze(-1)
    return (sign * absv * eff).reshape(E, R, K).to(dtype)


def _make_nvfp4_moe_layer(E, hidden, intermediate, seed, block_size=NVFP4_BLOCK):
    """Build a synthetic NVFP4 MoE layer matching ModelOpt checkpoint params.

    ``block_size`` is the stock Pallas path's block-512 scale granularity.
    """
    g = torch.Generator().manual_seed(seed)
    w1f = (torch.randn(E, intermediate, hidden, generator=g) * 0.1).float()
    w3f = (torch.randn(E, intermediate, hidden, generator=g) * 0.1).float()
    w2f = (torch.randn(E, hidden, intermediate, generator=g) * 0.1).float()
    # Block-512 checkpoints store the global scale pre-folded into block scales.
    g_gate = torch.ones(E)
    g_up = torch.ones(E)
    g_down = torch.ones(E)

    w1_p, w1_s = _quantize_rows_nvfp4(w1f, g_gate, block_size)
    w3_p, w3_s = _quantize_rows_nvfp4(w3f, g_up, block_size)
    w2_p, w2_s = _quantize_rows_nvfp4(w2f, g_down, block_size)

    w13_weight = torch.cat([w1_p, w3_p], dim=1)  # (E, 2I, H/2) u8, [Gate, Up]
    w13_scale = torch.cat([w1_s, w3_s], dim=1)  # (E, 2I, H/G) fp8
    w13_scale_2 = torch.stack([g_gate, g_up], dim=1)  # (E, 2) fp32

    return {
        "w13_weight": w13_weight,
        "w2_weight": w2_p,
        "w13_weight_scale": w13_scale,
        "w2_weight_scale": w2_s,
        "w13_weight_scale_2": w13_scale_2,
        "w2_weight_scale_2": g_down,
        "globals": (g_gate, g_up, g_down),
    }


def _build_fake_layer(data, extra_params=True):
    """Wrap synthetic tensors in an nn.Module with a moe_runner_config."""
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    E = data["w13_weight"].shape[0]
    layer.w13_weight = torch.nn.Parameter(data["w13_weight"], requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(data["w2_weight"], requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        data["w13_weight_scale"], requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        data["w2_weight_scale"], requires_grad=False
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        data["w13_weight_scale_2"], requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        data["w2_weight_scale_2"], requires_grad=False
    )
    if extra_params:
        # Params the TPU path must clear.
        layer.w13_input_scale = torch.nn.Parameter(
            torch.ones(E, 2), requires_grad=False
        )
        layer.w2_input_scale = torch.nn.Parameter(torch.ones(E), requires_grad=False)
    return layer


# The CPU-only layout checks live in per-commit CI:
# test/registered/unit/layers/quantization/test_nvfp4_tpu_moe_fold.py


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestNativeNvfp4FusedMoE(unittest.TestCase):
    """Packed NVFP4 + block scales vs bf16-dequant reference on TPU."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang_tpu.patches import apply_tpu_patches

            apply_tpu_patches()
        except Exception:  # pragma: no cover
            pass
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        cls.op = _build_fused_moe_op()
        cls.device = "tpu"

    def _inputs(self, m, hidden, topk, E, seed):
        g = torch.Generator().manual_seed(seed)
        hs = (torch.randn(m, hidden, generator=g) * 0.5).to(torch.bfloat16)
        topk_ids = torch.randint(0, E, (m, topk), generator=g, dtype=torch.int32)
        topk_w = torch.softmax(torch.randn(m, topk, generator=g).float(), dim=-1).to(
            torch.bfloat16
        )
        return hs, topk_ids, topk_w

    def test_native_matches_bf16(self):
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )
        from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

        # Real routed-MoE expert shapes: hidden 4096, intermediate 4096, top-8.
        # E is a small subset of the full expert count (the kernel loops over
        # experts identically, so a subset exercises the same code paths).
        E, hidden, intermediate, topk = 8, 4096, 4096, 8
        data = _make_nvfp4_moe_layer(E, hidden, intermediate, seed=3)
        hs, topk_ids, topk_w = self._inputs(64, hidden, topk, E, seed=4)
        gg, gu, g2 = data["globals"]

        # --- bf16 reference: dequant the same codes+scales, run bf16 path ---
        w1 = _dequant_rows_nvfp4(
            data["w13_weight"][:, :intermediate],
            data["w13_weight_scale"][:, :intermediate],
            gg,
        )
        w3 = _dequant_rows_nvfp4(
            data["w13_weight"][:, intermediate:],
            data["w13_weight_scale"][:, intermediate:],
            gu,
        )
        w13_bf16 = torch.cat([w1, w3], dim=1).permute(0, 2, 1).contiguous()
        w2_bf16 = (
            _dequant_rows_nvfp4(data["w2_weight"], data["w2_weight_scale"], g2)
            .permute(0, 2, 1)
            .contiguous()
        )
        ref = (
            fused_moe_call(
                hidden_states=hs.to(self.device),
                w1=w13_bf16.to(self.device),
                w2=w2_bf16.to(self.device),
                topk_weights=topk_w.to(self.device),
                topk_ids=topk_ids.to(self.device),
                topk=topk,
                activation="silu",
            )
            .cpu()
            .float()
        )

        # --- native packed-FP4 path ---
        layer = _build_fake_layer(data)
        prepare_tpu_nvfp4_moe_weights(layer, group_size=NVFP4_BLOCK)
        native = (
            fused_moe_call(
                hidden_states=hs.to(self.device),
                w1=layer.w13_weight.to(self.device),
                w2=layer.w2_weight.to(self.device),
                w1_scale=layer.w13_weight_scale.to(self.device),
                w2_scale=layer.w2_weight_scale.to(self.device),
                topk_weights=topk_w.to(self.device),
                topk_ids=topk_ids.to(self.device),
                topk=topk,
                activation="silu",
            )
            .cpu()
            .float()
        )
        torch.accelerator.synchronize()

        self.assertEqual(native.shape, ref.shape)
        self.assertTrue(torch.isfinite(native).all().item(), "native NaN/Inf")
        # Both paths use the same fp4 codes; native is W4A8, the reference uses
        # dequantized bf16 weights, so the
        # only divergence is fp8 activation rounding over GMM1 -> SwiGLU -> GMM2.
        # Measured on tpu7x at these shapes: rel_mean ~= 0.049, cos ~= 0.9988;
        # bounds sit ~1.5x above that so a real wiring regression (rel_mean -> ~0.3,
        # cos < 0.99) trips them. (Recalibrate if a legit run exceeds them.)
        denom = ref.abs().mean().item() + 1e-6
        rel_mean = (native - ref).abs().mean().item() / denom
        cos = torch.nn.functional.cosine_similarity(
            native.flatten(), ref.flatten(), dim=0
        ).item()
        self.assertLess(rel_mean, 0.08, f"native vs bf16 ref rel mean err={rel_mean}")
        self.assertGreater(cos, 0.995, f"native vs bf16 ref cosine={cos}")


if __name__ == "__main__":
    unittest.main()
