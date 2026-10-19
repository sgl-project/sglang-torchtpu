"""Manual check: packed MXFP4 fused MoE vs a BF16-dequantized reference.

Uses GPT-OSS's interleaved, biased layout on the packed path, which production
does not take (GPT-OSS stays on BF16 dequantization in
``prepare_tpu_mxfp4_moe_weights``); the registered
``test_fused_moe_situ_mxfp4`` covers the production packed path.
Requires torch_tpu and a TPU device.
"""

from __future__ import annotations

import unittest

import torch

try:
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import compile as _torch_tpu_compile  # noqa: F401

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

MXFP4_BLOCK_SIZE = 32
E2M1_MAX = 6.0
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _quantize_to_hf_mxfp4(
    x_float: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor along its last axis to HF MXFP4 layout.

    Returns ``(packed_blocks, scales_e8m0)`` where:
    * ``packed_blocks``: ``(*leading, num_blocks, 16)`` uint8 — 2 FP4 per byte
    * ``scales_e8m0``  : ``(*leading, num_blocks)``     uint8 — e8m0 (biased exp + 127)
    """
    assert x_float.shape[-1] % block_size == 0, (
        f"last axis {x_float.shape[-1]} not multiple of block_size {block_size}"
    )
    *leading, K = x_float.shape
    num_blocks = K // block_size

    # (..., K) → (..., num_blocks, block_size)
    blocks = x_float.float().reshape(*leading, num_blocks, block_size)

    # Per-block max abs → derive e8m0 scale s.t. block / 2^scale fits in [-6, 6].
    abs_max = blocks.abs().amax(dim=-1, keepdim=True)  # (..., num_blocks, 1)
    # descale = abs_max / 6.0; scale_exp = ceil(log2(descale)), clamped low.
    descale = abs_max / E2M1_MAX
    min_value = torch.tensor(-127.0)
    scale_exp = torch.ceil(
        torch.maximum(torch.log2(descale.clamp(min=1e-30)), min_value)
    )  # (..., num_blocks, 1) fp32
    # Cast each block to FP4 codes after the scale.
    q = (blocks / torch.exp2(scale_exp)).clamp(-E2M1_MAX, E2M1_MAX)
    sign = torch.sign(q)
    sign_bit = ((2 - sign) // 2).to(torch.uint8)  # 0 for +, 1 for -
    # Find magnitude bucket: how many of E2M1_BOUNDS does |q| exceed?
    ord_ = (((q.abs().unsqueeze(-1) - E2M1_BOUNDS) > 0).sum(dim=-1)).to(torch.uint8)
    fp4_codes = sign_bit * 0b1000 + ord_  # (..., num_blocks, block_size) uint8

    # Pack pairs of 4-bit codes into uint8 bytes along the block_size axis.
    # Within a byte: low nibble = even-index, high nibble = odd-index (HF order).
    low = fp4_codes[..., 0::2]
    high = fp4_codes[..., 1::2] << 4
    packed = (low + high).to(torch.uint8)  # (..., num_blocks, block_size/2 = 16)

    scales_e8m0 = (scale_exp.squeeze(-1) + 127.0).to(torch.uint8)
    return packed, scales_e8m0


def _dequant_hf_mxfp4(
    packed_blocks: torch.Tensor,
    scales_e8m0: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize HF MXFP4 to a target float dtype (independent reference).

    Inverse of ``_quantize_to_hf_mxfp4``. Accepts the 3D flattened layout
    that the HF loader produces: ``packed_blocks`` shape
    ``(*leading, K_packed)``, ``scales_e8m0`` shape
    ``(*leading, num_blocks)``. Returns ``(*leading, K)``.
    """
    *leading, K_packed = packed_blocks.shape
    num_blocks = scales_e8m0.shape[-1]
    assert scales_e8m0.shape[:-1] == tuple(leading), (
        f"scale leading dims {scales_e8m0.shape[:-1]} != packed {tuple(leading)}"
    )
    block_size = (K_packed * 2) // num_blocks  # = MXFP4_BLOCK_SIZE = 32
    assert block_size == MXFP4_BLOCK_SIZE, (
        f"derived block_size={block_size}, expected {MXFP4_BLOCK_SIZE}"
    )
    block_bytes = block_size // 2  # = 16

    # Reshape to expose blocks: (*leading, num_blocks, block_bytes)
    blocks_view = packed_blocks.reshape(*leading, num_blocks, block_bytes)
    # Unpack uint8 → uint4 codes (block_size = 32 per block).
    low = blocks_view & 0x0F
    high = (blocks_view >> 4) & 0x0F
    codes = torch.zeros(
        *leading, num_blocks, block_size, dtype=torch.uint8, device=packed_blocks.device
    )
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    # Decode sign + magnitude.
    sign = 1.0 - 2.0 * ((codes & 0b1000) >> 3).to(torch.float32)
    magnitude = (codes & 0b0111).to(torch.long)
    values = E2M1_VALUES.to(packed_blocks.device)
    abs_vals = values[magnitude.reshape(-1)].reshape(magnitude.shape)
    signed_vals = sign * abs_vals  # (..., num_blocks, block_size) fp32
    # Apply e8m0 scale.
    scale_fp32 = torch.exp2(
        scales_e8m0.to(torch.int32).to(torch.float32) - 127.0
    ).unsqueeze(-1)  # (..., num_blocks, 1)
    dequant = signed_vals * scale_fp32  # (..., num_blocks, block_size) fp32
    return dequant.reshape(*leading, num_blocks * block_size).to(dtype)


def _build_synthetic_hf_gpt_oss_layer(
    E: int,
    hidden: int,
    intermediate: int,
    seed: int,
) -> dict:
    """Generate one synthetic GPT-OSS MoE layer's worth of HF-format weights.

    Returns a dict with the same keys / shapes as HF's checkpoint:
    * ``gate_up_proj_blocks``    (E, 2*intermediate, hidden/32, 16) uint8
    * ``gate_up_proj_scales``    (E, 2*intermediate, hidden/32)    uint8
    * ``gate_up_proj_bias``      (E, 2*intermediate)               bf16
    * ``down_proj_blocks``       (E, hidden, intermediate/32, 16)  uint8
    * ``down_proj_scales``       (E, hidden, intermediate/32)      uint8
    * ``down_proj_bias``         (E, hidden)                       bf16
    """
    g = torch.Generator().manual_seed(seed)
    # Small-magnitude weights so dequantization stays in FP4 dynamic range.
    w13_fp32 = (torch.randn(E, 2 * intermediate, hidden, generator=g) * 0.1).to(
        torch.float32
    )
    w2_fp32 = (torch.randn(E, hidden, intermediate, generator=g) * 0.1).to(
        torch.float32
    )
    w13_packed, w13_scales = _quantize_to_hf_mxfp4(w13_fp32)
    w2_packed, w2_scales = _quantize_to_hf_mxfp4(w2_fp32)
    w13_bias = (torch.randn(E, 2 * intermediate, generator=g) * 0.01).to(torch.bfloat16)
    w2_bias = (torch.randn(E, hidden, generator=g) * 0.01).to(torch.bfloat16)
    # The HF model loader (see ``gpt_oss.py``) flattens (num_blocks, 16) → K_packed
    # before storing into ``layer.w13_weight`` / ``layer.w2_weight``. Mirror
    # that so the synthetic data feeds the same 3D layout the real loader
    # produces.
    w13_packed = w13_packed.reshape(
        *w13_packed.shape[:-2], -1
    )  # (E, 2N, num_blocks*16)
    w2_packed = w2_packed.reshape(
        *w2_packed.shape[:-2], -1
    )  # (E, hidden, num_blocks*16)
    return {
        "gate_up_proj_blocks": w13_packed,
        "gate_up_proj_scales": w13_scales,
        "gate_up_proj_bias": w13_bias,
        "down_proj_blocks": w2_packed,
        "down_proj_scales": w2_scales,
        "down_proj_bias": w2_bias,
    }


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestNativeMxfp4FusedMoE(unittest.TestCase):
    """Native packed-FP4 + block scales vs BF16-dequant reference."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang_tpu.patches import apply_tpu_patches

            apply_tpu_patches()
        except Exception:  # pragma: no cover
            pass
        from sglang_tpu.layers.moe.pallas.fused_moe_op import (
            _build_fused_moe_op,
        )

        cls.op = _build_fused_moe_op()
        cls.device = "tpu"

    def _build_inputs(self, m, hidden, topk, E, intermediate, seed):
        g = torch.Generator().manual_seed(seed)
        hs = (torch.randn(m, hidden, generator=g) * 0.5).to(torch.bfloat16)
        topk_ids = torch.randint(0, E, (m, topk), generator=g, dtype=torch.int32)
        topk_w = torch.softmax(torch.randn(m, topk, generator=g).float(), dim=-1).to(
            torch.bfloat16
        )
        return hs, topk_ids, topk_w

    def _run_native(self, layer, hs, topk_ids, topk_w):
        """Run the native MXFP4 fused MoE call (uint8 packed + fp32 scales)."""
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_native,
        )
        from sglang_tpu.layers.moe.pallas.fused_moe_op import fused_moe_call

        w13, w13_s, w13_b, w2, w2_s, w2_b = process_mxfp4_moe_weights_native(
            layer["gate_up_proj_blocks"],
            layer["gate_up_proj_scales"],
            layer["gate_up_proj_bias"],
            layer["down_proj_blocks"],
            layer["down_proj_scales"],
            layer["down_proj_bias"],
            w13_interleaved=True,
        )
        out = fused_moe_call(
            hidden_states=hs.to(self.device),
            w1=w13.to(self.device),
            w1_scale=w13_s.to(self.device),
            w1_bias=w13_b.to(self.device),
            w2=w2.to(self.device),
            w2_scale=w2_s.to(self.device),
            w2_bias=w2_b.to(self.device),
            topk_weights=topk_w.to(self.device),
            topk_ids=topk_ids.to(self.device),
            topk=int(topk_ids.shape[-1]),
            activation="swigluoai",
        )
        return out.cpu().float()

    def _run_bf16_reference(self, layer, hs, topk_ids, topk_w):
        """Reference: dequant HF MXFP4 to bf16 ourselves, then run fused_moe_call
        on the bf16 weights with no scales (so the kernel takes the
        unquantized fast path).

        Layout transformations (deinterleave, N-pad, transpose) mirror
        the native path so the two are directly comparable.
        """
        from sglang_tpu.layers.moe.pallas.fused_moe_op import fused_moe_call

        # Dequant
        w13_bf16 = _dequant_hf_mxfp4(
            layer["gate_up_proj_blocks"], layer["gate_up_proj_scales"], torch.bfloat16
        )  # (E, 2N_int, hidden) bf16
        w2_bf16 = _dequant_hf_mxfp4(
            layer["down_proj_blocks"], layer["down_proj_scales"], torch.bfloat16
        )  # (E, hidden, intermediate) bf16

        # Replicate the deinterleave + N-pad + transpose path so the
        # comparison is "same layout, different weight dtype".
        E, two_N_int, hidden = w13_bf16.shape
        N_int = two_N_int // 2
        align_to = 128
        pad_n = (align_to - N_int % align_to) % align_to

        def _deinterleave(t):
            E_, two_N = t.shape[:2]
            rest = t.shape[2:]
            N_ = two_N // 2
            perm = (0, 2, 1) + tuple(range(3, 3 + len(rest)))
            return (
                t.reshape(E_, N_, 2, *rest)
                .permute(*perm)
                .contiguous()
                .reshape(E_, 2 * N_, *rest)
            )

        def _pad_halves(t, n, pad):
            if pad == 0:
                return t
            E_ = t.shape[0]
            rest = t.shape[2:]
            gate, up = t[:, :n, ...], t[:, n:, ...]
            zeros = torch.zeros((E_, pad, *rest), dtype=t.dtype, device=t.device)
            return torch.cat([gate, zeros, up, zeros], dim=1)

        w13_bf16 = _deinterleave(w13_bf16)
        w13_bf16 = _pad_halves(w13_bf16, N_int, pad_n)  # (E, 2*N_padded, hidden)
        w13_bf16 = w13_bf16.permute(0, 2, 1).contiguous()  # (E, hidden, 2*N_padded)

        w13_bias = _deinterleave(layer["gate_up_proj_bias"].unsqueeze(-1)).squeeze(-1)
        w13_bias = _pad_halves(w13_bias.unsqueeze(-1), N_int, pad_n).squeeze(-1)
        w13_bias_bf16 = w13_bias.to(torch.float32).unsqueeze(1).contiguous()

        # w2: pad K from intermediate to N_padded
        w2_bf16 = torch.cat(
            [
                w2_bf16,
                torch.zeros(
                    E, hidden, pad_n, dtype=w2_bf16.dtype, device=w2_bf16.device
                ),
            ],
            dim=-1,
        )
        w2_bf16 = w2_bf16.permute(0, 2, 1).contiguous()  # (E, N_padded, hidden)
        w2_bias_bf16 = (
            layer["down_proj_bias"].to(torch.float32).unsqueeze(1).contiguous()
        )

        out = fused_moe_call(
            hidden_states=hs.to(self.device),
            w1=w13_bf16.to(self.device),
            w1_bias=w13_bias_bf16.to(self.device),
            w2=w2_bf16.to(self.device),
            w2_bias=w2_bias_bf16.to(self.device),
            topk_weights=topk_w.to(self.device),
            topk_ids=topk_ids.to(self.device),
            topk=int(topk_ids.shape[-1]),
            activation="swigluoai",
        )
        return out.cpu().float()

    def test_small_layer_native_matches_bf16(self):
        """Small layer (E=4, hidden=128, intermediate=128): native = bf16 ref."""
        E, hidden, intermediate = 4, 128, 128
        layer = _build_synthetic_hf_gpt_oss_layer(E, hidden, intermediate, seed=0)
        hs, topk_ids, topk_w = self._build_inputs(
            m=32, hidden=hidden, topk=2, E=E, intermediate=intermediate, seed=1
        )

        native_out = self._run_native(layer, hs, topk_ids, topk_w)
        ref_out = self._run_bf16_reference(layer, hs, topk_ids, topk_w)
        torch.accelerator.synchronize()

        self.assertEqual(native_out.shape, ref_out.shape)
        self.assertTrue(torch.isfinite(native_out).all().item(), "native NaN/Inf")
        self.assertTrue(torch.isfinite(ref_out).all().item(), "bf16 ref NaN/Inf")

        max_diff = (native_out - ref_out).abs().max().item()
        mean_diff = (native_out - ref_out).abs().mean().item()
        # Both paths use the same dequantized values; the matmul is bf16
        # in both cases. Differences are purely accumulation-order /
        # rounding noise from gmm_v2's tiling — should be very small.
        self.assertLess(
            max_diff,
            5e-2,
            f"native vs bf16 ref max abs diff = {max_diff} (mean {mean_diff})",
        )


if __name__ == "__main__":
    unittest.main()
