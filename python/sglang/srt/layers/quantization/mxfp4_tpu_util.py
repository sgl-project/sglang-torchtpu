# Adapted from vllm-project/tpu-inference: the MXFP4 MoE weight processing
# (tpu_inference/layers/vllm/quantization/mxfp4.py — process_mxfp4_moe_weights)
# and the packed-FP4 dequant primitive
# (tpu_inference/layers/common/quantization/__init__.py —
#  dequantize_tensor_from_mxfp4_packed):
#   https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/vllm/quantization/mxfp4.py
#   https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/quantization/__init__.py
"""GPT-OSS MXFP4 → native packed-FP4 weight rewrite for the TPU Pallas fused-MoE.

This is the **lossless** native path: HF's MXFP4 representation
(packed FP4 codes + per-32-element e8m0 scales) is preserved end-to-end
through to the Pallas ``gmm_v2`` kernel, which dequantizes inside VMEM
right before the MXU. No bf16 dequant materialised in HBM, no
re-quantization to a coarser block size — same numerics as if you ran
bf16 matmul on the dequantized weights, at ~4× lower HBM footprint.

HF GPT-OSS-20B MoE checkpoint layout (per layer)
------------------------------------------------
* ``gate_up_proj_blocks``    : ``(E, 2*N_int, num_blocks_hidden, 16)`` uint8 — packed FP4 codes
* ``gate_up_proj_scales``    : ``(E, 2*N_int, num_blocks_hidden)``    uint8 — e8m0 scales
* ``gate_up_proj_bias``      : ``(E, 2*N_int)``                       bf16
* ``down_proj_blocks``       : ``(E, hidden, num_blocks_inter, 16)``  uint8
* ``down_proj_scales``       : ``(E, hidden, num_blocks_inter)``      uint8
* ``down_proj_bias``         : ``(E, hidden)``                        bf16

Where:
* ``E`` = num_experts (32 for 20B).
* ``N_int`` = intermediate_size (2880).
* ``2*N_int`` axis of w13 is **interleaved**: rows alternate
  ``[gate_0, up_0, gate_1, up_1, ...]``.
* ``hidden`` = hidden_size (2880).
* ``num_blocks_<dim>`` = K / 32 where K is the reduction (=contracting)
  dim of the matmul.
* Each 16-byte block holds 32 packed FP4 values (2 codes per byte).

Output layout (gmm_v2-ready)
-----------------------------
gmm_v2 expects ``rhs.shape = (E, K, N)`` with the K element count as the
middle dim, ``rhs_scale.shape = (E, num_blocks_K, 1, N)``,
``rhs_bias.shape = (E, 1, N)``. We pass ``rhs`` as packed ``uint8`` of
shape ``(E, K_packed = K/2, N)`` — ``_fused_moe_jax`` bitcasts to
``float4_e2m1fn`` and reshapes to ``(E, K, N)`` before the kernel call.

For w13 (gate||up):
* ``w13_weight``       : ``(E, K_packed = hidden/2, 2*N_int_padded)`` uint8
* ``w13_weight_scale`` : ``(E, num_blocks_hidden, 1, 2*N_int_padded)`` fp32
* ``w13_weight_bias``  : ``(E, 1, 2*N_int_padded)`` fp32

For w2 (down):
* ``w2_weight``       : ``(E, K_packed = N_int_padded/2, hidden)`` uint8
* ``w2_weight_scale`` : ``(E, num_blocks_inter_padded, 1, hidden)`` fp32
* ``w2_weight_bias``  : ``(E, 1, hidden)`` fp32

The N-dim padding (intermediate) is gate/up-symmetric: gate and up are
each padded independently with zeros along their N axis so the
midpoint split inside the activation still lines up. w2's K dim is
padded to match the *padded* intermediate produced by w13 (since GMM2's
contracting dim must equal the post-activation tensor's last dim).
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

# block size is always 32 for MXFP4
MXFP4_BLOCK_SIZE = 32
PACKING = 2  # two FP4 codes per uint8 byte


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------


def _e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """Convert HF's e8m0 scale (uint8, biased exponent) to fp32 multiplier.

    e8m0 stores ``exp + 127`` in a single uint8 byte. The corresponding
    multiplier is ``2 ** exp``. Special case: e8m0 = 255 represents NaN
    in the strict spec, but HF never emits that for MoE weights — we
    treat all values uniformly.
    """
    assert scale_u8.dtype == torch.uint8, f"expected uint8 e8m0, got {scale_u8.dtype}"
    return torch.exp2(scale_u8.to(torch.int32).to(torch.float32) - 127.0)


def _flatten_block_bytes(packed_blocks: torch.Tensor) -> torch.Tensor:
    """``(..., num_blocks, 16)`` uint8 → ``(..., num_blocks * 16)`` uint8.

    Bytes from successive 32-element MXFP4 blocks become contiguous along
    the last axis. Total bytes per row = ``num_blocks * 16``, which holds
    ``num_blocks * 32`` packed FP4 values (since 2 codes per byte) =
    ``K`` elements.
    """
    assert packed_blocks.dtype == torch.uint8
    *leading, num_blocks, block_bytes = packed_blocks.shape
    assert block_bytes == 16, f"expected 16 bytes per block, got {block_bytes}"
    return packed_blocks.reshape(*leading, num_blocks * block_bytes)


def _deinterleave_axis1(t: torch.Tensor) -> torch.Tensor:
    """``[gate_0, up_0, gate_1, up_1, ...]`` → ``[gate||up]`` along axis 1.

    For ``(E, 2N, ...)`` input, produces ``(E, 2N, ...)`` where the first
    half along axis 1 is the gate rows and the second half is up.
    """
    E, two_N = t.shape[:2]
    assert two_N % 2 == 0
    rest = t.shape[2:]
    N = two_N // 2
    # (E, N, 2, *rest) → (E, 2, N, *rest) → (E, 2N, *rest)
    perm = (0, 2, 1) + tuple(range(3, 3 + len(rest)))
    return (
        t.reshape(E, N, 2, *rest).permute(*perm).contiguous().reshape(E, 2 * N, *rest)
    )


def _pad_gate_up_halves_axis1(
    t: torch.Tensor, n_per_half: int, pad_n: int
) -> torch.Tensor:
    """Pad gate and up halves independently along axis 1 with zeros.

    Input ``t`` shape: ``(E, 2 * n_per_half, *rest)`` with first half
    gate, second half up. Output shape: ``(E, 2 * (n_per_half + pad_n), *rest)``.
    """
    if pad_n == 0:
        return t
    E = t.shape[0]
    rest = t.shape[2:]
    gate = t[:, :n_per_half, ...]
    up = t[:, n_per_half:, ...]
    zeros = torch.zeros((E, pad_n, *rest), dtype=t.dtype, device=t.device)
    return torch.cat([gate, zeros, up, zeros], dim=1)


def _pad_last_axis(t: torch.Tensor, pad_amount: int) -> torch.Tensor:
    """Append ``pad_amount`` zeros along the last axis."""
    if pad_amount == 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[-1] = pad_amount
    return torch.cat(
        [t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=-1
    )


def process_mxfp4_moe_weights_bf16(
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    *,
    w13_interleaved: bool,
    padded_intermediate: int | None = None,
) -> tuple[
    torch.Tensor,  # w13_weight  (E, hidden_K_padded, 2*N_int_padded)  bfloat16
    torch.Tensor,  # w13_bias    (E, 1, 2*N_int_padded)                fp32
    torch.Tensor,  # w2_weight   (E, N_int_padded, hidden_N_padded)    bfloat16
    torch.Tensor,  # w2_bias     (E, 1, hidden_N_padded)               fp32
]:
    """Dequantize HF GPT-OSS MXFP4 weights to **bf16** in gmm_v2's (g, k, n)
    layout (the torchtpu-vllm "dequant at load" approach). Same layout as
    :func:`process_mxfp4_moe_weights_native`

    ``padded_intermediate`` is ``N_int_padded``, the width each of gate and up
    (and w2's K) is zero-padded to; by default the next multiple of 128.
    """
    E_w13, two_N_int, K_packed_w13 = w13_packed.shape
    N_int = two_N_int // 2
    hidden = K_packed_w13 * PACKING
    if padded_intermediate is None:
        padded_intermediate = -(-N_int // 128) * 128
    assert padded_intermediate >= N_int and padded_intermediate % 128 == 0, (
        f"padded_intermediate={padded_intermediate} must be a multiple of 128 "
        f"and at least the intermediate size {N_int}"
    )
    pad_n = padded_intermediate - N_int

    # w13: dequant (along K=hidden) → deinterleave gate/up → pad N → transpose.
    # ``w13_interleaved`` follows the checkpoint's gate/up layout: SWIGLUOAI
    # (GPT-OSS) stores w13 interleaved as [gate_0, up_0, gate_1, up_1, ...] and
    # must be deinterleaved into [gate.., up..] halves; a non-interleaved
    # checkpoint is already in that layout and skips this step. Mirrors
    # tpu-inference's ``w13_interleave = layer.activation == SWIGLUOAI``.
    w13 = MXFP4QuantizeUtil.dequantize(
        quantized_data=w13_packed,
        scale=w13_scale,
        dtype=torch.bfloat16,
        block_sizes=[MXFP4_BLOCK_SIZE],
    )  # (E, 2N, hidden) bf16
    if w13_interleaved:
        w13 = _deinterleave_axis1(w13)
    w13 = _pad_gate_up_halves_axis1(w13, N_int, pad_n)  # (E, 2*N_padded, hidden)
    w13 = w13.permute(0, 2, 1).contiguous()  # (E, hidden, 2*N_padded) bf16

    # requires K % 128 == 0 for gmm_v2 (the TPU lane width)
    w13 = torch.nn.functional.pad(
        w13, (0, 0, 0, (-hidden) % 128)
    )  # pad K (axis 1) → (E, hidden_padded, 2*N_padded)

    w13_b = w13_bias
    if w13_interleaved:
        w13_b = _deinterleave_axis1(w13_b.unsqueeze(-1)).squeeze(-1)
    w13_b = _pad_gate_up_halves_axis1(w13_b.unsqueeze(-1), N_int, pad_n).squeeze(-1)
    w13_b = w13_b.to(torch.float32).unsqueeze(1).contiguous()  # (E, 1, 2*N_padded)

    # w2: dequant (along K=intermediate) → pad K to match N_padded → transpose.
    w2 = MXFP4QuantizeUtil.dequantize(
        quantized_data=w2_packed,
        scale=w2_scale,
        dtype=torch.bfloat16,
        block_sizes=[MXFP4_BLOCK_SIZE],
    )  # (E, hidden, inter) bf16
    w2 = _pad_last_axis(w2, pad_n)  # (E, hidden, N_padded)
    w2 = w2.permute(0, 2, 1).contiguous()  # (E, N_padded, hidden) bf16

    # pad for irregular shape
    pad_n_out = (-hidden) % 128
    w2 = _pad_last_axis(w2, pad_n_out)  # (E, N_padded, hidden_padded_out)
    w2_b = w2_bias.to(torch.float32).unsqueeze(1).contiguous()  # (E, 1, hidden)
    w2_b = _pad_last_axis(w2_b, pad_n_out)  # (E, 1, hidden_padded_out)

    return w13, w13_b, w2, w2_b


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def process_mxfp4_moe_weights_native(
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    *,
    w13_interleaved: bool,
    align_to: int = 128,
) -> tuple[
    torch.Tensor,  # w13_weight  (E, K_packed = hidden/2, 2*N_int_padded) uint8
    torch.Tensor,  # w13_scale   (E, num_blocks_K, 1, 2*N_int_padded)    fp32
    torch.Tensor,  # w13_bias    (E, 1, 2*N_int_padded)                  fp32
    torch.Tensor,  # w2_weight   (E, K_packed = N_int_padded/2, hidden)  uint8
    torch.Tensor,  # w2_scale    (E, num_blocks_K_padded, 1, hidden)     fp32
    torch.Tensor,  # w2_bias     (E, 1, hidden)                          fp32
]:
    """Convert HF GPT-OSS MXFP4 weights to gmm_v2's native packed-FP4 layout.

    Lossless: HF's per-32-element scales are preserved (no requantization
    to a coarser block size). The kernel uses these scales directly via
    its block-wise quant code path.
    """
    # ============================================================
    # w13 (gate_up_proj): deinterleave + N-pad + transpose
    # ============================================================
    # The HF loader already flattens (num_blocks, 16) → (K_packed = K/2),
    # so we accept the 3D shape directly. ``num_blocks_hidden`` is
    # derived from the scale tensor's last axis.
    E_w13, two_N_int, K_packed_w13 = w13_packed.shape
    assert two_N_int % 2 == 0
    N_int = two_N_int // 2
    hidden = K_packed_w13 * PACKING  # K dim for w13's matmul
    assert hidden % MXFP4_BLOCK_SIZE == 0
    num_blocks_hidden = hidden // MXFP4_BLOCK_SIZE

    # Sanity-check companion tensors.
    assert w13_scale.shape == (E_w13, two_N_int, num_blocks_hidden), (
        f"w13_scale shape mismatch: got {w13_scale.shape}, "
        f"expected {(E_w13, two_N_int, num_blocks_hidden)}"
    )
    assert w13_bias.shape == (E_w13, two_N_int)

    pad_n = (align_to - N_int % align_to) % align_to
    N_int_padded = N_int + pad_n
    two_N_int_padded = 2 * N_int_padded

    # --- Weights ---
    # Step 1: deinterleave the 2N axis → first half gate, second half up
    #         (interleaved checkpoints only; see process_mxfp4_moe_weights_bf16)
    w13_w = _deinterleave_axis1(w13_packed) if w13_interleaved else w13_packed
    # Step 3: pad gate and up halves of N with zeros
    w13_w = _pad_gate_up_halves_axis1(w13_w, N_int, pad_n)  # (E, 2*N_padded, K_packed)
    # Step 4: transpose to gmm_v2 (E, K_packed, N) layout
    w13_w = w13_w.permute(0, 2, 1).contiguous()
    assert w13_w.shape == (E_w13, K_packed_w13, two_N_int_padded), w13_w.shape

    # --- Scales (e8m0 → fp32) ---
    # HF layout: (E, 2N, num_blocks) uint8.
    # Deinterleave first (same gate/up axis as weights).
    w13_s = (
        _deinterleave_axis1(w13_scale) if w13_interleaved else w13_scale
    )  # (E, 2N, num_blocks)
    # Pad N with zeros (corresponding fp4 weights are zero → no contribution).
    w13_s = _pad_gate_up_halves_axis1(
        w13_s, N_int, pad_n
    )  # (E, 2*N_padded, num_blocks)
    # Convert e8m0 → fp32.
    w13_s = _e8m0_to_fp32(w13_s)  # (E, 2*N_padded, num_blocks) fp32
    # Transpose to (E, num_blocks, 2*N_padded) and add the size-1 mid dim.
    # Unsqueeze before contiguous: torch_tpu re-materializes a rank-4 view of a
    # rank-3 buffer on every compiled call.
    w13_s = (
        w13_s.permute(0, 2, 1).unsqueeze(2).contiguous()
    )  # (E, num_blocks, 1, 2*N_padded)
    assert w13_s.shape == (E_w13, num_blocks_hidden, 1, two_N_int_padded)

    # --- Bias (bf16 → fp32) ---
    # Deinterleave needs at least one trailing dim; use a temporary unsqueeze.
    w13_b = (
        _deinterleave_axis1(w13_bias.unsqueeze(-1)).squeeze(-1)
        if w13_interleaved
        else w13_bias
    )  # (E, 2N)
    w13_b = _pad_gate_up_halves_axis1(w13_b.unsqueeze(-1), N_int, pad_n).squeeze(
        -1
    )  # (E, 2*N_padded)
    w13_b = w13_b.to(torch.float32).unsqueeze(1).contiguous()  # (E, 1, 2*N_padded)
    assert w13_b.shape == (E_w13, 1, two_N_int_padded)

    # ============================================================
    # w2 (down_proj): K-pad to match w13's intermediate, then transpose
    # ============================================================
    # Same 3D loader-flat layout as w13: ``(E, hidden, K_packed=K/2)``.
    E_w2, hidden_w2, K_packed_w2_raw = w2_packed.shape
    assert E_w2 == E_w13
    assert hidden_w2 == hidden
    inter = K_packed_w2_raw * PACKING  # K dim for w2 = intermediate_size
    assert inter % MXFP4_BLOCK_SIZE == 0
    num_blocks_inter = inter // MXFP4_BLOCK_SIZE
    assert inter == N_int, f"w2's K = {inter} should equal w13's N_int = {N_int}"

    # We need to pad w2's K to match w13's padded N_int.
    pad_k_elements = pad_n
    pad_k_packed = pad_k_elements // PACKING  # padded bytes per row
    pad_k_num_blocks = pad_k_elements // MXFP4_BLOCK_SIZE  # padded scale blocks
    # (For align_to=128 and MXFP4_BLOCK_SIZE=32, pad_n is a multiple of 32 iff
    #  align_to is. 128 % 32 == 0, so this holds.)
    assert pad_k_elements % MXFP4_BLOCK_SIZE == 0, (
        f"align_to={align_to} must be a multiple of MXFP4_BLOCK_SIZE={MXFP4_BLOCK_SIZE}"
    )
    num_blocks_inter_padded = num_blocks_inter + pad_k_num_blocks

    # --- Weights ---
    # Step 1: pad K_packed (zeros append along last axis → zero fp4 weights).
    w2_w = _pad_last_axis(w2_packed, pad_k_packed)  # (E, hidden, inter_padded/2)
    # Step 2: transpose to gmm_v2 (E, K_packed, N) layout where N = hidden.
    w2_w = w2_w.permute(0, 2, 1).contiguous()
    K_packed_w2 = w2_w.shape[1]
    assert w2_w.shape == (E_w2, K_packed_w2, hidden), w2_w.shape

    # --- Scales ---
    w2_s = _pad_last_axis(w2_scale, pad_k_num_blocks)  # (E, hidden, num_blocks_padded)
    w2_s = _e8m0_to_fp32(w2_s)  # fp32
    w2_s = (
        w2_s.permute(0, 2, 1).unsqueeze(2).contiguous()
    )  # (E, num_blocks_padded, 1, hidden)
    assert w2_s.shape == (E_w2, num_blocks_inter_padded, 1, hidden)

    # --- Bias ---
    w2_b = w2_bias.to(torch.float32).unsqueeze(1).contiguous()  # (E, 1, hidden)
    assert w2_b.shape == (E_w2, 1, hidden)

    return w13_w, w13_s, w13_b, w2_w, w2_s, w2_b


# torch_tpu returns wrong values from the MXFP4 dequantization once its
# float32 intermediates (one element per 4-bit value, two per packed byte)
# pass 2**30 elements, i.e. 4 GiB; measured correct at 1.06e9, wrong at
# 1.08e9. A layer above that is converted a few experts at a time.
_MAX_DEQUANT_ELEMENTS = 2**30
_EXPERTS_PER_CHUNK = 16


def _process_mxfp4_moe_weights_in_chunks(
    layer: torch.nn.Module, w13_interleaved: bool, padded_intermediate: int
):
    """``process_mxfp4_moe_weights_bf16`` over expert chunks, concatenated one
    output at a time so the chunks of a finished output are freed before the
    next one is assembled."""
    num_experts = layer.w13_weight.data.shape[0]
    outputs = ([], [], [], [])
    for i in range(0, num_experts, _EXPERTS_PER_CHUNK):
        chunk = process_mxfp4_moe_weights_bf16(
            layer.w13_weight.data[i : i + _EXPERTS_PER_CHUNK],
            layer.w13_weight_scale.data[i : i + _EXPERTS_PER_CHUNK],
            layer.w13_weight_bias.data[i : i + _EXPERTS_PER_CHUNK],
            layer.w2_weight.data[i : i + _EXPERTS_PER_CHUNK],
            layer.w2_weight_scale.data[i : i + _EXPERTS_PER_CHUNK],
            layer.w2_weight_bias.data[i : i + _EXPERTS_PER_CHUNK],
            w13_interleaved=w13_interleaved,
            padded_intermediate=padded_intermediate,
        )
        for parts, part in zip(outputs, chunk):
            parts.append(part)
    results = []
    for parts in outputs:
        results.append(torch.cat(parts, dim=0))
        parts.clear()
    return tuple(results)


def _gmm_padded_intermediate(
    hidden_size: int, intermediate_size: int, activation: str
) -> int:
    """Intermediate width at which the fused MoE's first GEMM runs the fewest
    N tiles on this TPU.

    ``_fused_moe_jax`` runs GMM1 on bf16 activations and these bf16 weights,
    contracting over hidden padded to 128 lanes, with the gated activation
    fused, so N is tiled per gate/up half; see
    ``gmm_v2.pad_size_n_to_fewest_tiles``. The tile depends on the chip's VMEM:
    gpt-oss's 2880 pads to 3072 on tpu7x (64 MiB) and to 2944, the next
    multiple of 128, where 2944 fits one tile (128 MiB).
    """
    from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
    from sglang.srt.layers.moe.pallas.fused_moe_op import _fuse_act_name
    from sglang.srt.layers.moe.pallas.gmm_v2 import pad_size_n_to_fewest_tiles

    fuse_act = "situ:0:none" if activation == "situ" else _fuse_act_name(activation)
    # JAX runs CPU-only next to TorchTPU; this swaps in the physical TPU's
    # metadata, the VMEM the kernel's tiles are sized against.
    with tpu_info_for_cpu_export():
        return pad_size_n_to_fewest_tiles(
            size_k=-(-hidden_size // 128) * 128,
            size_n=intermediate_size,
            fuse_act=fuse_act,
        )


def prepare_tpu_mxfp4_moe_weights(layer: torch.nn.Module, *, with_bias: bool) -> None:
    """Format MXFP4 MoE weights for the TPU Pallas gmm_v2 backend.

    Non-interleaved checkpoints with a lane-aligned hidden size (e.g. Kimi-K3)
    keep packed FP4 + per-32 scales and dequantize in-kernel
    (:func:`process_mxfp4_moe_weights_native`). Interleaved GPT-OSS (SWIGLUOAI)
    stays on the bf16 dequant-at-load path, with its intermediate zero-padded
    to the width at which the kernel runs the fewest N tiles
    (:func:`_gmm_padded_intermediate`): gpt-oss's 2880 becomes 3072 on tpu7x,
    4.3% more MoE weight bytes than 2944, for a first GEMM that runs 3 N tiles
    instead of 23.
    """
    activation = layer.moe_runner_config.activation
    # GPT-OSS (SWIGLUOAI) ships w13 interleaved as [gate_0, up_0, ...].
    w13_interleaved = activation == "swigluoai"
    # The packed path does not pad K, so it needs a lane-aligned hidden size.
    hidden_aligned = (layer.w13_weight.shape[-1] * PACKING) % 128 == 0
    w13_scale = w2_scale = None
    if not w13_interleaved and hidden_aligned:
        w13, w13_scale, w13_bias, w2, w2_scale, w2_bias = (
            process_mxfp4_moe_weights_native(
                layer.w13_weight.data,
                layer.w13_weight_scale.data,
                layer.w13_weight_bias.data,
                layer.w2_weight.data,
                layer.w2_weight_scale.data,
                layer.w2_weight_bias.data,
                w13_interleaved=False,
            )
        )
        w13_scale = Parameter(w13_scale, requires_grad=False)
        w2_scale = Parameter(w2_scale, requires_grad=False)
    else:
        _, two_n_int, k_packed = layer.w13_weight.data.shape
        padded_intermediate = _gmm_padded_intermediate(
            k_packed * PACKING, two_n_int // 2, activation
        )
        if 2 * layer.w13_weight.data.numel() > _MAX_DEQUANT_ELEMENTS:
            # All 128 experts of gpt-oss-120b on one device (a pipeline stage:
            # 2.1e9 values before padding) are processed a few at a time.
            w13, w13_bias, w2, w2_bias = _process_mxfp4_moe_weights_in_chunks(
                layer, w13_interleaved, padded_intermediate
            )
        else:
            w13, w13_bias, w2, w2_bias = process_mxfp4_moe_weights_bf16(
                layer.w13_weight.data,
                layer.w13_weight_scale.data,
                layer.w13_weight_bias.data,
                layer.w2_weight.data,
                layer.w2_weight_scale.data,
                layer.w2_weight_bias.data,
                w13_interleaved=w13_interleaved,
                padded_intermediate=padded_intermediate,
            )
    layer.w13_weight = Parameter(w13, requires_grad=False)
    layer.w13_weight_scale = w13_scale
    layer.w13_weight_bias = Parameter(w13_bias, requires_grad=False)
    layer.w2_weight = Parameter(w2, requires_grad=False)
    layer.w2_weight_scale = w2_scale
    layer.w2_weight_bias = Parameter(w2_bias, requires_grad=False)
    # Without checkpoint biases create_weights zero-fills them; skip the adds.
    if not with_bias:
        layer.w13_weight_bias = None
        layer.w2_weight_bias = None
