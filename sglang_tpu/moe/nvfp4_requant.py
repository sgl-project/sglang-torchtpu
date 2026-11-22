# SPDX-License-Identifier: Apache-2.0
"""NVFP4 expert requantization to wider scale blocks (pure torch, any device)."""

from __future__ import annotations

import torch

# Served requant blocks: the decode megakernel's and fused-RS's W4A16/W4A8 block-scaled arms.
QBLOCK_SIZES = (256, 512, 1024, 2048, 4096)
NVFP4_SERVED_GROUP = 16
_E2M1_MAX = 6.0
# e2m1 magnitude midpoints; True where a tie rounds up (to the even code).
_E2M1_MIDPOINTS = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    exp = ((codes >> 1) & 3).float()
    mant = (codes & 1).float()
    mag = torch.where(exp == 0, 0.5 * mant, (1.0 + 0.5 * mant) * torch.exp2(exp - 1.0))
    return torch.where(codes >= 8, -mag, mag)


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    e, k, half_n = packed.shape
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(e, k, 2 * half_n)
    return _e2m1_values(codes) * scale.float().repeat_interleave(
        k // scale.shape[1], dim=1
    )


def requant_nvfp4_blocks(
    packed: torch.Tensor, scale: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused-RS NVFP4 [E, K, N/2] uint8 (low nibble column 2m) + [E, K/16, N] scales -> codes
    re-rounded to one bf16 scale (amax / 6) per block K rows, and those scales [E, K/block, N].
    """
    e, k, half_n = packed.shape
    n = 2 * half_n
    if k % block or block % NVFP4_SERVED_GROUP:
        raise ValueError(f"cannot requantize K={k} to blocks of {block}")
    w = _dequant(packed, scale)
    wb = w.reshape(e, k // block, block, n)
    sb = (wb.abs().amax(dim=2) / _E2M1_MAX).to(torch.bfloat16)
    # |w| against midpoint x scale: both products are exact in f32 (few significant bits), so
    # ties resolve exactly where a TPU division would round them away.
    sbf = torch.where(sb > 0, sb, 1.0).float()[:, :, None]
    a = wb.abs()
    idx = torch.zeros_like(a, dtype=torch.uint8)
    for mid, tie_up in _E2M1_MIDPOINTS:
        idx += (a >= mid * sbf if tie_up else a > mid * sbf).to(torch.uint8)
    out = (idx | (wb < 0).to(torch.uint8) << 3).reshape(e, k, half_n, 2)
    return out[..., 0] | (out[..., 1] << 4), sb


def probe_nvfp4_blocks(
    packed: torch.Tensor, scale: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """requant_nvfp4_blocks written back in the input's block-16 format (each block's scale
    repeated over its 16-row groups), plus the squared L2 norms of the weight error and of
    the original weights."""
    codes, sb = requant_nvfp4_blocks(packed, scale, block)
    s16 = sb.to(scale.dtype).repeat_interleave(block // NVFP4_SERVED_GROUP, dim=1)
    ref = _dequant(packed, scale)
    err = _dequant(codes, s16) - ref
    return codes, s16, err.square().sum(), ref.square().sum()


def _quantize_wb_two_level_pow2(
    wb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE-calibrated two-level power-of-2 quantization of ``wb [E, K_blks, block, N]`` (f32).

    Finds per-(expert, col) bf16 base scale ``c[e, n]`` and integer shift ``p[e, kb, n]`` in
    ``[-5, +6]`` (12 normal octaves, ``p_max - p <= 11``) minimizing ``||rec - wb||_2^2``, and
    returns ``(packed_u8 [E, K, N//2], s_2l_bf16 [E, K_blks, N], err_sq_sum)``.
    Because ``s_2l = c * 2^p`` is stored as standard ``bfloat16`` block scales, decode kernels
    consume ``s_2l`` without modification while ``inner_kernel`` folds ``2^p`` into ``float8_e4m3fn``
    exponents losslessly for full-K / K1024 MXU accumulation.
    """
    e, n_blks, block, n = wb.shape
    k = n_blks * block
    col_amax = wb.abs().amax(dim=(1, 2))  # [E, N]
    blk_amax = wb.abs().amax(dim=2)  # [E, n_blks, N]
    best_err = torch.full((e, n), float("inf"), dtype=torch.float32, device=wb.device)
    best_codes = torch.zeros((e, k, n // 2), dtype=torch.uint8, device=wb.device)
    best_sb = torch.zeros((e, n_blks, n), dtype=torch.bfloat16, device=wb.device)
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=wb.device,
    )
    a_abs = wb.abs()
    sign_bit = (wb < 0).to(torch.uint8) << 3

    for alpha in (0.76, 0.82, 0.88, 0.94, 1.00):
        c_bf16 = ((col_amax * alpha) / (_E2M1_MAX * 64.0)).to(torch.bfloat16)
        c_f32 = c_bf16.float()
        c_safe = torch.where(c_f32 > 0, c_f32, torch.ones_like(c_f32))
        raw_p = torch.log2(
            torch.clamp(blk_amax / (_E2M1_MAX * c_safe[:, None, :]), min=2.0**-5)
        )
        p_lo = torch.clamp(torch.floor(raw_p), -5.0, 6.0)
        p_hi = torch.clamp(torch.ceil(raw_p), -5.0, 6.0)

        cand_idx = None
        cand_sb = None
        cand_err = None
        for p_cand in (p_lo, p_hi):
            sb = (c_f32[:, None, :] * torch.exp2(p_cand)).to(torch.bfloat16)
            sbf = torch.where(sb > 0, sb, torch.ones_like(sb)).float()[:, :, None, :]
            idx = torch.zeros_like(a_abs, dtype=torch.uint8)
            for mid, tie_up in _E2M1_MIDPOINTS:
                idx += (a_abs >= mid * sbf if tie_up else a_abs > mid * sbf).to(
                    torch.uint8
                )
            mag = e2m1_lut[idx.long()] * sb.float()[:, :, None, :]
            rec = torch.where(wb < 0, -mag, mag)
            err_b = (rec - wb).square().sum(dim=2)  # [E, n_blks, N]
            if cand_err is None:
                cand_idx, cand_sb, cand_err = idx, sb, err_b
            else:
                imp_b = err_b < cand_err
                cand_idx = torch.where(imp_b[:, :, None, :], idx, cand_idx)
                cand_sb = torch.where(imp_b, sb, cand_sb)
                cand_err = torch.minimum(cand_err, err_b)

        col_err = cand_err.sum(dim=1)  # [E, N]
        imp_c = col_err < best_err
        packed_pair = (cand_idx | sign_bit).reshape(e, k, n // 2, 2)
        packed_c = packed_pair[..., 0] | (packed_pair[..., 1] << 4)
        imp_pair = imp_c.reshape(e, 1, n // 2, 2)
        # Update even/odd columns independently before nibble-packing:
        cur_lo = best_codes & 0xF
        cur_hi = best_codes >> 4
        new_lo = torch.where(imp_pair[..., 0], packed_pair[..., 0], cur_lo)
        new_hi = torch.where(imp_pair[..., 1], packed_pair[..., 1], cur_hi)
        best_codes = new_lo | (new_hi << 4)
        best_sb = torch.where(imp_c[:, None, :], cand_sb, best_sb)
        best_err = torch.minimum(best_err, col_err)

    return best_codes, best_sb, best_err.sum()


def compact_nvfp4_two_level_pow2(
    packed: torch.Tensor, scale: torch.Tensor, block: int = 256
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE-calibrated two-level pow2 requantization returning ``(codes, s_2l, err2, ref2)``."""
    e, k, half_n = packed.shape
    n = 2 * half_n
    if k % block or block % NVFP4_SERVED_GROUP:
        raise ValueError(f"cannot requantize K={k} to two-level blocks of {block}")
    w = _dequant(packed, scale)
    wb = w.reshape(e, k // block, block, n)
    codes, sb, err2 = _quantize_wb_two_level_pow2(wb)
    return codes, sb.to(scale.dtype), err2, w.square().sum()


def compact_nvfp4_blocks(
    packed: torch.Tensor, scale: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """requant_nvfp4_blocks returning compact [E, K // block, N] bf16 scales (without repeating
    each block scale 16x), plus the squared L2 norms of the weight error and original weights."""
    codes, sb = requant_nvfp4_blocks(packed, scale, block)
    sb_out = sb.to(scale.dtype)
    ref = _dequant(packed, scale)
    err = _dequant(codes, sb_out) - ref
    return codes, sb_out, err.square().sum(), ref.square().sum()


_COMPACT_EXPERT_CHUNK = 4
_COMPILED_COMPACT = None


def _get_compiled_compact():
    global _COMPILED_COMPACT
    if _COMPILED_COMPACT is None:
        _COMPILED_COMPACT = torch.compile(
            compact_nvfp4_blocks, fullgraph=True, dynamic=False
        )
    return _COMPILED_COMPACT


def _copy_chunk(t: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(t).copy_(t)


def compact_tpu_nvfp4_block_scales(layer: torch.nn.Module, block: int) -> list[float]:
    """SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES / COMPACT_FP4_QBLOCK / FP4_TWO_LEVEL_POW2:
    requantize a fused-RS-prepared layer's fp4 experts in 4-expert chunks to compact
    [E, K // block, N] bf16 scales without marking qblock_layout."""
    from sglang.srt.layers.utils import copy_or_rebind_param
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.expert_loading import _wait_for
    from sglang_tpu.fp4_moe import _materialize_packed_fp4
    from sglang_tpu.fp8_moe import _free_replaced_storage

    qblock_override = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK.get() or 0)
    if qblock_override > 0:
        block = qblock_override
    if block not in QBLOCK_SIZES:
        raise NotImplementedError(
            f"NVFP4 compact block scales to block {block}: supported {QBLOCK_SIZES}"
        )
    use_two_level = (
        int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2.get() or 0) == 1
        and block == 256
    )
    compiled_fn = (
        compact_nvfp4_two_level_pow2
        if use_two_level
        else (_get_compiled_compact() if layer.w13_weight.device.type == "tpu" else compact_nvfp4_blocks)
    )
    errs: list[float] = []
    for name in ("w13_weight", "w2_weight"):
        w = getattr(layer, name).data
        s = getattr(layer, f"{name}_scale").data
        if (
            w.dtype != torch.float4_e2m1fn_x2
            or s.dtype != torch.bfloat16
            or s.shape[1] * NVFP4_SERVED_GROUP != w.shape[1]
        ):
            raise NotImplementedError(
                "NVFP4 compact block scales need fp4 experts with bf16 block-16 scales, "
                f"got {w.dtype} {tuple(w.shape)} with {s.dtype} {tuple(s.shape)}"
            )
        u = w.view(torch.uint8)
        codes, scales, err2, ref2 = [], [], 0.0, 0.0
        for e0 in range(0, w.shape[0], _COMPACT_EXPERT_CHUNK):
            c, sb, d2, r2 = compiled_fn(
                _copy_chunk(u[e0 : e0 + _COMPACT_EXPERT_CHUNK]),
                _copy_chunk(s[e0 : e0 + _COMPACT_EXPERT_CHUNK]),
                block,
            )
            codes.append(c)
            scales.append(sb)
            err2 += d2.item()
            ref2 += r2.item()
        new_w = _materialize_packed_fp4(torch.cat(codes))
        new_s = torch.cat(scales)
        del codes, scales
        _wait_for(new_w, new_s)
        copy_or_rebind_param(layer, name, new_w)
        copy_or_rebind_param(layer, f"{name}_scale", new_s)
        _free_replaced_storage(w, getattr(layer, name).data)
        _free_replaced_storage(s, getattr(layer, f"{name}_scale").data)
        del w, s, u, new_w, new_s
        errs.append((err2 / ref2) ** 0.5 if ref2 else 0.0)
    return errs


def quantize_fp8_to_nvfp4_blocks(
    w_fp8: torch.Tensor, scale: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize channelwise FP8 experts ``w_fp8 [E, K, N]`` + ``scale [E, ..., N]``
    into packed ``uint8 [E, K, N // 2]`` E2M1 nibbles + ``bf16 [E, K // block, N]`` scales,
    plus ``(err_sq_sum, ref_sq_sum)`` for relative L2 telemetry."""
    e, k, n = w_fp8.shape
    if n % 2 != 0 or k % block != 0 or block not in (16, *QBLOCK_SIZES):
        raise ValueError(
            f"Unsupported FP8->NVFP4 shapes: w={tuple(w_fp8.shape)}, block={block}"
        )
    s_2d = scale.reshape(e, 1, n).float()
    w = w_fp8.float() * s_2d
    wb = w.reshape(e, k // block, block, n)
    sb = (wb.abs().amax(dim=2) / _E2M1_MAX).to(torch.bfloat16)
    sbf = torch.where(sb > 0, sb, 1.0).float()[:, :, None]
    a = wb.abs()
    idx = torch.zeros_like(a, dtype=torch.uint8)
    for mid, tie_up in _E2M1_MIDPOINTS:
        idx += (a >= mid * sbf if tie_up else a > mid * sbf).to(torch.uint8)
    packed_pair = (idx | ((wb < 0).to(torch.uint8) << 3)).reshape(e, k, n // 2, 2)
    codes = packed_pair[..., 0] | (packed_pair[..., 1] << 4)
    deq = _dequant(codes, sb)
    err = deq - w
    return codes, sb, err.square().sum(), w.square().sum()


_COMPILED_FP8_TO_FP4 = None


def _get_compiled_fp8_to_fp4():
    global _COMPILED_FP8_TO_FP4
    if _COMPILED_FP8_TO_FP4 is None:
        _COMPILED_FP8_TO_FP4 = torch.compile(
            quantize_fp8_to_nvfp4_blocks, fullgraph=True, dynamic=False
        )
    return _COMPILED_FP8_TO_FP4


def convert_tpu_fp8_moe_to_nvfp4(
    layer: torch.nn.Module, block: int, *, in_place: bool = False
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Convert a channelwise FP8 MoE expert module (``L0`` / ``L61``) in 4-expert chunks
    to ``float4_e2m1fn_x2`` + ``bfloat16 [E, K // block, N]`` scales.

    When ``in_place=True``, replaces ``w13_weight`` / ``w2_weight`` and frees the FP8
    storage (unifying prefill ``gmm_v2`` and decode ``muse_sparse`` on NVFP4). When
    ``in_place=False``, returns a decode-only shadow dict preserving exact FP8 prefill.
    """
    from sglang.srt.layers.utils import copy_or_rebind_param
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.expert_loading import _wait_for
    from sglang_tpu.fp4_moe import _materialize_packed_fp4
    from sglang_tpu.fp8_moe import _free_replaced_storage

    if in_place:
        qblock_override = int(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK.get() or 0
        )
        if qblock_override > 0:
            block = qblock_override
    if block not in (16, *QBLOCK_SIZES):
        raise ValueError(
            f"FP8->NVFP4 boundary block must be 16 or {QBLOCK_SIZES}, got {block}"
        )
    use_two_level = (
        in_place
        and int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2.get() or 0) == 1
        and block == 256
    )

    def _two_level_from_fp8(w_f8: torch.Tensor, s_ch: torch.Tensor, blk: int):
        ef, kf, nf = w_f8.shape
        wf = w_f8.float() * s_ch.reshape(ef, 1, nf).float()
        c_u8, sb_b, err2_t = _quantize_wb_two_level_pow2(
            wf.reshape(ef, kf // blk, blk, nf)
        )
        return c_u8, sb_b, err2_t, wf.square().sum()

    fn = (
        _two_level_from_fp8
        if use_two_level
        else (
            _get_compiled_fp8_to_fp4()
            if layer.w13_weight.device.type == "tpu"
            else quantize_fp8_to_nvfp4_blocks
        )
    )
    out_tensors: dict[str, torch.Tensor] = {}
    errs: list[float] = []
    for name in ("w13_weight", "w2_weight"):
        w = getattr(layer, name).data
        s = getattr(layer, f"{name}_scale").data
        codes, scales, err2, ref2 = [], [], 0.0, 0.0
        for e0 in range(0, w.shape[0], _COMPACT_EXPERT_CHUNK):
            c, sb, d2, r2 = fn(
                _copy_chunk(w[e0 : e0 + _COMPACT_EXPERT_CHUNK]),
                _copy_chunk(s[e0 : e0 + _COMPACT_EXPERT_CHUNK]),
                block,
            )
            codes.append(c)
            scales.append(sb)
            err2 += float(d2.item())
            ref2 += float(r2.item())
        new_w = _materialize_packed_fp4(torch.cat(codes))
        new_s = torch.cat(scales)
        del codes, scales
        _wait_for(new_w, new_s)
        errs.append((err2 / ref2) ** 0.5 if ref2 else 0.0)
        if in_place:
            copy_or_rebind_param(layer, name, new_w)
            copy_or_rebind_param(layer, f"{name}_scale", new_s)
            _free_replaced_storage(w, getattr(layer, name).data)
            _free_replaced_storage(s, getattr(layer, f"{name}_scale").data)
            del w, s, new_w, new_s
        else:
            out_tensors[name] = new_w
            out_tensors[f"{name}_scale"] = new_s
    return out_tensors, errs
