# SPDX-License-Identifier: Apache-2.0
"""Wave-8 Decode Collective Bridges, Token-Sharded Stream, Fused QKV/RoPE Epilogue, & Entry/Exit.

Advances Wave-7 ``decode_collective_bridges.py`` with four zero-spill Pallas upgrades:
  1. Token-sharded residual ``stream [B // TP, 8192]`` + chipmate-sharded ``TP_chip=2``
     FP8 ``W_pre [8192, 2048]`` & ``W_post [4096, 4096]`` across Bridge 1b, Bridge 2b,
     Entry, and L61 Final Norm (collapsing ``14 -> 9`` remote-copy sync stages and ``7 -> 4``
     inter-chip ICI stages per layer while cutting VPU RMSNorm/residual work ``8x``!).
  2. In-VMEM fused decode QKV RMSNorm + interleaved FP32 RoPE + ``KV_PACK_HEADS=1`` + FP8 KV
     quantization inside Bridge 2b and L0 ``fp8_dual_qkv_gate``, eliminating ``[B, 1280] ->
     [B, 10, 128]`` ``as_strided`` sublane transposes and 62 separate epilogue kernel launches.
  3. Final-layer L61 ``moe_post_ag_final_norm`` (and token-sharded variant) fusing ``model.norm``
     and skipping the dead ``2 MiB`` L61 ``out_stream_hbm`` write (100% bit-exact).
  4. Pallas ``decode_vocab_embed_norm`` (and token-sharded entry variant) replacing
     ``VocabParallelEmbedding``'s ``1 MiB`` XLA ring ``all-reduce`` + eager RMSNorm.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from collections.abc import Callable
from threading import Lock

from sglang_tpu.kernels.decode_collective_bridges import (
    SUPPORTED_WORLD_SIZES,
    _COL_STRIP,
    _DOT_TN,
    _dma_wait,
    _dot_fp8_row_block_to_ref,
    _dot_fp8_scaled_to_ref,
    _fori_col_blocks,
    _fori_strips,
    _gather_shard_xor1,
    _gather_token_shard_xor1_and_dot_post,
    _get_tp_mesh,
    _pick_strip,
    _prescale_s_buf_in_vmem,
    _reduce_col_blocks,
    _run_post_ffn_slice_epilogue,
    _run_rep_b1_vmem_core,
    _signal_barrier,
    _wait_barrier,
)

_W8_CID_BASE = 40
_W8_LOCK = Lock()

_MOE_POST_FINAL_NORM_OPS: dict[tuple[int, bool, bool], Callable] = {}
_MOE_TO_ATTN_QKV_EPI_OPS: dict[tuple[int, bool, bool, bool], Callable] = {}
_DUAL_QKV_EPI_GATE_OPS: dict[bool, Callable] = {}
_EMBED_NORM_OPS: dict[int, Callable] = {}
_EMBED_TOK_SHARDED_OPS: dict[int, Callable] = {}
_ATTN_TO_MOE_TOK_SH_OPS: dict[tuple[int, bool, bool], Callable] = {}
_MOE_TO_ATTN_TOK_SH_OPS: dict[tuple[int, bool, bool, bool], Callable] = {}
_MOE_POST_TOK_SH_FINAL_OPS: dict[tuple[int, bool, bool], Callable] = {}
_EMBED_CHIP_SHARDED_OPS: dict[int, Callable] = {}
_ATTN_TO_MOE_CHIP_SH_OPS: dict[tuple[int, bool, bool | None, bool], Callable] = {}
_MOE_TO_ATTN_CHIP_SH_OPS: dict[tuple[int, bool, bool, bool], Callable] = {}
_MOE_POST_CHIP_SH_FINAL_OPS: dict[tuple[int, bool, bool], Callable] = {}
_AG_CHIP_CAP_OPS: dict[int, Callable] = {}


def _as_2d_f32(x, cols: int):
    import jax.numpy as jnp

    if x.ndim == 2 and x.shape == (1, cols) and x.dtype == jnp.float32:
        return x
    return x.reshape(1, cols).astype(jnp.float32)


def _as_dot_scale_f32(x, cols: int):
    """Preserve 1D `{0:T(1024)}` scales when `cols % 1024 != 0` (`1280`, `640`, `512`) to avoid HLO `reshape`."""
    import jax.numpy as jnp

    if cols % 1024 != 0:
        if x.ndim == 1 and x.shape == (cols,) and x.dtype == jnp.float32:
            return x
        return x.reshape(cols).astype(jnp.float32)
    return _as_2d_f32(x, cols)


def _qepi_norm_before_s3_enabled() -> bool:
    """True unless ``SGLANG_TPU_MUSE_DECODE_QEPI_NORM_BEFORE_S3=0`` selects legacy Stage-3."""
    import os

    return os.environ.get("SGLANG_TPU_MUSE_DECODE_QEPI_NORM_BEFORE_S3", "1").strip() not in (
        "0",
        "false",
        "False",
    )


def _run_my_half_post_ffn_norm_epilogue(
    s_my_buf,
    post_my_buf,
    post_ff_gain_ref,
    pf_beta_ref,
    norm2_gain_ref,
    out_norm2_ref,
    *,
    hidden: int,
    my_r0,
    n_r: int,
    align: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    norm2_eps: float,
    norm2_scale_input: bool,
    cast_stream_bf16: bool = False,
) -> None:
    """Zero-spill post-FFN norm + residual + next/final norm on ``my_half`` rows ONLY."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    bm = _pick_strip(n_r, 32)

    def _strip(r_loc):
        rs_l = pl.ds(r_loc, bm)
        rs_g = pl.ds(pl.multiple_of(my_r0 + r_loc, align), bm)

        def _bvar(c_sl, acc):
            bc = post_my_buf[rs_l, c_sl].astype(jnp.float32)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_sl]
            sq = bc * bc
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        b_sq = _reduce_col_blocks(
            hidden, _bvar, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        inv_b = lax.rsqrt(
            jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )

        def _res_nvar(c_sl, acc):
            bc = post_my_buf[rs_l, c_sl].astype(jnp.float32)
            bg = post_ff_gain_ref[0:1, c_sl]
            b_n = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
            ns = s_my_buf[rs_l, c_sl] + pf_beta_ref[0:1, c_sl] * b_n
            if cast_stream_bf16:
                ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
            s_my_buf[rs_l, c_sl] = ns
            nv = ns * norm2_gain_ref[0:1, c_sl] if norm2_scale_input else ns
            sq = nv * nv
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        n_sq = _reduce_col_blocks(
            hidden, _res_nvar, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        inv_n = lax.rsqrt(
            jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(norm2_eps)
        )

        def _nout(c_sl):
            sv = s_my_buf[rs_l, c_sl]
            ng = norm2_gain_ref[0:1, c_sl]
            nn = (sv * ng) * inv_n if norm2_scale_input else (sv * inv_n) * ng
            out_norm2_ref[rs_g, c_sl] = nn.astype(out_norm2_ref.dtype)

        _fori_col_blocks(hidden, _nout)

    _fori_strips(n_r, bm, _strip)


def _pallas_call_maybe_interpret(kernel_fn, *, interpret: bool, compiler_params, **kwargs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if isinstance(interpret, pltpu.InterpretParams):
        return pl.pallas_call(
            kernel_fn,
            compiler_params=compiler_params,
            interpret=interpret,
            **kwargs,
        )
    if interpret:
        return pl.pallas_call(
            kernel_fn,
            compiler_params=compiler_params,
            interpret=pltpu.InterpretParams(detect_races=False),
            **kwargs,
        )
    return pl.pallas_call(kernel_fn, compiler_params=compiler_params, **kwargs)


# =============================================================================
# 1. Shared In-VMEM Decode QKV + RoPE + FP8 KV Quantization Epilogue
# =============================================================================
def _run_in_vmem_qkv_epilogue(
    qkv_2d_ref,  # bf16[B, 1280] in VMEM (no 3D sublane transpose!)
    cos_sin_vmem_ref,  # f32[B, 2, 128] or None
    q_out_vmem_ref,  # bf16[B, 16, 128]
    kv_out_vmem_ref,  # fp8[B, 1, 4, 128]
    *,
    norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Bit-exact in-VMEM fusion of ``decode_qkv_epilogue._decode_qkv_epilogue_kernel``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    fp8_min = float(jnp.finfo(jnp.float8_e4m3fn).min)
    b_rows = qkv_2d_ref.shape[0]
    tile_r = 32 if (b_rows >= 32 and b_rows % 32 == 0) else (16 if b_rows % 16 == 0 else 8)
    n_tiles = b_rows // tile_r
    lane_iota = lax.broadcasted_iota(jnp.int32, (tile_r, 128), 1)
    mask_lo = lane_iota < 64
    even_lane = (lane_iota & 1) == 0
    inv_k_scale = 1.0 / float(k_scale)
    inv_v_scale = 1.0 / float(v_scale)

    def _roll(x, shift: int):
        if interpret_mode:
            return jnp.roll(x, shift=shift, axis=1)
        return pltpu.roll(x, shift=shift, axis=1)

    def _norm_pair(x_bf16, scale: float):
        x = x_bf16.astype(jnp.float32)
        x2 = x * x
        ms_lo = jnp.broadcast_to(
            jnp.sum(jnp.where(mask_lo, x2, 0.0), axis=1, keepdims=True), (tile_r, 128)
        ) * (1.0 / 64.0)
        ms_hi = jnp.broadcast_to(
            jnp.sum(jnp.where(mask_lo, 0.0, x2), axis=1, keepdims=True), (tile_r, 128)
        ) * (1.0 / 64.0)
        inv_rms = jnp.where(
            mask_lo, lax.rsqrt(ms_lo + norm_eps), lax.rsqrt(ms_hi + norm_eps)
        )
        y = x * inv_rms
        if scale != 1.0:
            y = y * scale
        return y.astype(jnp.bfloat16)

    def _rope_pair(x_bf16, cos_128, sin_s128):
        xv = x_bf16.astype(jnp.float32)
        partner = jnp.where(even_lane, _roll(xv, 127), _roll(xv, 1))
        return (xv * cos_128 + partner * sin_s128).astype(jnp.bfloat16)

    def _quant_fp8_u8(x_bf16, inv_sc: float):
        xf = x_bf16.astype(jnp.float32)
        if inv_sc != 1.0:
            xf = xf * inv_sc
        f8 = jnp.clip(xf, fp8_min, fp8_max).astype(jnp.float8_e4m3fn)
        if interpret_mode:
            return lax.bitcast_convert_type(f8, jnp.uint8)
        return pltpu.bitcast(f8, jnp.uint8)

    z_bf16 = jnp.zeros((tile_r, 128), jnp.bfloat16)
    for t_i in range(n_tiles):
        b0 = t_i * tile_r
        rs = pl.ds(b0, tile_r)
        if has_rope:
            cos_128 = cos_sin_vmem_ref[rs, 0, :]
            sin_s128 = cos_sin_vmem_ref[rs, 1, :]
        else:
            cos_128 = None
            sin_s128 = None

        q_heads: list = []
        for p_idx in range(8):
            qp = qkv_2d_ref[rs, pl.ds(p_idx * 128, 128)]
            qp = _norm_pair(qp, query_scale)
            if has_rope:
                qp = _rope_pair(qp, cos_128, sin_s128)
            qp_swapped = _roll(qp, 64)
            if p_idx < 4:
                h_even = jnp.where(mask_lo, qp, z_bf16)
                h_odd = jnp.where(mask_lo, qp_swapped, z_bf16)
            else:
                h_even = jnp.where(mask_lo, z_bf16, qp_swapped)
                h_odd = jnp.where(mask_lo, z_bf16, qp)
            q_heads.append(h_even)
            q_heads.append(h_odd)
        # Full 16-head T(16, 128) store at tile_r=32 (32 VREGs, 0 sublane RMWs!).
        q_out_vmem_ref[rs] = jnp.stack(q_heads, axis=1)

        kp = qkv_2d_ref[rs, pl.ds(8 * 128, 128)]
        vp = qkv_2d_ref[rs, pl.ds(9 * 128, 128)]
        kp = _norm_pair(kp, 1.0)
        if has_rope:
            kp = _rope_pair(kp, cos_128, sin_s128)
        k_u8 = _quant_fp8_u8(kp, inv_k_scale)
        v_u8 = _quant_fp8_u8(vp, inv_v_scale)
        if interpret_mode:
            z_u8 = jnp.zeros((tile_r, 128), jnp.uint8)
            packed_u8 = jnp.stack([k_u8, v_u8, z_u8, z_u8], axis=1)[:, None, :, :]
            kv_out_vmem_ref[rs] = lax.bitcast_convert_type(
                packed_u8, jnp.float8_e4m3fn
            )
        else:
            kv_u32 = k_u8.astype(jnp.uint32) | (v_u8.astype(jnp.uint32) << 8)
            kv_out_u32_ref = kv_out_vmem_ref.bitcast(jnp.uint32).reshape(b_rows, 128)
            kv_out_u32_ref[rs] = kv_u32


def _run_in_vmem_qkv_epilogue_dispatch(
    qkv_2d_ref,
    cos_sin_vmem_ref,
    q_out_vmem_ref,
    kv_out_vmem_ref,
    *,
    norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Dispatch `TP=8` (`qkv_cols=1280`) directly and `TP=16` (`qkv_cols=640`) via shared M2M epilogue."""
    if qkv_2d_ref.shape[1] == 1280:
        _run_in_vmem_qkv_epilogue(
            qkv_2d_ref,
            cos_sin_vmem_ref,
            q_out_vmem_ref,
            kv_out_vmem_ref,
            norm_eps=norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
        return
    from sglang_tpu.muse_megakernel.decode.m2m.kernel import (
        _run_in_vmem_qkv_epilogue_any,
    )

    _run_in_vmem_qkv_epilogue_any(
        qkv_2d_ref,
        cos_sin_vmem_ref,
        q_out_vmem_ref,
        kv_out_vmem_ref,
        norm_eps=norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
    )


# =============================================================================
# 2. L0 Fused Dual QKV+Gate + QKV/RoPE Epilogue & L61 Final Norm Sub-Bridge
# =============================================================================
def _fp8_dual_qkv_epi_gate_kernel(
    attn_in_ref,
    w_qkv_hbm,
    s_qkv_ref,
    w_gate_hbm,
    s_gate_ref,
    cos_sin_ref,
    out_q_ref,
    out_kv_ref,
    out_gate_ref,
    w_qkv_buf,
    w_gate_buf,
    qkv_vmem,
    w_qkv_sem,
    w_gate_sem,
    *,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    from jax.experimental.pallas import tpu as pltpu

    rows = attn_in_ref.shape[0]
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()
    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        attn_in_ref, w_qkv_buf, s_qkv_ref, 0, qkv_vmem, 0,
        rows=rows, n_cols=w_qkv_hbm.shape[1],
    )
    # Overlap VPU QKV RMSNorm + RoPE + KV FP8 quant underneath W_gate HBM DMA!
    _run_in_vmem_qkv_epilogue_dispatch(
        qkv_vmem,
        cos_sin_ref,
        out_q_ref,
        out_kv_ref,
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
    )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        attn_in_ref, w_gate_buf, s_gate_ref, 0, out_gate_ref, 0,
        rows=rows, n_cols=w_gate_hbm.shape[1],
    )


def fp8_dual_qkv_epi_gate_jax(
    attn_in,
    w_qkv_w8k,
    s_qkv,
    w_gate_w8k,
    s_gate,
    cos_sin_step=None,
    *,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    interpret: bool = False,
):
    """L0 single-read FP8 ``qkv+gate`` + in-VMEM QKV/RoPE epilogue -> ``(q_5d, kv_merged, gate)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, hidden = attn_in.shape
    qk, qkv_cols = w_qkv_w8k.shape
    gk, gate_cols = w_gate_w8k.shape
    if (
        not (rows == 8 or (16 <= rows <= 384 and rows % 16 == 0))
        or (qkv_cols, gate_cols) not in ((1280, 1024), (640, 512))
        or qk != hidden
        or gk != hidden
    ):
        raise ValueError("Unsupported fp8_dual_qkv_epi_gate shapes")
    has_rope = cos_sin_step is not None
    cs_in = cos_sin_step[:rows] if (has_rope and cos_sin_step.shape[0] != rows) else cos_sin_step

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
    q_3d, kv_merged, gate = _pallas_call_maybe_interpret(
        functools.partial(
            _fp8_dual_qkv_epi_gate_kernel,
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            has_rope=has_rope,
            interpret_mode=interpret,
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=[v_sp, h_sp, v_sp, h_sp, v_sp, v_sp if has_rope else None],
        out_specs=(v_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((hidden, qkv_cols), w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), w_gate_w8k.dtype),
            vmem((rows, qkv_cols), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=42 * 1024**2,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8_dual_qkv_epi_b{rows}_{'rope' if has_rope else 'nope'}",
    )(
        attn_in.astype(jnp.bfloat16),
        w_qkv_w8k,
        _as_dot_scale_f32(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_dot_scale_f32(s_gate, gate_cols),
        cs_in,
    )
    return q_3d.reshape(1, rows, 8, 2, 128), kv_merged, gate


def _moe_post_ag_final_norm_kernel(
    stream_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    final_gain_ref,
    out_hidden_ref,
    w_post_buf,
    s_in_buf,
    post_shard,
    half_a,
    half_b,
    mate_tok_buf,
    moe_full_buf,
    w_post_sem,
    s_in_sem,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    final_eps: float,
    final_scale_input: bool,
    sharded_moe_in: bool,
    norm_before_s3: bool,
) -> None:
    """L61 Final Sub-Bridge: fuses ``model.norm(stream.to(bf16))`` & skips dead ``out_stream_hbm``!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    use_halfrow = rows >= 32 and rows % 32 == 0
    use_nbs3 = use_halfrow and norm_before_s3
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    if use_nbs3:
        half = rows // 2
        parity = me & jnp.int32(1)
        my_r0 = pl.multiple_of(parity * half, 16)
        pltpu.make_async_copy(
            stream_hbm.at[pl.ds(my_r0, half), :], s_in_buf, s_in_sem.at[0]
        ).start()
    else:
        pltpu.make_async_copy(stream_hbm, s_in_buf, s_in_sem.at[0]).start()
    if sharded_moe_in:
        _gather_token_shard_xor1_and_dot_post(
            moe_out_ref, mate_tok_buf, moe_full_buf, w_post_buf, s_post_ref, post_shard, w_post_sem,
            tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
            me=me, world_size=world_size,
        )
    else:
        _dma_wait(w_post_buf, w_post_sem.at[0])
        _dot_fp8_scaled_to_ref(
            moe_out_ref, w_post_buf, s_post_ref, 0, post_shard, 0,
            rows=rows, n_cols=post_cols,
        )
        _wait_barrier(world_size)

    if use_nbs3:
        half = rows // 2
        bm_s1 = _pick_strip(half, 32)
        parity = me & jnp.int32(1)
        mate = me ^ jnp.int32(1)
        chip = me >> jnp.int32(1)
        n_chips = world_size // 2
        pair_cols = 2 * post_cols
        me_c0 = pl.multiple_of(me * post_cols, 128)
        chip_c0 = pl.multiple_of(chip * pair_cols, 128)
        my_r0 = pl.multiple_of(parity * half, 16)
        mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half, 16)

        s1 = pltpu.make_async_remote_copy(
            post_shard.at[pl.ds(mate_r0, half), :],
            half_a.at[:, pl.ds(me_c0, post_cols)],
            ag_s1_send.at[0],
            ag_s1_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s1.start()
        _fori_strips(
            half,
            bm_s1,
            lambda r0: half_a.__setitem__(
                (pl.ds(r0, bm_s1), pl.ds(me_c0, post_cols)),
                post_shard[pl.ds(my_r0 + r0, bm_s1), :],
            ),
        )
        s1.wait()

        s2_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                half_a.at[:, pl.ds(chip_c0, pair_cols)],
                half_a.at[:, pl.ds(chip_c0, pair_cols)],
                ag_s2_send.at[d - 1],
                ag_s2_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)
        _dma_wait(s_in_buf, s_in_sem.at[0])
        _prescale_s_buf_in_vmem(
            s_in_buf, pf_alpha_ref, rows=half, hidden=hidden, bm=_pick_strip(half, 32)
        )
        for op in s2_ops:
            op.wait()

        _run_my_half_post_ffn_norm_epilogue(
            s_in_buf,
            half_a,
            post_ff_gain_ref,
            pf_beta_ref,
            final_gain_ref,
            out_hidden_ref,
            hidden=hidden,
            my_r0=my_r0,
            n_r=half,
            align=16,
            post_ff_eps=post_ff_eps,
            post_ff_scale_input=post_ff_scale_input,
            norm2_eps=final_eps,
            norm2_scale_input=final_scale_input,
            cast_stream_bf16=True,
        )
        s3 = pltpu.make_async_remote_copy(
            out_hidden_ref.at[pl.ds(my_r0, half), :],
            out_hidden_ref.at[pl.ds(my_r0, half), :],
            ag_s3_send.at[0],
            ag_s3_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s3.start()
        s3.wait()
        return

    def _epi_final(src_buf, r_base, n_r):
        bm = _pick_strip(n_r, 32)

        def _strip(r_loc):
            rs_l = pl.ds(r_loc, bm)
            rs_g = pl.ds(pl.multiple_of(r_base + r_loc, align), bm)

            def _bvar(c_sl, acc):
                bc = src_buf[rs_l, c_sl].astype(jnp.float32)
                if post_ff_scale_input:
                    bc = bc * post_ff_gain_ref[0:1, c_sl]
                sq = bc * bc
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            b_sq = _reduce_col_blocks(
                hidden, _bvar, jnp.zeros((bm, 128), dtype=jnp.float32)
            )
            inv_b = lax.rsqrt(
                jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(post_ff_eps)
            )

            def _res_fvar(c_sl, acc):
                bc = src_buf[rs_l, c_sl].astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_sl]
                b_n = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                ns = pf_alpha_ref[0:1, c_sl] * s_in_buf[rs_g, c_sl] + pf_beta_ref[0:1, c_sl] * b_n
                ns_bf = ns.astype(jnp.bfloat16)
                s_in_buf[rs_g, c_sl] = ns_bf.astype(jnp.float32)
                nv = s_in_buf[rs_g, c_sl]
                if final_scale_input:
                    nv = nv * final_gain_ref[0:1, c_sl]
                sq = nv * nv
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            f_sq = _reduce_col_blocks(
                hidden, _res_fvar, jnp.zeros((bm, 128), dtype=jnp.float32)
            )
            inv_f = lax.rsqrt(
                jnp.sum(f_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(final_eps)
            )

            def _fout(c_sl):
                sv = s_in_buf[rs_g, c_sl]
                fg = final_gain_ref[0:1, c_sl]
                fn = (sv * fg) * inv_f if final_scale_input else (sv * inv_f) * fg
                out_hidden_ref[rs_g, c_sl] = fn.astype(out_hidden_ref.dtype)

            _fori_col_blocks(hidden, _fout)

        _fori_strips(n_r, bm, _strip)

    _gather_shard_xor1(
        post_shard, half_a, half_b,
        ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
        me=me, world_size=world_size, use_halfrow=use_halfrow,
        during_ici_fn=lambda: _dma_wait(s_in_buf, s_in_sem.at[0]),
        my_half_epilogue_fn=_epi_final,
        mate_half_epilogue_fn=_epi_final,
    )


def moe_post_ag_final_norm_jax(
    stream,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    final_gain,
    *,
    world_size: int,
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    final_eps: float = 1e-5,
    final_scale_input: bool = False,
    sharded_moe_in: bool = False,
    norm_before_s3: bool | None = None,
    collective_id: int = _W8_CID_BASE,
    interpret: bool = False,
):
    """L61 Final Sub-Bridge returning ``hidden_bf16 [B, H]`` directly (zero ``out_stream`` write)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    _, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    n_chips = world_size // 2
    use_halfrow = rows >= 32 and rows % 32 == 0
    nbs3 = _qepi_norm_before_s3_enabled() if norm_before_s3 is None else bool(norm_before_s3)
    use_nbs3 = use_halfrow and nbs3
    half_rows = rows // 2 if use_halfrow else rows
    s_rows = half_rows if use_nbs3 else rows
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _moe_post_ag_final_norm_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            final_eps=float(final_eps),
            final_scale_input=bool(final_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
            norm_before_s3=nbs3,
        ),
        interpret=interpret,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=v_sp,
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((s_rows, hidden), jnp.float32),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((8, 128) if use_nbs3 else (half_rows, hidden), jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem((rows, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=46 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8_post_ag_fnorm_b{rows}_tp{world_size}{'_sh' if sharded_moe_in else ''}",
    )(
        stream.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_2d_f32(s_post, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(final_gain, hidden),
    )


def _run_rep_b2_vmem_core(
    stream_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_w_qkv_hbm,
    next_s_qkv_ref,
    next_w_gate_hbm,
    next_s_gate_ref,
    cos_sin_ref,
    out_stream_hbm,
    out_next_q_ref,
    out_next_kv_ref,
    out_next_gate_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_vmem,
    qkv_vmem,
    post_shard,
    half_a,
    half_b,
    mate_tok_buf,
    moe_full_buf,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    sharded_moe_in: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    norm_before_s3: bool,
    interpret_mode: bool,
    skip_barrier_signal: bool = False,
    skip_barrier_wait: bool = False,
    skip_stream_load: bool = False,
    skip_stream_store: bool = False,
    prestarted_w_post_dma: bool = False,
    prestarted_w_qkv_dma: bool = False,
    prestarted_w_gate_dma: bool = False,
    gate_dst_ref=None,
    after_qkv_epi_fn=None,
    defer_stream_store_wait: bool = False,
) -> None:
    """Reusable VMEM-resident replicated-stream Bridge 2b + in-VMEM QKV/RoPE epilogue core (e.g. B=16)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    use_halfrow = rows >= 32 and rows % 32 == 0
    use_nbs3 = use_halfrow and norm_before_s3
    align = 16 if use_halfrow else 8
    half = rows // 2
    parity = lax.axis_index("model").astype(jnp.int32) & jnp.int32(1)
    my_r0 = pl.multiple_of(parity * half, 16) if use_halfrow else jnp.int32(0)
    mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half, 16) if use_halfrow else jnp.int32(0)

    me = lax.axis_index("model").astype(jnp.int32)
    if not skip_barrier_signal:
        _signal_barrier(me, world_size)
    if not prestarted_w_post_dma:
        pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    if not skip_stream_load:
        if use_nbs3:
            pltpu.make_async_copy(
                stream_hbm.at[pl.ds(my_r0, half), :], s_buf, s_in_sem.at[0]
            ).start()
        else:
            pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
    if not prestarted_w_qkv_dma:
        pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    if not prestarted_w_gate_dma:
        pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    if sharded_moe_in:
        _gather_token_shard_xor1_and_dot_post(
            moe_out_ref, mate_tok_buf, moe_full_buf, w_post_buf, s_post_ref, post_shard, w_post_sem,
            tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
            me=me, world_size=world_size, skip_barrier_wait=skip_barrier_wait,
        )
    else:
        _dma_wait(w_post_buf, w_post_sem.at[0])
        _dot_fp8_scaled_to_ref(
            moe_out_ref, w_post_buf, s_post_ref, 0, post_shard, 0,
            rows=rows, n_cols=post_cols,
        )
        if not skip_barrier_wait:
            _wait_barrier(world_size)

    if use_nbs3:
        bm_s1 = _pick_strip(half, 32)
        mate = me ^ jnp.int32(1)
        chip = me >> jnp.int32(1)
        n_chips = world_size // 2
        pair_cols = 2 * post_cols
        me_c0 = pl.multiple_of(me * post_cols, 128)
        chip_c0 = pl.multiple_of(chip * pair_cols, 128)

        s1 = pltpu.make_async_remote_copy(
            post_shard.at[pl.ds(mate_r0, half), :],
            half_a.at[:, pl.ds(me_c0, post_cols)],
            ag_s1_send.at[0],
            ag_s1_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s1.start()
        _fori_strips(
            half,
            bm_s1,
            lambda r0: half_a.__setitem__(
                (pl.ds(r0, bm_s1), pl.ds(me_c0, post_cols)),
                post_shard[pl.ds(my_r0 + r0, bm_s1), :],
            ),
        )
        s1.wait()

        s2_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                half_a.at[:, pl.ds(chip_c0, pair_cols)],
                half_a.at[:, pl.ds(chip_c0, pair_cols)],
                ag_s2_send.at[d - 1],
                ag_s2_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)
        if not skip_stream_load:
            _dma_wait(s_buf, s_in_sem.at[0])
        _prescale_s_buf_in_vmem(
            s_buf, pf_alpha_ref, rows=half, hidden=hidden, bm=_pick_strip(half, 32)
        )
        for op in s2_ops:
            op.wait()

        _run_my_half_post_ffn_norm_epilogue(
            s_buf,
            half_a,
            post_ff_gain_ref,
            pf_beta_ref,
            next_in_gain_ref,
            next_in_vmem,
            hidden=hidden,
            my_r0=my_r0,
            n_r=half,
            align=16,
            post_ff_eps=post_ff_eps,
            post_ff_scale_input=post_ff_scale_input,
            norm2_eps=next_in_eps,
            norm2_scale_input=next_in_scale_input,
            cast_stream_bf16=False,
        )

        # Stage-3a D2D (critical path: 512 KiB bf16 next_in[my_r0:+half] -> mate) + local HBM store
        s3_nin = pltpu.make_async_remote_copy(
            next_in_vmem.at[pl.ds(my_r0, half), :],
            next_in_vmem.at[pl.ds(my_r0, half), :],
            ag_s3_send.at[0],
            ag_s3_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s3_nin.start()
        if not skip_stream_store:
            pltpu.make_async_copy(
                s_buf, out_stream_hbm.at[pl.ds(my_r0, half), :], s_out_sem.at[0]
            ).start()
        _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
        s3_nin.wait()

        # Stage-3b D2D (non-critical: 1 MiB f32 s_buf -> mate's half_b f32, hidden under W_qkv MXU!)
        s3_stream = pltpu.make_async_remote_copy(
            s_buf,
            half_b,
            ag_s3_send.at[1],
            ag_s3_recv.at[1],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s3_stream.start()
        _dot_fp8_scaled_to_ref(
            next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_vmem, 0,
            rows=rows, n_cols=qkv_cols,
        )
        s3_stream.wait()
        if not skip_stream_store:
            pltpu.make_async_copy(
                half_b, out_stream_hbm.at[pl.ds(mate_r0, half), :], s_out_sem.at[1]
            ).start()
    elif (not use_halfrow) and (not sharded_moe_in) and (post_cols % 128 == 0) and ((hidden // 2) % 128 == 0):
        # Wave-18 Zero-s1 -50% ICI half-col AG + D2D-overlapped own_b_sq (B in {8, 16}):
        # eliminates serial s1 D2D before ICI, halves ICI bytes across peers, and hides
        # the single contiguous [B, H/2] D2D swap underneath own_b_sq.
        half_h = hidden // 2
        n_chips = world_size // 2
        mate = me ^ jnp.int32(1)
        side = me & jnp.int32(1)
        other_side = jnp.int32(1) - side
        chip = me >> jnp.int32(1)
        own_h0 = pl.multiple_of(side * half_h, 128)
        oth_h0 = pl.multiple_of(other_side * half_h, 128)
        my_slot_c0 = pl.multiple_of(own_h0 + chip * post_cols, 128)
        max_blk = 1024 if rows == 8 else 512
        post_spans = _tile_spans(post_cols, max_blk)
        hid_spans = _tile_spans(hidden, max_blk)

        s2_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                post_shard,
                half_a.at[:, pl.ds(my_slot_c0, post_cols)],
                ag_s2_send.at[d - 1],
                ag_s2_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)

        for off, cw in post_spans:
            half_a[:, pl.ds(pl.multiple_of(my_slot_c0 + off, 128), cw)] = post_shard[
                :, pl.ds(pl.multiple_of(off, 128), cw)
            ]
        if not skip_stream_load:
            _dma_wait(s_buf, s_in_sem.at[0])
        for off, cw in hid_spans:
            c_sl = pl.ds(pl.multiple_of(off, 128), cw)
            s_buf[:, c_sl] = pf_alpha_ref[0:1, c_sl] * s_buf[:, c_sl]
        for op in s2_ops:
            op.wait()

        op_d2d = pltpu.make_async_remote_copy(
            half_a.at[:, pl.ds(own_h0, half_h)],
            half_a.at[:, pl.ds(own_h0, half_h)],
            ag_s3_send.at[0],
            ag_s3_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_d2d.start()

        own_b_sq = jnp.zeros((rows, 128), dtype=jnp.float32)
        for c in range(n_chips):
            for off, cw in post_spans:
                c_src = pl.ds(pl.multiple_of(own_h0 + c * post_cols + off, 128), cw)
                c_can = pl.ds(pl.multiple_of((2 * c) * post_cols + side * post_cols + off, 128), cw)
                bc = half_a[:, c_src].astype(jnp.float32)
                if post_ff_scale_input:
                    bc = bc * post_ff_gain_ref[0:1, c_can]
                own_b_sq = own_b_sq + _fold_cols_to_128(bc * bc)
        op_d2d.wait()

        oth_b_sq = jnp.zeros((rows, 128), dtype=jnp.float32)
        for c in range(n_chips):
            for off, cw in post_spans:
                c_src = pl.ds(pl.multiple_of(oth_h0 + c * post_cols + off, 128), cw)
                c_can = pl.ds(
                    pl.multiple_of((2 * c) * post_cols + other_side * post_cols + off, 128), cw
                )
                bc = half_a[:, c_src].astype(jnp.float32)
                if post_ff_scale_input:
                    bc = bc * post_ff_gain_ref[0:1, c_can]
                oth_b_sq = oth_b_sq + _fold_cols_to_128(bc * bc)

        inv_b = lax.rsqrt(
            jnp.sum(own_b_sq + oth_b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )
        n_sq = jnp.zeros((rows, 128), dtype=jnp.float32)
        for r_idx in range(world_size):
            src_base = (r_idx & 1) * half_h + (r_idx >> 1) * post_cols
            dst_base = r_idx * post_cols
            for off, cw in post_spans:
                c_src = pl.ds(pl.multiple_of(src_base + off, 128), cw)
                c_can = pl.ds(pl.multiple_of(dst_base + off, 128), cw)
                bc = half_a[:, c_src].astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_can]
                bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                ns = s_buf[:, c_can] + pf_beta_ref[0:1, c_can] * bn
                s_buf[:, c_can] = ns
                nv = ns * next_in_gain_ref[0:1, c_can] if next_in_scale_input else ns
                n_sq = n_sq + _fold_cols_to_128(nv * nv)

        if not skip_stream_store:
            pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
        inv_n = lax.rsqrt(
            jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(next_in_eps)
        )
        for off, cw in hid_spans:
            c_sl = pl.ds(pl.multiple_of(off, 128), cw)
            sv = s_buf[:, c_sl]
            ng = next_in_gain_ref[0:1, c_sl]
            nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
            next_in_vmem[:, c_sl] = nn.astype(jnp.bfloat16)

        _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
        _dot_fp8_scaled_to_ref(
            next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_vmem, 0,
            rows=rows, n_cols=qkv_cols,
        )
    else:
        bm_p = _pick_strip(rows, 32)

        def _wait_and_prescale_stream():
            if not skip_stream_load:
                _dma_wait(s_buf, s_in_sem.at[0])
            _prescale_s_buf_in_vmem(s_buf, pf_alpha_ref, rows=rows, hidden=hidden, bm=bm_p)

        def _epi_and_store(buf, r_base, n_r, sem_slot: int):
            _run_post_ffn_slice_epilogue(
                s_buf, buf, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
                s_buf, next_in_vmem,
                hidden=hidden, row_base=r_base, n_r=n_r, align=align,
                post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
                next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
                with_next_norm=True, pre_scaled_alpha=True,
            )
            if use_halfrow and not skip_stream_store:
                pltpu.make_async_copy(
                    s_buf.at[pl.ds(r_base, n_r)],
                    out_stream_hbm.at[pl.ds(r_base, n_r)],
                    s_out_sem.at[sem_slot],
                ).start()

        _gather_shard_xor1(
            post_shard, half_a, half_b,
            ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
            me=me, world_size=world_size, use_halfrow=use_halfrow,
            during_ici_fn=_wait_and_prescale_stream,
            my_half_epilogue_fn=lambda b, r0, nr: _epi_and_store(b, r0, nr, 0),
            mate_half_epilogue_fn=lambda b, r0, nr: _epi_and_store(b, r0, nr, 1),
        )

        if not use_halfrow and not skip_stream_store:
            pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
        _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
        _dot_fp8_scaled_to_ref(
            next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_vmem, 0,
            rows=rows, n_cols=qkv_cols,
        )

    _run_in_vmem_qkv_epilogue_dispatch(
        qkv_vmem,
        cos_sin_ref,
        out_next_q_ref,
        out_next_kv_ref,
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
    )
    gate_dst = gate_dst_ref if gate_dst_ref is not None else out_next_gate_ref
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    if after_qkv_epi_fn is not None:
        after_qkv_epi_fn()
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, gate_dst, 0,
        rows=rows, n_cols=gate_cols,
    )
    if not skip_stream_store and not defer_stream_store_wait:
        if use_halfrow:
            _dma_wait(out_stream_hbm.at[pl.ds(my_r0, half)], s_out_sem.at[0])
            _dma_wait(out_stream_hbm.at[pl.ds(mate_r0, half)], s_out_sem.at[1])
        else:
            _dma_wait(out_stream_hbm, s_out_sem.at[0])


def _moe_to_attn_qkv_epi_bridge_kernel(
    *args,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    sharded_moe_in: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    norm_before_s3: bool,
    interpret_mode: bool,
) -> None:
    """Wave-8 Bridge 2b with fused in-VMEM QKV/RoPE epilogue -> ``(stream, q_3d, kv, gate)``."""
    _run_rep_b2_vmem_core(
        *args,
        world_size=world_size,
        post_ff_eps=post_ff_eps,
        post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps,
        next_in_scale_input=next_in_scale_input,
        sharded_moe_in=sharded_moe_in,
        qkv_norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        norm_before_s3=norm_before_s3,
        interpret_mode=interpret_mode,
    )


def moe_to_attn_qkv_epi_bridge_jax(
    stream,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    next_in_gain,
    next_w_qkv_w8k,
    next_s_qkv,
    next_w_gate_w8k,
    next_s_gate,
    cos_sin_step=None,
    *,
    world_size: int,
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    sharded_moe_in: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    norm_before_s3: bool | None = None,
    collective_id: int = _W8_CID_BASE + 1,
    interpret: bool = False,
):
    """Bridge 2b + in-VMEM QKV/RoPE epilogue -> ``(stream, next_q_5d, next_kv_merged, next_gate)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    _, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    _, qkv_cols = next_w_qkv_w8k.shape
    _, gate_cols = next_w_gate_w8k.shape
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or not (rows == 8 or (16 <= rows <= 384 and rows % 16 == 0))
        or (qkv_cols, gate_cols) != ((1280, 1024) if world_size == 8 else (640, 512))
    ):
        raise ValueError("Unsupported moe_to_attn_qkv_epi shapes")
    has_rope = cos_sin_step is not None
    cs_in = cos_sin_step[:rows] if (has_rope and cos_sin_step.shape[0] != rows) else cos_sin_step
    n_chips = world_size // 2
    use_halfrow = rows >= 32 and rows % 32 == 0
    nbs3 = _qepi_norm_before_s3_enabled() if norm_before_s3 is None else bool(norm_before_s3)
    use_nbs3 = use_halfrow and nbs3
    half_rows = rows // 2 if use_halfrow else rows
    s_rows = half_rows if use_nbs3 else rows
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
    is_interp = bool(interpret)

    new_stream, q_3d, kv_merged, next_gate = _pallas_call_maybe_interpret(
        functools.partial(
            _moe_to_attn_qkv_epi_bridge_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            has_rope=has_rope,
            norm_before_s3=nbs3,
            interpret_mode=is_interp,
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=[
            h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp,
            h_sp, v_sp, h_sp, v_sp, v_sp if has_rope else None,
        ],
        out_specs=(h_sp, v_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((s_rows, hidden), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, qkv_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.float32 if use_nbs3 else jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem((rows, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((2 if use_halfrow else 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((2 if use_nbs3 else 1,)),
            sem((2 if use_nbs3 else 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=56 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w8_moe_to_qkv_epi_b{rows}_tp{world_size}"
            f"{'_sh' if sharded_moe_in else ''}_{'rope' if has_rope else 'nope'}"
        ),
    )(
        stream.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_dot_scale_f32(s_post, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_dot_scale_f32(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_dot_scale_f32(next_s_gate, gate_cols),
        cs_in,
    )
    return new_stream, q_3d.reshape(1, rows, 8, 2, 128), kv_merged, next_gate


# =============================================================================
# 3. Pallas Vocab-Parallel Embedding Gather / Token-Shard RS + Fused RMSNorm
# =============================================================================
def _lookup_local_vocab_rows_in_kernel(
    input_ids_smem,
    embed_hbm,
    partial_emb_buf,
    slot_2d_buf,
    row_sems,
    *,
    me,
    actual_rows: int,
) -> None:
    """Owner-only HBM vocab gather inside Pallas supporting ``v_shard % 2 == 0`` & skipping empty 16-row groups.

    Uses ``tail_r0 = v_shard & ~7`` so ``tail_len = v_shard - tail_r0`` is strictly in ``{0, 2, 4, 6}``
    (never 8, which stalls TPU7x DMA, while avoiding tail-8 zeroing on ``v_shard % 8 == 0``) with
    ``pl.ds(kk * 16, ...)`` 16-row VMEM slot alignment and ``pl.multiple_of(..., 8)`` HBM alignment.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    pad_rows = partial_emb_buf.shape[0]
    v_shard, hidden = embed_hbm.shape
    v_start = me * jnp.int32(v_shard)
    v_end = v_start + jnp.int32(v_shard)
    tail_r0 = v_shard & ~7
    tail_len = v_shard - tail_r0
    max_r0 = tail_r0 - 16
    row_iota_16 = lax.broadcasted_iota(jnp.int32, (16, _COL_STRIP), 0)

    def _grp_body(g0):
        def _zero_grp():
            _fori_strips(
                hidden,
                _COL_STRIP,
                lambda c0: partial_emb_buf.__setitem__(
                    (pl.ds(g0, 16), pl.ds(c0, _COL_STRIP)),
                    jnp.zeros((16, _COL_STRIP), dtype=jnp.bfloat16),
                ),
            )

        def _run_grp():
            for k in range(16):
                valid_k = (g0 + k) < actual_rows
                safe_idx = jnp.minimum(g0 + k, actual_rows - 1)
                tid = input_ids_smem[safe_idx]
                in_r = valid_k & (tid >= v_start) & (tid < v_end)
                loc_id = jnp.where(in_r, tid - v_start, jnp.int32(0))
                if tail_len == 0:
                    b_r0 = pl.multiple_of(
                        jnp.minimum((loc_id >> jnp.int32(4)) << jnp.int32(4), max_r0), 8
                    )
                    lax.cond(
                        in_r,
                        lambda br=b_r0, kk=k: pltpu.make_async_copy(
                            embed_hbm.at[pl.ds(br, 16), :],
                            slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                            row_sems.at[kk],
                        ).start(),
                        lambda: None,
                    )
                else:

                    def _issue(lid=loc_id, kk=k):
                        in_tail = lid >= jnp.int32(tail_r0)

                        def _t():
                            pltpu.make_async_copy(
                                embed_hbm.at[pl.ds(tail_r0, tail_len), :],
                                slot_2d_buf.at[pl.ds(kk * 16, tail_len), :],
                                row_sems.at[kk],
                            ).start()

                        def _b():
                            br = pl.multiple_of(
                                jnp.minimum((lid >> jnp.int32(4)) << jnp.int32(4), max_r0), 8
                            )
                            pltpu.make_async_copy(
                                embed_hbm.at[pl.ds(br, 16), :],
                                slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                                row_sems.at[kk],
                            ).start()

                        lax.cond(in_tail, _t, _b)

                    lax.cond(in_r, _issue, lambda: None)

            for k in range(16):
                valid_k = (g0 + k) < actual_rows
                safe_idx = jnp.minimum(g0 + k, actual_rows - 1)
                tid = input_ids_smem[safe_idx]
                in_r = valid_k & (tid >= v_start) & (tid < v_end)
                if tail_len == 0:
                    lax.cond(
                        in_r,
                        lambda kk=k: pltpu.make_async_copy(
                            slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                            slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                            row_sems.at[kk],
                        ).wait(),
                        lambda: None,
                    )
                else:
                    loc_id = jnp.where(in_r, tid - v_start, jnp.int32(0))

                    def _wait_k(lid=loc_id, kk=k):
                        in_tail = lid >= jnp.int32(tail_r0)
                        lax.cond(
                            in_tail,
                            lambda: pltpu.make_async_copy(
                                slot_2d_buf.at[pl.ds(kk * 16, tail_len), :],
                                slot_2d_buf.at[pl.ds(kk * 16, tail_len), :],
                                row_sems.at[kk],
                            ).wait(),
                            lambda: pltpu.make_async_copy(
                                slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                                slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                                row_sems.at[kk],
                            ).wait(),
                        )

                    lax.cond(in_r, _wait_k, lambda: None)

            def _col_body(c0):
                c_sl = pl.ds(c0, _COL_STRIP)
                acc = jnp.zeros((16, _COL_STRIP), dtype=jnp.float32)
                for k in range(16):
                    valid_k = (g0 + k) < actual_rows
                    safe_idx = jnp.minimum(g0 + k, actual_rows - 1)
                    tid = input_ids_smem[safe_idx]
                    in_r = valid_k & (tid >= v_start) & (tid < v_end)
                    loc_id = jnp.where(in_r, tid - v_start, jnp.int32(0))
                    eff_r0 = jnp.where(
                        (loc_id >= jnp.int32(tail_r0)) if tail_len != 0 else jnp.bool_(False),
                        jnp.int32(tail_r0),
                        jnp.minimum((loc_id >> jnp.int32(4)) << jnp.int32(4), jnp.int32(max_r0)),
                    )
                    sub_r = loc_id - eff_r0
                    blk = slot_2d_buf[pl.ds(k * 16, 16), c_sl].astype(jnp.float32)
                    row_b = jnp.broadcast_to(
                        jnp.sum(
                            jnp.where(row_iota_16 == sub_r, blk, 0.0),
                            axis=0,
                            keepdims=True,
                        ),
                        (16, _COL_STRIP),
                    )
                    acc = jnp.where((row_iota_16 == k) & in_r, row_b, acc)
                partial_emb_buf[pl.ds(g0, 16), c_sl] = acc.astype(jnp.bfloat16)

            _fori_strips(hidden, _COL_STRIP, _col_body)

        any_hit = jnp.bool_(False)
        for k in range(16):
            valid_k = (g0 + k) < actual_rows
            tid = input_ids_smem[jnp.minimum(g0 + k, actual_rows - 1)]
            any_hit = any_hit | (valid_k & (tid >= v_start) & (tid < v_end))
        lax.cond(any_hit, _run_grp, _zero_grp)

    _fori_strips(pad_rows, 16, _grp_body)


def _decode_embed_ar_norm_kernel(
    input_ids_smem,
    embed_hbm,
    out_stream_hbm,
    partial_emb_buf,
    slot_2d_buf,
    own_half_buf,
    ici_half_buf,
    oth_half_buf,
    s_out_buf,
    row_sems,
    d2d_send,
    d2d_recv,
    ici_send,
    ici_recv,
    s_out_sem,
    *,
    world_size: int,
    norm_eps: float,
) -> None:
    """In-kernel owner lookup + half-col XOR=1 D2D + 1-shot half-col ICI + D2D-overlapped RMSNorm."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    actual_rows = out_stream_hbm.shape[0]
    pad_rows, hidden = partial_emb_buf.shape
    half_h = hidden // 2
    n_chips = world_size // 2
    bm = _pick_strip(pad_rows, 32)
    n_strips = hidden // _COL_STRIP
    n_half_strips = half_h // _COL_STRIP

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_h0 = pl.multiple_of(side * half_h, 128)
    oth_h0 = pl.multiple_of(other_side * half_h, 128)

    _signal_barrier(me, world_size)
    _lookup_local_vocab_rows_in_kernel(
        input_ids_smem,
        embed_hbm,
        partial_emb_buf,
        slot_2d_buf,
        row_sems,
        me=me,
        actual_rows=actual_rows,
    )
    _wait_barrier(world_size)

    # Half-col XOR=1 D2D: exchange complementary half_h slice (-50% D2D bytes vs full [B, H]).
    c1 = pltpu.make_async_remote_copy(
        partial_emb_buf.at[:, pl.ds(oth_h0, half_h)],
        own_half_buf,
        d2d_send.at[0],
        d2d_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c1.start()
    c1.wait()

    def _sum_mate(r0):
        rs = pl.ds(r0, bm)

        def _col(s, _):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_g = pl.ds(pl.multiple_of(own_h0 + s * _COL_STRIP, 128), _COL_STRIP)
            v = (
                partial_emb_buf[rs, c_g].astype(jnp.float32)
                + own_half_buf[rs, c_l].astype(jnp.float32)
            )
            own_half_buf[rs, c_l] = v.astype(own_half_buf.dtype)
            return jnp.int32(0)

        lax.fori_loop(0, n_half_strips, _col, jnp.int32(0))

    _fori_strips(pad_rows, bm, _sum_mate)

    # 1-shot half-col ICI across all same-side peers (-50% ICI bytes vs full [B, H]).
    c2 = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            own_half_buf,
            ici_half_buf.at[d - 1],
            ici_send.at[d - 1],
            ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        c2.append(op)
    for op in c2:
        op.wait()

    def _sum_own_half(r0):
        rs = pl.ds(r0, bm)

        def _col(s, _):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            tot = own_half_buf[rs, c_l].astype(jnp.float32)
            for slot in range(n_chips - 1):
                tot = tot + ici_half_buf[slot, rs, c_l].astype(jnp.float32)
            s_out_buf[rs, pl.ds(pl.multiple_of(own_h0 + s * _COL_STRIP, 128), _COL_STRIP)] = tot
            return jnp.int32(0)

        lax.fori_loop(0, n_half_strips, _col, jnp.int32(0))

    _fori_strips(pad_rows, bm, _sum_own_half)

    # Exchange reduced f32 own_h0 -> mate's oth_half_buf overlapped underneath strip-0 own_sq.
    c3 = pltpu.make_async_remote_copy(
        s_out_buf.at[:, pl.ds(own_h0, half_h)],
        oth_half_buf,
        d2d_send.at[1],
        d2d_recv.at[1],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c3.start()

    def _finish_norm(r0, *, wait_c3: bool):
        rs = pl.ds(r0, bm)

        def _sq_own(s, acc):
            c_g = pl.ds(pl.multiple_of(own_h0 + s * _COL_STRIP, 128), _COL_STRIP)
            v = s_out_buf[rs, c_g]
            sq = v * v
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        own_sq = lax.fori_loop(
            0, n_half_strips, _sq_own, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        if wait_c3:
            c3.wait()

        def _sq_oth(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_g = pl.ds(pl.multiple_of(oth_h0 + s * _COL_STRIP, 128), _COL_STRIP)
            v = oth_half_buf[rs, c_l]
            s_out_buf[rs, c_g] = v
            sq = v * v
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        oth_sq = lax.fori_loop(
            0, n_half_strips, _sq_oth, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        inv = lax.rsqrt(
            jnp.sum(own_sq + oth_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(norm_eps)
        )

        def _norm_out(s, _):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            yn = (s_out_buf[rs, c_sl] * inv).astype(jnp.bfloat16).astype(jnp.float32)
            s_out_buf[rs, c_sl] = yn
            return jnp.int32(0)

        lax.fori_loop(0, n_strips, _norm_out, jnp.int32(0))

    _finish_norm(0, wait_c3=True)
    if pad_rows > bm:
        n_row_strips = pad_rows // bm
        lax.fori_loop(
            1,
            n_row_strips,
            lambda i, _: (_finish_norm(pl.multiple_of(i * bm, bm), wait_c3=False) or jnp.int32(0)),
            jnp.int32(0),
        )

    s_src = s_out_buf if pad_rows == actual_rows else s_out_buf.at[pl.ds(0, actual_rows), :]
    pltpu.make_async_copy(s_src, out_stream_hbm, s_out_sem.at[0]).start()
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def decode_vocab_embed_norm_jax(
    input_ids,
    embed_weight_shard,
    *,
    world_size: int,
    norm_eps: float = 1e-5,
    collective_id: int = _W8_CID_BASE + 2,
    interpret: bool = False,
):
    """Pallas ``VocabParallelEmbedding`` gather + ``_rms_normalize.to(bf16).float()``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = input_ids.shape[0]
    v_shard, hidden = embed_weight_shard.shape
    if (
        not (rows == 8 or (16 <= rows <= 384 and rows % 16 == 0))
        or v_shard < 16
        or v_shard % 2 != 0
        or hidden % (2 * _COL_STRIP) != 0
    ):
        raise ValueError("Unsupported decode_vocab_embed_norm shapes")
    pad_rows = max(rows, 16)
    half_h = hidden // 2
    n_chips = world_size // 2
    smem, vmem, hbm, sem = pltpu.SMEM, pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA

    return _pallas_call_maybe_interpret(
        functools.partial(
            _decode_embed_ar_norm_kernel,
            world_size=world_size,
            norm_eps=float(norm_eps),
        ),
        interpret=interpret,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=smem), pl.BlockSpec(memory_space=hbm)],
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=[
            vmem((pad_rows, hidden), jnp.bfloat16),
            vmem((256, hidden), jnp.bfloat16),
            vmem((pad_rows, half_h), jnp.bfloat16),
            vmem((n_chips - 1, pad_rows, half_h), jnp.bfloat16),
            vmem((pad_rows, half_h), jnp.float32),
            vmem((pad_rows, hidden), jnp.float32),
            sem((16,)),
            sem((2,)),
            sem((2,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=56 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8_embed_norm_b{rows}_tp{world_size}",
    )(input_ids.astype(jnp.int32), embed_weight_shard)


def _decode_embed_tok_sharded_kernel(
    input_ids_smem,
    embed_hbm,
    l0_gain_ref,
    out_stream_shard_hbm,
    out_attn_in_ref,
    partial_emb_buf,
    slot_2d_buf,
    ici_pair_recv,
    d2d_send_tok,
    d2d_recv_tok,
    s_out_buf,
    attn_my_tok,
    attn_mate_tok,
    row_sems,
    ici_send,
    ici_recv,
    d2d1_send,
    d2d1_recv,
    d2d2_send,
    d2d2_recv,
    ag_send,
    ag_recv,
    s_out_sem,
    *,
    world_size: int,
    embed_eps: float,
    l0_eps: float,
    l0_scale_input: bool,
) -> None:
    """In-kernel owner lookup + Token-RS + 8-row ``embed_norm`` & ``L0 input_ln`` + Token-AG!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, hidden = partial_emb_buf.shape
    r_tok = rows // world_size
    r_pair = 2 * r_tok
    n_chips = world_size // 2
    n_strips = hidden // _COL_STRIP
    bm = _pick_strip(r_tok, 16)

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)

    _signal_barrier(me, world_size)
    _lookup_local_vocab_rows_in_kernel(
        input_ids_smem,
        embed_hbm,
        partial_emb_buf,
        slot_2d_buf,
        row_sems,
        me=me,
        actual_rows=rows,
    )
    _wait_barrier(world_size)

    # 1. Inter-chip Token-RS on 16-row-aligned chip pairs [r_pair, hidden].
    c_ici = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        peer_r0 = pl.multiple_of((chip ^ jnp.int32(d)) * r_pair, 16)
        op = pltpu.make_async_remote_copy(
            partial_emb_buf.at[pl.ds(peer_r0, r_pair), :],
            ici_pair_recv.at[d - 1],
            ici_send.at[d - 1],
            ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        c_ici.append(op)
    for op in c_ici:
        op.wait()

    is_even = side == jnp.int32(0)
    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        tot_pair = partial_emb_buf[pl.ds(chip_r0, r_pair), c_sl].astype(jnp.float32)
        for slot in range(n_chips - 1):
            tot_pair = tot_pair + ici_pair_recv[slot, :, c_sl].astype(jnp.float32)
        top_r = tot_pair[0:r_tok, :].astype(jnp.bfloat16)
        bot_r = tot_pair[r_tok:r_pair, :].astype(jnp.bfloat16)
        attn_my_tok[:, c_sl] = lax.select(is_even, top_r, bot_r)
        d2d_send_tok[:, c_sl] = lax.select(is_even, bot_r, top_r)

    # 2. Intra-chip D2D [r_tok, hidden] -> exact 8-rank sum for my r_tok rows!
    c_d2d = pltpu.make_async_remote_copy(
        d2d_send_tok, d2d_recv_tok, d2d1_send.at[0], d2d1_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    c_d2d.start()
    c_d2d.wait()

    def _norms_8row(r0):
        rs = pl.ds(r0, bm)

        def _emb_sq(s, acc):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            v = attn_my_tok[rs, c_sl].astype(jnp.float32) + d2d_recv_tok[rs, c_sl].astype(jnp.float32)
            s_out_buf[rs, c_sl] = v
            sq = v * v
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq0 = lax.fori_loop(0, n_strips, _emb_sq, jnp.zeros((bm, 128), dtype=jnp.float32))
        inv0 = lax.rsqrt(
            jnp.sum(sq0, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(embed_eps)
        )

        def _l0_sq(s, acc):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            st = (s_out_buf[rs, c_sl] * inv0).astype(jnp.bfloat16).astype(jnp.float32)
            s_out_buf[rs, c_sl] = st
            lv = st * l0_gain_ref[0:1, c_sl] if l0_scale_input else st
            sq = lv * lv
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq1 = lax.fori_loop(0, n_strips, _l0_sq, jnp.zeros((bm, 128), dtype=jnp.float32))
        inv1 = lax.rsqrt(
            jnp.sum(sq1, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(l0_eps)
        )

        def _l0_out(s, _):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            st = s_out_buf[rs, c_sl]
            g0 = l0_gain_ref[0:1, c_sl]
            a0 = (st * g0) * inv1 if l0_scale_input else (st * inv1) * g0
            attn_my_tok[rs, c_sl] = a0.astype(jnp.bfloat16)
            return jnp.int32(0)

        lax.fori_loop(0, n_strips, _l0_out, jnp.int32(0))

    _fori_strips(r_tok, bm, _norms_8row)
    pltpu.make_async_copy(s_out_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    # 3. Token-AG [r_tok, hidden] -> [rows, hidden] for Layer-0 attention!
    ag_d2d = pltpu.make_async_remote_copy(
        attn_my_tok, attn_mate_tok, d2d2_send.at[0], d2d2_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    ag_d2d.start()
    ag_d2d.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_v = attn_my_tok[:, c_sl]
        mate_v = attn_mate_tok[:, c_sl]
        top_v = lax.select(is_even, own_v, mate_v)
        bot_v = lax.select(is_even, mate_v, own_v)
        out_attn_in_ref[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate([top_v, bot_v], axis=0)

    ag_ici = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            out_attn_in_ref.at[pl.ds(chip_r0, r_pair), :],
            out_attn_in_ref.at[pl.ds(chip_r0, r_pair), :],
            ag_send.at[d - 1],
            ag_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ag_ici.append(op)
    for op in ag_ici:
        op.wait()
    _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def decode_vocab_embed_tok_sharded_jax(
    input_ids,
    embed_weight_shard,
    l0_in_gain,
    *,
    world_size: int,
    embed_eps: float = 1e-5,
    l0_eps: float = 1e-5,
    l0_scale_input: bool = False,
    collective_id: int = _W8_CID_BASE + 3,
    interpret: bool = False,
):
    """Token-sharded Entry: ``(input_ids, embed_w, l0_gain) -> (stream_shard_0, attn_in_0)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = input_ids.shape[0]
    v_shard, hidden = embed_weight_shard.shape
    if (
        world_size != 8
        or rows not in (64, 128, 256, 384)
        or v_shard < 16
        or v_shard % 2 != 0
        or hidden % 512 != 0
    ):
        raise ValueError("Unsupported decode_vocab_embed_tok_sharded shapes")
    r_tok = rows // world_size
    r_pair = 2 * r_tok
    n_chips = world_size // 2
    smem, vmem, hbm, sem = pltpu.SMEM, pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    s_sp, v_sp, h_sp = (
        pl.BlockSpec(memory_space=smem),
        pl.BlockSpec(memory_space=vmem),
        pl.BlockSpec(memory_space=hbm),
    )

    return _pallas_call_maybe_interpret(
        functools.partial(
            _decode_embed_tok_sharded_kernel,
            world_size=world_size,
            embed_eps=float(embed_eps),
            l0_eps=float(l0_eps),
            l0_scale_input=bool(l0_scale_input),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((r_tok, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        ),
        in_specs=[s_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp),
        scratch_shapes=[
            vmem((rows, hidden), jnp.bfloat16),
            vmem((256, hidden), jnp.bfloat16),
            vmem((n_chips - 1, r_pair, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            sem((16,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=36 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8_embed_tok_sh_b{rows}_tp{world_size}",
    )(input_ids.astype(jnp.int32), embed_weight_shard, _as_2d_f32(l0_in_gain, hidden))


# =============================================================================
# 4. Token-Sharded Bridge 1b (`stream [B//TP, 8192]` + `TP_chip=2` `W_pre`)
# =============================================================================
def _strip_unroll(n_strips: int, max_u: int = 4) -> int:
    for u in (max_u, 2):
        if u <= max_u and n_strips >= u and n_strips % u == 0:
            return u
    return 1


def _dot_fp8_w8_slice_to_ref(
    x_ref,
    x_r0,
    w_buf,
    scale_ref,
    scale_c0,
    dst_ref,
    dst_r0,
    dst_c0,
    *,
    n_r: int,
    n_cols: int,
    align: int = 16,
) -> None:
    """Zero-spill adaptive-N FP8 MXU dot `(x_ref[x_r0:+n_r] @ w_buf) * scale -> dst_ref`.

    Uses `tn = 1024` for `bm = 16`, `tn = 512` for `bm = 32`, and `tn = 256` for `bm = 64`
    so every accumulator stays `<= 16` `f32` VREGs (`0 / 0` LLO spills) while cutting
    `x_ref` VMEM re-reads up to `4x` on `16`- and `32`-row token-sharded sub-batches!
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    bm = _pick_strip(n_r, 64)
    max_tn = 1024 if bm <= 16 else (512 if bm <= 32 else _DOT_TN)
    static_x0 = isinstance(x_r0, int) and x_r0 == 0
    static_d0 = isinstance(dst_r0, int) and dst_r0 == 0

    tiles: list[tuple[int, int]] = []
    if n_cols <= 2048:
        cn = 0
        while cn < n_cols:
            rem = n_cols - cn
            tn = max_tn
            while tn > _DOT_TN and rem < tn:
                tn //= 2
            tiles.append((cn, tn))
            cn += tn

    def _m_body(r_loc):
        rs_x = pl.ds(r_loc, bm) if static_x0 else pl.ds(pl.multiple_of(x_r0 + r_loc, align), bm)
        rs_d = pl.ds(r_loc, bm) if static_d0 else pl.ds(pl.multiple_of(dst_r0 + r_loc, align), bm)

        def _dot_tile(cn: int, tn: int):
            acc = jnp.dot(
                x_ref[rs_x, :],
                w_buf[:, pl.ds(cn, tn)],
                preferred_element_type=jnp.float32,
            )
            sc = (
                scale_ref[pl.ds(scale_c0 + cn, tn)].reshape(1, tn)
                if scale_ref.ndim == 1
                else scale_ref[0:1, pl.ds(scale_c0 + cn, tn)]
            )
            dst_ref[rs_d, pl.ds(dst_c0 + cn, tn)] = (acc * sc).astype(dst_ref.dtype)

        if tiles:
            for cn, tn in tiles:
                _dot_tile(cn, tn)
        elif n_cols % (4 * _DOT_TN) == 0:
            def _n_quad(c4):
                for sub in range(0, 4 * _DOT_TN, _DOT_TN):
                    _dot_tile(c4 + sub, _DOT_TN)

            _fori_strips(n_cols, 4 * _DOT_TN, _n_quad)
        else:
            _fori_strips(n_cols, _DOT_TN, lambda cn: _dot_tile(cn, _DOT_TN))

    _fori_strips(n_r, bm, _m_body)


def _tok_b1_fast_enabled() -> bool:
    import os

    return os.environ.get("SGLANG_TPU_MUSE_DECODE_TOK_B1_FAST", "0").strip() in (
        "1",
        "true",
        "True",
        "on",
        "ON",
    )


def _fold_cols_to_128(v):
    w = v.shape[1]
    while w > 128:
        h = w // 2
        v = v[:, 0:h] + v[:, h:w]
        w = h
    return v


def _attn_to_moe_tok_sharded_bridge_kernel(
    stream_shard_hbm,
    gated_ref,
    wo_hbm,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_pair_hbm,
    w_pre_pair_scale_ref,
    w_router_hbm,
    pre_exp_gain_ref,
    out_stream_shard_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
    wo_buf,
    w_pre_pair_buf,
    w_router_buf,
    s_in_buf,
    s_out_buf,
    send_half,
    mate_recv,
    own_half,
    rs_a_recv,
    rs_b_recv,
    d2d_my_tok,
    d2d_send_tok,
    d2d_recv_tok,
    sq_tok_buf,
    ffn_my_tok,
    ffn_mate_tok,
    ffn_pair_buf,
    router_my_tok,
    router_mate_tok,
    router_pair_buf,
    pre_my_half,
    pre_mate_half,
    sq_pair_buf,
    wo_sem,
    w_pre_sem,
    w_router_sem,
    s_in_sem,
    s_out_sem,
    c1a_send,
    c1a_recv,
    rs1_send,
    rs1_recv,
    rs2_send,
    rs2_recv,
    c1c_send,
    c1c_recv,
    pair_send,
    pair_recv,
    rpair_send,
    rpair_recv,
    pre_send,
    pre_recv,
    ag_exp_send,
    ag_exp_recv,
    ag_log_send,
    ag_log_recv,
    *,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    use_fast_vpu: bool = True,
) -> None:
    """Wave-8 Token-Sharded Bridge 1b: back-to-back HBM DMAs, 4x-unrolled VPU & early router ICI!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    r_tok = stream_shard_hbm.shape[0]
    r_pair = 2 * r_tok
    two_pair = 2 * r_pair
    hidden = wo_hbm.shape[1]
    half = hidden // 2  # 4096
    q_cols = half // 2  # 2048
    pre_pair_cols = w_pre_pair_hbm.shape[1]  # 2048
    h_pre = 2 * pre_pair_cols  # 4096
    n_exp = w_router_hbm.shape[1]  # 256
    n_chips = world_size // 2  # 4

    bm_full = _pick_strip(rows, 32)
    bm_tok = _pick_strip(r_tok, 16)
    bm_pair = _pick_strip(r_pair, 16)
    n_strips = hidden // _COL_STRIP
    half_strips = half // _COL_STRIP
    pre_half_strips = pre_pair_cols // _COL_STRIP
    u_full = _strip_unroll(n_strips, 4)
    u_half = _strip_unroll(half_strips, 4)
    u_pre = _strip_unroll(pre_half_strips, 4)
    q_step = 512 if (q_cols % 512 == 0) else _COL_STRIP
    is_r16 = w_router_buf.dtype == jnp.bfloat16

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    cx = chip & jnp.int32(1)
    cy = chip >> jnp.int32(1)
    peer_x = me ^ jnp.int32(2)
    peer_y = me ^ jnp.int32(4)

    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    own_pre_c0 = pl.multiple_of(side * pre_pair_cols, 128)
    other_pre_c0 = pl.multiple_of(other_side * pre_pair_cols, 128)
    is_even = side == jnp.int32(0)

    _signal_barrier(me, world_size)

    # Queue ALL 5 HBM DMAs upfront back-to-back so HBM runs 100% bubble-free across Wo + RS + norms!
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
    ).start()
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
    ).start()
    pltpu.make_async_copy(stream_shard_hbm, s_in_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_pre_pair_hbm, w_pre_pair_buf, w_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[0], wo_scale_ref, other_c0, send_half, 0,
        rows=rows, n_cols=half,
    )

    # Stage 1 (c1a D2D): overlapped 100% with own-half Wo[1] MXU!
    _wait_barrier(world_size)
    op_c1a = pltpu.make_async_remote_copy(
        send_half, mate_recv, c1a_send.at[0], c1a_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1a.start()

    _dma_wait(wo_buf.at[1], wo_sem.at[1])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[1], wo_scale_ref, own_c0, own_half, 0,
        rows=rows, n_cols=half,
    )
    op_c1a.wait()

    if use_fast_vpu and rows <= 128:
        for r0 in range(0, rows, 32):
            rs = pl.ds(r0, 32)
            for c0 in range(0, half, 512):
                c_sl = pl.ds(c0, 512)
                v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                own_half[rs, c_sl] = v.astype(own_half.dtype)
    else:

        def _sum_c1a(r0):
            rs = pl.ds(r0, bm_full)

            def _col_grp(g, _):
                base_s = g * u_half
                for sub in range(u_half):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                    own_half[rs, c_sl] = v.astype(own_half.dtype)
                return jnp.int32(0)

            lax.fori_loop(0, half_strips // u_half, _col_grp, jnp.int32(0))

        _fori_strips(rows, bm_full, _sum_c1a)

    # Stage 2 (rs1 + rs2): Dual-Tree 2D-XY Token Reduce-Scatter [rows, 4096] -> [r_pair, 4096]!
    a_send_r1 = pl.multiple_of((jnp.int32(1) - cy) * two_pair, 32)
    a_keep_r1 = pl.multiple_of(cy * two_pair, 32)
    b_send_r1_0 = pl.multiple_of((jnp.int32(1) - cx) * r_pair, 16)
    b_send_r1_1 = pl.multiple_of(two_pair + (jnp.int32(1) - cx) * r_pair, 16)
    b_keep_r1_0 = pl.multiple_of(cx * r_pair, 16)
    b_keep_r1_1 = pl.multiple_of(two_pair + cx * r_pair, 16)

    op_rs1_a = pltpu.make_async_remote_copy(
        own_half.at[pl.ds(a_send_r1, two_pair), pl.ds(0, q_cols)],
        rs_a_recv,
        rs1_send.at[0],
        rs1_recv.at[0],
        device_id=(peer_y,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_rs1_b0 = pltpu.make_async_remote_copy(
        own_half.at[pl.ds(b_send_r1_0, r_pair), pl.ds(q_cols, q_cols)],
        rs_b_recv.at[pl.ds(0, r_pair), :],
        rs1_send.at[1],
        rs1_recv.at[1],
        device_id=(peer_x,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_rs1_b1 = pltpu.make_async_remote_copy(
        own_half.at[pl.ds(b_send_r1_1, r_pair), pl.ds(q_cols, q_cols)],
        rs_b_recv.at[pl.ds(r_pair, r_pair), :],
        rs1_send.at[2],
        rs1_recv.at[2],
        device_id=(peer_x,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_rs1_a.start()
    op_rs1_b0.start()
    op_rs1_b1.start()
    op_rs1_a.wait()
    op_rs1_b0.wait()
    op_rs1_b1.wait()

    for sub in range(2):
        r_loc = sub * r_pair
        ra = pl.ds(pl.multiple_of(a_keep_r1 + r_loc, 16), r_pair)
        rb = pl.ds(b_keep_r1_0 if sub == 0 else b_keep_r1_1, r_pair)
        rl = pl.ds(r_loc, r_pair)
        for off in range(0, q_cols, q_step):
            ca = pl.ds(off, q_step)
            cb = pl.ds(q_cols + off, q_step)
            va = own_half[ra, ca].astype(jnp.float32) + rs_a_recv[rl, ca].astype(jnp.float32)
            vb = own_half[rb, cb].astype(jnp.float32) + rs_b_recv[rl, ca].astype(jnp.float32)
            own_half[ra, ca] = va.astype(own_half.dtype)
            own_half[rb, cb] = vb.astype(own_half.dtype)

    a_send_r2 = pl.multiple_of(a_keep_r1 + (jnp.int32(1) - cx) * r_pair, 16)
    b_send_r2 = pl.multiple_of((jnp.int32(1) - cy) * two_pair + cx * r_pair, 16)

    op_rs2_a = pltpu.make_async_remote_copy(
        own_half.at[pl.ds(a_send_r2, r_pair), pl.ds(0, q_cols)],
        rs_a_recv.at[pl.ds(0, r_pair), :],
        rs2_send.at[0],
        rs2_recv.at[0],
        device_id=(peer_x,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_rs2_b = pltpu.make_async_remote_copy(
        own_half.at[pl.ds(b_send_r2, r_pair), pl.ds(q_cols, q_cols)],
        rs_b_recv.at[pl.ds(0, r_pair), :],
        rs2_send.at[1],
        rs2_recv.at[1],
        device_id=(peer_y,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_rs2_a.start()
    op_rs2_b.start()
    op_rs2_a.wait()
    op_rs2_b.wait()

    rc_chip = pl.ds(chip_r0, r_pair)
    rl0 = pl.ds(0, r_pair)
    for off in range(0, q_cols, q_step):
        ca = pl.ds(off, q_step)
        cb = pl.ds(q_cols + off, q_step)
        va = (
            own_half[rc_chip, ca].astype(jnp.float32) + rs_a_recv[rl0, ca].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        vb = (
            own_half[rc_chip, cb].astype(jnp.float32) + rs_b_recv[rl0, ca].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        # Keep local d2d_my_tok in f32 (T(8, 128) aligned at B=64 -> zero bf16 unpack!).
        d2d_my_tok[:, ca] = lax.select(is_even, va[0:r_tok, :], va[r_tok:r_pair, :]).astype(jnp.float32)
        d2d_send_tok[:, ca] = lax.select(is_even, va[r_tok:r_pair, :], va[0:r_tok, :])
        d2d_my_tok[:, cb] = lax.select(is_even, vb[0:r_tok, :], vb[r_tok:r_pair, :]).astype(jnp.float32)
        d2d_send_tok[:, cb] = lax.select(is_even, vb[r_tok:r_pair, :], vb[0:r_tok, :])

    # Stage 3 (c1c D2D): [r_pair, 4096] -> [r_tok, 8192] overlapped with own-half pa_sq!
    op_c1c = pltpu.make_async_remote_copy(
        d2d_send_tok, d2d_recv_tok, c1c_send.at[0], c1c_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1c.start()

    if use_fast_vpu and rows <= 128:
        c_blk = min(half, 1024)
        h_blk = min(hidden, 1024)
        own_sq = jnp.zeros((r_tok, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = d2d_my_tok[:, pl.ds(c0, c_blk)]
            own_sq = own_sq + _fold_cols_to_128(bf * bf)
        op_c1c.wait()
        _dma_wait(s_in_buf, s_in_sem.at[0])
        oth_sq = jnp.zeros((r_tok, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = d2d_recv_tok[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
        pa_sq = own_sq + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )
        pff_sq = jnp.zeros((r_tok, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            c_l = pl.ds(c0, c_blk)
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            ns_own = (
                pa_alpha_ref[0:1, c_own] * s_in_buf[:, c_own]
                + pa_beta_ref[0:1, c_own] * (d2d_my_tok[:, c_l] * inv_pa)
            )
            ns_oth = (
                pa_alpha_ref[0:1, c_oth] * s_in_buf[:, c_oth]
                + pa_beta_ref[0:1, c_oth] * (d2d_recv_tok[:, c_l].astype(jnp.float32) * inv_pa)
            )
            s_out_buf[:, c_own] = ns_own
            s_out_buf[:, c_oth] = ns_oth
            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            pff_sq = pff_sq + _fold_cols_to_128(v_own * v_own + v_oth * v_oth)
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        for c0 in range(0, hidden, h_blk):
            c_g = pl.ds(c0, h_blk)
            sv = s_out_buf[:, c_g]
            gn = pre_ff_gain_ref[0:1, c_g]
            rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
            if not is_r16:
                router_my_tok[:, c_g] = rn
            ffn_my_tok[:, c_g] = rn.astype(jnp.bfloat16)
    else:

        def _own_pa_sq_tok(r0):
            rs = pl.ds(r0, bm_tok)

            def _col_grp(g, acc):
                base_s = g * u_half
                for sub in range(u_half):
                    c_l = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    bf = d2d_my_tok[rs, c_l]
                    sq = bf * bf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            sq_tok_buf[rs, :] = lax.fori_loop(
                0, half_strips // u_half, _col_grp, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
            )

        _fori_strips(r_tok, bm_tok, _own_pa_sq_tok)
        op_c1c.wait()
        _dma_wait(s_in_buf, s_in_sem.at[0])

        # 8-row VPU post_attn_ln + gated_res + pre_ffn_ln (4x unrolled loops, 8x fewer rows than W7!)
        def _finish_pa_and_pre_ffn_tok(r0):
            rs = pl.ds(r0, bm_tok)

            def _oth_sq(g, acc):
                base_s = g * u_half
                for sub in range(u_half):
                    c_l = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    bf = d2d_recv_tok[rs, c_l].astype(jnp.float32)
                    sq = bf * bf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            oth_sq = lax.fori_loop(
                0, half_strips // u_half, _oth_sq, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
            )
            pa_sq = sq_tok_buf[rs, :] + oth_sq
            inv_pa = lax.rsqrt(
                jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(pa_eps)
            )

            def _res_both(g, acc):
                base_s = g * u_half
                for sub in range(u_half):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_own = pl.ds(pl.multiple_of(own_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_oth = pl.ds(pl.multiple_of(other_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    ns_own = (
                        pa_alpha_ref[0:1, c_own] * s_in_buf[rs, c_own]
                        + pa_beta_ref[0:1, c_own] * (d2d_my_tok[rs, c_l] * inv_pa)
                    )
                    ns_oth = (
                        pa_alpha_ref[0:1, c_oth] * s_in_buf[rs, c_oth]
                        + pa_beta_ref[0:1, c_oth] * (d2d_recv_tok[rs, c_l].astype(jnp.float32) * inv_pa)
                    )
                    s_out_buf[rs, c_own] = ns_own
                    s_out_buf[rs, c_oth] = ns_oth
                    v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
                    v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
                    sq = v_own * v_own + v_oth * v_oth
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            pff_sq = lax.fori_loop(
                0, half_strips // u_half, _res_both, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
            )
            inv_pff = lax.rsqrt(
                jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(pre_ff_eps)
            )

            def _pre_ffn_out(g, _):
                base_s = g * u_full
                for sub in range(u_full):
                    c_g = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    sv = s_out_buf[rs, c_g]
                    gn = pre_ff_gain_ref[0:1, c_g]
                    rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
                    if not is_r16:
                        router_my_tok[rs, c_g] = rn
                    ffn_my_tok[rs, c_g] = rn.astype(jnp.bfloat16)
                return jnp.int32(0)

            lax.fori_loop(0, n_strips // u_full, _pre_ffn_out, jnp.int32(0))

        _fori_strips(r_tok, bm_tok, _finish_pa_and_pre_ffn_tok)
    pltpu.make_async_copy(s_out_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    # Stage 4 (pair_d2d): intra-chip [r_tok -> r_pair, 8192] for TP_chip=2 W_pre_pair!
    op_pair = pltpu.make_async_remote_copy(
        ffn_my_tok, ffn_mate_tok, pair_send.at[0], pair_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_pair.start()
    if not is_r16:
        op_rpair = pltpu.make_async_remote_copy(
            router_my_tok, router_mate_tok, rpair_send.at[0], rpair_recv.at[0],
            device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
        )
        op_rpair.start()
        op_rpair.wait()
    op_pair.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_f = ffn_my_tok[:, c_sl]
        mate_f = ffn_mate_tok[:, c_sl]
        ffn_pair_buf[:, c_sl] = jnp.concatenate(
            [lax.select(is_even, own_f, mate_f), lax.select(is_even, mate_f, own_f)], axis=0
        )
        if not is_r16:
            own_r = router_my_tok[:, c_sl]
            mate_r = router_mate_tok[:, c_sl]
            router_pair_buf[:, c_sl] = jnp.concatenate(
                [lax.select(is_even, own_r, mate_r), lax.select(is_even, mate_r, own_r)], axis=0
            )

    # Adaptive-N W_pre_pair MXU [r_pair, 8192] @ [8192, 2048] (tn=1024 at B=64 -> 2 dots, 0 scf.for!).
    _dma_wait(w_pre_pair_buf, w_pre_sem.at[0])
    _dot_fp8_w8_slice_to_ref(
        ffn_pair_buf, 0, w_pre_pair_buf, w_pre_pair_scale_ref, 0, pre_my_half, 0, 0,
        n_r=r_pair, n_cols=pre_pair_cols, align=16,
    )

    # Stage 5 (pre_d2d): [r_pair, 2048 -> 4096] OVERLAPPED with W_router + early ag_log ICI + own pre_sq!
    op_pre = pltpu.make_async_remote_copy(
        pre_my_half, pre_mate_half, pre_send.at[0], pre_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_pre.start()

    _dma_wait(w_router_buf, w_router_sem.at[0])
    rmul = jnp.float32(router_multiplier)
    r_lhs = ffn_pair_buf if is_r16 else router_pair_buf
    if r_pair <= 64 and n_exp == 256:
        lg = jnp.dot(r_lhs[...], w_router_buf[...], preferred_element_type=jnp.float32)
        out_router_logits_ref[pl.ds(chip_r0, r_pair), :] = lg * rmul
    else:
        for ce in range(0, n_exp, _DOT_TN):
            c_sl = pl.ds(ce, _DOT_TN)
            lg = jnp.dot(r_lhs[...], w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
            out_router_logits_ref[pl.ds(chip_r0, r_pair), c_sl] = lg * rmul

    # Start inter-chip router-logits AG immediately so it finishes 100% hidden under pre_exp_norm!
    ag_log_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op_l = pltpu.make_async_remote_copy(
            out_router_logits_ref.at[pl.ds(chip_r0, r_pair), :],
            out_router_logits_ref.at[pl.ds(chip_r0, r_pair), :],
            ag_log_send.at[d - 1],
            ag_log_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_l.start()
        ag_log_ops.append(op_l)

    if use_fast_vpu and rows <= 128:
        p_blk = min(pre_pair_cols, 1024 if r_pair <= 16 else 512)
        own_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            xf = pre_my_half[:, c_l].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_own]
            own_pre_sq = own_pre_sq + _fold_cols_to_128(xf * xf)
        op_pre.wait()
        oth_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_oth = pl.ds(pl.multiple_of(other_pre_c0 + c0, 128), p_blk)
            xf = pre_mate_half[:, c_l].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_oth]
            oth_pre_sq = oth_pre_sq + _fold_cols_to_128(xf * xf)
        pre_sq = own_pre_sq + oth_pre_sq
        inv = lax.rsqrt(
            jnp.sum(pre_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
            + jnp.float32(pre_exp_eps)
        )
        rs_g = pl.ds(chip_r0, r_pair)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            c_oth = pl.ds(pl.multiple_of(other_pre_c0 + c0, 128), p_blk)
            x_own = pre_my_half[:, c_l].astype(jnp.float32)
            x_oth = pre_mate_half[:, c_l].astype(jnp.float32)
            g_own = pre_exp_gain_ref[0:1, c_own]
            g_oth = pre_exp_gain_ref[0:1, c_oth]
            y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
            y_oth = (x_oth * g_oth) * inv if pre_exp_scale_input else (x_oth * inv) * g_oth
            out_expert_in_ref[rs_g, c_own] = y_own.astype(out_expert_in_ref.dtype)
            out_expert_in_ref[rs_g, c_oth] = y_oth.astype(out_expert_in_ref.dtype)
    else:

        def _own_pre_sq(r0):
            rs = pl.ds(r0, bm_pair)

            def _col_grp(g, acc):
                base_s = g * u_pre
                for sub in range(u_pre):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    xf = pre_my_half[rs, c_l].astype(jnp.float32)
                    if pre_exp_scale_input:
                        xf = xf * pre_exp_gain_ref[0:1, c_own]
                    sq = xf * xf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            sq_pair_buf[rs, :] = lax.fori_loop(
                0, pre_half_strips // u_pre, _col_grp, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )

        _fori_strips(r_pair, bm_pair, _own_pre_sq)
        op_pre.wait()

        def _finish_pre_exp_norm(r0):
            rs_l = pl.ds(r0, bm_pair)
            rs_g = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm_pair)

            def _oth_sq(g, acc):
                base_s = g * u_pre
                for sub in range(u_pre):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_oth = pl.ds(pl.multiple_of(other_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    xf = pre_mate_half[rs_l, c_l].astype(jnp.float32)
                    if pre_exp_scale_input:
                        xf = xf * pre_exp_gain_ref[0:1, c_oth]
                    sq = xf * xf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            oth_sq = lax.fori_loop(
                0, pre_half_strips // u_pre, _oth_sq, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )
            inv = lax.rsqrt(
                jnp.sum(sq_pair_buf[rs_l, :] + oth_sq, axis=-1, keepdims=True)
                * jnp.float32(1.0 / h_pre)
                + jnp.float32(pre_exp_eps)
            )

            def _out_both(g, _):
                base_s = g * u_pre
                for sub in range(u_pre):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_oth = pl.ds(pl.multiple_of(other_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    x_own = pre_my_half[rs_l, c_l].astype(jnp.float32)
                    x_oth = pre_mate_half[rs_l, c_l].astype(jnp.float32)
                    g_own = pre_exp_gain_ref[0:1, c_own]
                    g_oth = pre_exp_gain_ref[0:1, c_oth]
                    y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
                    y_oth = (x_oth * g_oth) * inv if pre_exp_scale_input else (x_oth * inv) * g_oth
                    out_expert_in_ref[rs_g, c_own] = y_own.astype(out_expert_in_ref.dtype)
                    out_expert_in_ref[rs_g, c_oth] = y_oth.astype(out_expert_in_ref.dtype)
                return jnp.int32(0)

            lax.fori_loop(0, pre_half_strips // u_pre, _out_both, jnp.int32(0))

        _fori_strips(r_pair, bm_pair, _finish_pre_exp_norm)

    # Stage 6 (tok_ag): 1-stage inter-chip Token-AG [r_pair -> rows] of expert_in (ag_log already in flight!).
    ag_exp_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op_e = pltpu.make_async_remote_copy(
            out_expert_in_ref.at[pl.ds(chip_r0, r_pair), :],
            out_expert_in_ref.at[pl.ds(chip_r0, r_pair), :],
            ag_exp_send.at[d - 1],
            ag_exp_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_e.start()
        ag_exp_ops.append(op_e)
    for op in ag_log_ops:
        op.wait()
    for op in ag_exp_ops:
        op.wait()
    _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def attn_to_moe_tok_sharded_bridge_jax(
    stream_shard,
    gated_attn,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_pair_w8k,
    w_pre_pair_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int = 8,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    use_fast_vpu: bool | None = None,
    collective_id: int = _W8_CID_BASE + 4,
    interpret: bool = False,
):
    """Token-sharded Bridge 1b (`stream_shard [B//8, 8192] -> (new_stream_shard, logits, expert_in)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        raise ValueError(f"Token-sharded Bridge 1b requires world_size=8, got {world_size}")
    rows, o_in = gated_attn.shape
    r_tok, hidden = stream_shard.shape
    r_pair = 2 * r_tok
    two_pair = 2 * r_pair
    _, pre_pair_cols = w_pre_pair_w8k.shape
    h_pre = 2 * pre_pair_cols
    _, n_exp = w_router_km.shape
    if rows not in (64, 128, 256, 384) or rows != world_size * r_tok or r_tok % 8 != 0:
        raise ValueError(f"Unsupported token-sharded Bridge 1b shapes stream={stream_shard.shape}, gated={gated_attn.shape}")
    half = hidden // 2
    q_cols = half // 2
    n_chips = world_size // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    fast_vpu = _tok_b1_fast_enabled() if use_fast_vpu is None else bool(use_fast_vpu)
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _attn_to_moe_tok_sharded_bridge_kernel,
            world_size=world_size,
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            use_fast_vpu=fast_vpu,
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((r_tok, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
            jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((hidden, pre_pair_cols), w_pre_pair_w8k.dtype),
            vmem((hidden, n_exp), w_router_km.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.float32),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((two_pair, q_cols), jnp.bfloat16),
            vmem((two_pair, q_cols), jnp.bfloat16),
            vmem((r_tok, half), jnp.float32),
            vmem((r_tok, half), jnp.bfloat16),
            vmem((r_tok, half), jnp.bfloat16),
            vmem((r_tok, 128), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((8, 128) if is_r16 else (r_tok, hidden), jnp.float32),
            vmem((8, 128) if is_r16 else (r_tok, hidden), jnp.float32),
            vmem((8, 128) if is_r16 else (r_pair, hidden), jnp.float32),
            vmem((r_pair, pre_pair_cols), jnp.bfloat16),
            vmem((r_pair, pre_pair_cols), jnp.bfloat16),
            vmem((r_pair, 128), jnp.float32),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((3,)),
            sem((3,)),
            sem((2,)),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=58 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8_tok_b1_b{rows}_tp{world_size}_{'r16' if is_r16 else 'r32'}{'' if fast_vpu else '_legacy'}",
    )(
        stream_shard.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_2d_f32(wo_scale, hidden),
        _as_2d_f32(pa_alpha, hidden),
        _as_2d_f32(pa_beta, hidden),
        _as_2d_f32(pre_ff_gain, hidden),
        w_pre_pair_w8k,
        _as_2d_f32(w_pre_pair_scale, pre_pair_cols),
        w_router_km,
        _as_2d_f32(pre_exp_gain, h_pre),
    )


# =============================================================================
# 5. Token-Sharded Bridge 2b & L61 Final Norm (`stream [B//TP, 8192]` + `TP_chip=2` `W_post`)
# =============================================================================
def _tok_b2_fast_enabled() -> bool:
    import os

    return os.environ.get("SGLANG_TPU_MUSE_DECODE_TOK_B2_FAST", "1").strip() not in (
        "0",
        "false",
        "False",
        "off",
        "OFF",
    )


def _run_tok_sharded_post_tp8_fast_core(
    moe_out_ref,
    w_post_buf,
    s_post_ref,
    s_buf,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_in_my_tok,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_pair_full,
    w_post_sem,
    s_in_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    *,
    me,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    round_stream_bf16_for_norm: bool,
    on_stream_updated_fn=None,
) -> None:
    """Wave-10 4-MiB `TP=8` `W_post` token-sharded core (1-pass `W_post` + 2D `post_pair_full` + 0-`scf.for` VPU).

    Eliminates 3x16-row `W_post` MXU re-staging (`tpu.matmul` `25 -> 13`), 3D `post_chip_blocks`
    dynamic gathers, and all 4 runtime `scf.for` VPU loops + 64 dynamic RMW dummy loads at `B<=128`.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok, k_in = moe_out_ref.shape
    rows = world_size * r_tok
    r_pair = 2 * r_tok
    post_cols = w_post_buf.shape[1]
    pair_cols = 2 * post_cols
    hidden = world_size * post_cols
    n_chips = world_size // 2

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    chip_c0 = pl.multiple_of(chip * pair_cols, 128)
    is_even = side == jnp.int32(0)

    _wait_barrier(world_size)
    ts1 = pltpu.make_async_remote_copy(
        moe_out_ref, mate_tok_buf, tok_s1_send.at[0], tok_s1_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    ts1.start()
    ts1.wait()

    k_w = min(k_in, 1024 if r_tok <= 16 else 512)
    for c0 in range(0, k_in, k_w):
        c_sl = pl.ds(c0, k_w)
        own_v = moe_out_ref[:, c_sl]
        mate_v = mate_tok_buf[:, c_sl]
        moe_full_buf[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    ts2_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            moe_full_buf.at[pl.ds(chip_r0, r_pair), :],
            moe_full_buf.at[pl.ds(chip_r0, r_pair), :],
            tok_s2_send.at[d - 1],
            tok_s2_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ts2_ops.append(op)

    _dma_wait(s_buf, s_in_sem.at[0])
    _dma_wait(w_post_buf, w_post_sem.at[0])
    if rows <= 128:
        for op in ts2_ops:
            op.wait()
        _dot_fp8_scaled_to_ref(
            moe_full_buf, w_post_buf, s_post_ref, 0, post_shard, 0,
            rows=rows, n_cols=post_cols,
        )
    else:
        _dot_fp8_row_block_to_ref(
            moe_full_buf, chip_r0, w_post_buf, s_post_ref, post_shard,
            n_r=r_pair, n_cols=post_cols, align=16,
        )
        for op in ts2_ops:
            op.wait()
        for d in range(1, n_chips):
            p_r0 = pl.multiple_of((chip ^ jnp.int32(d)) * r_pair, 16)
            _dot_fp8_row_block_to_ref(
                moe_full_buf, p_r0, w_post_buf, s_post_ref, post_shard,
                n_r=r_pair, n_cols=post_cols, align=16,
            )

    op_pd2d = pltpu.make_async_remote_copy(
        post_shard, mate_post_shard, post_d2d_send.at[0], post_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_pd2d.start()
    op_pd2d.wait()

    cs_w = min(post_cols, 512)
    for c0 in range(0, post_cols, cs_w):
        c_sl = pl.ds(c0, cs_w)
        s_v = post_shard[:, c_sl]
        m_v = mate_post_shard[:, c_sl]
        chip_cols_buf[:, pl.ds(c0, cs_w)] = lax.select(is_even, s_v, m_v)
        chip_cols_buf[:, pl.ds(post_cols + c0, cs_w)] = lax.select(is_even, m_v, s_v)

    pair_w = min(pair_cols, 1024 if r_pair <= 16 else 512)
    for c0 in range(0, pair_cols, pair_w):
        c_sl = pl.ds(c0, pair_w)
        post_pair_full[:, pl.ds(pl.multiple_of(chip_c0 + c0, 128), pair_w)] = chip_cols_buf[
            pl.ds(chip_r0, r_pair), c_sl
        ]

    pici_ops = []
    for d in range(1, n_chips):
        dst_chip = chip ^ jnp.int32(d)
        dst_r0 = pl.multiple_of(dst_chip * r_pair, 16)
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            chip_cols_buf.at[pl.ds(dst_r0, r_pair), :],
            post_pair_full.at[:, pl.ds(chip_c0, pair_cols)],
            post_ici_send.at[d - 1],
            post_ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        pici_ops.append(op)
    for op in pici_ops:
        op.wait()

    col_blk = min(hidden, 1024)
    if r_tok <= 16:
        my_p_off = pl.multiple_of(side * r_tok, 16) if r_tok == 16 else None

        def _load_my_post_static(c_sl):
            if r_tok == 8:
                pv = post_pair_full[:, c_sl]
                return lax.select(is_even, pv[0:8, :], pv[8:16, :]).astype(jnp.float32)
            return post_pair_full[pl.ds(my_p_off, 16), c_sl].astype(jnp.float32)

        b_sq = jnp.zeros((r_tok, 128), dtype=jnp.float32)
        for c0 in range(0, hidden, col_blk):
            c_sl = pl.ds(c0, col_blk)
            bc = _load_my_post_static(c_sl)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_sl]
            b_sq = b_sq + _fold_cols_to_128(bc * bc)
        inv_b = lax.rsqrt(
            jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )

        n_sq = jnp.zeros((r_tok, 128), dtype=jnp.float32)
        for c0 in range(0, hidden, col_blk):
            c_sl = pl.ds(c0, col_blk)
            bc = _load_my_post_static(c_sl)
            bg = post_ff_gain_ref[0:1, c_sl]
            bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
            ns = pf_alpha_ref[0:1, c_sl] * s_buf[:, c_sl] + pf_beta_ref[0:1, c_sl] * bn
            if round_stream_bf16_for_norm:
                ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
            s_buf[:, c_sl] = ns
            nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
            n_sq = n_sq + _fold_cols_to_128(nv * nv)

        if on_stream_updated_fn is not None:
            on_stream_updated_fn()

        inv_n = lax.rsqrt(
            jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(next_in_eps)
        )
        for c0 in range(0, hidden, col_blk):
            c_sl = pl.ds(c0, col_blk)
            sv = s_buf[:, c_sl]
            ng = next_in_gain_ref[0:1, c_sl]
            nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
            next_in_my_tok[:, c_sl] = nn.astype(jnp.bfloat16)
    else:
        bm_tok = _pick_strip(r_tok, 32)
        n_strips = hidden // _COL_STRIP
        u_full = _strip_unroll(n_strips, 4)

        def _row_epi(r0):
            rs = pl.ds(r0, bm_tok)
            p_off = pl.multiple_of(side * r_tok + r0, 16)
            rs_p = pl.ds(p_off, bm_tok)

            def _bvar(g, acc):
                base_s = g * u_full
                for sub in range(u_full):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    bc = post_pair_full[rs_p, c_sl].astype(jnp.float32)
                    if post_ff_scale_input:
                        bc = bc * post_ff_gain_ref[0:1, c_sl]
                    sq = bc * bc
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            b_sq = lax.fori_loop(
                0, n_strips // u_full, _bvar, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
            )
            inv_b = lax.rsqrt(
                jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(post_ff_eps)
            )

            def _res_nvar(g, acc):
                base_s = g * u_full
                for sub in range(u_full):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    bc = post_pair_full[rs_p, c_sl].astype(jnp.float32)
                    bg = post_ff_gain_ref[0:1, c_sl]
                    bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                    ns = pf_alpha_ref[0:1, c_sl] * s_buf[rs, c_sl] + pf_beta_ref[0:1, c_sl] * bn
                    if round_stream_bf16_for_norm:
                        ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                    s_buf[rs, c_sl] = ns
                    nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
                    sq = nv * nv
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            n_sq = lax.fori_loop(
                0, n_strips // u_full, _res_nvar, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
            )
            inv_n = lax.rsqrt(
                jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(next_in_eps)
            )

            def _nout(g, _):
                base_s = g * u_full
                for sub in range(u_full):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    sv = s_buf[rs, c_sl]
                    ng = next_in_gain_ref[0:1, c_sl]
                    nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                    next_in_my_tok[rs, c_sl] = nn.astype(jnp.bfloat16)
                return jnp.int32(0)

            lax.fori_loop(0, n_strips // u_full, _nout, jnp.int32(0))

        _fori_strips(r_tok, bm_tok, _row_epi)
        if on_stream_updated_fn is not None:
            on_stream_updated_fn()


def _ag_tok_sharded_half_col_xor1(
    my_tok_ref,
    mate_half_buf,
    out_full_ref,
    d2d1_send,
    d2d1_recv,
    ici_send,
    ici_recv,
    d2d2_send,
    d2d2_recv,
    *,
    me,
    world_size: int,
    during_d2d2_fn=None,
) -> None:
    """Half-column `XOR=1` token All-Gather (`[B/8, H] -> [B, H] bf16`, `-50%` inter-chip ICI!).

    Splits `H` across the two intra-chip cores (`side=0` gathers `0:H/2`, `side=1` gathers `H/2:H`
    over inter-chip ICI = `384 KiB` vs `768 KiB` at `B=64`), finishing with one `733 GB/s` D2D swap!
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok, hidden = my_tok_ref.shape
    r_pair = 2 * r_tok
    half_h = hidden // 2
    n_chips = world_size // 2

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    side_c0 = pl.multiple_of(side * half_h, 128)
    other_c0 = pl.multiple_of(other_side * half_h, 128)
    is_even = side == jnp.int32(0)

    op_d2d1 = pltpu.make_async_remote_copy(
        my_tok_ref.at[:, pl.ds(other_c0, half_h)],
        mate_half_buf,
        d2d1_send.at[0],
        d2d1_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_d2d1.start()
    op_d2d1.wait()

    ag_w = min(half_h, 1024 if r_tok <= 16 else 512)
    for c0 in range(0, half_h, ag_w):
        c_l = pl.ds(c0, ag_w)
        c_g = pl.ds(pl.multiple_of(side_c0 + c0, 128), ag_w)
        own_v = my_tok_ref[:, c_g]
        mate_v = mate_half_buf[:, c_l]
        out_full_ref[pl.ds(chip_r0, r_pair), c_g] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    ici_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            out_full_ref.at[pl.ds(chip_r0, r_pair), pl.ds(side_c0, half_h)],
            out_full_ref.at[pl.ds(chip_r0, r_pair), pl.ds(side_c0, half_h)],
            ici_send.at[d - 1],
            ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ici_ops.append(op)
    for op in ici_ops:
        op.wait()

    op_d2d2 = pltpu.make_async_remote_copy(
        out_full_ref.at[:, pl.ds(side_c0, half_h)],
        out_full_ref.at[:, pl.ds(side_c0, half_h)],
        d2d2_send.at[0],
        d2d2_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_d2d2.start()
    if during_d2d2_fn is not None:
        during_d2d2_fn()
    op_d2d2.wait()


def _moe_to_attn_tok_sharded_tp8_fast_bridge_kernel(
    stream_shard_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_w_qkv_hbm,
    next_s_qkv_ref,
    next_w_gate_hbm,
    next_s_gate_ref,
    cos_sin_ref,
    out_stream_shard_hbm,
    out_qkv_or_q_ref,
    out_kv_or_gate_ref,
    out_gate_epi_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_my_tok,
    next_in_mate_half,
    next_in_vmem,
    qkv_vmem,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_pair_full,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    in_d2d_send,
    in_d2d_recv,
    in_ici_send,
    in_ici_recv,
    in_d2d2_send,
    in_d2d2_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    fuse_qkv_epi: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Wave-10 Fast Token-Sharded Bridge 2b (`13` `tpu.matmul`s, `-50%` `in_ici`, `0` `scf.for`s!)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    r_tok = moe_out_ref.shape[0]
    rows = world_size * r_tok
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(stream_shard_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    _run_tok_sharded_post_tp8_fast_core(
        moe_out_ref, w_post_buf, s_post_ref, s_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
        next_in_my_tok, mate_tok_buf, moe_full_buf,
        post_shard, mate_post_shard, chip_cols_buf, post_pair_full,
        w_post_sem, s_in_sem,
        tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
        post_d2d_send, post_d2d_recv, post_ici_send, post_ici_recv,
        me=me, world_size=world_size,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
        round_stream_bf16_for_norm=False,
        on_stream_updated_fn=lambda: pltpu.make_async_copy(
            s_buf, out_stream_shard_hbm, s_out_sem.at[0]
        ).start(),
    )

    _ag_tok_sharded_half_col_xor1(
        next_in_my_tok, next_in_mate_half, next_in_vmem,
        in_d2d_send, in_d2d_recv, in_ici_send, in_ici_recv, in_d2d2_send, in_d2d2_recv,
        me=me, world_size=world_size,
        during_d2d2_fn=lambda: _dma_wait(w_qkv_buf, w_qkv_sem.at[0]),
    )

    qkv_dst = qkv_vmem if fuse_qkv_epi else out_qkv_or_q_ref
    gate_dst = out_gate_epi_ref if fuse_qkv_epi else out_kv_or_gate_ref
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_dst, 0,
        rows=rows, n_cols=qkv_cols,
    )
    if fuse_qkv_epi:
        _run_in_vmem_qkv_epilogue(
            qkv_vmem,
            cos_sin_ref,
            out_qkv_or_q_ref,
            out_kv_or_gate_ref,
            norm_eps=qkv_norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, gate_dst, 0,
        rows=rows, n_cols=gate_cols,
    )
    _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def _moe_post_tok_sharded_tp8_fast_final_norm_kernel(
    stream_shard_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    final_gain_ref,
    out_stream_shard_hbm,
    out_hidden_ref,
    w_post_buf,
    s_buf,
    hidden_my_tok,
    hidden_mate_half,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_pair_full,
    w_post_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    ag_d2d_send,
    ag_d2d_recv,
    ag_ici_send,
    ag_ici_recv,
    ag_d2d2_send,
    ag_d2d2_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    final_eps: float,
    final_scale_input: bool,
    return_stream_too: bool,
) -> None:
    """Wave-10 Fast Token-Sharded L61 Final Norm / Bridge 2a."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(stream_shard_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()

    _run_tok_sharded_post_tp8_fast_core(
        moe_out_ref, w_post_buf, s_post_ref, s_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, final_gain_ref,
        hidden_my_tok, mate_tok_buf, moe_full_buf,
        post_shard, mate_post_shard, chip_cols_buf, post_pair_full,
        w_post_sem, s_in_sem,
        tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
        post_d2d_send, post_d2d_recv, post_ici_send, post_ici_recv,
        me=me, world_size=world_size,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=final_eps, next_in_scale_input=final_scale_input,
        round_stream_bf16_for_norm=not return_stream_too,
        on_stream_updated_fn=(
            (lambda: pltpu.make_async_copy(s_buf, out_stream_shard_hbm, s_out_sem.at[0]).start())
            if return_stream_too
            else None
        ),
    )

    _ag_tok_sharded_half_col_xor1(
        hidden_my_tok, hidden_mate_half, out_hidden_ref,
        ag_d2d_send, ag_d2d_recv, ag_ici_send, ag_ici_recv, ag_d2d2_send, ag_d2d2_recv,
        me=me, world_size=world_size,
    )
    if return_stream_too:
        _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def _run_tok_sharded_post_tp8_core(
    moe_out_ref,
    w_post_buf,
    s_post_ref,
    s_buf,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_in_my_tok,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_chip_blocks,
    sq_tok_buf,
    w_post_sem,
    s_in_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    *,
    me,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    round_stream_bf16_for_norm: bool,
) -> None:
    """4-MiB ``TP=8`` ``W_post [4096, 1024]`` token-sharded Bridge-2 core (-12 MiB/L HBM!)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok, k_in = moe_out_ref.shape
    r_pair = 2 * r_tok
    post_cols = w_post_buf.shape[1]  # 1024
    pair_cols = 2 * post_cols  # 2048
    hidden = world_size * post_cols  # 8192
    n_chips = world_size // 2  # 4
    bm_tok = 8 if r_tok == 8 else 16
    pair_strips = pair_cols // _COL_STRIP

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    is_even = side == jnp.int32(0)

    # Stage 1: Token-AG moe_out [r_tok, 4096] -> moe_full_buf [rows, 4096] overlapped with W_post!
    _wait_barrier(world_size)
    ts1 = pltpu.make_async_remote_copy(
        moe_out_ref, mate_tok_buf, tok_s1_send.at[0], tok_s1_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    ts1.start()
    ts1.wait()

    k_w = min(k_in, 512)
    for c0 in range(0, k_in, k_w):
        c_sl = pl.ds(c0, k_w)
        own_v = moe_out_ref[:, c_sl]
        mate_v = mate_tok_buf[:, c_sl]
        moe_full_buf[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    ts2_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            moe_full_buf.at[pl.ds(chip_r0, r_pair), :],
            moe_full_buf.at[pl.ds(chip_r0, r_pair), :],
            tok_s2_send.at[d - 1],
            tok_s2_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ts2_ops.append(op)

    _dma_wait(w_post_buf, w_post_sem.at[0])
    _dot_fp8_row_block_to_ref(
        moe_full_buf, chip_r0, w_post_buf, s_post_ref, post_shard,
        n_r=r_pair, n_cols=post_cols, align=16,
    )
    for op in ts2_ops:
        op.wait()
    for d in range(1, n_chips):
        p_r0 = pl.multiple_of((chip ^ jnp.int32(d)) * r_pair, 16)
        _dot_fp8_row_block_to_ref(
            moe_full_buf, p_r0, w_post_buf, s_post_ref, post_shard,
            n_r=r_pair, n_cols=post_cols, align=16,
        )

    # Stage 2: Chipmate D2D [rows, 1024] -> [rows, 2048] + 192 KiB 1-stage ICI scatter!
    op_pd2d = pltpu.make_async_remote_copy(
        post_shard, mate_post_shard, post_d2d_send.at[0], post_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_pd2d.start()
    op_pd2d.wait()

    cs_w = min(post_cols, 512)
    for c0 in range(0, post_cols, cs_w):
        c_sl = pl.ds(c0, cs_w)
        s_v = post_shard[:, c_sl]
        m_v = mate_post_shard[:, c_sl]
        chip_cols_buf[:, pl.ds(c0, cs_w)] = lax.select(is_even, s_v, m_v)
        chip_cols_buf[:, pl.ds(post_cols + c0, cs_w)] = lax.select(is_even, m_v, s_v)

    pair_w = min(pair_cols, 512)
    for c0 in range(0, pair_cols, pair_w):
        c_sl = pl.ds(c0, pair_w)
        post_chip_blocks[chip, :, c_sl] = chip_cols_buf[pl.ds(chip_r0, r_pair), c_sl]

    pici_ops = []
    for d in range(1, n_chips):
        dst_chip = chip ^ jnp.int32(d)
        dst_r0 = pl.multiple_of(dst_chip * r_pair, 16)
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            chip_cols_buf.at[pl.ds(dst_r0, r_pair), :],
            post_chip_blocks.at[chip],
            post_ici_send.at[d - 1],
            post_ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        pici_ops.append(op)

    def _load_my_post_strip(b_idx, r0_tok, c_l):
        if r_tok == 8:
            pair_v = post_chip_blocks[b_idx, :, c_l]
            return lax.select(is_even, pair_v[0:8, :], pair_v[8:16, :])
        my_off = pl.multiple_of(side * r_tok + r0_tok, 16)
        return post_chip_blocks[b_idx, pl.ds(my_off, bm_tok), c_l]

    own_c_base = pl.multiple_of(chip * pair_cols, 128)

    def _own_pf_sq(r0):
        rs = pl.ds(r0, bm_tok)

        def _col(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_g = pl.ds(pl.multiple_of(own_c_base + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            bc = _load_my_post_strip(chip, r0, c_l).astype(jnp.float32)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_g]
            sq = bc * bc
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq_tok_buf[rs, :] = lax.fori_loop(
            0, pair_strips, _col, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
        )

    _fori_strips(r_tok, bm_tok, _own_pf_sq)
    for op in pici_ops:
        op.wait()
    _dma_wait(s_buf, s_in_sem.at[0])

    def _finish_pf_and_next_ln(r0):
        rs = pl.ds(r0, bm_tok)
        b1 = chip ^ jnp.int32(1)
        b2 = chip ^ jnp.int32(2)
        b3 = chip ^ jnp.int32(3)
        c1_base = pl.multiple_of(b1 * pair_cols, 128)
        c2_base = pl.multiple_of(b2 * pair_cols, 128)
        c3_base = pl.multiple_of(b3 * pair_cols, 128)

        def _all_oth_sq(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_g1 = pl.ds(pl.multiple_of(c1_base + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_g2 = pl.ds(pl.multiple_of(c2_base + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_g3 = pl.ds(pl.multiple_of(c3_base + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            v1 = _load_my_post_strip(b1, r0, c_l).astype(jnp.float32)
            v2 = _load_my_post_strip(b2, r0, c_l).astype(jnp.float32)
            v3 = _load_my_post_strip(b3, r0, c_l).astype(jnp.float32)
            if post_ff_scale_input:
                v1 = v1 * post_ff_gain_ref[0:1, c_g1]
                v2 = v2 * post_ff_gain_ref[0:1, c_g2]
                v3 = v3 * post_ff_gain_ref[0:1, c_g3]
            sq = v1 * v1 + v2 * v2 + v3 * v3
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        tot_sq = sq_tok_buf[rs, :] + lax.fori_loop(
            0, pair_strips, _all_oth_sq, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
        )
        inv_b = lax.rsqrt(
            jnp.sum(tot_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )

        def _all_res_blk(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            sq_sum = jnp.zeros((bm_tok, _COL_STRIP), dtype=jnp.float32)
            for b_static in range(n_chips):
                c_g = pl.ds(pl.multiple_of(b_static * pair_cols + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                bc = _load_my_post_strip(b_static, r0, c_l).astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_g]
                bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                ns = pf_alpha_ref[0:1, c_g] * s_buf[rs, c_g] + pf_beta_ref[0:1, c_g] * bn
                if round_stream_bf16_for_norm:
                    ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                s_buf[rs, c_g] = ns
                nv = ns * next_in_gain_ref[0:1, c_g] if next_in_scale_input else ns
                sq_sum = sq_sum + nv * nv
            return acc + (sq_sum[:, 0:128] + sq_sum[:, 128:256])

        n_sq = lax.fori_loop(
            0, pair_strips, _all_res_blk, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
        )
        inv_n = lax.rsqrt(
            jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(next_in_eps)
        )

        def _nout(s, _):
            c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            sv = s_buf[rs, c_g]
            ng = next_in_gain_ref[0:1, c_g]
            nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
            next_in_my_tok[rs, c_g] = nn.astype(jnp.bfloat16)
            return jnp.int32(0)

        lax.fori_loop(0, hidden // _COL_STRIP, _nout, jnp.int32(0))

    _fori_strips(r_tok, bm_tok, _finish_pf_and_next_ln)


def _run_tok_sharded_post_core(
    moe_out_ref,
    w_post_pair_buf,
    s_post_pair_ref,
    s_in_buf,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    s_out_buf,
    next_in_my_tok,
    mate_tok_buf,
    moe_pair_buf,
    post_pair_half,
    post_my_tok,
    post_send_tok,
    post_recv_tok,
    sq_tok_buf,
    w_post_sem,
    s_in_sem,
    tok_s1_send,
    tok_s1_recv,
    post_d2d_send,
    post_d2d_recv,
    *,
    me,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    round_stream_bf16_for_norm: bool,
) -> None:
    """Shared token-sharded Bridge-2 core: 1 D2D + 16-row ``W_post`` + 1 D2D + 8-row VPU norms!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok, k_in = moe_out_ref.shape
    r_pair = 2 * r_tok
    post_half_cols = w_post_pair_buf.shape[1]  # 4096
    hidden = 2 * post_half_cols  # 8192
    bm_tok = _pick_strip(r_tok, 16)
    half_strips = post_half_cols // _COL_STRIP

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_c0 = pl.multiple_of(side * post_half_cols, 128)
    other_c0 = pl.multiple_of(other_side * post_half_cols, 128)
    is_even = side == jnp.int32(0)

    # Stage 1 (ts1 D2D): [r_tok, 4096] -> [r_pair, 4096] (DELETES ts2 4-chip ICI!).
    _wait_barrier(world_size)
    ts1 = pltpu.make_async_remote_copy(
        moe_out_ref, mate_tok_buf, tok_s1_send.at[0], tok_s1_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    ts1.start()
    ts1.wait()

    for c0 in range(0, k_in, 512):
        c_sl = pl.ds(c0, 512)
        own_v = moe_out_ref[:, c_sl]
        mate_v = mate_tok_buf[:, c_sl]
        moe_pair_buf[:, c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    # 16-row W_post_pair MXU [r_pair, 4096] @ [4096, 4096]!
    _dma_wait(w_post_pair_buf, w_post_sem.at[0])
    _dot_fp8_scaled_to_ref(
        moe_pair_buf, w_post_pair_buf, s_post_pair_ref, 0, post_pair_half, 0,
        rows=r_pair, n_cols=post_half_cols,
    )

    for c0 in range(0, post_half_cols, 512):
        c_sl = pl.ds(c0, 512)
        blk = post_pair_half[:, c_sl]
        post_my_tok[:, c_sl] = lax.select(is_even, blk[0:r_tok, :], blk[r_tok:r_pair, :])
        post_send_tok[:, c_sl] = lax.select(is_even, blk[r_tok:r_pair, :], blk[0:r_tok, :])

    # Stage 2 (post_d2d D2D): [r_pair, 4096] -> [r_tok, 8192] (DELETES ag_s1+s2+s3!)
    op_pd2d = pltpu.make_async_remote_copy(
        post_send_tok, post_recv_tok, post_d2d_send.at[0], post_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_pd2d.start()

    def _own_pf_sq(r0):
        rs = pl.ds(r0, bm_tok)

        def _col(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_own = pl.ds(pl.multiple_of(own_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            bc = post_my_tok[rs, c_l].astype(jnp.float32)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_own]
            sq = bc * bc
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq_tok_buf[rs, :] = lax.fori_loop(
            0, half_strips, _col, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
        )

    _fori_strips(r_tok, bm_tok, _own_pf_sq)
    op_pd2d.wait()
    _dma_wait(s_in_buf, s_in_sem.at[0])

    # 8-row VPU post_ffn_ln + gated_res + next_in_ln (8x less VPU work!)
    def _finish_pf_and_next_ln(r0):
        rs = pl.ds(r0, bm_tok)

        def _oth_sq(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_oth = pl.ds(pl.multiple_of(other_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            bc = post_recv_tok[rs, c_l].astype(jnp.float32)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_oth]
            sq = bc * bc
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        oth_sq = lax.fori_loop(
            0, half_strips, _oth_sq, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
        )
        inv_b = lax.rsqrt(
            jnp.sum(sq_tok_buf[rs, :] + oth_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )

        def _res_nvar(s, acc):
            c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_own = pl.ds(pl.multiple_of(own_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            c_oth = pl.ds(pl.multiple_of(other_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            bc_own = post_my_tok[rs, c_l].astype(jnp.float32)
            bc_oth = post_recv_tok[rs, c_l].astype(jnp.float32)
            bg_own = post_ff_gain_ref[0:1, c_own]
            bg_oth = post_ff_gain_ref[0:1, c_oth]
            bn_own = (bc_own * bg_own) * inv_b if post_ff_scale_input else (bc_own * inv_b) * bg_own
            bn_oth = (bc_oth * bg_oth) * inv_b if post_ff_scale_input else (bc_oth * inv_b) * bg_oth
            ns_own = pf_alpha_ref[0:1, c_own] * s_in_buf[rs, c_own] + pf_beta_ref[0:1, c_own] * bn_own
            ns_oth = pf_alpha_ref[0:1, c_oth] * s_in_buf[rs, c_oth] + pf_beta_ref[0:1, c_oth] * bn_oth
            if round_stream_bf16_for_norm:
                ns_own = ns_own.astype(jnp.bfloat16).astype(jnp.float32)
                ns_oth = ns_oth.astype(jnp.bfloat16).astype(jnp.float32)
            s_out_buf[rs, c_own] = ns_own
            s_out_buf[rs, c_oth] = ns_oth
            nv_own = ns_own * next_in_gain_ref[0:1, c_own] if next_in_scale_input else ns_own
            nv_oth = ns_oth * next_in_gain_ref[0:1, c_oth] if next_in_scale_input else ns_oth
            sq = nv_own * nv_own + nv_oth * nv_oth
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        n_sq = lax.fori_loop(
            0, half_strips, _res_nvar, jnp.zeros((bm_tok, 128), dtype=jnp.float32)
        )
        inv_n = lax.rsqrt(
            jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(next_in_eps)
        )

        def _nout(s, _):
            c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            sv = s_out_buf[rs, c_g]
            ng = next_in_gain_ref[0:1, c_g]
            nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
            next_in_my_tok[rs, c_g] = nn.astype(jnp.bfloat16)
            return jnp.int32(0)

        lax.fori_loop(0, hidden // _COL_STRIP, _nout, jnp.int32(0))

    _fori_strips(r_tok, bm_tok, _finish_pf_and_next_ln)


def _moe_to_attn_tok_sharded_tp8_bridge_kernel(
    stream_shard_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_w_qkv_hbm,
    next_s_qkv_ref,
    next_w_gate_hbm,
    next_s_gate_ref,
    cos_sin_ref,
    out_stream_shard_hbm,
    out_qkv_or_q_ref,
    out_kv_or_gate_ref,
    out_gate_epi_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_my_tok,
    next_in_mate_tok,
    next_in_vmem,
    qkv_vmem,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_chip_blocks,
    sq_tok_buf,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    in_d2d_send,
    in_d2d_recv,
    in_ici_send,
    in_ici_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    fuse_qkv_epi: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """4-MiB ``TP=8`` ``W_post`` Token-Sharded Bridge 2b (-12 MiB/L HBM & 0 extra weight HBM!)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok = moe_out_ref.shape[0]
    r_pair = 2 * r_tok
    hidden = world_size * w_post_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    n_chips = world_size // 2

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    is_even = side == jnp.int32(0)

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_shard_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    _run_tok_sharded_post_tp8_core(
        moe_out_ref, w_post_buf, s_post_ref, s_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
        next_in_my_tok, mate_tok_buf, moe_full_buf,
        post_shard, mate_post_shard, chip_cols_buf, post_chip_blocks, sq_tok_buf,
        w_post_sem, s_in_sem,
        tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
        post_d2d_send, post_d2d_recv, post_ici_send, post_ici_recv,
        me=me, world_size=world_size,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
        round_stream_bf16_for_norm=False,
    )
    pltpu.make_async_copy(s_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    op_id2d = pltpu.make_async_remote_copy(
        next_in_my_tok, next_in_mate_tok, in_d2d_send.at[0], in_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_id2d.start()
    op_id2d.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_v = next_in_my_tok[:, c_sl]
        mate_v = next_in_mate_tok[:, c_sl]
        next_in_vmem[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    ici_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            next_in_vmem.at[pl.ds(chip_r0, r_pair), :],
            next_in_vmem.at[pl.ds(chip_r0, r_pair), :],
            in_ici_send.at[d - 1],
            in_ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ici_ops.append(op)

    # Finish 768-KiB in_ici under W_qkv (10 MiB) HBM DMA, run 1 full-batch W_qkv dot,
    # and overlap VPU QKV/RoPE epilogue directly underneath W_gate (8 MiB) HBM DMA!
    rows = world_size * r_tok
    qkv_dst = qkv_vmem if fuse_qkv_epi else out_qkv_or_q_ref
    gate_dst = out_gate_epi_ref if fuse_qkv_epi else out_kv_or_gate_ref
    for op in ici_ops:
        op.wait()

    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_dst, 0,
        rows=rows, n_cols=qkv_cols,
    )
    if fuse_qkv_epi:
        _run_in_vmem_qkv_epilogue(
            qkv_vmem,
            cos_sin_ref,
            out_qkv_or_q_ref,
            out_kv_or_gate_ref,
            norm_eps=qkv_norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, gate_dst, 0,
        rows=rows, n_cols=gate_cols,
    )
    _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def _moe_to_attn_tok_sharded_bridge_kernel(
    stream_shard_hbm,
    moe_out_ref,
    w_post_pair_hbm,
    s_post_pair_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_w_qkv_hbm,
    next_s_qkv_ref,
    next_w_gate_hbm,
    next_s_gate_ref,
    cos_sin_ref,
    out_stream_shard_hbm,
    out_qkv_or_q_ref,
    out_kv_or_gate_ref,
    out_gate_epi_ref,
    w_post_pair_buf,
    w_qkv_buf,
    w_gate_buf,
    s_in_buf,
    s_out_buf,
    next_in_my_tok,
    next_in_mate_tok,
    next_in_vmem,
    qkv_vmem,
    mate_tok_buf,
    moe_pair_buf,
    post_pair_half,
    post_my_tok,
    post_send_tok,
    post_recv_tok,
    sq_tok_buf,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    post_d2d_send,
    post_d2d_recv,
    in_d2d_send,
    in_d2d_recv,
    in_ici_send,
    in_ici_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    fuse_qkv_epi: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Wave-8 Token-Sharded Bridge 2b: ONLY 1 ICI stage (vs 2) & 8x less VPU RMSNorm work!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok = moe_out_ref.shape[0]
    r_pair = 2 * r_tok
    hidden = 2 * w_post_pair_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    n_chips = world_size // 2

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    is_even = side == jnp.int32(0)

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_pair_hbm, w_post_pair_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_shard_hbm, s_in_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    _run_tok_sharded_post_core(
        moe_out_ref, w_post_pair_buf, s_post_pair_ref, s_in_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
        s_out_buf, next_in_my_tok, mate_tok_buf, moe_pair_buf,
        post_pair_half, post_my_tok, post_send_tok, post_recv_tok, sq_tok_buf,
        w_post_sem, s_in_sem, tok_s1_send, tok_s1_recv, post_d2d_send, post_d2d_recv,
        me=me, world_size=world_size,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
        round_stream_bf16_for_norm=False,
    )
    pltpu.make_async_copy(s_out_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    # Stage 3 (in_d2d): [r_tok -> r_pair, 8192]
    op_id2d = pltpu.make_async_remote_copy(
        next_in_my_tok, next_in_mate_tok, in_d2d_send.at[0], in_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_id2d.start()
    op_id2d.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_v = next_in_my_tok[:, c_sl]
        mate_v = next_in_mate_tok[:, c_sl]
        next_in_vmem[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    # Stage 4 (in_ici): 1-stage inter-chip Token-AG [r_pair -> rows, 8192] OVERLAPPED with
    # own-chip r_pair rows of next_qkv & next_gate MXU!
    ici_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            next_in_vmem.at[pl.ds(chip_r0, r_pair), :],
            next_in_vmem.at[pl.ds(chip_r0, r_pair), :],
            in_ici_send.at[d - 1],
            in_ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ici_ops.append(op)

    rows = world_size * r_tok
    qkv_dst = qkv_vmem if fuse_qkv_epi else out_qkv_or_q_ref
    gate_dst = out_gate_epi_ref if fuse_qkv_epi else out_kv_or_gate_ref
    for op in ici_ops:
        op.wait()

    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_dst, 0,
        rows=rows, n_cols=qkv_cols,
    )
    if fuse_qkv_epi:
        _run_in_vmem_qkv_epilogue(
            qkv_vmem,
            cos_sin_ref,
            out_qkv_or_q_ref,
            out_kv_or_gate_ref,
            norm_eps=qkv_norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, gate_dst, 0,
        rows=rows, n_cols=gate_cols,
    )
    _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def moe_to_attn_tok_sharded_bridge_jax(
    stream_shard,
    moe_out,
    w_post_pair_w8k,
    s_post_pair,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    next_in_gain,
    next_w_qkv_w8k,
    next_s_qkv,
    next_w_gate_w8k,
    next_s_gate,
    cos_sin_step=None,
    *,
    world_size: int = 8,
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    fuse_qkv_epi: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    collective_id: int = _W8_CID_BASE + 5,
    interpret: bool = False,
):
    """Token-sharded Bridge 2b (`(stream_shard, moe_out) -> (stream_shard, qkv / q+kv, gate)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        raise ValueError(f"Token-sharded Bridge 2b requires world_size=8, got {world_size}")
    r_tok, k_in = moe_out.shape
    rows = world_size * r_tok
    r_pair = 2 * r_tok
    _, post_cols = w_post_pair_w8k.shape
    _, hidden = stream_shard.shape
    use_tp8_post = post_cols * world_size == hidden
    if (
        rows not in (64, 128, 256, 384)
        or r_tok % 8 != 0
        or (not use_tp8_post and 2 * post_cols != hidden)
    ):
        raise ValueError("Unsupported token-sharded Bridge 2b shapes")
    has_rope = bool(fuse_qkv_epi and cos_sin_step is not None)
    cs_in = cos_sin_step[:rows] if (has_rope and cos_sin_step.shape[0] != rows) else cos_sin_step
    _, qkv_cols = next_w_qkv_w8k.shape
    _, gate_cols = next_w_gate_w8k.shape
    n_chips = world_size // 2
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    if fuse_qkv_epi:
        out_shape = (
            jax.ShapeDtypeStruct((r_tok, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        )
        out_specs = (h_sp, v_sp, v_sp, v_sp)
    else:
        out_shape = (
            jax.ShapeDtypeStruct((r_tok, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, qkv_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        )
        out_specs = (h_sp, v_sp, v_sp, v_sp)

    use_fast_tp8 = use_tp8_post and _tok_b2_fast_enabled()
    if use_fast_tp8:
        pair_cols = 2 * post_cols
        half_h = hidden // 2
        scratch_b2 = [
            vmem((k_in, post_cols), w_post_pair_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, half_h), jnp.bfloat16),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, qkv_cols) if fuse_qkv_epi else (8, 128), jnp.bfloat16),
            vmem((r_tok, k_in), jnp.bfloat16),
            vmem((rows, k_in), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, pair_cols), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
        ]
        kern_b2 = _moe_to_attn_tok_sharded_tp8_fast_bridge_kernel
    elif use_tp8_post:
        pair_cols = 2 * post_cols
        scratch_b2 = [
            vmem((k_in, post_cols), w_post_pair_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, qkv_cols) if fuse_qkv_epi else (8, 128), jnp.bfloat16),
            vmem((r_tok, k_in), jnp.bfloat16),
            vmem((rows, k_in), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, pair_cols), jnp.bfloat16),
            vmem((n_chips, r_pair, pair_cols), jnp.bfloat16),
            vmem((r_tok, 128), jnp.float32),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ]
        kern_b2 = _moe_to_attn_tok_sharded_tp8_bridge_kernel
    else:
        scratch_b2 = [
            vmem((k_in, post_cols), w_post_pair_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, qkv_cols) if fuse_qkv_epi else (8, 128), jnp.bfloat16),
            vmem((r_tok, k_in), jnp.bfloat16),
            vmem((r_pair, k_in), jnp.bfloat16),
            vmem((r_pair, post_cols), jnp.bfloat16),
            vmem((r_tok, post_cols), jnp.bfloat16),
            vmem((r_tok, post_cols), jnp.bfloat16),
            vmem((r_tok, post_cols), jnp.bfloat16),
            vmem((r_tok, 128), jnp.float32),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ]
        kern_b2 = _moe_to_attn_tok_sharded_bridge_kernel

    res = _pallas_call_maybe_interpret(
        functools.partial(
            kern_b2,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            fuse_qkv_epi=bool(fuse_qkv_epi),
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            has_rope=has_rope,
            interpret_mode=interpret,
        ),
        interpret=interpret,
        out_shape=out_shape,
        in_specs=[
            h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp,
            h_sp, v_sp, h_sp, v_sp, v_sp if has_rope else None,
        ],
        out_specs=out_specs,
        scratch_shapes=scratch_b2,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=56 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w8_tok_b2_b{rows}_tp{world_size}"
            f"{'_p8' if use_tp8_post else ''}"
            f"{'_epi_' + ('rope' if has_rope else 'nope') if fuse_qkv_epi else ''}"
        ),
    )(
        stream_shard.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_pair_w8k,
        _as_2d_f32(s_post_pair, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_2d_f32(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_2d_f32(next_s_gate, gate_cols),
        cs_in if has_rope else None,
    )
    if fuse_qkv_epi:
        st_sh, q_3d, kv_m, gate = res
        return st_sh, q_3d.reshape(1, rows, 8, 2, 128), kv_m, gate
    st_sh, qkv, gate, _ = res
    return st_sh, qkv, gate


def _moe_post_tok_sharded_tp8_final_norm_kernel(
    stream_shard_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    final_gain_ref,
    out_stream_shard_hbm,
    out_hidden_ref,
    w_post_buf,
    s_buf,
    hidden_my_tok,
    hidden_mate_tok,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_chip_blocks,
    sq_tok_buf,
    w_post_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    ag_d2d_send,
    ag_d2d_recv,
    ag_ici_send,
    ag_ici_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    final_eps: float,
    final_scale_input: bool,
    return_stream_too: bool,
) -> None:
    """4-MiB ``TP=8`` ``W_post`` Token-sharded L61 Final Norm / Bridge 2a."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok = moe_out_ref.shape[0]
    r_pair = 2 * r_tok
    hidden = world_size * w_post_hbm.shape[1]
    n_chips = world_size // 2

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    is_even = side == jnp.int32(0)

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_shard_hbm, s_buf, s_in_sem.at[0]).start()

    _run_tok_sharded_post_tp8_core(
        moe_out_ref, w_post_buf, s_post_ref, s_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, final_gain_ref,
        hidden_my_tok, mate_tok_buf, moe_full_buf,
        post_shard, mate_post_shard, chip_cols_buf, post_chip_blocks, sq_tok_buf,
        w_post_sem, s_in_sem,
        tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
        post_d2d_send, post_d2d_recv, post_ici_send, post_ici_recv,
        me=me, world_size=world_size,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=final_eps, next_in_scale_input=final_scale_input,
        round_stream_bf16_for_norm=not return_stream_too,
    )
    if return_stream_too:
        pltpu.make_async_copy(s_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    op_d2d = pltpu.make_async_remote_copy(
        hidden_my_tok, hidden_mate_tok, ag_d2d_send.at[0], ag_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_d2d.start()
    op_d2d.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_v = hidden_my_tok[:, c_sl]
        mate_v = hidden_mate_tok[:, c_sl]
        out_hidden_ref[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    ici_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            out_hidden_ref.at[pl.ds(chip_r0, r_pair), :],
            out_hidden_ref.at[pl.ds(chip_r0, r_pair), :],
            ag_ici_send.at[d - 1],
            ag_ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ici_ops.append(op)
    for op in ici_ops:
        op.wait()
    if return_stream_too:
        _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def _moe_post_tok_sharded_final_norm_kernel(
    stream_shard_hbm,
    moe_out_ref,
    w_post_pair_hbm,
    s_post_pair_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    final_gain_ref,
    out_stream_shard_hbm,
    out_hidden_ref,
    w_post_pair_buf,
    s_in_buf,
    s_out_buf,
    hidden_my_tok,
    hidden_mate_tok,
    mate_tok_buf,
    moe_pair_buf,
    post_pair_half,
    post_my_tok,
    post_send_tok,
    post_recv_tok,
    sq_tok_buf,
    w_post_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    post_d2d_send,
    post_d2d_recv,
    ag_d2d_send,
    ag_d2d_recv,
    ag_ici_send,
    ag_ici_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    final_eps: float,
    final_scale_input: bool,
    return_stream_too: bool,
) -> None:
    """Token-sharded L61 Final Norm (or Bridge 2a when ``return_stream_too=True``)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_tok = moe_out_ref.shape[0]
    r_pair = 2 * r_tok
    hidden = 2 * w_post_pair_hbm.shape[1]
    n_chips = world_size // 2

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)
    is_even = side == jnp.int32(0)

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_pair_hbm, w_post_pair_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_shard_hbm, s_in_buf, s_in_sem.at[0]).start()

    _run_tok_sharded_post_core(
        moe_out_ref, w_post_pair_buf, s_post_pair_ref, s_in_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, final_gain_ref,
        s_out_buf, hidden_my_tok, mate_tok_buf, moe_pair_buf,
        post_pair_half, post_my_tok, post_send_tok, post_recv_tok, sq_tok_buf,
        w_post_sem, s_in_sem, tok_s1_send, tok_s1_recv, post_d2d_send, post_d2d_recv,
        me=me, world_size=world_size,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=final_eps, next_in_scale_input=final_scale_input,
        round_stream_bf16_for_norm=not return_stream_too,
    )
    if return_stream_too:
        pltpu.make_async_copy(s_out_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    op_d2d = pltpu.make_async_remote_copy(
        hidden_my_tok, hidden_mate_tok, ag_d2d_send.at[0], ag_d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_d2d.start()
    op_d2d.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_v = hidden_my_tok[:, c_sl]
        mate_v = hidden_mate_tok[:, c_sl]
        out_hidden_ref[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
            [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
        )

    ici_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            out_hidden_ref.at[pl.ds(chip_r0, r_pair), :],
            out_hidden_ref.at[pl.ds(chip_r0, r_pair), :],
            ag_ici_send.at[d - 1],
            ag_ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ici_ops.append(op)
    for op in ici_ops:
        op.wait()
    if return_stream_too:
        _dma_wait(out_stream_shard_hbm, s_out_sem.at[0])


def moe_post_tok_sharded_final_norm_jax(
    stream_shard,
    moe_out,
    w_post_pair_w8k,
    s_post_pair,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    final_gain,
    *,
    world_size: int = 8,
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    final_eps: float = 1e-5,
    final_scale_input: bool = False,
    return_stream_too: bool = False,
    collective_id: int = _W8_CID_BASE + 6,
    interpret: bool = False,
):
    """Token-sharded L61 Final Norm (`return_stream_too=False -> hidden [B, H] bf16`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        raise ValueError(f"Token-sharded Final Norm requires world_size=8, got {world_size}")
    r_tok, k_in = moe_out.shape
    rows = world_size * r_tok
    r_pair = 2 * r_tok
    _, post_cols = w_post_pair_w8k.shape
    _, hidden = stream_shard.shape
    use_tp8_post = post_cols * world_size == hidden
    if (
        rows not in (64, 128, 256, 384)
        or r_tok % 8 != 0
        or (not use_tp8_post and 2 * post_cols != hidden)
    ):
        raise ValueError("Unsupported token-sharded Final Norm shapes")
    n_chips = world_size // 2
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    use_fast_tp8 = use_tp8_post and _tok_b2_fast_enabled()
    if use_fast_tp8:
        pair_cols = 2 * post_cols
        half_h = hidden // 2
        scratch_fn = [
            vmem((k_in, post_cols), w_post_pair_w8k.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, half_h), jnp.bfloat16),
            vmem((r_tok, k_in), jnp.bfloat16),
            vmem((rows, k_in), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, pair_cols), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
        ]
        kern_fn = _moe_post_tok_sharded_tp8_fast_final_norm_kernel
    elif use_tp8_post:
        pair_cols = 2 * post_cols
        scratch_fn = [
            vmem((k_in, post_cols), w_post_pair_w8k.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, k_in), jnp.bfloat16),
            vmem((rows, k_in), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((rows, pair_cols), jnp.bfloat16),
            vmem((n_chips, r_pair, pair_cols), jnp.bfloat16),
            vmem((r_tok, 128), jnp.float32),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ]
        kern_fn = _moe_post_tok_sharded_tp8_final_norm_kernel
    else:
        scratch_fn = [
            vmem((k_in, post_cols), w_post_pair_w8k.dtype),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.float32),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, hidden), jnp.bfloat16),
            vmem((r_tok, k_in), jnp.bfloat16),
            vmem((r_pair, k_in), jnp.bfloat16),
            vmem((r_pair, post_cols), jnp.bfloat16),
            vmem((r_tok, post_cols), jnp.bfloat16),
            vmem((r_tok, post_cols), jnp.bfloat16),
            vmem((r_tok, post_cols), jnp.bfloat16),
            vmem((r_tok, 128), jnp.float32),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ]
        kern_fn = _moe_post_tok_sharded_final_norm_kernel

    st_out, hid_out = _pallas_call_maybe_interpret(
        functools.partial(
            kern_fn,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            final_eps=float(final_eps),
            final_scale_input=bool(final_scale_input),
            return_stream_too=bool(return_stream_too),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((r_tok, hidden) if return_stream_too else (8, 128), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=(h_sp if return_stream_too else v_sp, v_sp),
        scratch_shapes=scratch_fn,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=42 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w8_tok_fnorm_b{rows}_tp{world_size}"
            f"{'_p8' if use_tp8_post else ''}{'_st' if return_stream_too else ''}"
        ),
    )(
        stream_shard.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_pair_w8k,
        _as_2d_f32(s_post_pair, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(final_gain, hidden),
    )
    return (st_out, hid_out) if return_stream_too else hid_out


# =============================================================================
# 5b. Wave-9 Chip-Granularity Stream Bridges (`stream [B//(TP//2), 8192]`, TP=8 & TP=16)
# =============================================================================
def _run_ag_ici_step(
    specs,
    ici_send,
    ici_recv,
    *,
    sem_base: int,
    n_r: int,
    overlap_fn=None,
) -> None:
    """Issue one parallel multi-slab ICI doubling step and optionally overlap work."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    ops = [
        pltpu.make_async_remote_copy(
            slab.at[pl.ds(pl.multiple_of(r0, 16), n_r), :],
            slab.at[pl.ds(pl.multiple_of(r0, 16), n_r), :],
            ici_send.at[sem_base + idx],
            ici_recv.at[sem_base + idx],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for idx, (slab, r0, peer) in enumerate(specs)
    ]
    for op in ops:
        op.start()
    if overlap_fn is not None:
        overlap_fn()
    for op in ops:
        op.wait()


def _use_tp16_parallel_xyz_ag(
    *,
    world_size: int,
    is_3d_out: bool,
    r_pair: int,
    half_h: int,
) -> bool:
    """Return True when 3-slab parallel XYZ doubling beats 7-peer AG on TP=16 silicon.

    Verified bit-exact on 2x2x2 TPU7x across 2D ``[B, H]`` all-gathers with
    ``r_pair * half_h >= 65536`` (``B=128 H=8192``, ``B=256 H>=4096``, ``B=384``);
    smaller 2D payloads (``<= 32768``) and 3D ``out_expert_3d_ref`` stay on 7-peer.
    """
    return (
        world_size == 16
        and not is_3d_out
        and r_pair >= 16
        and r_pair % 16 == 0
        and half_h >= 384
        and (half_h % 128 == 0)
        and (r_pair * half_h) >= 65536
    )


def _use_tp8_parallel_xy_ag(
    *,
    world_size: int,
    is_3d_out: bool,
    r_pair: int,
    half_h: int,
    n_ici_sems: int = 5,
) -> bool:
    """Return True when in-place 2-slab parallel XY doubling beats 3-peer AG on TP=8 silicon.

    Verified bit-exact on 2x2x1 TPU7x for ``B >= 128, H >= 8192`` (``r_pair >= 32,
    half_h >= 4096``: ``-3.74 us/L`` on ``B=128`` ``Step (c)``, ``-2.47..-2.60 us/L``
    on ``B=128/256`` ``ChipB2_epi``, ``-3.22 us`` on ``L61_Final B=256``). Smaller
    ``B <= 64`` (`r_pair=16`) or ``half_h < 4096`` stay on 3-peer 1-shot.
    Requires ``>= 5`` ICI send/recv semaphores (Hop 1-2 disjoint + 2nd X-slab chunk).
    """
    return (
        world_size == 8
        and not is_3d_out
        and n_ici_sems >= 5
        and r_pair >= 32
        and (r_pair % 16 == 0)
        and half_h >= 4096
        and (half_h % 128 == 0)
    )


def _tp16_parallel_xyz_ag(
    pair_ref,
    out_full_ref,
    ici_send,
    ici_recv,
    d2d2_send,
    d2d2_recv,
    *,
    me,
    mate,
    side,
    other_side,
    chip,
    r_pair: int,
    half_h: int,
    side_c0,
    during_ici_fn,
    during_d2d2_fn,
    copy_local_from_pair: bool,
    during_ici_early: bool = False,
) -> None:
    """3-slab parallel XYZ doubling all-gather for TP=16 (``-42%`` link volume, ``0`` extra VMEM).

    Keeps ``slab_z`` in place on ``side_c0`` and stages permuted ``slab_y`` / ``slab_x`` in the
    dead ``other_c0`` half during ICI. After unpacking into ``side_c0``, swaps ``slab_z`` over
    ``ici[6]`` before ``d2d2_yx`` on ``d2d2[0]`` so mate completes unpack before overwrite.
    Pass ``during_ici_early=True`` for async DMA pre-starts (e.g. RPAd ``bkv``) to span all 3 hops.
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    cx = chip & jnp.int32(1)
    cy = (chip >> jnp.int32(1)) & jnp.int32(1)
    cz = (chip >> jnp.int32(2)) & jnp.int32(1)
    peer_x = me ^ jnp.int32(2)
    peer_y = me ^ jnp.int32(4)
    peer_z = me ^ jnp.int32(8)
    other_c0 = pl.multiple_of(other_side * half_h, 128)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)

    n_tiles = half_h // 128
    t_z = n_tiles // 3
    t_y = n_tiles // 3
    t_x = n_tiles - t_z - t_y
    w_z, w_y, w_x = t_z * 128, t_y * 128, t_x * 128
    off_y_c = w_z
    off_x_c = w_z + w_y

    # Cyclic axis schedules: z-slab X->Y->Z (in-place), y-slab Y->Z->X, x-slab Z->X->Y.
    q_y = (cx << jnp.int32(2)) | (cz << jnp.int32(1)) | cy
    q_x = (cy << jnp.int32(2)) | (cx << jnp.int32(1)) | cz
    qy_r0 = pl.multiple_of(q_y * r_pair, 16)
    qx_r0 = pl.multiple_of(q_x * r_pair, 16)

    slab_z = out_full_ref.at[:, pl.ds(pl.multiple_of(side_c0, 128), w_z)]
    slab_y_dst = out_full_ref.at[:, pl.ds(pl.multiple_of(side_c0 + off_y_c, 128), w_y)]
    slab_x_dst = out_full_ref.at[:, pl.ds(pl.multiple_of(side_c0 + off_x_c, 128), w_x)]
    slab_y = out_full_ref.at[:, pl.ds(pl.multiple_of(other_c0 + off_y_c, 128), w_y)]
    slab_x = out_full_ref.at[:, pl.ds(pl.multiple_of(other_c0 + off_x_c, 128), w_x)]

    bm = _pick_strip(r_pair, 32)
    max_step = 1024 if r_pair <= 16 else 512
    spans_z = _tile_spans(w_z, max_step)
    spans_y = _tile_spans(w_y, max_step)
    spans_x = _tile_spans(w_x, max_step)

    if copy_local_from_pair:
        assert pair_ref is not None
        for r0 in range(0, r_pair, bm):
            rs_l = pl.ds(r0, bm)
            rs_ch = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm)
            rs_qy = pl.ds(pl.multiple_of(qy_r0 + r0, 16), bm)
            rs_qx = pl.ds(pl.multiple_of(qx_r0 + r0, 16), bm)
            for off, cw in spans_z:
                c_l = pl.ds(pl.multiple_of(off, 128), cw)
                slab_z[rs_ch, c_l] = pair_ref[rs_l, pl.ds(pl.multiple_of(side_c0 + off, 128), cw)]
            for off, cw in spans_y:
                c_l = pl.ds(pl.multiple_of(off, 128), cw)
                slab_y[rs_qy, c_l] = pair_ref[
                    rs_l, pl.ds(pl.multiple_of(side_c0 + off_y_c + off, 128), cw)
                ]
            for off, cw in spans_x:
                c_l = pl.ds(pl.multiple_of(off, 128), cw)
                slab_x[rs_qx, c_l] = pair_ref[
                    rs_l, pl.ds(pl.multiple_of(side_c0 + off_x_c + off, 128), cw)
                ]
    else:
        for r0 in range(0, r_pair, bm):
            rs_src = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm)
            rs_qy = pl.ds(pl.multiple_of(qy_r0 + r0, 16), bm)
            rs_qx = pl.ds(pl.multiple_of(qx_r0 + r0, 16), bm)
            for off, cw in spans_y:
                cs = pl.ds(pl.multiple_of(off, 128), cw)
                slab_y[rs_qy, cs] = slab_y_dst[rs_src, cs]
            for off, cw in spans_x:
                cs = pl.ds(pl.multiple_of(off, 128), cw)
                slab_x[rs_qx, cs] = slab_x_dst[rs_src, cs]

    _run_ag_ici_step(
        ((slab_z, chip * r_pair, peer_x), (slab_y, q_y * r_pair, peer_y), (slab_x, q_x * r_pair, peer_z)),
        ici_send,
        ici_recv,
        sem_base=0,
        n_r=r_pair,
        overlap_fn=(during_ici_fn if during_ici_early else None),
    )
    _run_ag_ici_step(
        (
            (slab_z, ((chip >> jnp.int32(1)) << jnp.int32(1)) * r_pair, peer_y),
            (slab_y, ((q_y >> jnp.int32(1)) << jnp.int32(1)) * r_pair, peer_z),
            (slab_x, ((q_x >> jnp.int32(1)) << jnp.int32(1)) * r_pair, peer_x),
        ),
        ici_send,
        ici_recv,
        sem_base=3,
        n_r=2 * r_pair,
    )
    # Step 2 wait clears peers X/Y/Z before Step 3 reuses semaphores 0..2.
    _run_ag_ici_step(
        ((slab_z, cz * (4 * r_pair), peer_z), (slab_y, cx * (4 * r_pair), peer_x), (slab_x, cy * (4 * r_pair), peer_y)),
        ici_send,
        ici_recv,
        sem_base=0,
        n_r=4 * r_pair,
        overlap_fn=(None if during_ici_early else during_ici_fn),
    )

    perm_y = (0, 2, 4, 6, 1, 3, 5, 7)
    perm_x = (0, 4, 1, 5, 2, 6, 3, 7)
    for q in range(8):
        cy_dst = perm_y[q] * r_pair
        cx_dst = perm_x[q] * r_pair
        q_r0 = q * r_pair
        for r0 in range(0, r_pair, bm):
            rs_q = pl.ds(pl.multiple_of(q_r0 + r0, 16), bm)
            rs_yd = pl.ds(pl.multiple_of(cy_dst + r0, 16), bm)
            rs_xd = pl.ds(pl.multiple_of(cx_dst + r0, 16), bm)
            for off, cw in spans_y:
                cs = pl.ds(pl.multiple_of(off, 128), cw)
                slab_y_dst[rs_yd, cs] = slab_y[rs_q, cs]
            for off, cw in spans_x:
                cs = pl.ds(pl.multiple_of(off, 128), cw)
                slab_x_dst[rs_xd, cs] = slab_x[rs_q, cs]

    # Swap slab_z first on ici[6] so wait(op_dz) synchronizes mate before d2d2_yx overwrites other_c0[w_z:].
    op_dz = pltpu.make_async_remote_copy(
        slab_z,
        slab_z,
        ici_send.at[6],
        ici_recv.at[6],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_dz.start()
    op_dz.wait()

    d2_yx = out_full_ref.at[:, pl.ds(pl.multiple_of(side_c0 + off_y_c, 128), w_y + w_x)]
    op_dyx = pltpu.make_async_remote_copy(
        d2_yx,
        d2_yx,
        d2d2_send.at[0],
        d2d2_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_dyx.start()
    if during_d2d2_fn is not None:
        during_d2d2_fn()
    op_dyx.wait()


class _CombinedSemSeq:
    """Concatenate semaphore slices behind ``.shape`` and ``.at[idx]`` without allocation."""

    def __init__(self, sems):
        self._sems = tuple(sems)

    @property
    def shape(self) -> tuple[int]:
        return (len(self._sems),)

    @property
    def at(self):
        sems = self._sems

        class _At:
            def __getitem__(self, idx):
                return sems[idx]

        return _At()


def _tp8_parallel_xy_ag(
    pair_ref,
    out_full_ref,
    ici_send,
    ici_recv,
    d2d2_send,
    d2d2_recv,
    *,
    me,
    mate,
    side,
    other_side,
    chip,
    r_pair: int,
    half_h: int,
    side_c0,
    during_ici_fn,
    during_d2d2_fn,
    copy_local_from_pair: bool,
    during_ici_early: bool = False,
) -> None:
    """In-place zero-permute 2-slab parallel XY doubling all-gather for TP=8.

    Both slabs (`slab_x` via X->Y and `slab_y` via Y->X) write strictly into their own
    canonical row slots on `side_c0`, avoiding both `perm_y` VPU permutes and the serial
    `op_dx` sync barrier: `other_c0` stays untouched until one single `op_d2d2` swaps
    `[B, H/2]` cleanly with `mate`.
    """
    del side, other_side
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    cx = chip & jnp.int32(1)
    cy = (chip >> jnp.int32(1)) & jnp.int32(1)
    peer_x = me ^ jnp.int32(2)
    peer_y = me ^ jnp.int32(4)
    chip_r0 = pl.multiple_of(chip * r_pair, 16)

    n_tiles = half_h // 128
    t_x = n_tiles // 2
    t_y = n_tiles - t_x
    w_x, w_y = t_x * 128, t_y * 128

    slab_x = out_full_ref.at[:, pl.ds(pl.multiple_of(side_c0, 128), w_x)]
    slab_y = out_full_ref.at[:, pl.ds(pl.multiple_of(side_c0 + w_x, 128), w_y)]

    if copy_local_from_pair:
        assert pair_ref is not None
        bm = _pick_strip(r_pair, 32)
        spans_all = _tile_spans(half_h, 1024)
        for r0 in range(0, r_pair, bm):
            rs_l = pl.ds(r0, bm)
            rs_ch = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm)
            for off, cw in spans_all:
                cs = pl.ds(pl.multiple_of(side_c0 + off, 128), cw)
                out_full_ref[rs_ch, cs] = pair_ref[rs_l, cs]

    # Hop 1: slab_x -> peer_x (slot chip_r0) and slab_y -> peer_y (slot chip_r0).
    op1_x = pltpu.make_async_remote_copy(
        slab_x.at[pl.ds(chip_r0, r_pair), :],
        slab_x.at[pl.ds(chip_r0, r_pair), :],
        ici_send.at[0],
        ici_recv.at[0],
        device_id=(peer_x,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op1_y = pltpu.make_async_remote_copy(
        slab_y.at[pl.ds(chip_r0, r_pair), :],
        slab_y.at[pl.ds(chip_r0, r_pair), :],
        ici_send.at[1],
        ici_recv.at[1],
        device_id=(peer_y,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op1_x.start()
    op1_y.start()
    if during_ici_early and during_ici_fn is not None:
        during_ici_fn()
    op1_x.wait()
    op1_y.wait()

    # Hop 2: slab_x sends contiguous 2*r_pair on Y; slab_y sends its two r_pair chunks on X.
    x2_r0 = pl.multiple_of(cy * (2 * r_pair), 16)
    y2_r0a = pl.multiple_of(cx * r_pair, 16)
    y2_r0b = pl.multiple_of((2 + cx) * r_pair, 16)
    op2_x = pltpu.make_async_remote_copy(
        slab_x.at[pl.ds(x2_r0, 2 * r_pair), :],
        slab_x.at[pl.ds(x2_r0, 2 * r_pair), :],
        ici_send.at[2],
        ici_recv.at[2],
        device_id=(peer_y,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op2_ya = pltpu.make_async_remote_copy(
        slab_y.at[pl.ds(y2_r0a, r_pair), :],
        slab_y.at[pl.ds(y2_r0a, r_pair), :],
        ici_send.at[3],
        ici_recv.at[3],
        device_id=(peer_x,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op2_yb = pltpu.make_async_remote_copy(
        slab_y.at[pl.ds(y2_r0b, r_pair), :],
        slab_y.at[pl.ds(y2_r0b, r_pair), :],
        ici_send.at[4],
        ici_recv.at[4],
        device_id=(peer_x,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op2_x.start()
    op2_ya.start()
    op2_yb.start()
    if (not during_ici_early) and during_ici_fn is not None:
        during_ici_fn()
    op2_x.wait()
    op2_ya.wait()
    op2_yb.wait()

    # Single intra-chip D2D swap: `other_c0` was untouched during ICI.
    op_d2d2 = pltpu.make_async_remote_copy(
        out_full_ref.at[:, pl.ds(side_c0, half_h)],
        out_full_ref.at[:, pl.ds(side_c0, half_h)],
        d2d2_send.at[0],
        d2d2_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_d2d2.start()
    if during_d2d2_fn is not None:
        during_d2d2_fn()
    op_d2d2.wait()


def _ag_chip_sharded_half_col_xor1(
    pair_ref,
    out_full_ref,
    ici_send,
    ici_recv,
    d2d2_send,
    d2d2_recv,
    *,
    me,
    world_size: int,
    ag_3d_buf=None,
    during_ici_fn=None,
    during_d2d2_fn=None,
    r_pair_in: int | None = None,
    copy_local_from_pair: bool = True,
    during_ici_early: bool = False,
) -> None:
    """Zero-`d2d1` half-column `XOR=1` All-Gather (`[B//(TP//2), H] -> [B, H] bf16`, `-50%` ICI).

    Because both cores on a chip already hold the identical `r_pair`-row `pair_ref`, `side=0`
    gathers columns `0:H/2` and `side=1` gathers `H/2:H` directly over inter-chip ICI with
    ZERO pre-ICI D2D, finishing with a single `733 GB/s` D2D swap. Uses `ag_3d_buf [n_chips, 8, H/2]`
    when `r_pair == 8` (`TP=8 B=32` / `TP=16 B=64`) to respect TPU7x `(16, 128)` `bf16` VMEM tiling,
    dispatches TP=16 large 2D payloads (`r_pair * half_h >= 65536`) to 3-slab parallel XYZ, and
    dispatches TP=8 ``B>=256, H>=8192`` 2D payloads to 2-slab parallel XY.
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    is_3d_out = len(out_full_ref.shape) == 3
    hidden = out_full_ref.shape[1] * out_full_ref.shape[2] if is_3d_out else out_full_ref.shape[1]
    r_pair = int(r_pair_in) if r_pair_in is not None else pair_ref.shape[0]
    half_h = hidden // 2
    half_tiles = out_full_ref.shape[1] // 2 if is_3d_out else 0
    n_chips = world_size // 2

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    side_c0 = pl.multiple_of(side * half_h, 128)
    side_t0 = pl.multiple_of(side * half_tiles, 16) if is_3d_out else 0
    ag_w = min(half_h, 1024 if r_pair <= 16 else 512)

    if _use_tp16_parallel_xyz_ag(
        world_size=world_size,
        is_3d_out=is_3d_out,
        r_pair=r_pair,
        half_h=half_h,
    ):
        _tp16_parallel_xyz_ag(
            pair_ref,
            out_full_ref,
            ici_send,
            ici_recv,
            d2d2_send,
            d2d2_recv,
            me=me,
            mate=mate,
            side=side,
            other_side=other_side,
            chip=chip,
            r_pair=r_pair,
            half_h=half_h,
            side_c0=side_c0,
            during_ici_fn=during_ici_fn,
            during_d2d2_fn=during_d2d2_fn,
            copy_local_from_pair=copy_local_from_pair,
            during_ici_early=during_ici_early,
        )
        return

    if _use_tp8_parallel_xy_ag(
        world_size=world_size,
        is_3d_out=is_3d_out,
        r_pair=r_pair,
        half_h=half_h,
        n_ici_sems=min(int(ici_send.shape[0]), int(ici_recv.shape[0])),
    ):
        _tp8_parallel_xy_ag(
            pair_ref,
            out_full_ref,
            ici_send,
            ici_recv,
            d2d2_send,
            d2d2_recv,
            me=me,
            mate=mate,
            side=side,
            other_side=other_side,
            chip=chip,
            r_pair=r_pair,
            half_h=half_h,
            side_c0=side_c0,
            during_ici_fn=during_ici_fn,
            during_d2d2_fn=during_d2d2_fn,
            copy_local_from_pair=copy_local_from_pair,
            during_ici_early=during_ici_early,
        )
        return

    if r_pair == 8 and not is_3d_out:
        assert ag_3d_buf is not None and pair_ref is not None
        for off, cw in _tile_spans(half_h, ag_w):
            c_l = pl.ds(pl.multiple_of(off, 128), cw)
            c_g = pl.ds(pl.multiple_of(side_c0 + off, 128), cw)
            ag_3d_buf[chip, :, c_l] = pair_ref[:, c_g]

        ici_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                ag_3d_buf.at[chip],
                ag_3d_buf.at[chip],
                ici_send.at[d - 1],
                ici_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            ici_ops.append(op)
        if during_ici_fn is not None:
            during_ici_fn()
        for op in ici_ops:
            op.wait()

        for p in range(n_chips // 2):
            r16 = pl.ds(16 * p, 16)
            for off, cw in _tile_spans(half_h, ag_w):
                c_l = pl.ds(pl.multiple_of(off, 128), cw)
                c_g = pl.ds(pl.multiple_of(side_c0 + off, 128), cw)
                out_full_ref[r16, c_g] = jnp.concatenate(
                    [ag_3d_buf[2 * p, :, c_l], ag_3d_buf[2 * p + 1, :, c_l]], axis=0
                )
    else:
        chip_r0 = pl.multiple_of(chip * r_pair, 16)
        if copy_local_from_pair:
            assert pair_ref is not None and not is_3d_out
            for off, cw in _tile_spans(half_h, ag_w):
                c_g = pl.ds(pl.multiple_of(side_c0 + off, 128), cw)
                out_full_ref[pl.ds(chip_r0, r_pair), c_g] = pair_ref[:, c_g]

        ici_slice = (
            out_full_ref.at[pl.ds(chip_r0, r_pair), pl.ds(side_t0, half_tiles), :]
            if is_3d_out
            else out_full_ref.at[pl.ds(chip_r0, r_pair), pl.ds(side_c0, half_h)]
        )
        ici_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                ici_slice,
                ici_slice,
                ici_send.at[d - 1],
                ici_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            ici_ops.append(op)
        if during_ici_fn is not None:
            during_ici_fn()
        for op in ici_ops:
            op.wait()

    d2d2_slice = (
        out_full_ref.at[:, pl.ds(side_t0, half_tiles), :]
        if is_3d_out
        else out_full_ref.at[:, pl.ds(side_c0, half_h)]
    )
    op_d2d2 = pltpu.make_async_remote_copy(
        d2d2_slice,
        d2d2_slice,
        d2d2_send.at[0],
        d2d2_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_d2d2.start()
    if during_d2d2_fn is not None:
        during_d2d2_fn()
    op_d2d2.wait()


def _decode_embed_chip_sharded_kernel(
    input_ids_smem,
    embed_hbm,
    l0_gain_ref,
    out_stream_chip_hbm,
    out_attn_in_ref,
    partial_emb_buf,
    slot_2d_buf,
    ici_pair_recv,
    my_pair_bf16,
    mate_pair_bf16,
    s_out_buf,
    attn_pair,
    emb_3d_buf,
    ag_3d_buf,
    row_sems,
    ici_send,
    ici_recv,
    d2d1_send,
    d2d1_recv,
    ag_ici_send,
    ag_ici_recv,
    ag_d2d2_send,
    ag_d2d2_recv,
    s_out_sem,
    *,
    world_size: int,
    embed_eps: float,
    l0_eps: float,
    l0_scale_input: bool,
) -> None:
    """Chip-sharded Entry (`B in {32,64,128,256,384}` on `TP=8`, `0-scf.for` norms + `-50%` ICI AG)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, hidden = partial_emb_buf.shape
    n_chips = world_size // 2
    r_pair = rows // n_chips
    n_strips = hidden // _COL_STRIP

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    chip = me >> jnp.int32(1)

    _signal_barrier(me, world_size)
    _lookup_local_vocab_rows_in_kernel(
        input_ids_smem,
        embed_hbm,
        partial_emb_buf,
        slot_2d_buf,
        row_sems,
        me=me,
        actual_rows=rows,
    )
    _wait_barrier(world_size)

    if r_pair == 8:
        for p in range(n_chips // 2):
            r16 = pl.ds(16 * p, 16)
            for c0 in range(0, hidden, 512):
                c_sl = pl.ds(c0, 512)
                blk16 = partial_emb_buf[r16, c_sl]
                emb_3d_buf[2 * p, :, c_sl] = blk16[0:8, :]
                emb_3d_buf[2 * p + 1, :, c_sl] = blk16[8:16, :]
        c_ici = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                emb_3d_buf.at[chip ^ jnp.int32(d)],
                ici_pair_recv.at[d - 1],
                ici_send.at[d - 1],
                ici_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            c_ici.append(op)
        for op in c_ici:
            op.wait()
        for c0 in range(0, hidden, 512):
            c_sl = pl.ds(c0, 512)
            tot_pair = emb_3d_buf[chip, :, c_sl].astype(jnp.float32)
            for slot in range(n_chips - 1):
                tot_pair = tot_pair + ici_pair_recv[slot, :, c_sl].astype(jnp.float32)
            my_pair_bf16[:, c_sl] = tot_pair.astype(jnp.bfloat16)
    else:
        chip_r0 = pl.multiple_of(chip * r_pair, 16)
        c_ici = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            peer_r0 = pl.multiple_of((chip ^ jnp.int32(d)) * r_pair, 16)
            op = pltpu.make_async_remote_copy(
                partial_emb_buf.at[pl.ds(peer_r0, r_pair), :],
                ici_pair_recv.at[d - 1],
                ici_send.at[d - 1],
                ici_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            c_ici.append(op)
        for op in c_ici:
            op.wait()
        for c0 in range(0, hidden, 512):
            c_sl = pl.ds(c0, 512)
            tot_pair = partial_emb_buf[pl.ds(chip_r0, r_pair), c_sl].astype(jnp.float32)
            for slot in range(n_chips - 1):
                tot_pair = tot_pair + ici_pair_recv[slot, :, c_sl].astype(jnp.float32)
            my_pair_bf16[:, c_sl] = tot_pair.astype(jnp.bfloat16)

    c_d2d = pltpu.make_async_remote_copy(
        my_pair_bf16,
        mate_pair_bf16,
        d2d1_send.at[0],
        d2d1_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_d2d.start()
    c_d2d.wait()

    if r_pair <= 32:
        col_blk = 1024 if r_pair <= 16 else 512
        sq0 = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, hidden, col_blk):
            c_sl = pl.ds(c0, col_blk)
            v = my_pair_bf16[:, c_sl].astype(jnp.float32) + mate_pair_bf16[:, c_sl].astype(jnp.float32)
            s_out_buf[:, c_sl] = v
            sq0 = sq0 + _fold_cols_to_128(v * v)
        inv0 = lax.rsqrt(
            jnp.sum(sq0, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(embed_eps)
        )
        sq1 = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, hidden, col_blk):
            c_sl = pl.ds(c0, col_blk)
            st = (s_out_buf[:, c_sl] * inv0).astype(jnp.bfloat16).astype(jnp.float32)
            s_out_buf[:, c_sl] = st
            lv = st * l0_gain_ref[0:1, c_sl] if l0_scale_input else st
            sq1 = sq1 + _fold_cols_to_128(lv * lv)
        pltpu.make_async_copy(s_out_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()
        inv1 = lax.rsqrt(
            jnp.sum(sq1, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(l0_eps)
        )
        for c0 in range(0, hidden, col_blk):
            c_sl = pl.ds(c0, col_blk)
            st = s_out_buf[:, c_sl]
            g0 = l0_gain_ref[0:1, c_sl]
            a0 = (st * g0) * inv1 if l0_scale_input else (st * inv1) * g0
            attn_pair[:, c_sl] = a0.astype(jnp.bfloat16)
    else:
        bm = _pick_strip(r_pair, 32)
        u_full = _strip_unroll(n_strips, 4)

        def _norms_pair(r0):
            rs = pl.ds(r0, bm)

            def _emb_sq(g, acc):
                base_s = g * u_full
                for sub in range(u_full):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    v = my_pair_bf16[rs, c_sl].astype(jnp.float32) + mate_pair_bf16[rs, c_sl].astype(jnp.float32)
                    s_out_buf[rs, c_sl] = v
                    sq = v * v
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            sq0 = lax.fori_loop(
                0, n_strips // u_full, _emb_sq, jnp.zeros((bm, 128), dtype=jnp.float32)
            )
            inv0 = lax.rsqrt(
                jnp.sum(sq0, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(embed_eps)
            )

            def _l0_sq(g, acc):
                base_s = g * u_full
                for sub in range(u_full):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    st = (s_out_buf[rs, c_sl] * inv0).astype(jnp.bfloat16).astype(jnp.float32)
                    s_out_buf[rs, c_sl] = st
                    lv = st * l0_gain_ref[0:1, c_sl] if l0_scale_input else st
                    sq = lv * lv
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            sq1 = lax.fori_loop(
                0, n_strips // u_full, _l0_sq, jnp.zeros((bm, 128), dtype=jnp.float32)
            )
            inv1 = lax.rsqrt(
                jnp.sum(sq1, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(l0_eps)
            )

            def _l0_out(g, _):
                base_s = g * u_full
                for sub in range(u_full):
                    c_sl = pl.ds(pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    st = s_out_buf[rs, c_sl]
                    g0 = l0_gain_ref[0:1, c_sl]
                    a0 = (st * g0) * inv1 if l0_scale_input else (st * inv1) * g0
                    attn_pair[rs, c_sl] = a0.astype(jnp.bfloat16)
                return jnp.int32(0)

            lax.fori_loop(0, n_strips // u_full, _l0_out, jnp.int32(0))

        _fori_strips(r_pair, bm, _norms_pair)
        pltpu.make_async_copy(s_out_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()

    _ag_chip_sharded_half_col_xor1(
        attn_pair,
        out_attn_in_ref,
        ag_ici_send,
        ag_ici_recv,
        ag_d2d2_send,
        ag_d2d2_recv,
        me=me,
        world_size=world_size,
        ag_3d_buf=ag_3d_buf,
    )
    _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def decode_vocab_embed_chip_sharded_jax(
    input_ids,
    embed_weight_shard,
    l0_in_gain,
    *,
    world_size: int,
    embed_eps: float = 1e-5,
    l0_eps: float = 1e-5,
    l0_scale_input: bool = False,
    collective_id: int = _W8_CID_BASE + 7,
    interpret: bool = False,
):
    """Chip-sharded Entry: ``(input_ids, embed_w, l0_gain) -> (stream_chip_0, attn_in_0)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = input_ids.shape[0]
    v_shard, hidden = embed_weight_shard.shape
    valid_rows = (32, 64, 128, 256, 384) if world_size == 8 else (64, 128, 256, 384)
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or rows not in valid_rows
        or v_shard < 16
        or v_shard % 2 != 0
        or hidden % 1024 != 0
    ):
        raise ValueError("Unsupported decode_vocab_embed_chip_sharded shapes")
    n_chips = world_size // 2
    r_pair = rows // n_chips
    half_h = hidden // 2
    smem, vmem, hbm, sem = pltpu.SMEM, pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    s_sp, v_sp, h_sp = (
        pl.BlockSpec(memory_space=smem),
        pl.BlockSpec(memory_space=vmem),
        pl.BlockSpec(memory_space=hbm),
    )

    return _pallas_call_maybe_interpret(
        functools.partial(
            _decode_embed_chip_sharded_kernel,
            world_size=world_size,
            embed_eps=float(embed_eps),
            l0_eps=float(l0_eps),
            l0_scale_input=bool(l0_scale_input),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        ),
        in_specs=[s_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp),
        scratch_shapes=[
            vmem((rows, hidden), jnp.bfloat16),
            vmem((256, hidden), jnp.bfloat16),
            vmem((n_chips - 1, r_pair, hidden), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.float32),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((n_chips, 8, hidden) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((n_chips, 8, half_h) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((16,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=42 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w9_embed_chip_sh_b{rows}_tp{world_size}",
    )(input_ids.astype(jnp.int32), embed_weight_shard, _as_2d_f32(l0_in_gain, hidden))


def _tp16_parallel_rs_override() -> bool | None:
    from sglang_tpu.envs import envs

    field = envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS
    if field.is_set():
        return bool(field.get())
    return None


def _tp16_parallel_rs_enabled(*, r_pair: int | None = None) -> bool:
    ov = _tp16_parallel_rs_override()
    if ov is not None:
        return ov
    # Default ON for TP=16 r_pair == 8 (B=64: -8.53 us/L on Step(c) SWA_BD4, -6.35 us/L on ChipB1 w/ bc1s)
    # and r_pair >= 16 multiples of 16 (B=128/256/384: -10.8..-34.5 us/L on ChipB1, -12.8..-20.8 us/L on Step(b)).
    return r_pair is not None and (r_pair == 8 or (r_pair >= 16 and r_pair % 16 == 0))


def _bug_c_oneshot_enabled() -> bool:
    from sglang_tpu.envs import envs

    return bool(envs.SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT.get())


def _tile_spans(total: int, max_step: int = 512) -> tuple[tuple[int, int], ...]:
    """Split a 128-aligned width into ``(offset, width)`` spans <= ``max_step``."""
    spans: list[tuple[int, int]] = []
    off = 0
    while off < total:
        step = min(max_step, total - off)
        spans.append((off, step))
        off += step
    return tuple(spans)


def _tp16_parallel_xyz_rs(
    own_half,
    mate_recv,
    send_half,
    rs1_a_or_z_recv,
    rs1_b_or_y_recv,
    rs2_a_or_x_recv,
    my_pair_bf16,
    ag_exp_send,
    ag_exp_recv,
    *,
    cx,
    cy,
    cz,
    peer_x,
    peer_y,
    peer_z,
    r_pair: int,
    half: int,
) -> None:
    """Orthogonal 3-slab parallel XYZ reduce-scatter for ``TP=16`` (``r_pair >= 8``).

    Fuses the Stage-1 ``own_half + mate_recv`` sum directly into three 128-aligned
    column slabs ``(slab_z, slab_y, slab_x)`` with orthogonal chip row permutations:
      - ``slab_z``: identity order -> ``Z(4*r_pair) -> Y(2*r_pair) -> X(r_pair)``
      - ``slab_y``: ``perm_y = (0, 4, 1, 5, 2, 6, 3, 7)`` -> ``Y -> X -> Z``
      - ``slab_x``: ``perm_x = (0, 2, 4, 6, 1, 3, 5, 7)`` -> ``X -> Z -> Y``
    All 3 steps push into step-disjoint ``(4*r_pair/2*r_pair/r_pair, half)`` receive
    buffers. For ``r_pair >= 16``, kept Step-2 rows go directly into ``my_pair_bf16``
    and Step-3 send rows into dead ``src[4*r_pair : 5*r_pair]`` so every strip fits in
    <= 16 VREGs with zero extra VMEM. Step 2 provides a transitive happens-before
    barrier across all three Step-3 peers so Step 3 safely reuses ``[0:3]``.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    two_pair = 2 * r_pair
    four_pair = 4 * r_pair
    rows = 8 * r_pair

    n_tiles = half // 128
    t_z = n_tiles // 3
    t_y = n_tiles // 3
    t_x = n_tiles - t_z - t_y
    w_z, w_y, w_x = t_z * 128, t_y * 128, t_x * 128
    off_y = w_z
    off_x = w_z + w_y

    slab_z = own_half.at[:, pl.ds(0, w_z)]
    slab_y = send_half.at[:, pl.ds(0, w_y)]
    slab_x = send_half.at[:, pl.ds(w_y, w_x)]

    spans_z = _tile_spans(w_z)
    spans_y = _tile_spans(w_y)
    spans_x = _tile_spans(w_x)

    def _run_strips(total_rows: int, bm: int, body_fn) -> None:
        if total_rows // bm <= 4:
            for r0 in range(0, total_rows, bm):
                body_fn(r0)
        else:
            _fori_strips(total_rows, bm, body_fn)

    r1_z = rs1_a_or_z_recv.at[:, pl.ds(0, w_z)]
    r1_y = rs1_a_or_z_recv.at[:, pl.ds(off_y, w_y)]
    r1_x = rs1_a_or_z_recv.at[:, pl.ds(off_x, w_x)]
    r2_z = rs1_b_or_y_recv.at[:, pl.ds(0, w_z)]
    r2_y = rs1_b_or_y_recv.at[:, pl.ds(off_y, w_y)]
    r2_x = rs1_b_or_y_recv.at[:, pl.ds(off_x, w_x)]
    r3_z = rs2_a_or_x_recv.at[:, pl.ds(0, w_z)]
    r3_y = rs2_a_or_x_recv.at[:, pl.ds(off_y, w_y)]
    r3_x = rs2_a_or_x_recv.at[:, pl.ds(off_x, w_x)]

    s1_meta = (
        (slab_z, r1_z, spans_z, cz, peer_z, 0),
        (slab_y, r1_y, spans_y, cy, peer_y, 1),
        (slab_x, r1_x, spans_x, cx, peer_x, 2),
    )
    s2_meta = (
        (slab_z, r2_z, r3_z, spans_z, 0, cy, cx, peer_y, 3),
        (slab_y, r2_y, r3_y, spans_y, off_y, cx, cz, peer_x, 4),
        (slab_x, r2_x, r3_x, spans_x, off_x, cz, cy, peer_z, 5),
    )

    if r_pair == 8:
        for off, cw in spans_z:
            cs = pl.ds(pl.multiple_of(off, 128), cw)
            for p in range(4):
                rs16 = pl.ds(p * 16, 16)
                slab_z[rs16, cs] = (
                    own_half[rs16, cs].astype(jnp.float32) + mate_recv[rs16, cs].astype(jnp.float32)
                ).astype(own_half.dtype)

        for dst_slab, g_off, spans, perm in (
            (slab_y, off_y, spans_y, (0, 4, 1, 5, 2, 6, 3, 7)),
            (slab_x, off_x, spans_x, (0, 2, 4, 6, 1, 3, 5, 7)),
        ):
            for off, cw in spans:
                cg = pl.ds(pl.multiple_of(g_off + off, 128), cw)
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                chips = [None] * 8
                for p in range(4):
                    rs16 = pl.ds(p * 16, 16)
                    v = (
                        own_half[rs16, cg].astype(jnp.float32) + mate_recv[rs16, cg].astype(jnp.float32)
                    ).astype(own_half.dtype)
                    chips[2 * p] = v[0:8, :]
                    chips[2 * p + 1] = v[8:16, :]
                for q in range(4):
                    dst_slab[pl.ds(q * 16, 16), cl] = jnp.concatenate(
                        [chips[perm[2 * q]], chips[perm[2 * q + 1]]], axis=0
                    )

        # Step 1 (4*r_pair rows parallel): Z(slab_z), Y(slab_y), X(slab_x)
        s1_ops = []
        for src, dst, _, bit, peer, sem_idx in s1_meta:
            sr = pl.multiple_of((jnp.int32(1) - bit) * four_pair, 16)
            op = pltpu.make_async_remote_copy(
                src.at[pl.ds(sr, four_pair)],
                dst,
                ag_exp_send.at[sem_idx],
                ag_exp_recv.at[sem_idx],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s1_ops.append(op)
        for op in s1_ops:
            op.wait()
        for src, dst, spans, bit, _, _ in s1_meta:
            kr = pl.multiple_of(bit * four_pair, 16)

            def _s1_reduce(r0, _s=src, _d=dst, _kr=kr, _sp=spans):
                rw = pl.ds(r0, 16)
                rk = pl.ds(pl.multiple_of(_kr + r0, 16), 16)
                for off, cw in _sp:
                    cl = pl.ds(pl.multiple_of(off, 128), cw)
                    _s[rw, cl] = (
                        _s[rk, cl].astype(jnp.float32) + _d[rw, cl].astype(jnp.float32)
                    ).astype(_s.dtype)

            _run_strips(four_pair, 16, _s1_reduce)

        # Step 2 (2*r_pair rows parallel): Y(slab_z), X(slab_y), Z(slab_x)
        s2_ops = []
        for src, dst, _, _, _, bit, _, peer, sem_idx in s2_meta:
            sr = pl.multiple_of((jnp.int32(1) - bit) * two_pair, 16)
            op = pltpu.make_async_remote_copy(
                src.at[pl.ds(sr, two_pair)],
                dst,
                ag_exp_send.at[sem_idx],
                ag_exp_recv.at[sem_idx],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)
        for op in s2_ops:
            op.wait()

        keep_chunks: list[tuple[object, object, object, object]] = []
        for src, dst, r3_src, spans, g_off, bit, next_bit, _, _ in s2_meta:
            kr = pl.multiple_of(bit * 16, 16)
            is_b0 = next_bit == jnp.int32(0)
            for off, cw in spans:
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                cg = pl.ds(pl.multiple_of(g_off + off, 128), cw)
                s2_v = (
                    src[pl.ds(kr, 16), cl].astype(jnp.float32) + dst[:, cl].astype(jnp.float32)
                ).astype(src.dtype)
                src[pl.ds(0, 8), cl] = lax.select(is_b0, s2_v[8:16, :], s2_v[0:8, :])
                keep_chunks.append((r3_src, cg, cl, lax.select(is_b0, s2_v[0:8, :], s2_v[8:16, :])))

        # Step 3 (r_pair rows parallel): X(slab_z), Z(slab_y), Y(slab_x)
        s3_ops = []
        for src, dst, peer, sem_idx in (
            (slab_z, r3_z, peer_x, 0),
            (slab_y, r3_y, peer_z, 1),
            (slab_x, r3_x, peer_y, 2),
        ):
            op = pltpu.make_async_remote_copy(
                src.at[pl.ds(0, r_pair)],
                dst,
                ag_exp_send.at[sem_idx],
                ag_exp_recv.at[sem_idx],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s3_ops.append(op)
        for op in s3_ops:
            op.wait()

        for r3_src, cg, cl, k_v in keep_chunks:
            my_pair_bf16[:, cg] = (
                k_v.astype(jnp.float32) + r3_src[:, cl].astype(jnp.float32)
            ).astype(my_pair_bf16.dtype)
        return

    # Wave-18 Send-First 3-Hop Pipeline Overlap for r_pair >= 16 (0-ULP bit-exact):
    # Compute only the 50% send slice per slab -> launch ICI immediately -> compute the
    # disjoint 50% keep slice overlapped underneath ICI. Waiting all Step-1 ops before
    # Step 2 and all Step-2 ops before Step 3 preserves the transitive barrier for [0:3].
    bm_s1 = 16
    sr1_z = pl.multiple_of((jnp.int32(1) - cz) * four_pair, 16)
    kr1_z = pl.multiple_of(cz * four_pair, 16)
    sr1_y = pl.multiple_of((jnp.int32(1) - cy) * four_pair, 16)
    sr1_x = pl.multiple_of((jnp.int32(1) - cx) * four_pair, 16)

    def _sum_z_half(base_r):
        def _body(r0, _b=base_r):
            rs = pl.ds(pl.multiple_of(_b + r0, 16), bm_s1)
            for off, cw in spans_z:
                cs = pl.ds(pl.multiple_of(off, 128), cw)
                slab_z[rs, cs] = (
                    own_half[rs, cs].astype(jnp.float32) + mate_recv[rs, cs].astype(jnp.float32)
                ).astype(own_half.dtype)

        _run_strips(four_pair, bm_s1, _body)

    def _sum_perm_half(dst_slab, g_off, spans, base_chips, stride_mul, h_bit):
        for k in range(4):
            chip_src = jnp.int32(base_chips[k]) + h_bit * jnp.int32(stride_mul)
            src_base = chip_src * r_pair
            dst_base = (h_bit * jnp.int32(4) + jnp.int32(k)) * r_pair

            def _body(r0, _sb=src_base, _db=dst_base, _ds=dst_slab, _go=g_off, _sp=spans):
                rs_s = pl.ds(pl.multiple_of(_sb + r0, 16), bm_s1)
                rs_d = pl.ds(pl.multiple_of(_db + r0, 16), bm_s1)
                for off, cw in _sp:
                    cg = pl.ds(pl.multiple_of(_go + off, 128), cw)
                    cl = pl.ds(pl.multiple_of(off, 128), cw)
                    _ds[rs_d, cl] = (
                        own_half[rs_s, cg].astype(jnp.float32) + mate_recv[rs_s, cg].astype(jnp.float32)
                    ).astype(own_half.dtype)

            _run_strips(r_pair, bm_s1, _body)

    op1_z = pltpu.make_async_remote_copy(
        slab_z.at[pl.ds(sr1_z, four_pair)], r1_z, ag_exp_send.at[0], ag_exp_recv.at[0],
        device_id=(peer_z,), device_id_type=pl.DeviceIdType.MESH,
    )
    op1_y = pltpu.make_async_remote_copy(
        slab_y.at[pl.ds(sr1_y, four_pair)], r1_y, ag_exp_send.at[1], ag_exp_recv.at[1],
        device_id=(peer_y,), device_id_type=pl.DeviceIdType.MESH,
    )
    op1_x = pltpu.make_async_remote_copy(
        slab_x.at[pl.ds(sr1_x, four_pair)], r1_x, ag_exp_send.at[2], ag_exp_recv.at[2],
        device_id=(peer_x,), device_id_type=pl.DeviceIdType.MESH,
    )

    _sum_z_half(sr1_z)
    op1_z.start()
    _sum_perm_half(slab_y, off_y, spans_y, (0, 4, 1, 5), 2, jnp.int32(1) - cy)
    op1_y.start()
    _sum_perm_half(slab_x, off_x, spans_x, (0, 2, 4, 6), 1, jnp.int32(1) - cx)
    op1_x.start()

    _sum_z_half(kr1_z)
    _sum_perm_half(slab_y, off_y, spans_y, (0, 4, 1, 5), 2, cy)
    _sum_perm_half(slab_x, off_x, spans_x, (0, 2, 4, 6), 1, cx)

    op1_z.wait()
    op1_y.wait()
    op1_x.wait()

    # Step 1 reduce -> Step 2 (2*r_pair rows parallel): send half -> start -> keep half overlapped.
    s2_full = (
        (slab_z, r1_z, r2_z, r3_z, spans_z, 0, cz, cy, cx, peer_y, 3),
        (slab_y, r1_y, r2_y, r3_y, spans_y, off_y, cy, cx, cz, peer_x, 4),
        (slab_x, r1_x, r2_x, r3_x, spans_x, off_x, cx, cz, cy, peer_z, 5),
    )
    s2_ops = []
    for src, r1_d, r2_d, _, spans, _, b1, b2, _, peer2, sem2 in s2_full:
        kr1 = pl.multiple_of(b1 * four_pair, 16)
        sr2 = pl.multiple_of((jnp.int32(1) - b2) * two_pair, 16)

        def _s1_red_send(r0, _s=src, _d=r1_d, _kr1=kr1, _sr2=sr2, _sp=spans):
            rw = pl.ds(pl.multiple_of(_sr2 + r0, 16), 16)
            rk = pl.ds(pl.multiple_of(_kr1 + _sr2 + r0, 16), 16)
            for off, cw in _sp:
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                _s[rw, cl] = (_s[rk, cl].astype(jnp.float32) + _d[rw, cl].astype(jnp.float32)).astype(_s.dtype)

        _run_strips(two_pair, 16, _s1_red_send)
        op2 = pltpu.make_async_remote_copy(
            src.at[pl.ds(sr2, two_pair)], r2_d, ag_exp_send.at[sem2], ag_exp_recv.at[sem2],
            device_id=(peer2,), device_id_type=pl.DeviceIdType.MESH,
        )
        op2.start()
        s2_ops.append(op2)

    for src, r1_d, _, _, spans, _, b1, b2, _, _, _ in s2_full:
        kr1 = pl.multiple_of(b1 * four_pair, 16)
        kr2 = pl.multiple_of(b2 * two_pair, 16)

        def _s1_red_keep(r0, _s=src, _d=r1_d, _kr1=kr1, _kr2=kr2, _sp=spans):
            rw = pl.ds(pl.multiple_of(_kr2 + r0, 16), 16)
            rk = pl.ds(pl.multiple_of(_kr1 + _kr2 + r0, 16), 16)
            for off, cw in _sp:
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                _s[rw, cl] = (_s[rk, cl].astype(jnp.float32) + _d[rw, cl].astype(jnp.float32)).astype(_s.dtype)

        _run_strips(two_pair, 16, _s1_red_keep)

    for op in s2_ops:
        op.wait()

    # Step 2 reduce -> Step 3 (r_pair rows parallel): send into dead [4*r_pair:5*r_pair], keep under ICI.
    s3_peers = (peer_x, peer_z, peer_y)
    s3_ops = []
    for idx, (src, _, r2_d, r3_d, spans, _, _, b2, b3, _, _) in enumerate(s2_full):
        kr2 = pl.multiple_of(b2 * two_pair, 16)
        send_base = lax.select(b3 == jnp.int32(0), jnp.int32(r_pair), jnp.int32(0))

        def _s2_red_send(r0, _s=src, _d=r2_d, _kr2=kr2, _sb=send_base, _sp=spans):
            rs_s = pl.ds(pl.multiple_of(_kr2 + _sb + r0, 16), 16)
            rs_d = pl.ds(pl.multiple_of(_sb + r0, 16), 16)
            rs_out = pl.ds(pl.multiple_of(four_pair + r0, 16), 16)
            for off, cw in _sp:
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                _s[rs_out, cl] = (_s[rs_s, cl].astype(jnp.float32) + _d[rs_d, cl].astype(jnp.float32)).astype(_s.dtype)

        _run_strips(r_pair, 16, _s2_red_send)
        op3 = pltpu.make_async_remote_copy(
            src.at[pl.ds(four_pair, r_pair)], r3_d, ag_exp_send.at[idx], ag_exp_recv.at[idx],
            device_id=(s3_peers[idx],), device_id_type=pl.DeviceIdType.MESH,
        )
        op3.start()
        s3_ops.append(op3)

    for src, _, r2_d, _, spans, g_off, _, b2, b3, _, _ in s2_full:
        kr2 = pl.multiple_of(b2 * two_pair, 16)
        keep_base = lax.select(b3 == jnp.int32(0), jnp.int32(0), jnp.int32(r_pair))

        def _s2_red_keep(r0, _s=src, _d=r2_d, _kr2=kr2, _kb=keep_base, _go=g_off, _sp=spans):
            rl = pl.ds(r0, 16)
            rs_s = pl.ds(pl.multiple_of(_kr2 + _kb + r0, 16), 16)
            rs_d = pl.ds(pl.multiple_of(_kb + r0, 16), 16)
            for off, cw in _sp:
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                cg = pl.ds(pl.multiple_of(_go + off, 128), cw)
                my_pair_bf16[rl, cg] = (
                    _s[rs_s, cl].astype(jnp.float32) + _d[rs_d, cl].astype(jnp.float32)
                ).astype(my_pair_bf16.dtype)

        _run_strips(r_pair, 16, _s2_red_keep)

    for op in s3_ops:
        op.wait()

    for r3_src, g_off, spans in (
        (r3_z, 0, spans_z),
        (r3_y, off_y, spans_y),
        (r3_x, off_x, spans_x),
    ):

        def _s3_finish(r0, _r3=r3_src, _go=g_off, _sp=spans):
            rl = pl.ds(r0, 16)
            for off, cw in _sp:
                cl = pl.ds(pl.multiple_of(off, 128), cw)
                cg = pl.ds(pl.multiple_of(_go + off, 128), cw)
                my_pair_bf16[rl, cg] = (
                    my_pair_bf16[rl, cg].astype(jnp.float32) + _r3[rl, cl].astype(jnp.float32)
                ).astype(my_pair_bf16.dtype)

        _run_strips(r_pair, 16, _s3_finish)


def _run_chip_b1_vmem_core(
    stream_chip_hbm,
    gated_ref,
    wo_hbm,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_pair_hbm,
    w_pre_pair_scale_ref,
    w_router_hbm,
    pre_exp_gain_ref,
    out_stream_chip_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
    wo_buf,
    w_pre_pair_buf,
    w_router_buf,
    s_buf,
    send_half,
    mate_recv,
    own_half,
    rs1_a_or_z_recv,
    rs1_b_or_y_recv,
    rs2_a_or_x_recv,
    rs2_b_recv,
    my_pair_bf16,
    recv_pair_bf16,
    sq_pair_pa_buf,
    ffn_pair_buf,
    router_pair_buf,
    pre_my_half,
    pre_mate_half,
    sq_pair_pre_buf,
    exp_3d_buf,
    wo_sem,
    w_pre_sem,
    w_router_sem,
    s_in_sem,
    s_out_sem,
    c1a_send,
    c1a_recv,
    rs1_send,
    rs1_recv,
    rs2_send,
    rs2_recv,
    rs3_send,
    rs3_recv,
    c1c_send,
    c1c_recv,
    pre_send,
    pre_recv,
    ag_exp_send,
    ag_exp_recv,
    ag_exp_d2d_send,
    ag_exp_d2d_recv,
    ag_log_send,
    ag_log_recv,
    *,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    skip_barrier_signal: bool = False,
    skip_barrier_wait: bool = False,
    skip_stream_load: bool = False,
    skip_stream_store: bool = False,
    early_router_and_log_ag: bool = False,
    during_ag_exp_ici_fn=None,
    during_ag_exp_d2d2_fn=None,
    out_expert_3d_ref=None,
    parallel_rs: bool | None = None,
    bug_c_oneshot: bool | None = None,
) -> None:
    """Reusable VMEM-resident Wave-9 chip-sharded Bridge 1b core (`standalone` & `b1+MoE(+b2)`).

    Gain vectors (`pa_alpha`, `pa_beta`, `pre_ff_gain`, `pre_exp_gain`) are `(1, cols) f32` via `_as_2d_f32`;
    dot scales (`wo_scale`, `w_pre_pair_scale`) follow `_as_dot_scale_f32`. Passing `out_expert_3d_ref`
    (`(B, H_PRE // 128, 128)`) writes Stage-4 pre-expert activations directly in MoE 3D layout (`out_expert_in_ref`
    may be `None`), eliminating the 2D->3D VPU relayout and `expert_in_vmem` buffer.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    r_pair = s_buf.shape[0]
    two_pair = 2 * r_pair
    four_pair = 4 * r_pair
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    q_cols = half // 2
    pre_pair_cols = w_pre_pair_hbm.shape[1]
    h_pre = 2 * pre_pair_cols
    half_tiles = pre_pair_cols // 128
    if out_expert_3d_ref is not None and (half_tiles % 16 != 0 or pre_pair_cols % 128 != 0):
        raise ValueError(
            f"out_expert_3d_ref requires pre_pair_cols multiple of 2048 (16*128), got {pre_pair_cols}"
        )
    exp_dtype = out_expert_3d_ref.dtype if out_expert_3d_ref is not None else out_expert_in_ref.dtype
    n_exp = w_router_hbm.shape[1]
    n_chips = world_size // 2
    r_align = 16 if (r_pair % 16 == 0) else 8
    two_align = 32 if (two_pair % 32 == 0) else 16

    bm_full = _pick_strip(rows, 32)
    bm_pair = _pick_strip(r_pair, 32)
    c_blk_half = 512 if (half % 512 == 0) else _COL_STRIP
    c_blk_full = 512 if (hidden % 512 == 0) else _COL_STRIP
    c_blk_pre = 512 if (pre_pair_cols % 512 == 0) else _COL_STRIP
    q_step = 512 if (q_cols % 512 == 0) else _COL_STRIP
    is_r16 = w_router_buf.dtype == jnp.bfloat16

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    cx = chip & jnp.int32(1)
    cy = (chip >> jnp.int32(1)) & jnp.int32(1)
    cz = (chip >> jnp.int32(2)) & jnp.int32(1)
    peer_x = me ^ jnp.int32(2)
    peer_y = me ^ jnp.int32(4)
    peer_z = me ^ jnp.int32(8)

    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)
    chip_r0 = pl.multiple_of(chip * r_pair, r_align)
    own_pre_c0 = pl.multiple_of(side * pre_pair_cols, 128)
    other_pre_c0 = pl.multiple_of(other_side * pre_pair_cols, 128)
    side_t0 = pl.multiple_of(side * half_tiles, 16)

    if not skip_barrier_signal:
        _signal_barrier(me, world_size)
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
    ).start()
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
    ).start()
    if not skip_stream_load:
        pltpu.make_async_copy(stream_chip_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_pre_pair_hbm, w_pre_pair_buf, w_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[0], wo_scale_ref, other_c0, send_half, 0,
        rows=rows, n_cols=half,
    )

    if not skip_barrier_wait:
        _wait_barrier(world_size)
    op_c1a = pltpu.make_async_remote_copy(
        send_half, mate_recv, c1a_send.at[0], c1a_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1a.start()

    _dma_wait(wo_buf.at[1], wo_sem.at[1])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[1], wo_scale_ref, own_c0, own_half, 0,
        rows=rows, n_cols=half,
    )
    op_c1a.wait()

    use_prs = (
        world_size == 16
        and (r_pair == 8 or (r_pair >= 16 and r_pair % 16 == 0))
        and half >= 384
        and send_half is not own_half
        and send_half is not mate_recv
        and (_tp16_parallel_rs_enabled(r_pair=r_pair) if parallel_rs is None else bool(parallel_rs))
    )

    if not use_prs:
        if rows <= 128:
            for r0 in range(0, rows, 32):
                rs = pl.ds(r0, 32)
                for c0 in range(0, half, 512):
                    c_sl = pl.ds(c0, 512)
                    v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                    own_half[rs, c_sl] = v.astype(own_half.dtype)
        else:

            def _sum_c1a(r0):
                rs = pl.ds(r0, bm_full)
                for c0 in range(0, half, c_blk_half):
                    c_sl = pl.ds(c0, c_blk_half)
                    v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                    own_half[rs, c_sl] = v.astype(own_half.dtype)

            _fori_strips(rows, bm_full, _sum_c1a)

    # Stage 2: Race-free Inter-Chip Token Reduce-Scatter own_half [rows, half] -> my_pair_bf16 [r_pair, half].
    if use_prs:
        _tp16_parallel_xyz_rs(
            own_half,
            mate_recv,
            send_half,
            rs1_a_or_z_recv,
            rs1_b_or_y_recv,
            rs2_a_or_x_recv,
            my_pair_bf16,
            ag_exp_send,
            ag_exp_recv,
            cx=cx,
            cy=cy,
            cz=cz,
            peer_x=peer_x,
            peer_y=peer_y,
            peer_z=peer_z,
            r_pair=r_pair,
            half=half,
        )
    elif world_size == 8 and r_pair >= 16:
        a_send_r1 = pl.multiple_of((jnp.int32(1) - cy) * two_pair, 32)
        a_keep_r1 = pl.multiple_of(cy * two_pair, 32)
        b_send_r1_0 = pl.multiple_of((jnp.int32(1) - cx) * r_pair, 16)
        b_send_r1_1 = pl.multiple_of(two_pair + (jnp.int32(1) - cx) * r_pair, 16)
        b_keep_r1_0 = pl.multiple_of(cx * r_pair, 16)
        b_keep_r1_1 = pl.multiple_of(two_pair + cx * r_pair, 16)

        op_rs1_a = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(a_send_r1, two_pair), pl.ds(0, q_cols)],
            rs1_a_or_z_recv,
            rs1_send.at[0],
            rs1_recv.at[0],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs1_b0 = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(b_send_r1_0, r_pair), pl.ds(q_cols, q_cols)],
            rs1_b_or_y_recv.at[pl.ds(0, r_pair), :],
            rs1_send.at[1],
            rs1_recv.at[1],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs1_b1 = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(b_send_r1_1, r_pair), pl.ds(q_cols, q_cols)],
            rs1_b_or_y_recv.at[pl.ds(r_pair, r_pair), :],
            rs1_send.at[2],
            rs1_recv.at[2],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs1_a.start()
        op_rs1_b0.start()
        op_rs1_b1.start()
        op_rs1_a.wait()
        op_rs1_b0.wait()
        op_rs1_b1.wait()

        for sub in range(2):
            r_loc = sub * r_pair
            ra = pl.ds(pl.multiple_of(a_keep_r1 + r_loc, 16), r_pair)
            rb = pl.ds(b_keep_r1_0 if sub == 0 else b_keep_r1_1, r_pair)
            rl = pl.ds(r_loc, r_pair)
            for off in range(0, q_cols, q_step):
                ca = pl.ds(off, q_step)
                cb = pl.ds(q_cols + off, q_step)
                va = own_half[ra, ca].astype(jnp.float32) + rs1_a_or_z_recv[rl, ca].astype(jnp.float32)
                vb = own_half[rb, cb].astype(jnp.float32) + rs1_b_or_y_recv[rl, ca].astype(jnp.float32)
                own_half[ra, ca] = va.astype(own_half.dtype)
                own_half[rb, cb] = vb.astype(own_half.dtype)

        a_send_r2 = pl.multiple_of(a_keep_r1 + (jnp.int32(1) - cx) * r_pair, 16)
        b_send_r2 = pl.multiple_of((jnp.int32(1) - cy) * two_pair + cx * r_pair, 16)

        op_rs2_a = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(a_send_r2, r_pair), pl.ds(0, q_cols)],
            rs2_a_or_x_recv,
            rs2_send.at[0],
            rs2_recv.at[0],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs2_b = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(b_send_r2, r_pair), pl.ds(q_cols, q_cols)],
            rs2_b_recv,
            rs2_send.at[1],
            rs2_recv.at[1],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs2_a.start()
        op_rs2_b.start()
        op_rs2_a.wait()
        op_rs2_b.wait()

        rc_chip = pl.ds(pl.multiple_of(chip * r_pair, 16), r_pair)
        for off in range(0, q_cols, q_step):
            ca = pl.ds(off, q_step)
            cb = pl.ds(q_cols + off, q_step)
            my_pair_bf16[:, ca] = (
                own_half[rc_chip, ca].astype(jnp.float32) + rs2_a_or_x_recv[:, ca].astype(jnp.float32)
            ).astype(jnp.bfloat16)
            my_pair_bf16[:, cb] = (
                own_half[rc_chip, cb].astype(jnp.float32) + rs2_b_recv[:, ca].astype(jnp.float32)
            ).astype(jnp.bfloat16)
    else:
        if world_size == 16:
            z_send_r = pl.multiple_of((jnp.int32(1) - cz) * four_pair, 16)
            z_keep_r = pl.multiple_of(cz * four_pair, 16)
            op_rs_z = pltpu.make_async_remote_copy(
                own_half.at[pl.ds(z_send_r, four_pair), :],
                rs1_a_or_z_recv,
                rs1_send.at[0],
                rs1_recv.at[0],
                device_id=(peer_z,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_rs_z.start()
            op_rs_z.wait()
            bm_z = _pick_strip(four_pair, 32)
            if four_pair <= 64:
                for r0 in range(0, four_pair, bm_z):
                    rk = pl.ds(pl.multiple_of(z_keep_r + r0, 16), bm_z)
                    rl = pl.ds(r0, bm_z)
                    for c0 in range(0, half, 512):
                        cs = pl.ds(c0, 512)
                        v = own_half[rk, cs].astype(jnp.float32) + rs1_a_or_z_recv[rl, cs].astype(jnp.float32)
                        own_half[rl, cs] = v.astype(own_half.dtype)
            else:

                def _sum_z(r0):
                    rk = pl.ds(pl.multiple_of(z_keep_r + r0, 16), bm_z)
                    rl = pl.ds(r0, bm_z)
                    for c0 in range(0, half, c_blk_half):
                        cs = pl.ds(c0, c_blk_half)
                        v = own_half[rk, cs].astype(jnp.float32) + rs1_a_or_z_recv[rl, cs].astype(jnp.float32)
                        own_half[rl, cs] = v.astype(own_half.dtype)

                _fori_strips(four_pair, bm_z, _sum_z)

        y_send_r = pl.multiple_of((jnp.int32(1) - cy) * two_pair, 16)
        y_keep_r = pl.multiple_of(cy * two_pair, 16)
        op_rs_y = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(y_send_r, two_pair), :],
            rs1_b_or_y_recv,
            rs2_send.at[0],
            rs2_recv.at[0],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs_y.start()
        op_rs_y.wait()
        bm_y = _pick_strip(two_pair, 32)
        for r0 in range(0, two_pair, bm_y):
            rk = pl.ds(pl.multiple_of(y_keep_r + r0, 16), bm_y)
            rl = pl.ds(r0, bm_y)
            for c0 in range(0, half, 512):
                cs = pl.ds(c0, 512)
                v = own_half[rk, cs].astype(jnp.float32) + rs1_b_or_y_recv[rl, cs].astype(jnp.float32)
                own_half[rl, cs] = v.astype(own_half.dtype)

        c_step_x = min(half, 1024 if r_pair <= 16 else 512)
        if r_pair == 8:
            is_cx0 = cx == jnp.int32(0)
            for c0 in range(0, half, c_step_x):
                cs = pl.ds(c0, c_step_x)
                v16 = own_half[pl.ds(0, 16), cs]
                my_pair_bf16[:, cs] = lax.select(is_cx0, v16[0:8, :], v16[8:16, :])
                rs2_b_recv[:, cs] = lax.select(is_cx0, v16[8:16, :], v16[0:8, :])
            op_rs_x = pltpu.make_async_remote_copy(
                rs2_b_recv,
                rs2_a_or_x_recv,
                rs3_send.at[0],
                rs3_recv.at[0],
                device_id=(peer_x,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_rs_x.start()
            op_rs_x.wait()
            for c0 in range(0, half, c_step_x):
                cs = pl.ds(c0, c_step_x)
                my_pair_bf16[:, cs] = (
                    my_pair_bf16[:, cs].astype(jnp.float32) + rs2_a_or_x_recv[:, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)
        else:
            x_send_r = pl.multiple_of((jnp.int32(1) - cx) * r_pair, 16)
            x_keep_r = pl.multiple_of(cx * r_pair, 16)
            op_rs_x = pltpu.make_async_remote_copy(
                own_half.at[pl.ds(x_send_r, r_pair), :],
                rs2_a_or_x_recv,
                rs3_send.at[0],
                rs3_recv.at[0],
                device_id=(peer_x,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_rs_x.start()
            op_rs_x.wait()
            rk_x = pl.ds(x_keep_r, r_pair)
            for c0 in range(0, half, c_step_x):
                cs = pl.ds(c0, c_step_x)
                my_pair_bf16[:, cs] = (
                    own_half[rk_x, cs].astype(jnp.float32) + rs2_a_or_x_recv[:, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)

    # Stage 3: Single `[r_pair, 4096]` intra-chip D2D swap overlapped with own-half `pa_sq`!
    op_c1c = pltpu.make_async_remote_copy(
        my_pair_bf16,
        recv_pair_bf16,
        c1c_send.at[0],
        c1c_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1c.start()

    if r_pair == 16:
        c_blk = min(half, 1024)
        own_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        own_chunks = []
        for c0 in range(0, half, c_blk):
            bf = my_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            own_chunks.append(bf)
            own_sq = own_sq + _fold_cols_to_128(bf * bf)
        op_c1c.wait()
        if not skip_stream_load:
            _dma_wait(s_buf, s_in_sem.at[0])
        oth_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        oth_chunks = []
        for c0 in range(0, half, c_blk):
            bf = recv_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            oth_chunks.append(bf)
            oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
        pa_sq = own_sq + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )
        own_pff_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        oth_pff_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        ns_own_list = []
        ns_oth_list = []
        for idx_c, c0 in enumerate(range(0, half, c_blk)):
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            ns_own = (
                pa_alpha_ref[0:1, c_own] * s_buf[:, c_own]
                + pa_beta_ref[0:1, c_own] * (own_chunks[idx_c] * inv_pa)
            )
            ns_oth = (
                pa_alpha_ref[0:1, c_oth] * s_buf[:, c_oth]
                + pa_beta_ref[0:1, c_oth] * (oth_chunks[idx_c] * inv_pa)
            )
            s_buf[:, c_own] = ns_own
            s_buf[:, c_oth] = ns_oth
            ns_own_list.append(ns_own)
            ns_oth_list.append(ns_oth)
            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            own_pff_sq = own_pff_sq + _fold_cols_to_128(v_own * v_own)
            oth_pff_sq = oth_pff_sq + _fold_cols_to_128(v_oth * v_oth)
        pff_sq = own_pff_sq + oth_pff_sq
        if not skip_stream_store:
            pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        for idx_c, c0 in enumerate(range(0, half, c_blk)):
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            go = pre_ff_gain_ref[0:1, c_own]
            gt = pre_ff_gain_ref[0:1, c_oth]
            ro = (
                (ns_own_list[idx_c] * go) * inv_pff
                if pre_ff_scale_input
                else (ns_own_list[idx_c] * inv_pff) * go
            )
            rt = (
                (ns_oth_list[idx_c] * gt) * inv_pff
                if pre_ff_scale_input
                else (ns_oth_list[idx_c] * inv_pff) * gt
            )
            if not is_r16:
                router_pair_buf[:, c_own] = ro
                router_pair_buf[:, c_oth] = rt
            ffn_pair_buf[:, c_own] = ro.astype(jnp.bfloat16)
            ffn_pair_buf[:, c_oth] = rt.astype(jnp.bfloat16)
    elif r_pair <= 32:
        c_blk = min(half, 1024 if r_pair <= 16 else 512)
        h_blk = min(hidden, 1024 if r_pair <= 16 else 512)
        own_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = my_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            own_sq = own_sq + _fold_cols_to_128(bf * bf)
        op_c1c.wait()
        if not skip_stream_load:
            _dma_wait(s_buf, s_in_sem.at[0])
        oth_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = recv_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
        pa_sq = own_sq + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )
        own_pff_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        oth_pff_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            c_l = pl.ds(c0, c_blk)
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            ns_own = (
                pa_alpha_ref[0:1, c_own] * s_buf[:, c_own]
                + pa_beta_ref[0:1, c_own] * (my_pair_bf16[:, c_l].astype(jnp.float32) * inv_pa)
            )
            ns_oth = (
                pa_alpha_ref[0:1, c_oth] * s_buf[:, c_oth]
                + pa_beta_ref[0:1, c_oth] * (recv_pair_bf16[:, c_l].astype(jnp.float32) * inv_pa)
            )
            s_buf[:, c_own] = ns_own
            s_buf[:, c_oth] = ns_oth
            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            own_pff_sq = own_pff_sq + _fold_cols_to_128(v_own * v_own)
            oth_pff_sq = oth_pff_sq + _fold_cols_to_128(v_oth * v_oth)
        pff_sq = own_pff_sq + oth_pff_sq
        if not skip_stream_store:
            pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        for c0 in range(0, hidden, h_blk):
            c_g = pl.ds(c0, h_blk)
            sv = s_buf[:, c_g]
            gn = pre_ff_gain_ref[0:1, c_g]
            rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
            if not is_r16:
                router_pair_buf[:, c_g] = rn
            ffn_pair_buf[:, c_g] = rn.astype(jnp.bfloat16)
    else:

        def _own_pa_sq(r0):
            rs = pl.ds(r0, bm_pair)
            acc = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            for c0 in range(0, half, c_blk_half):
                bf = my_pair_bf16[rs, pl.ds(c0, c_blk_half)].astype(jnp.float32)
                acc = acc + _fold_cols_to_128(bf * bf)
            sq_pair_pa_buf[rs, :] = acc

        _fori_strips(r_pair, bm_pair, _own_pa_sq)
        op_c1c.wait()
        if not skip_stream_load:
            _dma_wait(s_buf, s_in_sem.at[0])

        def _finish_pa_and_pre_ffn_pair(r0):
            rs = pl.ds(r0, bm_pair)
            oth_sq = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            for c0 in range(0, half, c_blk_half):
                bf = recv_pair_bf16[rs, pl.ds(c0, c_blk_half)].astype(jnp.float32)
                oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
            inv_pa = lax.rsqrt(
                jnp.sum(sq_pair_pa_buf[rs, :] + oth_sq, axis=-1, keepdims=True)
                * jnp.float32(1.0 / hidden)
                + jnp.float32(pa_eps)
            )
            own_pff_sq = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            oth_pff_sq = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            for c0 in range(0, half, c_blk_half):
                c_l = pl.ds(c0, c_blk_half)
                c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk_half)
                c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk_half)
                ns_own = (
                    pa_alpha_ref[0:1, c_own] * s_buf[rs, c_own]
                    + pa_beta_ref[0:1, c_own]
                    * (my_pair_bf16[rs, c_l].astype(jnp.float32) * inv_pa)
                )
                ns_oth = (
                    pa_alpha_ref[0:1, c_oth] * s_buf[rs, c_oth]
                    + pa_beta_ref[0:1, c_oth]
                    * (recv_pair_bf16[rs, c_l].astype(jnp.float32) * inv_pa)
                )
                s_buf[rs, c_own] = ns_own
                s_buf[rs, c_oth] = ns_oth
                v_own = (
                    ns_own * pre_ff_gain_ref[0:1, c_own]
                    if pre_ff_scale_input
                    else ns_own
                )
                v_oth = (
                    ns_oth * pre_ff_gain_ref[0:1, c_oth]
                    if pre_ff_scale_input
                    else ns_oth
                )
                own_pff_sq = own_pff_sq + _fold_cols_to_128(v_own * v_own)
                oth_pff_sq = oth_pff_sq + _fold_cols_to_128(v_oth * v_oth)
            pff_sq = own_pff_sq + oth_pff_sq
            inv_pff = lax.rsqrt(
                jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(pre_ff_eps)
            )
            for c0 in range(0, hidden, c_blk_full):
                c_g = pl.ds(c0, c_blk_full)
                sv = s_buf[rs, c_g]
                gn = pre_ff_gain_ref[0:1, c_g]
                rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
                if not is_r16:
                    router_pair_buf[rs, c_g] = rn
                ffn_pair_buf[rs, c_g] = rn.astype(jnp.bfloat16)

        _fori_strips(r_pair, bm_pair, _finish_pa_and_pre_ffn_pair)
        if not skip_stream_store:
            pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()

    # Stage 4: Direct W_pre_pair MXU `[r_pair, 8192] @ [8192, 2048]` (ZERO `op_pair`/`op_rpair`!).
    rmul = jnp.float32(router_multiplier)
    r_lhs = ffn_pair_buf if is_r16 else router_pair_buf
    ag_log_ops = []

    def _run_router_and_start_log_ag():
        _dma_wait(w_router_buf, w_router_sem.at[0])
        if r_pair <= 64 and n_exp == 256:
            lg = jnp.dot(r_lhs[...], w_router_buf[...], preferred_element_type=jnp.float32)
            out_router_logits_ref[pl.ds(chip_r0, r_pair), :] = lg * rmul
        else:
            for ce in range(0, n_exp, _DOT_TN):
                c_sl = pl.ds(ce, _DOT_TN)
                lg = jnp.dot(r_lhs[...], w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
                out_router_logits_ref[pl.ds(chip_r0, r_pair), c_sl] = lg * rmul
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op_l = pltpu.make_async_remote_copy(
                out_router_logits_ref.at[pl.ds(chip_r0, r_pair), :],
                out_router_logits_ref.at[pl.ds(chip_r0, r_pair), :],
                ag_log_send.at[d - 1],
                ag_log_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_l.start()
            ag_log_ops.append(op_l)

    def _wait_log_ag():
        for op in ag_log_ops:
            op.wait()

    run_early_router = early_router_and_log_ag or use_prs
    if run_early_router:
        _run_router_and_start_log_ag()

    _dma_wait(w_pre_pair_buf, w_pre_sem.at[0])
    _dot_fp8_w8_slice_to_ref(
        ffn_pair_buf, 0, w_pre_pair_buf, w_pre_pair_scale_ref, 0, pre_my_half, 0, 0,
        n_r=r_pair, n_cols=pre_pair_cols, align=r_align,
    )

    use_bugc_1shot = (
        r_pair == 8
        and (_bug_c_oneshot_enabled() if bug_c_oneshot is None else bool(bug_c_oneshot))
    )

    if r_pair == 8 and use_bugc_1shot:
        # Bug C + 1-shot half-col ag_exp (TP=8 B=32 & TP=16 B=64): swap 4 KiB sq_pair_pre_buf
        # <-> sq_pair_pa_buf (idle for r_pair=8), normalize own half (-50% VPU), gather 1-shot
        # half-cols over ICI (-50% ICI bytes), unpack pairs, then overlap d2d2 with during_d2d2_fn.
        p_blk = min(pre_pair_cols, 1024)
        own_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        pre_f_chunks = []
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            xf = pre_my_half[:, c_l].astype(jnp.float32)
            pre_f_chunks.append(xf)
            xf_s = xf * pre_exp_gain_ref[0:1, c_own] if pre_exp_scale_input else xf
            own_pre_sq = own_pre_sq + _fold_cols_to_128(xf_s * xf_s)
        sq_pair_pre_buf[...] = own_pre_sq
        op_sq = pltpu.make_async_remote_copy(
            sq_pair_pre_buf,
            sq_pair_pa_buf,
            pre_send.at[0],
            pre_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_sq.start()
        if not run_early_router:
            _run_router_and_start_log_ag()
        op_sq.wait()
        pre_sq = own_pre_sq + sq_pair_pa_buf[...]
        inv = lax.rsqrt(
            jnp.sum(pre_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
            + jnp.float32(pre_exp_eps)
        )
        for idx_p, c0 in enumerate(range(0, pre_pair_cols, p_blk)):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            x_own = pre_f_chunks[idx_p]
            g_own = pre_exp_gain_ref[0:1, c_own]
            y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
            exp_3d_buf[chip, :, c_l] = y_own.astype(exp_dtype)

        if run_early_router:
            _wait_log_ag()

        half_exp_slice = exp_3d_buf.at[chip, :, pl.ds(0, pre_pair_cols)]
        ag_exp_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op_e = pltpu.make_async_remote_copy(
                half_exp_slice,
                half_exp_slice,
                ag_exp_send.at[d - 1],
                ag_exp_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_e.start()
            ag_exp_ops.append(op_e)
        if not run_early_router:
            _wait_log_ag()
        if during_ag_exp_ici_fn is not None:
            during_ag_exp_ici_fn()
        for op in ag_exp_ops:
            op.wait()

        ag_exp_dst = out_expert_3d_ref if out_expert_3d_ref is not None else out_expert_in_ref
        if out_expert_3d_ref is not None:
            for p in range(n_chips // 2):
                r16 = pl.ds(16 * p, 16)
                v16 = jnp.concatenate(
                    [
                        exp_3d_buf[2 * p, :, pl.ds(0, pre_pair_cols)],
                        exp_3d_buf[2 * p + 1, :, pl.ds(0, pre_pair_cols)],
                    ],
                    axis=0,
                )
                out_expert_3d_ref[r16, pl.ds(side_t0, half_tiles), :] = v16.reshape(
                    16, half_tiles, 128
                )
        else:
            e_w = min(pre_pair_cols, 1024)
            for p in range(n_chips // 2):
                r16 = pl.ds(16 * p, 16)
                for c0 in range(0, pre_pair_cols, e_w):
                    c_l = pl.ds(c0, e_w)
                    c_g = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), e_w)
                    out_expert_in_ref[r16, c_g] = jnp.concatenate(
                        [exp_3d_buf[2 * p, :, c_l], exp_3d_buf[2 * p + 1, :, c_l]],
                        axis=0,
                    )
        d2d2_slice = (
            ag_exp_dst.at[:, pl.ds(side_t0, half_tiles), :]
            if out_expert_3d_ref is not None
            else ag_exp_dst.at[:, pl.ds(own_pre_c0, pre_pair_cols)]
        )
        op_d2d2 = pltpu.make_async_remote_copy(
            d2d2_slice,
            d2d2_slice,
            ag_exp_d2d_send.at[0],
            ag_exp_d2d_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_d2d2.start()
        if during_ag_exp_d2d2_fn is not None:
            during_ag_exp_d2d2_fn()
        op_d2d2.wait()
    elif r_pair == 8:
        op_pre = pltpu.make_async_remote_copy(
            pre_my_half, pre_mate_half, pre_send.at[0], pre_recv.at[0],
            device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
        )
        op_pre.start()

        if not run_early_router:
            _run_router_and_start_log_ag()

        p_blk = min(pre_pair_cols, 1024)
        own_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            xf = pre_my_half[:, c_l].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_own]
            own_pre_sq = own_pre_sq + _fold_cols_to_128(xf * xf)
        op_pre.wait()
        oth_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_oth = pl.ds(pl.multiple_of(other_pre_c0 + c0, 128), p_blk)
            xf = pre_mate_half[:, c_l].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_oth]
            oth_pre_sq = oth_pre_sq + _fold_cols_to_128(xf * xf)
        pre_sq = own_pre_sq + oth_pre_sq
        inv = lax.rsqrt(
            jnp.sum(pre_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
            + jnp.float32(pre_exp_eps)
        )
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            c_oth = pl.ds(pl.multiple_of(other_pre_c0 + c0, 128), p_blk)
            x_own = pre_my_half[:, c_l].astype(jnp.float32)
            x_oth = pre_mate_half[:, c_l].astype(jnp.float32)
            g_own = pre_exp_gain_ref[0:1, c_own]
            g_oth = pre_exp_gain_ref[0:1, c_oth]
            y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
            y_oth = (x_oth * g_oth) * inv if pre_exp_scale_input else (x_oth * inv) * g_oth
            exp_3d_buf[chip, :, c_own] = y_own.astype(exp_dtype)
            exp_3d_buf[chip, :, c_oth] = y_oth.astype(exp_dtype)

        if run_early_router:
            _wait_log_ag()

        ag_exp_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op_e = pltpu.make_async_remote_copy(
                exp_3d_buf.at[chip],
                exp_3d_buf.at[chip],
                ag_exp_send.at[d - 1],
                ag_exp_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_e.start()
            ag_exp_ops.append(op_e)
        if not run_early_router:
            _wait_log_ag()
        if during_ag_exp_ici_fn is not None:
            during_ag_exp_ici_fn()
        if during_ag_exp_d2d2_fn is not None:
            during_ag_exp_d2d2_fn()
        for op in ag_exp_ops:
            op.wait()
        if out_expert_3d_ref is not None:
            for p in range(n_chips // 2):
                r16 = pl.ds(16 * p, 16)
                for h_i in range(2):
                    h_c0 = pl.multiple_of(h_i * pre_pair_cols, 128)
                    h_t0 = pl.multiple_of(h_i * half_tiles, 16)
                    v16 = jnp.concatenate(
                        [
                            exp_3d_buf[2 * p, :, pl.ds(h_c0, pre_pair_cols)],
                            exp_3d_buf[2 * p + 1, :, pl.ds(h_c0, pre_pair_cols)],
                        ],
                        axis=0,
                    )
                    out_expert_3d_ref[r16, pl.ds(h_t0, half_tiles), :] = v16.reshape(
                        16, half_tiles, 128
                    )
        else:
            e_w = min(h_pre, 1024)
            for p in range(n_chips // 2):
                r16 = pl.ds(16 * p, 16)
                for c0 in range(0, h_pre, e_w):
                    c_sl = pl.ds(c0, e_w)
                    out_expert_in_ref[r16, c_sl] = jnp.concatenate(
                        [exp_3d_buf[2 * p, :, c_sl], exp_3d_buf[2 * p + 1, :, c_sl]], axis=0
                    )
    else:
        # Bug C Fix (r_pair >= 16): swap 8..48 KiB `sq_pair_pre_buf [r_pair, 128]`, normalize ONLY
        # own half (`-50%` VPU), and launch half-col `ag_exp` ICI overlapped under `W_router`!
        rs_g = pl.ds(chip_r0, r_pair)
        ag_exp_dst = out_expert_3d_ref if out_expert_3d_ref is not None else out_expert_in_ref
        if r_pair <= 32:
            p_blk = min(pre_pair_cols, 1024 if r_pair <= 16 else 512)
            own_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
            pre_f_chunks = []
            for c0 in range(0, pre_pair_cols, p_blk):
                c_l = pl.ds(c0, p_blk)
                c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
                xf = pre_my_half[:, c_l].astype(jnp.float32)
                if r_pair == 16:
                    pre_f_chunks.append(xf)
                xf_s = xf * pre_exp_gain_ref[0:1, c_own] if pre_exp_scale_input else xf
                own_pre_sq = own_pre_sq + _fold_cols_to_128(xf_s * xf_s)
            sq_pair_pre_buf[...] = own_pre_sq
            op_sq = pltpu.make_async_remote_copy(
                sq_pair_pre_buf, pre_mate_half, pre_send.at[0], pre_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            op_sq.start()
            op_sq.wait()
            pre_sq = own_pre_sq + pre_mate_half[...]
            inv = lax.rsqrt(
                jnp.sum(pre_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
                + jnp.float32(pre_exp_eps)
            )
            y_chunks_3d = []
            for idx_p, c0 in enumerate(range(0, pre_pair_cols, p_blk)):
                c_l = pl.ds(c0, p_blk)
                c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
                x_own = pre_f_chunks[idx_p] if r_pair == 16 else pre_my_half[:, c_l].astype(jnp.float32)
                g_own = pre_exp_gain_ref[0:1, c_own]
                y_own = ((x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own).astype(exp_dtype)
                if out_expert_3d_ref is not None:
                    y_chunks_3d.append(y_own)
                else:
                    out_expert_in_ref[rs_g, c_own] = y_own
            if out_expert_3d_ref is not None:
                out_expert_3d_ref[rs_g, pl.ds(side_t0, half_tiles), :] = jnp.concatenate(
                    y_chunks_3d, axis=1
                ).reshape(r_pair, half_tiles, 128)
        else:

            def _own_pre_sq(r0):
                rs = pl.ds(r0, bm_pair)
                acc = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
                for c0 in range(0, pre_pair_cols, c_blk_pre):
                    c_l = pl.ds(c0, c_blk_pre)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), c_blk_pre)
                    xf = pre_my_half[rs, c_l].astype(jnp.float32)
                    if pre_exp_scale_input:
                        xf = xf * pre_exp_gain_ref[0:1, c_own]
                    acc = acc + _fold_cols_to_128(xf * xf)
                sq_pair_pre_buf[rs, :] = acc

            _fori_strips(r_pair, bm_pair, _own_pre_sq)
            op_sq = pltpu.make_async_remote_copy(
                sq_pair_pre_buf, pre_mate_half, pre_send.at[0], pre_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            op_sq.start()
            op_sq.wait()

            def _own_pre_exp_norm(r0):
                rs_l = pl.ds(r0, bm_pair)
                rs_go = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm_pair)
                inv = lax.rsqrt(
                    jnp.sum(sq_pair_pre_buf[rs_l, :] + pre_mate_half[rs_l, :], axis=-1, keepdims=True)
                    * jnp.float32(1.0 / h_pre)
                    + jnp.float32(pre_exp_eps)
                )
                y_chunks_3d = []
                for c0 in range(0, pre_pair_cols, c_blk_pre):
                    c_l = pl.ds(c0, c_blk_pre)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), c_blk_pre)
                    x_own = pre_my_half[rs_l, c_l].astype(jnp.float32)
                    g_own = pre_exp_gain_ref[0:1, c_own]
                    y_own = ((x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own).astype(exp_dtype)
                    if out_expert_3d_ref is not None:
                        y_chunks_3d.append(y_own)
                    else:
                        out_expert_in_ref[rs_go, c_own] = y_own
                if out_expert_3d_ref is not None:
                    out_expert_3d_ref[rs_go, pl.ds(side_t0, half_tiles), :] = jnp.concatenate(
                        y_chunks_3d, axis=1
                    ).reshape(bm_pair, half_tiles, 128)

            _fori_strips(r_pair, bm_pair, _own_pre_exp_norm)

        if run_early_router:
            _wait_log_ag()

        _ag_chip_sharded_half_col_xor1(
            None,
            ag_exp_dst,
            ag_exp_send,
            ag_exp_recv,
            ag_exp_d2d_send,
            ag_exp_d2d_recv,
            me=me,
            world_size=world_size,
            during_ici_fn=(
                during_ag_exp_ici_fn if run_early_router else _run_router_and_start_log_ag
            ),
            during_d2d2_fn=during_ag_exp_d2d2_fn if run_early_router else _wait_log_ag,
            r_pair_in=r_pair,
            copy_local_from_pair=False,
        )
    if not skip_stream_store:
        _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def _attn_to_moe_chip_sharded_bridge_kernel(
    *args,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    parallel_rs: bool | None = None,
    bug_c_oneshot: bool | None = None,
) -> None:
    """Wave-9 Chip-Sharded Bridge 1b (`TP=8` & `TP=16`): 0 `op_pair`/`op_rpair`, 0 concat, 0 `scf.for`!"""
    _run_chip_b1_vmem_core(
        *args,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        parallel_rs=parallel_rs,
        bug_c_oneshot=bug_c_oneshot,
    )


def attn_to_moe_chip_sharded_bridge_jax(
    stream_chip,
    gated_attn,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_pair_w8k,
    w_pre_pair_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int = 8,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    parallel_rs: bool | None = None,
    bug_c_oneshot: bool | None = None,
    collective_id: int = _W8_CID_BASE + 8,
    interpret: bool = False,
):
    """Chip-sharded Bridge 1b (`stream_chip [B//(TP//2), 8192] -> (new_stream_chip, logits, expert_in)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Chip-sharded Bridge 1b requires world_size in (8, 16), got {world_size}")
    rows, o_in = gated_attn.shape
    r_pair, hidden = stream_chip.shape
    n_chips = world_size // 2
    two_pair = 2 * r_pair
    _, pre_pair_cols = w_pre_pair_w8k.shape
    h_pre = 2 * pre_pair_cols
    _, n_exp = w_router_km.shape
    valid_rows = (32, 64, 128, 256, 384) if world_size == 8 else (64, 128, 256, 384)
    if (
        rows not in valid_rows
        or rows != n_chips * r_pair
        or r_pair < 8
        or r_pair % 8 != 0
        or hidden % 1024 != 0
    ):
        raise ValueError(
            f"Unsupported chip-sharded Bridge 1b shapes world_size={world_size}, "
            f"stream={stream_chip.shape}, gated={gated_attn.shape}"
        )
    half = hidden // 2
    q_cols = half // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    use_xy = world_size == 8 and r_pair >= 16
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _attn_to_moe_chip_sharded_bridge_kernel,
            world_size=world_size,
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            parallel_rs=parallel_rs,
            bug_c_oneshot=bug_c_oneshot,
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
            jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((hidden, pre_pair_cols), w_pre_pair_w8k.dtype),
            vmem((hidden, n_exp), w_router_km.dtype),
            vmem((r_pair, hidden), jnp.float32),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem(
                (two_pair, q_cols)
                if use_xy
                else ((4 * r_pair, half) if world_size == 16 else (8, 128)),
                jnp.bfloat16,
            ),
            vmem((two_pair, q_cols) if use_xy else (two_pair, half), jnp.bfloat16),
            vmem((r_pair, q_cols) if use_xy else (r_pair, half), jnp.bfloat16),
            vmem(
                (r_pair, q_cols)
                if use_xy
                else ((r_pair, half) if r_pair == 8 else (8, 128)),
                jnp.bfloat16,
            ),
            vmem((r_pair, half), jnp.bfloat16),
            vmem((r_pair, half), jnp.bfloat16),
            vmem((r_pair, 128), jnp.float32),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((8, 128) if is_r16 else (r_pair, hidden), jnp.float32),
            vmem((r_pair, pre_pair_cols), jnp.bfloat16),
            (
                vmem((r_pair, pre_pair_cols), jnp.bfloat16)
                if r_pair == 8
                else vmem((r_pair, 128), jnp.float32)
            ),
            vmem((r_pair, 128), jnp.float32),
            vmem((n_chips, 8, h_pre) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((3 if use_xy else 1,)),
            sem((3 if use_xy else 1,)),
            sem((2 if use_xy else 1,)),
            sem((2 if use_xy else 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(60 if rows > 256 else 58) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w9_chip_b1_b{rows}_tp{world_size}_{'r16' if is_r16 else 'r32'}",
    )(
        stream_chip.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_dot_scale_f32(wo_scale, hidden),
        _as_2d_f32(pa_alpha, hidden),
        _as_2d_f32(pa_beta, hidden),
        _as_2d_f32(pre_ff_gain, hidden),
        w_pre_pair_w8k,
        _as_dot_scale_f32(w_pre_pair_scale, pre_pair_cols),
        w_router_km,
        _as_2d_f32(pre_exp_gain, h_pre),
    )


def _run_chip_sharded_post_fast_core(
    moe_out_ref,
    w_post_buf,
    s_post_ref,
    s_buf,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_in_pair,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_pair_full,
    cols_3d_buf,
    w_post_sem,
    s_in_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    *,
    me,
    world_size: int,
    sharded_moe_in: bool,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    round_stream_bf16_for_norm: bool,
    on_stream_updated_fn=None,
    tok_d2d2_send=None,
    tok_d2d2_recv=None,
    out_next_full_ref=None,
    skip_barrier_wait: bool = False,
    skip_stream_load: bool = False,
) -> None:
    """Wave-9 chip-sharded `W_post` + `0-scf.for` epilogue (`TP=8` & `TP=16`, sharded or full `moe_out`)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_pair, hidden = s_buf.shape
    half_h = hidden // 2
    n_chips = world_size // 2
    rows = n_chips * r_pair
    k_in = moe_out_ref.shape[1]
    post_cols = w_post_buf.shape[1]
    pair_cols = 2 * post_cols
    is_tp2_wpost = pair_cols == hidden
    r_align = 16 if (r_pair % 16 == 0) else 8

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * r_pair, r_align)
    is_even = side == jnp.int32(0)

    if not skip_barrier_wait:
        _wait_barrier(world_size)
    if is_tp2_wpost:
        # Opt-in TP_chip=2 W_post [k_in, H/2]: eliminates BOTH tok_s2 and post_ici inter-chip ICI!
        k_w = min(k_in, 1024 if r_pair <= 16 else 512)
        if sharded_moe_in:
            ts1 = pltpu.make_async_remote_copy(
                moe_out_ref, mate_tok_buf, tok_s1_send.at[0], tok_s1_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            ts1.start()
            ts1.wait()
            for c0 in range(0, k_in, k_w):
                c_sl = pl.ds(c0, k_w)
                own_v = moe_out_ref[:, c_sl]
                mate_v = mate_tok_buf[:, c_sl]
                moe_full_buf[:, c_sl] = jnp.concatenate(
                    [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)], axis=0
                )
            _dma_wait(w_post_buf, w_post_sem.at[0])
            _dot_fp8_scaled_to_ref(
                moe_full_buf, w_post_buf, s_post_ref, 0, chip_cols_buf, 0,
                rows=r_pair, n_cols=half_h,
            )
        elif r_pair == 8:
            cx = chip & jnp.int32(1)
            r16_c = pl.multiple_of((chip >> jnp.int32(1)) * 16, 16)
            for c0 in range(0, k_in, k_w):
                c_sl = pl.ds(c0, k_w)
                blk16 = moe_out_ref[pl.ds(r16_c, 16), c_sl]
                moe_full_buf[:, c_sl] = lax.select(cx == jnp.int32(0), blk16[0:8, :], blk16[8:16, :])
            _dma_wait(w_post_buf, w_post_sem.at[0])
            _dot_fp8_scaled_to_ref(
                moe_full_buf, w_post_buf, s_post_ref, 0, chip_cols_buf, 0,
                rows=8, n_cols=half_h,
            )
        else:
            _dma_wait(w_post_buf, w_post_sem.at[0])
            _dot_fp8_w8_slice_to_ref(
                moe_out_ref, chip_r0, w_post_buf, s_post_ref, 0, chip_cols_buf, 0, 0,
                n_r=r_pair, n_cols=half_h, align=16,
            )

        op_pd2d = pltpu.make_async_remote_copy(
            chip_cols_buf, mate_post_shard, post_d2d_send.at[0], post_d2d_recv.at[0],
            device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
        )
        op_pd2d.start()

        if r_pair == 8 or (world_size == 8 and r_pair == 32):
            # Wave-18/19 Zero-post_pair_full + op_pd2d overlap (TP=8 B=32/128, TP=16 B=64):
            # hides op_pd2d D2D underneath own_b_sq, skips post_pair_full write + 2x reads,
            # keeps VREG-cached chunks for r_pair==8 and spill-free own/oth passes for r_pair==32,
            # keeps TP=8 B=64 (r_pair==16) on the spill-free balanced path, and preserves 0.0
            # chipmate bit-exactness via commutative own_sq + oth_sq sums.
            own_c0 = pl.multiple_of(side * half_h, 128)
            other_c0 = pl.multiple_of(other_side * half_h, 128)
            half_spans = _tile_spans(half_h, 1024 if r_pair == 8 else 512)
            if r_pair == 8:
                own_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                own_bc, own_bg = [], []
                for off, cw in half_spans:
                    c_l = pl.ds(pl.multiple_of(off, 128), cw)
                    c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                    bc = chip_cols_buf[:, c_l].astype(jnp.float32)
                    bg = post_ff_gain_ref[0:1, c_own]
                    own_bc.append(bc)
                    own_bg.append(bg)
                    bcs = bc * bg if post_ff_scale_input else bc
                    own_b_sq = own_b_sq + _fold_cols_to_128(bcs * bcs)
                op_pd2d.wait()
                if not skip_stream_load:
                    _dma_wait(s_buf, s_in_sem.at[0])
                oth_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                oth_bc, oth_bg = [], []
                for off, cw in half_spans:
                    c_l = pl.ds(pl.multiple_of(off, 128), cw)
                    c_oth = pl.ds(pl.multiple_of(other_c0 + off, 128), cw)
                    bc = mate_post_shard[:, c_l].astype(jnp.float32)
                    bg = post_ff_gain_ref[0:1, c_oth]
                    oth_bc.append(bc)
                    oth_bg.append(bg)
                    bcs = bc * bg if post_ff_scale_input else bc
                    oth_b_sq = oth_b_sq + _fold_cols_to_128(bcs * bcs)
                inv_b = lax.rsqrt(
                    jnp.sum(own_b_sq + oth_b_sq, axis=-1, keepdims=True)
                    * jnp.float32(1.0 / hidden)
                    + jnp.float32(post_ff_eps)
                )
                own_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                ns_own_list = []
                for idx_c, (off, cw) in enumerate(half_spans):
                    c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                    bc, bg = own_bc[idx_c], own_bg[idx_c]
                    bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                    ns = pf_alpha_ref[0:1, c_own] * s_buf[:, c_own] + pf_beta_ref[0:1, c_own] * bn
                    if round_stream_bf16_for_norm:
                        ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                    s_buf[:, c_own] = ns
                    ns_own_list.append(ns)
                    nv = ns * next_in_gain_ref[0:1, c_own] if next_in_scale_input else ns
                    own_n_sq = own_n_sq + _fold_cols_to_128(nv * nv)

                oth_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                ns_oth_list = []
                for idx_c, (off, cw) in enumerate(half_spans):
                    c_oth = pl.ds(pl.multiple_of(other_c0 + off, 128), cw)
                    bc, bg = oth_bc[idx_c], oth_bg[idx_c]
                    bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                    ns = pf_alpha_ref[0:1, c_oth] * s_buf[:, c_oth] + pf_beta_ref[0:1, c_oth] * bn
                    if round_stream_bf16_for_norm:
                        ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                    s_buf[:, c_oth] = ns
                    if out_next_full_ref is None or r_pair == 8:
                        ns_oth_list.append(ns)
                    nv = ns * next_in_gain_ref[0:1, c_oth] if next_in_scale_input else ns
                    oth_n_sq = oth_n_sq + _fold_cols_to_128(nv * nv)

                if on_stream_updated_fn is not None:
                    on_stream_updated_fn()
                inv_n = lax.rsqrt(
                    jnp.sum(own_n_sq + oth_n_sq, axis=-1, keepdims=True)
                    * jnp.float32(1.0 / hidden)
                    + jnp.float32(next_in_eps)
                )
                if out_next_full_ref is not None and r_pair >= 16:
                    for idx_c, (off, cw) in enumerate(half_spans):
                        c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                        ng = next_in_gain_ref[0:1, c_own]
                        nn = (
                            (ns_own_list[idx_c] * ng) * inv_n
                            if next_in_scale_input
                            else (ns_own_list[idx_c] * inv_n) * ng
                        )
                        out_next_full_ref[pl.ds(chip_r0, r_pair), c_own] = nn.astype(jnp.bfloat16)
                else:
                    for idx_c, (off, cw) in enumerate(half_spans):
                        c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                        ng = next_in_gain_ref[0:1, c_own]
                        nn = (
                            (ns_own_list[idx_c] * ng) * inv_n
                            if next_in_scale_input
                            else (ns_own_list[idx_c] * inv_n) * ng
                        )
                        next_in_pair[:, c_own] = nn.astype(jnp.bfloat16)
                    for idx_c, (off, cw) in enumerate(half_spans):
                        c_oth = pl.ds(pl.multiple_of(other_c0 + off, 128), cw)
                        ng = next_in_gain_ref[0:1, c_oth]
                        nn = (
                            (ns_oth_list[idx_c] * ng) * inv_n
                            if next_in_scale_input
                            else (ns_oth_list[idx_c] * inv_n) * ng
                        )
                        next_in_pair[:, c_oth] = nn.astype(jnp.bfloat16)
            else:
                # r_pair == 32 (TP=8 B=128): separate own/oth passes prevent simultaneous VREG spilling.
                own_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                for off, cw in half_spans:
                    c_l = pl.ds(pl.multiple_of(off, 128), cw)
                    c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                    bc = chip_cols_buf[:, c_l].astype(jnp.float32)
                    if post_ff_scale_input:
                        bc = bc * post_ff_gain_ref[0:1, c_own]
                    own_b_sq = own_b_sq + _fold_cols_to_128(bc * bc)
                op_pd2d.wait()
                if not skip_stream_load:
                    _dma_wait(s_buf, s_in_sem.at[0])
                oth_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                for off, cw in half_spans:
                    c_l = pl.ds(pl.multiple_of(off, 128), cw)
                    c_oth = pl.ds(pl.multiple_of(other_c0 + off, 128), cw)
                    bc = mate_post_shard[:, c_l].astype(jnp.float32)
                    if post_ff_scale_input:
                        bc = bc * post_ff_gain_ref[0:1, c_oth]
                    oth_b_sq = oth_b_sq + _fold_cols_to_128(bc * bc)
                inv_b = lax.rsqrt(
                    jnp.sum(own_b_sq + oth_b_sq, axis=-1, keepdims=True)
                    * jnp.float32(1.0 / hidden)
                    + jnp.float32(post_ff_eps)
                )
                own_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                for off, cw in half_spans:
                    c_l = pl.ds(pl.multiple_of(off, 128), cw)
                    c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                    bc = chip_cols_buf[:, c_l].astype(jnp.float32)
                    bg = post_ff_gain_ref[0:1, c_own]
                    bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                    ns = pf_alpha_ref[0:1, c_own] * s_buf[:, c_own] + pf_beta_ref[0:1, c_own] * bn
                    if round_stream_bf16_for_norm:
                        ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                    s_buf[:, c_own] = ns
                    nv = ns * next_in_gain_ref[0:1, c_own] if next_in_scale_input else ns
                    own_n_sq = own_n_sq + _fold_cols_to_128(nv * nv)

                oth_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
                for off, cw in half_spans:
                    c_l = pl.ds(pl.multiple_of(off, 128), cw)
                    c_oth = pl.ds(pl.multiple_of(other_c0 + off, 128), cw)
                    bc = mate_post_shard[:, c_l].astype(jnp.float32)
                    bg = post_ff_gain_ref[0:1, c_oth]
                    bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                    ns = pf_alpha_ref[0:1, c_oth] * s_buf[:, c_oth] + pf_beta_ref[0:1, c_oth] * bn
                    if round_stream_bf16_for_norm:
                        ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                    s_buf[:, c_oth] = ns
                    nv = ns * next_in_gain_ref[0:1, c_oth] if next_in_scale_input else ns
                    oth_n_sq = oth_n_sq + _fold_cols_to_128(nv * nv)

                if on_stream_updated_fn is not None:
                    on_stream_updated_fn()
                inv_n = lax.rsqrt(
                    jnp.sum(own_n_sq + oth_n_sq, axis=-1, keepdims=True)
                    * jnp.float32(1.0 / hidden)
                    + jnp.float32(next_in_eps)
                )
                if out_next_full_ref is not None:
                    for off, cw in half_spans:
                        c_own = pl.ds(pl.multiple_of(own_c0 + off, 128), cw)
                        sv = s_buf[:, c_own]
                        ng = next_in_gain_ref[0:1, c_own]
                        nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                        out_next_full_ref[pl.ds(chip_r0, r_pair), c_own] = nn.astype(jnp.bfloat16)
                else:
                    for off, cw in _tile_spans(hidden, 512):
                        c_sl = pl.ds(pl.multiple_of(off, 128), cw)
                        sv = s_buf[:, c_sl]
                        ng = next_in_gain_ref[0:1, c_sl]
                        nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                        next_in_pair[:, c_sl] = nn.astype(jnp.bfloat16)
            return

        if not skip_stream_load:
            _dma_wait(s_buf, s_in_sem.at[0])
        op_pd2d.wait()

        p_w = min(half_h, 1024 if r_pair <= 16 else 512)
        for c0 in range(0, half_h, p_w):
            c_l = pl.ds(c0, p_w)
            my_c = chip_cols_buf[:, c_l]
            mate_c = mate_post_shard[:, c_l]
            post_pair_full[:, pl.ds(c0, p_w)] = lax.select(is_even, my_c, mate_c)
            post_pair_full[:, pl.ds(half_h + c0, p_w)] = lax.select(is_even, mate_c, my_c)
    else:
        chip_c0 = pl.multiple_of(chip * pair_cols, 128)
        if sharded_moe_in:
            r_tok = moe_out_ref.shape[0]
            half_k = k_in // 2
            if (
                r_pair >= 16
                and half_k % 128 == 0
                and tok_d2d2_send is not None
                and tok_d2d2_recv is not None
            ):
                # Bug A Fix: race-free half-column `ts1` (`other_k0` push) + `-50%` `tok_s2` ICI!
                side_k0 = pl.multiple_of(side * half_k, 128)
                other_k0 = pl.multiple_of(other_side * half_k, 128)
                k_w = min(half_k, 1024 if r_tok <= 16 else 512)
                ts1 = pltpu.make_async_remote_copy(
                    moe_out_ref.at[:, pl.ds(other_k0, half_k)],
                    mate_tok_buf.at[:, pl.ds(other_k0, half_k)],
                    tok_s1_send.at[0],
                    tok_s1_recv.at[0],
                    device_id=(mate,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                ts1.start()
                ts1.wait()
                for c0 in range(0, half_k, k_w):
                    c_g = pl.ds(pl.multiple_of(side_k0 + c0, 128), k_w)
                    own_v = moe_out_ref[:, c_g]
                    mate_v = mate_tok_buf[:, c_g]
                    moe_full_buf[pl.ds(chip_r0, r_pair), c_g] = jnp.concatenate(
                        [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)],
                        axis=0,
                    )

                def _wait_s_and_wpost():
                    if not skip_stream_load:
                        _dma_wait(s_buf, s_in_sem.at[0])
                    _dma_wait(w_post_buf, w_post_sem.at[0])

                _ag_chip_sharded_half_col_xor1(
                    None,
                    moe_full_buf,
                    tok_s2_send,
                    tok_s2_recv,
                    tok_d2d2_send,
                    tok_d2d2_recv,
                    me=me,
                    world_size=world_size,
                    during_ici_fn=_wait_s_and_wpost,
                    r_pair_in=r_pair,
                    copy_local_from_pair=False,
                )
                _dot_fp8_scaled_to_ref(
                    moe_full_buf, w_post_buf, s_post_ref, 0, post_shard, 0,
                    rows=rows, n_cols=post_cols,
                )
            else:
                ts1 = pltpu.make_async_remote_copy(
                    moe_out_ref, mate_tok_buf, tok_s1_send.at[0], tok_s1_recv.at[0],
                    device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
                )
                ts1.start()
                ts1.wait()

                k_w = min(k_in, 1024 if r_tok <= 16 else 512)
                for c0 in range(0, k_in, k_w):
                    c_sl = pl.ds(c0, k_w)
                    own_v = moe_out_ref[:, c_sl]
                    mate_v = mate_tok_buf[:, c_sl]
                    moe_full_buf[pl.ds(chip_r0, r_pair), c_sl] = jnp.concatenate(
                        [lax.select(is_even, own_v, mate_v), lax.select(is_even, mate_v, own_v)],
                        axis=0,
                    )

                ts2_ops = []
                for d in range(1, n_chips):
                    peer = me ^ jnp.int32(d << 1)
                    op = pltpu.make_async_remote_copy(
                        moe_full_buf.at[pl.ds(chip_r0, r_pair), :],
                        moe_full_buf.at[pl.ds(chip_r0, r_pair), :],
                        tok_s2_send.at[d - 1],
                        tok_s2_recv.at[d - 1],
                        device_id=(peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    op.start()
                    ts2_ops.append(op)

                if not skip_stream_load:
                    _dma_wait(s_buf, s_in_sem.at[0])
                _dma_wait(w_post_buf, w_post_sem.at[0])
                for op in ts2_ops:
                    op.wait()
                _dot_fp8_scaled_to_ref(
                    moe_full_buf, w_post_buf, s_post_ref, 0, post_shard, 0,
                    rows=rows, n_cols=post_cols,
                )
        else:
            if not skip_stream_load:
                _dma_wait(s_buf, s_in_sem.at[0])
            _dma_wait(w_post_buf, w_post_sem.at[0])
            _dot_fp8_scaled_to_ref(
                moe_out_ref, w_post_buf, s_post_ref, 0, post_shard, 0,
                rows=rows, n_cols=post_cols,
            )

        use_compact_post_ag = (
            r_pair >= 16
            and mate_post_shard.shape == (r_pair, half_h)
            and chip_cols_buf.shape == (r_pair, half_h)
        )
        if not use_compact_post_ag:
            op_pd2d = pltpu.make_async_remote_copy(
                post_shard, mate_post_shard, post_d2d_send.at[0], post_d2d_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            op_pd2d.start()
            op_pd2d.wait()

            cs_w = min(post_cols, 512)
            for c0 in range(0, post_cols, cs_w):
                c_sl = pl.ds(c0, cs_w)
                s_v = post_shard[:, c_sl]
                m_v = mate_post_shard[:, c_sl]
                chip_cols_buf[:, pl.ds(c0, cs_w)] = lax.select(is_even, s_v, m_v)
                chip_cols_buf[:, pl.ds(post_cols + c0, cs_w)] = lax.select(is_even, m_v, s_v)

            pair_w = min(pair_cols, 1024 if r_pair <= 16 else 512)
            if r_pair == 8:
                for p in range(n_chips // 2):
                    r16 = pl.ds(16 * p, 16)
                    for c0 in range(0, pair_cols, pair_w):
                        c_sl = pl.ds(c0, pair_w)
                        blk16 = chip_cols_buf[r16, c_sl]
                        cols_3d_buf[2 * p, :, c_sl] = blk16[0:8, :]
                        cols_3d_buf[2 * p + 1, :, c_sl] = blk16[8:16, :]
                for c0 in range(0, pair_cols, pair_w):
                    c_sl = pl.ds(c0, pair_w)
                    post_pair_full[:, pl.ds(pl.multiple_of(chip_c0 + c0, 128), pair_w)] = cols_3d_buf[
                        chip, :, c_sl
                    ]
                pici_ops = []
                for d in range(1, n_chips):
                    dst_chip = chip ^ jnp.int32(d)
                    peer = me ^ jnp.int32(d << 1)
                    op = pltpu.make_async_remote_copy(
                        cols_3d_buf.at[dst_chip],
                        post_pair_full.at[:, pl.ds(chip_c0, pair_cols)],
                        post_ici_send.at[d - 1],
                        post_ici_recv.at[d - 1],
                        device_id=(peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    op.start()
                    pici_ops.append(op)
                for op in pici_ops:
                    op.wait()
            else:
                for c0 in range(0, pair_cols, pair_w):
                    c_sl = pl.ds(c0, pair_w)
                    post_pair_full[:, pl.ds(pl.multiple_of(chip_c0 + c0, 128), pair_w)] = chip_cols_buf[
                        pl.ds(chip_r0, r_pair), c_sl
                    ]
                pici_ops = []
                for d in range(1, n_chips):
                    dst_chip = chip ^ jnp.int32(d)
                    dst_r0 = pl.multiple_of(dst_chip * r_pair, 16)
                    peer = me ^ jnp.int32(d << 1)
                    op = pltpu.make_async_remote_copy(
                        chip_cols_buf.at[pl.ds(dst_r0, r_pair), :],
                        post_pair_full.at[:, pl.ds(chip_c0, pair_cols)],
                        post_ici_send.at[d - 1],
                        post_ici_recv.at[d - 1],
                        device_id=(peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    op.start()
                    pici_ops.append(op)
                for op in pici_ops:
                    op.wait()
        else:
            # Bug B Fix (r_pair >= 16): `-50%` `post_ici` direct all-to-all into `[r_pair, half_h]`
            # + single `[r_pair, half_h]` intra-chip D2D swap (eliminates `[rows, 2*post_cols]` select)!
            my_c_loc = pl.multiple_of(chip * post_cols, 128)
            post_w = min(post_cols, 1024 if r_pair <= 16 else 512)
            for c0 in range(0, post_cols, post_w):
                c_l = pl.ds(c0, post_w)
                chip_cols_buf[:, pl.ds(pl.multiple_of(my_c_loc + c0, 128), post_w)] = post_shard[
                    pl.ds(chip_r0, r_pair), c_l
                ]
            pici_ops = []
            for d in range(1, n_chips):
                dst_chip = chip ^ jnp.int32(d)
                dst_r0 = pl.multiple_of(dst_chip * r_pair, 16)
                peer = me ^ jnp.int32(d << 1)
                op = pltpu.make_async_remote_copy(
                    post_shard.at[pl.ds(dst_r0, r_pair), :],
                    chip_cols_buf.at[:, pl.ds(my_c_loc, post_cols)],
                    post_ici_send.at[d - 1],
                    post_ici_recv.at[d - 1],
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                op.start()
                pici_ops.append(op)
            for op in pici_ops:
                op.wait()

            op_pd2d = pltpu.make_async_remote_copy(
                chip_cols_buf, mate_post_shard, post_d2d_send.at[0], post_d2d_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            op_pd2d.start()
            op_pd2d.wait()

            for k in range(n_chips):
                for c0 in range(0, post_cols, post_w):
                    c_src = pl.ds(k * post_cols + c0, post_w)
                    my_k = chip_cols_buf[:, c_src]
                    mate_k = mate_post_shard[:, c_src]
                    post_pair_full[:, pl.ds((2 * k) * post_cols + c0, post_w)] = lax.select(
                        is_even, my_k, mate_k
                    )
                    post_pair_full[:, pl.ds((2 * k + 1) * post_cols + c0, post_w)] = lax.select(
                        is_even, mate_k, my_k
                    )

    side_c0 = pl.multiple_of(side * half_h, 128)
    if r_pair == 16:
        half_spans = _tile_spans(half_h, 1024)
        lo_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        bc_chunks = []
        bg_chunks = []
        for off, cw in half_spans:
            c_sl = pl.ds(pl.multiple_of(off, 128), cw)
            bc = post_pair_full[:, c_sl].astype(jnp.float32)
            bg = post_ff_gain_ref[0:1, c_sl]
            bc_chunks.append(bc)
            bg_chunks.append(bg)
            bcs = bc * bg if post_ff_scale_input else bc
            lo_b_sq = lo_b_sq + _fold_cols_to_128(bcs * bcs)
        hi_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for off, cw in half_spans:
            c_sl = pl.ds(pl.multiple_of(half_h + off, 128), cw)
            bc = post_pair_full[:, c_sl].astype(jnp.float32)
            bg = post_ff_gain_ref[0:1, c_sl]
            bc_chunks.append(bc)
            bg_chunks.append(bg)
            bcs = bc * bg if post_ff_scale_input else bc
            hi_b_sq = hi_b_sq + _fold_cols_to_128(bcs * bcs)
        inv_b = lax.rsqrt(
            jnp.sum(lo_b_sq + hi_b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )

        n_half_blks = len(half_spans)
        lo_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        ns_chunks = []
        for idx_c, (off, cw) in enumerate(half_spans):
            c_sl = pl.ds(pl.multiple_of(off, 128), cw)
            bc = bc_chunks[idx_c]
            bg = bg_chunks[idx_c]
            bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
            ns = pf_alpha_ref[0:1, c_sl] * s_buf[:, c_sl] + pf_beta_ref[0:1, c_sl] * bn
            if round_stream_bf16_for_norm:
                ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
            s_buf[:, c_sl] = ns
            ns_chunks.append(ns)
            nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
            lo_n_sq = lo_n_sq + _fold_cols_to_128(nv * nv)
        hi_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for idx_c, (off, cw) in enumerate(half_spans):
            c_sl = pl.ds(pl.multiple_of(half_h + off, 128), cw)
            bc = bc_chunks[n_half_blks + idx_c]
            bg = bg_chunks[n_half_blks + idx_c]
            bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
            ns = pf_alpha_ref[0:1, c_sl] * s_buf[:, c_sl] + pf_beta_ref[0:1, c_sl] * bn
            if round_stream_bf16_for_norm:
                ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
            s_buf[:, c_sl] = ns
            ns_chunks.append(ns)
            nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
            hi_n_sq = hi_n_sq + _fold_cols_to_128(nv * nv)

        if on_stream_updated_fn is not None:
            on_stream_updated_fn()

        inv_n = lax.rsqrt(
            jnp.sum(lo_n_sq + hi_n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(next_in_eps)
        )
        if out_next_full_ref is not None:
            for sub_i, (off, cw) in enumerate(half_spans):
                c_g = pl.ds(pl.multiple_of(side_c0 + off, 128), cw)
                sv = lax.select(is_even, ns_chunks[sub_i], ns_chunks[n_half_blks + sub_i])
                ng = next_in_gain_ref[0:1, c_g]
                nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                out_next_full_ref[pl.ds(chip_r0, r_pair), c_g] = nn.astype(jnp.bfloat16)
        else:
            for idx_c, (off, cw) in enumerate(half_spans):
                c_sl = pl.ds(pl.multiple_of(off, 128), cw)
                sv = ns_chunks[idx_c]
                ng = next_in_gain_ref[0:1, c_sl]
                nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                next_in_pair[:, c_sl] = nn.astype(jnp.bfloat16)
            for idx_c, (off, cw) in enumerate(half_spans):
                c_sl = pl.ds(pl.multiple_of(half_h + off, 128), cw)
                sv = ns_chunks[n_half_blks + idx_c]
                ng = next_in_gain_ref[0:1, c_sl]
                nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                next_in_pair[:, c_sl] = nn.astype(jnp.bfloat16)
    elif r_pair <= 32:
        half_spans = _tile_spans(half_h, 1024 if r_pair <= 16 else 512)
        lo_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for off, cw in half_spans:
            c_sl = pl.ds(pl.multiple_of(off, 128), cw)
            bc = post_pair_full[:, c_sl].astype(jnp.float32)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_sl]
            lo_b_sq = lo_b_sq + _fold_cols_to_128(bc * bc)
        hi_b_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for off, cw in half_spans:
            c_sl = pl.ds(pl.multiple_of(half_h + off, 128), cw)
            bc = post_pair_full[:, c_sl].astype(jnp.float32)
            if post_ff_scale_input:
                bc = bc * post_ff_gain_ref[0:1, c_sl]
            hi_b_sq = hi_b_sq + _fold_cols_to_128(bc * bc)
        inv_b = lax.rsqrt(
            jnp.sum(lo_b_sq + hi_b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(post_ff_eps)
        )

        lo_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for off, cw in half_spans:
            c_sl = pl.ds(pl.multiple_of(off, 128), cw)
            bc = post_pair_full[:, c_sl].astype(jnp.float32)
            bg = post_ff_gain_ref[0:1, c_sl]
            bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
            ns = pf_alpha_ref[0:1, c_sl] * s_buf[:, c_sl] + pf_beta_ref[0:1, c_sl] * bn
            if round_stream_bf16_for_norm:
                ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
            s_buf[:, c_sl] = ns
            nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
            lo_n_sq = lo_n_sq + _fold_cols_to_128(nv * nv)
        hi_n_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for off, cw in half_spans:
            c_sl = pl.ds(pl.multiple_of(half_h + off, 128), cw)
            bc = post_pair_full[:, c_sl].astype(jnp.float32)
            bg = post_ff_gain_ref[0:1, c_sl]
            bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
            ns = pf_alpha_ref[0:1, c_sl] * s_buf[:, c_sl] + pf_beta_ref[0:1, c_sl] * bn
            if round_stream_bf16_for_norm:
                ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
            s_buf[:, c_sl] = ns
            nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
            hi_n_sq = hi_n_sq + _fold_cols_to_128(nv * nv)

        if on_stream_updated_fn is not None:
            on_stream_updated_fn()

        inv_n = lax.rsqrt(
            jnp.sum(lo_n_sq + hi_n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(next_in_eps)
        )
        if out_next_full_ref is not None and r_pair >= 16:
            for off, cw in half_spans:
                c_g = pl.ds(pl.multiple_of(side_c0 + off, 128), cw)
                sv = s_buf[:, c_g]
                ng = next_in_gain_ref[0:1, c_g]
                nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                out_next_full_ref[pl.ds(chip_r0, r_pair), c_g] = nn.astype(jnp.bfloat16)
        else:
            for off, cw in _tile_spans(hidden, 1024 if r_pair <= 16 else 512):
                c_sl = pl.ds(pl.multiple_of(off, 128), cw)
                sv = s_buf[:, c_sl]
                ng = next_in_gain_ref[0:1, c_sl]
                nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                next_in_pair[:, c_sl] = nn.astype(jnp.bfloat16)
    else:
        bm_p = _pick_strip(r_pair, 32)
        col_blk = 512 if (half_h % 512 == 0) else _COL_STRIP

        def _row_epi(r0):
            rs = pl.ds(r0, bm_p)
            b_sq = jnp.zeros((bm_p, 128), dtype=jnp.float32)
            for c0 in range(0, hidden, col_blk):
                c_sl = pl.ds(c0, col_blk)
                bc = post_pair_full[rs, c_sl].astype(jnp.float32)
                if post_ff_scale_input:
                    bc = bc * post_ff_gain_ref[0:1, c_sl]
                b_sq = b_sq + _fold_cols_to_128(bc * bc)
            inv_b = lax.rsqrt(
                jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(post_ff_eps)
            )
            n_sq = jnp.zeros((bm_p, 128), dtype=jnp.float32)
            for c0 in range(0, hidden, col_blk):
                c_sl = pl.ds(c0, col_blk)
                bc = post_pair_full[rs, c_sl].astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_sl]
                bn = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                ns = pf_alpha_ref[0:1, c_sl] * s_buf[rs, c_sl] + pf_beta_ref[0:1, c_sl] * bn
                if round_stream_bf16_for_norm:
                    ns = ns.astype(jnp.bfloat16).astype(jnp.float32)
                s_buf[rs, c_sl] = ns
                nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
                n_sq = n_sq + _fold_cols_to_128(nv * nv)
            inv_n = lax.rsqrt(
                jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(next_in_eps)
            )
            if out_next_full_ref is not None:
                rs_go = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm_p)
                for c0 in range(0, half_h, col_blk):
                    c_g = pl.ds(pl.multiple_of(side_c0 + c0, 128), col_blk)
                    sv = s_buf[rs, c_g]
                    ng = next_in_gain_ref[0:1, c_g]
                    nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                    out_next_full_ref[rs_go, c_g] = nn.astype(jnp.bfloat16)
            else:
                for c0 in range(0, hidden, col_blk):
                    c_sl = pl.ds(c0, col_blk)
                    sv = s_buf[rs, c_sl]
                    ng = next_in_gain_ref[0:1, c_sl]
                    nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                    next_in_pair[rs, c_sl] = nn.astype(jnp.bfloat16)

        _fori_strips(r_pair, bm_p, _row_epi)
        if on_stream_updated_fn is not None:
            on_stream_updated_fn()


def _run_chip_b2_vmem_core(
    stream_chip_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_w_qkv_hbm,
    next_s_qkv_ref,
    next_w_gate_hbm,
    next_s_gate_ref,
    cos_sin_ref,
    out_stream_chip_hbm,
    out_qkv_or_q_ref,
    out_kv_or_gate_ref,
    out_gate_epi_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_pair,
    next_in_vmem,
    qkv_vmem,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_pair_full,
    cols_3d_buf,
    ag_3d_buf,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    tok_d2d2_send,
    tok_d2d2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    in_ici_send,
    in_ici_recv,
    in_d2d2_send,
    in_d2d2_recv,
    *,
    world_size: int,
    sharded_moe_in: bool,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    fuse_qkv_epi: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
    skip_barrier_signal: bool = False,
    skip_barrier_wait: bool = False,
    skip_stream_load: bool = False,
    skip_stream_store: bool = False,
    prestarted_w_post_dma: bool = False,
    prestarted_w_qkv_dma: bool = False,
    prestarted_w_gate_dma: bool = False,
    gate_dst_ref=None,
    after_qkv_epi_fn=None,
    defer_stream_store_wait: bool = False,
    during_in_ici_fn=None,
    during_in_ici_early: bool = False,
) -> None:
    """Reusable VMEM-resident Wave-9 chip-sharded Bridge 2b core (`standalone` & `b1+MoE+b2`).

    Gain vectors (`post_ff_gain`, `pf_alpha`, `pf_beta`, `next_in_gain`) are `(1, cols) f32` via `_as_2d_f32`;
    dot scales (`s_post`, `next_s_qkv`, `next_s_gate`) follow `_as_dot_scale_f32`. When chained after `b1+MoE`
    in a single superkernel (`s_buf` already holds live `f32` residual), pass `skip_stream_load=True`,
    `skip_barrier_signal/wait` as appropriate, and `prestarted_w_*_dma=True` for weights overlapped with `MoE RS`.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    r_pair = s_buf.shape[0]
    rows = (world_size // 2) * r_pair
    qkv_cols = w_qkv_buf.shape[1]
    gate_cols = w_gate_buf.shape[1]

    me = lax.axis_index("model").astype(jnp.int32)
    if not skip_barrier_signal:
        _signal_barrier(me, world_size)
    if not skip_stream_load:
        pltpu.make_async_copy(stream_chip_hbm, s_buf, s_in_sem.at[0]).start()
    if not prestarted_w_post_dma:
        pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    if not prestarted_w_qkv_dma:
        pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    if not prestarted_w_gate_dma:
        pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    _run_chip_sharded_post_fast_core(
        moe_out_ref, w_post_buf, s_post_ref, s_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
        next_in_pair, mate_tok_buf, moe_full_buf,
        post_shard, mate_post_shard, chip_cols_buf, post_pair_full, cols_3d_buf,
        w_post_sem, s_in_sem,
        tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
        post_d2d_send, post_d2d_recv, post_ici_send, post_ici_recv,
        me=me, world_size=world_size, sharded_moe_in=sharded_moe_in,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
        round_stream_bf16_for_norm=False,
        on_stream_updated_fn=(
            None
            if skip_stream_store
            else (
                lambda: pltpu.make_async_copy(
                    s_buf, out_stream_chip_hbm, s_out_sem.at[0]
                ).start()
            )
        ),
        tok_d2d2_send=tok_d2d2_send,
        tok_d2d2_recv=tok_d2d2_recv,
        out_next_full_ref=next_in_vmem if r_pair >= 16 else None,
        skip_barrier_wait=skip_barrier_wait,
        skip_stream_load=skip_stream_load,
    )

    # When is_tp2_wpost=True, post_ici_send/recv (ag_log) are unused after PAIR_RS / barrier;
    # borrow 2 disjoint semaphores so TP=8 B>=128 Step (c) unlocks 5-sem in-place 2D-XY in_ag.
    eff_in_s, eff_in_r = in_ici_send, in_ici_recv
    is_tp2_wpost = (2 * w_post_buf.shape[1]) == s_buf.shape[1]
    if (
        world_size == 8
        and r_pair >= 32
        and is_tp2_wpost
        and post_ici_send is not in_ici_send
        and post_ici_recv is not in_ici_recv
        and min(int(in_ici_send.shape[0]), int(in_ici_recv.shape[0])) < 5
        and int(post_ici_send.shape[0]) >= 2
        and int(post_ici_recv.shape[0]) >= 2
    ):
        eff_in_s = _CombinedSemSeq(
            [in_ici_send.at[0], in_ici_send.at[1], in_ici_send.at[2], post_ici_send.at[0], post_ici_send.at[1]]
        )
        eff_in_r = _CombinedSemSeq(
            [in_ici_recv.at[0], in_ici_recv.at[1], in_ici_recv.at[2], post_ici_recv.at[0], post_ici_recv.at[1]]
        )

    _ag_chip_sharded_half_col_xor1(
        next_in_pair if r_pair == 8 else None, next_in_vmem,
        eff_in_s, eff_in_r, in_d2d2_send, in_d2d2_recv,
        me=me, world_size=world_size, ag_3d_buf=ag_3d_buf,
        during_ici_fn=during_in_ici_fn,
        during_d2d2_fn=lambda: _dma_wait(w_qkv_buf, w_qkv_sem.at[0]),
        r_pair_in=r_pair,
        copy_local_from_pair=(r_pair == 8),
        during_ici_early=(during_in_ici_early or (world_size == 8 and r_pair >= 32)),
    )

    qkv_dst = qkv_vmem if fuse_qkv_epi else out_qkv_or_q_ref
    gate_dst = (
        gate_dst_ref
        if gate_dst_ref is not None
        else (out_gate_epi_ref if fuse_qkv_epi else out_kv_or_gate_ref)
    )
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_dst, 0,
        rows=rows, n_cols=qkv_cols,
    )
    if fuse_qkv_epi:
        _run_in_vmem_qkv_epilogue_dispatch(
            qkv_vmem,
            cos_sin_ref,
            out_qkv_or_q_ref,
            out_kv_or_gate_ref,
            norm_eps=qkv_norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    if after_qkv_epi_fn is not None:
        after_qkv_epi_fn()
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, gate_dst, 0,
        rows=rows, n_cols=gate_cols,
    )
    if not skip_stream_store and not defer_stream_store_wait:
        _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def _moe_to_attn_chip_sharded_bridge_kernel(
    *args,
    world_size: int,
    sharded_moe_in: bool,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    fuse_qkv_epi: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Wave-9 Chip-Sharded Bridge 2b (`TP=8` & `TP=16`, `0` `op_d2d1`, `-50%` `in_ici`, `0` `scf.for`)."""
    _run_chip_b2_vmem_core(
        *args,
        world_size=world_size,
        sharded_moe_in=sharded_moe_in,
        post_ff_eps=post_ff_eps,
        post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps,
        next_in_scale_input=next_in_scale_input,
        fuse_qkv_epi=fuse_qkv_epi,
        qkv_norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
    )


def moe_to_attn_chip_sharded_bridge_jax(
    stream_chip,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    next_in_gain,
    next_w_qkv_w8k,
    next_s_qkv,
    next_w_gate_w8k,
    next_s_gate,
    cos_sin_step=None,
    *,
    world_size: int = 8,
    sharded_moe_in: bool = True,
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    fuse_qkv_epi: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    collective_id: int = _W8_CID_BASE + 9,
    interpret: bool = False,
):
    """Chip-sharded Bridge 2b (`(stream_chip, moe_out) -> (stream_chip, qkv / q+kv, gate)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Chip-sharded Bridge 2b requires world_size in (8, 16), got {world_size}")
    n_chips = world_size // 2
    r_pair, hidden = stream_chip.shape
    rows = n_chips * r_pair
    in_r, k_in = moe_out.shape
    _, post_cols = w_post_w8k.shape
    is_tp2_wpost = 2 * post_cols == hidden
    expected_in_r = rows // world_size if sharded_moe_in else rows
    valid_rows = (32, 64, 128, 256, 384) if world_size == 8 else (64, 128, 256, 384)
    if (
        rows not in valid_rows
        or r_pair < 8
        or r_pair % 8 != 0
        or in_r != expected_in_r
        or in_r < 8
        or in_r % 8 != 0
        or post_cols % 128 != 0
        or (post_cols * world_size != hidden and not is_tp2_wpost)
    ):
        raise ValueError(
            f"Unsupported chip-sharded Bridge 2b shapes world_size={world_size}, "
            f"stream={stream_chip.shape}, moe_out={moe_out.shape}, sharded_moe_in={sharded_moe_in}"
        )
    has_rope = bool(fuse_qkv_epi and cos_sin_step is not None)
    cs_in = cos_sin_step[:rows] if (has_rope and cos_sin_step.shape[0] != rows) else cos_sin_step
    _, qkv_cols = next_w_qkv_w8k.shape
    _, gate_cols = next_w_gate_w8k.shape
    pair_cols = 2 * post_cols
    half_h = hidden // 2
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    if fuse_qkv_epi:
        out_shape = (
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        )
    else:
        out_shape = (
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, qkv_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        )
    out_specs = (h_sp, v_sp, v_sp, v_sp)
    moe_full_sh = (
        ((r_pair, k_in) if (sharded_moe_in or r_pair == 8) else (8, 128))
        if is_tp2_wpost
        else ((rows, k_in) if sharded_moe_in else (8, 128))
    )

    res = _pallas_call_maybe_interpret(
        functools.partial(
            _moe_to_attn_chip_sharded_bridge_kernel,
            world_size=world_size,
            sharded_moe_in=bool(sharded_moe_in),
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            fuse_qkv_epi=bool(fuse_qkv_epi),
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            has_rope=has_rope,
            interpret_mode=bool(interpret),
        ),
        interpret=interpret,
        out_shape=out_shape,
        in_specs=[
            h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp,
            h_sp, v_sp, h_sp, v_sp, v_sp if has_rope else None,
        ],
        out_specs=out_specs,
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((r_pair, hidden), jnp.float32),
            vmem((8, hidden) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, qkv_cols) if fuse_qkv_epi else (8, 128), jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem(moe_full_sh, jnp.bfloat16),
            vmem((8, 128) if is_tp2_wpost else (rows, post_cols), jnp.bfloat16),
            vmem((r_pair, half_h) if (is_tp2_wpost or r_pair >= 16) else (rows, post_cols), jnp.bfloat16),
            vmem((r_pair, half_h) if (is_tp2_wpost or r_pair >= 16) else (rows, pair_cols), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((n_chips, 8, pair_cols) if (r_pair == 8 and not is_tp2_wpost) else (8, 128), jnp.bfloat16),
            vmem((n_chips, 8, half_h) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(60 if (rows > 256 or is_tp2_wpost) else 56) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w9_chip_b2_b{rows}_tp{world_size}"
            f"{'_tp2wp' if is_tp2_wpost else ''}"
            f"{'_sh' if sharded_moe_in else '_full'}"
            f"{'_epi_' + ('rope' if has_rope else 'nope') if fuse_qkv_epi else ''}"
        ),
    )(
        stream_chip.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_dot_scale_f32(s_post, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_dot_scale_f32(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_dot_scale_f32(next_s_gate, gate_cols),
        cs_in if has_rope else None,
    )
    if fuse_qkv_epi:
        st_ch, q_3d, kv_m, gate = res
        return st_ch, q_3d.reshape(1, rows, 8, 2, 128), kv_m, gate
    st_ch, qkv, gate, _ = res
    return st_ch, qkv, gate


def _moe_post_chip_sharded_final_norm_kernel(
    stream_chip_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    final_gain_ref,
    out_stream_chip_hbm,
    out_hidden_ref,
    w_post_buf,
    s_buf,
    hidden_pair,
    mate_tok_buf,
    moe_full_buf,
    post_shard,
    mate_post_shard,
    chip_cols_buf,
    post_pair_full,
    cols_3d_buf,
    ag_3d_buf,
    w_post_sem,
    s_in_sem,
    s_out_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    tok_d2d2_send,
    tok_d2d2_recv,
    post_d2d_send,
    post_d2d_recv,
    post_ici_send,
    post_ici_recv,
    ag_ici_send,
    ag_ici_recv,
    ag_d2d2_send,
    ag_d2d2_recv,
    *,
    world_size: int,
    sharded_moe_in: bool,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    final_eps: float,
    final_scale_input: bool,
    return_stream_too: bool,
) -> None:
    """Wave-9 Chip-Sharded L61 Final Norm / Bridge 2a (`TP=8` & `TP=16`)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    r_pair = stream_chip_hbm.shape[0]
    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(stream_chip_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()

    _run_chip_sharded_post_fast_core(
        moe_out_ref, w_post_buf, s_post_ref, s_buf,
        post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, final_gain_ref,
        hidden_pair, mate_tok_buf, moe_full_buf,
        post_shard, mate_post_shard, chip_cols_buf, post_pair_full, cols_3d_buf,
        w_post_sem, s_in_sem,
        tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
        post_d2d_send, post_d2d_recv, post_ici_send, post_ici_recv,
        me=me, world_size=world_size, sharded_moe_in=sharded_moe_in,
        post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
        next_in_eps=final_eps, next_in_scale_input=final_scale_input,
        round_stream_bf16_for_norm=not return_stream_too,
        on_stream_updated_fn=(
            (lambda: pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start())
            if return_stream_too
            else None
        ),
        tok_d2d2_send=tok_d2d2_send,
        tok_d2d2_recv=tok_d2d2_recv,
        out_next_full_ref=out_hidden_ref if r_pair >= 16 else None,
    )

    _ag_chip_sharded_half_col_xor1(
        hidden_pair if r_pair == 8 else None, out_hidden_ref,
        ag_ici_send, ag_ici_recv, ag_d2d2_send, ag_d2d2_recv,
        me=me, world_size=world_size, ag_3d_buf=ag_3d_buf,
        r_pair_in=r_pair,
        copy_local_from_pair=(r_pair == 8),
    )
    if return_stream_too:
        _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def moe_post_chip_sharded_final_norm_jax(
    stream_chip,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    final_gain,
    *,
    world_size: int = 8,
    sharded_moe_in: bool = True,
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    final_eps: float = 1e-5,
    final_scale_input: bool = False,
    return_stream_too: bool = False,
    collective_id: int = _W8_CID_BASE + 10,
    interpret: bool = False,
):
    """Chip-sharded L61 Final Norm / Bridge 2a (`TP=8` & `TP=16`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Chip-sharded Final Norm requires world_size in (8, 16), got {world_size}")
    n_chips = world_size // 2
    r_pair, hidden = stream_chip.shape
    rows = n_chips * r_pair
    in_r, k_in = moe_out.shape
    _, post_cols = w_post_w8k.shape
    is_tp2_wpost = 2 * post_cols == hidden
    expected_in_r = rows // world_size if sharded_moe_in else rows
    valid_rows = (32, 64, 128, 256, 384) if world_size == 8 else (64, 128, 256, 384)
    if (
        rows not in valid_rows
        or r_pair < 8
        or r_pair % 8 != 0
        or in_r != expected_in_r
        or in_r < 8
        or in_r % 8 != 0
        or post_cols % 128 != 0
        or (post_cols * world_size != hidden and not is_tp2_wpost)
    ):
        raise ValueError("Unsupported chip-sharded Final Norm shapes")
    pair_cols = 2 * post_cols
    half_h = hidden // 2
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
    moe_full_sh = (
        ((r_pair, k_in) if (sharded_moe_in or r_pair == 8) else (8, 128))
        if is_tp2_wpost
        else ((rows, k_in) if sharded_moe_in else (8, 128))
    )

    st_out, hid_out = _pallas_call_maybe_interpret(
        functools.partial(
            _moe_post_chip_sharded_final_norm_kernel,
            world_size=world_size,
            sharded_moe_in=bool(sharded_moe_in),
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            final_eps=float(final_eps),
            final_scale_input=bool(final_scale_input),
            return_stream_too=bool(return_stream_too),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((r_pair, hidden) if return_stream_too else (8, 128), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=(h_sp if return_stream_too else v_sp, v_sp),
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((r_pair, hidden), jnp.float32),
            vmem((8, hidden) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem(moe_full_sh, jnp.bfloat16),
            vmem((8, 128) if is_tp2_wpost else (rows, post_cols), jnp.bfloat16),
            vmem((r_pair, half_h) if (is_tp2_wpost or r_pair >= 16) else (rows, post_cols), jnp.bfloat16),
            vmem((r_pair, half_h) if (is_tp2_wpost or r_pair >= 16) else (rows, pair_cols), jnp.bfloat16),
            vmem((r_pair, hidden), jnp.bfloat16),
            vmem((n_chips, 8, pair_cols) if (r_pair == 8 and not is_tp2_wpost) else (8, 128), jnp.bfloat16),
            vmem((n_chips, 8, half_h) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(54 if is_tp2_wpost else 46) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w9_chip_fnorm_b{rows}_tp{world_size}"
            f"{'_tp2wp' if is_tp2_wpost else ''}"
            f"{'_sh' if sharded_moe_in else '_full'}{'_st' if return_stream_too else ''}"
        ),
    )(
        stream_chip.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_dot_scale_f32(s_post, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(final_gain, hidden),
    )
    return (st_out, hid_out) if return_stream_too else hid_out


def ag_chip_sharded_capture_jax(
    local_cap_bf16,
    *,
    world_size: int = 8,
    collective_id: int = _W8_CID_BASE + 11,
    interpret: bool = False,
):
    """Gather chip-sharded `[B // (TP // 2), K * H]` bf16 captures -> `[B, K * H]` via 1 XOR=1 AG."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"Chip-sharded capture AG requires world_size in (8, 16), got {world_size}"
        )
    r_pair, tot_cols = local_cap_bf16.shape
    n_chips = world_size // 2
    rows = n_chips * r_pair
    valid_rows = (32, 64, 128, 256, 384) if world_size == 8 else (64, 128, 256, 384)
    half_cols = tot_cols // 2
    ag_w = min(half_cols, 1024 if r_pair <= 16 else 512) if tot_cols >= 256 else 0
    if (
        rows not in valid_rows
        or r_pair < 8
        or r_pair % 8 != 0
        or tot_cols % 256 != 0
        or half_cols % 128 != 0
        or ag_w == 0
        or half_cols % ag_w != 0
    ):
        raise ValueError(
            f"Unsupported chip-sharded capture AG shapes world_size={world_size}, "
            f"local_cap={local_cap_bf16.shape}"
        )
    vmem, sem = pltpu.VMEM, pltpu.SemaphoreType.DMA
    v_sp = pl.BlockSpec(memory_space=vmem)

    def _kern(in_ref, out_ref, ag_3d_buf, ici_send, ici_recv, d2d_send, d2d_recv):
        me = lax.axis_index("model").astype(jnp.int32)
        _signal_barrier(me, world_size)
        _wait_barrier(world_size)
        _ag_chip_sharded_half_col_xor1(
            in_ref,
            out_ref,
            ici_send,
            ici_recv,
            d2d_send,
            d2d_recv,
            me=me,
            world_size=world_size,
            ag_3d_buf=ag_3d_buf if r_pair == 8 else None,
            r_pair_in=r_pair,
            copy_local_from_pair=True,
        )

    return _pallas_call_maybe_interpret(
        _kern,
        interpret=interpret,
        out_shape=jax.ShapeDtypeStruct((rows, tot_cols), jnp.bfloat16),
        in_specs=[v_sp],
        out_specs=v_sp,
        scratch_shapes=[
            vmem((n_chips, 8, half_cols) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((5 if world_size == 8 else n_chips - 1,)),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=56 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w9_chip_ag_cap_b{rows}_c{tot_cols}_tp{world_size}",
    )(local_cap_bf16.astype(jnp.bfloat16))


# =============================================================================
# 6. TorchTPU `cpu_export_jax_op` Registrations
# =============================================================================
def get_fp8_dual_qkv_epi_gate_op(has_rope: bool) -> Callable:
    key = bool(has_rope)
    op = _DUAL_QKV_EPI_GATE_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _DUAL_QKV_EPI_GATE_OPS.get(key)
        if op is None:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            if has_rope:

                def _wrapped_r(
                    attn_in: jax.Array,
                    w_qkv: jax.Array,
                    s_qkv: jax.Array,
                    w_gate: jax.Array,
                    s_gate: jax.Array,
                    cos_sin_step: jax.Array,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return fp8_dual_qkv_epi_gate_jax(
                        attn_in, w_qkv, s_qkv, w_gate, s_gate, cos_sin_step,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )

                op = cpu_export_jax_op(
                    "sglang_tpu_pallas::muse_w8_dual_qkv_epi_gate_rope", _wrapped_r
                )
            else:

                def _wrapped_n(
                    attn_in: jax.Array,
                    w_qkv: jax.Array,
                    s_qkv: jax.Array,
                    w_gate: jax.Array,
                    s_gate: jax.Array,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return fp8_dual_qkv_epi_gate_jax(
                        attn_in, w_qkv, s_qkv, w_gate, s_gate, None,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )

                op = cpu_export_jax_op(
                    "sglang_tpu_pallas::muse_w8_dual_qkv_epi_gate_nope", _wrapped_n
                )
            _DUAL_QKV_EPI_GATE_OPS[key] = op
    return op


def get_moe_post_ag_final_norm_op(
    world_size: int, *, sharded_moe_in: bool = False
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    nbs3 = _qepi_norm_before_s3_enabled()
    key = (int(world_size), bool(sharded_moe_in), nbs3)
    op = _MOE_POST_FINAL_NORM_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _MOE_POST_FINAL_NORM_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            t_specs = (P(), moe_sp, P(None, "model"), P("model"), P(), P(), P(), P())
            in_specs = (*t_specs, None, None, None, None)

            def _wrapped(
                stream: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                final_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                final_eps: float,
                final_scale_input: bool,
            ) -> jax.Array:
                fn = functools.partial(
                    moe_post_ag_final_norm_jax,
                    world_size=world_size,
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    final_eps=float(final_eps),
                    final_scale_input=bool(final_scale_input),
                    sharded_moe_in=bool(sharded_moe_in),
                    norm_before_s3=nbs3,
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=P(), check_vma=False
                )(stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta, final_gain)

            sfx = f"{'_sh' if sharded_moe_in else ''}{'' if nbs3 else '_leg'}"
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8_post_ag_fnorm_tp{world_size}{sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=(P(),),
            )
            _MOE_POST_FINAL_NORM_OPS[key] = op
    return op


def get_moe_to_attn_qkv_epi_bridge_op(
    world_size: int, *, sharded_moe_in: bool = False, has_rope: bool = True
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Fused QKV epilogue Bridge 2b requires world_size in (8, 16), got {world_size}")
    nbs3 = _qepi_norm_before_s3_enabled()
    key = (int(world_size), bool(sharded_moe_in), bool(has_rope), nbs3)
    op = _MOE_TO_ATTN_QKV_EPI_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _MOE_TO_ATTN_QKV_EPI_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            base_t_specs = (
                P(),
                moe_sp,
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(None, "model"),
                P("model"),
            )
            out_specs = (
                P(),
                P(None, None, "model", None, None),
                P(None, "model", None, None),
                P(None, "model"),
            )
            sfx = f"{'_sh' if sharded_moe_in else ''}_{'rope' if has_rope else 'nope'}{'' if nbs3 else '_leg'}"

            if has_rope:
                t_specs_r = (*base_t_specs, P())
                in_specs_r = (*t_specs_r, None, None, None, None, None, None, None, None)

                def _wrapped_r(
                    stream: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    cos_sin_step: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    fn = functools.partial(
                        moe_to_attn_qkv_epi_bridge_jax,
                        world_size=world_size,
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        sharded_moe_in=bool(sharded_moe_in),
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        norm_before_s3=nbs3,
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs_r, out_specs=out_specs, check_vma=False
                    )(
                        stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta,
                        next_in_gain, next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
                        cos_sin_step,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_moe_to_qkv_epi_tp{world_size}{sfx}",
                    _wrapped_r,
                    mesh=mesh,
                    input_partition_specs=in_specs_r,
                    output_partition_specs=out_specs,
                )
            else:
                in_specs_n = (*base_t_specs, None, None, None, None, None, None, None, None)

                def _wrapped_n(
                    stream: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    def _local(*args):
                        return moe_to_attn_qkv_epi_bridge_jax(
                            *args,
                            None,
                            world_size=world_size,
                            post_ff_eps=float(post_ff_eps),
                            post_ff_scale_input=bool(post_ff_scale_input),
                            next_in_eps=float(next_in_eps),
                            next_in_scale_input=bool(next_in_scale_input),
                            sharded_moe_in=bool(sharded_moe_in),
                            qkv_norm_eps=float(qkv_norm_eps),
                            query_scale=float(query_scale),
                            k_scale=float(k_scale),
                            v_scale=float(v_scale),
                            norm_before_s3=nbs3,
                        )

                    return jax.shard_map(
                        _local, mesh=mesh, in_specs=base_t_specs, out_specs=out_specs, check_vma=False
                    )(
                        stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta,
                        next_in_gain, next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_moe_to_qkv_epi_tp{world_size}{sfx}",
                    _wrapped_n,
                    mesh=mesh,
                    input_partition_specs=in_specs_n,
                    output_partition_specs=out_specs,
                )
            _MOE_TO_ATTN_QKV_EPI_OPS[key] = op
    return op


def get_decode_vocab_embed_norm_op(world_size: int) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    op = _EMBED_NORM_OPS.get(world_size)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _EMBED_NORM_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (P(), P("model", None))
            in_specs = (*t_specs, None)

            def _wrapped(
                input_ids: jax.Array,
                embed_weight_shard: jax.Array,
                norm_eps: float,
            ) -> jax.Array:
                fn = functools.partial(
                    decode_vocab_embed_norm_jax,
                    world_size=world_size,
                    norm_eps=float(norm_eps),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=P(), check_vma=False
                )(input_ids, embed_weight_shard)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8_embed_norm_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=(P(),),
            )
            _EMBED_NORM_OPS[world_size] = op
    return op


def get_decode_vocab_embed_tok_sharded_op(world_size: int) -> Callable:
    if world_size != 8:
        raise ValueError(f"Token-sharded embed entry requires world_size=8, got {world_size}")
    op = _EMBED_TOK_SHARDED_OPS.get(world_size)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _EMBED_TOK_SHARDED_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (P(), P("model", None), P())
            in_specs = (*t_specs, None, None, None)
            out_specs = (P("model", None), P())

            def _wrapped(
                input_ids: jax.Array,
                embed_weight_shard: jax.Array,
                l0_in_gain: jax.Array,
                embed_eps: float,
                l0_eps: float,
                l0_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array]:
                fn = functools.partial(
                    decode_vocab_embed_tok_sharded_jax,
                    world_size=world_size,
                    embed_eps=float(embed_eps),
                    l0_eps=float(l0_eps),
                    l0_scale_input=bool(l0_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(input_ids, embed_weight_shard, l0_in_gain)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8_embed_tok_sh_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _EMBED_TOK_SHARDED_OPS[world_size] = op
    return op


def get_attn_to_moe_tok_sharded_bridge_op(
    world_size: int, *, bf16_router: bool = True
) -> Callable:
    if world_size != 8:
        raise ValueError(f"Token-sharded Bridge 1b requires world_size=8, got {world_size}")
    fast_flag = _tok_b1_fast_enabled()
    key = (int(world_size), bool(bf16_router), fast_flag)
    fast_sfx = "" if fast_flag else "_legacy"
    op = _ATTN_TO_MOE_TOK_SH_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _ATTN_TO_MOE_TOK_SH_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (
                P("model", None),
                P(None, "model"),
                P("model", None),
                P("model"),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(),
                P(),
            )
            in_specs = (*t_specs, None, None, None, None, None, None)
            out_specs = (P("model", None), P(), P())

            def _wrapped(
                stream_shard: jax.Array,
                gated_attn: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                def _local(st, ga, wo, ws, a, b, pfg, wp, wps, wr, peg):
                    return attn_to_moe_tok_sharded_bridge_jax(
                        st, ga, wo, ws, a, b, pfg, wp, wps, wr, peg,
                        world_size=world_size,
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                        use_fast_vpu=fast_flag,
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream_shard, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
                    pre_ff_gain, w_pre_pair_w8k, w_pre_pair_scale, w_router_km, pre_exp_gain,
                )

            tag = "r16" if bf16_router else "r32"
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8_tok_attn_to_moe_{tag}_tp{world_size}{fast_sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _ATTN_TO_MOE_TOK_SH_OPS[key] = op
    return op


def get_moe_to_attn_tok_sharded_bridge_op(
    world_size: int, *, fuse_qkv_epi: bool = False, has_rope: bool = True
) -> Callable:
    if world_size != 8:
        raise ValueError(f"Token-sharded Bridge 2b requires world_size=8, got {world_size}")
    eff_rope = bool(fuse_qkv_epi and has_rope)
    fast_flag = _tok_b2_fast_enabled()
    key = (int(world_size), bool(fuse_qkv_epi), eff_rope, fast_flag)
    fast_sfx = "" if fast_flag else "_legacy"
    op = _MOE_TO_ATTN_TOK_SH_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _MOE_TO_ATTN_TOK_SH_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            base_t_specs = (
                P("model", None),
                P("model", None),
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(None, "model"),
                P("model"),
            )

            if not fuse_qkv_epi:
                in_specs_plain = (*base_t_specs, None, None, None, None)
                out_specs_plain = (P("model", None), P(None, "model"), P(None, "model"))

                def _wrapped_plain(
                    stream_shard: jax.Array,
                    moe_out: jax.Array,
                    w_post_pair_w8k: jax.Array,
                    s_post_pair: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    def _local(*args):
                        return moe_to_attn_tok_sharded_bridge_jax(
                            *args,
                            None,
                            world_size=world_size,
                            post_ff_eps=float(post_ff_eps),
                            post_ff_scale_input=bool(post_ff_scale_input),
                            next_in_eps=float(next_in_eps),
                            next_in_scale_input=bool(next_in_scale_input),
                            fuse_qkv_epi=False,
                        )

                    return jax.shard_map(
                        _local, mesh=mesh, in_specs=base_t_specs, out_specs=out_specs_plain, check_vma=False
                    )(
                        stream_shard, moe_out, w_post_pair_w8k, s_post_pair, post_ff_gain,
                        pf_alpha, pf_beta, next_in_gain, next_w_qkv_w8k, next_s_qkv,
                        next_w_gate_w8k, next_s_gate,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_tok_moe_to_qkvg_tp{world_size}{fast_sfx}",
                    _wrapped_plain,
                    mesh=mesh,
                    input_partition_specs=in_specs_plain,
                    output_partition_specs=out_specs_plain,
                )
            elif eff_rope:
                t_specs_r = (*base_t_specs, P())
                in_specs_r = (*t_specs_r, None, None, None, None, None, None, None, None)
                out_specs_epi = (
                    P("model", None),
                    P(None, None, "model", None, None),
                    P(None, "model", None, None),
                    P(None, "model"),
                )

                def _wrapped_epi_r(
                    stream_shard: jax.Array,
                    moe_out: jax.Array,
                    w_post_pair_w8k: jax.Array,
                    s_post_pair: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    cos_sin_step: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    fn = functools.partial(
                        moe_to_attn_tok_sharded_bridge_jax,
                        world_size=world_size,
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        fuse_qkv_epi=True,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs_r, out_specs=out_specs_epi, check_vma=False
                    )(
                        stream_shard, moe_out, w_post_pair_w8k, s_post_pair, post_ff_gain,
                        pf_alpha, pf_beta, next_in_gain, next_w_qkv_w8k, next_s_qkv,
                        next_w_gate_w8k, next_s_gate, cos_sin_step,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_tok_moe_to_qkv_epi_rope_tp{world_size}",
                    _wrapped_epi_r,
                    mesh=mesh,
                    input_partition_specs=in_specs_r,
                    output_partition_specs=out_specs_epi,
                )
            else:
                in_specs_n = (*base_t_specs, None, None, None, None, None, None, None, None)
                out_specs_epi = (
                    P("model", None),
                    P(None, None, "model", None, None),
                    P(None, "model", None, None),
                    P(None, "model"),
                )

                def _wrapped_epi_n(
                    stream_shard: jax.Array,
                    moe_out: jax.Array,
                    w_post_pair_w8k: jax.Array,
                    s_post_pair: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    def _local(*args):
                        return moe_to_attn_tok_sharded_bridge_jax(
                            *args,
                            None,
                            world_size=world_size,
                            post_ff_eps=float(post_ff_eps),
                            post_ff_scale_input=bool(post_ff_scale_input),
                            next_in_eps=float(next_in_eps),
                            next_in_scale_input=bool(next_in_scale_input),
                            fuse_qkv_epi=True,
                            qkv_norm_eps=float(qkv_norm_eps),
                            query_scale=float(query_scale),
                            k_scale=float(k_scale),
                            v_scale=float(v_scale),
                        )

                    return jax.shard_map(
                        _local, mesh=mesh, in_specs=base_t_specs, out_specs=out_specs_epi, check_vma=False
                    )(
                        stream_shard, moe_out, w_post_pair_w8k, s_post_pair, post_ff_gain,
                        pf_alpha, pf_beta, next_in_gain, next_w_qkv_w8k, next_s_qkv,
                        next_w_gate_w8k, next_s_gate,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_tok_moe_to_qkv_epi_nope_tp{world_size}",
                    _wrapped_epi_n,
                    mesh=mesh,
                    input_partition_specs=in_specs_n,
                    output_partition_specs=out_specs_epi,
                )
            _MOE_TO_ATTN_TOK_SH_OPS[key] = op
    return op


def get_moe_post_tok_sharded_final_norm_op(
    world_size: int, *, return_stream_too: bool = False
) -> Callable:
    if world_size != 8:
        raise ValueError(f"Token-sharded Final Norm requires world_size=8, got {world_size}")
    fast_flag = _tok_b2_fast_enabled()
    key = (int(world_size), bool(return_stream_too), fast_flag)
    fast_sfx = "" if fast_flag else "_legacy"
    op = _MOE_POST_TOK_SH_FINAL_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _MOE_POST_TOK_SH_FINAL_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (
                P("model", None),
                P("model", None),
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
            )
            in_specs = (*t_specs, None, None, None, None)
            out_sp = (P("model", None), P()) if return_stream_too else P()
            out_sp_tuple = out_sp if return_stream_too else (P(),)

            if return_stream_too:

                def _wrapped_st(
                    stream_shard: jax.Array,
                    moe_out: jax.Array,
                    w_post_pair_w8k: jax.Array,
                    s_post_pair: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    final_gain: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    final_eps: float,
                    final_scale_input: bool,
                ) -> tuple[jax.Array, jax.Array]:
                    fn = functools.partial(
                        moe_post_tok_sharded_final_norm_jax,
                        world_size=world_size,
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        final_eps=float(final_eps),
                        final_scale_input=bool(final_scale_input),
                        return_stream_too=True,
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs, out_specs=out_sp, check_vma=False
                    )(
                        stream_shard, moe_out, w_post_pair_w8k, s_post_pair,
                        post_ff_gain, pf_alpha, pf_beta, final_gain,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_tok_post_rn_tp{world_size}{fast_sfx}",
                    _wrapped_st,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_sp_tuple,
                )
            else:

                def _wrapped_fn(
                    stream_shard: jax.Array,
                    moe_out: jax.Array,
                    w_post_pair_w8k: jax.Array,
                    s_post_pair: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    final_gain: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    final_eps: float,
                    final_scale_input: bool,
                ) -> jax.Array:
                    fn = functools.partial(
                        moe_post_tok_sharded_final_norm_jax,
                        world_size=world_size,
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        final_eps=float(final_eps),
                        final_scale_input=bool(final_scale_input),
                        return_stream_too=False,
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs, out_specs=out_sp, check_vma=False
                    )(
                        stream_shard, moe_out, w_post_pair_w8k, s_post_pair,
                        post_ff_gain, pf_alpha, pf_beta, final_gain,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8_tok_post_fnorm_tp{world_size}{fast_sfx}",
                    _wrapped_fn,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_sp_tuple,
                )
            _MOE_POST_TOK_SH_FINAL_OPS[key] = op
    return op


def get_decode_vocab_embed_chip_sharded_op(world_size: int) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"Chip-sharded embed entry requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    op = _EMBED_CHIP_SHARDED_OPS.get(world_size)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _EMBED_CHIP_SHARDED_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (P(), P("model", None), P())
            in_specs = (*t_specs, None, None, None)
            out_specs = (P("model", None), P())

            def _wrapped(
                input_ids: jax.Array,
                embed_weight_shard: jax.Array,
                l0_in_gain: jax.Array,
                embed_eps: float,
                l0_eps: float,
                l0_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array]:
                fn = functools.partial(
                    decode_vocab_embed_chip_sharded_jax,
                    world_size=world_size,
                    embed_eps=float(embed_eps),
                    l0_eps=float(l0_eps),
                    l0_scale_input=bool(l0_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(input_ids, embed_weight_shard, l0_in_gain)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w9_embed_chip_sh_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _EMBED_CHIP_SHARDED_OPS[world_size] = op
    return op


def get_attn_to_moe_chip_sharded_bridge_op(
    world_size: int, *, bf16_router: bool = True
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Chip-sharded Bridge 1b requires world_size in (8, 16), got {world_size}")
    prs_flag: bool | None = _tp16_parallel_rs_override() if world_size == 16 else False
    bc1s_flag = _bug_c_oneshot_enabled()
    prs_sfx = "_prs0" if prs_flag is False and world_size == 16 else ("_prs" if prs_flag else "")
    bc1s_sfx = "_bc1s" if bc1s_flag else ""
    key = (int(world_size), bool(bf16_router), prs_flag, bc1s_flag)
    op = _ATTN_TO_MOE_CHIP_SH_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _ATTN_TO_MOE_CHIP_SH_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (
                P("model", None),
                P(None, "model"),
                P("model", None),
                P("model"),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(),
                P(),
            )
            in_specs = (*t_specs, None, None, None, None, None, None)
            out_specs = (P("model", None), P(), P())

            def _wrapped(
                stream_chip: jax.Array,
                gated_attn: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                def _local(st, ga, wo, ws, a, b, pfg, wp, wps, wr, peg):
                    return attn_to_moe_chip_sharded_bridge_jax(
                        st, ga, wo, ws, a, b, pfg, wp, wps, wr, peg,
                        world_size=world_size,
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                        parallel_rs=prs_flag,
                        bug_c_oneshot=bc1s_flag,
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream_chip, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
                    pre_ff_gain, w_pre_pair_w8k, w_pre_pair_scale, w_router_km, pre_exp_gain,
                )

            tag = "r16" if bf16_router else "r32"
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w9_chip_attn_to_moe_{tag}_tp{world_size}{prs_sfx}{bc1s_sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _ATTN_TO_MOE_CHIP_SH_OPS[key] = op
    return op


def get_moe_to_attn_chip_sharded_bridge_op(
    world_size: int,
    *,
    sharded_moe_in: bool = True,
    fuse_qkv_epi: bool = False,
    has_rope: bool = True,
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Chip-sharded Bridge 2b requires world_size in (8, 16), got {world_size}")
    eff_rope = bool(fuse_qkv_epi and has_rope)
    key = (int(world_size), bool(sharded_moe_in), bool(fuse_qkv_epi), eff_rope)
    op = _MOE_TO_ATTN_CHIP_SH_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _MOE_TO_ATTN_CHIP_SH_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_spec = P("model", None) if sharded_moe_in else P()
            base_t_specs = (
                P("model", None),
                moe_spec,
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(None, "model"),
                P("model"),
            )
            sh_sfx = "_sh" if sharded_moe_in else "_full"

            if not fuse_qkv_epi:
                in_specs_plain = (*base_t_specs, None, None, None, None)
                out_specs_plain = (P("model", None), P(None, "model"), P(None, "model"))

                def _wrapped_plain(
                    stream_chip: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    def _local(*args):
                        return moe_to_attn_chip_sharded_bridge_jax(
                            *args,
                            None,
                            world_size=world_size,
                            sharded_moe_in=bool(sharded_moe_in),
                            post_ff_eps=float(post_ff_eps),
                            post_ff_scale_input=bool(post_ff_scale_input),
                            next_in_eps=float(next_in_eps),
                            next_in_scale_input=bool(next_in_scale_input),
                            fuse_qkv_epi=False,
                        )

                    return jax.shard_map(
                        _local, mesh=mesh, in_specs=base_t_specs, out_specs=out_specs_plain, check_vma=False
                    )(
                        stream_chip, moe_out, w_post_w8k, s_post, post_ff_gain,
                        pf_alpha, pf_beta, next_in_gain, next_w_qkv_w8k, next_s_qkv,
                        next_w_gate_w8k, next_s_gate,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w9_chip_moe_to_qkvg_tp{world_size}{sh_sfx}",
                    _wrapped_plain,
                    mesh=mesh,
                    input_partition_specs=in_specs_plain,
                    output_partition_specs=out_specs_plain,
                )
            elif eff_rope:
                t_specs_r = (*base_t_specs, P())
                in_specs_r = (*t_specs_r, None, None, None, None, None, None, None, None)
                out_specs_epi = (
                    P("model", None),
                    P(None, None, "model", None, None),
                    P(None, "model", None, None),
                    P(None, "model"),
                )

                def _wrapped_epi_r(
                    stream_chip: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    cos_sin_step: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    fn = functools.partial(
                        moe_to_attn_chip_sharded_bridge_jax,
                        world_size=world_size,
                        sharded_moe_in=bool(sharded_moe_in),
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        fuse_qkv_epi=True,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs_r, out_specs=out_specs_epi, check_vma=False
                    )(
                        stream_chip, moe_out, w_post_w8k, s_post, post_ff_gain,
                        pf_alpha, pf_beta, next_in_gain, next_w_qkv_w8k, next_s_qkv,
                        next_w_gate_w8k, next_s_gate, cos_sin_step,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w9_chip_moe_to_qkv_epi_rope_tp{world_size}{sh_sfx}",
                    _wrapped_epi_r,
                    mesh=mesh,
                    input_partition_specs=in_specs_r,
                    output_partition_specs=out_specs_epi,
                )
            else:
                in_specs_n = (*base_t_specs, None, None, None, None, None, None, None, None)
                out_specs_epi = (
                    P("model", None),
                    P(None, None, "model", None, None),
                    P(None, "model", None, None),
                    P(None, "model"),
                )

                def _wrapped_epi_n(
                    stream_chip: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    next_in_gain: jax.Array,
                    next_w_qkv_w8k: jax.Array,
                    next_s_qkv: jax.Array,
                    next_w_gate_w8k: jax.Array,
                    next_s_gate: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    def _local(*args):
                        return moe_to_attn_chip_sharded_bridge_jax(
                            *args,
                            None,
                            world_size=world_size,
                            sharded_moe_in=bool(sharded_moe_in),
                            post_ff_eps=float(post_ff_eps),
                            post_ff_scale_input=bool(post_ff_scale_input),
                            next_in_eps=float(next_in_eps),
                            next_in_scale_input=bool(next_in_scale_input),
                            fuse_qkv_epi=True,
                            qkv_norm_eps=float(qkv_norm_eps),
                            query_scale=float(query_scale),
                            k_scale=float(k_scale),
                            v_scale=float(v_scale),
                        )

                    return jax.shard_map(
                        _local, mesh=mesh, in_specs=base_t_specs, out_specs=out_specs_epi, check_vma=False
                    )(
                        stream_chip, moe_out, w_post_w8k, s_post, post_ff_gain,
                        pf_alpha, pf_beta, next_in_gain, next_w_qkv_w8k, next_s_qkv,
                        next_w_gate_w8k, next_s_gate,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w9_chip_moe_to_qkv_epi_nope_tp{world_size}{sh_sfx}",
                    _wrapped_epi_n,
                    mesh=mesh,
                    input_partition_specs=in_specs_n,
                    output_partition_specs=out_specs_epi,
                )
            _MOE_TO_ATTN_CHIP_SH_OPS[key] = op
    return op


def get_moe_post_chip_sharded_final_norm_op(
    world_size: int, *, sharded_moe_in: bool = True, return_stream_too: bool = False
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Chip-sharded Final Norm requires world_size in (8, 16), got {world_size}")
    key = (int(world_size), bool(sharded_moe_in), bool(return_stream_too))
    op = _MOE_POST_CHIP_SH_FINAL_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _MOE_POST_CHIP_SH_FINAL_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_spec = P("model", None) if sharded_moe_in else P()
            t_specs = (
                P("model", None),
                moe_spec,
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
            )
            in_specs = (*t_specs, None, None, None, None)
            out_sp = (P("model", None), P()) if return_stream_too else P()
            out_sp_tuple = out_sp if return_stream_too else (P(),)
            sh_sfx = "_sh" if sharded_moe_in else "_full"

            if return_stream_too:

                def _wrapped_st(
                    stream_chip: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    final_gain: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    final_eps: float,
                    final_scale_input: bool,
                ) -> tuple[jax.Array, jax.Array]:
                    fn = functools.partial(
                        moe_post_chip_sharded_final_norm_jax,
                        world_size=world_size,
                        sharded_moe_in=bool(sharded_moe_in),
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        final_eps=float(final_eps),
                        final_scale_input=bool(final_scale_input),
                        return_stream_too=True,
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs, out_specs=out_sp, check_vma=False
                    )(
                        stream_chip, moe_out, w_post_w8k, s_post,
                        post_ff_gain, pf_alpha, pf_beta, final_gain,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w9_chip_post_rn_tp{world_size}{sh_sfx}",
                    _wrapped_st,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_sp_tuple,
                )
            else:

                def _wrapped_fn(
                    stream_chip: jax.Array,
                    moe_out: jax.Array,
                    w_post_w8k: jax.Array,
                    s_post: jax.Array,
                    post_ff_gain: jax.Array,
                    pf_alpha: jax.Array,
                    pf_beta: jax.Array,
                    final_gain: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    final_eps: float,
                    final_scale_input: bool,
                ) -> jax.Array:
                    fn = functools.partial(
                        moe_post_chip_sharded_final_norm_jax,
                        world_size=world_size,
                        sharded_moe_in=bool(sharded_moe_in),
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        final_eps=float(final_eps),
                        final_scale_input=bool(final_scale_input),
                        return_stream_too=False,
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs, out_specs=out_sp, check_vma=False
                    )(
                        stream_chip, moe_out, w_post_w8k, s_post,
                        post_ff_gain, pf_alpha, pf_beta, final_gain,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w9_chip_post_fnorm_tp{world_size}{sh_sfx}",
                    _wrapped_fn,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_sp_tuple,
                )
            _MOE_POST_CHIP_SH_FINAL_OPS[key] = op
    return op


def get_ag_chip_sharded_capture_op(world_size: int) -> Callable:
    """Return TorchTPU custom op gathering chip-sharded `[r_pair, K*H]` capture tensors."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"Chip-sharded capture AG requires world_size in (8, 16), got {world_size}"
        )
    key = int(world_size)
    op = _AG_CHIP_CAP_OPS.get(key)
    if op is not None:
        return op
    with _W8_LOCK:
        op = _AG_CHIP_CAP_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)

            def _wrapped(local_cap_bf16: jax.Array) -> jax.Array:
                fn = functools.partial(ag_chip_sharded_capture_jax, world_size=world_size)
                return jax.shard_map(
                    fn,
                    mesh=mesh,
                    in_specs=(P("model", None),),
                    out_specs=P(),
                    check_vma=False,
                )(local_cap_bf16)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w9_chip_ag_cap_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=(P("model", None),),
                output_partition_specs=(P(),),
            )
            _AG_CHIP_CAP_OPS[key] = op
    return op


def prewarm_w8_decode_bridge_ops(
    world_size: int,
    mode: int,
    *,
    bf16_router: bool = False,
    tok_sharded: bool = False,
    chip_sharded: bool = False,
    bridge_qkv_epi: bool = False,
    fuse_final_norm: bool = False,
    pallas_embed: bool = False,
) -> None:
    """Eagerly register Wave-8 / Wave-9 custom ops outside ``torch.compile``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        return
    if pallas_embed:
        get_decode_vocab_embed_norm_op(world_size)
    if mode <= 0:
        return
    if fuse_final_norm and mode in (2, 3, 4, 5, 6):
        for sh in (False, True):
            get_moe_post_ag_final_norm_op(world_size, sharded_moe_in=sh)
    if bridge_qkv_epi and world_size in SUPPORTED_WORLD_SIZES and mode in (2, 3, 5):
        for r_flag in (False, True):
            get_fp8_dual_qkv_epi_gate_op(r_flag)
            for sh in (False, True):
                get_moe_to_attn_qkv_epi_bridge_op(
                    world_size, sharded_moe_in=sh, has_rope=r_flag
                )
    if tok_sharded and world_size == 8:
        if pallas_embed:
            get_decode_vocab_embed_tok_sharded_op(world_size)
        if mode in (1, 3, 4):
            for r16 in (False, True):
                get_attn_to_moe_tok_sharded_bridge_op(world_size, bf16_router=r16)
        if mode in (2, 3, 4, 5, 6):
            for ret_st in (False, True):
                get_moe_post_tok_sharded_final_norm_op(
                    world_size, return_stream_too=ret_st
                )
        if mode in (2, 3, 5):
            get_moe_to_attn_tok_sharded_bridge_op(world_size, fuse_qkv_epi=False)
            if bridge_qkv_epi:
                for r_flag in (False, True):
                    get_moe_to_attn_tok_sharded_bridge_op(
                        world_size, fuse_qkv_epi=True, has_rope=r_flag
                    )
    if chip_sharded and world_size in SUPPORTED_WORLD_SIZES:
        get_ag_chip_sharded_capture_op(world_size)
        if pallas_embed:
            get_decode_vocab_embed_chip_sharded_op(world_size)
        if mode in (1, 3, 4):
            for r16 in (False, True):
                get_attn_to_moe_chip_sharded_bridge_op(world_size, bf16_router=r16)
        if mode in (2, 3, 4, 5, 6):
            for sh in (False, True):
                for ret_st in (False, True):
                    get_moe_post_chip_sharded_final_norm_op(
                        world_size, sharded_moe_in=sh, return_stream_too=ret_st
                    )
        if mode in (2, 3, 5):
            for sh in (False, True):
                get_moe_to_attn_chip_sharded_bridge_op(
                    world_size, sharded_moe_in=sh, fuse_qkv_epi=False
                )
                if bridge_qkv_epi:
                    for r_flag in (False, True):
                        get_moe_to_attn_chip_sharded_bridge_op(
                            world_size,
                            sharded_moe_in=sh,
                            fuse_qkv_epi=True,
                            has_rope=r_flag,
                        )
