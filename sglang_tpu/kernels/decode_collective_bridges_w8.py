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
    _fori_strips,
    _gather_shard_xor1,
    _gather_token_shard_xor1_and_dot_post,
    _get_tp_mesh,
    _pick_strip,
    _signal_barrier,
    _wait_barrier,
)

_W8_CID_BASE = 40
_W8_LOCK = Lock()

_MOE_POST_FINAL_NORM_OPS: dict[tuple[int, bool], Callable] = {}
_MOE_TO_ATTN_QKV_EPI_OPS: dict[tuple[int, bool, bool], Callable] = {}
_DUAL_QKV_EPI_GATE_OPS: dict[bool, Callable] = {}
_EMBED_NORM_OPS: dict[int, Callable] = {}
_EMBED_TOK_SHARDED_OPS: dict[int, Callable] = {}
_ATTN_TO_MOE_TOK_SH_OPS: dict[tuple[int, bool], Callable] = {}
_MOE_TO_ATTN_TOK_SH_OPS: dict[tuple[int, bool, bool], Callable] = {}
_MOE_POST_TOK_SH_FINAL_OPS: dict[tuple[int, bool], Callable] = {}


def _as_2d_f32(x, cols: int):
    import jax.numpy as jnp

    if x.ndim == 2 and x.shape == (1, cols) and x.dtype == jnp.float32:
        return x
    return x.reshape(1, cols).astype(jnp.float32)


def _pallas_call_maybe_interpret(kernel_fn, *, interpret: bool, compiler_params, **kwargs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

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
    tile_r = 32
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
    # Overlap VPU QKV RMSNorm + RoPE + KV FP8 quant underneath 8-MiB W_gate HBM DMA!
    _run_in_vmem_qkv_epilogue(
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
        rows < 32
        or rows > 384
        or rows % 32 != 0
        or qkv_cols != 1280
        or gate_cols != 1024
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
        _as_2d_f32(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_2d_f32(s_gate, gate_cols),
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
    align = 16 if use_halfrow else 8
    n_strips = hidden // _COL_STRIP

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
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

    def _epi_final(src_buf, r_base, n_r):
        bm = _pick_strip(n_r, 32)

        def _strip(r_loc):
            rs_l = pl.ds(r_loc, bm)
            rs_g = pl.ds(pl.multiple_of(r_base + r_loc, align), bm)

            def _bvar(s, acc):
                c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                bc = src_buf[rs_l, c_sl].astype(jnp.float32)
                if post_ff_scale_input:
                    bc = bc * post_ff_gain_ref[0:1, c_sl]
                sq = bc * bc
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            b_sq = lax.fori_loop(0, n_strips, _bvar, jnp.zeros((bm, 128), dtype=jnp.float32))
            inv_b = lax.rsqrt(
                jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(post_ff_eps)
            )

            def _res_fvar(s, acc):
                c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                bc = src_buf[rs_l, c_sl].astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_sl]
                b_n = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                ns = pf_alpha_ref[0:1, c_sl] * s_in_buf[rs_g, c_sl] + pf_beta_ref[0:1, c_sl] * b_n
                # Match eager model.norm(stream.to(torch.bfloat16)) bit-for-bit!
                ns_bf = ns.astype(jnp.bfloat16)
                s_in_buf[rs_g, c_sl] = ns_bf.astype(jnp.float32)
                nv = s_in_buf[rs_g, c_sl]
                if final_scale_input:
                    nv = nv * final_gain_ref[0:1, c_sl]
                sq = nv * nv
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            f_sq = lax.fori_loop(0, n_strips, _res_fvar, jnp.zeros((bm, 128), dtype=jnp.float32))
            inv_f = lax.rsqrt(
                jnp.sum(f_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(final_eps)
            )

            def _fout(s, _):
                c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                sv = s_in_buf[rs_g, c_sl]
                fg = final_gain_ref[0:1, c_sl]
                fn = (sv * fg) * inv_f if final_scale_input else (sv * inv_f) * fg
                out_hidden_ref[rs_g, c_sl] = fn.astype(out_hidden_ref.dtype)
                return jnp.int32(0)

            lax.fori_loop(0, n_strips, _fout, jnp.int32(0))

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
    half_rows = rows // 2 if (rows >= 32 and rows % 32 == 0) else rows
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
        ),
        interpret=interpret,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=v_sp,
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
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


def _moe_to_attn_qkv_epi_bridge_kernel(
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
    interpret_mode: bool,
) -> None:
    """Wave-8 Bridge 2b with fused in-VMEM QKV/RoPE epilogue -> ``(stream, q_3d, kv, gate)``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    from jax.experimental import pallas as pl
    from sglang_tpu.kernels.decode_collective_bridges import (
        _prescale_s_buf_in_vmem,
        _run_post_ffn_slice_epilogue,
    )

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    use_halfrow = rows >= 32 and rows % 32 == 0
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

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

    bm_p = _pick_strip(rows, 32)
    half = rows // 2
    parity = me & jnp.int32(1)
    my_r0 = pl.multiple_of(parity * half, 16)
    mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half, 16)

    def _wait_and_prescale_stream():
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
        if use_halfrow:
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

    if not use_halfrow:
        pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_vmem, 0,
        rows=rows, n_cols=qkv_cols,
    )
    _run_in_vmem_qkv_epilogue(
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
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, out_next_gate_ref, 0,
        rows=rows, n_cols=gate_cols,
    )
    if use_halfrow:
        _dma_wait(out_stream_hbm.at[pl.ds(my_r0, half)], s_out_sem.at[0])
        _dma_wait(out_stream_hbm.at[pl.ds(mate_r0, half)], s_out_sem.at[1])
    else:
        _dma_wait(out_stream_hbm, s_out_sem.at[0])


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
    if rows < 32 or rows > 384 or rows % 32 != 0 or qkv_cols != 1280 or gate_cols != 1024:
        raise ValueError("Unsupported moe_to_attn_qkv_epi shapes")
    has_rope = cos_sin_step is not None
    cs_in = cos_sin_step[:rows] if (has_rope and cos_sin_step.shape[0] != rows) else cos_sin_step
    n_chips = world_size // 2
    half_rows = rows // 2
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

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
            interpret_mode=interpret,
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
            vmem((rows, hidden), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, qkv_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem((rows, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((2 if (rows >= 32 and rows % 32 == 0) else 1,)),
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
        _as_2d_f32(s_post, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_2d_f32(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_2d_f32(next_s_gate, gate_cols),
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
    """T(16, 128)-aligned owner-only HBM vocab gather inside Pallas (fixes #1494 & #1531 e2003!)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    pad_rows = partial_emb_buf.shape[0]
    v_shard, hidden = embed_hbm.shape
    v_start = me * jnp.int32(v_shard)
    v_end = v_start + jnp.int32(v_shard)
    max_r0 = jnp.int32((v_shard - 16) & ~15)
    row_iota_16 = lax.broadcasted_iota(jnp.int32, (16, _COL_STRIP), 0)

    def _grp_body(g0):
        for k in range(16):
            valid_k = (g0 + k) < actual_rows
            safe_idx = jnp.minimum(g0 + k, actual_rows - 1)
            tid = input_ids_smem[safe_idx]
            in_r = valid_k & (tid >= v_start) & (tid < v_end)
            loc_id = jnp.where(in_r, tid - v_start, jnp.int32(0))
            b_r0 = pl.multiple_of(
                jnp.minimum((loc_id >> jnp.int32(4)) << jnp.int32(4), max_r0), 16
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

        for k in range(16):
            valid_k = (g0 + k) < actual_rows
            safe_idx = jnp.minimum(g0 + k, actual_rows - 1)
            tid = input_ids_smem[safe_idx]
            in_r = valid_k & (tid >= v_start) & (tid < v_end)
            lax.cond(
                in_r,
                lambda kk=k: pltpu.make_async_copy(
                    slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                    slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                    row_sems.at[kk],
                ).wait(),
                lambda: None,
            )

        def _col_body(c0):
            c_sl = pl.ds(c0, _COL_STRIP)
            acc = jnp.zeros((16, _COL_STRIP), dtype=jnp.float32)
            for k in range(16):
                valid_k = (g0 + k) < actual_rows
                safe_idx = jnp.minimum(g0 + k, actual_rows - 1)
                tid = input_ids_smem[safe_idx]
                in_r = valid_k & (tid >= v_start) & (tid < v_end)
                loc_id = jnp.where(in_r, tid - v_start, jnp.int32(0))
                b_r0 = jnp.minimum((loc_id >> jnp.int32(4)) << jnp.int32(4), max_r0)
                sub_r = loc_id - b_r0
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

    _fori_strips(pad_rows, 16, _grp_body)


def _decode_embed_ar_norm_kernel(
    input_ids_smem,
    embed_hbm,
    out_stream_hbm,
    partial_emb_buf,
    slot_2d_buf,
    mate_buf,
    ici_buf,
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
    """In-kernel owner lookup + XOR=1 D2D + 1-shot ICI sum + RMSNorm -> ``stream [B, H] f32``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    actual_rows = out_stream_hbm.shape[0]
    pad_rows, hidden = partial_emb_buf.shape
    n_chips = world_size // 2
    bm = _pick_strip(pad_rows, 32)
    n_strips = hidden // _COL_STRIP

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
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

    c1 = pltpu.make_async_remote_copy(
        partial_emb_buf, mate_buf, d2d_send.at[0], d2d_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    c1.start()
    c1.wait()

    def _sum_mate(r0):
        rs = pl.ds(r0, bm)

        def _col(s, _):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            v = partial_emb_buf[rs, c_sl].astype(jnp.float32) + mate_buf[rs, c_sl].astype(jnp.float32)
            mate_buf[rs, c_sl] = v.astype(mate_buf.dtype)
            return jnp.int32(0)

        lax.fori_loop(0, n_strips, _col, jnp.int32(0))

    _fori_strips(pad_rows, bm, _sum_mate)

    c2 = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            mate_buf, ici_buf.at[d - 1], ici_send.at[d - 1], ici_recv.at[d - 1],
            device_id=(peer,), device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        c2.append(op)
    for op in c2:
        op.wait()

    def _sum_and_norm(r0):
        rs = pl.ds(r0, bm)

        def _sq_pass(s, acc):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            tot = mate_buf[rs, c_sl].astype(jnp.float32)
            for slot in range(n_chips - 1):
                tot = tot + ici_buf[slot, rs, c_sl].astype(jnp.float32)
            s_out_buf[rs, c_sl] = tot
            sq = tot * tot
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq_acc = lax.fori_loop(0, n_strips, _sq_pass, jnp.zeros((bm, 128), dtype=jnp.float32))
        inv = lax.rsqrt(
            jnp.sum(sq_acc, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(norm_eps)
        )

        def _norm_out(s, _):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            yn = (s_out_buf[rs, c_sl] * inv).astype(jnp.bfloat16).astype(jnp.float32)
            s_out_buf[rs, c_sl] = yn
            return jnp.int32(0)

        lax.fori_loop(0, n_strips, _norm_out, jnp.int32(0))

    _fori_strips(pad_rows, bm, _sum_and_norm)
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
        or v_shard % 8 != 0
        or hidden % _COL_STRIP != 0
    ):
        raise ValueError("Unsupported decode_vocab_embed_norm shapes")
    pad_rows = max(rows, 16)
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
            vmem((pad_rows, hidden), jnp.bfloat16),
            vmem((n_chips - 1, pad_rows, hidden), jnp.bfloat16),
            vmem((pad_rows, hidden), jnp.float32),
            sem((16,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=52 * 1024**2,
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
        or v_shard % 8 != 0
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
    max_tn = min(1024, max(_DOT_TN, (16 // max(bm // 8, 1)) * 128))
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
            sc = scale_ref[0:1, pl.ds(scale_c0 + cn, tn)]
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
        name=f"muse_w8_tok_b1_b{rows}_tp{world_size}_{'r16' if is_r16 else 'r32'}",
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


def _fold_cols_to_128(v):
    w = v.shape[1]
    while w > 128:
        h = w // 2
        v = v[:, 0:h] + v[:, h:w]
        w = h
    return v


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
    key = (int(world_size), bool(sharded_moe_in))
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
            t_specs = (P(), moe_sp, P(None, "model"), P(None, "model"), P(), P(), P(), P())
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
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=P(), check_vma=False
                )(stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta, final_gain)

            sfx = "_sh" if sharded_moe_in else ""
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
    if world_size != 8:
        raise ValueError(f"Fused QKV epilogue Bridge 2b requires world_size=8, got {world_size}")
    key = (int(world_size), bool(sharded_moe_in), bool(has_rope))
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
                P(None, "model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P(None, "model"),
                P(None, "model"),
                P(None, "model"),
            )
            out_specs = (
                P(),
                P(None, None, "model", None, None),
                P(None, "model", None, None),
                P(None, "model"),
            )
            sfx = f"{'_sh' if sharded_moe_in else ''}_{'rope' if has_rope else 'nope'}"

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
    key = (int(world_size), bool(bf16_router))
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
                P(None, "model"),
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
                        st, ga, wo, ws[0], a, b, pfg, wp, wps, wr, peg,
                        world_size=world_size,
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream_shard, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
                    pre_ff_gain, w_pre_pair_w8k, w_pre_pair_scale, w_router_km, pre_exp_gain,
                )

            tag = "r16" if bf16_router else "r32"
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8_tok_attn_to_moe_{tag}_tp{world_size}",
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
                P(None, "model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P(None, "model"),
                P(None, "model"),
                P(None, "model"),
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
                P(None, "model"),
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


def prewarm_w8_decode_bridge_ops(
    world_size: int,
    mode: int,
    *,
    bf16_router: bool = False,
    tok_sharded: bool = False,
    bridge_qkv_epi: bool = False,
    fuse_final_norm: bool = False,
    pallas_embed: bool = False,
) -> None:
    """Eagerly register Wave-8 custom ops outside ``torch.compile``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        return
    if pallas_embed:
        get_decode_vocab_embed_norm_op(world_size)
    if mode <= 0:
        return
    if fuse_final_norm and mode in (2, 3, 4, 5, 6):
        for sh in (False, True):
            get_moe_post_ag_final_norm_op(world_size, sharded_moe_in=sh)
    if bridge_qkv_epi and world_size == 8 and mode in (2, 3, 5):
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
