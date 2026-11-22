# SPDX-License-Identifier: Apache-2.0
"""Zero-spill HBM-under-ICI overlapped collective + FP8 projection bridges for Muse Spark.

Collapses all 15 non-attention / non-expert ops of a Muse Spark 816B decode layer into
two zero-spill Pallas megabridges that hide dense-projection and residual-stream HBM DMAs
directly underneath TPU7x ``XOR=1`` chip-hierarchical ICI collectives:

  1. ``attn_to_moe_bridge_jax`` (bridges ``RPAd_fast(onorm+gate)`` -> ``muse_sparse``):
     - Streams ``Wo_fp8 [O_in, H]`` chipmate-half-first so own-half MXU/HBM overlaps the
       Phase-1a ``733 GB/s`` ``XOR=1`` D2D exchange AND ``stream [B, H]`` HBM prefetch.
     - Prefetches ``W_pre_fp8 [H, 4096/W]`` and ``W_router [H, 256]`` from HBM during the
       Phase-1b/2a ``XOR=1`` inter-chip all-reduce (zero exposed weight-read stall!).
     - Overlaps Phase-2b D2D with 50% of ``post_attention_layernorm`` variance, updates
       ``stream`` (stored asynchronously to HBM), keeps ``router_input`` / ``ffn_hidden``
       purely in VMEM, computes ``FP8 pre_expert_proj``, overlaps its ``XOR=1`` all-gather
       under ``mlp.gate * router_multiplier``, and applies ``pre_expert_norm`` in VMEM.
     - Modular sub-mode ``attn_oproj_ar_res_norm_jax`` stops after ``pre_feedforward_ln``
       for standalone A/B or fallback when MoE pre-projection fusion is disabled.

  2. ``moe_to_attn_bridge_jax`` (bridges ``muse_sparse`` -> next layer's ``RPAd_fast``):
     - Overlaps ``W_post_fp8 [4096, H/W]`` and ``stream [B, H]`` HBM reads, computes
       ``FP8 post_expert_proj``, and prefetches ``8 MiB`` of next-layer ``W_qkv_fp8`` from
       HBM underneath the ``XOR=1`` half-row all-gather.
     - Hides Stage-3 D2D behind ``B/2`` rows of ``post_ffn_ln + gated_residual +
       next_input_ln``, stores ``new_stream`` asynchronously to HBM, keeps ``next_attn_in``
       in VMEM, and streams ``next W_qkv_fp8`` + ``next W_gate_fp8`` into TWO separate
       contiguous HBM outputs ``(next_qkv, next_gate)`` (zero slice/data-formatting copies).
     - Modular sub-modes: ``moe_post_ag_res_norm_jax`` (stops at ``next_attn_in``),
       ``moe_post_ag_res_only_jax`` (final layer L61), and ``fp8_dual_qkv_gate_jax``
       (layer-0 entry emitting contiguous ``(qkv, gate)`` in one ``attn_in`` read).
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
import os
from collections.abc import Callable
from threading import Lock

import torch

SUPPORTED_WORLD_SIZES = frozenset((8, 16))
_DEFAULT_CID = 31
_COL_STRIP = 256
_DOT_TN = 256
_QUANT_CHUNK = 4096

# Bitmask bits for SGLANG_TPU_MUSE_DECODE_BRIDGE_FP8_MASK:
FP8_BIT_O_PROJ = 1
FP8_BIT_PRE_EXPERT = 2
FP8_BIT_POST_EXPERT = 4
FP8_BIT_QKV_GATE = 8
FP8_BIT_ROUTER_BF16 = 16
FP8_MASK_ALL = (
    FP8_BIT_O_PROJ
    | FP8_BIT_PRE_EXPERT
    | FP8_BIT_POST_EXPERT
    | FP8_BIT_QKV_GATE
    | FP8_BIT_ROUTER_BF16
)

_OPS_LOCK = Lock()
_ATTN_TO_MOE_OPS: dict[tuple[int, bool, bool], Callable] = {}
_POST_ATTN_TO_MOE_OPS: dict[tuple[int, bool], Callable] = {}
_ATTN_OPROJ_AR_RN_OPS: dict[int, Callable] = {}
_MOE_TO_ATTN_QKVG_OPS: dict[tuple[int, bool], Callable] = {}
_MOE_POST_AG_RN_OPS: dict[tuple[int, bool], Callable] = {}
_MOE_POST_AG_RO_OPS: dict[tuple[int, bool], Callable] = {}
_DUAL_QKVG_OP: Callable | None = None


def quantize_weight_fp8_k_major(weight_nk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` weight to K-major ``float8_e4m3fn [K, N]`` + ``float32 [1, N]`` scale.

    ``N`` must be a multiple of 256 (holds for every Muse Spark projection on TP8 & TP16).
    Constructed from fresh contiguous allocations so Dynamo/AOT never re-materializes views.
    """
    n, k = weight_nk.shape
    if n % _DOT_TN != 0 or k % 128 != 0:
        raise ValueError(f"Expected (256*p, 128*q) weight, got {tuple(weight_nk.shape)}")
    wk = weight_nk.detach().t().contiguous()
    q_parts, s_parts = [], []
    for c0 in range(0, n, _QUANT_CHUNK):
        wf = wk[:, c0 : c0 + _QUANT_CHUNK].float()
        sc = (wf.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
        q_parts.append((wf / sc).to(torch.float8_e4m3fn))
        s_parts.append(sc.reshape(1, -1))
    w8k = q_parts[0].contiguous() if len(q_parts) == 1 else torch.cat(q_parts, dim=1).contiguous()
    scale = (
        s_parts[0].contiguous() if len(s_parts) == 1 else torch.cat(s_parts, dim=1).contiguous()
    )
    return w8k, scale


# =============================================================================
# Shared Zero-Spill Pallas Primitives (<= 16 live VREGs -> 0 / 0 LLO spills)
# =============================================================================
def _as_2d_f32_row(x, cols: int):
    import jax.numpy as jnp

    if x.ndim == 2 and x.shape == (1, cols) and x.dtype == jnp.float32:
        return x
    return x.reshape(1, cols).astype(jnp.float32)


def _signal_barrier(me, world_size: int) -> None:
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    barrier = pltpu.get_barrier_semaphore()
    mask = jnp.int32(world_size - 1)
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=((me + jnp.int32(offset)) & mask,),
            device_id_type=pl.DeviceIdType.MESH,
        )


def _wait_barrier(world_size: int) -> None:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    pl.semaphore_wait(pltpu.get_barrier_semaphore(), world_size - 1)


def _pick_strip(n_rows: int, cap: int = 32) -> int:
    bm = min(n_rows, cap)
    while bm > 8 and n_rows % bm != 0:
        bm //= 2
    return bm


def _use_dual_tree_xy(world_size: int, rows: int, hidden: int) -> bool:
    """Dual-Tree 2D-XY Butterfly needs 8 ranks, >=32 rows, and >=2 col strips per chip."""
    return world_size == 8 and rows >= 32 and (hidden // world_size) >= 2 * _COL_STRIP


def _fori_strips(n: int, step: int, body_fn) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    cnt = n // step

    def _loop(i, _):
        body_fn(pl.multiple_of(i * step, step))
        return jnp.int32(0)

    if cnt <= 2:
        for i in range(cnt):
            body_fn(pl.multiple_of(i * step, step))
    else:
        lax.fori_loop(0, cnt, _loop, jnp.int32(0))


def _dot_fp8_scaled_to_ref(
    x_ref,
    w_buf,
    scale_ref,
    scale_c0,
    dst_ref,
    dst_c0,
    *,
    rows: int,
    n_cols: int,
    w8a8_scale_ref=None,
) -> None:
    """Zero-spill ``(x_ref @ w_buf) * scale -> dst_ref[:, dst_c0:dst_c0+n_cols]``.

    Sub-tiles ``M`` by ``min(rows, 64)`` and ``N`` by ``256`` so each accumulator holds at
    most ``(64, 256) float32`` = 16 VREGs -> guaranteed ``0 / 0`` hardware LLO spills!
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    bm = _pick_strip(rows, 64)
    # Use tn=1024 (bm<=16) / tn=512 (bm==32) so B<=32 LC decodes cut x_ref VMEM re-reads 2-4x
    # while still keeping <= 16 f32 accumulator VREGs (0/0 LLO spills); bm==64 keeps _DOT_TN=256.
    max_tn = min(1024, max(_DOT_TN, (16 // max(bm // 8, 1)) * 128))
    tiles: list[tuple[int, int]] = []
    if max_tn > _DOT_TN and n_cols <= 2048:
        cn = 0
        while cn < n_cols:
            rem = n_cols - cn
            tn = max_tn
            while tn > _DOT_TN and rem < tn:
                tn //= 2
            tiles.append((cn, tn))
            cn += tn

    def _m_body(r0):
        rs = pl.ds(r0, bm)
        row_s = w8a8_scale_ref[rs, :] if w8a8_scale_ref is not None else None

        def _dot_tile(cn, tn: int):
            acc = jnp.dot(
                x_ref[rs, :],
                w_buf[:, pl.ds(cn, tn)],
                preferred_element_type=jnp.float32,
            )
            sc = scale_ref[0:1, pl.ds(scale_c0 + cn, tn)]
            out = acc * sc
            if row_s is not None:
                out = out * row_s
            dst_ref[rs, pl.ds(dst_c0 + cn, tn)] = out.astype(dst_ref.dtype)

        # Unroll up to 5 tiles (QKV=1280, Gate=1024, Post=1024, Pre=512) or 4-way inside
        # fori_loop (Wo half=4096 -> only 4 loop trips!) so Mosaic LLO back-to-back pipelines
        # MXU dots with VPU scale+bf16 pack while keeping <= 16 live f32 VREGs (0/0 spills!).
        if tiles:
            for cn, tn in tiles:
                _dot_tile(cn, tn)
        elif max_tn > _DOT_TN and n_cols % max_tn == 0:
            _fori_strips(n_cols, max_tn, lambda cn: _dot_tile(cn, max_tn))
        elif n_cols <= 5 * _DOT_TN:
            for cn in range(0, n_cols, _DOT_TN):
                _dot_tile(cn, _DOT_TN)
        elif n_cols % (4 * _DOT_TN) == 0:
            def _n_quad(c4):
                for sub in range(0, 4 * _DOT_TN, _DOT_TN):
                    _dot_tile(c4 + sub, _DOT_TN)

            _fori_strips(n_cols, 4 * _DOT_TN, _n_quad)
        else:
            _fori_strips(n_cols, _DOT_TN, lambda cn: _dot_tile(cn, _DOT_TN))

    _fori_strips(rows, bm, _m_body)


def _quantize_act_fp8_per_row(x_ref, x_fp8_buf, x_scale_buf) -> None:
    """Zero-spill per-row dynamic FP8 activation quantization ``x [B, K] -> (x_fp8, s [B, 1])``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    rows, k_cols = x_ref.shape
    bm = _pick_strip(rows, 32)
    n_k = k_cols // _COL_STRIP

    def _row_body(r0):
        rs = pl.ds(r0, bm)

        def _amax_step(s, cur):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            ax = jnp.abs(x_ref[rs, c_sl].astype(jnp.float32))
            return jnp.maximum(cur, jnp.maximum(ax[:, 0:128], ax[:, 128:256]))

        amax_128 = lax.fori_loop(0, n_k, _amax_step, jnp.zeros((bm, 128), dtype=jnp.float32))
        sc = jnp.maximum(jnp.max(amax_128, axis=-1, keepdims=True) * jnp.float32(1.0 / 448.0), jnp.float32(1e-12))
        inv_sc = jnp.float32(1.0) / sc
        x_scale_buf[rs, :] = sc

        def _q_step(s, _):
            c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            qf = x_ref[rs, c_sl].astype(jnp.float32) * inv_sc
            x_fp8_buf[rs, c_sl] = qf.astype(jnp.float8_e4m3fn)
            return jnp.int32(0)

        lax.fori_loop(0, n_k, _q_step, jnp.int32(0))

    _fori_strips(rows, bm, _row_body)


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


def _fori_col_blocks(n_cols: int, step_fn, *, unroll_k: int = 4) -> None:
    """Iterate ``0..n_cols`` in ``_COL_STRIP`` steps with ``unroll_k``-way loop unrolling."""
    from jax.experimental import pallas as pl

    blk = unroll_k * _COL_STRIP
    if n_cols >= blk and n_cols % blk == 0:
        def _blk(b0):
            for sub in range(unroll_k):
                step_fn(pl.ds(pl.multiple_of(b0 + sub * _COL_STRIP, _COL_STRIP), _COL_STRIP))

        _fori_strips(n_cols, blk, _blk)
        return

    _fori_strips(
        n_cols,
        _COL_STRIP,
        lambda c0: step_fn(pl.ds(c0, _COL_STRIP)),
    )


def _reduce_col_blocks(n_cols: int, step_fn, init, *, unroll_k: int = 4):
    """Accumulate across ``0..n_cols`` in ``unroll_k * _COL_STRIP`` loop bodies (0 spills)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    blk = unroll_k * _COL_STRIP
    if n_cols >= blk and n_cols % blk == 0:
        cnt = n_cols // blk

        def _blk_step(bi, acc):
            b0 = pl.multiple_of(bi * blk, blk)
            for sub in range(unroll_k):
                acc = step_fn(
                    pl.ds(pl.multiple_of(b0 + sub * _COL_STRIP, _COL_STRIP), _COL_STRIP),
                    acc,
                )
            return acc

        if cnt == 1:
            return _blk_step(jnp.int32(0), init)
        return lax.fori_loop(0, cnt, _blk_step, init)

    cnt = n_cols // _COL_STRIP

    def _s_step(si, acc):
        return step_fn(pl.ds(pl.multiple_of(si * _COL_STRIP, _COL_STRIP), _COL_STRIP), acc)

    if cnt == 1:
        return _s_step(jnp.int32(0), init)
    return lax.fori_loop(0, cnt, _s_step, init)


def _prescale_s_buf_in_vmem(s_buf, alpha_ref, *, rows: int, hidden: int, bm: int) -> None:
    """Pre-scale ``s_buf *= alpha_ref`` in-place (overlapped inside ICI wait windows)."""
    from jax.experimental import pallas as pl

    def _row(r0):
        rs = pl.ds(r0, bm)
        _fori_col_blocks(
            hidden,
            lambda c_sl: s_buf.__setitem__(
                (rs, c_sl), alpha_ref[0:1, c_sl] * s_buf[rs, c_sl]
            ),
        )

    _fori_strips(rows, bm, _row)


def _dma_wait(buf_ref, sem_ref) -> None:
    from jax.experimental.pallas import tpu as pltpu

    pltpu.make_async_copy(buf_ref, buf_ref, sem_ref).wait()


def _gather_shard_xor1(
    shard_ref,
    my_half_or_out,
    mate_half,
    s1_send,
    s1_recv,
    s2_send,
    s2_recv,
    s3_send,
    s3_recv,
    *,
    me,
    world_size: int,
    use_halfrow: bool,
    during_ici_fn=None,
    my_half_epilogue_fn=None,
    mate_half_epilogue_fn=None,
) -> None:
    """Zero-spill ``XOR=1`` column all-gather supporting overlapped compute under ICI & D2D!

    - When ``use_halfrow`` (``B >= 32, B % 32 == 0``): runs 3-stage half-row ``XOR=1`` AG
      (``-57%`` ICI), invokes ``during_ici_fn()`` while Stage-2 ICI DMAs fly, then starts
      Stage-3 D2D ``my_half -> mate_half`` overlapped with ``my_half_epilogue_fn(my_half, my_r0)``,
      waits Stage-3, and runs ``mate_half_epilogue_fn(mate_half, mate_r0)``.
    - When not ``use_halfrow`` (``B in {8, 16}``): runs zero-scratch 2-stage ``XOR=1`` AG
      directly into ``my_half_or_out [B, W*C]``, invokes ``during_ici_fn()`` under Stage-2 ICI,
      and runs ``my_half_epilogue_fn(my_half_or_out, 0)`` across all ``B`` rows.
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, cols = shard_ref.shape
    pair_cols = 2 * cols
    n_chips = world_size // 2
    mate = me ^ jnp.int32(1)
    chip = me >> jnp.int32(1)
    me_c0 = pl.multiple_of(me * cols, 128)
    chip_c0 = pl.multiple_of(chip * pair_cols, 128)

    if use_halfrow:
        half = rows // 2
        bm = _pick_strip(half, 32)
        parity = me & jnp.int32(1)
        mate_par = jnp.int32(1) - parity
        my_r0 = pl.multiple_of(parity * half, 16)
        mate_r0 = pl.multiple_of(mate_par * half, 16)

        s1 = pltpu.make_async_remote_copy(
            shard_ref.at[pl.ds(mate_r0, half), :],
            my_half_or_out.at[:, pl.ds(me_c0, cols)],
            s1_send.at[0],
            s1_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s1.start()
        _fori_strips(
            half,
            bm,
            lambda r0: my_half_or_out.__setitem__(
                (pl.ds(r0, bm), pl.ds(me_c0, cols)), shard_ref[pl.ds(my_r0 + r0, bm), :]
            ),
        )
        s1.wait()

        s2_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                my_half_or_out.at[:, pl.ds(chip_c0, pair_cols)],
                my_half_or_out.at[:, pl.ds(chip_c0, pair_cols)],
                s2_send.at[d - 1],
                s2_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)
        if during_ici_fn is not None:
            during_ici_fn()
        for op in s2_ops:
            op.wait()

        s3 = pltpu.make_async_remote_copy(
            my_half_or_out,
            mate_half,
            s3_send.at[0],
            s3_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s3.start()
        if my_half_epilogue_fn is not None:
            my_half_epilogue_fn(my_half_or_out, my_r0, half)
        s3.wait()
        if mate_half_epilogue_fn is not None:
            mate_half_epilogue_fn(mate_half, mate_r0, half)
        return

    # B in {8, 16}: 2-stage gather directly into my_half_or_out [rows, W*C].
    bm = _pick_strip(rows, 16)
    s1 = pltpu.make_async_remote_copy(
        shard_ref,
        my_half_or_out.at[:, pl.ds(me_c0, cols)],
        s1_send.at[0],
        s1_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    s1.start()
    _fori_strips(
        rows,
        bm,
        lambda r0: my_half_or_out.__setitem__(
            (pl.ds(r0, bm), pl.ds(me_c0, cols)), shard_ref[pl.ds(r0, bm), :]
        ),
    )
    s1.wait()

    s2_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            my_half_or_out.at[:, pl.ds(chip_c0, pair_cols)],
            my_half_or_out.at[:, pl.ds(chip_c0, pair_cols)],
            s2_send.at[d - 1],
            s2_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        s2_ops.append(op)
    if during_ici_fn is not None:
        during_ici_fn()
    for op in s2_ops:
        op.wait()
    if my_half_epilogue_fn is not None:
        my_half_epilogue_fn(my_half_or_out, jnp.int32(0), rows)


# =============================================================================
# 1. BRIDGE 1: `attn_to_moe_bridge_jax` & `attn_oproj_ar_res_norm_jax`
# =============================================================================
def _run_oproj_ar_and_post_attn_norm(
    stream_hbm,
    gated_ref,
    wo_hbm,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    out_stream_hbm,
    wo_buf,
    s_buf,
    r_out_buf,
    h_out_buf,
    send_half,
    mate_recv,
    own_half,
    scatter_ref,
    own_chunk,
    sq_buf,
    wo_sem,
    s_in_sem,
    s_out_sem,
    ar_s1a_send,
    ar_s1a_recv,
    ar_s1b_send,
    ar_s1b_recv,
    ar_s2a_send,
    ar_s2a_recv,
    ar_s2b_send,
    ar_s2b_recv,
    *,
    me,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    start_extra_prefetch_fn=None,
    x_fp8_buf=None,
    x_scale_buf=None,
    inv_pff_buf=None,
    defer_stream_store: bool = False,
) -> None:
    """Shared core: D2D-overlapped FP8 ``o_proj`` + ``XOR=1`` AR + ``pa_ln + res + pre_ffn_ln``.

    Updates ``s_buf [B, H] f32`` in-place (saving ``12 MiB`` VMEM at ``B=384``), leaves
    ``r_out_buf`` / ``h_out_buf [B, H] bf16`` live in VMEM, and kicks off an asynchronous
    HBM store of ``s_buf -> out_stream_hbm`` on ``s_out_sem`` unless ``defer_stream_store``.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    hidden = wo_hbm.shape[1]
    half = hidden // 2  # 4096 (single contiguous 4 MiB FP8 DMA per chipmate half!)
    chunk = hidden // world_size  # 1024 (TP8) or 512 (TP16)
    n_chips = world_size // 2
    bm = _pick_strip(rows, 32)
    bm64 = _pick_strip(rows, 64)
    half_strips = half // _COL_STRIP
    chunk_strips = chunk // _COL_STRIP

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)

    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)
    chip_c0 = pl.multiple_of(chip * chunk, 128)

    # Start BOTH 4 MiB Wo halves back-to-back: chipmate half [0] finishes first (2.9 us)!
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
    ).start()
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
    ).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()

    if x_fp8_buf is not None:
        _quantize_act_fp8_per_row(gated_ref, x_fp8_buf, x_scale_buf)
        lhs = x_fp8_buf
    else:
        lhs = gated_ref

    # Compute chipmate-half o_proj -> send_half [B, H/2] while own-half Wo + stream stream in!
    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    # Queue W_pre + W_router HBM prefetch right behind own-half Wo + stream so HBM stays 100%
    # saturated throughout own-half MXU + Phase-1a D2D + Phase-1b/2a ICI RS+AG!
    if start_extra_prefetch_fn is not None:
        start_extra_prefetch_fn()
    _dot_fp8_scaled_to_ref(
        lhs, wo_buf.at[0], wo_scale_ref, other_c0, send_half, 0,
        rows=rows, n_cols=half, w8a8_scale_ref=x_scale_buf,
    )

    # Ensure all peers are resident, then FIRE Phase-1a D2D overlapped with own-half o_proj!
    _wait_barrier(world_size)
    c1a = pltpu.make_async_remote_copy(
        send_half,
        mate_recv,
        ar_s1a_send.at[0],
        ar_s1a_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c1a.start()

    # Compute own-half o_proj -> own_half [B, H/2] 100% OVERLAPPED with Phase-1a D2D!
    _dma_wait(wo_buf.at[1], wo_sem.at[1])
    _dot_fp8_scaled_to_ref(
        lhs, wo_buf.at[1], wo_scale_ref, own_c0, own_half, 0,
        rows=rows, n_cols=half, w8a8_scale_ref=x_scale_buf,
    )

    # Wait Phase-1a D2D (already finished!) and sum own_half + mate_recv -> own_half.
    c1a.wait()

    def _sum_s1(r0):
        rs = pl.ds(r0, bm64)

        def _col_step(c_sl):
            v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
            own_half[rs, c_sl] = v.astype(own_half.dtype)

        _fori_col_blocks(half, _col_step)

    _fori_strips(rows, bm64, _sum_s1)

    # Phase 1b + 2a: Inter-chip reduce-scatter + all-gather across n_chips.
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    if use_xy:
        # Dual-Tree Bidirectional 2D-XY Butterfly on 2x2 torus (zero 2-hop diagonals, zero
        # 3-to-1 DMA incast, simultaneous X + Y link saturation on every step!):
        #   Tree A [0 : 2048] reduces X (me^2) then Y (me^4);
        #   Tree B [2048 : 4096] reduces Y (me^4) then X (me^2).
        e_cols = chunk // 2  # 512
        e_strips = e_cols // _COL_STRIP
        cx = chip & jnp.int32(1)
        cy = chip >> jnp.int32(1)
        peer_x = me ^ jnp.int32(2)
        peer_y = me ^ jnp.int32(4)

        a_send1 = pl.multiple_of((jnp.int32(1) - cx) * chunk, 128)
        a_keep1 = pl.multiple_of(cx * chunk, 128)
        b_send1 = pl.multiple_of(2 * chunk + (jnp.int32(1) - cy) * chunk, 128)
        b_keep1 = pl.multiple_of(2 * chunk + cy * chunk, 128)

        rs1_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_send1, chunk)], scatter_ref.at[0],
            ar_s1b_send.at[0], ar_s1b_recv.at[0],
            device_id=(peer_x,), device_id_type=pl.DeviceIdType.MESH,
        )
        rs1_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_send1, chunk)], scatter_ref.at[1],
            ar_s1b_send.at[1], ar_s1b_recv.at[1],
            device_id=(peer_y,), device_id_type=pl.DeviceIdType.MESH,
        )
        rs1_a.start()
        rs1_b.start()
        rs1_a.wait()
        rs1_b.wait()

        def _sum_xy1(r0):
            rs = pl.ds(r0, bm64)
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                c_l = pl.ds(off, _COL_STRIP)
                ca = pl.ds(pl.multiple_of(a_keep1 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_keep1 + off, _COL_STRIP), _COL_STRIP)
                va = own_half[rs, ca].astype(jnp.float32) + scatter_ref[0, rs, c_l].astype(jnp.float32)
                vb = own_half[rs, cb].astype(jnp.float32) + scatter_ref[1, rs, c_l].astype(jnp.float32)
                own_half[rs, ca] = va.astype(own_half.dtype)
                own_half[rs, cb] = vb.astype(own_half.dtype)

        _fori_strips(rows, bm64, _sum_xy1)

        a_send2 = pl.multiple_of(a_keep1 + (jnp.int32(1) - cy) * e_cols, 128)
        a_keep2 = pl.multiple_of(a_keep1 + cy * e_cols, 128)
        b_send2 = pl.multiple_of(b_keep1 + (jnp.int32(1) - cx) * e_cols, 128)
        b_keep2 = pl.multiple_of(b_keep1 + cx * e_cols, 128)

        rs2_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_send2, e_cols)], scatter_ref.at[0, :, pl.ds(0, e_cols)],
            ar_s1b_send.at[2], ar_s1b_recv.at[2],
            device_id=(peer_y,), device_id_type=pl.DeviceIdType.MESH,
        )
        rs2_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_send2, e_cols)], scatter_ref.at[1, :, pl.ds(0, e_cols)],
            ar_s1b_send.at[3], ar_s1b_recv.at[3],
            device_id=(peer_x,), device_id_type=pl.DeviceIdType.MESH,
        )
        rs2_a.start()
        rs2_b.start()
        rs2_a.wait()
        rs2_b.wait()

        def _sum_xy2(r0):
            rs = pl.ds(r0, bm64)
            for s in range(e_strips):
                off = s * _COL_STRIP
                c_l = pl.ds(off, _COL_STRIP)
                ca = pl.ds(pl.multiple_of(a_keep2 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_keep2 + off, _COL_STRIP), _COL_STRIP)
                va = own_half[rs, ca].astype(jnp.float32) + scatter_ref[0, rs, c_l].astype(jnp.float32)
                vb = own_half[rs, cb].astype(jnp.float32) + scatter_ref[1, rs, c_l].astype(jnp.float32)
                own_half[rs, ca] = va.astype(own_half.dtype)
                own_half[rs, cb] = vb.astype(own_half.dtype)

        _fori_strips(rows, bm64, _sum_xy2)

        # Dual-Tree AG-1 overlapped with s_buf *= pa_alpha (0-ULP bit-identical)!
        ag1_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_keep2, e_cols)], own_half.at[:, pl.ds(a_keep2, e_cols)],
            ar_s2a_send.at[0], ar_s2a_recv.at[0],
            device_id=(peer_y,), device_id_type=pl.DeviceIdType.MESH,
        )
        ag1_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_keep2, e_cols)], own_half.at[:, pl.ds(b_keep2, e_cols)],
            ar_s2a_send.at[1], ar_s2a_recv.at[1],
            device_id=(peer_x,), device_id_type=pl.DeviceIdType.MESH,
        )
        ag1_a.start()
        ag1_b.start()
        _dma_wait(s_buf, s_in_sem.at[0])
        _prescale_s_buf_in_vmem(s_buf, pa_alpha_ref, rows=rows, hidden=hidden, bm=bm64)
        ag1_a.wait()
        ag1_b.wait()

        # Dual-Tree AG-2 writes (a_other1, b_other1); (a_keep1, b_keep1) are already final,
        # so overlap the first 50% of _own_pa_sq directly underneath AG-2 ICI!
        a_other1 = pl.multiple_of((jnp.int32(1) - cx) * chunk, 128)
        b_other1 = pl.multiple_of(2 * chunk + (jnp.int32(1) - cy) * chunk, 128)
        ag2_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_keep1, chunk)], own_half.at[:, pl.ds(a_keep1, chunk)],
            ar_s2a_send.at[2], ar_s2a_recv.at[2],
            device_id=(peer_x,), device_id_type=pl.DeviceIdType.MESH,
        )
        ag2_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_keep1, chunk)], own_half.at[:, pl.ds(b_keep1, chunk)],
            ar_s2a_send.at[3], ar_s2a_recv.at[3],
            device_id=(peer_y,), device_id_type=pl.DeviceIdType.MESH,
        )
        ag2_a.start()
        ag2_b.start()

        def _own_pa_sq_keep1(r0):
            rs = pl.ds(r0, bm64)
            acc = jnp.zeros((bm64, 128), dtype=jnp.float32)
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                ca = pl.ds(pl.multiple_of(a_keep1 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_keep1 + off, _COL_STRIP), _COL_STRIP)
                fa = own_half[rs, ca].astype(jnp.float32)
                fb = own_half[rs, cb].astype(jnp.float32)
                sq = fa * fa + fb * fb
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            sq_buf[rs, :] = acc

        _fori_strips(rows, bm64, _own_pa_sq_keep1)
        ag2_a.wait()
        ag2_b.wait()
    else:
        # Small-B / TP16 latency-optimal 1-shot symmetric XOR RS + AG.
        c1b = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            peer_chip = chip ^ jnp.int32(d)
            op = pltpu.make_async_remote_copy(
                own_half.at[:, pl.ds(pl.multiple_of(peer_chip * chunk, 128), chunk)],
                scatter_ref.at[d - 1],
                ar_s1b_send.at[d - 1],
                ar_s1b_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            c1b.append(op)
        for op in c1b:
            op.wait()

        def _sum_s2(r0):
            rs = pl.ds(r0, bm64)
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                c_l = pl.ds(pl.multiple_of(off, _COL_STRIP), _COL_STRIP)
                c_h = pl.ds(pl.multiple_of(chip_c0 + off, _COL_STRIP), _COL_STRIP)
                tot = own_half[rs, c_h].astype(jnp.float32)
                for slot in range(n_chips - 1):
                    tot = tot + scatter_ref[slot, rs, c_l].astype(jnp.float32)
                vb = tot.astype(own_chunk.dtype)
                own_chunk[rs, c_l] = vb
                own_half[rs, c_h] = vb

        _fori_strips(rows, bm64, _sum_s2)

        c2a = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                own_chunk,
                own_half.at[:, pl.ds(chip_c0, chunk)],
                ar_s2a_send.at[d - 1],
                ar_s2a_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            c2a.append(op)
        _dma_wait(s_buf, s_in_sem.at[0])
        _prescale_s_buf_in_vmem(s_buf, pa_alpha_ref, rows=rows, hidden=hidden, bm=bm64)
        for op in c2a:
            op.wait()

    # Phase 2b: Single [B, H/2] D2D AG into mate_recv OVERLAPPED with remaining own_half sum(b^2)!
    c2b = pltpu.make_async_remote_copy(
        own_half,
        mate_recv,
        ar_s2b_send.at[0],
        ar_s2b_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c2b.start()

    if use_xy:
        def _own_pa_sq_rest(r0):
            rs = pl.ds(r0, bm64)
            acc = sq_buf[rs, :]
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                ca = pl.ds(pl.multiple_of(a_other1 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_other1 + off, _COL_STRIP), _COL_STRIP)
                fa = own_half[rs, ca].astype(jnp.float32)
                fb = own_half[rs, cb].astype(jnp.float32)
                sq = fa * fa + fb * fb
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            sq_buf[rs, :] = acc

        _fori_strips(rows, bm64, _own_pa_sq_rest)
    else:
        def _own_pa_sq(r0):
            rs = pl.ds(r0, bm64)

            def _col(c_sl, acc):
                bf = own_half[rs, c_sl].astype(jnp.float32)
                sq = bf * bf
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            sq_buf[rs, :] = _reduce_col_blocks(
                half, _col, jnp.zeros((bm64, 128), dtype=jnp.float32)
            )

        _fori_strips(rows, bm64, _own_pa_sq)
    c2b.wait()

    # Finish other-half post_attn_ln variance + gated residual + pre_feedforward_ln in VMEM!
    def _finish_pa_and_pre_ffn(r0):
        rs = pl.ds(r0, bm)

        def _oth_sq(c_sl, acc):
            bf = mate_recv[rs, c_sl].astype(jnp.float32)
            sq = bf * bf
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        oth_sq = _reduce_col_blocks(
            half, _oth_sq, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        # IEEE-754 f32 addition is strictly commutative: own_sq + oth_sq == oth_sq + own_sq
        # bit-for-bit across even (side=0) and odd (side=1) chipmate cores!
        pa_sq = sq_buf[rs, :] + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )

        def _res_both_halves(c_l, acc):
            c_own = pl.ds(pl.multiple_of(own_c0 + c_l.start, _COL_STRIP), _COL_STRIP)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c_l.start, _COL_STRIP), _COL_STRIP)

            ns_own = s_buf[rs, c_own] + pa_beta_ref[0:1, c_own] * (
                own_half[rs, c_l].astype(jnp.float32) * inv_pa
            )
            ns_oth = s_buf[rs, c_oth] + pa_beta_ref[0:1, c_oth] * (
                mate_recv[rs, c_l].astype(jnp.float32) * inv_pa
            )
            s_buf[rs, c_own] = ns_own
            s_buf[rs, c_oth] = ns_oth

            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            sq = v_own * v_own + v_oth * v_oth
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        pff_sq = _reduce_col_blocks(
            half, _res_both_halves, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        if inv_pff_buf is not None:
            inv_pff_buf[rs, :] = inv_pff

        def _pre_ffn_out(c_g):
            sv = s_buf[rs, c_g]
            gn = pre_ff_gain_ref[0:1, c_g]
            rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
            if r_out_buf is not None:
                r_out_buf[rs, c_g] = rn
            h_out_buf[rs, c_g] = rn.astype(h_out_buf.dtype)

        _fori_col_blocks(hidden, _pre_ffn_out)

    _fori_strips(rows, bm, _finish_pa_and_pre_ffn)
    if not defer_stream_store:
        # Stream updated residual out to HBM in the background while MoE pre-proj runs!
        pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()


def _attn_oproj_ar_res_norm_kernel(
    stream_hbm,
    gated_ref,
    wo_hbm,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    out_stream_hbm,
    out_router_ref,
    out_ffn_ref,
    wo_buf,
    s_buf,
    send_half,
    mate_recv,
    own_half,
    scatter_ref,
    own_chunk,
    sq_buf,
    wo_sem,
    s_in_sem,
    s_out_sem,
    ar_s1a_send,
    ar_s1a_recv,
    ar_s1b_send,
    ar_s1b_recv,
    ar_s2a_send,
    ar_s2a_recv,
    ar_s2b_send,
    ar_s2b_recv,
    *,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
) -> None:
    import jax.numpy as jnp
    from jax import lax

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    _run_oproj_ar_and_post_attn_norm(
        stream_hbm, gated_ref, wo_hbm, wo_scale_ref, pa_alpha_ref, pa_beta_ref,
        pre_ff_gain_ref, out_stream_hbm, wo_buf, s_buf, out_router_ref,
        out_ffn_ref, send_half, mate_recv, own_half, scatter_ref, own_chunk, sq_buf,
        wo_sem, s_in_sem, s_out_sem, ar_s1a_send, ar_s1a_recv, ar_s1b_send, ar_s1b_recv,
        ar_s2a_send, ar_s2a_recv, ar_s2b_send, ar_s2b_recv,
        me=me, world_size=world_size, pa_eps=pa_eps, pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
    )
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def attn_oproj_ar_res_norm_jax(
    stream,
    gated_attn,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    *,
    world_size: int,
    pa_eps: float = 1e-5,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    collective_id: int = _DEFAULT_CID,
):
    """Modular Sub-Bridge 1a: ``FP8 o_proj + XOR=1 AR + pa_ln + res + pre_ffn_ln``.

    Consumes already-onorm-gated ``gated_attn [B, H/W]`` from ``RPAd_fast-...-onorm-gate``
    and returns ``(new_stream_f32 [B, H], router_input_f32 [B, H], ffn_hidden_bf16 [B, H])``.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    rows, o_in = gated_attn.shape
    wo_k, hidden = wo_w8k.shape
    if rows < 8 or rows > 384 or rows % 8 != 0 or wo_k != o_in or hidden % (256 * world_size) != 0:
        raise ValueError(f"Unsupported shapes gated={gated_attn.shape}, wo={wo_w8k.shape}")
    half = hidden // 2
    chunk = hidden // world_size
    n_chips = world_size // 2
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_ar_sems = 4 if use_xy else (n_chips - 1)
    n_sc_slots = 2 if use_xy else (n_chips - 1)

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_sp = pl.BlockSpec(memory_space=vmem)
    h_sp = pl.BlockSpec(memory_space=hbm)

    return pl.pallas_call(
        functools.partial(
            _attn_oproj_ar_res_norm_kernel,
            world_size=world_size,
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((n_sc_slots, rows, chunk), jnp.bfloat16),
            vmem((8, 128) if use_xy else (rows, chunk), jnp.bfloat16),
            vmem((rows, 128), jnp.float32),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_ar_sems,)),
            sem((n_ar_sems,)),
            sem((n_ar_sems,)),
            sem((n_ar_sems,)),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(58 if rows > 256 else 56) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w7_oproj_ar_rn_b{rows}_tp{world_size}",
    )(
        stream.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_2d_f32_row(wo_scale, hidden),
        _as_2d_f32_row(pa_alpha, hidden),
        _as_2d_f32_row(pa_beta, hidden),
        _as_2d_f32_row(pre_ff_gain, hidden),
    )


def _norm_pre_expert_slice(
    src_buf,
    pre_exp_gain_ref,
    out_expert_in_ref,
    *,
    h_pre: int,
    row_base,
    n_r: int,
    align: int,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
) -> None:
    """Zero-spill pre_expert RMSNorm using unrolled 1024-col strips."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    bm_n = _pick_strip(n_r, 32)

    def _strip(r_loc):
        rs_l = pl.ds(r_loc, bm_n)
        rs_g = pl.ds(pl.multiple_of(row_base + r_loc, align), bm_n)

        def _v_step(c_sl, acc):
            xf = src_buf[rs_l, c_sl].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_sl]
            sq = xf * xf
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq_s = _reduce_col_blocks(
            h_pre, _v_step, jnp.zeros((bm_n, 128), dtype=jnp.float32)
        )
        inv = lax.rsqrt(
            jnp.sum(sq_s, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
            + jnp.float32(pre_exp_eps)
        )

        def _o_step(c_sl):
            xf = src_buf[rs_l, c_sl].astype(jnp.float32)
            g = pre_exp_gain_ref[0:1, c_sl]
            yn = (xf * g) * inv if pre_exp_scale_input else (xf * inv) * g
            out_expert_in_ref[rs_g, c_sl] = yn.astype(out_expert_in_ref.dtype)

        _fori_col_blocks(h_pre, _o_step)

    _fori_strips(n_r, bm_n, _strip)


def _attn_to_moe_bridge_kernel(
    stream_hbm,
    gated_ref,
    wo_hbm,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_hbm,
    w_pre_scale_ref,
    w_router_hbm,
    pre_exp_gain_ref,
    out_stream_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
    wo_buf,
    w_pre_buf,
    w_router_buf,
    s_buf,
    router_vmem,
    inv_pff_buf,
    ffn_vmem,
    send_half,
    mate_recv,
    own_half,
    scatter_ref,
    own_chunk,
    sq_buf,
    pre_shard,
    pre_half_a,
    pre_half_b,
    x_fp8_buf,
    x_scale_buf,
    wo_sem,
    w_pre_sem,
    w_router_sem,
    s_in_sem,
    s_out_sem,
    ar_s1a_send,
    ar_s1a_recv,
    ar_s1b_send,
    ar_s1b_recv,
    ar_s2a_send,
    ar_s2a_recv,
    ar_s2b_send,
    ar_s2b_recv,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    *,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    use_w8a8: bool,
    router_d2d_overlap: bool = False,
) -> None:
    """10-op zero-spill megabridge from ``RPAd_fast(onorm+gate)`` straight into ``muse_sparse``!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    pre_cols = w_pre_hbm.shape[1]
    h_pre = world_size * pre_cols
    n_exp = w_router_hbm.shape[1]
    use_halfrow = rows >= 32 and rows % 32 == 0
    pre_buf_rows = rows // 2 if use_halfrow else rows
    is_r16 = w_router_buf.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    # Prefetch BOTH W_pre (4 MiB FP8) and W_router (4 MiB BF16 / 8 MiB FP32) from HBM
    # right underneath the 7.5 us o_proj ICI reduce-scatter + all-gather -> 0.0 us wait!
    def _prefetch_moe_weights():
        pltpu.make_async_copy(w_pre_hbm, w_pre_buf, w_pre_sem.at[0]).start()
        pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    r_buf_arg = None if (is_r16 or tile_r32) else router_vmem
    _run_oproj_ar_and_post_attn_norm(
        stream_hbm, gated_ref, wo_hbm, wo_scale_ref, pa_alpha_ref, pa_beta_ref,
        pre_ff_gain_ref, out_stream_hbm, wo_buf, s_buf, r_buf_arg,
        ffn_vmem, send_half, mate_recv, own_half, scatter_ref, own_chunk, sq_buf,
        wo_sem, s_in_sem, s_out_sem, ar_s1a_send, ar_s1a_recv, ar_s1b_send, ar_s1b_recv,
        ar_s2a_send, ar_s2a_recv, ar_s2b_send, ar_s2b_recv,
        me=me, world_size=world_size, pa_eps=pa_eps, pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        start_extra_prefetch_fn=_prefetch_moe_weights,
        x_fp8_buf=x_fp8_buf if use_w8a8 else None,
        x_scale_buf=x_scale_buf if use_w8a8 else None,
        inv_pff_buf=inv_pff_buf if tile_r32 else None,
        defer_stream_store=tile_r32,
    )

    # W_pre is ALREADY resident in VMEM! Run FP8 pre_expert_proj [B, 8192] @ [8192, 512]!
    _dma_wait(w_pre_buf, w_pre_sem.at[0])
    _dot_fp8_scaled_to_ref(
        ffn_vmem, w_pre_buf, w_pre_scale_ref, 0, pre_shard, 0,
        rows=rows, n_cols=pre_cols,
    )

    # Compute router logits (router_vmem @ w_router) * router_multiplier RIGHT UNDERNEATH
    # the pre_expert XOR=1 Stage-2 inter-chip ICI all-gather!
    def _compute_router_under_ici():
        _dma_wait(w_router_buf, w_router_sem.at[0])
        bm_g = _pick_strip(rows, 64)
        rmul = jnp.float32(router_multiplier)

        def _r_step(r0):
            rs = pl.ds(r0, bm_g)
            if tile_r32:
                inv_r = inv_pff_buf[rs, :]

                def _fill_rtile(s, _):
                    c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    sv = s_buf[rs, c_g]
                    gn = pre_ff_gain_ref[0:1, c_g]
                    router_vmem[:, c_g] = (sv * gn) * inv_r if pre_ff_scale_input else (sv * inv_r) * gn
                    return jnp.int32(0)

                lax.fori_loop(0, hidden // _COL_STRIP, _fill_rtile, jnp.int32(0))
                lhs_slice = router_vmem[:, :]
            else:
                r_lhs = ffn_vmem if is_r16 else router_vmem
                lhs_slice = r_lhs[rs, :]

            def _e_step(ce):
                c_sl = pl.ds(ce, _DOT_TN)
                lg = jnp.dot(lhs_slice, w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
                out_router_logits_ref[rs, c_sl] = lg * rmul

            _fori_strips(n_exp, _DOT_TN, _e_step)

        _fori_strips(rows, bm_g, _r_step)
        if tile_r32:
            pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()

    # Zero-spill pre_expert_norm overlapped behind Stage-3 D2D!
    align = 16 if use_halfrow else 8

    def _norm_pre_slice(src_buf, row_base, n_r):
        _norm_pre_expert_slice(
            src_buf, pre_exp_gain_ref, out_expert_in_ref,
            h_pre=h_pre, row_base=row_base, n_r=n_r, align=align,
            pre_exp_eps=pre_exp_eps, pre_exp_scale_input=pre_exp_scale_input,
        )

    # Reuse dead send_half VMEM buffer when h_pre == half (saves 3 MiB @ B=384!).
    eff_pre_a = send_half.at[pl.ds(0, pre_buf_rows), :] if h_pre == half else pre_half_a

    # Note: no second barrier needed! Every rank already synchronized at Phase 2b c2b.wait()
    # and all ranks enter Stage 1 with pre_half_a live from kernel start!
    if use_halfrow:
        half_r = rows // 2
        bm_s1 = _pick_strip(half_r, 32)
        parity = me & jnp.int32(1)
        mate = me ^ jnp.int32(1)
        chip = me >> jnp.int32(1)
        n_chips = world_size // 2
        me_c0 = pl.multiple_of(me * pre_cols, 128)
        pair_cols = 2 * pre_cols
        chip_c0 = pl.multiple_of(chip * pair_cols, 128)
        my_r0 = pl.multiple_of(parity * half_r, 16)
        mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half_r, 16)

        s1 = pltpu.make_async_remote_copy(
            pre_shard.at[pl.ds(mate_r0, half_r), :],
            eff_pre_a.at[:, pl.ds(me_c0, pre_cols)],
            ag_s1_send.at[0],
            ag_s1_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s1.start()
        _fori_strips(
            half_r,
            bm_s1,
            lambda r0: eff_pre_a.__setitem__(
                (pl.ds(r0, bm_s1), pl.ds(me_c0, pre_cols)), pre_shard[pl.ds(my_r0 + r0, bm_s1), :]
            ),
        )
        s1.wait()

        s2_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                eff_pre_a.at[:, pl.ds(chip_c0, pair_cols)],
                eff_pre_a.at[:, pl.ds(chip_c0, pair_cols)],
                ag_s2_send.at[d - 1],
                ag_s2_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)
        overlap_router_d2d = bool(router_d2d_overlap) and not tile_r32
        if not overlap_router_d2d:
            _compute_router_under_ici()
        for op in s2_ops:
            op.wait()

        # Normalize ONLY my_half into out_expert_in_ref[my_r0:my_r0+half_r], then Stage-3 D2D
        # copies normalized bf16 straight to mate (halves pre_expert_norm VPU rows!).
        if overlap_router_d2d:
            _compute_router_under_ici()
        _norm_pre_slice(eff_pre_a, my_r0, half_r)
        s3 = pltpu.make_async_remote_copy(
            out_expert_in_ref.at[pl.ds(my_r0, half_r), :],
            out_expert_in_ref.at[pl.ds(my_r0, half_r), :],
            ag_s3_send.at[0],
            ag_s3_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s3.start()
        _dma_wait(out_stream_hbm, s_out_sem.at[0])
        s3.wait()
    else:
        eff_pre_b = mate_recv.at[pl.ds(0, pre_buf_rows), :] if h_pre == half else pre_half_b
        _gather_shard_xor1(
            pre_shard,
            eff_pre_a,
            eff_pre_b,
            ag_s1_send,
            ag_s1_recv,
            ag_s2_send,
            ag_s2_recv,
            ag_s3_send,
            ag_s3_recv,
            me=me,
            world_size=world_size,
            use_halfrow=False,
            during_ici_fn=_compute_router_under_ici,
            my_half_epilogue_fn=_norm_pre_slice,
            mate_half_epilogue_fn=_norm_pre_slice,
        )
        _dma_wait(out_stream_hbm, s_out_sem.at[0])


def attn_to_moe_bridge_jax(
    stream,
    gated_attn,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_w8k,
    w_pre_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int,
    pa_eps: float = 1e-5,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    use_w8a8: bool = False,
    router_d2d_overlap: bool | None = None,
    interpret: bool = False,
    collective_id: int = _DEFAULT_CID + 1,
):
    """Full 10-op Bridge 1b: ``RPAd_fast(onorm+gate) -> (new_stream, router_logits, expert_in)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    rows, o_in = gated_attn.shape
    wo_k, hidden = wo_w8k.shape
    wp_k, pre_cols = w_pre_w8k.shape
    wr_k, n_exp = w_router_km.shape
    h_pre = world_size * pre_cols
    if (
        rows < 8
        or rows > 384
        or rows % 8 != 0
        or wo_k != o_in
        or wp_k != hidden
        or wr_k != hidden
        or pre_cols % _DOT_TN != 0
        or n_exp % _DOT_TN != 0
    ):
        raise ValueError(
            f"Unsupported attn_to_moe shapes gated={gated_attn.shape}, wo={wo_w8k.shape}, "
            f"w_pre={w_pre_w8k.shape}, w_router={w_router_km.shape}"
        )
    half = hidden // 2
    chunk = hidden // world_size
    n_chips = world_size // 2
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_ar_sems = 4 if use_xy else (n_chips - 1)
    n_sc_slots = 2 if use_xy else (n_chips - 1)
    is_r16 = w_router_km.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64
    bm_g = _pick_strip(rows, 64)
    use_halfrow = rows >= 32 and rows % 32 == 0
    pre_buf_rows = rows // 2 if use_halfrow else rows
    reuse_pre_ab = h_pre == half

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_sp = pl.BlockSpec(memory_space=vmem)
    h_sp = pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _attn_to_moe_bridge_kernel,
            world_size=world_size,
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            use_w8a8=bool(use_w8a8),
            router_d2d_overlap=(
                (os.environ.get("SGLANG_TPU_MUSE_BRIDGE1_ROUTER_D2D_OVERLAP", "0") == "1")
                if router_d2d_overlap is None
                else bool(router_d2d_overlap)
            ),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
            jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((hidden, pre_cols), w_pre_w8k.dtype),
            vmem((hidden, n_exp), w_router_km.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((8, 128) if is_r16 else ((bm_g, hidden) if tile_r32 else (rows, hidden)), jnp.float32),
            vmem((rows, 1) if tile_r32 else (8, 1), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((n_sc_slots, rows, chunk), jnp.bfloat16),
            vmem((8, 128) if use_xy else (rows, chunk), jnp.bfloat16),
            vmem((rows, 128), jnp.float32),
            vmem((rows, pre_cols), jnp.bfloat16),
            vmem((8, 128) if reuse_pre_ab else (pre_buf_rows, h_pre), jnp.bfloat16),
            vmem((8, 128) if (reuse_pre_ab or use_halfrow) else (pre_buf_rows, h_pre), jnp.bfloat16),
            vmem((rows, o_in) if use_w8a8 else (8, 128), jnp.float8_e4m3fn),
            vmem((rows, 1) if use_w8a8 else (8, 1), jnp.float32),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_ar_sems,)),
            sem((n_ar_sems,)),
            sem((n_ar_sems,)),
            sem((n_ar_sems,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=58 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w7_attn_to_moe_b{rows}_tp{world_size}"
            f"_{'r16' if w_router_km.dtype == jnp.bfloat16 else 'r32'}"
            f"{'_a8' if use_w8a8 else ''}"
        ),
    )(
        stream.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_2d_f32_row(wo_scale, hidden),
        _as_2d_f32_row(pa_alpha, hidden),
        _as_2d_f32_row(pa_beta, hidden),
        _as_2d_f32_row(pre_ff_gain, hidden),
        w_pre_w8k,
        _as_2d_f32_row(w_pre_scale, pre_cols),
        w_router_km,
        _as_2d_f32_row(pre_exp_gain, h_pre),
    )


# =============================================================================
# 2. BRIDGE 2: `moe_to_attn_bridge_jax`, `moe_post_ag_res_norm_jax`,
#              `moe_post_ag_res_only_jax`, & L0 `fp8_dual_qkv_gate_jax`
# =============================================================================
def _run_post_ffn_slice_epilogue(
    s_in_buf,
    src_buf,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    s_out_buf,
    next_in_buf,
    *,
    hidden: int,
    row_base,
    n_r: int,
    align: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    with_next_norm: bool,
    pre_scaled_alpha: bool = False,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    bm = _pick_strip(n_r, 32)

    def _strip(r_loc):
        rs_l = pl.ds(r_loc, bm)
        rs_g = pl.ds(pl.multiple_of(row_base + r_loc, align), bm)

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

        if with_next_norm:
            def _res_nvar(c_sl, acc):
                bc = src_buf[rs_l, c_sl].astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_sl]
                b_n = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                s_base = s_in_buf[rs_g, c_sl] if pre_scaled_alpha else (pf_alpha_ref[0:1, c_sl] * s_in_buf[rs_g, c_sl])
                ns = s_base + pf_beta_ref[0:1, c_sl] * b_n
                s_out_buf[rs_g, c_sl] = ns
                nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
                sq = nv * nv
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            n_sq = _reduce_col_blocks(
                hidden, _res_nvar, jnp.zeros((bm, 128), dtype=jnp.float32)
            )
            inv_n = lax.rsqrt(
                jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(next_in_eps)
            )

            def _nout(c_sl):
                sv = s_out_buf[rs_g, c_sl]
                ng = next_in_gain_ref[0:1, c_sl]
                nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
                next_in_buf[rs_g, c_sl] = nn.astype(next_in_buf.dtype)

            _fori_col_blocks(hidden, _nout)
        else:
            def _ro(c_sl):
                bc = src_buf[rs_l, c_sl].astype(jnp.float32)
                bg = post_ff_gain_ref[0:1, c_sl]
                b_n = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
                s_base = s_in_buf[rs_g, c_sl] if pre_scaled_alpha else (pf_alpha_ref[0:1, c_sl] * s_in_buf[rs_g, c_sl])
                s_out_buf[rs_g, c_sl] = s_base + pf_beta_ref[0:1, c_sl] * b_n

            _fori_col_blocks(hidden, _ro)

    _fori_strips(n_r, bm, _strip)


def _post_attn_to_moe_bridge_kernel(
    stream_hbm,
    attn_out_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_hbm,
    w_pre_scale_ref,
    w_router_hbm,
    pre_exp_gain_ref,
    out_stream_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
    w_pre_buf,
    w_router_buf,
    s_buf,
    router_vmem,
    ffn_vmem,
    inv_pa_buf,
    pre_shard,
    pre_half_a,
    pre_half_b,
    w_pre_sem,
    w_router_sem,
    s_in_sem,
    s_out_sem,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    *,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
) -> None:
    """Zero-spill 7-op Sub-Bridge 1c: ``attn_out -> (new_stream, router_logits, expert_in)``!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, hidden = attn_out_ref.shape
    pre_cols = w_pre_hbm.shape[1]
    h_pre = world_size * pre_cols
    n_exp = w_router_hbm.shape[1]
    is_r16 = w_router_buf.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64
    use_halfrow = rows >= 32 and rows % 32 == 0
    bm = _pick_strip(rows, 32)
    h_strips = hidden // _COL_STRIP
    pre_strips = h_pre // _COL_STRIP

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    # Saturate HBM from cycle 0: stream_in -> W_pre -> W_router!
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_pre_hbm, w_pre_buf, w_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    # Compute post_attn_ln inverse-RMS from attn_out_ref WHILE stream_in DMA completes!
    def _pa_var(r0):
        rs = pl.ds(r0, bm)

        def _col(s, acc):
            c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            bf = attn_out_ref[rs, c_g].astype(jnp.float32)
            sq = bf * bf
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        sq = lax.fori_loop(0, h_strips, _col, jnp.zeros((bm, 128), dtype=jnp.float32))
        inv_pa_buf[rs, :] = lax.rsqrt(
            jnp.sum(sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden) + jnp.float32(pa_eps)
        )

    _fori_strips(rows, bm, _pa_var)
    _dma_wait(s_buf, s_in_sem.at[0])

    # Gated residual + pre_ffn_ln purely in VMEM while W_pre + W_router finish streaming in!
    def _res_and_pre_ffn(r0):
        rs = pl.ds(r0, bm)
        inv_pa = inv_pa_buf[rs, :]

        def _res(s, acc):
            c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            ns = (
                pa_alpha_ref[0:1, c_g] * s_buf[rs, c_g]
                + pa_beta_ref[0:1, c_g] * (attn_out_ref[rs, c_g].astype(jnp.float32) * inv_pa)
            )
            s_buf[rs, c_g] = ns
            vf = ns * pre_ff_gain_ref[0:1, c_g] if pre_ff_scale_input else ns
            sq = vf * vf
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        pff_sq = lax.fori_loop(0, h_strips, _res, jnp.zeros((bm, 128), dtype=jnp.float32))
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        if tile_r32:
            inv_pa_buf[rs, :] = inv_pff

        def _out(s, _):
            c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
            sv = s_buf[rs, c_g]
            gn = pre_ff_gain_ref[0:1, c_g]
            rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
            if (not is_r16) and (not tile_r32):
                router_vmem[rs, c_g] = rn
            ffn_vmem[rs, c_g] = rn.astype(ffn_vmem.dtype)
            return jnp.int32(0)

        lax.fori_loop(0, h_strips, _out, jnp.int32(0))

    _fori_strips(rows, bm, _res_and_pre_ffn)
    if not tile_r32:
        pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()

    _dma_wait(w_pre_buf, w_pre_sem.at[0])
    _dot_fp8_scaled_to_ref(
        ffn_vmem, w_pre_buf, w_pre_scale_ref, 0, pre_shard, 0,
        rows=rows, n_cols=pre_cols,
    )
    _wait_barrier(world_size)

    def _router_under_ici():
        _dma_wait(w_router_buf, w_router_sem.at[0])
        bm_g = _pick_strip(rows, 64)
        rmul = jnp.float32(router_multiplier)

        def _r_step(r0):
            rs = pl.ds(r0, bm_g)
            if tile_r32:
                inv_r = inv_pa_buf[rs, :]

                def _fill_rt(s, _):
                    c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    sv = s_buf[rs, c_g]
                    gn = pre_ff_gain_ref[0:1, c_g]
                    router_vmem[:, c_g] = (sv * gn) * inv_r if pre_ff_scale_input else (sv * inv_r) * gn
                    return jnp.int32(0)

                lax.fori_loop(0, h_strips, _fill_rt, jnp.int32(0))
                lhs_slice = router_vmem[:, :]
            else:
                r_lhs = ffn_vmem if is_r16 else router_vmem
                lhs_slice = r_lhs[rs, :]
            for ce in range(0, n_exp, _DOT_TN):
                c_sl = pl.ds(ce, _DOT_TN)
                lg = jnp.dot(lhs_slice, w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
                out_router_logits_ref[rs, c_sl] = lg * rmul

        _fori_strips(rows, bm_g, _r_step)
        if tile_r32:
            pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()

    align = 16 if use_halfrow else 8

    def _norm_pre(src_buf, row_base, n_r):
        _norm_pre_expert_slice(
            src_buf, pre_exp_gain_ref, out_expert_in_ref,
            h_pre=h_pre, row_base=row_base, n_r=n_r, align=align,
            pre_exp_eps=pre_exp_eps, pre_exp_scale_input=pre_exp_scale_input,
        )

    _gather_shard_xor1(
        pre_shard, pre_half_a, pre_half_b,
        ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
        me=me, world_size=world_size, use_halfrow=use_halfrow,
        during_ici_fn=_router_under_ici,
        my_half_epilogue_fn=_norm_pre,
        mate_half_epilogue_fn=_norm_pre,
    )
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def post_attn_to_moe_bridge_jax(
    stream,
    attn_out,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_w8k,
    w_pre_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int,
    pa_eps: float = 1e-5,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _DEFAULT_CID + 5,
):
    """7-op Sub-Bridge 1c: ``(stream, attn_ar_out) -> (new_stream, router_logits, expert_in)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    rows, hidden = attn_out.shape
    wp_k, pre_cols = w_pre_w8k.shape
    wr_k, n_exp = w_router_km.shape
    h_pre = world_size * pre_cols
    if (
        rows < 8
        or rows > 384
        or rows % 8 != 0
        or wp_k != hidden
        or wr_k != hidden
        or pre_cols % _DOT_TN != 0
        or n_exp % _DOT_TN != 0
    ):
        raise ValueError("Unsupported post_attn_to_moe shapes")
    n_chips = world_size // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64
    bm_g = _pick_strip(rows, 64)
    use_halfrow = rows >= 32 and rows % 32 == 0
    pre_buf_rows = rows // 2 if use_halfrow else rows

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return pl.pallas_call(
        functools.partial(
            _post_attn_to_moe_bridge_kernel,
            world_size=world_size,
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
            jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((hidden, pre_cols), w_pre_w8k.dtype),
            vmem((hidden, n_exp), w_router_km.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((8, 128) if is_r16 else ((bm_g, hidden) if tile_r32 else (rows, hidden)), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, 1), jnp.float32),
            vmem((rows, pre_cols), jnp.bfloat16),
            vmem((pre_buf_rows, h_pre), jnp.bfloat16),
            vmem((pre_buf_rows, h_pre), jnp.bfloat16),
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
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(56 if rows > 256 else 48) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w7_pa_to_moe_b{rows}_tp{world_size}_{'r16' if is_r16 else 'r32'}",
    )(
        stream.astype(jnp.float32),
        attn_out.astype(jnp.bfloat16),
        _as_2d_f32_row(pa_alpha, hidden),
        _as_2d_f32_row(pa_beta, hidden),
        _as_2d_f32_row(pre_ff_gain, hidden),
        w_pre_w8k,
        _as_2d_f32_row(w_pre_scale, pre_cols),
        w_router_km,
        _as_2d_f32_row(pre_exp_gain, h_pre),
    )


def _dot_fp8_row_block_to_ref(
    x_buf,
    r0,
    w_buf,
    scale_ref,
    dst_ref,
    *,
    n_r: int,
    n_cols: int,
    align: int = 16,
) -> None:
    """Zero-spill ``(x_buf[r0:r0+n_r, :] @ w_buf) * scale -> dst_ref[r0:r0+n_r, :n_cols]``."""
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    bm = _pick_strip(n_r, 64)

    def _sub(r_loc):
        rs = pl.ds(pl.multiple_of(r0 + r_loc, align), bm)
        for cn in range(0, n_cols, _DOT_TN):
            acc = jnp.dot(
                x_buf[rs, :],
                w_buf[:, pl.ds(cn, _DOT_TN)],
                preferred_element_type=jnp.float32,
            )
            sc = scale_ref[0:1, pl.ds(cn, _DOT_TN)]
            dst_ref[rs, pl.ds(cn, _DOT_TN)] = (acc * sc).astype(dst_ref.dtype)

    _fori_strips(n_r, bm, _sub)


def _gather_token_shard_xor1_and_dot_post(
    moe_shard_ref,
    mate_tok_buf,
    moe_full_buf,
    w_post_buf,
    s_post_ref,
    post_shard,
    w_post_sem,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    *,
    me,
    world_size: int,
) -> None:
    """Overlap ``XOR=1`` token AG (``[B//W, 4096] -> [B, 4096]``) with ``W_post`` DMA & MXU!

    Uses whole-buffer D2D into ``mate_tok_buf [r_per, K]`` followed by 16-row-aligned pair
    pack into ``moe_full_buf[chip_r0 : chip_r0 + 2*r_per, :]`` so TPU7x ``(16, 128)`` bf16
    VMEM tiling is 100% statically aligned even when ``r_per == 8`` (``B=64, TP=8``)!
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r_per, k_in = moe_shard_ref.shape
    post_cols = w_post_buf.shape[1]
    pair_r = 2 * r_per
    n_chips = world_size // 2
    mate = me ^ jnp.int32(1)
    parity = me & jnp.int32(1)
    chip = me >> jnp.int32(1)
    chip_r0 = pl.multiple_of(chip * pair_r, 16)

    _wait_barrier(world_size)
    ts1 = pltpu.make_async_remote_copy(
        moe_shard_ref,
        mate_tok_buf,
        tok_s1_send.at[0],
        tok_s1_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    ts1.start()
    ts1.wait()

    rows = n_chips * pair_r
    is_even = parity == jnp.int32(0)
    pack_w = 1024 if (k_in % 1024 == 0 and pair_r <= 32) else 512
    for c0 in range(0, k_in, pack_w):
        c_sl = pl.ds(c0, pack_w)
        own_v = moe_shard_ref[:, c_sl]
        mate_v = mate_tok_buf[:, c_sl]
        top_v = lax.select(is_even, own_v, mate_v)
        bot_v = lax.select(is_even, mate_v, own_v)
        moe_full_buf[pl.ds(chip_r0, pair_r), c_sl] = jnp.concatenate([top_v, bot_v], axis=0)

    ts2 = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            moe_full_buf.at[pl.ds(chip_r0, pair_r), :],
            moe_full_buf.at[pl.ds(chip_r0, pair_r), :],
            tok_s2_send.at[d - 1],
            tok_s2_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ts2.append(op)

    if n_chips == 4 and rows <= 128:
        for op in ts2:
            op.wait()
        _dma_wait(w_post_buf, w_post_sem.at[0])
        # Single full-batch (B, 4096) @ (4096, 1024) MXU pass stages w_post exactly once (4 MiB).
        _dot_fp8_scaled_to_ref(
            moe_full_buf, w_post_buf, s_post_ref, 0, post_shard, 0,
            rows=rows, n_cols=post_cols,
        )
    else:
        # Overlap W_post HBM wait + own-chip 2*r_per rows of post_expert_proj underneath Stage-2 ICI!
        _dma_wait(w_post_buf, w_post_sem.at[0])
        _dot_fp8_row_block_to_ref(
            moe_full_buf, chip_r0, w_post_buf, s_post_ref, post_shard,
            n_r=pair_r, n_cols=post_cols, align=16,
        )
        for op in ts2:
            op.wait()
        for d in range(1, n_chips):
            p_r0 = (chip ^ jnp.int32(d)) * pair_r
            _dot_fp8_row_block_to_ref(
                moe_full_buf, p_r0, w_post_buf, s_post_ref, post_shard,
                n_r=pair_r, n_cols=post_cols, align=16,
            )


def _moe_post_ag_epilogue_kernel(
    stream_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    out_stream_hbm,
    out_next_in_ref,
    w_post_buf,
    s_buf,
    post_shard,
    half_a,
    half_b,
    mate_tok_buf,
    moe_full_buf,
    w_post_sem,
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
) -> None:
    """Modular Sub-Bridge 2a: ``FP8 post_expert_proj + XOR=1 AG + post_ffn_res + next_in_ln``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    use_halfrow = rows >= 32 and rows % 32 == 0
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    # Stream W_post (4 MiB FP8) and stream_in (2 MiB FP32) concurrently!
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
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

    bm_p = _pick_strip(rows, 64)

    def _wait_and_prescale_stream():
        _dma_wait(s_buf, s_in_sem.at[0])
        _prescale_s_buf_in_vmem(s_buf, pf_alpha_ref, rows=rows, hidden=hidden, bm=bm_p)

    def _epi(buf, r_base, n_r):
        _run_post_ffn_slice_epilogue(
            s_buf, buf, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
            s_buf, out_next_in_ref,
            hidden=hidden, row_base=r_base, n_r=n_r, align=align,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
            with_next_norm=True, pre_scaled_alpha=True,
        )

    _gather_shard_xor1(
        post_shard, half_a, half_b,
        ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
        me=me, world_size=world_size, use_halfrow=use_halfrow,
        during_ici_fn=_wait_and_prescale_stream,
        my_half_epilogue_fn=_epi,
        mate_half_epilogue_fn=_epi,
    )
    pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def _moe_post_ag_res_only_kernel(
    stream_hbm,
    moe_out_ref,
    w_post_hbm,
    s_post_ref,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    out_stream_hbm,
    w_post_buf,
    s_buf,
    post_shard,
    half_a,
    half_b,
    mate_tok_buf,
    moe_full_buf,
    w_post_sem,
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
    sharded_moe_in: bool,
) -> None:
    """Final-layer L61 Sub-Bridge: ``FP8 post_expert_proj + XOR=1 AG + post_ffn_res``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    use_halfrow = rows >= 32 and rows % 32 == 0
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
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

    bm_p = _pick_strip(rows, 64)

    def _wait_and_prescale_stream():
        _dma_wait(s_buf, s_in_sem.at[0])
        _prescale_s_buf_in_vmem(s_buf, pf_alpha_ref, rows=rows, hidden=hidden, bm=bm_p)

    def _epi(buf, r_base, n_r):
        _run_post_ffn_slice_epilogue(
            s_buf, buf, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, None,
            s_buf, None,
            hidden=hidden, row_base=r_base, n_r=n_r, align=align,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=0.0, next_in_scale_input=False,
            with_next_norm=False, pre_scaled_alpha=True,
        )

    _gather_shard_xor1(
        post_shard, half_a, half_b,
        ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
        me=me, world_size=world_size, use_halfrow=use_halfrow,
        during_ici_fn=_wait_and_prescale_stream,
        my_half_epilogue_fn=_epi,
        mate_half_epilogue_fn=_epi,
    )
    pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def _moe_to_attn_qkvg_bridge_kernel(
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
    out_stream_hbm,
    out_next_qkv_ref,
    out_next_gate_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_vmem,
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
) -> None:
    """Full 7-op Inter-Layer Bridge 2b: ``moe_out -> (new_stream, next_qkv, next_gate)``!"""
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
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    # Queue W_post (4 MiB) -> stream_in (2 MiB) -> FULL W_qkv (10 MiB) -> FULL W_gate (8 MiB)
    # on the HBM DMA engine so HBM runs at 100% peak bandwidth non-stop during ICI AG!
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

    bm_p = _pick_strip(rows, 64)
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
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, out_next_qkv_ref, 0,
        rows=rows, n_cols=qkv_cols,
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


def moe_post_ag_res_norm_jax(
    stream,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    next_in_gain,
    *,
    world_size: int,
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    sharded_moe_in: bool = False,
    collective_id: int = _DEFAULT_CID + 2,
):
    """Modular Sub-Bridge 2a: ``moe_out -> (new_stream_f32, next_attn_in_bf16)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    wk, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    if (
        in_r < 8
        or in_r % 8 != 0
        or rows > 384
        or stream.shape[0] != rows
        or wk != k_in
        or post_cols % _DOT_TN != 0
    ):
        raise ValueError(f"Unsupported moe_post_ag_rn shapes moe={moe_out.shape}, w={w_post_w8k.shape}")
    n_chips = world_size // 2
    half_rows = rows // 2 if (rows >= 32 and rows % 32 == 0) else rows

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return pl.pallas_call(
        functools.partial(
            _moe_post_ag_epilogue_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=(h_sp, v_sp),
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
            vmem_limit_bytes=(56 if rows > 256 else 48) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w7_post_ag_rn_b{rows}_tp{world_size}{'_sh' if sharded_moe_in else ''}",
    )(
        stream.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_2d_f32_row(s_post, post_cols),
        _as_2d_f32_row(post_ff_gain, hidden),
        _as_2d_f32_row(pf_alpha, hidden),
        _as_2d_f32_row(pf_beta, hidden),
        _as_2d_f32_row(next_in_gain, hidden),
    )


def moe_post_ag_res_only_jax(
    stream,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    *,
    world_size: int,
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    sharded_moe_in: bool = False,
    collective_id: int = _DEFAULT_CID + 3,
):
    """Final-layer L61 Sub-Bridge: ``moe_out -> final_stream_f32 [B, H]``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    wk, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    if (
        in_r < 8
        or in_r % 8 != 0
        or rows > 384
        or stream.shape[0] != rows
        or wk != k_in
        or post_cols % _DOT_TN != 0
    ):
        raise ValueError(f"Unsupported moe_post_ag_ro shapes moe={moe_out.shape}, w={w_post_w8k.shape}")
    n_chips = world_size // 2
    half_rows = rows // 2 if (rows >= 32 and rows % 32 == 0) else rows

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return pl.pallas_call(
        functools.partial(
            _moe_post_ag_res_only_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp],
        out_specs=h_sp,
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
            vmem_limit_bytes=(56 if rows > 256 else 48) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w7_post_ag_ro_b{rows}_tp{world_size}{'_sh' if sharded_moe_in else ''}",
    )(
        stream.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_2d_f32_row(s_post, post_cols),
        _as_2d_f32_row(post_ff_gain, hidden),
        _as_2d_f32_row(pf_alpha, hidden),
        _as_2d_f32_row(pf_beta, hidden),
    )


def moe_to_attn_bridge_jax(
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
    *,
    world_size: int,
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    sharded_moe_in: bool = False,
    interpret: bool = False,
    collective_id: int = _DEFAULT_CID + 4,
):
    """Full 7-op Bridge 2b: ``moe_out -> (new_stream_f32, next_qkv_bf16, next_gate_bf16)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    wk, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    qkv_k, qkv_cols = next_w_qkv_w8k.shape
    gt_k, gate_cols = next_w_gate_w8k.shape
    if (
        in_r < 8
        or in_r % 8 != 0
        or rows > 384
        or stream.shape[0] != rows
        or wk != k_in
        or qkv_k != hidden
        or gt_k != hidden
        or post_cols % _DOT_TN != 0
        or qkv_cols % _DOT_TN != 0
        or gate_cols % _DOT_TN != 0
    ):
        raise ValueError("Unsupported moe_to_attn_bridge shapes")
    n_chips = world_size // 2
    use_halfrow = rows >= 32 and rows % 32 == 0
    half_rows = rows // 2 if use_halfrow else rows

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _moe_to_attn_qkvg_bridge_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, qkv_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
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
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(58 if rows > 256 else 56) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w7_moe_to_qkvg_b{rows}_tp{world_size}{'_sh' if sharded_moe_in else ''}",
    )(
        stream.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_2d_f32_row(s_post, post_cols),
        _as_2d_f32_row(post_ff_gain, hidden),
        _as_2d_f32_row(pf_alpha, hidden),
        _as_2d_f32_row(pf_beta, hidden),
        _as_2d_f32_row(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_2d_f32_row(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_2d_f32_row(next_s_gate, gate_cols),
    )


def _fp8_dual_qkv_gate_kernel(
    attn_in_ref,
    w_qkv_hbm,
    s_qkv_ref,
    w_gate_hbm,
    s_gate_ref,
    out_qkv_ref,
    out_gate_ref,
    w_qkv_buf,
    w_gate_buf,
    w_qkv_sem,
    w_gate_sem,
) -> None:
    """Zero-spill L0 entry FP8 ``qkv_proj + gate_proj`` -> two separate contiguous tensors!"""
    from jax.experimental.pallas import tpu as pltpu

    rows = attn_in_ref.shape[0]
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()
    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        attn_in_ref, w_qkv_buf, s_qkv_ref, 0, out_qkv_ref, 0,
        rows=rows, n_cols=w_qkv_hbm.shape[1],
    )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        attn_in_ref, w_gate_buf, s_gate_ref, 0, out_gate_ref, 0,
        rows=rows, n_cols=w_gate_hbm.shape[1],
    )


def fp8_dual_qkv_gate_jax(attn_in, w_qkv_w8k, s_qkv, w_gate_w8k, s_gate):
    """Single-read FP8 ``qkv_proj + gate_proj`` emitting two separate contiguous bf16 outputs."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, hidden = attn_in.shape
    qk, qkv_cols = w_qkv_w8k.shape
    gk, gate_cols = w_gate_w8k.shape
    if rows < 8 or rows > 384 or rows % 8 != 0 or qk != hidden or gk != hidden:
        raise ValueError("Unsupported fp8_dual_qkv_gate shapes")
    if qkv_cols % _DOT_TN != 0 or gate_cols % _DOT_TN != 0:
        raise ValueError("Expected 256-aligned qkv/gate columns")

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return pl.pallas_call(
        _fp8_dual_qkv_gate_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, qkv_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=[v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(v_sp, v_sp),
        scratch_shapes=[
            vmem((hidden, qkv_cols), w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), w_gate_w8k.dtype),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(48 if rows > 256 else 40) * 1024**2,
            disable_bounds_checks=True,
        ),
        name=f"muse_w7_dual_qkvg_b{rows}_qkv{qkv_cols}_g{gate_cols}",
    )(
        attn_in.astype(jnp.bfloat16),
        w_qkv_w8k,
        _as_2d_f32_row(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_2d_f32_row(s_gate, gate_cols),
    )


# =============================================================================
# 3. TorchTPU `cpu_export_jax_op` Registration (Static-Arg Closure Safe)
# =============================================================================
@functools.cache
def _get_tp_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def get_attn_oproj_ar_res_norm_op(world_size: int) -> Callable:
    """Register Sub-Bridge 1a (`gated_attn -> (new_stream, router_in, ffn_hidden)`)."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    op = _ATTN_OPROJ_AR_RN_OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _ATTN_OPROJ_AR_RN_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (P(), P(None, "model"), P("model", None), P("model"), P(), P(), P())
            in_specs = (*t_specs, None, None, None)
            out_specs = (P(), P(), P())

            def _wrapped(
                stream: jax.Array,
                gated_attn: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                def _local(st, ga, wo, ws, a, b, pg):
                    return attn_oproj_ar_res_norm_jax(
                        st, ga, wo, ws[0], a, b, pg,
                        world_size=world_size,
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(stream, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta, pre_ff_gain)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w7_oproj_ar_rn_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _ATTN_OPROJ_AR_RN_OPS[world_size] = op
    return op


def get_attn_to_moe_bridge_op(
    world_size: int,
    *,
    bf16_router: bool = True,
    router_d2d_overlap: bool | None = None,
) -> Callable:
    """Register Full Bridge 1b (`gated_attn -> (new_stream, scaled_router_logits, expert_in)`)."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    r_d2d_ovl = (
        (os.environ.get("SGLANG_TPU_MUSE_BRIDGE1_ROUTER_D2D_OVERLAP", "0") == "1")
        if router_d2d_overlap is None
        else bool(router_d2d_overlap)
    )
    key = (int(world_size), bool(bf16_router), r_d2d_ovl)
    op = _ATTN_TO_MOE_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _ATTN_TO_MOE_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (
                P(),
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
            out_specs = (P(), P(), P())

            def _wrapped(
                stream: jax.Array,
                gated_attn: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_w8k: jax.Array,
                w_pre_scale: jax.Array,
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
                    return attn_to_moe_bridge_jax(
                        st, ga, wo, ws[0], a, b, pfg, wp, wps, wr, peg,
                        world_size=world_size,
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                        router_d2d_overlap=r_d2d_ovl,
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
                    pre_ff_gain, w_pre_w8k, w_pre_scale, w_router_km, pre_exp_gain,
                )

            tag = "r16" if bf16_router else "r32"
            ovl_tag = "_rov" if r_d2d_ovl else ""
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w7_attn_to_moe_{tag}_tp{world_size}{ovl_tag}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _ATTN_TO_MOE_OPS[key] = op
    return op


def get_moe_post_ag_res_norm_op(
    world_size: int, *, sharded_moe_in: bool = False
) -> Callable:
    """Register Sub-Bridge 2a (`moe_out -> (new_stream, next_attn_in)`)."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(sharded_moe_in))
    op = _MOE_POST_AG_RN_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _MOE_POST_AG_RN_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            t_specs = (P(), moe_sp, P(None, "model"), P(None, "model"), P(), P(), P(), P())
            in_specs = (*t_specs, None, None, None, None)
            out_specs = (P(), P())

            def _wrapped(
                stream: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array]:
                fn = functools.partial(
                    moe_post_ag_res_norm_jax,
                    world_size=world_size,
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_scale_input),
                    sharded_moe_in=bool(sharded_moe_in),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta, next_in_gain)

            sfx = "_sh" if sharded_moe_in else ""
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w7_post_ag_rn_tp{world_size}{sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _MOE_POST_AG_RN_OPS[key] = op
    return op


def get_moe_post_ag_res_only_op(
    world_size: int, *, sharded_moe_in: bool = False
) -> Callable:
    """Register L61 Sub-Bridge (`moe_out -> final_stream`)."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(sharded_moe_in))
    op = _MOE_POST_AG_RO_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _MOE_POST_AG_RO_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            t_specs = (P(), moe_sp, P(None, "model"), P(None, "model"), P(), P(), P())
            in_specs = (*t_specs, None, None)

            def _wrapped(
                stream: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
            ) -> jax.Array:
                fn = functools.partial(
                    moe_post_ag_res_only_jax,
                    world_size=world_size,
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    sharded_moe_in=bool(sharded_moe_in),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=P(), check_vma=False
                )(stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta)

            sfx = "_sh" if sharded_moe_in else ""
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w7_post_ag_ro_tp{world_size}{sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=(P(),),
            )
            _MOE_POST_AG_RO_OPS[key] = op
    return op


def get_moe_to_attn_bridge_op(
    world_size: int, *, sharded_moe_in: bool = False
) -> Callable:
    """Register Full Bridge 2b (`moe_out -> (new_stream, next_qkv_contig, next_gate_contig)`)."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(sharded_moe_in))
    op = _MOE_TO_ATTN_QKVG_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _MOE_TO_ATTN_QKVG_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            t_specs = (
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
            in_specs = (*t_specs, None, None, None, None)
            out_specs = (P(), P(None, "model"), P(None, "model"))

            def _wrapped(
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
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                fn = functools.partial(
                    moe_to_attn_bridge_jax,
                    world_size=world_size,
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_scale_input),
                    sharded_moe_in=bool(sharded_moe_in),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta,
                    next_in_gain, next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
                )

            sfx = "_sh" if sharded_moe_in else ""
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w7_moe_to_qkvg_tp{world_size}{sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _MOE_TO_ATTN_QKVG_OPS[key] = op
    return op


def get_fp8_dual_qkv_gate_op() -> Callable:
    """Register L0 entry `fp8_dual_qkv_gate` (`attn_in -> (qkv_contig, gate_contig)`)."""
    global _DUAL_QKVG_OP
    if _DUAL_QKVG_OP is not None:
        return _DUAL_QKVG_OP
    with _OPS_LOCK:
        if _DUAL_QKVG_OP is None:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return fp8_dual_qkv_gate_jax(attn_in, w_qkv_w8k, s_qkv, w_gate_w8k, s_gate)

            _DUAL_QKVG_OP = cpu_export_jax_op(
                "sglang_tpu_pallas::muse_w7_dual_qkv_gate", _wrapped
            )
    return _DUAL_QKVG_OP


def get_post_attn_to_moe_bridge_op(world_size: int, *, use_w8a8: bool = False) -> Callable:
    """Register 7-op Sub-Bridge 1c (`(stream, attn_ar_out) -> (stream, logits, expert_in)`).

    Pairs with ``FP8 o_proj + XLA all-reduce``; ``use_w8a8`` is kept reserved as False so the
    cache key stays uniform with ``get_attn_to_moe_bridge_op``.
    """
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (world_size, bool(use_w8a8))
    op = _POST_ATTN_TO_MOE_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _POST_ATTN_TO_MOE_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (P(), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(), P())
            in_specs = (*t_specs, None, None, None, None, None, None)
            out_specs = (P(), P(), P())

            def _wrapped(
                stream: jax.Array,
                attn_out: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_w8k: jax.Array,
                w_pre_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                fn = functools.partial(
                    post_attn_to_moe_bridge_jax,
                    world_size=world_size,
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream, attn_out, pa_alpha, pa_beta, pre_ff_gain,
                    w_pre_w8k, w_pre_scale, w_router_km, pre_exp_gain,
                )

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w7_pa_to_moe_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _POST_ATTN_TO_MOE_OPS[key] = op
    return op


def prewarm_decode_collective_bridge_ops(
    world_size: int, mode: int, *, bf16_router: bool = False
) -> None:
    """Eagerly register all custom ops used by ``mode`` BEFORE ``torch.compile`` starts.

    TorchDynamo rejects ``threading.Lock.__enter__`` during cold graph tracing; warming the
    module caches during model load ensures every forward getter hits the lock-free fast path.
    """
    if mode <= 0 or world_size not in SUPPORTED_WORLD_SIZES:
        return
    if mode in (1, 3, 4, 5, 6):
        from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
            get_muse_decode_fp8_linear_bf16_op,
        )

        get_muse_decode_fp8_linear_bf16_op()
    if mode in (1, 3, 4):
        get_attn_to_moe_bridge_op(world_size, bf16_router=bool(bf16_router))
    if mode in (5, 6):
        get_post_attn_to_moe_bridge_op(world_size)
    if mode in (2, 3, 4, 5, 6):
        for sh in (False, True):
            get_moe_post_ag_res_norm_op(world_size, sharded_moe_in=sh)
            get_moe_post_ag_res_only_op(world_size, sharded_moe_in=sh)
    if mode in (2, 3, 5):
        for sh in (False, True):
            get_moe_to_attn_bridge_op(world_size, sharded_moe_in=sh)
        get_fp8_dual_qkv_gate_op()
