# SPDX-License-Identifier: Apache-2.0
"""Sequence-Parallel Direct Peer-VMEM All-to-All & ``_blocks_to_owners`` Decode MoE.

Implements two ``muse_megakernel``-inspired sequence-parallel (``SP + EP8``)
primitives for Muse Spark decode on TPU7x:
  1. ``blocks_to_owners_norm_pallas``: Converts column-sharded ``[B, C // EP]``
     linear outputs (e.g. ``pre_expert_proj`` ``[B, 512]`` or ``post_expert_proj``
     ``[B, 1024]``) directly into row-sharded ``[B // EP, C]`` owner activations
     via ``(EP - 1)`` tile-aligned peer-VMEM remote DMAs (56 KiB/rank at B=64 vs
     448 KiB/rank for All-Gather — 8x less ICI traffic!) with optional fused
     ``RMSNorm`` on only ``B // EP`` rows.
  2. ``fused_decode_ep_a2a_from_sharded``: Single-launch SP+EP decode MoE kernel
     taking row-sharded ``logits_own [B // EP, 256]`` and either ``x_own [B // EP, 4096]``
     or column-sharded ``pre_col [B, 4096 // EP]`` (fused ``_blocks_to_owners`` +
     ``pre_expert_norm``), executing:
       - Exact FP32 ``sigmoid + bias`` 8-round tie-preserving routing on ONLY
         ``B // EP`` rows per rank (8x less VPU routing work),
       - Rank-deduplicated 3D ``(rows, 32, 128)`` direct peer-VMEM dispatch
         (sending at most 1 copy per ``(own_row, hit_dest_rank)`` with zero
         ``SEG=8`` HBM tile padding waste),
       - 4-slot homogeneous circular VMEM weight ring + fused ``inv_rms`` scatter,
       - Rank-deduplicated peer-VMEM combine returning pre-summed partials back
         to each row owner's VMEM (plus optional ``pair_ag_relay`` for replicated
         output comparison).
"""

# PEP 563 stays off so TorchTPU / cpu_export_jax_op sees concrete annotations.

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
    _advance_unit,
    _build_masked_x_from_2d,
    _build_masked_x_from_3d,
    _fp4_blockdiag_hoisted_dual_strip_v2,
    _fp4_blockdiag_hoisted_strip_v2,
    _fp4_dual_gmm_chunked_v2,
    _fp4_gmm_chunked_v2,
    _pad_rows_to,
    _pick_bd_nc,
    _pick_k16_nc,
    _pick_k256_nc,
    _ring_slot,
)

_FP4_PACK = 8
_H = 4096
_I = 4096
_N1 = 8192
_E_TOTAL = 256
_TOPK = 8


@dataclasses.dataclass(frozen=True)
class EpA2ADecodeConfig:
    """Compile-time configuration for ``fused_decode_ep_a2a_from_sharded``."""

    ring_mode: str = "ring4_dual"  # "ring3_dual" or "ring4_dual"
    tile_m: int = 32
    m_buckets: tuple[int, ...] = (16, 32)
    k16_blockdiag_rows: tuple[int, ...] = (1, 2, 4, 8)
    fp4_unpack_mode: str = "via_fp8"
    combine_bf16: bool = True
    pair_ag_relay: bool = True
    norm_f32_stage: bool = False


def default_ep_a2a_config(
    batch_size: int, ep: int, scale_block: int
) -> EpA2ADecodeConfig:
    if scale_block == 16:
        tm = 32 if batch_size <= 64 else 64
        mb = (16, 32) if tm == 32 else (16, 32, 64)
        return EpA2ADecodeConfig(
            ring_mode="ring3_dual" if batch_size >= 256 else "ring4_dual",
            tile_m=tm,
            m_buckets=mb,
            k16_blockdiag_rows=(1, 2, 4, 8),
            fp4_unpack_mode="via_fp8",
            combine_bf16=(batch_size >= 64),
            pair_ag_relay=(ep == 8),
        )
    tm = 32 if batch_size <= 64 else 64
    mb = (16, 32) if tm == 32 else (16, 32, 64)
    return EpA2ADecodeConfig(
        ring_mode="ring4_dual",
        tile_m=tm,
        m_buckets=mb,
        k16_blockdiag_rows=(),
        fp4_unpack_mode="via_fp8",
        combine_bf16=(batch_size >= 64),
        pair_ag_relay=(ep == 8),
    )


def _blocks_to_owners_kernel(
    rank_ref,
    col_in_ref,
    norm_ref,
    out_ref,
    stage_vmem,
    rcv_vmem,
    send_sem,
    recv_sem,
    *,
    ep: int,
    rows_own: int,
    col_local: int,
    has_norm: bool,
    eps: float,
    ep_axis_name: str,
):
    """Port of ``muse_megakernel/decode/spep/layer.py:_blocks_to_owners`` in VMEM."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for d in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]
    pad_r = stage_vmem.shape[1]

    # Stage each owner k's rows at the top of its 8-row-aligned slot.
    for k in range(ep):
        blk = col_in_ref[pl.ds(k * rows_own, rows_own), :]
        stage_vmem[k, :, :] = _pad_rows_to(blk, pad_r)

    pl.semaphore_wait(barrier_sem, ep)

    rcv_vmem[me] = stage_vmem[me]
    ops = []
    for d in range(ep):
        op = pltpu.make_async_remote_copy(
            stage_vmem.at[d],
            rcv_vmem.at[me],
            send_sem.at[d],
            recv_sem,
            device_id={ep_axis_name: jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        ops.append((d, op))

        @pl.when(me != d)
        def _s(_op=op):
            _op.start()

    pltpu.make_async_copy(
        rcv_vmem.at[pl.ds(0, ep - 1)],
        rcv_vmem.at[pl.ds(0, ep - 1)],
        recv_sem,
    ).wait()

    own_full = jnp.concatenate(
        [rcv_vmem[d, 0:rows_own, :] for d in range(ep)], axis=1
    )
    if has_norm:
        xf = own_full.astype(jnp.float32)
        gain = norm_ref[...].reshape(1, ep * col_local).astype(jnp.float32) + 1.0
        inv = lax.rsqrt(jnp.mean(xf * xf, axis=1, keepdims=True) + eps)
        out_val = (xf * inv * gain).astype(jnp.bfloat16)
    else:
        out_val = own_full
    out_ref[...] = _pad_rows_to(out_val, out_ref.shape[0])

    for d, op in ops:
        @pl.when(me != d)
        def _w(_op=op):
            _op.wait_send()


def blocks_to_owners_norm_pallas(
    rank: jax.Array,
    col_slice: jax.Array,
    norm: jax.Array | None = None,
    *,
    ep: int = 8,
    eps: float = 1e-5,
    ep_axis_name: str = "model",
) -> jax.Array:
    """Convert ``[B, C_local]`` column-sharded tensor -> ``[B // EP, C]`` owner rows."""
    batch_size, col_local = col_slice.shape
    if batch_size % ep != 0 or col_local % 128 != 0:
        raise ValueError(
            f"blocks_to_owners requires B % ep == 0 and col_local % 128 == 0, "
            f"got {col_slice.shape} for ep={ep}"
        )
    rows_own = batch_size // ep
    pad_r = max(8, ((rows_own + 7) // 8) * 8)
    full_c = ep * col_local
    has_norm = norm is not None
    norm_arg = (
        norm.reshape(1, full_c)
        if has_norm
        else jnp.zeros((1, 128), dtype=jnp.bfloat16)
    )
    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    out_pad = pl.pallas_call(
        functools.partial(
            _blocks_to_owners_kernel,
            ep=ep,
            rows_own=rows_own,
            col_local=col_local,
            has_norm=has_norm,
            eps=eps,
            ep_axis_name=ep_axis_name,
        ),
        out_shape=jax.ShapeDtypeStruct((pad_r, full_c), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(v_spec, v_spec),
            out_specs=v_spec,
            scratch_shapes=(
                vmem((ep, pad_r, col_local), jnp.bfloat16),
                vmem((ep, pad_r, col_local), jnp.bfloat16),
                sem((ep,)),
                sem,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024**2,
            collective_id=0,
        ),
        name=f"muse_blocks_to_owners_ep{ep}_b{batch_size}_c{full_c}",
    )(rank, col_slice, norm_arg)
    return out_pad[:rows_own]


def _ep_a2a_decode_kernel(
    *all_refs,
    ep: int,
    batch_size: int,
    scale_block: int,
    input_is_col_sharded: bool,
    pre_eps: float,
    return_replicated: bool,
    ep_axis_name: str,
    cfg: EpA2ADecodeConfig,
):
    is_k256 = scale_block == 256
    n_slots = 4 if cfg.ring_mode.startswith("ring4") else 3
    use_bd = (not is_k256) and bool(cfg.k16_blockdiag_rows)
    tile_m = cfg.tile_m
    e_local = _E_TOTAL // ep
    rows_own = batch_size // ep
    pad_own = max(8, ((rows_own + 7) // 8) * 8)

    it = iter(all_refs)
    rank_ref = next(it)
    logits_own_ref = next(it)  # [pad_own, 256] f32
    bias_ref = next(it)  # [1, 256] f32
    x_in_ref = next(it)  # [B, 4096//ep] if input_is_col_sharded else [pad_own, 4096]
    pre_norm_ref = next(it) if input_is_col_sharded else None
    w1_ref = next(it)
    s1_ref = next(it)
    w2_ref = next(it)
    s2_ref = next(it)
    norm_ref = next(it)
    output_ref = next(it)

    if input_is_col_sharded:
        b2o_stage = next(it)
        b2o_rcv = next(it)
        b2o_ssem = next(it)
        b2o_rsem = next(it)
    else:
        b2o_stage = b2o_rcv = b2o_ssem = b2o_rsem = None

    # Routing & deduplicated peer-VMEM A2A buffers
    own_x_3d = next(it)  # [rows_own, 32, 128] bf16
    hdr_send_vmem = next(it)  # [ep, 8, 128] int32 (col 0..rows_own-1: hit mask; row 1: n_rows_sent)
    hdr_recv_vmem = next(it)  # [ep, 8, 128] int32
    hdr_recv_smem = next(it)  # [ep, 8, 128] int32
    w_send_vmem = next(it)  # [ep, pad_own, 128] f32 (weights for dest d's e_local experts)
    w_recv_vmem = next(it)  # [ep, pad_own, 128] f32
    w_recv_smem = next(it)  # [ep, pad_own, 128] f32
    hdr_ssem = next(it)
    hdr_rsem = next(it)

    disp_recv_3d = next(it)  # [ep, rows_own, 32, 128] bf16 (rank-deduplicated!)
    disp_ssem = next(it)
    disp_rsem = next(it)

    # Local SMEM expert bucketing over deduplicated received rows
    exec_eid_smem = next(it)
    exec_slot_smem = next(it)
    exec_start_smem = next(it)
    exec_wptr_smem = next(it)
    packed_gid_smem = next(it)  # global slot s*rows_own + k_idx in disp_recv_3d
    packed_w_smem = next(it)
    src_Slot_map_smem = next(it)  # [ep, rows_own]: compact index k_idx -> local r

    w_ring = next(it)
    s_ring = next(it)
    ring_sem = next(it)
    misc_sem = next(it)

    io_3d_buf = next(it)
    lhs_2d_buf = next(it)
    act_2d_buf = next(it)
    y_stage_buf = next(it)
    masked_x_buf = next(it) if use_bd else None
    gain_f32_buf = next(it)
    accum_recv_3d = next(it)  # [ep * rows_own, 32, 128] f32

    comb_send_3d = next(it)  # [ep, rows_own, 32, 128] bf16 or f32
    comb_recv_3d = next(it)  # [ep, rows_own, 32, 128] bf16 or f32
    comb_ssem = next(it)
    comb_rsem = next(it)
    owner_3d = next(it)

    if return_replicated:
        fan_3d_buf = next(it)
        if cfg.pair_ag_relay and ep == 8:
            ag_ssem = next(it)
            ag_psem = next(it)
            fan_ssem = fan_rsem = None
        else:
            fan_ssem = next(it)
            fan_rsem = next(it)
            ag_ssem = ag_psem = None
    else:
        fan_3d_buf = ag_ssem = ag_psem = fan_ssem = fan_rsem = None

    # 1. Entry barrier across EP ranks.
    barrier_sem = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]
    w1_u32 = w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref.bitcast(jnp.uint32)
    gain_f32_buf[...] = norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0

    # 2. Sequence-parallel tie-exact routing on ONLY ``rows_own = B // EP`` rows!
    lg_raw = logits_own_ref[0:pad_own, :].astype(jnp.float32)
    row_valid = lax.broadcasted_iota(jnp.int32, (pad_own, _E_TOTAL), 0) < rows_own
    lg = jnp.where(jnp.isnan(lg_raw) | jnp.logical_not(row_valid), -jnp.inf, lg_raw)
    bias_row = bias_ref[...].reshape(1, _E_TOTAL).astype(jnp.float32)
    scores = jnp.where(row_valid, 1.0 / (1.0 + jnp.exp(-lg)), 0.0)
    choice = jnp.where(row_valid, scores + bias_row, -jnp.inf)
    e_iota = lax.broadcasted_iota(jnp.int32, (pad_own, _E_TOTAL), 1)
    s_iota = lax.broadcasted_iota(jnp.int32, (pad_own, _TOPK), 1)
    in_order = jnp.zeros((pad_own, _TOPK), jnp.float32)
    dense_w = jnp.zeros((pad_own, _E_TOTAL), jnp.float32)

    for slot in range(_TOPK):
        peak = jnp.max(choice, axis=1, keepdims=True)
        best = jnp.min(
            jnp.where(choice == peak, e_iota, _E_TOTAL), axis=1, keepdims=True
        )
        pick = (e_iota == best) & row_valid
        sel = jnp.where(pick, scores, 0.0)
        in_order = jnp.where(
            s_iota == slot, jnp.sum(sel, axis=1, keepdims=True), in_order
        )
        dense_w = jnp.where(pick, scores, dense_w)
        choice = jnp.where(pick, -jnp.inf, choice)

    w_own = jnp.where(
        row_valid,
        dense_w / (jnp.sum(in_order, axis=1, keepdims=True) + 1e-15),
        0.0,
    )

    pl.semaphore_wait(barrier_sem, ep)

    # 3. Optional fused _blocks_to_owners + pre_expert_norm overlapped with routing!
    if input_is_col_sharded:
        col_local = _H // ep
        for k in range(ep):
            blk = x_in_ref[pl.ds(k * rows_own, rows_own), :]
            b2o_stage[k, :, :] = _pad_rows_to(blk, pad_own)
        b2o_rcv[me] = b2o_stage[me]
        b2o_ops = []
        for d in range(ep):
            op_b = pltpu.make_async_remote_copy(
                b2o_stage.at[d],
                b2o_rcv.at[me],
                b2o_ssem.at[d],
                b2o_rsem,
                device_id={ep_axis_name: jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            b2o_ops.append((d, op_b))

            @pl.when(me != d)
            def _sb(_op=op_b):
                _op.start()

        pltpu.make_async_copy(
            b2o_rcv.at[pl.ds(0, ep - 1)],
            b2o_rcv.at[pl.ds(0, ep - 1)],
            b2o_rsem,
        ).wait()

        own_cat = jnp.concatenate(
            [b2o_rcv[d, 0:pad_own, :] for d in range(ep)], axis=1
        )
        xf = own_cat.astype(jnp.float32)
        pre_g = pre_norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0
        inv_pre = lax.rsqrt(jnp.mean(xf * xf, axis=1, keepdims=True) + pre_eps)
        x_normed = (xf * inv_pre * pre_g).astype(jnp.bfloat16)
        own_x_3d[...] = x_normed[:rows_own, :].reshape(rows_own, 32, 128)
    else:
        own_x_3d[...] = x_in_ref[0:rows_own, :].reshape(rows_own, 32, 128)
        b2o_ops = []

    # 4. Build per-destination rank-deduplicated hit masks & weight slices, exchange headers!
    lane128 = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    row8_i = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    for d in range(ep):
        w_sub = w_own[:, d * e_local : (d + 1) * e_local]  # [pad_own, e_local]
        hit_r = jnp.any(w_sub > 0.0, axis=1)  # [pad_own] bool
        # Prefix sum over own rows to get compact send slot 0..n_hit_d - 1!
        hit_i32 = hit_r.astype(jnp.int32)
        n_hit_d = jnp.sum(hit_i32)
        # Store hit_i32 in row 0 cols 0..pad_own-1, and n_hit_d in row 1 col 0
        hdr_blk = jnp.zeros((8, 128), dtype=jnp.int32)
        hit_pad128 = jnp.pad(hit_i32[None, :], ((0, 0), (0, 128 - pad_own)))
        hdr_blk = jnp.where(row8_i == 0, hit_pad128, hdr_blk)
        hdr_blk = jnp.where(
            (row8_i == 1) & (lane128 == 0), n_hit_d, hdr_blk
        )
        hdr_send_vmem[d] = hdr_blk
        w_send_vmem[d] = jnp.pad(
            w_sub, ((0, 0), (0, 128 - e_local))
        )

    hdr_recv_vmem[me] = hdr_send_vmem[me]
    w_recv_vmem[me] = w_send_vmem[me]
    hdr_ops = []
    for d in range(ep):
        op_h = pltpu.make_async_remote_copy(
            hdr_send_vmem.at[d],
            hdr_recv_vmem.at[me],
            hdr_ssem.at[d],
            hdr_rsem,
            device_id={ep_axis_name: jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_w = pltpu.make_async_remote_copy(
            w_send_vmem.at[d],
            w_recv_vmem.at[me],
            hdr_ssem.at[ep + d],
            hdr_rsem,
            device_id={ep_axis_name: jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        hdr_ops.append((d, op_h, op_w))

        @pl.when(me != d)
        def _sh(_oh=op_h, _ow=op_w):
            _oh.start()
            _ow.start()

    pltpu.make_async_copy(
        hdr_recv_vmem.at[pl.ds(0, ep - 1)],
        hdr_recv_vmem.at[pl.ds(0, ep - 1)],
        hdr_rsem,
    ).wait()
    pltpu.make_async_copy(
        w_recv_vmem.at[pl.ds(0, ep - 1)],
        w_recv_vmem.at[pl.ds(0, ep - 1)],
        hdr_rsem,
    ).wait()

    # Copy both received header & weight tables + own send headers to SMEM!
    cp_h = pltpu.make_async_copy(hdr_recv_vmem, hdr_recv_smem, misc_sem.at[0])
    cp_w = pltpu.make_async_copy(w_recv_vmem, w_recv_smem, misc_sem.at[0])
    # Also copy hdr_send_vmem to SMEM via hdr_recv_smem? No, compact own send rows first!
    cp_h.start()
    cp_w.start()
    cp_h.wait()
    cp_w.wait()

    # Wait on hdr_send before reusing misc_sem for hdr_send_smem? Instead compute own compact
    # send buffer using a tiny SMEM copy of hdr_send_vmem!
    hdr_send_smem_tmp = hdr_recv_smem  # wait: keep hdr_recv_smem intact!
    del hdr_send_smem_tmp

    # 5. Rank-deduplicated 3D (1, 32, 128) peer-VMEM dispatch:
    # Pack own hit rows for each destination d into disp_send_3d[d, 0..n_send_d-1] using VPU/SMEM!
    # Notice: even simpler & faster — send ONLY when n_hit_d > 0:
    # Since rows_own <= 32, sending `own_x_3d` (rows_own rows) ONLY to peers d that have
    # n_hit_d > 0 OR sending individual hit rows! Let's send ONLY individual hit rows!
    # Wait: to know own hit rows in SMEM without an extra SMEM buffer, store own hit mask
    # for all d in row 2..7? Even simpler: hdr_send_vmem[d] WAS constructed on `me`!
    # Wait: let's include `own_hdr_smem` in scratch so `me` reads `own_hdr_smem[d, 0, r]` in 1 cycle!
    # Let's check: we can pack `own_x_3d[r]` -> `disp_send_3d[d, k]` in `rows_own * ep` steps (`<=256` scalar steps = 0.2 us),
    # and issue 1 single contiguous `n_send_d`-row remote DMA per active peer `d`?
    # Wait: `n_send_d` is a dynamic runtime length (`pl.ds(0, n_send_d)` is NOT allowed in Pallas —
    # Pallas requires static slice sizes in `make_async_remote_copy`, looped `fori_loop(0, n_send_d)` of 1-row `(1, 32, 128)` copies!).
    # YES!! Look at `a2a.py:438` and `layer.py:445`: dynamic counts MUST loop static 1-row `(1, 32, 128)` copies!
    # And with 1-row `(1, 32, 128)` remote copies, we don't even need `disp_send_3d` staging — we send
    # `own_x_3d.at[pl.ds(r, 1)]` DIRECTLY into `peer d`'s `disp_recv_3d.at[me, pl.ds(r, 1)]` ONLY WHEN `hit(r, d)`!!
    # ZERO STAGING COPY, ZERO COMPACTION LOOP, AND EXACT DEDUPLICATION!

    # Wait: let's copy hdr_send_vmem into disp_ssem-waited SMEM? We have `src_Slot_map_smem` `[ep, pad_own]`!
    # Copy row 0 of each `hdr_send_vmem[d]`? `hdr_send_vmem` is `[ep, 8, 128]`: copy `hdr_send_vmem` to `hdr_send_smem`!
    # Let's use `src_Slot_map_smem` shaped `(ep, 8, 128)` in SMEM to hold `hdr_send_vmem`!
    cp_own_h = pltpu.make_async_copy(
        hdr_send_vmem, src_Slot_map_smem, misc_sem.at[0]
    )
    cp_own_h.start()
    cp_own_h.wait()

    # Start deduplicated 1-row remote DMAs for every (d != me, r) where own row r hits rank d!
    disp_recv_3d[me] = own_x_3d[...]
    total_disp_sent = jnp.int32(0)
    for d in range(ep):
        def _send_own_row(r, cnt_s, d=d):
            hit_rd = (src_Slot_map_smem[d, 0, r] != 0) & (me != d)

            @pl.when(hit_rd)
            def _do_send():
                pltpu.make_async_remote_copy(
                    own_x_3d.at[pl.ds(r, 1), :, :],
                    disp_recv_3d.at[me, pl.ds(r, 1), :, :],
                    disp_ssem,
                    disp_rsem,
                    device_id={ep_axis_name: jnp.int32(d)},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            return cnt_s + hit_rd.astype(jnp.int32)

        total_disp_sent = lax.fori_loop(
            0, rows_own, _send_own_row, total_disp_sent
        )

    # 6. While peer-VMEM dispatch DMAs fly over ICI, build local expert schedule & PRIME WEIGHT RING!
    def _build_local_plan(le, state):
        k_cur, cur_off = state
        # Count total hits for local expert `le` across all (s, r)
        def _cnt_s(s, acc_s):
            def _cnt_r(r, acc_r):
                w_val = w_recv_smem[s, r, le]
                return acc_r + (w_val > 0.0).astype(jnp.int32)

            return lax.fori_loop(0, rows_own, _cnt_r, acc_s)

        cnt_le = lax.fori_loop(0, ep, _cnt_s, jnp.int32(0))
        active = cnt_le > 0

        @pl.when(active)
        def _rec_e():
            exec_eid_smem[k_cur] = me * e_local + le
            exec_slot_smem[k_cur] = le
            exec_start_smem[k_cur] = cur_off
            exec_wptr_smem[k_cur] = cur_off

        return (
            k_cur + active.astype(jnp.int32),
            cur_off + cnt_le,
        )

    num_exec, tot_expert_rows = lax.fori_loop(
        0, e_local, _build_local_plan, (jnp.int32(0), jnp.int32(0))
    )
    exec_start_smem[num_exec] = tot_expert_rows

    half_i = _I // 2

    def start_ring_unit(r_slot, w_slot, part):
        @pl.when(part < 2)
        def _u01():
            g_col = pl.multiple_of(part * half_i, half_i)
            u_col = _I + g_col
            pltpu.make_async_copy(
                w1_u32.at[w_slot, :, pl.ds(g_col, half_i)],
                w_ring.at[r_slot, :, 0:half_i],
                ring_sem.at[r_slot],
            ).start()
            pltpu.make_async_copy(
                w1_u32.at[w_slot, :, pl.ds(u_col, half_i)],
                w_ring.at[r_slot, :, half_i:_I],
                ring_sem.at[r_slot],
            ).start()
            pltpu.make_async_copy(
                s1_ref.at[w_slot, :, pl.ds(g_col, half_i)],
                s_ring.at[r_slot, :, 0:half_i],
                ring_sem.at[r_slot],
            ).start()
            pltpu.make_async_copy(
                s1_ref.at[w_slot, :, pl.ds(u_col, half_i)],
                s_ring.at[r_slot, :, half_i:_I],
                ring_sem.at[r_slot],
            ).start()

        @pl.when(part == 2)
        def _u2():
            pltpu.make_async_copy(
                w2_u32.at[w_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
            ).start()
            pltpu.make_async_copy(
                s2_ref.at[w_slot], s_ring.at[r_slot], ring_sem.at[r_slot]
            ).start()

    def wait_ring_slot(r_slot):
        pltpu.make_async_copy(
            w_ring.at[r_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
        ).wait()
        pltpu.make_async_copy(
            s_ring.at[r_slot], s_ring.at[r_slot], ring_sem.at[r_slot]
        ).wait()

    # IMMEDIATELY prime circular weight ring while ICI dispatch DMAs land!
    n_units_total = num_exec * 3
    for ahead in range(n_slots - 1):
        @pl.when(ahead < n_units_total)
        def _prime_a2a(ahead=ahead):
            start_ring_unit(
                jnp.int32(ahead),
                exec_slot_smem[ahead // 3],
                jnp.int32(ahead % 3),
            )

    # Populate (gid = s * rows_own + r, weight) per active local expert & zero accum_recv_3d!
    def _pack_e(k_idx, _):
        le = exec_slot_smem[k_idx]

        def _ps(s, ptr):
            def _pr(r, p_in):
                wv = w_recv_smem[s, r, le]
                hit = wv > 0.0

                @pl.when(hit)
                def _wr():
                    packed_gid_smem[p_in] = s * rows_own + r
                    packed_w_smem[p_in] = wv

                return p_in + hit.astype(jnp.int32)

            return lax.fori_loop(0, rows_own, _pr, ptr)

        lax.fori_loop(0, ep, _ps, exec_start_smem[k_idx])
        return _

    lax.fori_loop(0, num_exec, _pack_e, jnp.int32(0))

    zero_rc = min(batch_size, 8)
    zeros_acc = jnp.zeros((zero_rc, 32, 128), dtype=jnp.float32)

    def _zero_acc_recv(ci, _):
        accum_recv_3d[pl.ds(ci * zero_rc, zero_rc), :, :] = zeros_acc
        return _

    lax.fori_loop(0, batch_size // zero_rc, _zero_acc_recv, jnp.int32(0))

    # Wait for exact count of deduplicated incoming x rows on disp_rsem!
    total_disp_recv = jnp.int32(0)
    for s in range(ep):
        n_from_s = jnp.where(s == me, jnp.int32(0), hdr_recv_smem[s, 1, 0])
        total_disp_recv = total_disp_recv + n_from_s

    disp_recv_flat = disp_recv_3d.reshape(batch_size, 32, 128)
    tmpl_disp_r = pltpu.make_async_copy(
        disp_recv_flat.at[pl.ds(0, 1)],
        disp_recv_flat.at[pl.ds(0, 1)],
        disp_rsem,
    )
    lax.fori_loop(
        0, total_disp_recv, lambda _, c: (tmpl_disp_r.wait(), c)[1], jnp.int32(0)
    )

    # 7. Circular-ring expert compute over deduplicated received rows!
    bd_rows_tuple = cfg.k16_blockdiag_rows if use_bd else ()
    m_buckets = (
        tuple(b for b in cfg.m_buckets if b > bd_rows_tuple[-1])
        if bd_rows_tuple
        else cfg.m_buckets
    )

    def _run_u01_std(part_idx: int, r_slot: jax.Array, m_rows: int):
        width = _I // 2
        col_offset = part_idx * width
        if is_k256:
            nc = _pick_k256_nc(m_rows, width, "wide")
            use_dual = m_rows <= 16
        else:
            use_dual = m_rows <= 32
            nc = _pick_k16_nc(m_rows, width, 512, use_dual=use_dual)
        write_m = 16 if m_rows == 8 else m_rows

        def _st(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            if use_dual:
                g, u = _fp4_dual_gmm_chunked_v2(
                    lhs_2d_buf,
                    w_ring.at[r_slot],
                    s_ring.at[r_slot],
                    m_rows,
                    _H,
                    n0,
                    width + n0,
                    nc,
                    is_k256,
                    scale_block,
                    unpack_mode=cfg.fp4_unpack_mode,
                )
                h = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)
                act_2d_buf[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
            else:
                g = _fp4_gmm_chunked_v2(
                    lhs_2d_buf,
                    w_ring.at[r_slot],
                    s_ring.at[r_slot],
                    m_rows,
                    _H,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    unpack_mode=cfg.fp4_unpack_mode,
                )
                act_2d_buf[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    jax.nn.silu(g).astype(jnp.bfloat16), write_m
                )
                u = _fp4_gmm_chunked_v2(
                    lhs_2d_buf,
                    w_ring.at[r_slot],
                    s_ring.at[r_slot],
                    m_rows,
                    _H,
                    width + n0,
                    nc,
                    is_k256,
                    scale_block,
                    unpack_mode=cfg.fp4_unpack_mode,
                )
                sg = act_2d_buf[pl.ds(0, write_m), col_out].astype(jnp.float32)
                act_2d_buf[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    (sg * u).astype(jnp.bfloat16), write_m
                )
            return carry

        n_st = width // nc
        if n_st == 1:
            _st(0, 0)
        elif n_st == 2:
            _st(0, 0)
            _st(1, 0)
        else:
            lax.fori_loop(0, n_st, _st, 0)

    def _run_u2_std(r_slot: jax.Array, m_rows: int) -> jax.Array:
        nc = (
            _pick_k256_nc(m_rows, _H, "wide")
            if is_k256
            else _pick_k16_nc(m_rows, _H, 512, use_dual=False)
        )
        sq_init = jnp.zeros((m_rows, 1), dtype=jnp.float32)

        def _st(j, sq_acc):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(n0, nc)
            d = _fp4_gmm_chunked_v2(
                act_2d_buf,
                w_ring.at[r_slot],
                s_ring.at[r_slot],
                m_rows,
                _I,
                n0,
                nc,
                is_k256,
                scale_block,
                unpack_mode=cfg.fp4_unpack_mode,
            ).astype(jnp.bfloat16)
            u = d.astype(jnp.float32) * gain_f32_buf[0:1, col_out]
            y_stage_buf[pl.ds(0, m_rows), col_out] = u
            return sq_acc + jnp.sum(u * u, axis=1, keepdims=True)

        sq_m = lax.fori_loop(0, _H // nc, _st, sq_init)
        return _pad_rows_to(sq_m, y_stage_buf.shape[0])

    def _run_u01_bd(part_idx: int, r_slot: jax.Array, bd_rows: int):
        width = _I // 2
        col_offset = part_idx * width
        nc = _pick_bd_nc(bd_rows, width, False)
        use_dual_bd = (bd_rows * nc) // 128 <= 4

        def _st(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            if use_dual_bd:
                g, u = _fp4_blockdiag_hoisted_dual_strip_v2(
                    masked_x_buf,
                    w_ring.at[r_slot],
                    s_ring.at[r_slot],
                    bd_rows,
                    n0,
                    width + n0,
                    nc,
                )
                h = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)
                act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(h, 16)
            else:
                g = _fp4_blockdiag_hoisted_strip_v2(
                    masked_x_buf, w_ring.at[r_slot], s_ring.at[r_slot], bd_rows, n0, nc
                )
                act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                    jax.nn.silu(g).astype(jnp.bfloat16), 16
                )
                u = _fp4_blockdiag_hoisted_strip_v2(
                    masked_x_buf,
                    w_ring.at[r_slot],
                    s_ring.at[r_slot],
                    bd_rows,
                    width + n0,
                    nc,
                )
                sg = act_2d_buf[pl.ds(0, 16), col_out][:bd_rows, :].astype(
                    jnp.float32
                )
                act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                    (sg * u).astype(jnp.bfloat16), 16
                )
            return carry

        lax.fori_loop(0, width // nc, _st, 0)

    def _run_u2_bd(r_slot: jax.Array, bd_rows: int) -> jax.Array:
        nc = _pick_bd_nc(bd_rows, _H, False)
        pad_f32 = 16 if bd_rows > 8 else 8

        def _st(j, sq_acc):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(n0, nc)
            d = _fp4_blockdiag_hoisted_strip_v2(
                masked_x_buf, w_ring.at[r_slot], s_ring.at[r_slot], bd_rows, n0, nc
            ).astype(jnp.bfloat16)
            u = d.astype(jnp.float32) * gain_f32_buf[0:1, col_out]
            y_stage_buf[pl.ds(0, pad_f32), col_out] = _pad_rows_to(u, pad_f32)
            return sq_acc + jnp.sum(u * u, axis=1, keepdims=True)

        sq_b = lax.fori_loop(
            0, _H // nc, _st, jnp.zeros((bd_rows, 1), dtype=jnp.float32)
        )
        return _pad_rows_to(sq_b, y_stage_buf.shape[0])

    def _expert_body(k_exp, _):
        row_start = exec_start_smem[k_exp]
        row_end = exec_start_smem[k_exp + 1]
        r_e = row_end - row_start
        n_rt = (r_e + tile_m - 1) // tile_m

        has_k1 = (k_exp + 1) < num_exec
        k1_slot = exec_slot_smem[jnp.where(has_k1, k_exp + 1, jnp.int32(0))]
        has_k2 = (k_exp + 2) < num_exec
        k2_slot = exec_slot_smem[jnp.where(has_k2, k_exp + 2, jnp.int32(0))]

        u_base = k_exp * 3
        s0 = _ring_slot(u_base, n_slots)
        s1 = _ring_slot(u_base + 1, n_slots)
        s2 = _ring_slot(u_base + 2, n_slots)

        def _rt_body(ti, _c):
            off = ti * tile_m
            base_pos = row_start + off
            cur_rows = jnp.minimum(tile_m, r_e - off)
            is_first = ti == 0
            is_last = ti == (n_rt - 1)
            do_pf_k1 = jnp.logical_and(is_last, has_k1)
            do_pf_k2 = jnp.logical_and(is_last, has_k2)

            @pl.when(
                jnp.logical_and(
                    jnp.logical_and(k_exp == 0, is_first),
                    (n_slots - 1) < n_units_total,
                )
            )
            def _pf_init():
                if n_slots == 3:
                    start_ring_unit(
                        jnp.int32(2), exec_slot_smem[0], jnp.int32(2)
                    )
                else:
                    start_ring_unit(jnp.int32(3), k1_slot, jnp.int32(0))

            # Gather cur_rows from disp_recv_flat into io_3d_buf / lhs_2d_buf
            def _g_row(i, _g):
                gid = packed_gid_smem[base_pos + i]
                io_3d_buf[pl.ds(i, 1), :, :] = disp_recv_flat[
                    pl.ds(gid, 1), :, :
                ]
                return _g

            lax.fori_loop(0, cur_rows, _g_row, jnp.int32(0))

            n_c16 = (cur_rows + 15) >> 4

            def _cp2d(ci, _x):
                rr = pl.multiple_of(ci * 16, 16)
                for h_idx in range(2):
                    lhs_2d_buf[pl.ds(rr, 16), pl.ds(h_idx * 2048, 2048)] = (
                        io_3d_buf[
                            pl.ds(rr, 16), pl.ds(h_idx * 16, 16), :
                        ].reshape(16, 2048)
                    )
                return _x

            if use_bd:
                max_bd = bd_rows_tuple[-1]

                @pl.when(cur_rows <= max_bd)
                def _p_bdx():
                    bi = jnp.int32(0)
                    for _r in bd_rows_tuple[:-1]:
                        bi = bi + (cur_rows > _r).astype(jnp.int32)
                    lax.switch(
                        bi,
                        [
                            (
                                lambda r=r: (
                                    _build_masked_x_from_3d(
                                        io_3d_buf, masked_x_buf, r
                                    ),
                                    jnp.int32(0),
                                )[1]
                            )
                            for r in bd_rows_tuple
                        ],
                    )

                @pl.when(cur_rows > max_bd)
                def _p_stdx():
                    lax.fori_loop(0, n_c16, _cp2d, jnp.int32(0))
            else:
                lax.fori_loop(0, n_c16, _cp2d, jnp.int32(0))

            def _call_u01(p_i: int, r_sl: jax.Array):
                if use_bd:
                    bi = jnp.int32(0)
                    for _r in bd_rows_tuple[:-1]:
                        bi = bi + (cur_rows > _r).astype(jnp.int32)
                    mi = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        mi = mi + (cur_rows > _b).astype(jnp.int32)
                    idx = jnp.where(
                        cur_rows <= bd_rows_tuple[-1],
                        bi,
                        len(bd_rows_tuple) + mi,
                    )
                    lax.switch(
                        idx,
                        [
                            (
                                lambda r=r: (
                                    _run_u01_bd(p_i, r_sl, r),
                                    jnp.int32(0),
                                )[1]
                            )
                            for r in bd_rows_tuple
                        ]
                        + [
                            (
                                lambda b=b: (
                                    _run_u01_std(p_i, r_sl, b),
                                    jnp.int32(0),
                                )[1]
                            )
                            for b in m_buckets
                        ],
                    )
                else:
                    mi = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        mi = mi + (cur_rows > _b).astype(jnp.int32)
                    lax.switch(
                        mi,
                        [
                            (
                                lambda b=b: (
                                    _run_u01_std(p_i, r_sl, b),
                                    jnp.int32(0),
                                )[1]
                            )
                            for b in m_buckets
                        ],
                    )

            @pl.when(is_first)
            def _w0():
                wait_ring_slot(s0)

            _call_u01(0, s0)

            @pl.when(do_pf_k1)
            def _pf0():
                start_ring_unit(
                    s0, k1_slot, jnp.int32(0 if n_slots == 3 else 1)
                )

            @pl.when(is_first)
            def _w1():
                wait_ring_slot(s1)

            _call_u01(1, s1)

            @pl.when(do_pf_k1)
            def _pf1():
                start_ring_unit(
                    s1, k1_slot, jnp.int32(1 if n_slots == 3 else 2)
                )

            if use_bd:
                max_bd = bd_rows_tuple[-1]

                @pl.when(cur_rows <= max_bd)
                def _p_bda():
                    bi = jnp.int32(0)
                    for _r in bd_rows_tuple[:-1]:
                        bi = bi + (cur_rows > _r).astype(jnp.int32)
                    lax.switch(
                        bi,
                        [
                            (
                                lambda r=r: (
                                    _build_masked_x_from_2d(
                                        act_2d_buf, masked_x_buf, r
                                    ),
                                    jnp.int32(0),
                                )[1]
                            )
                            for r in bd_rows_tuple
                        ],
                    )

            @pl.when(is_first)
            def _w2():
                wait_ring_slot(s2)

            if use_bd:
                bi = jnp.int32(0)
                for _r in bd_rows_tuple[:-1]:
                    bi = bi + (cur_rows > _r).astype(jnp.int32)
                mi = jnp.int32(0)
                for _b in m_buckets[:-1]:
                    mi = mi + (cur_rows > _b).astype(jnp.int32)
                idx2 = jnp.where(
                    cur_rows <= bd_rows_tuple[-1],
                    bi,
                    len(bd_rows_tuple) + mi,
                )
                sq_tile = lax.switch(
                    idx2,
                    [(lambda r=r: _run_u2_bd(s2, r)) for r in bd_rows_tuple]
                    + [(lambda b=b: _run_u2_std(s2, b)) for b in m_buckets],
                )
            else:
                mi = jnp.int32(0)
                for _b in m_buckets[:-1]:
                    mi = mi + (cur_rows > _b).astype(jnp.int32)
                sq_tile = lax.switch(
                    mi, [(lambda b=b: _run_u2_std(s2, b)) for b in m_buckets]
                )

            if n_slots == 3:
                @pl.when(do_pf_k1)
                def _pf2_3():
                    start_ring_unit(s2, k1_slot, jnp.int32(2))
            else:
                @pl.when(do_pf_k2)
                def _pf2_4():
                    start_ring_unit(s2, k2_slot, jnp.int32(0))

            # Fused RMSNorm + scatter into accum_recv_3d[gid] (zero dynamic_slice on VREGs)!
            del sq_tile
            n8 = (cur_rows + 7) >> 3

            def _ns8(q, carry_ns):
                i0 = pl.multiple_of(q << 3, 8)
                u0 = y_stage_buf[pl.ds(i0, 8), pl.ds(0, 2048)]
                u1 = y_stage_buf[pl.ds(i0, 8), pl.ds(2048, 2048)]
                u0_f = u0.astype(jnp.float32)
                u1_f = u1.astype(jnp.float32)
                sq8 = jnp.sum(u0_f * u0_f, axis=1, keepdims=True) + jnp.sum(
                    u1_f * u1_f, axis=1, keepdims=True
                )
                inv8 = lax.rsqrt(sq8 * (1.0 / _H) + 1e-8)
                if cfg.norm_f32_stage:
                    y0 = (u0_f * inv8).astype(jnp.bfloat16).reshape(8, 16, 128)
                    y1 = (u1_f * inv8).astype(jnp.bfloat16).reshape(8, 16, 128)
                else:
                    inv8_bf = inv8.astype(jnp.bfloat16)
                    y0 = (u0 * inv8_bf).reshape(8, 16, 128)
                    y1 = (u1 * inv8_bf).reshape(8, 16, 128)
                y8 = jnp.concatenate([y0, y1], axis=1)
                for s_i in range(8):
                    ii = i0 + s_i

                    @pl.when(ii < cur_rows)
                    def _add_r(s_i=s_i, ii=ii):
                        gid = packed_gid_smem[base_pos + ii]
                        wt = packed_w_smem[base_pos + ii]
                        accum_recv_3d[pl.ds(gid, 1), :, :] = (
                            accum_recv_3d[pl.ds(gid, 1), :, :]
                            + y8[s_i : s_i + 1, :, :].astype(jnp.float32) * wt
                        )

                return carry_ns

            lax.fori_loop(0, n8, _ns8, jnp.int32(0))
            return _c

        lax.fori_loop(0, n_rt, _rt_body, jnp.int32(0))
        return _

    lax.fori_loop(0, num_exec, _expert_body, jnp.int32(0))

    # 8. Rank-deduplicated peer-VMEM combine: send ONLY hit rows (s, r) back to owner s!
    comb_dtype = comb_send_3d.dtype
    for s in range(ep):
        blk = accum_recv_3d[pl.ds(s * rows_own, rows_own), :, :]
        comb_send_3d[s] = blk.astype(comb_dtype)

    comb_recv_3d[me] = comb_send_3d[me]
    total_comb_sent = jnp.int32(0)
    for s in range(ep):
        def _ret_row(r, c_cnt, s=s):
            hit_sr = (hdr_recv_smem[s, 0, r] != 0) & (s != me)

            @pl.when(hit_sr)
            def _do_ret():
                pltpu.make_async_remote_copy(
                    comb_send_3d.at[s, pl.ds(r, 1), :, :],
                    comb_recv_3d.at[me, pl.ds(r, 1), :, :],
                    comb_ssem,
                    comb_rsem,
                    device_id={ep_axis_name: jnp.int32(s)},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            return c_cnt + hit_sr.astype(jnp.int32)

        total_comb_sent = lax.fori_loop(0, rows_own, _ret_row, total_comb_sent)

    # Wait for exact count of incoming returned partials (= total_disp_sent!)
    comb_recv_flat = comb_recv_3d.reshape(batch_size, 32, 128)
    tmpl_comb_r = pltpu.make_async_copy(
        comb_recv_flat.at[pl.ds(0, 1)],
        comb_recv_flat.at[pl.ds(0, 1)],
        comb_rsem,
    )
    lax.fori_loop(
        0, total_disp_sent, lambda _, c: (tmpl_comb_r.wait(), c)[1], jnp.int32(0)
    )

    # Sum own + hit remote peers for each of own rows r in 0..rows_own-1!
    def _sum_own(r, _):
        acc_r = accum_recv_3d[pl.ds(me * rows_own + r, 1), :, :]
        for d in range(ep):
            has_d = (src_Slot_map_smem[d, 0, r] != 0) & (d != me)
            part_d = comb_recv_3d[d, pl.ds(r, 1), :, :].astype(jnp.float32)
            acc_r = acc_r + jnp.where(has_d, part_d, 0.0)
        owner_3d[pl.ds(r, 1), :, :] = acc_r.astype(jnp.bfloat16)
        return _

    lax.fori_loop(0, rows_own, _sum_own, jnp.int32(0))

    if not return_replicated:
        if rows_own >= 16:
            for r0 in range(0, rows_own, 16):
                output_ref[pl.ds(r0, 16), :] = owner_3d[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)
        else:
            out_own = owner_3d[...].reshape(rows_own, _H)
            output_ref[...] = _pad_rows_to(out_own, output_ref.shape[0])
    elif cfg.pair_ag_relay and ep == 8:
        core = me & jnp.int32(1)
        chip = me >> 1
        partner = jnp.bitwise_xor(me, jnp.int32(1))
        my_row_off = me * rows_own
        fan_3d_buf[pl.ds(my_row_off, rows_own), :, :] = owner_3d[...]

        def _ag_send(slot_rank, peer_rank, ssem_idx: int):
            r_off = slot_rank * rows_own
            op = pltpu.make_async_remote_copy(
                fan_3d_buf.at[pl.ds(r_off, rows_own), :, :],
                fan_3d_buf.at[pl.ds(r_off, rows_own), :, :],
                ag_ssem.at[ssem_idx],
                ag_psem.at[slot_rank],
                device_id={ep_axis_name: peer_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            return op

        def _ag_wait(slot_rank):
            pltpu.make_async_copy(owner_3d, owner_3d, ag_psem.at[slot_rank]).wait()

        far_ranks = [
            jnp.int32(2) * jnp.bitwise_xor(chip, jnp.int32(d)) + core
            for d in (1, 2, 3)
        ]
        ag_ops = [_ag_send(me, partner, 0)]
        for idx_f, fk in enumerate(far_ranks):
            ag_ops.append(_ag_send(me, fk, 1 + idx_f))
        for idx_f, fk in enumerate(far_ranks):
            _ag_wait(fk)
            ag_ops.append(_ag_send(fk, partner, 4 + idx_f))
        _ag_wait(partner)
        for fk in far_ranks:
            _ag_wait(jnp.bitwise_xor(fk, jnp.int32(1)))

        for r0 in range(0, batch_size, 16):
            output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                pl.ds(r0, 16), :, :
            ].reshape(16, _H)
        for op_a in ag_ops:
            op_a.wait_send()
    else:
        my_row_off = me * rows_own
        fan_3d_buf[pl.ds(my_row_off, rows_own), :, :] = owner_3d[...]
        fan_ops = []
        for dest in range(ep):
            fop = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_own), :, :],
                fan_ssem.at[dest],
                fan_rsem,
                device_id={ep_axis_name: jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            fan_ops.append((dest, fop))

            @pl.when(me != dest)
            def _sf(_op=fop):
                _op.start()

        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, (ep - 1) * rows_own), :, :],
            fan_3d_buf.at[pl.ds(0, (ep - 1) * rows_own), :, :],
            fan_rsem,
        ).wait()
        for r0 in range(0, batch_size, 16):
            output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                pl.ds(r0, 16), :, :
            ].reshape(16, _H)
        for dest, fop in fan_ops:
            @pl.when(me != dest)
            def _wf(_op=fop):
                _op.wait_send()

    # Wait on all outbound sends cleanly before exit.
    for d, op_b in b2o_ops:
        @pl.when(me != d)
        def _wb(_op=op_b):
            _op.wait_send()

    for d, op_h, op_w in hdr_ops:
        @pl.when(me != d)
        def _wh(_oh=op_h, _ow=op_w):
            _oh.wait_send()
            _ow.wait_send()

    tmpl_disp_s = pltpu.make_async_copy(
        disp_recv_flat.at[pl.ds(0, 1)],
        disp_recv_flat.at[pl.ds(0, 1)],
        disp_ssem,
    )
    lax.fori_loop(
        0, total_disp_sent, lambda _, c: (tmpl_disp_s.wait(), c)[1], jnp.int32(0)
    )
    tmpl_comb_s = pltpu.make_async_copy(
        comb_recv_flat.at[pl.ds(0, 1)],
        comb_recv_flat.at[pl.ds(0, 1)],
        comb_ssem,
    )
    lax.fori_loop(
        0, total_comb_sent, lambda _, c: (tmpl_comb_s.wait(), c)[1], jnp.int32(0)
    )


def fused_decode_ep_a2a_from_sharded(
    rank: jax.Array,
    logits_own: jax.Array,
    x_or_pre_col: jax.Array,
    bias: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    norm: jax.Array,
    pre_norm: jax.Array | None = None,
    *,
    input_is_col_sharded: bool = False,
    pre_eps: float = 1e-5,
    ep: int = 8,
    return_replicated: bool = False,
    ep_axis_name: str = "model",
    config: EpA2ADecodeConfig | None = None,
) -> jax.Array:
    """Sequence-Parallel Direct Peer-VMEM A2A Decode MoE (+ optional ``_blocks_to_owners``)."""
    rows_own = logits_own.shape[0]
    batch_size = rows_own * ep
    if batch_size not in (16, 32, 64, 128, 256):
        raise ValueError(f"Expected B in (16..256), got B={batch_size}")
    scale_block = _H // s1.shape[1]
    e_local = _E_TOTAL // ep
    if config is None:
        config = default_ep_a2a_config(batch_size, ep, scale_block)

    pad_own = max(8, ((rows_own + 7) // 8) * 8)
    logits_pad = _pad_rows_to(logits_own.astype(jnp.float32), pad_own)
    bias_2d = bias.reshape(1, _E_TOTAL).astype(jnp.float32)
    norm_2d = norm.reshape(1, _H)

    if input_is_col_sharded:
        if x_or_pre_col.shape != (batch_size, _H // ep):
            raise ValueError(
                f"Expected pre_col shape ({batch_size}, {_H // ep}), got {x_or_pre_col.shape}"
            )
        x_arg = x_or_pre_col
        pre_norm_2d = (
            pre_norm.reshape(1, _H)
            if pre_norm is not None
            else jnp.zeros((1, _H), dtype=jnp.bfloat16)
        )
    else:
        if x_or_pre_col.shape != (rows_own, _H):
            raise ValueError(
                f"Expected x_own shape ({rows_own}, {_H}), got {x_or_pre_col.shape}"
            )
        x_arg = _pad_rows_to(x_or_pre_col, pad_own)
        pre_norm_2d = None

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    n_slots = 4 if config.ring_mode.startswith("ring4") else 3
    nqb = _H // scale_block
    tile_m = config.tile_m
    use_bd = (scale_block == 16) and bool(config.k16_blockdiag_rows)
    max_packed = max(128, batch_size * _TOPK)
    comb_dtype = jnp.bfloat16 if config.combine_bf16 else jnp.float32

    scratch_shapes: list = []
    if input_is_col_sharded:
        scratch_shapes.extend(
            [
                vmem((ep, pad_own, _H // ep), jnp.bfloat16),
                vmem((ep, pad_own, _H // ep), jnp.bfloat16),
                sem((ep,)),
                sem,
            ]
        )

    scratch_shapes.extend(
        [
            vmem((rows_own, 32, 128), jnp.bfloat16),  # own_x_3d
            vmem((ep, 8, 128), jnp.int32),  # hdr_send_vmem
            vmem((ep, 8, 128), jnp.int32),  # hdr_recv_vmem
            smem((ep, 8, 128), jnp.int32),  # hdr_recv_smem
            vmem((ep, pad_own, 128), jnp.float32),  # w_send_vmem
            vmem((ep, pad_own, 128), jnp.float32),  # w_recv_vmem
            smem((ep, pad_own, 128), jnp.float32),  # w_recv_smem
            sem((2 * ep,)),  # hdr_ssem
            sem,  # hdr_rsem
            vmem((ep, rows_own, 32, 128), jnp.bfloat16),  # disp_recv_3d
            sem,  # disp_ssem
            sem,  # disp_rsem
            smem((e_local,), jnp.int32),  # exec_eid_smem
            smem((e_local,), jnp.int32),  # exec_slot_smem
            smem((e_local + 1,), jnp.int32),  # exec_start_smem
            smem((e_local,), jnp.int32),  # exec_wptr_smem
            smem((max_packed,), jnp.int32),  # packed_gid_smem
            smem((max_packed,), jnp.float32),  # packed_w_smem
            smem((ep, 8, 128), jnp.int32),  # src_Slot_map_smem (own hdr_send_smem!)
            vmem((n_slots, _H // _FP4_PACK, _I), jnp.uint32),  # w_ring
            vmem((n_slots, nqb, _I), s1.dtype),  # s_ring
            sem((n_slots,)),  # ring_sem
            sem((1,)),  # misc_sem
            vmem((tile_m, 32, 128), jnp.bfloat16),  # io_3d_buf
            vmem((tile_m, _H), jnp.bfloat16),  # lhs_2d_buf
            vmem((tile_m, _I), jnp.bfloat16),  # act_2d_buf
            vmem((max(16, tile_m), _H), jnp.float32),  # y_stage_buf
        ]
    )
    if use_bd:
        scratch_shapes.append(
            vmem((max(config.k16_blockdiag_rows) * 16, _H), jnp.bfloat16)
        )
    scratch_shapes.extend(
        [
            vmem((1, _H), jnp.float32),  # gain_f32_buf
            vmem((batch_size, 32, 128), jnp.float32),  # accum_recv_3d
            vmem((ep, rows_own, 32, 128), comb_dtype),  # comb_send_3d
            vmem((ep, rows_own, 32, 128), comb_dtype),  # comb_recv_3d
            sem,  # comb_ssem
            sem,  # comb_rsem
            vmem((rows_own, 32, 128), jnp.bfloat16),  # owner_3d
        ]
    )
    if return_replicated:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
        if config.pair_ag_relay and ep == 8:
            scratch_shapes.extend([sem((7,)), sem((8,))])
        else:
            scratch_shapes.extend([sem((ep,)), sem])

    in_specs = (
        [v_spec, v_spec, v_spec, v_spec, h_spec, h_spec, h_spec, h_spec, v_spec]
        if input_is_col_sharded
        else [v_spec, v_spec, v_spec, h_spec, h_spec, h_spec, h_spec, v_spec]
    )
    call_args = (
        (
            rank,
            logits_pad,
            bias_2d,
            x_arg,
            pre_norm_2d,
            w1,
            s1,
            w2,
            s2,
            norm_2d,
        )
        if input_is_col_sharded
        else (rank, logits_pad, bias_2d, x_arg, w1, s1, w2, s2, norm_2d)
    )
    out_rows_pad = batch_size if return_replicated else max(8, rows_own)
    res = pl.pallas_call(
        functools.partial(
            _ep_a2a_decode_kernel,
            ep=ep,
            batch_size=batch_size,
            scale_block=scale_block,
            input_is_col_sharded=input_is_col_sharded,
            pre_eps=pre_eps,
            return_replicated=return_replicated,
            ep_axis_name=ep_axis_name,
            cfg=config,
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows_pad, _H), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=tuple(in_specs),
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=f"muse_ep_a2a_ep{ep}_b{batch_size}_k{scale_block}",
    )(*call_args)
    return res if return_replicated else res[:rows_own]
