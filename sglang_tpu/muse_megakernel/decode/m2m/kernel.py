# SPDX-License-Identifier: Apache-2.0
"""Wave-9 MoE-to-MoE (`M2M`) Per-Layer Decode Megakernel for Muse Spark on TPU7x.

Fuses across the layer boundary `(l - 1) -> l`:
  1. Early KV-page ring prefetch (`seq 0..2`) at `t = 0` under Bridge-2b collectives.
  2. Wave-10 fast token-sharded Bridge-2b post-MoE core (`_run_tok_sharded_post_tp8_fast_core`)
     keeping `stream_shard [B // TP, 8192] f32` resident in VMEM (`s_buf`) straight into Bridge-1b.
  3. Half-column `XOR=1` `next_in` All-Gather + `W_qkv` dot + in-VMEM QKV RMSNorm / RoPE / FP8 KV.
  4. Overlapped HBM streaming of `W_gate (8 MiB)`, `Wo (8 MiB)`, `W_pre_pair (16 MiB, Slab A reuse)`,
     and `W_router (4 MiB)` underneath in-VMEM `RPAd` (`swabatch` / `one_block`, `ps=128/256/512`,
     `nod2` & zero-concat `dense2` even/odd, plus `ps=512` `1536/2048` intermediate gears).
  5. In-VMEM post-RPA per-head RMSNorm + `sigmoid(gate) * out` -> `gated_vmem`.
  6. Factored token-sharded Bridge-1b core consuming `s_buf` and `gated_vmem` directly in VMEM.
  7. Read-only `kv_cache` inside the collective call (`0` HLO `CopyInsertion` copies) paired with
     async emission of ready-to-scatter `512 B` KV rows (`nod2` or `dense2` merged) for one
     end-of-step `kv_write_batch_jax` launch across all `M2M` layers.
"""

# PEP 563 must stay off so cpu_export_jax_op sees concrete jax.Array annotations.

import functools
import math

from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
    _MASK_LOGIT,
    _bkv_col_iota,
    _broadcast_minor_128,
    _broadcast_minor_128_any,
    _load_bkv_fp8_chunk,
    _load_bkv_fp8_split,
    _load_new_kv_1row_f32,
    _load_new_kv_1row_u16,
    _roll_lanes_fast,
    _u8_to_fp8,
)
from sglang_tpu.attention.pallas.util import align_to, cdiv
from sglang_tpu.kernels.decode_collective_bridges import (
    _COL_STRIP,
    _DOT_TN,
    _dma_wait,
    _dot_fp8_scaled_to_ref,
    _fori_strips,
    _pick_strip,
    _signal_barrier,
    _wait_barrier,
)
from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    _ag_tok_sharded_half_col_xor1,
    _as_2d_f32,
    _as_dot_scale_f32,
    _dot_fp8_w8_slice_to_ref,
    _pallas_call_maybe_interpret,
    _run_in_vmem_qkv_epilogue,
    _run_tok_sharded_post_tp8_fast_core,
    _strip_unroll,
)

_LOG2_E = math.log2(math.e)
_M2M_CID_BASE = 52
_NUM_KV_SLOTS = 4


def _max_m2m_ring_pages(
    *,
    sliding_window: int | None,
    pages_per_seq: int,
    page_size: int,
) -> int:
    if sliding_window is not None:
        return cdiv(sliding_window + page_size - 1, page_size)
    return max(1, pages_per_seq)


def _m2m_seq_info(
    seq,
    kv_lens_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    *,
    sliding_window: int | None,
    page_size: int,
):
    import jax.numpy as jnp

    q_start = cu_q_lens_ref[seq]
    q_end = cu_q_lens_ref[seq + 1]
    kv_len = kv_lens_ref[seq]
    active = (q_end - q_start > 0) & (kv_len > 0)
    cache_kv = jnp.maximum(kv_len - jnp.int32(1), jnp.int32(0))
    if sliding_window is not None:
        first_key = jnp.maximum(
            cache_kv - jnp.int32(sliding_window - 1), jnp.int32(0)
        )
        base = (first_key // page_size) * page_size
    else:
        first_key = jnp.int32(0)
        base = jnp.int32(0)
    rel_first = first_key - base
    rel_end = jnp.where(active, jnp.maximum(cache_kv - base, jnp.int32(0)), jnp.int32(0))
    n_pages = (rel_end + page_size - 1) // page_size
    seq_pg_base = cdiv(cu_kv_lens_ref[seq], page_size) + (base // page_size)
    return q_start, active, kv_len, rel_first, rel_end, n_pages, seq_pg_base


def _start_fetch_kv_pages(
    seq_pg_base,
    n_pages,
    slot,
    page_indices_ref,
    kv_cache_ro_hbm,
    bkv_vmem_ref,
    rpa_smem_ref,
    bkv_sems,
    *,
    max_pages: int | None = None,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    num_slots = bkv_vmem_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    rpa_smem_ref[2 + num_slots + slot] = n_pages

    if max_pages is not None and max_pages <= 8:
        for i in range(max_pages):

            @pl.when(i < n_pages)
            def _(i=i):
                pg_idx = jnp.minimum(seq_pg_base + i, num_page_indices - 1)
                pg = page_indices_ref[pg_idx]
                pltpu.make_async_copy(
                    kv_cache_ro_hbm.at[pl.ds(pg, 1)],
                    bkv_vmem_ref.at[slot, pl.ds(i, 1)],
                    bkv_sems.at[slot],
                ).start()

        return

    def _dma_page(i, _):
        pg_idx = jnp.minimum(seq_pg_base + i, num_page_indices - 1)
        pg = page_indices_ref[pg_idx]
        pltpu.make_async_copy(
            kv_cache_ro_hbm.at[pl.ds(pg, 1)],
            bkv_vmem_ref.at[slot, pl.ds(i, 1)],
            bkv_sems.at[slot],
        ).start()

    lax.fori_loop(0, n_pages, _dma_page, None)


def _prologue_prefetch_kv_pages(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    kv_cache_ro_hbm,
    bkv_vmem_ref,
    rpa_smem_ref,
    bkv_sems,
    *,
    sliding_window: int | None,
    page_size: int,
    num_slots: int = _NUM_KV_SLOTS,
) -> None:
    """Kick off all `num_slots` (`0..3`) sequence KV page DMAs at `t = 0`."""
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    max_num_seqs = kv_lens_ref.shape[0]
    max_ring_pages = bkv_vmem_ref.shape[1]
    n_active = jnp.clip(distribution_ref[0], 0, max_num_seqs)
    rpa_smem_ref[0] = jnp.int32(0)
    for s_z in range(2 * num_slots):
        rpa_smem_ref[2 + s_z] = jnp.int32(0)

    for s_init in range(num_slots):

        @pl.when(s_init < n_active)
        def _(si=s_init):
            _, _, _, _, _, n_pg, pg_base = _m2m_seq_info(
                si,
                kv_lens_ref,
                cu_q_lens_ref,
                cu_kv_lens_ref,
                sliding_window=sliding_window,
                page_size=page_size,
            )
            _start_fetch_kv_pages(
                pg_base,
                n_pg,
                si,
                page_indices_ref,
                kv_cache_ro_hbm,
                bkv_vmem_ref,
                rpa_smem_ref,
                bkv_sems,
                max_pages=max_ring_pages,
            )


def _run_in_vmem_rpad_and_gate(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    q_vmem_ref,  # bf16[B, 16, 128]
    kv_merged_vmem_ref,  # fp8[B_pad32, 1, 4, 128]
    kv_cache_ro_hbm,  # fp8[num_pages, phys_page_rows, 1, 4, 128] (read-only inside M2M!)
    out_kvw_rows_ref,  # VMEM output fp8[B, 1, 4, 128] scattered by _kvw_1layer_jax (0 HLO copies!)
    bkv_vmem_ref,  # fp8[num_slots, max_ring_pages, phys_page_rows, 1, 4, 128]
    o_vmem_ref,  # bf16[B, 16, 128]
    next_in_vmem,  # bf16[B, 8192]
    w_gate_buf,  # fp8[8192, 1024]
    w_gate_sem,
    next_s_gate_ref,  # f32[1, 1024]
    gate_vmem_ref,  # bf16[B, 1024]
    gated_vmem_ref,  # bf16[B, 1024]
    rpa_smem_ref,  # i32[2 + 2 * num_slots]
    bkv_sems,
    *,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float,
    v_scale: float,
    out_norm_eps: float,
    kv_pack_dense2: bool,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    early_prefetched: bool = True,
    pv_bf16: bool = True,
    interpret_mode: bool = False,
    on_rpad_kv_done_fn=None,
    init_sv_vmem_ref=None,
    unroll_quad: bool = False,
    mp5_uniform: bool = True,
    nq8: bool = True,
    u32_vmask: bool = True,
) -> None:
    """In-VMEM single-token `RPAd` (`4`-seq unrolled quad + VREG V-san + hoisted new-KV) + `W_gate`."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    max_num_tokens = q_vmem_ref.shape[0]
    max_num_seqs = kv_lens_ref.shape[0]
    num_slots = bkv_vmem_ref.shape[0]
    max_ring_pages = bkv_vmem_ref.shape[1]
    phys_page_rows = (
        kv_cache_ro_hbm.shape[2] if kv_cache_ro_hbm.ndim == 6 else kv_cache_ro_hbm.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    num_page_indices = page_indices_ref.shape[0]
    has_hi_heads = w_gate_buf.shape[1] > 512
    n_q_act = 8 if (nq8 and not has_hi_heads) else 16
    lane_128_lt_64 = lax.broadcasted_iota(jnp.int32, (1, 128), 1) < 64

    s_scale = float(sm_scale * k_scale)
    exp2_scale = float(s_scale * _LOG2_E)
    v_scale_val = float(v_scale)
    n_active = jnp.clip(distribution_ref[0], 0, max_num_seqs)

    if not early_prefetched:
        _prologue_prefetch_kv_pages(
            kv_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
            cu_kv_lens_ref,
            distribution_ref,
            kv_cache_ro_hbm,
            bkv_vmem_ref,
            rpa_smem_ref,
            bkv_sems,
            sliding_window=sliding_window,
            page_size=page_size,
            num_slots=num_slots,
        )

    if init_sv_vmem_ref is not None and sliding_window is None:
        b_pad = kv_merged_vmem_ref.shape[0]
        step_b = 32 if max_num_tokens >= 32 else max_num_tokens
        for b0 in range(0, max_num_tokens, step_b):
            cur_b = min(step_b, max_num_tokens - b0)
            if cur_b <= 0:
                break
            if interpret_mode:
                blk = kv_merged_vmem_ref[pl.ds(b0, step_b)].reshape(step_b, 4, 128)
                u8 = lax.bitcast_convert_type(blk, jnp.uint8)
                u32 = u8[:, 0, :].astype(jnp.uint32) | (
                    u8[:, 1, :].astype(jnp.uint32) << 8
                )
            else:
                u32_ref = kv_merged_vmem_ref.bitcast(jnp.uint32).reshape(b_pad, 128)
                u32 = u32_ref[pl.ds(b0, step_b)]
            k_f32 = _u8_to_fp8(
                u32.astype(jnp.uint8), interpret_mode=interpret_mode
            ).astype(jnp.float32)
            v_f32 = (
                _u8_to_fp8(
                    (u32 >> 8).astype(jnp.uint8), interpret_mode=interpret_mode
                ).astype(jnp.float32)
                * v_scale_val
            )
            q_blk = q_vmem_ref[pl.ds(b0, cur_b), pl.ds(0, n_q_act), :].astype(
                jnp.float32
            )
            s_blk = _broadcast_minor_128(
                jnp.sum(q_blk * k_f32[:cur_b, None, :], axis=2, keepdims=True)
            )
            v_blk = jnp.broadcast_to(v_f32[:cur_b, None, :], (cur_b, n_q_act, 128))
            init_sv_vmem_ref[0, pl.ds(b0, cur_b), pl.ds(0, n_q_act)] = s_blk
            init_sv_vmem_ref[1, pl.ds(b0, cur_b), pl.ds(0, n_q_act)] = v_blk

    first_inactive_row = cu_q_lens_ref[n_active]

    @pl.when(first_inactive_row < max_num_tokens)
    def _zero_inactive_o():
        z = jnp.zeros((o_vmem_ref.shape[1], 128), jnp.bfloat16)
        lax.fori_loop(
            first_inactive_row,
            max_num_tokens,
            lambda r, _: o_vmem_ref.__setitem__(r, z),
            None,
        )

    def _wait_fetch_pages_slot(slot):
        n_pages = rpa_smem_ref[2 + num_slots + slot]

        @pl.when(n_pages > 0)
        def _():
            dst = bkv_vmem_ref.at[slot, pl.ds(0, n_pages)]
            pltpu.make_async_copy(dst, dst, bkv_sems.at[slot]).wait()

    def _get_page_u32(slot, pg_idx):
        if interpret_mode:
            blk = bkv_vmem_ref[slot, pg_idx].reshape(phys_page_rows, 4, 128)
            u8 = lax.bitcast_convert_type(blk, jnp.uint8)
            if kv_pack_dense2:
                return (
                    u8[:, 0, :].astype(jnp.uint32)
                    | (u8[:, 1, :].astype(jnp.uint32) << 8)
                    | (u8[:, 2, :].astype(jnp.uint32) << 16)
                    | (u8[:, 3, :].astype(jnp.uint32) << 24)
                )
            return u8[:, 0, :].astype(jnp.uint32) | (
                u8[:, 1, :].astype(jnp.uint32) << 8
            )
        u32_ref = (
            bkv_vmem_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(max_ring_pages, phys_page_rows, 128)
        )
        return u32_ref[pg_idx]

    def _mask_v_fp8_vregs(v_f8, valid_mask):
        if interpret_mode:
            v_u8 = lax.bitcast_convert_type(v_f8, jnp.uint8)
            return lax.bitcast_convert_type(
                jnp.where(valid_mask, v_u8, jnp.uint8(0)), jnp.float8_e4m3fn
            )
        v_u8 = pltpu.bitcast(v_f8, jnp.uint8)
        return pltpu.bitcast(
            jnp.where(valid_mask, v_u8, jnp.uint8(0)), jnp.float8_e4m3fn
        )

    if not kv_pack_dense2:
        if out_kvw_rows_ref is not kv_merged_vmem_ref:
            out_kvw_rows_ref[...] = kv_merged_vmem_ref[pl.ds(0, max_num_tokens)]
    else:
        out_kvw_rows_ref[...] = jnp.zeros_like(out_kvw_rows_ref[...])

    def _record_kv_write_uncond(seq, q_row, _w_cur, kv_len, slot, prev_loc_tok):
        """Stage 512B `dense2` pair row into `out_kvw_rows_ref[q_row]` (`nod2` is 0-op!)."""
        if not kv_pack_dense2:
            return
        pos = kv_len - 1
        pair_r = (pos % page_size) // 2
        u16_new = _load_new_kv_1row_u16(
            kv_merged_vmem_ref, q_row, interpret_mode=interpret_mode
        )
        is_odd = (pos & 1) == 1
        loc_clamped = jnp.maximum(prev_loc_tok, 0)
        slot_pg = jnp.clip(loc_clamped // page_size, 0, max_ring_pages - 1)
        if interpret_mode:
            pg_u32 = _get_page_u32(slot, slot_pg)
            r_io = lax.broadcasted_iota(jnp.int32, (phys_page_rows, 128), 0)
            prev_u16 = jnp.sum(
                jnp.where(
                    r_io == pair_r,
                    (pg_u32 & jnp.uint32(0xFFFF)).astype(jnp.int32),
                    jnp.int32(0),
                ),
                axis=0,
                keepdims=True,
            ).astype(jnp.uint32)
        else:
            sub_base = (pair_r // 8) * 8
            sub_off = pair_r & 7
            u32_ref = (
                bkv_vmem_ref.bitcast(jnp.uint32)
                .at[slot]
                .reshape(max_ring_pages, phys_page_rows, 128)
            )
            sub_u32 = u32_ref[slot_pg, pl.ds(pl.multiple_of(sub_base, 8), 8)]
            r_sub = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
            prev_u16 = jnp.sum(
                jnp.where(
                    r_sub == sub_off,
                    (sub_u32 & jnp.uint32(0xFFFF)).astype(jnp.int32),
                    jnp.int32(0),
                ),
                axis=0,
                keepdims=True,
            ).astype(jnp.uint32)
        if not has_hi_heads:
            prev_u16 = jnp.where(lane_128_lt_64, prev_u16, jnp.uint32(0))
        packed_1x128 = jnp.where(
            is_odd, prev_u16 | (u16_new << jnp.uint32(16)), u16_new
        )
        if interpret_mode:
            b0 = packed_1x128.astype(jnp.uint8)
            b1 = (packed_1x128 >> 8).astype(jnp.uint8)
            b2 = (packed_1x128 >> 16).astype(jnp.uint8)
            b3 = (packed_1x128 >> 24).astype(jnp.uint8)
            row_f8 = lax.bitcast_convert_type(
                jnp.stack([b0, b1, b2, b3], axis=1)[:, None, :, :],
                jnp.float8_e4m3fn,
            )
            out_kvw_rows_ref[pl.ds(q_row, 1)] = row_f8
        else:
            aligned8 = (q_row // 8) * 8
            sub8 = q_row - aligned8
            u32_w = out_kvw_rows_ref.bitcast(jnp.uint32).reshape(-1, 128)
            cur8 = u32_w[pl.ds(pl.multiple_of(aligned8, 8), 8)]
            r_io = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
            u32_w[pl.ds(pl.multiple_of(aligned8, 8), 8)] = jnp.where(
                r_io == sub8, jnp.broadcast_to(packed_1x128, (8, 128)), cur8
            )

    def _record_kv_write(seq, q_row, active, kv_len, slot, prev_loc_tok):
        """Stage 512B `dense2` pair row in `out_kvw_rows_ref` when `active` (`nod2` is 0-op!)."""
        if not kv_pack_dense2:
            return

        @pl.when(active)
        def _():
            _record_kv_write_uncond(seq, q_row, q_row, kv_len, slot, prev_loc_tok)

    def _write_o_seq(r_idx, val_bf16):
        if o_vmem_ref.shape[1] == n_q_act:
            o_vmem_ref[r_idx] = val_bf16
        else:
            o_vmem_ref[r_idx, pl.ds(0, n_q_act)] = val_bf16

    def _compute_seq(
        slot,
        q_bf16,
        q_row,
        rel_first,
        rel_end,
        n_slot_pages: int,
    ):
        q_act = q_bf16[:n_q_act] if q_bf16.shape[0] > n_q_act else q_bf16
        if init_sv_vmem_ref is not None and sliding_window is None:
            s_new = init_sv_vmem_ref[0, q_row, pl.ds(0, n_q_act)]
            acc0 = init_sv_vmem_ref[1, q_row, pl.ds(0, n_q_act)]
        else:
            k_new_f32, v_new_f32 = _load_new_kv_1row_f32(
                kv_merged_vmem_ref, q_row, interpret_mode=interpret_mode
            )
            q_f32 = q_act.astype(jnp.float32)
            s_new = _broadcast_minor_128(
                jnp.sum(
                    q_f32 * jnp.broadcast_to(k_new_f32, (n_q_act, 128)),
                    axis=1,
                    keepdims=True,
                )
            )
            acc0 = jnp.broadcast_to(v_new_f32, (n_q_act, 128)) * v_scale_val
        w_tot = n_slot_pages * page_size
        q_dot = q_act
        s_new_eff = s_new
        exp2_sc_eff = exp2_scale

        if kv_pack_dense2 and w_tot >= 512 and (w_tot % 256 == 0):
            half_w = w_tot // 2
            if u32_vmask or not has_hi_heads:
                if interpret_mode:
                    blk_k = bkv_vmem_ref[slot, pl.ds(0, n_slot_pages)].reshape(
                        half_w, 4, 128
                    )
                    u8_k = lax.bitcast_convert_type(blk_k, jnp.uint8)
                    ke_u8 = u8_k[:, 0, :]
                    ko_u8 = u8_k[:, 2, :]
                    if not has_hi_heads:
                        ke_u8 = jnp.where(lane_128_lt_64, ke_u8, jnp.uint8(0))
                        ko_u8 = jnp.where(lane_128_lt_64, ko_u8, jnp.uint8(0))
                    k_e = _u8_to_fp8(ke_u8, interpret_mode=True)
                    k_o = _u8_to_fp8(ko_u8, interpret_mode=True)
                else:
                    u32_ref_k = (
                        bkv_vmem_ref.bitcast(jnp.uint32)
                        .at[slot]
                        .reshape(max_ring_pages * phys_page_rows, 128)
                    )
                    pair_u32_k = u32_ref_k[pl.ds(0, half_w)]
                    if not has_hi_heads:
                        pair_u32_k = jnp.where(
                            lane_128_lt_64, pair_u32_k, jnp.uint32(0)
                        )
                    k_e = _u8_to_fp8(pair_u32_k.astype(jnp.uint8), interpret_mode=False)
                    k_o = _u8_to_fp8(
                        lax.shift_right_logical(pair_u32_k, jnp.uint32(16)).astype(
                            jnp.uint8
                        ),
                        interpret_mode=False,
                    )
            else:
                k_e, k_o = _load_bkv_fp8_split(
                    bkv_vmem_ref,
                    slot,
                    0,
                    w_tot,
                    kind="k",
                    page_size=page_size,
                    interpret_mode=interpret_mode,
                )
            s_e = jnp.matmul(q_dot, k_e.T, preferred_element_type=jnp.float32)
            h_io = lax.broadcasted_iota(jnp.int32, (n_q_act, half_w), 1)
            c_e = 2 * h_io
            if sliding_window is not None:
                s_e = jnp.where((c_e >= rel_first) & (c_e < rel_end), s_e, _MASK_LOGIT)
            else:
                s_e = jnp.where(c_e < rel_end, s_e, _MASK_LOGIT)
            del c_e

            s_o = jnp.matmul(q_dot, k_o.T, preferred_element_type=jnp.float32)
            c_o = 2 * h_io + 1
            if sliding_window is not None:
                s_o = jnp.where((c_o >= rel_first) & (c_o < rel_end), s_o, _MASK_LOGIT)
            else:
                s_o = jnp.where(c_o < rel_end, s_o, _MASK_LOGIT)
            del c_o

            s_eo_max = jnp.maximum(s_e, s_o)
            m = jnp.maximum(
                s_new_eff,
                _broadcast_minor_128(jnp.max(s_eo_max, axis=1, keepdims=True)),
            )
            m_hw = _broadcast_minor_128_any(m, half_w)
            p_e = jnp.exp2((s_e - m_hw) * exp2_sc_eff)
            p_o = jnp.exp2((s_o - m_hw) * exp2_sc_eff)
            p_new = jnp.exp2((s_new_eff - m) * exp2_sc_eff)
            p_eo_sum = p_e + p_o
            l_sum = (
                _broadcast_minor_128(jnp.sum(p_eo_sum, axis=1, keepdims=True))
                + p_new
            )
            r_hw = lax.broadcasted_iota(jnp.int32, (half_w, 128), 0)
            r_ve = 2 * r_hw
            r_vo = r_ve + 1
            valid_ve = (
                (r_ve >= rel_first) & (r_ve < rel_end)
                if sliding_window is not None
                else (r_ve < rel_end)
            )
            valid_vo = (
                (r_vo >= rel_first) & (r_vo < rel_end)
                if sliding_window is not None
                else (r_vo < rel_end)
            )
            if not has_hi_heads:
                valid_ve = valid_ve & lane_128_lt_64
                valid_vo = valid_vo & lane_128_lt_64
            if u32_vmask:
                if interpret_mode:
                    blk_v = bkv_vmem_ref[slot, pl.ds(0, n_slot_pages)].reshape(
                        half_w, 4, 128
                    )
                    u8_v = lax.bitcast_convert_type(blk_v, jnp.uint8)
                    ve_u32 = jnp.where(
                        valid_ve, u8_v[:, 1, :].astype(jnp.uint32), jnp.uint32(0)
                    )
                    vo_u32 = jnp.where(
                        valid_vo, u8_v[:, 3, :].astype(jnp.uint32), jnp.uint32(0)
                    )
                else:
                    u32_ref_v = (
                        bkv_vmem_ref.bitcast(jnp.uint32)
                        .at[slot]
                        .reshape(max_ring_pages * phys_page_rows, 128)
                    )
                    pair_u32_v = u32_ref_v[pl.ds(0, half_w)]
                    ve_u32 = jnp.where(
                        valid_ve,
                        lax.shift_right_logical(pair_u32_v, jnp.uint32(8)),
                        jnp.uint32(0),
                    )
                    vo_u32 = jnp.where(
                        valid_vo,
                        lax.shift_right_logical(pair_u32_v, jnp.uint32(24)),
                        jnp.uint32(0),
                    )
                v_e = _u8_to_fp8(ve_u32.astype(jnp.uint8), interpret_mode=interpret_mode)
                v_o = _u8_to_fp8(vo_u32.astype(jnp.uint8), interpret_mode=interpret_mode)
            else:
                v_e, v_o = _load_bkv_fp8_split(
                    bkv_vmem_ref,
                    slot,
                    0,
                    w_tot,
                    kind="v",
                    page_size=page_size,
                    interpret_mode=interpret_mode,
                )
                v_e = _mask_v_fp8_vregs(v_e, valid_ve)
                v_o = _mask_v_fp8_vregs(v_o, valid_vo)
            pe_dot = p_e.astype(jnp.bfloat16) if pv_bf16 else p_e
            po_dot = p_o.astype(jnp.bfloat16) if pv_bf16 else p_o
            pv = jnp.matmul(pe_dot, v_e, preferred_element_type=jnp.float32) + jnp.matmul(
                po_dot, v_o, preferred_element_type=jnp.float32
            )
            if v_scale_val != 1.0:
                pv = pv * v_scale_val
            acc = pv + p_new * acc0
            if interpret_mode:
                return (acc / l_sum).astype(jnp.bfloat16)
            return (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)

        if not kv_pack_dense2 and (u32_vmask or not has_hi_heads):
            if interpret_mode:
                blk_k = bkv_vmem_ref[slot, pl.ds(0, n_slot_pages)].reshape(
                    w_tot, 4, 128
                )
                k_u8 = lax.bitcast_convert_type(blk_k[:, 0, :], jnp.uint8)
                if not has_hi_heads:
                    k_u8 = jnp.where(lane_128_lt_64, k_u8, jnp.uint8(0))
                k_all = _u8_to_fp8(k_u8, interpret_mode=True)
            else:
                u32_ref_k = (
                    bkv_vmem_ref.bitcast(jnp.uint32)
                    .at[slot]
                    .reshape(max_ring_pages * phys_page_rows, 128)
                )
                row_u32_k = u32_ref_k[pl.ds(0, w_tot)]
                if not has_hi_heads:
                    row_u32_k = jnp.where(lane_128_lt_64, row_u32_k, jnp.uint32(0))
                k_all = _u8_to_fp8(row_u32_k.astype(jnp.uint8), interpret_mode=False)
        else:
            k_all = _load_bkv_fp8_chunk(
                bkv_vmem_ref,
                slot,
                0,
                w_tot,
                kind="k",
                page_size=page_size,
                kv_pack_dense2=kv_pack_dense2,
                interpret_mode=interpret_mode,
            )
            if not has_hi_heads:
                k_all = _mask_v_fp8_vregs(
                    k_all, jnp.broadcast_to(lane_128_lt_64, (w_tot, 128))
                )
        s_all = jnp.matmul(q_dot, k_all.T, preferred_element_type=jnp.float32)
        k_col = _bkv_col_iota(0, w_tot, n_q_act, kv_pack_dense2=kv_pack_dense2)
        valid_s = (
            (k_col >= rel_first) & (k_col < rel_end)
            if sliding_window is not None
            else (k_col < rel_end)
        )
        s_all = jnp.where(valid_s, s_all, _MASK_LOGIT)
        use_early_pbf16 = pv_bf16 and w_tot >= 2560
        m = jnp.maximum(
            s_new_eff, _broadcast_minor_128(jnp.max(s_all, axis=1, keepdims=True))
        )
        m_sub = m[:, :1] if use_early_pbf16 else _broadcast_minor_128_any(m, w_tot)
        p_all = jnp.exp2((s_all - m_sub) * exp2_sc_eff)
        p_new = jnp.exp2((s_new_eff - m) * exp2_sc_eff)
        l_sum = _broadcast_minor_128(jnp.sum(p_all, axis=1, keepdims=True)) + p_new
        if use_early_pbf16:
            p_dot = p_all.astype(jnp.bfloat16)
            del p_all
        v_row_idx = lax.broadcasted_iota(jnp.int32, (w_tot, 128), 0)
        if kv_pack_dense2:
            # _load_bkv_fp8_chunk(d2) returns [even_rows, odd_rows] stacked along axis 0!
            half_w_c = w_tot // 2
            v_tok_idx = jnp.where(
                v_row_idx < half_w_c,
                2 * v_row_idx,
                2 * (v_row_idx - half_w_c) + 1,
            )
        else:
            v_tok_idx = v_row_idx
        valid_v = (
            (v_tok_idx >= rel_first) & (v_tok_idx < rel_end)
            if sliding_window is not None
            else (v_tok_idx < rel_end)
        )
        if not has_hi_heads:
            valid_v = valid_v & lane_128_lt_64
        if u32_vmask and not kv_pack_dense2:
            if interpret_mode:
                blk_v = bkv_vmem_ref[slot, pl.ds(0, n_slot_pages)].reshape(
                    w_tot, 4, 128
                )
                v_u8_raw = lax.bitcast_convert_type(blk_v[:, 1, :], jnp.uint8)
                v_u32 = jnp.where(
                    valid_v, v_u8_raw.astype(jnp.uint32), jnp.uint32(0)
                )
            else:
                u32_ref_v = (
                    bkv_vmem_ref.bitcast(jnp.uint32)
                    .at[slot]
                    .reshape(max_ring_pages * phys_page_rows, 128)
                )
                row_u32_v = u32_ref_v[pl.ds(0, w_tot)]
                v_u32 = jnp.where(
                    valid_v,
                    lax.shift_right_logical(row_u32_v, jnp.uint32(8)),
                    jnp.uint32(0),
                )
            v_all = _u8_to_fp8(v_u32.astype(jnp.uint8), interpret_mode=interpret_mode)
        else:
            v_all = _load_bkv_fp8_chunk(
                bkv_vmem_ref,
                slot,
                0,
                w_tot,
                kind="v",
                page_size=page_size,
                kv_pack_dense2=kv_pack_dense2,
                interpret_mode=interpret_mode,
            )
            v_all = _mask_v_fp8_vregs(v_all, valid_v)
        if not use_early_pbf16:
            p_dot = p_all.astype(jnp.bfloat16) if pv_bf16 else p_all
        pv = jnp.matmul(p_dot, v_all, preferred_element_type=jnp.float32)
        if v_scale_val != 1.0:
            pv = pv * v_scale_val
        acc = pv + p_new * acc0
        if interpret_mode:
            return (acc / l_sum).astype(jnp.bfloat16)
        return (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)

    def _scan_bounds(s_i, carry):
        mx_re, mn_re, all_act = carry
        _, act_i, _, _, re_i, _, _ = _m2m_seq_info(
            s_i,
            kv_lens_ref,
            cu_q_lens_ref,
            cu_kv_lens_ref,
            sliding_window=sliding_window,
            page_size=page_size,
        )
        return (
            jnp.where(act_i, jnp.maximum(mx_re, re_i), mx_re),
            jnp.where(act_i, jnp.minimum(mn_re, re_i), mn_re),
            all_act & act_i,
        )

    init_bounds = (jnp.int32(0), jnp.int32(1 << 29), jnp.bool_(True))
    if max_num_seqs <= 128:
        max_re_b, min_re_b, all_act_b = init_bounds
        q_prev = cu_q_lens_ref[0]
        kv_prev = cu_kv_lens_ref[0]
        for s_i in range(max_num_seqs):
            in_b_s = s_i < n_active
            q_next = cu_q_lens_ref[s_i + 1]
            kv_next = cu_kv_lens_ref[s_i + 1]
            kvl_s = kv_lens_ref[s_i]
            act_s = (q_next - q_prev == 1) & (kv_next > kv_prev) & (kvl_s > 0)
            q_prev = q_next
            kv_prev = kv_next
            cache_s = kvl_s - jnp.int32(1)
            if sliding_window is not None:
                fk_s = jnp.maximum(
                    cache_s - jnp.int32(sliding_window - 1), jnp.int32(0)
                )
                bs_s = (fk_s // page_size) * page_size
            else:
                bs_s = jnp.int32(0)
            re_s = jnp.where(act_s, jnp.maximum(cache_s - bs_s, jnp.int32(0)), jnp.int32(0))
            max_re_b = jnp.where(in_b_s & act_s, jnp.maximum(max_re_b, re_s), max_re_b)
            min_re_b = jnp.where(in_b_s & act_s, jnp.minimum(min_re_b, re_s), min_re_b)
            all_act_b = jnp.where(in_b_s, all_act_b & act_s, jnp.bool_(False))
    else:
        max_re_b, min_re_b, all_act_b = lax.fori_loop(
            0, n_active, _scan_bounds, init_bounds
        )
        all_act_b = all_act_b & (n_active == max_num_seqs)

    z_act_bf16 = jnp.zeros((n_q_act, 128), jnp.bfloat16)

    def _run_fixed_pages(w_pg: int):
        if (
            num_slots == 4
            and max_num_seqs >= 8
            and max_num_seqs % 4 == 0
            and max_num_seqs == max_num_tokens
            and 1 <= w_pg <= 8
        ):
            is_fast_uniform = (
                (
                    (n_active == max_num_seqs)
                    & all_act_b
                    & (cu_q_lens_ref[max_num_seqs] == max_num_tokens)
                )
                if mp5_uniform
                else (
                    (n_active == max_num_seqs)
                    & all_act_b
                    & (cu_q_lens_ref[max_num_seqs] == max_num_tokens)
                    & (min_re_b > (w_pg - 1) * page_size)
                )
            )

            @pl.when(is_fast_uniform)
            def _fast_uniform_path():
                def _consume_slot_static(seq, slot_c: int, prologue_wait: bool = False):
                    if prologue_wait:
                        _wait_fetch_pages_slot(slot_c)
                    else:
                        dst = bkv_vmem_ref.at[slot_c, pl.ds(0, w_pg)]
                        pltpu.make_async_copy(dst, dst, bkv_sems.at[slot_c]).wait()
                    kvl = kv_lens_ref[seq]
                    cache_kv = kvl - jnp.int32(1)
                    if sliding_window is not None:
                        fk = jnp.maximum(
                            cache_kv - jnp.int32(sliding_window - 1), jnp.int32(0)
                        )
                        bs = (fk // page_size) * page_size
                    else:
                        fk = jnp.int32(0)
                        bs = jnp.int32(0)
                    rf = fk - bs
                    re = cache_kv - bs
                    _record_kv_write_uncond(seq, seq, seq, kvl, slot_c, re - 1)
                    _write_o_seq(
                        seq,
                        _compute_seq(
                            slot_c,
                            q_vmem_ref[seq],
                            seq,
                            rf,
                            re,
                            w_pg,
                        ),
                    )

                def _prefetch_slot_static(fut_seq, slot_c: int):
                    kvl_f = kv_lens_ref[fut_seq]
                    cache_kv_f = kvl_f - jnp.int32(1)
                    bs_f = (
                        (
                            jnp.maximum(
                                cache_kv_f - jnp.int32(sliding_window - 1),
                                jnp.int32(0),
                            )
                            // page_size
                        )
                        * page_size
                        if sliding_window is not None
                        else jnp.int32(0)
                    )
                    pg_base_f = cdiv(cu_kv_lens_ref[fut_seq], page_size) + (
                        bs_f // page_size
                    )
                    n_pg_f = jnp.maximum(
                        cdiv(jnp.maximum(cache_kv_f - bs_f, 0), page_size), 1
                    )
                    for i in range(w_pg):
                        i_eff = jnp.minimum(i, n_pg_f - 1) if mp5_uniform else i
                        pg_idx = jnp.minimum(pg_base_f + i_eff, num_page_indices - 1)
                        pg = page_indices_ref[pg_idx]
                        pltpu.make_async_copy(
                            kv_cache_ro_hbm.at[pl.ds(pg, 1)],
                            bkv_vmem_ref.at[slot_c, pl.ds(i, 1)],
                            bkv_sems.at[slot_c],
                        ).start()

                if unroll_quad and w_pg <= 2:
                    for slot_c in range(4):
                        _consume_slot_static(slot_c, slot_c, prologue_wait=mp5_uniform)
                        _prefetch_slot_static(slot_c + 4, slot_c)
                    n_quads = max_num_seqs // 4

                    @pl.loop(1, n_quads - 1)
                    def _quad_steady(q_idx):
                        base_seq = q_idx * 4
                        for slot_c in range(4):
                            seq_c = base_seq + slot_c
                            _consume_slot_static(seq_c, slot_c)
                            _prefetch_slot_static(seq_c + 4, slot_c)

                    tail_base = max_num_seqs - 4
                    for slot_c in range(4):
                        _consume_slot_static(tail_base + slot_c, slot_c)
                elif mp5_uniform:

                    @pl.loop(0, max_num_seqs - 4)
                    def _steady_step_mp5(seq):
                        slot = seq & jnp.int32(3)
                        n_w = jnp.where(
                            seq < 4,
                            rpa_smem_ref[2 + num_slots + slot],
                            jnp.int32(w_pg),
                        )

                        @pl.when(n_w > 0)
                        def _():
                            dst = bkv_vmem_ref.at[slot, pl.ds(0, n_w)]
                            pltpu.make_async_copy(dst, dst, bkv_sems.at[slot]).wait()

                        kvl = kv_lens_ref[seq]
                        cache_kv = kvl - jnp.int32(1)
                        if sliding_window is not None:
                            fk = jnp.maximum(
                                cache_kv - jnp.int32(sliding_window - 1),
                                jnp.int32(0),
                            )
                            bs = (fk // page_size) * page_size
                        else:
                            fk = jnp.int32(0)
                            bs = jnp.int32(0)
                        rf = fk - bs
                        re = cache_kv - bs
                        _record_kv_write_uncond(seq, seq, seq, kvl, slot, re - 1)
                        _write_o_seq(
                            seq,
                            _compute_seq(
                                slot,
                                q_vmem_ref[seq],
                                seq,
                                rf,
                                re,
                                w_pg,
                            ),
                        )
                        _prefetch_slot_static(seq + 4, slot)

                    @pl.loop(max_num_seqs - 4, max_num_seqs)
                    def _drain_step_mp5(seq):
                        slot = seq & jnp.int32(3)
                        dst = bkv_vmem_ref.at[slot, pl.ds(0, w_pg)]
                        pltpu.make_async_copy(dst, dst, bkv_sems.at[slot]).wait()
                        kvl = kv_lens_ref[seq]
                        cache_kv = kvl - jnp.int32(1)
                        if sliding_window is not None:
                            fk = jnp.maximum(
                                cache_kv - jnp.int32(sliding_window - 1),
                                jnp.int32(0),
                            )
                            bs = (fk // page_size) * page_size
                        else:
                            fk = jnp.int32(0)
                            bs = jnp.int32(0)
                        rf = fk - bs
                        re = cache_kv - bs
                        _record_kv_write_uncond(seq, seq, seq, kvl, slot, re - 1)
                        _write_o_seq(
                            seq,
                            _compute_seq(
                                slot,
                                q_vmem_ref[seq],
                                seq,
                                rf,
                                re,
                                w_pg,
                            ),
                        )
                else:

                    @pl.loop(0, max_num_seqs - 4)
                    def _steady_step(seq):
                        slot = seq & jnp.int32(3)
                        dst = bkv_vmem_ref.at[slot, pl.ds(0, w_pg)]
                        pltpu.make_async_copy(dst, dst, bkv_sems.at[slot]).wait()
                        kvl = kv_lens_ref[seq]
                        cache_kv = kvl - jnp.int32(1)
                        if sliding_window is not None:
                            fk = jnp.maximum(
                                cache_kv - jnp.int32(sliding_window - 1),
                                jnp.int32(0),
                            )
                            bs = (fk // page_size) * page_size
                        else:
                            fk = jnp.int32(0)
                            bs = jnp.int32(0)
                        rf = fk - bs
                        re = cache_kv - bs
                        _record_kv_write_uncond(seq, seq, seq, kvl, slot, re - 1)
                        _write_o_seq(
                            seq,
                            _compute_seq(
                                slot,
                                q_vmem_ref[seq],
                                seq,
                                rf,
                                re,
                                w_pg,
                            ),
                        )
                        fut_seq = seq + 4
                        kvl_f = kv_lens_ref[fut_seq]
                        cache_kv_f = kvl_f - jnp.int32(1)
                        bs_f = (
                            (
                                jnp.maximum(
                                    cache_kv_f - jnp.int32(sliding_window - 1),
                                    jnp.int32(0),
                                )
                                // page_size
                            )
                            * page_size
                            if sliding_window is not None
                            else jnp.int32(0)
                        )
                        pg_base_f = cdiv(cu_kv_lens_ref[fut_seq], page_size) + (
                            bs_f // page_size
                        )
                        for i in range(w_pg):
                            pg_idx = jnp.minimum(pg_base_f + i, num_page_indices - 1)
                            pg = page_indices_ref[pg_idx]
                            pltpu.make_async_copy(
                                kv_cache_ro_hbm.at[pl.ds(pg, 1)],
                                bkv_vmem_ref.at[slot, pl.ds(i, 1)],
                                bkv_sems.at[slot],
                            ).start()

                    @pl.loop(max_num_seqs - 4, max_num_seqs)
                    def _drain_step(seq):
                        slot = seq & jnp.int32(3)
                        dst = bkv_vmem_ref.at[slot, pl.ds(0, w_pg)]
                        pltpu.make_async_copy(dst, dst, bkv_sems.at[slot]).wait()
                        kvl = kv_lens_ref[seq]
                        cache_kv = kvl - jnp.int32(1)
                        if sliding_window is not None:
                            fk = jnp.maximum(
                                cache_kv - jnp.int32(sliding_window - 1),
                                jnp.int32(0),
                            )
                            bs = (fk // page_size) * page_size
                        else:
                            fk = jnp.int32(0)
                            bs = jnp.int32(0)
                        rf = fk - bs
                        re = cache_kv - bs
                        _record_kv_write_uncond(seq, seq, seq, kvl, slot, re - 1)
                        _write_o_seq(
                            seq,
                            _compute_seq(
                                slot,
                                q_vmem_ref[seq],
                                seq,
                                rf,
                                re,
                                w_pg,
                            ),
                        )

                rpa_smem_ref[0] = jnp.int32(max_num_seqs)

            @pl.when(~is_fast_uniform)
            def _general_fallback():
                @pl.loop(0, n_active)
                def _seq_step_gen(seq):
                    slot = seq % num_slots
                    q_r, act, kvl, rf, re, _, _ = _m2m_seq_info(
                        seq,
                        kv_lens_ref,
                        cu_q_lens_ref,
                        cu_kv_lens_ref,
                        sliding_window=sliding_window,
                        page_size=page_size,
                    )
                    _wait_fetch_pages_slot(slot)
                    _record_kv_write(seq, q_r, act, kvl, slot, re - 1)

                    @pl.when(act)
                    def _():
                        _write_o_seq(
                            q_r,
                            _compute_seq(
                                slot, q_vmem_ref[q_r], q_r, rf, re, w_pg
                            ),
                        )

                    @pl.when(~act)
                    def _():
                        _write_o_seq(q_r, z_act_bf16)

                    fut_seq = seq + num_slots

                    @pl.when(fut_seq < n_active)
                    def _():
                        _, _, _, _, _, n_pg_f, pg_base_f = _m2m_seq_info(
                            fut_seq,
                            kv_lens_ref,
                            cu_q_lens_ref,
                            cu_kv_lens_ref,
                            sliding_window=sliding_window,
                            page_size=page_size,
                        )
                        _start_fetch_kv_pages(
                            pg_base_f,
                            n_pg_f,
                            slot,
                            page_indices_ref,
                            kv_cache_ro_hbm,
                            bkv_vmem_ref,
                            rpa_smem_ref,
                            bkv_sems,
                            max_pages=w_pg,
                        )

                    @pl.when(fut_seq >= n_active)
                    def _():
                        rpa_smem_ref[2 + num_slots + slot] = jnp.int32(0)

            return

        @pl.loop(0, n_active)
        def _seq_step(seq):
            slot = seq % num_slots
            q_r, act, kvl, rf, re, _, _ = _m2m_seq_info(
                seq,
                kv_lens_ref,
                cu_q_lens_ref,
                cu_kv_lens_ref,
                sliding_window=sliding_window,
                page_size=page_size,
            )
            _wait_fetch_pages_slot(slot)
            _record_kv_write(seq, q_r, act, kvl, slot, re - 1)

            @pl.when(act)
            def _():
                _write_o_seq(
                    q_r,
                    _compute_seq(
                        slot, q_vmem_ref[q_r], q_r, rf, re, w_pg
                    ),
                )

            @pl.when(~act)
            def _():
                _write_o_seq(q_r, z_act_bf16)

            fut_seq = seq + num_slots

            @pl.when(fut_seq < n_active)
            def _():
                _, _, _, _, _, n_pg_f, pg_base_f = _m2m_seq_info(
                    fut_seq,
                    kv_lens_ref,
                    cu_q_lens_ref,
                    cu_kv_lens_ref,
                    sliding_window=sliding_window,
                    page_size=page_size,
                )
                _start_fetch_kv_pages(
                    pg_base_f,
                    n_pg_f,
                    slot,
                    page_indices_ref,
                    kv_cache_ro_hbm,
                    bkv_vmem_ref,
                    rpa_smem_ref,
                    bkv_sems,
                    max_pages=None,
                )

            @pl.when(fut_seq >= n_active)
            def _():
                rpa_smem_ref[2 + num_slots + slot] = jnp.int32(0)

    # Deduplicated token-cap gears (includes ps=512 1536/2048 gears so 1/1K >1024 never over-pads!).
    cand_caps = (256, 512, 1024, 1536, 2048)
    gears: list[tuple[int, int]] = []
    seen_pgs: set[int] = set()
    for tok_cap in cand_caps:
        pg_cap = min(max_ring_pages, cdiv(tok_cap, page_size))
        if pg_cap < max_ring_pages and pg_cap not in seen_pgs:
            seen_pgs.add(pg_cap)
            gears.append((pg_cap * page_size, pg_cap))

    if not gears:
        _run_fixed_pages(max_ring_pages)
    else:
        prev_tok = 0
        for hi_tok, pg_g in gears:
            cond = (
                (max_re_b <= hi_tok)
                if prev_tok == 0
                else ((max_re_b > prev_tok) & (max_re_b <= hi_tok))
            )

            @pl.when(cond)
            def _gear(pg_g=pg_g):
                _run_fixed_pages(pg_g)

            prev_tok = hi_tok

        @pl.when(max_re_b > prev_tok)
        def _gear_full():
            _run_fixed_pages(max_ring_pages)

    if on_rpad_kv_done_fn is not None:
        on_rpad_kv_done_fn()

    # W_gate streamed during RPAd -> compute gate_vmem & fuse per-head RMSNorm + sigmoid(gate).
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem,
        w_gate_buf,
        next_s_gate_ref,
        0,
        gate_vmem_ref,
        0,
        rows=max_num_tokens,
        n_cols=w_gate_buf.shape[1],
    )

    has_hi_heads = w_gate_buf.shape[1] > 512
    if max_num_tokens <= 64:
        lane_lt_64 = lax.broadcasted_iota(jnp.int32, (max_num_tokens, 128), 1) < 64

        def _finish_pair_full(pair_bf16, col_off: int):
            pf = pair_bf16.astype(jnp.float32)
            sq = pf * pf
            sq_lo = jnp.sum(jnp.where(lane_lt_64, sq, 0.0), axis=1, keepdims=True)
            sq_hi = jnp.sum(jnp.where(lane_lt_64, 0.0, sq), axis=1, keepdims=True)
            inv_lo = _broadcast_minor_128(
                lax.rsqrt(sq_lo * (1.0 / 64.0) + float(out_norm_eps))
            )
            inv_hi = _broadcast_minor_128(
                lax.rsqrt(sq_hi * (1.0 / 64.0) + float(out_norm_eps))
            )
            out_p = (pf * jnp.where(lane_lt_64, inv_lo, inv_hi)).astype(jnp.bfloat16)
            g_p = gate_vmem_ref[:, pl.ds(col_off, 128)]
            sig_g = jax.nn.sigmoid(g_p.astype(jnp.float32)).astype(jnp.bfloat16)
            gated_vmem_ref[:, pl.ds(col_off, 128)] = sig_g * out_p

        for p in range(4):
            e_lo = o_vmem_ref[:, 2 * p, :]
            d_lo = o_vmem_ref[:, 2 * p + 1, :]
            pair_lo = jnp.where(
                lane_lt_64,
                e_lo,
                _roll_lanes_fast(d_lo, 64, interpret_mode=interpret_mode),
            )
            _finish_pair_full(pair_lo, p * 128)
            if has_hi_heads:
                e_hi = o_vmem_ref[:, 8 + 2 * p, :]
                d_hi = o_vmem_ref[:, 8 + 2 * p + 1, :]
                pair_hi = jnp.where(
                    lane_lt_64,
                    _roll_lanes_fast(e_hi, 64, interpret_mode=interpret_mode),
                    d_hi,
                )
                _finish_pair_full(pair_hi, 512 + p * 128)
        return

    one_trip = jnp.where(distribution_ref[2] >= 0, jnp.int32(1), jnp.int32(0))
    tile_r = 16
    n_gate_tiles = max_num_tokens // tile_r
    lane_lt_64 = lax.broadcasted_iota(jnp.int32, (tile_r, 128), 1) < 64

    def _finish_pair(pair_bf16, r_sl, col_off: int):
        pf = pair_bf16.astype(jnp.float32)
        sq = pf * pf
        sq_lo = jnp.sum(jnp.where(lane_lt_64, sq, 0.0), axis=1, keepdims=True)
        sq_hi = jnp.sum(jnp.where(lane_lt_64, 0.0, sq), axis=1, keepdims=True)
        inv_lo = _broadcast_minor_128(
            lax.rsqrt(sq_lo * (1.0 / 64.0) + float(out_norm_eps))
        )
        inv_hi = _broadcast_minor_128(
            lax.rsqrt(sq_hi * (1.0 / 64.0) + float(out_norm_eps))
        )
        out_p = (pf * jnp.where(lane_lt_64, inv_lo, inv_hi)).astype(jnp.bfloat16)
        g_p = gate_vmem_ref[r_sl, pl.ds(col_off, 128)]
        sig_g = jax.nn.sigmoid(g_p.astype(jnp.float32)).astype(jnp.bfloat16)
        gated_vmem_ref[r_sl, pl.ds(col_off, 128)] = sig_g * out_p

    def _gate_tile_step(t_i, _):
        r_sl = pl.ds(pl.multiple_of(t_i * tile_r, tile_r), tile_r)
        for p in range(4):
            e_lo = o_vmem_ref[r_sl, 2 * p, :]
            d_lo = o_vmem_ref[r_sl, 2 * p + 1, :]
            pair_lo = jnp.where(
                lane_lt_64,
                e_lo,
                _roll_lanes_fast(d_lo, 64, interpret_mode=interpret_mode),
            )
            _finish_pair(pair_lo, r_sl, p * 128)
            if has_hi_heads:
                e_hi = o_vmem_ref[r_sl, 8 + 2 * p, :]
                d_hi = o_vmem_ref[r_sl, 8 + 2 * p + 1, :]
                pair_hi = jnp.where(
                    lane_lt_64,
                    _roll_lanes_fast(e_hi, 64, interpret_mode=interpret_mode),
                    d_hi,
                )
                _finish_pair(pair_hi, r_sl, 512 + p * 128)
        return jnp.int32(0)

    lax.fori_loop(0, n_gate_tiles * one_trip, _gate_tile_step, jnp.int32(0))


def _llo_fence(distribution_ref, fn) -> None:
    """Opaque 1-trip `kWhile` region preventing DGO from hoisting VMEM loads across slab reuse."""
    import jax.numpy as jnp
    from jax import lax

    one = jnp.where(distribution_ref[2] >= 0, jnp.int32(1), jnp.int32(0))
    lax.fori_loop(0, one, lambda _, s: (fn(), s)[1], jnp.int32(0))


def _run_in_vmem_qkv_epilogue_any(
    qkv_2d_ref,  # bf16[B, 1280] (TP=8) or bf16[B, 640] (TP=16)
    cos_sin_vmem_ref,
    q_out_vmem_ref,  # bf16[B, 16, 128]
    kv_out_vmem_ref,  # fp8[B, 1, 4, 128]
    *,
    norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
    distribution_ref=None,
) -> None:
    """Zero-spill `fori_loop` (`tile_r=16`) QKV RMSNorm + RoPE + FP8 KV packing (`TP=8` & `TP=16`)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    qkv_cols = qkv_2d_ref.shape[1]
    if qkv_cols not in (640, 1280):
        raise ValueError(f"Unsupported qkv_cols={qkv_cols}")
    is_tp8 = qkv_cols == 1280
    if is_tp8 and qkv_2d_ref.shape[0] <= 64:
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

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    fp8_min = float(jnp.finfo(jnp.float8_e4m3fn).min)
    b_rows = qkv_2d_ref.shape[0]
    tile_r = 16 if b_rows >= 16 else 8
    n_tiles = b_rows // tile_r
    one_trip = (
        jnp.where(distribution_ref[2] >= 0, jnp.int32(1), jnp.int32(0))
        if distribution_ref is not None
        else jnp.int32(1)
    )
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
            jnp.sum(jnp.where(mask_lo, x2, 0.0), axis=1, keepdims=True),
            (tile_r, 128),
        ) * (1.0 / 64.0)
        ms_hi = jnp.broadcast_to(
            jnp.sum(jnp.where(mask_lo, 0.0, x2), axis=1, keepdims=True),
            (tile_r, 128),
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
    z_u8 = jnp.zeros((tile_r, 128), jnp.uint8)

    def _qkv_tile_step(t_i, _):
        rs = pl.ds(pl.multiple_of(t_i * tile_r, tile_r), tile_r)
        if has_rope:
            cos_128 = cos_sin_vmem_ref[rs, 0, :]
            sin_s128 = cos_sin_vmem_ref[rs, 1, :]
        else:
            cos_128 = None
            sin_s128 = None

        q_heads: list = []
        n_q_pairs = 8 if is_tp8 else 4
        for p_idx in range(n_q_pairs):
            qp = qkv_2d_ref[rs, pl.ds(p_idx * 128, 128)]
            qp = _norm_pair(qp, query_scale)
            if has_rope:
                qp = _rope_pair(qp, cos_128, sin_s128)
            qp_swapped = _roll(qp, 64)
            if p_idx < 4:
                q_heads.append(jnp.where(mask_lo, qp, z_bf16))
                q_heads.append(jnp.where(mask_lo, qp_swapped, z_bf16))
            else:
                q_heads.append(jnp.where(mask_lo, z_bf16, qp_swapped))
                q_heads.append(jnp.where(mask_lo, z_bf16, qp))
        if not is_tp8 and q_out_vmem_ref.shape[1] > 8:
            for _ in range(q_out_vmem_ref.shape[1] - 8):
                q_heads.append(z_bf16)
        q_out_vmem_ref[rs] = jnp.stack(q_heads, axis=1)

        if is_tp8:
            kp = qkv_2d_ref[rs, pl.ds(8 * 128, 128)]
            vp = qkv_2d_ref[rs, pl.ds(9 * 128, 128)]
            kp = _norm_pair(kp, 1.0)
            if has_rope:
                kp = _rope_pair(kp, cos_128, sin_s128)
            k_u8 = _quant_fp8_u8(kp, inv_k_scale)
            v_u8 = _quant_fp8_u8(vp, inv_v_scale)
        else:
            kv_128 = qkv_2d_ref[rs, pl.ds(512, 128)]
            kp = _norm_pair(kv_128, 1.0)
            if has_rope:
                kp = _rope_pair(kp, cos_128, sin_s128)
            vp = _roll(kv_128, 64)
            k_u8 = jnp.where(mask_lo, _quant_fp8_u8(kp, inv_k_scale), z_u8)
            v_u8 = jnp.where(mask_lo, _quant_fp8_u8(vp, inv_v_scale), z_u8)

        if interpret_mode:
            packed_u8 = jnp.stack([k_u8, v_u8, z_u8, z_u8], axis=1)[:, None, :, :]
            kv_out_vmem_ref[rs] = lax.bitcast_convert_type(
                packed_u8, jnp.float8_e4m3fn
            )
        else:
            kv_u32 = k_u8.astype(jnp.uint32) | (v_u8.astype(jnp.uint32) << 8)
            kv_out_u32_ref = kv_out_vmem_ref.bitcast(jnp.uint32).reshape(
                kv_out_vmem_ref.shape[0], 128
            )
            kv_out_u32_ref[rs] = kv_u32
        return jnp.int32(0)

    if b_rows <= 64:
        for t_i in range(n_tiles):
            _qkv_tile_step(t_i, jnp.int32(0))
        return
    lax.fori_loop(0, n_tiles * one_trip, _qkv_tile_step, jnp.int32(0))


def _run_tok_sharded_b1_from_vmem_core(
    s_buf,  # f32[r_tok, 8192] already in VMEM!
    gated_vmem_ref,  # bf16[rows, 1024] already in VMEM!
    wo_buf,  # fp8[2, 1024, 4096] DMA already started!
    wo_sem,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_pair_buf,  # fp8[8192, 2048] DMA already started!
    w_pre_sem,
    w_pre_pair_scale_ref,
    w_router_buf,  # bf16/f32[8192, 256] DMA already started!
    w_router_sem,
    pre_exp_gain_ref,
    out_stream_shard_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
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
    me,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    need_wait_barrier: bool = False,
    distribution_ref=None,
) -> None:
    """Factored Bridge-1b core taking `s_buf` and `gated_vmem_ref` directly in VMEM."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_vmem_ref.shape[0]
    r_tok, hidden = s_buf.shape
    r_pair = 2 * r_tok
    two_pair = 2 * r_pair
    half = hidden // 2
    q_cols = half // 2
    pre_pair_cols = w_pre_pair_buf.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_buf.shape[1]
    n_chips = world_size // 2

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

    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    _dot_fp8_scaled_to_ref(
        gated_vmem_ref,
        wo_buf.at[0],
        wo_scale_ref,
        other_c0,
        send_half,
        0,
        rows=rows,
        n_cols=half,
    )

    if need_wait_barrier:
        _wait_barrier(world_size)
    op_c1a = pltpu.make_async_remote_copy(
        send_half,
        mate_recv,
        c1a_send.at[0],
        c1a_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1a.start()

    _dma_wait(wo_buf.at[1], wo_sem.at[1])
    _dot_fp8_scaled_to_ref(
        gated_vmem_ref,
        wo_buf.at[1],
        wo_scale_ref,
        own_c0,
        own_half,
        0,
        rows=rows,
        n_cols=half,
    )
    op_c1a.wait()

    def _sum_c1a(r0):
        rs = pl.ds(r0, bm_full)

        def _col_grp(g, _):
            base_s = g * u_half
            for sub in range(u_half):
                c_sl = pl.ds(
                    pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[
                    rs, c_sl
                ].astype(jnp.float32)
                own_half[rs, c_sl] = v.astype(own_half.dtype)
            return jnp.int32(0)

        lax.fori_loop(0, half_strips // u_half, _col_grp, jnp.int32(0))

    _fori_strips(rows, bm_full, _sum_c1a)

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
            va = own_half[ra, ca].astype(jnp.float32) + rs_a_recv[
                rl, ca
            ].astype(jnp.float32)
            vb = own_half[rb, cb].astype(jnp.float32) + rs_b_recv[
                rl, ca
            ].astype(jnp.float32)
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
            own_half[rc_chip, ca].astype(jnp.float32)
            + rs_a_recv[rl0, ca].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        vb = (
            own_half[rc_chip, cb].astype(jnp.float32)
            + rs_b_recv[rl0, ca].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        d2d_my_tok[:, ca] = lax.select(
            is_even, va[0:r_tok, :], va[r_tok:r_pair, :]
        ).astype(jnp.float32)
        d2d_send_tok[:, ca] = lax.select(
            is_even, va[r_tok:r_pair, :], va[0:r_tok, :]
        )
        d2d_my_tok[:, cb] = lax.select(
            is_even, vb[0:r_tok, :], vb[r_tok:r_pair, :]
        ).astype(jnp.float32)
        d2d_send_tok[:, cb] = lax.select(
            is_even, vb[r_tok:r_pair, :], vb[0:r_tok, :]
        )

    op_c1c = pltpu.make_async_remote_copy(
        d2d_send_tok,
        d2d_recv_tok,
        c1c_send.at[0],
        c1c_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1c.start()

    def _own_pa_sq_tok(r0):
        rs = pl.ds(r0, bm_tok)

        def _col_grp(g, acc):
            base_s = g * u_half
            for sub in range(u_half):
                c_l = pl.ds(
                    pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                bf = d2d_my_tok[rs, c_l]
                sq = bf * bf
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            return acc

        sq_tok_buf[rs, :] = lax.fori_loop(
            0,
            half_strips // u_half,
            _col_grp,
            jnp.zeros((bm_tok, 128), dtype=jnp.float32),
        )

    _fori_strips(r_tok, bm_tok, _own_pa_sq_tok)
    op_c1c.wait()

    # Update `s_buf` in-place in VMEM (saves 256 KiB scratch + eliminates HBM roundtrip!).
    def _finish_pa_and_pre_ffn_tok(r0):
        rs = pl.ds(r0, bm_tok)

        def _oth_sq(g, acc):
            base_s = g * u_half
            for sub in range(u_half):
                c_l = pl.ds(
                    pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                bf = d2d_recv_tok[rs, c_l].astype(jnp.float32)
                sq = bf * bf
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            return acc

        oth_sq = lax.fori_loop(
            0,
            half_strips // u_half,
            _oth_sq,
            jnp.zeros((bm_tok, 128), dtype=jnp.float32),
        )
        inv_pa = lax.rsqrt(
            jnp.sum(sq_tok_buf[rs, :] + oth_sq, axis=-1, keepdims=True)
            * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )

        def _res_both(g, acc):
            base_s = g * u_half
            for sub in range(u_half):
                s = base_s + sub
                c_l = pl.ds(
                    pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP
                )
                c_own = pl.ds(
                    pl.multiple_of(own_c0 + s * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                c_oth = pl.ds(
                    pl.multiple_of(other_c0 + s * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                ns_own = pa_alpha_ref[0:1, c_own] * s_buf[
                    rs, c_own
                ] + pa_beta_ref[0:1, c_own] * (d2d_my_tok[rs, c_l] * inv_pa)
                ns_oth = pa_alpha_ref[0:1, c_oth] * s_buf[
                    rs, c_oth
                ] + pa_beta_ref[0:1, c_oth] * (
                    d2d_recv_tok[rs, c_l].astype(jnp.float32) * inv_pa
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
                sq = v_own * v_own + v_oth * v_oth
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            return acc

        pff_sq = lax.fori_loop(
            0,
            half_strips // u_half,
            _res_both,
            jnp.zeros((bm_tok, 128), dtype=jnp.float32),
        )
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )

        def _pre_ffn_out(g, _):
            base_s = g * u_full
            for sub in range(u_full):
                c_g = pl.ds(
                    pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                sv = s_buf[rs, c_g]
                gn = pre_ff_gain_ref[0:1, c_g]
                rn = (
                    (sv * gn) * inv_pff
                    if pre_ff_scale_input
                    else (sv * inv_pff) * gn
                )
                if not is_r16:
                    router_my_tok[rs, c_g] = rn
                ffn_my_tok[rs, c_g] = rn.astype(jnp.bfloat16)
            return jnp.int32(0)

        lax.fori_loop(0, n_strips // u_full, _pre_ffn_out, jnp.int32(0))

    _fori_strips(r_tok, bm_tok, _finish_pa_and_pre_ffn_tok)
    pltpu.make_async_copy(s_buf, out_stream_shard_hbm, s_out_sem.at[0]).start()

    op_pair = pltpu.make_async_remote_copy(
        ffn_my_tok,
        ffn_mate_tok,
        pair_send.at[0],
        pair_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_pair.start()
    if not is_r16:
        op_rpair = pltpu.make_async_remote_copy(
            router_my_tok,
            router_mate_tok,
            rpair_send.at[0],
            rpair_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rpair.start()
        op_rpair.wait()
    op_pair.wait()

    for c0 in range(0, hidden, 512):
        c_sl = pl.ds(c0, 512)
        own_f = ffn_my_tok[:, c_sl]
        mate_f = ffn_mate_tok[:, c_sl]
        ffn_pair_buf[:, c_sl] = jnp.concatenate(
            [lax.select(is_even, own_f, mate_f), lax.select(is_even, mate_f, own_f)],
            axis=0,
        )
        if not is_r16:
            own_r = router_my_tok[:, c_sl]
            mate_r = router_mate_tok[:, c_sl]
            router_pair_buf[:, c_sl] = jnp.concatenate(
                [lax.select(is_even, own_r, mate_r), lax.select(is_even, mate_r, own_r)],
                axis=0,
            )

    _dma_wait(w_pre_pair_buf, w_pre_sem.at[0])

    def _run_wpre_pair_dot():
        _dot_fp8_w8_slice_to_ref(
            ffn_pair_buf,
            0,
            w_pre_pair_buf,
            w_pre_pair_scale_ref,
            0,
            pre_my_half,
            0,
            0,
            n_r=r_pair,
            n_cols=pre_pair_cols,
            align=16,
        )

    if distribution_ref is not None:
        _llo_fence(distribution_ref, _run_wpre_pair_dot)
    else:
        _run_wpre_pair_dot()

    op_pre = pltpu.make_async_remote_copy(
        pre_my_half,
        pre_mate_half,
        pre_send.at[0],
        pre_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_pre.start()

    _dma_wait(w_router_buf, w_router_sem.at[0])
    rmul = jnp.float32(router_multiplier)
    r_lhs = ffn_pair_buf if is_r16 else router_pair_buf
    if r_pair <= 64 and n_exp == 256:
        lg = jnp.dot(
            r_lhs[...], w_router_buf[...], preferred_element_type=jnp.float32
        )
        out_router_logits_ref[pl.ds(chip_r0, r_pair), :] = lg * rmul
    else:
        for ce in range(0, n_exp, _DOT_TN):
            c_sl = pl.ds(ce, _DOT_TN)
            lg = jnp.dot(
                r_lhs[...],
                w_router_buf[:, c_sl],
                preferred_element_type=jnp.float32,
            )
            out_router_logits_ref[pl.ds(chip_r0, r_pair), c_sl] = lg * rmul

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
                c_l = pl.ds(
                    pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP
                )
                c_own = pl.ds(
                    pl.multiple_of(own_pre_c0 + s * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                xf = pre_my_half[rs, c_l].astype(jnp.float32)
                if pre_exp_scale_input:
                    xf = xf * pre_exp_gain_ref[0:1, c_own]
                sq = xf * xf
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            return acc

        sq_pair_buf[rs, :] = lax.fori_loop(
            0,
            pre_half_strips // u_pre,
            _col_grp,
            jnp.zeros((bm_pair, 128), dtype=jnp.float32),
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
                c_l = pl.ds(
                    pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP
                )
                c_oth = pl.ds(
                    pl.multiple_of(other_pre_c0 + s * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                xf = pre_mate_half[rs_l, c_l].astype(jnp.float32)
                if pre_exp_scale_input:
                    xf = xf * pre_exp_gain_ref[0:1, c_oth]
                sq = xf * xf
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            return acc

        oth_sq = lax.fori_loop(
            0,
            pre_half_strips // u_pre,
            _oth_sq,
            jnp.zeros((bm_pair, 128), dtype=jnp.float32),
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
                c_l = pl.ds(
                    pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP
                )
                c_own = pl.ds(
                    pl.multiple_of(own_pre_c0 + s * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                c_oth = pl.ds(
                    pl.multiple_of(other_pre_c0 + s * _COL_STRIP, _COL_STRIP),
                    _COL_STRIP,
                )
                x_own = pre_my_half[rs_l, c_l].astype(jnp.float32)
                x_oth = pre_mate_half[rs_l, c_l].astype(jnp.float32)
                g_own = pre_exp_gain_ref[0:1, c_own]
                g_oth = pre_exp_gain_ref[0:1, c_oth]
                y_own = (
                    (x_own * g_own) * inv
                    if pre_exp_scale_input
                    else (x_own * inv) * g_own
                )
                y_oth = (
                    (x_oth * g_oth) * inv
                    if pre_exp_scale_input
                    else (x_oth * inv) * g_oth
                )
                out_expert_in_ref[rs_g, c_own] = y_own.astype(
                    out_expert_in_ref.dtype
                )
                out_expert_in_ref[rs_g, c_oth] = y_oth.astype(
                    out_expert_in_ref.dtype
                )
            return jnp.int32(0)

        lax.fori_loop(0, pre_half_strips // u_pre, _out_both, jnp.int32(0))

    _fori_strips(r_pair, bm_pair, _finish_pre_exp_norm)

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


def _m2m_tok_sh_tp8_kernel(
    # 5 SMEM scalar prefetches:
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    # Inputs + Outputs + Scratches:
    *body_refs,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    qkv_norm_eps: float,
    query_scale: float,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float,
    v_scale: float,
    kv_pack_dense2: bool,
    layer_num_in_bundle: int,
    layer_offset_in_bundle: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    idx = 0
    stream_shard_hbm = body_refs[idx]
    moe_out_ref = body_refs[idx + 1]
    w_post_hbm = body_refs[idx + 2]
    s_post_ref = body_refs[idx + 3]
    post_ff_gain_ref = body_refs[idx + 4]
    pf_alpha_ref = body_refs[idx + 5]
    pf_beta_ref = body_refs[idx + 6]
    next_in_gain_ref = body_refs[idx + 7]
    next_w_qkv_hbm = body_refs[idx + 8]
    next_s_qkv_ref = body_refs[idx + 9]
    next_w_gate_hbm = body_refs[idx + 10]
    next_s_gate_ref = body_refs[idx + 11]
    idx += 12
    if has_rope:
        cos_sin_ref = body_refs[idx]
        idx += 1
    else:
        cos_sin_ref = None
    kv_cache_ro_hbm = body_refs[idx]
    wo_hbm = body_refs[idx + 1]
    wo_scale_ref = body_refs[idx + 2]
    pa_alpha_ref = body_refs[idx + 3]
    pa_beta_ref = body_refs[idx + 4]
    pre_ff_gain_ref = body_refs[idx + 5]
    w_pre_pair_hbm = body_refs[idx + 6]
    w_pre_pair_scale_ref = body_refs[idx + 7]
    w_router_hbm = body_refs[idx + 8]
    pre_exp_gain_ref = body_refs[idx + 9]
    idx += 10

    # Outputs (4 — `out_kvw_rows_ref` is scattered in-place by `_kvw_1layer_jax` with 0 HLO copies!):
    out_stream_shard_hbm = body_refs[idx]
    out_router_logits_ref = body_refs[idx + 1]
    out_expert_in_ref = body_refs[idx + 2]
    out_kvw_rows_ref = body_refs[idx + 3]
    idx += 4

    # Weight & KV ring slabs:
    slab_a = body_refs[idx]  # fp8[8192, 2048]: w_qkv[:, :1280] -> w_pre_pair
    slab_b = body_refs[idx + 1]  # fp8[8192, 1024]: w_post[:4096, :] -> w_gate
    wo_buf = body_refs[idx + 2]  # fp8[2, 1024, 4096]
    w_router_buf = body_refs[idx + 3]  # bf16[8192, 256]
    bkv_vmem_ref = body_refs[idx + 4]
    idx += 5

    # Shared & per-phase activation scratches:
    s_buf = body_refs[idx]
    next_in_my_tok = body_refs[idx + 1]
    next_in_mate_half = body_refs[idx + 2]
    next_in_vmem = body_refs[idx + 3]
    qkv_vmem = body_refs[idx + 4]
    q_vmem_ref = body_refs[idx + 5]
    kv_merged_vmem_ref = body_refs[idx + 6]
    o_vmem_ref = body_refs[idx + 7]
    gate_vmem_ref = body_refs[idx + 8]
    gated_vmem_ref = body_refs[idx + 9]
    mate_tok_buf = body_refs[idx + 10]
    moe_full_buf = body_refs[idx + 11]
    post_shard = body_refs[idx + 12]
    mate_post_shard = body_refs[idx + 13]
    chip_cols_buf = body_refs[idx + 14]
    post_pair_full = body_refs[idx + 15]
    send_half = body_refs[idx + 16]
    mate_recv = body_refs[idx + 17]
    own_half = body_refs[idx + 18]
    rs_a_recv = body_refs[idx + 19]
    rs_b_recv = body_refs[idx + 20]
    d2d_my_tok = body_refs[idx + 21]
    d2d_send_tok = body_refs[idx + 22]
    d2d_recv_tok = body_refs[idx + 23]
    sq_tok_buf = body_refs[idx + 24]
    ffn_my_tok = body_refs[idx + 25]
    ffn_mate_tok = body_refs[idx + 26]
    ffn_pair_buf = body_refs[idx + 27]
    router_my_tok = body_refs[idx + 28]
    router_mate_tok = body_refs[idx + 29]
    router_pair_buf = body_refs[idx + 30]
    pre_my_half = body_refs[idx + 31]
    pre_mate_half = body_refs[idx + 32]
    sq_pair_buf = body_refs[idx + 33]
    init_sv_vmem_ref = body_refs[idx + 34]
    rpa_smem_ref = body_refs[idx + 35]
    idx += 36

    # Semaphores (`<= 42` total DMA semaphores; causally ordered reuse across b2 -> b1):
    w_post_sem = body_refs[idx]
    w_qkv_sem = body_refs[idx + 1]
    w_gate_sem = body_refs[idx + 2]
    wo_sem = body_refs[idx + 3]
    w_pre_sem = body_refs[idx + 4]
    w_router_sem = body_refs[idx + 5]
    s_in_sem = body_refs[idx + 6]
    s_out_sem = body_refs[idx + 7]
    bkv_sems = body_refs[idx + 8]
    d2d_a_send = body_refs[idx + 9]
    d2d_a_recv = body_refs[idx + 10]
    d2d_b_send = body_refs[idx + 11]
    d2d_b_recv = body_refs[idx + 12]
    d2d_c_send = body_refs[idx + 13]
    d2d_c_recv = body_refs[idx + 14]
    d2d_d_send = body_refs[idx + 15]
    d2d_d_recv = body_refs[idx + 16]
    ici_a_send = body_refs[idx + 17]
    ici_a_recv = body_refs[idx + 18]
    ici_b_send = body_refs[idx + 19]
    ici_b_recv = body_refs[idx + 20]
    ici_c_send = body_refs[idx + 21]
    ici_c_recv = body_refs[idx + 22]
    rs1_send = body_refs[idx + 23]
    rs1_recv = body_refs[idx + 24]
    rs2_send = body_refs[idx + 25]
    rs2_recv = body_refs[idx + 26]

    r_tok = moe_out_ref.shape[0]
    rows = world_size * r_tok
    qkv_cols = next_w_qkv_hbm.shape[1]
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    phys_page_rows = (
        kv_cache_ro_hbm.shape[2] if kv_cache_ro_hbm.ndim == 6 else kv_cache_ro_hbm.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows

    me = lax.axis_index("model").astype(jnp.int32)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)

    w_post_buf = slab_b.at[pl.ds(0, w_post_hbm.shape[0]), :]
    w_qkv_buf = slab_a.at[:, pl.ds(0, qkv_cols)]

    # Phase 0 (`t = 0`): Signal barrier + start stream_shard, W_post, W_qkv, AND KV pages 0..2!
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(stream_shard_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    _prologue_prefetch_kv_pages(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        kv_cache_ro_hbm,
        bkv_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
        sliding_window=sliding_window,
        page_size=page_size,
        num_slots=bkv_vmem_ref.shape[0],
    )

    # Phase 1: Bridge-2b post-MoE core (`s_buf` stays in VMEM — zero HBM write/re-read!).
    _llo_fence(
        distribution_ref,
        lambda: _run_tok_sharded_post_tp8_fast_core(
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
            d2d_a_send,
            d2d_a_recv,
            ici_a_send,
            ici_a_recv,
            d2d_b_send,
            d2d_b_recv,
            ici_b_send,
            ici_b_recv,
            me=me,
            world_size=world_size,
            post_ff_eps=post_ff_eps,
            post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps,
            next_in_scale_input=next_in_scale_input,
            round_stream_bf16_for_norm=False,
            on_stream_updated_fn=None,
        ),
    )

    # `w_post_buf` is dead -> reuse `slab_b` for `W_gate` (defer `Wo`/`W_pre`/`W_router` past RPAd KV!).
    pltpu.make_async_copy(next_w_gate_hbm, slab_b, w_gate_sem.at[0]).start()

    _ag_tok_sharded_half_col_xor1(
        next_in_my_tok,
        next_in_mate_half,
        next_in_vmem,
        d2d_c_send,
        d2d_c_recv,
        ici_c_send,
        ici_c_recv,
        d2d_d_send,
        d2d_d_recv,
        me=me,
        world_size=world_size,
        during_d2d2_fn=lambda: _dma_wait(w_qkv_buf, w_qkv_sem.at[0]),
    )

    _llo_fence(
        distribution_ref,
        lambda: _dot_fp8_scaled_to_ref(
            next_in_vmem,
            w_qkv_buf,
            next_s_qkv_ref,
            0,
            qkv_vmem,
            0,
            rows=rows,
            n_cols=qkv_cols,
        ),
    )

    eff_kv_merged_ref = (
        out_kvw_rows_ref
        if (not kv_pack_dense2 and rows == kv_merged_vmem_ref.shape[0])
        else kv_merged_vmem_ref
    )
    _run_in_vmem_qkv_epilogue_any(
        qkv_vmem,
        cos_sin_ref,
        q_vmem_ref,
        eff_kv_merged_ref.at[pl.ds(0, rows)],
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        distribution_ref=distribution_ref,
    )

    def _start_wo_after_kv():
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
        ).start()
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
        ).start()

    # Phase 2: In-VMEM `RPAd` (zero weight HBM contention!) + `Wo` under `W_gate` dot + head-norm/gate!
    _run_in_vmem_rpad_and_gate(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        q_vmem_ref,
        eff_kv_merged_ref,
        kv_cache_ro_hbm,
        out_kvw_rows_ref,
        bkv_vmem_ref,
        o_vmem_ref,
        next_in_vmem,
        slab_b,
        w_gate_sem,
        next_s_gate_ref,
        gate_vmem_ref,
        gated_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        out_norm_eps=qkv_norm_eps,
        kv_pack_dense2=kv_pack_dense2,
        layer_num_in_bundle=layer_num_in_bundle,
        layer_offset_in_bundle=layer_offset_in_bundle,
        early_prefetched=True,
        interpret_mode=interpret_mode,
        on_rpad_kv_done_fn=_start_wo_after_kv,
        init_sv_vmem_ref=init_sv_vmem_ref,
    )

    # Stream `W_pre_pair (16 MiB)` + `W_router (4 MiB)` under `Wo` dot + 3-hop ICI RS (~16 us window)!
    pltpu.make_async_copy(w_pre_pair_hbm, slab_a, w_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    # Phase 3: Factored Bridge-1b core (`in_ici + in_d2d2` causally closed all b2 semaphores!).
    eff_send_half = (
        moe_full_buf if moe_full_buf.shape == (rows, half) else send_half
    )
    eff_ffn_pair_buf = (
        post_pair_full if post_pair_full.shape == (2 * r_tok, hidden) else ffn_pair_buf
    )
    _run_tok_sharded_b1_from_vmem_core(
        s_buf,
        gated_vmem_ref,
        wo_buf,
        wo_sem,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        slab_a,
        w_pre_sem,
        w_pre_pair_scale_ref,
        w_router_buf,
        w_router_sem,
        pre_exp_gain_ref,
        out_stream_shard_hbm,
        out_router_logits_ref,
        out_expert_in_ref,
        eff_send_half,
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
        eff_ffn_pair_buf,
        router_my_tok,
        router_mate_tok,
        router_pair_buf,
        pre_my_half,
        pre_mate_half,
        sq_pair_buf,
        s_out_sem,
        d2d_a_send,
        d2d_a_recv,
        rs1_send,
        rs1_recv,
        rs2_send,
        rs2_recv,
        d2d_b_send,
        d2d_b_recv,
        d2d_c_send,
        d2d_c_recv,
        d2d_d_send,
        d2d_d_recv,
        w_post_sem,  # reuse 1-slot DMA sem (w_post long completed!) as pre_send
        w_qkv_sem,   # reuse 1-slot DMA sem (w_qkv long completed!) as pre_recv
        ici_a_send,
        ici_a_recv,
        ici_b_send,
        ici_b_recv,
        me=me,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        need_wait_barrier=False,
        distribution_ref=distribution_ref,
    )


def _l0_a2m_tok_sh_tp8_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    *body_refs,
    world_size: int,
    qkv_norm_eps: float,
    query_scale: float,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float,
    v_scale: float,
    kv_pack_dense2: bool,
    layer_num_in_bundle: int,
    layer_offset_in_bundle: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Layer-0 fused `Dual-QKVG + QKV/RoPE epilogue + RPAd(0) + gate + Bridge-1b(0)`."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    idx = 0
    stream_shard_hbm = body_refs[idx]
    attn_in_ref = body_refs[idx + 1]
    w_qkv_hbm = body_refs[idx + 2]
    s_qkv_ref = body_refs[idx + 3]
    w_gate_hbm = body_refs[idx + 4]
    s_gate_ref = body_refs[idx + 5]
    idx += 6
    if has_rope:
        cos_sin_ref = body_refs[idx]
        idx += 1
    else:
        cos_sin_ref = None
    kv_cache_ro_hbm = body_refs[idx]
    wo_hbm = body_refs[idx + 1]
    wo_scale_ref = body_refs[idx + 2]
    pa_alpha_ref = body_refs[idx + 3]
    pa_beta_ref = body_refs[idx + 4]
    pre_ff_gain_ref = body_refs[idx + 5]
    w_pre_pair_hbm = body_refs[idx + 6]
    w_pre_pair_scale_ref = body_refs[idx + 7]
    w_router_hbm = body_refs[idx + 8]
    pre_exp_gain_ref = body_refs[idx + 9]
    idx += 10

    out_stream_shard_hbm = body_refs[idx]
    out_router_logits_ref = body_refs[idx + 1]
    out_expert_in_ref = body_refs[idx + 2]
    out_kvw_rows_ref = body_refs[idx + 3]
    idx += 4

    slab_a = body_refs[idx]
    w_gate_buf = body_refs[idx + 1]
    wo_buf = body_refs[idx + 2]
    w_router_buf = body_refs[idx + 3]
    bkv_vmem_ref = body_refs[idx + 4]
    s_buf = body_refs[idx + 5]
    qkv_vmem = body_refs[idx + 6]
    q_vmem_ref = body_refs[idx + 7]
    kv_merged_vmem_ref = body_refs[idx + 8]
    o_vmem_ref = body_refs[idx + 9]
    gate_vmem_ref = body_refs[idx + 10]
    gated_vmem_ref = body_refs[idx + 11]
    send_half = body_refs[idx + 12]
    mate_recv = body_refs[idx + 13]
    own_half = body_refs[idx + 14]
    rs_a_recv = body_refs[idx + 15]
    rs_b_recv = body_refs[idx + 16]
    d2d_my_tok = body_refs[idx + 17]
    d2d_send_tok = body_refs[idx + 18]
    d2d_recv_tok = body_refs[idx + 19]
    sq_tok_buf = body_refs[idx + 20]
    ffn_my_tok = body_refs[idx + 21]
    ffn_mate_tok = body_refs[idx + 22]
    ffn_pair_buf = body_refs[idx + 23]
    router_my_tok = body_refs[idx + 24]
    router_mate_tok = body_refs[idx + 25]
    router_pair_buf = body_refs[idx + 26]
    pre_my_half = body_refs[idx + 27]
    pre_mate_half = body_refs[idx + 28]
    sq_pair_buf = body_refs[idx + 29]
    init_sv_vmem_ref = body_refs[idx + 30]
    rpa_smem_ref = body_refs[idx + 31]
    idx += 32

    w_qkv_sem = body_refs[idx]
    w_gate_sem = body_refs[idx + 1]
    wo_sem = body_refs[idx + 2]
    w_pre_sem = body_refs[idx + 3]
    w_router_sem = body_refs[idx + 4]
    s_in_sem = body_refs[idx + 5]
    s_out_sem = body_refs[idx + 6]
    bkv_sems = body_refs[idx + 7]
    c1a_send = body_refs[idx + 8]
    c1a_recv = body_refs[idx + 9]
    rs1_send = body_refs[idx + 10]
    rs1_recv = body_refs[idx + 11]
    rs2_send = body_refs[idx + 12]
    rs2_recv = body_refs[idx + 13]
    c1c_send = body_refs[idx + 14]
    c1c_recv = body_refs[idx + 15]
    pair_send = body_refs[idx + 16]
    pair_recv = body_refs[idx + 17]
    rpair_send = body_refs[idx + 18]
    rpair_recv = body_refs[idx + 19]
    pre_send = body_refs[idx + 20]
    pre_recv = body_refs[idx + 21]
    ag_exp_send = body_refs[idx + 22]
    ag_exp_recv = body_refs[idx + 23]
    ag_log_send = body_refs[idx + 24]
    ag_log_recv = body_refs[idx + 25]

    rows = attn_in_ref.shape[0]
    qkv_cols = w_qkv_hbm.shape[1]
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    phys_page_rows = (
        kv_cache_ro_hbm.shape[2] if kv_cache_ro_hbm.ndim == 6 else kv_cache_ro_hbm.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows

    me = lax.axis_index("model").astype(jnp.int32)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)

    w_qkv_buf = slab_a.at[:, pl.ds(0, qkv_cols)]

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()
    pltpu.make_async_copy(stream_shard_hbm, s_buf, s_in_sem.at[0]).start()
    _prologue_prefetch_kv_pages(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        kv_cache_ro_hbm,
        bkv_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
        sliding_window=sliding_window,
        page_size=page_size,
        num_slots=bkv_vmem_ref.shape[0],
    )

    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _llo_fence(
        distribution_ref,
        lambda: _dot_fp8_scaled_to_ref(
            attn_in_ref,
            w_qkv_buf,
            s_qkv_ref,
            0,
            qkv_vmem,
            0,
            rows=rows,
            n_cols=qkv_cols,
        ),
    )

    eff_kv_merged_ref = (
        out_kvw_rows_ref
        if (not kv_pack_dense2 and rows == kv_merged_vmem_ref.shape[0])
        else kv_merged_vmem_ref
    )
    _run_in_vmem_qkv_epilogue_any(
        qkv_vmem,
        cos_sin_ref,
        q_vmem_ref,
        eff_kv_merged_ref.at[pl.ds(0, rows)],
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        distribution_ref=distribution_ref,
    )

    def _start_wo_l0():
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
        ).start()
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
        ).start()

    _run_in_vmem_rpad_and_gate(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        q_vmem_ref,
        eff_kv_merged_ref,
        kv_cache_ro_hbm,
        out_kvw_rows_ref,
        bkv_vmem_ref,
        o_vmem_ref,
        attn_in_ref,
        w_gate_buf,
        w_gate_sem,
        s_gate_ref,
        gate_vmem_ref,
        gated_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        out_norm_eps=qkv_norm_eps,
        kv_pack_dense2=kv_pack_dense2,
        layer_num_in_bundle=layer_num_in_bundle,
        layer_offset_in_bundle=layer_offset_in_bundle,
        early_prefetched=True,
        interpret_mode=interpret_mode,
        on_rpad_kv_done_fn=_start_wo_l0,
        init_sv_vmem_ref=init_sv_vmem_ref,
    )

    pltpu.make_async_copy(w_pre_pair_hbm, slab_a, w_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    _dma_wait(s_buf, s_in_sem.at[0])

    _run_tok_sharded_b1_from_vmem_core(
        s_buf,
        gated_vmem_ref,
        wo_buf,
        wo_sem,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        slab_a,
        w_pre_sem,
        w_pre_pair_scale_ref,
        w_router_buf,
        w_router_sem,
        pre_exp_gain_ref,
        out_stream_shard_hbm,
        out_router_logits_ref,
        out_expert_in_ref,
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
        me=me,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        need_wait_barrier=True,
        distribution_ref=distribution_ref,
    )


def _kvw_1layer_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    kvw_rows,
    kv_cache_ro,
    *,
    kv_pack_dense2: bool,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    interpret: bool = False,
    vmem_kvw: bool = True,
):
    """Zero-copy 1-layer KV scatter (`collective_id=None` so XLA aliases `kv_cache_ro` in-place!)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    is_interpret = bool(interpret)
    rows = kvw_rows.shape[0]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    phys_page_rows = (
        kv_cache_ro.shape[2] if kv_cache_ro.ndim == 6 else kv_cache_ro.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows

    def _kvw_1layer_kernel(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        kvw_rows_hbm,
        _kv_in_hbm,
        kv_out_hbm,
        w_cnt_smem_ref,
        sems,
    ):
        def _dst_slice(seq, kvl):
            seq_base = cdiv(cu_kv_lens_ref[seq], page_size)
            pos = kvl - jnp.int32(1)
            pg_off = seq_base + (pos // page_size)
            raw_pg = page_indices_ref[jnp.minimum(pg_off, num_page_indices - 1)]
            r_in_pg = (
                ((pos % page_size) // 2) if kv_pack_dense2 else (pos % page_size)
            )
            if kv_out_hbm.ndim == 6:
                return kv_out_hbm.at[
                    raw_pg, layer_offset_in_bundle, pl.ds(r_in_pg, 1)
                ]
            eff_pg = (
                raw_pg * layer_num_in_bundle + layer_offset_in_bundle
                if layer_num_in_bundle > 1
                else raw_pg
            )
            return kv_out_hbm.at[eff_pg, pl.ds(r_in_pg, 1)]

        n_active = jnp.clip(distribution_ref[0], 0, max_num_seqs)
        if not is_interpret and max_num_seqs == rows and rows <= 128:
            q_prev = cu_q_lens_ref[0]
            kv_prev = cu_kv_lens_ref[0]
            all_act = (n_active == max_num_seqs) & (q_prev == jnp.int32(0))
            for s_i in range(max_num_seqs):
                q_next = cu_q_lens_ref[s_i + 1]
                kv_next = cu_kv_lens_ref[s_i + 1]
                kvl_s = kv_lens_ref[s_i]
                act_s = (q_next - q_prev == 1) & (kv_next > kv_prev) & (kvl_s > 0)
                all_act = all_act & act_s
                q_prev = q_next
                kv_prev = kv_next

            @pl.when(all_act)
            def _fast_all():
                for seq in range(rows):
                    pltpu.make_async_copy(
                        kvw_rows_hbm.at[pl.ds(seq, 1)],
                        _dst_slice(seq, kv_lens_ref[seq]),
                        sems.at[0],
                    ).start()
                src_all = kvw_rows_hbm.at[pl.ds(0, rows)]
                pltpu.make_async_copy(src_all, src_all, sems.at[0]).wait()

            @pl.when(~all_act)
            def _partial():
                w_cnt_smem_ref[0] = jnp.int32(0)

                @pl.loop(0, n_active)
                def _step(seq):
                    q_s = cu_q_lens_ref[seq]
                    q_e = cu_q_lens_ref[seq + 1]
                    kvl = kv_lens_ref[seq]
                    act = (
                        (q_e - q_s == 1)
                        & (cu_kv_lens_ref[seq + 1] > cu_kv_lens_ref[seq])
                        & (kvl > 0)
                    )

                    @pl.when(act)
                    def _():
                        w_cnt_smem_ref[0] = w_cnt_smem_ref[0] + jnp.int32(1)
                        pltpu.make_async_copy(
                            kvw_rows_hbm.at[pl.ds(q_s, 1)],
                            _dst_slice(seq, kvl),
                            sems.at[0],
                        ).start()

                w_cnt = w_cnt_smem_ref[0]

                @pl.when(w_cnt > 0)
                def _():
                    src_w = kvw_rows_hbm.at[pl.ds(0, w_cnt)]
                    pltpu.make_async_copy(src_w, src_w, sems.at[0]).wait()

            return

        w_cnt_smem_ref[0] = jnp.int32(0)

        @pl.loop(0, n_active)
        def _step_any(seq):
            q_s = cu_q_lens_ref[seq]
            q_e = cu_q_lens_ref[seq + 1]
            kvl = kv_lens_ref[seq]
            act = (
                (q_e - q_s == 1)
                & (cu_kv_lens_ref[seq + 1] > cu_kv_lens_ref[seq])
                & (kvl > 0)
            )

            @pl.when(act)
            def _():
                w_cnt_smem_ref[0] = w_cnt_smem_ref[0] + jnp.int32(1)
                pltpu.make_async_copy(
                    kvw_rows_hbm.at[pl.ds(q_s, 1)],
                    _dst_slice(seq, kvl),
                    sems.at[0],
                ).start()

        w_cnt = w_cnt_smem_ref[0]

        @pl.when(w_cnt > 0)
        def _():
            src_w = kvw_rows_hbm.at[pl.ds(0, w_cnt)]
            pltpu.make_async_copy(src_w, src_w, sems.at[0]).wait()

    h_sp = pl.BlockSpec(memory_space=pltpu.HBM)
    v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        in_specs=[v_sp if vmem_kvw else h_sp, h_sp],
        out_specs=h_sp,
        grid=(1,),
        scratch_shapes=[pltpu.SMEM((2,), jnp.int32), pltpu.SemaphoreType.DMA((1,))],
    )
    out_shape = (
        jax.ShapeDtypeStruct(kv_cache_ro.shape, kv_cache_ro.dtype)
        if is_interpret
        else pltpu.HBM(kv_cache_ro.shape, kv_cache_ro.dtype)
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=4 * 1024**2,
        disable_bounds_checks=True,
    )
    kvw_arg = (
        kvw_rows
        if (is_interpret or vmem_kvw)
        else pltpu.with_memory_space_constraint(kvw_rows, pltpu.HBM)
    )
    kv_arg = (
        kv_cache_ro
        if is_interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    v_tag = "_vkvw" if vmem_kvw else ""
    return _pallas_call_maybe_interpret(
        _kvw_1layer_kernel,
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases={6: 0},
        compiler_params=cp,
        name=f"muse_m2m_kvw1_b{rows}_{d2_tag}{v_tag}",
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        kvw_arg,
        kv_arg,
    )


def kvw_group_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    kvw_rows_list: tuple | list,
    kv_buffers: tuple | list,
    *,
    kv_pack_dense2: bool,
    interpret: bool = False,
):
    """Zero-copy grouped `K`-layer VMEM->HBM KV scatter (`1` launch for `K` layers, `0` HLO copies!)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    n_grp = len(kvw_rows_list)
    if n_grp == 0:
        return ()
    is_interpret = bool(interpret)
    rows = kvw_rows_list[0].shape[0]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    phys_page_rows = kv_buffers[0].shape[1]
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    n_sems = min(n_grp, 16)

    def _kvw_grp_kernel(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        *body_refs,
    ):
        row_vmem_refs = body_refs[:n_grp]
        kv_out_refs = body_refs[2 * n_grp : 3 * n_grp]
        w_cnt_smem_ref = body_refs[3 * n_grp]
        sems = body_refs[3 * n_grp + 1]

        n_active = jnp.clip(distribution_ref[0], 0, max_num_seqs)
        if not is_interpret and max_num_seqs == rows and rows <= 128:
            q_prev = cu_q_lens_ref[0]
            kv_prev = cu_kv_lens_ref[0]
            all_act = (n_active == max_num_seqs) & (q_prev == jnp.int32(0))
            for s_i in range(max_num_seqs):
                q_next = cu_q_lens_ref[s_i + 1]
                kv_next = cu_kv_lens_ref[s_i + 1]
                kvl_s = kv_lens_ref[s_i]
                act_s = (q_next - q_prev == 1) & (kv_next > kv_prev) & (kvl_s > 0)
                all_act = all_act & act_s
                q_prev = q_next
                kv_prev = kv_next

            @pl.when(all_act)
            def _fast_grp():
                for seq in range(rows):
                    seq_base = cdiv(cu_kv_lens_ref[seq], page_size)
                    pos = kv_lens_ref[seq] - jnp.int32(1)
                    pg_off = seq_base + (pos // page_size)
                    raw_pg = page_indices_ref[jnp.minimum(pg_off, num_page_indices - 1)]
                    r_in_pg = (
                        ((pos % page_size) // 2) if kv_pack_dense2 else (pos % page_size)
                    )
                    for l_i in range(n_grp):
                        pltpu.make_async_copy(
                            row_vmem_refs[l_i].at[pl.ds(seq, 1)],
                            kv_out_refs[l_i].at[raw_pg, pl.ds(r_in_pg, 1)],
                            sems.at[l_i % n_sems],
                        ).start()
                for l_i in range(n_grp):
                    src_l = row_vmem_refs[l_i].at[pl.ds(0, rows)]
                    pltpu.make_async_copy(src_l, src_l, sems.at[l_i % n_sems]).wait()

            @pl.when(~all_act)
            def _partial_grp():
                w_cnt_smem_ref[0] = jnp.int32(0)

                @pl.loop(0, n_active)
                def _step(seq):
                    q_s = cu_q_lens_ref[seq]
                    q_e = cu_q_lens_ref[seq + 1]
                    kvl = kv_lens_ref[seq]
                    act = (
                        (q_e - q_s == 1)
                        & (cu_kv_lens_ref[seq + 1] > cu_kv_lens_ref[seq])
                        & (kvl > 0)
                    )

                    @pl.when(act)
                    def _():
                        w_cnt_smem_ref[0] = w_cnt_smem_ref[0] + jnp.int32(1)
                        seq_base = cdiv(cu_kv_lens_ref[seq], page_size)
                        pos = kvl - jnp.int32(1)
                        pg_off = seq_base + (pos // page_size)
                        raw_pg = page_indices_ref[
                            jnp.minimum(pg_off, num_page_indices - 1)
                        ]
                        r_in_pg = (
                            ((pos % page_size) // 2)
                            if kv_pack_dense2
                            else (pos % page_size)
                        )
                        for l_i in range(n_grp):
                            pltpu.make_async_copy(
                                row_vmem_refs[l_i].at[pl.ds(q_s, 1)],
                                kv_out_refs[l_i].at[raw_pg, pl.ds(r_in_pg, 1)],
                                sems.at[l_i % n_sems],
                            ).start()

                w_cnt = w_cnt_smem_ref[0]

                @pl.when(w_cnt > 0)
                def _():
                    for l_i in range(n_grp):
                        src_l = row_vmem_refs[l_i].at[pl.ds(0, w_cnt)]
                        pltpu.make_async_copy(
                            src_l, src_l, sems.at[l_i % n_sems]
                        ).wait()

            return

        w_cnt_smem_ref[0] = jnp.int32(0)

        @pl.loop(0, n_active)
        def _step_any(seq):
            q_s = cu_q_lens_ref[seq]
            q_e = cu_q_lens_ref[seq + 1]
            kvl = kv_lens_ref[seq]
            act = (
                (q_e - q_s == 1)
                & (cu_kv_lens_ref[seq + 1] > cu_kv_lens_ref[seq])
                & (kvl > 0)
            )

            @pl.when(act)
            def _():
                w_cnt_smem_ref[0] = w_cnt_smem_ref[0] + jnp.int32(1)
                seq_base = cdiv(cu_kv_lens_ref[seq], page_size)
                pos = kvl - jnp.int32(1)
                pg_off = seq_base + (pos // page_size)
                raw_pg = page_indices_ref[jnp.minimum(pg_off, num_page_indices - 1)]
                r_in_pg = (
                    ((pos % page_size) // 2) if kv_pack_dense2 else (pos % page_size)
                )
                for l_i in range(n_grp):
                    pltpu.make_async_copy(
                        row_vmem_refs[l_i].at[pl.ds(q_s, 1)],
                        kv_out_refs[l_i].at[raw_pg, pl.ds(r_in_pg, 1)],
                        sems.at[l_i % n_sems],
                    ).start()

        w_cnt = w_cnt_smem_ref[0]

        @pl.when(w_cnt > 0)
        def _():
            for l_i in range(n_grp):
                src_l = row_vmem_refs[l_i].at[pl.ds(0, w_cnt)]
                pltpu.make_async_copy(src_l, src_l, sems.at[l_i % n_sems]).wait()

    v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
    h_sp = pl.BlockSpec(memory_space=pltpu.HBM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        in_specs=[v_sp] * n_grp + [h_sp] * n_grp,
        out_specs=[h_sp] * n_grp,
        grid=(1,),
        scratch_shapes=[
            pltpu.SMEM((2,), jnp.int32),
            pltpu.SemaphoreType.DMA((n_sems,)),
        ],
    )
    out_shape = [
        (
            jax.ShapeDtypeStruct(kb.shape, kb.dtype)
            if is_interpret
            else pltpu.HBM(kb.shape, kb.dtype)
        )
        for kb in kv_buffers
    ]
    aliases = {5 + n_grp + j: j for j in range(n_grp)}
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=8 * 1024**2,
        disable_bounds_checks=True,
    )
    kv_args = [
        kb if is_interpret else pltpu.with_memory_space_constraint(kb, pltpu.HBM)
        for kb in kv_buffers
    ]
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    res = _pallas_call_maybe_interpret(
        _kvw_grp_kernel,
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases=aliases,
        compiler_params=cp,
        name=f"muse_m2m_kvwg{n_grp}_b{rows}_{d2_tag}",
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        *kvw_rows_list,
        *kv_args,
    )
    return tuple(res) if isinstance(res, (tuple, list)) else (res,)


def m2m_tok_sh_tp8_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    stream_shard,
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
    cos_sin,
    kv_cache_ro,
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
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    sm_scale: float = 0.125,
    sliding_window: int | None = 2048,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_pack_dense2: bool = True,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _M2M_CID_BASE,
    interpret: bool = False,
):
    """Per-layer `M2M` (`l >= 1`, `TP = 8` token-sharded stream)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        raise ValueError(f"m2m_tok_sh_tp8_jax requires world_size=8, got {world_size}")
    r_tok, hidden = stream_shard.shape
    rows = world_size * r_tok
    if rows not in (64, 128) or r_tok % 8 != 0:
        raise ValueError(f"Unsupported m2m_tok_sh_tp8 rows={rows}, r_tok={r_tok}")

    r_pair = 2 * r_tok
    two_pair = 2 * r_pair
    half = hidden // 2
    q_cols = half // 2
    post_cols = w_post_w8k.shape[1]
    pair_cols = 2 * post_cols
    qkv_cols = next_w_qkv_w8k.shape[1]
    gate_cols = next_w_gate_w8k.shape[1]
    o_in = wo_w8k.shape[0]
    pre_pair_cols = w_pre_pair_w8k.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_km.shape[1]
    n_chips = world_size // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    has_rope = cos_sin is not None
    b_pad32 = align_to(rows, 32)

    phys_page_rows = (
        kv_cache_ro.shape[2] if kv_cache_ro.ndim == 6 else kv_cache_ro.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    max_ring_pages = _max_m2m_ring_pages(
        sliding_window=sliding_window,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
    )

    vmem, smem, hbm, sem = (
        pltpu.VMEM,
        pltpu.SMEM,
        pltpu.HBM,
        pltpu.SemaphoreType.DMA,
    )
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    in_specs = [
        h_sp,  # stream_shard
        v_sp,  # moe_out
        h_sp,  # w_post_w8k
        v_sp,  # s_post
        v_sp,  # post_ff_gain
        v_sp,  # pf_alpha
        v_sp,  # pf_beta
        v_sp,  # next_in_gain
        h_sp,  # next_w_qkv_w8k
        v_sp,  # next_s_qkv
        h_sp,  # next_w_gate_w8k
        v_sp,  # next_s_gate
    ]
    if has_rope:
        in_specs.append(v_sp)  # cos_sin
    in_specs.extend(
        [
            h_sp,  # kv_cache_ro
            h_sp,  # wo_w8k
            v_sp,  # wo_scale
            v_sp,  # pa_alpha
            v_sp,  # pa_beta
            v_sp,  # pre_ff_gain
            h_sp,  # w_pre_pair_w8k
            v_sp,  # w_pre_pair_scale
            h_sp,  # w_router_km
            v_sp,  # pre_exp_gain
        ]
    )

    is_interpret = bool(interpret)
    out_shape = (
        jax.ShapeDtypeStruct((r_tok, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
        jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 1, 4, 128), kv_cache_ro.dtype),
    )
    out_specs = (h_sp, v_sp, v_sp, v_sp)

    scratch_shapes = [
        # 5 weight + KV ring slabs:
        vmem((hidden, pre_pair_cols), next_w_qkv_w8k.dtype),  # slab_a (16 MiB)
        vmem((hidden, gate_cols), next_w_gate_w8k.dtype),  # slab_b (8 MiB)
        vmem((2, o_in, half), wo_w8k.dtype),  # wo_buf (8 MiB)
        vmem((hidden, n_exp), w_router_km.dtype),  # w_router_buf (4 MiB)
        vmem(
            (_NUM_KV_SLOTS, max_ring_pages, phys_page_rows, 1, 4, 128),
            kv_cache_ro.dtype,
        ),
        # 35 activation + SMEM scratches:
        vmem((r_tok, hidden), jnp.float32),  # s_buf
        vmem((r_tok, hidden), jnp.bfloat16),  # next_in_my_tok
        vmem((r_tok, half), jnp.bfloat16),  # next_in_mate_half
        vmem((rows, hidden), jnp.bfloat16),  # next_in_vmem
        vmem((rows, qkv_cols), jnp.bfloat16),  # qkv_vmem
        vmem((rows, 16, 128), jnp.bfloat16),  # q_vmem_ref
        vmem((b_pad32, 1, 4, 128), kv_cache_ro.dtype),  # kv_merged_vmem_ref
        vmem((rows, 16, 128), jnp.bfloat16),  # o_vmem_ref
        vmem((rows, gate_cols), jnp.bfloat16),  # gate_vmem_ref
        vmem((rows, gate_cols), jnp.bfloat16),  # gated_vmem_ref
        vmem((r_tok, w_post_w8k.shape[0]), jnp.bfloat16),  # mate_tok_buf
        vmem((rows, w_post_w8k.shape[0]), jnp.bfloat16),  # moe_full_buf
        vmem((rows, post_cols), jnp.bfloat16),  # post_shard
        vmem((rows, post_cols), jnp.bfloat16),  # mate_post_shard
        vmem((rows, pair_cols), jnp.bfloat16),  # chip_cols_buf
        vmem((r_pair, hidden), jnp.bfloat16),  # post_pair_full (reused as ffn_pair_buf)
        vmem(
            (8, 128) if w_post_w8k.shape[0] == half else (rows, half), jnp.bfloat16
        ),  # send_half
        vmem((rows, half), jnp.bfloat16),  # mate_recv
        vmem((rows, half), jnp.bfloat16),  # own_half
        vmem((two_pair, q_cols), jnp.bfloat16),  # rs_a_recv
        vmem((two_pair, q_cols), jnp.bfloat16),  # rs_b_recv
        vmem((r_tok, half), jnp.float32),  # d2d_my_tok
        vmem((r_tok, half), jnp.bfloat16),  # d2d_send_tok
        vmem((r_tok, half), jnp.bfloat16),  # d2d_recv_tok
        vmem((r_tok, 128), jnp.float32),  # sq_tok_buf
        vmem((r_tok, hidden), jnp.bfloat16),  # ffn_my_tok
        vmem((r_tok, hidden), jnp.bfloat16),  # ffn_mate_tok
        vmem((8, 128), jnp.bfloat16),  # ffn_pair_buf (aliased onto post_pair_full)
        vmem((8, 128) if is_r16 else (r_tok, hidden), jnp.float32),
        vmem((8, 128) if is_r16 else (r_tok, hidden), jnp.float32),
        vmem((8, 128) if is_r16 else (r_pair, hidden), jnp.float32),
        vmem((r_pair, pre_pair_cols), jnp.bfloat16),  # pre_my_half
        vmem((r_pair, pre_pair_cols), jnp.bfloat16),  # pre_mate_half
        vmem((r_pair, 128), jnp.float32),  # sq_pair_buf
        vmem((2, b_pad32 if sliding_window is None else 1, 16, 128), jnp.float32),  # init_sv_vmem_ref
        smem((2 + 2 * _NUM_KV_SLOTS,), jnp.int32),  # rpa_smem_ref
        # 27 semaphore groups (42 total DMA semaphores < 64 limit!):
        sem((1,)),  # w_post_sem (reused as pre_send in Phase 3)
        sem((1,)),  # w_qkv_sem (reused as pre_recv in Phase 3)
        sem((1,)),  # w_gate_sem
        sem((2,)),  # wo_sem
        sem((1,)),  # w_pre_sem
        sem((1,)),  # w_router_sem
        sem((1,)),  # s_in_sem
        sem((1,)),  # s_out_sem
        sem((_NUM_KV_SLOTS,)),  # bkv_sems
        sem((1,)),  # d2d_a_send
        sem((1,)),  # d2d_a_recv
        sem((1,)),  # d2d_b_send
        sem((1,)),  # d2d_b_recv
        sem((1,)),  # d2d_c_send
        sem((1,)),  # d2d_c_recv
        sem((1,)),  # d2d_d_send
        sem((1,)),  # d2d_d_recv
        sem((n_chips - 1,)),  # ici_a_send
        sem((n_chips - 1,)),  # ici_a_recv
        sem((n_chips - 1,)),  # ici_b_send
        sem((n_chips - 1,)),  # ici_b_recv
        sem((n_chips - 1,)),  # ici_c_send
        sem((n_chips - 1,)),  # ici_c_recv
        sem((3,)),  # rs1_send
        sem((3,)),  # rs1_recv
        sem((2,)),  # rs2_send
        sem((2,)),  # rs2_recv
    ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(1,),
        scratch_shapes=scratch_shapes,
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=60 * 1024**2,
        collective_id=collective_id,
        disable_bounds_checks=True,
    )

    rope_args = (cos_sin.astype(jnp.float32),) if has_rope else ()
    kv_ro_arg = (
        kv_cache_ro
        if is_interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    sw_tag = f"sw{sliding_window}" if sliding_window is not None else f"full{max_ring_pages}p"
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    name = f"muse_m2m_tp8_b{rows}_{sw_tag}_{d2_tag}_{'rope' if has_rope else 'nope'}"

    st_out, logits_out, exp_in_out, kvw_rows_out = _pallas_call_maybe_interpret(
        functools.partial(
            _m2m_tok_sh_tp8_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            sm_scale=float(sm_scale),
            sliding_window=sliding_window,
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            kv_pack_dense2=bool(kv_pack_dense2),
            layer_num_in_bundle=int(layer_num_in_bundle),
            layer_offset_in_bundle=int(layer_offset_in_bundle),
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            has_rope=has_rope,
            interpret_mode=is_interpret,
        ),
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        compiler_params=cp,
        name=name,
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        stream_shard.astype(jnp.float32),
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
        *rope_args,
        kv_ro_arg,
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
    kv_cache_out = _kvw_1layer_jax(
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        kvw_rows_out,
        kv_cache_ro,
        kv_pack_dense2=bool(kv_pack_dense2),
        layer_num_in_bundle=int(layer_num_in_bundle),
        layer_offset_in_bundle=int(layer_offset_in_bundle),
        interpret=interpret,
    )
    return st_out, logits_out, exp_in_out, kv_cache_out


def l0_a2m_tok_sh_tp8_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    stream_shard,
    attn_in,
    w_qkv_w8k,
    s_qkv,
    w_gate_w8k,
    s_gate,
    cos_sin,
    kv_cache_ro,
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
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    sm_scale: float = 0.125,
    sliding_window: int | None = 2048,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_pack_dense2: bool = True,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _M2M_CID_BASE + 1,
    interpret: bool = False,
):
    """Layer-0 fused `A2M` (`TP = 8` token-sharded stream)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        raise ValueError(f"l0_a2m_tok_sh_tp8_jax requires world_size=8, got {world_size}")
    r_tok, hidden = stream_shard.shape
    rows = attn_in.shape[0]
    if rows not in (64, 128) or rows != world_size * r_tok or r_tok % 8 != 0:
        raise ValueError(f"Unsupported l0_a2m_tok_sh_tp8 rows={rows}, r_tok={r_tok}")

    r_pair = 2 * r_tok
    two_pair = 2 * r_pair
    half = hidden // 2
    q_cols = half // 2
    qkv_cols = w_qkv_w8k.shape[1]
    gate_cols = w_gate_w8k.shape[1]
    o_in = wo_w8k.shape[0]
    pre_pair_cols = w_pre_pair_w8k.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_km.shape[1]
    n_chips = world_size // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    has_rope = cos_sin is not None
    b_pad32 = align_to(rows, 32)

    phys_page_rows = (
        kv_cache_ro.shape[2] if kv_cache_ro.ndim == 6 else kv_cache_ro.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    max_ring_pages = _max_m2m_ring_pages(
        sliding_window=sliding_window,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
    )

    vmem, smem, hbm, sem = (
        pltpu.VMEM,
        pltpu.SMEM,
        pltpu.HBM,
        pltpu.SemaphoreType.DMA,
    )
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    in_specs = [h_sp, v_sp, h_sp, v_sp, h_sp, v_sp]
    if has_rope:
        in_specs.append(v_sp)
    in_specs.extend([h_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp])

    is_interpret = bool(interpret)
    out_shape = (
        jax.ShapeDtypeStruct((r_tok, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
        jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 1, 4, 128), kv_cache_ro.dtype),
    )
    out_specs = (h_sp, v_sp, v_sp, v_sp)

    scratch_shapes = [
        vmem((hidden, pre_pair_cols), w_qkv_w8k.dtype),
        vmem((hidden, gate_cols), w_gate_w8k.dtype),
        vmem((2, o_in, half), wo_w8k.dtype),
        vmem((hidden, n_exp), w_router_km.dtype),
        vmem(
            (_NUM_KV_SLOTS, max_ring_pages, phys_page_rows, 1, 4, 128),
            kv_cache_ro.dtype,
        ),
        vmem((r_tok, hidden), jnp.float32),
        vmem((rows, qkv_cols), jnp.bfloat16),
        vmem((rows, 16, 128), jnp.bfloat16),
        vmem((b_pad32, 1, 4, 128), kv_cache_ro.dtype),
        vmem((rows, 16, 128), jnp.bfloat16),
        vmem((rows, gate_cols), jnp.bfloat16),
        vmem((rows, gate_cols), jnp.bfloat16),
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
        vmem((2, b_pad32 if sliding_window is None else 1, 16, 128), jnp.float32),
        smem((2 + 2 * _NUM_KV_SLOTS,), jnp.int32),
        sem((1,)),
        sem((1,)),
        sem((2,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((_NUM_KV_SLOTS,)),
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
    ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(1,),
        scratch_shapes=scratch_shapes,
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=60 * 1024**2,
        collective_id=collective_id,
        disable_bounds_checks=True,
    )
    rope_args = (cos_sin.astype(jnp.float32),) if has_rope else ()
    kv_ro_arg = (
        kv_cache_ro
        if is_interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    sw_tag = f"sw{sliding_window}" if sliding_window is not None else f"full{max_ring_pages}p"
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    name = f"muse_l0_a2m_tp8_b{rows}_{sw_tag}_{d2_tag}_{'rope' if has_rope else 'nope'}"

    st_out, logits_out, exp_in_out, kvw_rows_out = _pallas_call_maybe_interpret(
        functools.partial(
            _l0_a2m_tok_sh_tp8_kernel,
            world_size=world_size,
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            sm_scale=float(sm_scale),
            sliding_window=sliding_window,
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            kv_pack_dense2=bool(kv_pack_dense2),
            layer_num_in_bundle=int(layer_num_in_bundle),
            layer_offset_in_bundle=int(layer_offset_in_bundle),
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            has_rope=has_rope,
            interpret_mode=is_interpret,
        ),
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        compiler_params=cp,
        name=name,
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        stream_shard.astype(jnp.float32),
        attn_in.astype(jnp.bfloat16),
        w_qkv_w8k,
        _as_dot_scale_f32(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_dot_scale_f32(s_gate, gate_cols),
        *rope_args,
        kv_ro_arg,
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
    kv_cache_out = _kvw_1layer_jax(
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        kvw_rows_out,
        kv_cache_ro,
        kv_pack_dense2=bool(kv_pack_dense2),
        layer_num_in_bundle=int(layer_num_in_bundle),
        layer_offset_in_bundle=int(layer_offset_in_bundle),
        interpret=interpret,
    )
    return st_out, logits_out, exp_in_out, kv_cache_out


def kv_write_batch_jax(
    kvw_rows_list: tuple | list,
    kvw_dst_list: tuple | list,
    kv_buffers: tuple | list,
    *,
    layer_to_buf_idx: tuple[int, ...],
    interpret: bool = False,
):
    """Single end-of-step in-place scatter of all `K` `M2M` layers' KV rows (`0` HLO copies)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    n_layers = len(kvw_rows_list)
    n_bufs = len(kv_buffers)
    if n_layers == 0:
        return tuple(kv_buffers)
    if len(kvw_dst_list) != n_layers or len(layer_to_buf_idx) != n_layers:
        raise ValueError("Mismatched kv_write_batch_jax list lengths")
    ref_row_shape = kvw_rows_list[0].shape
    ref_row_dtype = kvw_rows_list[0].dtype
    row_tail = ref_row_shape[1:]
    if any(r.shape != ref_row_shape or r.dtype != ref_row_dtype for r in kvw_rows_list):
        raise ValueError("All kvw_rows_list entries must share identical shape and dtype")
    if any(kb.shape[-3:] != row_tail or kb.dtype != ref_row_dtype for kb in kv_buffers):
        raise ValueError("All kv_buffers must match kvw_rows_list row tail shape and dtype")

    rows = ref_row_shape[0]
    n_sems = min(n_layers, 32)

    def _kvw_body(*refs):
        dst_smem_refs = refs[:n_layers]
        row_hbm_refs = refs[n_layers : 2 * n_layers]
        kv_out_refs = refs[2 * n_layers + n_bufs : 2 * n_layers + 2 * n_bufs]
        sems = refs[2 * n_layers + 2 * n_bufs]

        for l_i in range(n_layers):
            b_idx = layer_to_buf_idx[l_i]
            kv_ref = kv_out_refs[b_idx]
            dst_s = dst_smem_refs[l_i]
            w_cnt = jnp.clip(dst_s[7, 127], 0, rows)
            sem_slot = l_i % n_sems

            def _issue_one(i, _, _kv_ref=kv_ref, _l_i=l_i, _dst_s=dst_s, _sem=sem_slot):
                d = jnp.maximum(_dst_s[i // 128, i % 128], 0)
                if _kv_ref.ndim == 6:
                    layer_num = _kv_ref.shape[1]
                    phys_rows = _kv_ref.shape[2]
                    eff_pg = d // phys_rows
                    r_in_pg = d % phys_rows
                    dst_sl = _kv_ref.at[
                        eff_pg // layer_num, eff_pg % layer_num, pl.ds(r_in_pg, 1)
                    ]
                else:
                    phys_rows = _kv_ref.shape[1]
                    eff_pg = d // phys_rows
                    r_in_pg = d % phys_rows
                    dst_sl = _kv_ref.at[eff_pg, pl.ds(r_in_pg, 1)]
                pltpu.make_async_copy(
                    row_hbm_refs[_l_i].at[pl.ds(i, 1)],
                    dst_sl,
                    sems.at[_sem],
                ).start()

            lax.fori_loop(0, w_cnt, _issue_one, None)

        for l_i in range(n_layers):
            dst_s = dst_smem_refs[l_i]
            w_cnt = jnp.clip(dst_s[7, 127], 0, rows)
            sem_slot = l_i % n_sems

            @pl.when(w_cnt > 0)
            def _wait_layer(_l_i=l_i, _w_cnt=w_cnt, _sem=sem_slot):
                src = row_hbm_refs[_l_i].at[pl.ds(0, _w_cnt)]
                pltpu.make_async_copy(src, src, sems.at[_sem]).wait()

    h_sp = pl.BlockSpec(memory_space=pltpu.HBM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=n_layers,
        in_specs=[h_sp] * (n_layers + n_bufs),
        out_specs=[h_sp] * n_bufs,
        grid=(1,),
        scratch_shapes=[pltpu.SemaphoreType.DMA((n_sems,))],
    )
    out_shape = [
        (
            jax.ShapeDtypeStruct(kb.shape, kb.dtype)
            if interpret
            else pltpu.HBM(kb.shape, kb.dtype)
        )
        for kb in kv_buffers
    ]
    aliases = {2 * n_layers + j: j for j in range(n_bufs)}
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=8 * 1024**2,
        disable_bounds_checks=True,
    )
    kv_args = [
        kb if interpret else pltpu.with_memory_space_constraint(kb, pltpu.HBM)
        for kb in kv_buffers
    ]
    res = _pallas_call_maybe_interpret(
        _kvw_body,
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases=aliases,
        compiler_params=cp,
        name=f"muse_m2m_kvw_l{n_layers}_b{rows}",
    )(*kvw_dst_list, *kvw_rows_list, *kv_args)
    return tuple(res) if isinstance(res, (tuple, list)) else (res,)
