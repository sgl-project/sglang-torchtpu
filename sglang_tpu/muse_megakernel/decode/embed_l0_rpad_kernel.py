"""Fused L0 decode megakernel: Vocab Embed + Norms + L0 QKV/Gate/RoPE + L0 RPAd.

Replaces the 3-kernel L0 prologue (`embed_norm` -> `dual_qkv_epi` -> `RPAd_fast`)
and L0's standalone KV-cache scatter with a single Pallas megakernel that:
1. Uses an 8-row `in_r`-predicated HBM vocab gather (`1 MiB` slot buffer, 2x smaller
   HBM fetch per owned token, skips VPU extraction on non-owned tokens).
2. Overlaps `L0 W_qkv` and `L0 W_gate` HBM->VMEM DMAs behind vocab gather + ICI RS + norms.
3. Overlaps `L0 RPAd bkv` HBM->VMEM DMAs behind the pre-attention AG (chip-sharded) or
   ICI sum (replicated), reusing 5-sem 2D-XY AG on `TP=8 B>=128`.
4. Computes `L0 QKV + QK-Norm/RoPE + Gate + L0 RPAd` in VMEM and returns
   `(stream_chip_or_full, gated_attn_0, kv_merged_0)` so `L0` attention is skipped
   and `L0`'s KV write folds into the batched multi-layer KV-write pass.
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp

from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    _COL_STRIP,
    _W8_CID_BASE,
    SUPPORTED_WORLD_SIZES,
    _ag_chip_sharded_half_col_xor1,
    _as_2d_f32,
    _as_dot_scale_f32,
    _dma_wait,
    _dot_fp8_scaled_to_ref,
    _fold_cols_to_128,
    _fori_strips,
    _pick_strip,
    _run_in_vmem_qkv_epilogue_dispatch,
    _signal_barrier,
    _strip_unroll,
    _wait_barrier,
)
from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
    resolve_step_c_rpa_plan,
)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _lookup_local_vocab_rows_fast_16row(
    input_ids_smem,
    embed_hbm,
    partial_emb_buf,
    slot_2d_buf,
    row_sems,
    *,
    me,
    actual_rows: int,
    interpret_mode: bool = False,
) -> None:
    """Fast 16-row HBM-DMA, owner-predicated vocab gather supporting ``v_shard % 2 == 0``.

    Uses 16-row HBM->VMEM DMAs (`pl.ds(br, 16)`, avoiding TPU7x 8-row bf16 HBM DMA stalls,
    with `tail_len in {0, 2, 4, 6}`) into an 8-slot (`128 x H`) VMEM buffer, and skips
    extraction completely for non-owned tokens via `lax.cond(in_r, ...)`.
    """
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
    col_w = 512 if (hidden % 512 == 0) else _COL_STRIP
    row_iota_16 = lax.broadcasted_iota(jnp.int32, (16, col_w), 0)

    def _grp(g_idx, _):
        g0 = pl.multiple_of(g_idx * 16, 16)
        for c0 in range(0, hidden, col_w):
            partial_emb_buf[pl.ds(g0, 16), pl.ds(c0, col_w)] = jnp.zeros(
                (16, col_w), dtype=jnp.bfloat16
            )

        for half_idx in range(2):
            h_off = half_idx * 8
            for k in range(8):
                row_idx = g0 + h_off + k
                valid_k = row_idx < actual_rows
                safe_idx = jnp.minimum(row_idx, actual_rows - 1)
                tid = input_ids_smem[safe_idx]
                in_r = valid_k & (tid >= v_start) & (tid < v_end)
                loc = jnp.where(in_r, tid - v_start, jnp.int32(0))
                if tail_len == 0:
                    b_r0 = pl.multiple_of(
                        jnp.minimum(
                            (loc >> jnp.int32(4)) << jnp.int32(4), jnp.int32(max_r0)
                        ),
                        8,
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

                    def _issue(lid=loc, kk=k):
                        in_tail = lid >= jnp.int32(tail_r0)

                        def _t():
                            if interpret_mode:
                                for c0 in range(0, hidden, col_w):
                                    slot_2d_buf[
                                        pl.ds(kk * 16, 16), pl.ds(c0, col_w)
                                    ] = jnp.zeros((16, col_w), dtype=jnp.bfloat16)
                            pltpu.make_async_copy(
                                embed_hbm.at[pl.ds(tail_r0, tail_len), :],
                                slot_2d_buf.at[pl.ds(kk * 16, tail_len), :],
                                row_sems.at[kk],
                            ).start()

                        def _b():
                            br = pl.multiple_of(
                                jnp.minimum(
                                    (lid >> jnp.int32(4)) << jnp.int32(4),
                                    jnp.int32(max_r0),
                                ),
                                8,
                            )
                            pltpu.make_async_copy(
                                embed_hbm.at[pl.ds(br, 16), :],
                                slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                                row_sems.at[kk],
                            ).start()

                        lax.cond(in_tail, _t, _b)

                    lax.cond(in_r, _issue, lambda: None)

            for k in range(8):
                row_idx = g0 + h_off + k
                valid_k = row_idx < actual_rows
                safe_idx = jnp.minimum(row_idx, actual_rows - 1)
                tid = input_ids_smem[safe_idx]
                in_r = valid_k & (tid >= v_start) & (tid < v_end)
                loc = jnp.where(in_r, tid - v_start, jnp.int32(0))

                def _extract_k(lid=loc, kk=k, dst_row=h_off + k):
                    if tail_len == 0:
                        pltpu.make_async_copy(
                            slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                            slot_2d_buf.at[pl.ds(kk * 16, 16), :],
                            row_sems.at[kk],
                        ).wait()
                        eff_r0 = jnp.minimum(
                            (lid >> jnp.int32(4)) << jnp.int32(4), jnp.int32(max_r0)
                        )
                        sub_r = lid - eff_r0
                    else:
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
                        eff_r0 = jnp.where(
                            in_tail,
                            jnp.int32(tail_r0),
                            jnp.minimum(
                                (lid >> jnp.int32(4)) << jnp.int32(4), jnp.int32(max_r0)
                            ),
                        )
                        sub_r = lid - eff_r0
                    dst_mask = row_iota_16 == jnp.int32(dst_row)
                    src_mask = row_iota_16 == sub_r
                    for c0 in range(0, hidden, col_w):
                        c_sl = pl.ds(c0, col_w)
                        blk = slot_2d_buf[pl.ds(kk * 16, 16), c_sl].astype(jnp.float32)
                        row_b = jnp.broadcast_to(
                            jnp.sum(jnp.where(src_mask, blk, 0.0), axis=0, keepdims=True),
                            (16, col_w),
                        ).astype(jnp.bfloat16)
                        cur = partial_emb_buf[pl.ds(g0, 16), c_sl]
                        partial_emb_buf[pl.ds(g0, 16), c_sl] = jnp.where(
                            dst_mask, row_b, cur
                        )

                lax.cond(in_r, _extract_k, lambda: None)
        return jnp.int32(0)

    lax.fori_loop(0, pad_rows // 16, _grp, jnp.int32(0))


def _run_l0_qkv_epi_gate_rpad_body(
    *,
    attn_in_vmem,
    w_qkv_buf,
    s_qkv_ref,
    w_gate_buf,
    s_gate_ref,
    cos_sin_ref,
    w_qkv_sem,
    w_gate_sem,
    qkv_vmem,
    rpad_q_vmem,
    rpad_kv_vmem,
    rpad_gate_vmem,
    bkv_vmem_ref,
    state_vmem_ref,
    o_vmem_ref,
    kv_write_vmem_ref,
    init_sv_vmem_ref,
    written_cnt_smem_ref,
    rpad_bkv_sems,
    rpad_cache_write_sem,
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    kv_cache_hbm_ref,
    out_gated_ref,
    out_kv_or_cache_ref,
    prestarted_bkv_state,
    world_size: int,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    rpa_sm_scale: float,
    rpa_sliding_window: int | None,
    rpa_k_scale: float,
    rpa_v_scale: float,
    rpa_out_norm_eps: float,
    rpa_cfg: Any,
    rpa_one_block_per_seq: bool,
    rpa_use_grp_vec: bool,
    rpa_kv_pack_dense2: bool,
    rpa_kv_pack_bd4: bool,
    rpa_skip_kv_write: bool,
    interpret_mode: bool,
    waited_w_qkv: bool,
) -> None:
    """Shared Scope-2 body: QKV projection -> QKV/RoPE epilogue -> Gate -> RPAd."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
        _run_rpad_vmem_core,
        prepare_and_start_rpad_dmas,
    )

    rows = attn_in_vmem.shape[0]
    qkv_cols = w_qkv_buf.shape[1]
    gate_cols = w_gate_buf.shape[1]

    if not waited_w_qkv:
        _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        attn_in_vmem, w_qkv_buf, s_qkv_ref, 0, qkv_vmem, 0,
        rows=rows, n_cols=qkv_cols,
    )
    b2_out_kv = out_kv_or_cache_ref if rpa_skip_kv_write else rpad_kv_vmem
    _run_in_vmem_qkv_epilogue_dispatch(
        qkv_vmem,
        cos_sin_ref,
        rpad_q_vmem,
        b2_out_kv,
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
    )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    if prestarted_bkv_state is None:
        prestarted_bkv_state = prepare_and_start_rpad_dmas(
            kv_lens_ref=kv_lens_ref,
            page_indices_ref=page_indices_ref,
            cu_q_lens_ref=cu_q_lens_ref,
            cu_kv_lens_ref=cu_kv_lens_ref,
            distribution_ref=distribution_ref,
            kv_cache_hbm_ref=kv_cache_hbm_ref,
            bkv_vmem_ref=bkv_vmem_ref,
            bkv_sems=rpad_bkv_sems,
            sliding_window=rpa_sliding_window,
            cfg=rpa_cfg,
            one_block_per_seq=rpa_one_block_per_seq,
            decode_query_size=1,
            kv_pack_dense2=rpa_kv_pack_dense2,
            interpret_mode=interpret_mode,
            b_tokens=rows,
            init_sv_vmem_ref=init_sv_vmem_ref,
            use_grp_vec=rpa_use_grp_vec,
            kv_pack_bd4=rpa_kv_pack_bd4,
        )
    _dot_fp8_scaled_to_ref(
        attn_in_vmem, w_gate_buf, s_gate_ref, 0, rpad_gate_vmem, 0,
        rows=rows, n_cols=gate_cols,
    )

    if rpa_kv_pack_bd4:
        step_q = math.gcd(rows, 8)
        for q0 in range(0, rows, step_q):
            q_sl = pl.ds(q0, step_q)
            q_lo_2d = rpad_q_vmem[q_sl, pl.ds(0, 8), :].reshape(step_q * 8, 128)
            q_hi_2d = (
                jnp.roll(q_lo_2d, 64, axis=1)
                if interpret_mode
                else pltpu.roll(q_lo_2d, 64, axis=1)
            )
            rpad_q_vmem[q_sl, pl.ds(8, 8), :] = q_hi_2d.reshape(step_q, 8, 128)
        q_for_rpa = rpad_q_vmem
    elif world_size == 16:
        q_for_rpa = rpad_q_vmem.at[:, pl.ds(0, 8), :]
    else:
        q_for_rpa = rpad_q_vmem

    _run_rpad_vmem_core(
        kv_lens_ref=kv_lens_ref,
        page_indices_ref=page_indices_ref,
        cu_q_lens_ref=cu_q_lens_ref,
        cu_kv_lens_ref=cu_kv_lens_ref,
        distribution_ref=distribution_ref,
        q_vmem_ref=q_for_rpa,
        kv_merged_vmem_ref=b2_out_kv,
        gate_vmem_ref=rpad_gate_vmem,
        kv_cache_hbm_ref=kv_cache_hbm_ref,
        o_out_vmem_ref=out_gated_ref,
        updated_kv_cache_hbm_ref=(None if rpa_skip_kv_write else out_kv_or_cache_ref),
        bkv_vmem_ref=bkv_vmem_ref,
        state_vmem_ref=state_vmem_ref,
        o_vmem_ref=o_vmem_ref,
        kv_write_vmem_ref=kv_write_vmem_ref,
        init_sv_vmem_ref=init_sv_vmem_ref,
        written_cnt_smem_ref=written_cnt_smem_ref,
        bkv_sems=rpad_bkv_sems,
        cache_write_sem=rpad_cache_write_sem,
        sm_scale=rpa_sm_scale,
        sliding_window=rpa_sliding_window,
        k_scale=rpa_k_scale,
        v_scale=rpa_v_scale,
        cfg=rpa_cfg,
        one_block_per_seq=rpa_one_block_per_seq,
        decode_query_size=1,
        in_kernel_pack2=True,
        has_gate=True,
        out_norm_eps=rpa_out_norm_eps,
        kv_pack_dense2=rpa_kv_pack_dense2,
        interpret_mode=interpret_mode,
        prestarted_bkv_state=prestarted_bkv_state,
        use_grp_vec=rpa_use_grp_vec,
        skip_kv_write=rpa_skip_kv_write,
        kv_pack_bd4=rpa_kv_pack_bd4,
    )


def _embed_l0_rpad_chip_sharded_kernel(
    *args,
    world_size: int,
    embed_eps: float,
    l0_eps: float,
    l0_scale_input: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    rpa_sm_scale: float,
    rpa_sliding_window: int | None,
    rpa_k_scale: float,
    rpa_v_scale: float,
    rpa_out_norm_eps: float,
    rpa_cfg: Any,
    rpa_one_block_per_seq: bool,
    rpa_use_grp_vec: bool,
    rpa_kv_pack_dense2: bool,
    rpa_kv_pack_bd4: bool,
    rpa_skip_kv_write: bool,
    interpret_mode: bool,
    has_init_sv: bool,
    has_kv_write_buf: bool,
    scope1_shapes: tuple,
    scope2_shapes: tuple,
) -> None:
    """2-scope fused L0 chip-sharded kernel: Embed+RS+Norms+AG (Scope 1) -> QKV+Epi+Gate+RPAd (Scope 2)."""
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
        prepare_and_start_rpad_dmas,
    )

    it = iter(args)
    # Scalar prefetch (SMEM)
    input_ids_smem = next(it)
    kv_lens_ref = next(it)
    page_indices_ref = next(it)
    cu_q_lens_ref = next(it)
    cu_kv_lens_ref = next(it)
    distribution_ref = next(it)
    # Inputs
    embed_hbm = next(it)
    l0_gain_ref = next(it)
    w_qkv_hbm = next(it)
    s_qkv_ref = next(it)
    w_gate_hbm = next(it)
    s_gate_ref = next(it)
    cos_sin_ref = next(it) if has_rope else None
    kv_cache_hbm_ref = next(it)
    # Outputs
    out_stream_chip_hbm = next(it)
    out_gated_ref = next(it)
    out_kv_or_cache_ref = next(it)
    # Outer scratch
    attn_in_vmem = next(it)
    s_out_buf = next(it)
    attn_pair = next(it)
    w_qkv_buf = next(it)
    w_gate_buf = next(it)
    written_cnt_smem_ref = next(it)
    row_sems = next(it)
    ici_send = next(it)
    ici_recv = next(it)
    d2d1_send = next(it)
    d2d1_recv = next(it)
    ag_ici_send = next(it)
    ag_ici_recv = next(it)
    ag_d2d2_send = next(it)
    ag_d2d2_recv = next(it)
    s_out_sem = next(it)
    w_qkv_sem = next(it)
    w_gate_sem = next(it)
    rpad_bkv_sems = next(it)
    rpad_cache_write_sem = None if rpa_skip_kv_write else next(it)

    rows, hidden = attn_in_vmem.shape
    n_chips = world_size // 2
    r_pair = rows // n_chips
    n_strips = hidden // _COL_STRIP

    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)
    chip = me >> jnp.int32(1)

    _signal_barrier(me, world_size)
    # Prestart L0 W_qkv and W_gate DMAs so they hide completely behind embed + RS + norms + AG!
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    def _scope1_embed(*s1_refs) -> None:
        (
            partial_emb_buf,
            slot_2d_buf,
            ici_pair_recv,
            my_pair_bf16,
            mate_pair_bf16,
            emb_3d_buf,
        ) = s1_refs
        _lookup_local_vocab_rows_fast_16row(
            input_ids_smem,
            embed_hbm,
            partial_emb_buf,
            slot_2d_buf,
            row_sems,
            me=me,
            actual_rows=rows,
            interpret_mode=interpret_mode,
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
                v = my_pair_bf16[:, c_sl].astype(jnp.float32) + mate_pair_bf16[
                    :, c_sl
                ].astype(jnp.float32)
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
                        c_sl = pl.ds(
                            pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                            _COL_STRIP,
                        )
                        v = my_pair_bf16[rs, c_sl].astype(jnp.float32) + mate_pair_bf16[
                            rs, c_sl
                        ].astype(jnp.float32)
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
                        c_sl = pl.ds(
                            pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                            _COL_STRIP,
                        )
                        st = (s_out_buf[rs, c_sl] * inv0).astype(jnp.bfloat16).astype(
                            jnp.float32
                        )
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
                        c_sl = pl.ds(
                            pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                            _COL_STRIP,
                        )
                        st = s_out_buf[rs, c_sl]
                        g0 = l0_gain_ref[0:1, c_sl]
                        a0 = (st * g0) * inv1 if l0_scale_input else (st * inv1) * g0
                        attn_pair[rs, c_sl] = a0.astype(jnp.bfloat16)
                    return jnp.int32(0)

                lax.fori_loop(0, n_strips // u_full, _l0_out, jnp.int32(0))

            _fori_strips(r_pair, bm, _norms_pair)
            pltpu.make_async_copy(s_out_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()

    pl.run_scoped(_scope1_embed, *scope1_shapes)

    def _scope2_rpad(*s2_refs) -> None:
        s2_it = iter(s2_refs)
        ag_3d_buf = next(s2_it)
        qkv_vmem = next(s2_it)
        rpad_q_vmem = next(s2_it)
        rpad_kv_vmem = None if rpa_skip_kv_write else next(s2_it)
        rpad_gate_vmem = next(s2_it)
        bkv_vmem_ref = next(s2_it)
        state_vmem_ref = next(s2_it)
        o_vmem_ref = next(s2_it)
        kv_write_vmem_ref = next(s2_it) if has_kv_write_buf else None
        init_sv_vmem_ref = next(s2_it) if has_init_sv else None

        pre_state_holder: list = []
        use_bkv_in_ici = not rpa_kv_pack_dense2 or rpa_use_grp_vec

        def _start_bkv_in_ici() -> None:
            pre_state = prepare_and_start_rpad_dmas(
                kv_lens_ref=kv_lens_ref,
                page_indices_ref=page_indices_ref,
                cu_q_lens_ref=cu_q_lens_ref,
                cu_kv_lens_ref=cu_kv_lens_ref,
                distribution_ref=distribution_ref,
                kv_cache_hbm_ref=kv_cache_hbm_ref,
                bkv_vmem_ref=bkv_vmem_ref,
                bkv_sems=rpad_bkv_sems,
                sliding_window=rpa_sliding_window,
                cfg=rpa_cfg,
                one_block_per_seq=rpa_one_block_per_seq,
                decode_query_size=1,
                kv_pack_dense2=rpa_kv_pack_dense2,
                interpret_mode=interpret_mode,
                b_tokens=rows,
                init_sv_vmem_ref=init_sv_vmem_ref,
                use_grp_vec=rpa_use_grp_vec,
                kv_pack_bd4=rpa_kv_pack_bd4,
            )
            pre_state_holder.append(pre_state)

        _ag_chip_sharded_half_col_xor1(
            attn_pair,
            attn_in_vmem,
            ag_ici_send,
            ag_ici_recv,
            ag_d2d2_send,
            ag_d2d2_recv,
            me=me,
            world_size=world_size,
            ag_3d_buf=ag_3d_buf,
            during_ici_fn=(_start_bkv_in_ici if use_bkv_in_ici else None),
            during_d2d2_fn=lambda: _dma_wait(w_qkv_buf, w_qkv_sem.at[0]),
            r_pair_in=r_pair,
            copy_local_from_pair=True,
            during_ici_early=(
                use_bkv_in_ici
                and ((world_size == 16 and r_pair >= 16) or (world_size == 8 and r_pair >= 32))
            ),
        )
        _run_l0_qkv_epi_gate_rpad_body(
            attn_in_vmem=attn_in_vmem,
            w_qkv_buf=w_qkv_buf,
            s_qkv_ref=s_qkv_ref,
            w_gate_buf=w_gate_buf,
            s_gate_ref=s_gate_ref,
            cos_sin_ref=cos_sin_ref,
            w_qkv_sem=w_qkv_sem,
            w_gate_sem=w_gate_sem,
            qkv_vmem=qkv_vmem,
            rpad_q_vmem=rpad_q_vmem,
            rpad_kv_vmem=rpad_kv_vmem,
            rpad_gate_vmem=rpad_gate_vmem,
            bkv_vmem_ref=bkv_vmem_ref,
            state_vmem_ref=state_vmem_ref,
            o_vmem_ref=o_vmem_ref,
            kv_write_vmem_ref=kv_write_vmem_ref,
            init_sv_vmem_ref=init_sv_vmem_ref,
            written_cnt_smem_ref=written_cnt_smem_ref,
            rpad_bkv_sems=rpad_bkv_sems,
            rpad_cache_write_sem=rpad_cache_write_sem,
            kv_lens_ref=kv_lens_ref,
            page_indices_ref=page_indices_ref,
            cu_q_lens_ref=cu_q_lens_ref,
            cu_kv_lens_ref=cu_kv_lens_ref,
            distribution_ref=distribution_ref,
            kv_cache_hbm_ref=kv_cache_hbm_ref,
            out_gated_ref=out_gated_ref,
            out_kv_or_cache_ref=out_kv_or_cache_ref,
            prestarted_bkv_state=(pre_state_holder[0] if pre_state_holder else None),
            world_size=world_size,
            qkv_norm_eps=qkv_norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            rpa_sm_scale=rpa_sm_scale,
            rpa_sliding_window=rpa_sliding_window,
            rpa_k_scale=rpa_k_scale,
            rpa_v_scale=rpa_v_scale,
            rpa_out_norm_eps=rpa_out_norm_eps,
            rpa_cfg=rpa_cfg,
            rpa_one_block_per_seq=rpa_one_block_per_seq,
            rpa_use_grp_vec=rpa_use_grp_vec,
            rpa_kv_pack_dense2=rpa_kv_pack_dense2,
            rpa_kv_pack_bd4=rpa_kv_pack_bd4,
            rpa_skip_kv_write=rpa_skip_kv_write,
            interpret_mode=interpret_mode,
            waited_w_qkv=True,
        )

    pl.run_scoped(_scope2_rpad, *scope2_shapes)
    _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def _embed_l0_rpad_replicated_b16_kernel(
    *args,
    world_size: int,
    embed_eps: float,
    l0_eps: float,
    l0_scale_input: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    rpa_sm_scale: float,
    rpa_sliding_window: int | None,
    rpa_k_scale: float,
    rpa_v_scale: float,
    rpa_out_norm_eps: float,
    rpa_cfg: Any,
    rpa_one_block_per_seq: bool,
    rpa_use_grp_vec: bool,
    rpa_kv_pack_dense2: bool,
    rpa_kv_pack_bd4: bool,
    rpa_skip_kv_write: bool,
    interpret_mode: bool,
    has_init_sv: bool,
    has_kv_write_buf: bool,
) -> None:
    """Single-scope fused L0 replicated `B in {8, 16}` kernel: Embed+AR+Norms -> QKV+Epi+Gate+RPAd."""
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
        prepare_and_start_rpad_dmas,
    )

    it = iter(args)
    input_ids_smem = next(it)
    kv_lens_ref = next(it)
    page_indices_ref = next(it)
    cu_q_lens_ref = next(it)
    cu_kv_lens_ref = next(it)
    distribution_ref = next(it)
    embed_hbm = next(it)
    l0_gain_ref = next(it)
    w_qkv_hbm = next(it)
    s_qkv_ref = next(it)
    w_gate_hbm = next(it)
    s_gate_ref = next(it)
    cos_sin_ref = next(it) if has_rope else None
    kv_cache_hbm_ref = next(it)
    out_stream_hbm = next(it)
    out_gated_ref = next(it)
    out_kv_or_cache_ref = next(it)
    # Scratch
    partial_emb_buf = next(it)
    slot_2d_buf = next(it)
    mate_buf = next(it)
    ici_buf = next(it)
    s_out_buf = next(it)
    attn_in_vmem = next(it)
    w_qkv_buf = next(it)
    w_gate_buf = next(it)
    qkv_vmem = next(it)
    rpad_q_vmem = next(it)
    rpad_kv_vmem = None if rpa_skip_kv_write else next(it)
    rpad_gate_vmem = next(it)
    bkv_vmem_ref = next(it)
    state_vmem_ref = next(it)
    o_vmem_ref = next(it)
    kv_write_vmem_ref = next(it) if has_kv_write_buf else None
    init_sv_vmem_ref = next(it) if has_init_sv else None
    written_cnt_smem_ref = next(it)
    row_sems = next(it)
    d2d_send = next(it)
    d2d_recv = next(it)
    ici_send = next(it)
    ici_recv = next(it)
    s_out_sem = next(it)
    w_qkv_sem = next(it)
    w_gate_sem = next(it)
    rpad_bkv_sems = next(it)
    rpad_cache_write_sem = None if rpa_skip_kv_write else next(it)

    actual_rows = out_stream_hbm.shape[0]
    pad_rows, hidden = partial_emb_buf.shape
    n_chips = world_size // 2
    me = lax.axis_index("model").astype(jnp.int32)
    mate = me ^ jnp.int32(1)

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    _lookup_local_vocab_rows_fast_16row(
        input_ids_smem,
        embed_hbm,
        partial_emb_buf,
        slot_2d_buf,
        row_sems,
        me=me,
        actual_rows=actual_rows,
        interpret_mode=interpret_mode,
    )
    _wait_barrier(world_size)

    c1 = pltpu.make_async_remote_copy(
        partial_emb_buf,
        mate_buf,
        d2d_send.at[0],
        d2d_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c1.start()
    c1.wait()

    col_blk = 1024 if (hidden % 1024 == 0) else 512
    for c0 in range(0, hidden, col_blk):
        c_sl = pl.ds(c0, col_blk)
        v = partial_emb_buf[:, c_sl].astype(jnp.float32) + mate_buf[:, c_sl].astype(
            jnp.float32
        )
        mate_buf[:, c_sl] = v.astype(jnp.bfloat16)

    c2 = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op = pltpu.make_async_remote_copy(
            mate_buf,
            ici_buf.at[d - 1],
            ici_send.at[d - 1],
            ici_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        c2.append(op)
    # Overlap L0 RPAd bkv DMAs behind the 1-shot ICI all-reduce!
    pre_state = prepare_and_start_rpad_dmas(
        kv_lens_ref=kv_lens_ref,
        page_indices_ref=page_indices_ref,
        cu_q_lens_ref=cu_q_lens_ref,
        cu_kv_lens_ref=cu_kv_lens_ref,
        distribution_ref=distribution_ref,
        kv_cache_hbm_ref=kv_cache_hbm_ref,
        bkv_vmem_ref=bkv_vmem_ref,
        bkv_sems=rpad_bkv_sems,
        sliding_window=rpa_sliding_window,
        cfg=rpa_cfg,
        one_block_per_seq=rpa_one_block_per_seq,
        decode_query_size=1,
        kv_pack_dense2=rpa_kv_pack_dense2,
        interpret_mode=interpret_mode,
        b_tokens=actual_rows,
        init_sv_vmem_ref=init_sv_vmem_ref,
        use_grp_vec=rpa_use_grp_vec,
        kv_pack_bd4=rpa_kv_pack_bd4,
    )
    for op in c2:
        op.wait()

    sq0 = jnp.zeros((pad_rows, 128), dtype=jnp.float32)
    for c0 in range(0, hidden, col_blk):
        c_sl = pl.ds(c0, col_blk)
        tot = mate_buf[:, c_sl].astype(jnp.float32)
        for slot in range(n_chips - 1):
            tot = tot + ici_buf[slot, :, c_sl].astype(jnp.float32)
        s_out_buf[:, c_sl] = tot
        sq0 = sq0 + _fold_cols_to_128(tot * tot)
    inv0 = lax.rsqrt(
        jnp.sum(sq0, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
        + jnp.float32(embed_eps)
    )
    sq1 = jnp.zeros((pad_rows, 128), dtype=jnp.float32)
    for c0 in range(0, hidden, col_blk):
        c_sl = pl.ds(c0, col_blk)
        st = (s_out_buf[:, c_sl] * inv0).astype(jnp.bfloat16).astype(jnp.float32)
        s_out_buf[:, c_sl] = st
        lv = st * l0_gain_ref[0:1, c_sl] if l0_scale_input else st
        sq1 = sq1 + _fold_cols_to_128(lv * lv)
    s_src = s_out_buf if pad_rows == actual_rows else s_out_buf.at[pl.ds(0, actual_rows), :]
    pltpu.make_async_copy(s_src, out_stream_hbm, s_out_sem.at[0]).start()
    inv1 = lax.rsqrt(
        jnp.sum(sq1, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
        + jnp.float32(l0_eps)
    )
    for c0 in range(0, hidden, col_blk):
        c_sl = pl.ds(c0, col_blk)
        st = s_out_buf[pl.ds(0, actual_rows), c_sl]
        g0 = l0_gain_ref[0:1, c_sl]
        inv1_a = inv1[0:actual_rows, :]
        a0 = (st * g0) * inv1_a if l0_scale_input else (st * inv1_a) * g0
        attn_in_vmem[:, c_sl] = a0.astype(jnp.bfloat16)

    _run_l0_qkv_epi_gate_rpad_body(
        attn_in_vmem=attn_in_vmem,
        w_qkv_buf=w_qkv_buf,
        s_qkv_ref=s_qkv_ref,
        w_gate_buf=w_gate_buf,
        s_gate_ref=s_gate_ref,
        cos_sin_ref=cos_sin_ref,
        w_qkv_sem=w_qkv_sem,
        w_gate_sem=w_gate_sem,
        qkv_vmem=qkv_vmem,
        rpad_q_vmem=rpad_q_vmem,
        rpad_kv_vmem=rpad_kv_vmem,
        rpad_gate_vmem=rpad_gate_vmem,
        bkv_vmem_ref=bkv_vmem_ref,
        state_vmem_ref=state_vmem_ref,
        o_vmem_ref=o_vmem_ref,
        kv_write_vmem_ref=kv_write_vmem_ref,
        init_sv_vmem_ref=init_sv_vmem_ref,
        written_cnt_smem_ref=written_cnt_smem_ref,
        rpad_bkv_sems=rpad_bkv_sems,
        rpad_cache_write_sem=rpad_cache_write_sem,
        kv_lens_ref=kv_lens_ref,
        page_indices_ref=page_indices_ref,
        cu_q_lens_ref=cu_q_lens_ref,
        cu_kv_lens_ref=cu_kv_lens_ref,
        distribution_ref=distribution_ref,
        kv_cache_hbm_ref=kv_cache_hbm_ref,
        out_gated_ref=out_gated_ref,
        out_kv_or_cache_ref=out_kv_or_cache_ref,
        prestarted_bkv_state=pre_state,
        world_size=world_size,
        qkv_norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        rpa_sm_scale=rpa_sm_scale,
        rpa_sliding_window=rpa_sliding_window,
        rpa_k_scale=rpa_k_scale,
        rpa_v_scale=rpa_v_scale,
        rpa_out_norm_eps=rpa_out_norm_eps,
        rpa_cfg=rpa_cfg,
        rpa_one_block_per_seq=rpa_one_block_per_seq,
        rpa_use_grp_vec=rpa_use_grp_vec,
        rpa_kv_pack_dense2=rpa_kv_pack_dense2,
        rpa_kv_pack_bd4=rpa_kv_pack_bd4,
        rpa_skip_kv_write=rpa_skip_kv_write,
        interpret_mode=interpret_mode,
        waited_w_qkv=False,
    )
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def _prepare_l0_rpad_common(
    input_ids: jax.Array,
    embed_weight_shard: jax.Array,
    l0_in_gain: jax.Array,
    l0_w_qkv_w8k: jax.Array,
    l0_s_qkv: jax.Array,
    l0_w_gate_w8k: jax.Array,
    l0_s_gate: jax.Array,
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    cos_sin_step: jax.Array | None,
    layer_idx: jax.Array | None,
    *,
    world_size: int,
    valid_rows: tuple[int, ...],
    rpa_sliding_window: int | None,
    rpa_mode_str: str | None,
    kv_pack_dense2: bool,
    kv_pack_bd4: bool,
    skip_kv_write: bool,
    allow_large_b: bool,
    interpret: bool | object,
) -> dict:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = int(input_ids.shape[0])
    v_shard, hidden = embed_weight_shard.shape
    _, qkv_cols = l0_w_qkv_w8k.shape
    _, gate_cols = l0_w_gate_w8k.shape
    q_heads = 8 if world_size == 16 else 16
    pack2_cols = 512 if world_size == 16 else 1024
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or rows not in valid_rows
        or v_shard < 16
        or v_shard % 2 != 0
        or hidden % 1024 != 0
        or gate_cols != pack2_cols
    ):
        raise ValueError(
            f"Unsupported embed_l0_rpad shapes: world_size={world_size}, rows={rows}, "
            f"embed={embed_weight_shard.shape}, w_qkv={l0_w_qkv_w8k.shape}, w_gate={l0_w_gate_w8k.shape}"
        )

    eff_sw = (
        None
        if (rpa_sliding_window is None or rpa_sliding_window <= 0)
        else int(rpa_sliding_window)
    )
    kv_cache_flat = kv_cache_fused
    eff_page_indices = page_indices
    if layer_idx is not None:
        num_blocks, layer_num = kv_cache_fused.shape[:2]
        kv_cache_flat = kv_cache_fused.reshape(
            (num_blocks * layer_num,) + kv_cache_fused.shape[2:]
        )
        eff_page_indices = (
            page_indices * jnp.int32(layer_num)
            + layer_idx.reshape(()).astype(jnp.int32)
        )

    if rpa_mode_str is None:
        try:
            from sglang_tpu.envs import envs as tpu_envs

            rpa_mode_str = str(tpu_envs.SGLANG_TPU_RPA_V3_DECODE_FAST.get()).strip()
        except Exception:
            rpa_mode_str = ""
        if not rpa_mode_str:
            rpa_mode_str = "v2"

    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(eff_page_indices.shape[0]) // max(1, max_num_seqs)
    phys_page_rows = int(kv_cache_flat.shape[1])
    dist_len = int(distribution.shape[0])
    has_shared_prefix = dist_len > 3
    shared_group_size = (
        max_num_seqs // (dist_len - 3)
        if (has_shared_prefix and (max_num_seqs % (dist_len - 3)) == 0)
        else 0
    )
    cfg_rpa, one_block_per_seq, bkv_slot_pages, use_grp_vec, is_ok = resolve_step_c_rpa_plan(
        rpa_mode_str,
        sliding_window=eff_sw,
        pages_per_seq=pages_per_seq,
        phys_page_rows=phys_page_rows,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
        b_tokens=rows,
        max_num_seqs=max_num_seqs,
        shared_group_size=shared_group_size,
        has_shared_prefix=has_shared_prefix,
        allow_large_b=allow_large_b,
    )
    if not is_ok or (world_size == 16 and cfg_rpa.num_slots > 8):
        raise ValueError(
            "embed_l0_rpad requires compatible RPAd plan; "
            f"got num_slots={cfg_rpa.num_slots}, max_ring_pages={cfg_rpa.max_ring_pages}, "
            f"phys_page_rows={phys_page_rows}, loop_kind={cfg_rpa.loop_kind!r}, B={rows}"
        )

    if isinstance(interpret, bool):
        interpret_arg = (
            pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
            if interpret
            else False
        )
    else:
        interpret_arg = interpret
    interpret_mode = interpret_arg is not False

    has_rope = cos_sin_step is not None
    cs_in = (
        cos_sin_step[:rows]
        if (has_rope and cos_sin_step.shape[0] != rows)
        else cos_sin_step
    )
    vmem, hbm, smem, sem = pltpu.VMEM, pltpu.HBM, pltpu.SMEM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    max_blks_4k = (
        _cdiv(max(pages_per_seq, 1), cfg_rpa.max_ring_pages)
        if cfg_rpa.loop_kind == "ring4k"
        else 0
    )
    sh_grp = cfg_rpa.shared_group_size or (16 if cfg_rpa.loop_kind == "shared16" else 0)
    max_blks_sh = (
        _cdiv(max(pages_per_seq, 1), cfg_rpa.max_ring_pages) if sh_grp > 0 else 0
    )
    sh_smem_words = 4 * max(sh_grp * max_blks_sh, 0)
    if (max_num_seqs * pages_per_seq + 2 + cfg_rpa.num_slots + sh_smem_words) * 4 > 960 * 1024:
        sh_smem_words = 0
    smem_words = (
        2
        + cfg_rpa.num_slots
        + max(4 * max(max_num_seqs * max_blks_4k, 0), sh_smem_words)
    )
    rg_grp = (
        cfg_rpa.ring_group_size
        if (use_grp_vec and cfg_rpa.loop_kind == "ring4k")
        else 1
    )
    state_rows = q_heads * max(sh_grp, rg_grp)
    b_pad32 = _align_to(max(rows, 1), 32)
    has_init_sv = bool(cfg_rpa.hoist_new_kv or use_grp_vec)
    has_kv_write_buf = bool((kv_pack_dense2 or kv_pack_bd4) and not skip_kv_write)

    scalar_args = [
        input_ids.astype(jnp.int32),
        kv_lens,
        eff_page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
    ]
    kv_in_arg = (
        kv_cache_flat
        if interpret_mode
        else pltpu.with_memory_space_constraint(kv_cache_flat, hbm)
    )
    in_specs = (
        h_sp,
        v_sp,
        h_sp,
        v_sp,
        h_sp,
        v_sp,
        *((v_sp,) if has_rope else ()),
        h_sp,
    )
    call_args = (
        *scalar_args,
        embed_weight_shard,
        _as_2d_f32(l0_in_gain, hidden),
        l0_w_qkv_w8k,
        _as_dot_scale_f32(l0_s_qkv, qkv_cols),
        l0_w_gate_w8k,
        _as_dot_scale_f32(l0_s_gate, gate_cols),
        *((cs_in,) if has_rope else ()),
        kv_in_arg,
    )
    kv_cache_pallas_in = len(call_args) - 1
    if skip_kv_write:
        kv_out_shape = jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn)
        kv_out_spec = v_sp
        pallas_aliases = {}
    else:
        kv_out_shape = (
            jax.ShapeDtypeStruct(kv_cache_flat.shape, kv_cache_flat.dtype)
            if interpret_mode
            else pltpu.HBM(kv_cache_flat.shape, kv_cache_flat.dtype)
        )
        kv_out_spec = h_sp
        pallas_aliases = {kv_cache_pallas_in: 2}

    return dict(
        rows=rows,
        hidden=hidden,
        qkv_cols=qkv_cols,
        gate_cols=gate_cols,
        q_heads=q_heads,
        pack2_cols=pack2_cols,
        eff_sw=eff_sw,
        cfg_rpa=cfg_rpa,
        one_block_per_seq=one_block_per_seq,
        bkv_slot_pages=bkv_slot_pages,
        use_grp_vec=use_grp_vec,
        phys_page_rows=phys_page_rows,
        kv_dtype=kv_cache_flat.dtype,
        kv_orig_shape=kv_cache_fused.shape,
        has_rope=has_rope,
        has_init_sv=has_init_sv,
        has_kv_write_buf=has_kv_write_buf,
        smem_words=smem_words,
        state_rows=state_rows,
        b_pad32=b_pad32,
        vmem=vmem,
        hbm=hbm,
        smem=smem,
        sem=sem,
        v_sp=v_sp,
        h_sp=h_sp,
        interpret_arg=interpret_arg,
        interpret_mode=interpret_mode,
        scalar_args=scalar_args,
        in_specs=in_specs,
        call_args=call_args,
        kv_out_shape=kv_out_shape,
        kv_out_spec=kv_out_spec,
        pallas_aliases=pallas_aliases,
        pack_tag="_bd4" if kv_pack_bd4 else ("_d2" if kv_pack_dense2 else ""),
    )


def embed_l0_rpad_chip_sharded_jax(
    input_ids: jax.Array,
    embed_weight_shard: jax.Array,
    l0_in_gain: jax.Array,
    l0_w_qkv_w8k: jax.Array,
    l0_s_qkv: jax.Array,
    l0_w_gate_w8k: jax.Array,
    l0_s_gate: jax.Array,
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    cos_sin_step: jax.Array | None = None,
    layer_idx: jax.Array | None = None,
    *,
    world_size: int = 8,
    embed_eps: float = 1e-5,
    l0_eps: float = 1e-5,
    l0_scale_input: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rpa_sm_scale: float = 1.0,
    rpa_sliding_window: int | None = None,
    rpa_k_scale: float = 1.0,
    rpa_v_scale: float = 1.0,
    rpa_out_norm_eps: float = 1e-5,
    rpa_mode_str: str | None = None,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    skip_kv_write: bool = True,
    allow_large_b: bool = True,
    collective_id: int = _W8_CID_BASE + 34,
    interpret: bool | object = False,
):
    """Fused L0 chip-sharded megakernel (`TP=8 B in {32..256}`, `TP=16 B in {64..256}`)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    valid_rows = (32, 64, 128, 256) if world_size == 8 else (64, 128, 256)
    st = _prepare_l0_rpad_common(
        input_ids, embed_weight_shard, l0_in_gain,
        l0_w_qkv_w8k, l0_s_qkv, l0_w_gate_w8k, l0_s_gate,
        kv_cache_fused, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution,
        cos_sin_step, layer_idx,
        world_size=world_size,
        valid_rows=valid_rows,
        rpa_sliding_window=rpa_sliding_window,
        rpa_mode_str=rpa_mode_str,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
        skip_kv_write=skip_kv_write,
        allow_large_b=allow_large_b,
        interpret=interpret,
    )
    rows = st["rows"]
    hidden = st["hidden"]
    n_chips = world_size // 2
    r_pair = rows // n_chips
    half_h = hidden // 2
    vmem, smem, sem = st["vmem"], st["smem"], st["sem"]
    cfg_rpa = st["cfg_rpa"]

    scope1_shapes = (
        vmem((rows, hidden), jnp.bfloat16),  # partial_emb_buf
        vmem((128, hidden), jnp.bfloat16),  # slot_2d_buf (8 sems x 16 rows)
        vmem((n_chips - 1, r_pair, hidden), jnp.bfloat16),  # ici_pair_recv
        vmem((r_pair, hidden), jnp.bfloat16),  # my_pair_bf16
        vmem((r_pair, hidden), jnp.bfloat16),  # mate_pair_bf16
        vmem((n_chips, 8, hidden) if r_pair == 8 else (8, 128), jnp.bfloat16),  # emb_3d_buf
    )
    scope2_shapes = (
        vmem((n_chips, 8, half_h) if r_pair == 8 else (8, 128), jnp.bfloat16),  # ag_3d_buf
        vmem((rows, st["qkv_cols"]), jnp.bfloat16),  # qkv_vmem
        vmem((rows, 16, 128), jnp.bfloat16),  # rpad_q_vmem
        *(
            ()
            if skip_kv_write
            else (vmem((rows, 1, 4, 128), jnp.float8_e4m3fn),)
        ),
        vmem((rows, st["pack2_cols"]), jnp.bfloat16),  # rpad_gate_vmem
        vmem(
            (cfg_rpa.num_slots, st["bkv_slot_pages"], st["phys_page_rows"], 1, 4, 128),
            st["kv_dtype"],
        ),  # bkv_vmem_ref
        vmem((3, st["state_rows"], 128), jnp.float32),  # state_vmem_ref
        vmem((rows, st["q_heads"], 128), jnp.bfloat16),  # o_vmem_ref
        *(
            (vmem((max(rows, 8), 8, 1, 4, 128), st["kv_dtype"]),)
            if st["has_kv_write_buf"]
            else ()
        ),
        *(
            (vmem((2, st["b_pad32"], st["q_heads"], 128), jnp.float32),)
            if st["has_init_sv"]
            else ()
        ),
    )
    outer_scratch = [
        vmem((rows, hidden), jnp.bfloat16),  # attn_in_vmem
        vmem((r_pair, hidden), jnp.float32),  # s_out_buf
        vmem((r_pair, hidden), jnp.bfloat16),  # attn_pair
        vmem((hidden, st["qkv_cols"]), l0_w_qkv_w8k.dtype),  # w_qkv_buf
        vmem((hidden, st["gate_cols"]), l0_w_gate_w8k.dtype),  # w_gate_buf
        smem((st["smem_words"],), jnp.int32),  # written_cnt_smem_ref
        sem((8,)),  # row_sems
        sem((n_chips - 1,)),  # ici_send
        sem((n_chips - 1,)),  # ici_recv
        sem((1,)),  # d2d1_send
        sem((1,)),  # d2d1_recv
        sem((5 if world_size == 8 else n_chips - 1,)),  # ag_ici_send
        sem((5 if world_size == 8 else n_chips - 1,)),  # ag_ici_recv
        sem((1,)),  # ag_d2d2_send
        sem((1,)),  # ag_d2d2_recv
        sem((1,)),  # s_out_sem
        sem((1,)),  # w_qkv_sem
        sem((1,)),  # w_gate_sem
        sem((cfg_rpa.num_slots,)),  # rpad_bkv_sems
        *([] if skip_kv_write else [sem((1,))]),  # rpad_cache_write_sem
    ]

    kernel_fn = functools.partial(
        _embed_l0_rpad_chip_sharded_kernel,
        world_size=world_size,
        embed_eps=float(embed_eps),
        l0_eps=float(l0_eps),
        l0_scale_input=bool(l0_scale_input),
        qkv_norm_eps=float(qkv_norm_eps),
        query_scale=float(query_scale),
        k_scale=float(k_scale),
        v_scale=float(v_scale),
        has_rope=st["has_rope"],
        rpa_sm_scale=float(rpa_sm_scale),
        rpa_sliding_window=st["eff_sw"],
        rpa_k_scale=float(rpa_k_scale),
        rpa_v_scale=float(rpa_v_scale),
        rpa_out_norm_eps=float(rpa_out_norm_eps),
        rpa_cfg=cfg_rpa,
        rpa_one_block_per_seq=bool(st["one_block_per_seq"]),
        rpa_use_grp_vec=bool(st["use_grp_vec"]),
        rpa_kv_pack_dense2=bool(kv_pack_dense2),
        rpa_kv_pack_bd4=bool(kv_pack_bd4),
        rpa_skip_kv_write=bool(skip_kv_write),
        interpret_mode=st["interpret_mode"],
        has_init_sv=st["has_init_sv"],
        has_kv_write_buf=st["has_kv_write_buf"],
        scope1_shapes=scope1_shapes,
        scope2_shapes=scope2_shapes,
    )
    st_ch, gated_out, kv_out = pl.pallas_call(
        kernel_fn,
        out_shape=(
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, st["pack2_cols"]), jnp.bfloat16),
            st["kv_out_shape"],
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(st["scalar_args"]),
            grid=(1,),
            in_specs=st["in_specs"],
            out_specs=(st["h_sp"], st["v_sp"], st["kv_out_spec"]),
            scratch_shapes=outer_scratch,
        ),
        input_output_aliases=st["pallas_aliases"],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(63 if rows > 64 else 60) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=st["interpret_arg"],
        name=(
            f"muse_w18_embed_l0_rpad_b{rows}_tp{world_size}{st['pack_tag']}"
            f"_{'rope' if st['has_rope'] else 'nope'}"
        ),
    )(*st["call_args"])
    if (not skip_kv_write) and layer_idx is not None:
        kv_out = kv_out.reshape(st["kv_orig_shape"])
    return st_ch, gated_out, kv_out


def embed_l0_rpad_replicated_b16_jax(
    input_ids: jax.Array,
    embed_weight_shard: jax.Array,
    l0_in_gain: jax.Array,
    l0_w_qkv_w8k: jax.Array,
    l0_s_qkv: jax.Array,
    l0_w_gate_w8k: jax.Array,
    l0_s_gate: jax.Array,
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    cos_sin_step: jax.Array | None = None,
    layer_idx: jax.Array | None = None,
    *,
    world_size: int = 8,
    embed_eps: float = 1e-5,
    l0_eps: float = 1e-5,
    l0_scale_input: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rpa_sm_scale: float = 1.0,
    rpa_sliding_window: int | None = None,
    rpa_k_scale: float = 1.0,
    rpa_v_scale: float = 1.0,
    rpa_out_norm_eps: float = 1e-5,
    rpa_mode_str: str | None = None,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    skip_kv_write: bool = True,
    collective_id: int = _W8_CID_BASE + 35,
    interpret: bool | object = False,
):
    """Fused L0 replicated `B in {8, 16}` megakernel (`TP=8` & `TP=16`)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    st = _prepare_l0_rpad_common(
        input_ids, embed_weight_shard, l0_in_gain,
        l0_w_qkv_w8k, l0_s_qkv, l0_w_gate_w8k, l0_s_gate,
        kv_cache_fused, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution,
        cos_sin_step, layer_idx,
        world_size=world_size,
        valid_rows=(8, 16),
        rpa_sliding_window=rpa_sliding_window,
        rpa_mode_str=rpa_mode_str,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
        skip_kv_write=skip_kv_write,
        allow_large_b=False,
        interpret=interpret,
    )
    rows = st["rows"]
    pad_rows = max(rows, 16)
    hidden = st["hidden"]
    n_chips = world_size // 2
    vmem, smem, sem = st["vmem"], st["smem"], st["sem"]
    cfg_rpa = st["cfg_rpa"]

    scratch_shapes = [
        vmem((pad_rows, hidden), jnp.bfloat16),  # partial_emb_buf
        vmem((128, hidden), jnp.bfloat16),  # slot_2d_buf (8 sems x 16 rows)
        vmem((pad_rows, hidden), jnp.bfloat16),  # mate_buf
        vmem((n_chips - 1, pad_rows, hidden), jnp.bfloat16),  # ici_buf
        vmem((pad_rows, hidden), jnp.float32),  # s_out_buf
        vmem((rows, hidden), jnp.bfloat16),  # attn_in_vmem
        vmem((hidden, st["qkv_cols"]), l0_w_qkv_w8k.dtype),  # w_qkv_buf
        vmem((hidden, st["gate_cols"]), l0_w_gate_w8k.dtype),  # w_gate_buf
        vmem((rows, st["qkv_cols"]), jnp.bfloat16),  # qkv_vmem
        vmem((rows, 16, 128), jnp.bfloat16),  # rpad_q_vmem
        *(
            []
            if skip_kv_write
            else [vmem((rows, 1, 4, 128), jnp.float8_e4m3fn)]
        ),
        vmem((rows, st["pack2_cols"]), jnp.bfloat16),  # rpad_gate_vmem
        vmem(
            (cfg_rpa.num_slots, st["bkv_slot_pages"], st["phys_page_rows"], 1, 4, 128),
            st["kv_dtype"],
        ),  # bkv_vmem_ref
        vmem((3, st["state_rows"], 128), jnp.float32),  # state_vmem_ref
        vmem((rows, st["q_heads"], 128), jnp.bfloat16),  # o_vmem_ref
        *(
            [vmem((max(rows, 8), 8, 1, 4, 128), st["kv_dtype"])]
            if st["has_kv_write_buf"]
            else []
        ),
        *(
            [vmem((2, st["b_pad32"], st["q_heads"], 128), jnp.float32)]
            if st["has_init_sv"]
            else []
        ),
        smem((st["smem_words"],), jnp.int32),  # written_cnt_smem_ref
        sem((8,)),  # row_sems
        sem((1,)),  # d2d_send
        sem((1,)),  # d2d_recv
        sem((n_chips - 1,)),  # ici_send
        sem((n_chips - 1,)),  # ici_recv
        sem((1,)),  # s_out_sem
        sem((1,)),  # w_qkv_sem
        sem((1,)),  # w_gate_sem
        sem((cfg_rpa.num_slots,)),  # rpad_bkv_sems
        *([] if skip_kv_write else [sem((1,))]),  # rpad_cache_write_sem
    ]

    kernel_fn = functools.partial(
        _embed_l0_rpad_replicated_b16_kernel,
        world_size=world_size,
        embed_eps=float(embed_eps),
        l0_eps=float(l0_eps),
        l0_scale_input=bool(l0_scale_input),
        qkv_norm_eps=float(qkv_norm_eps),
        query_scale=float(query_scale),
        k_scale=float(k_scale),
        v_scale=float(v_scale),
        has_rope=st["has_rope"],
        rpa_sm_scale=float(rpa_sm_scale),
        rpa_sliding_window=st["eff_sw"],
        rpa_k_scale=float(rpa_k_scale),
        rpa_v_scale=float(rpa_v_scale),
        rpa_out_norm_eps=float(rpa_out_norm_eps),
        rpa_cfg=cfg_rpa,
        rpa_one_block_per_seq=bool(st["one_block_per_seq"]),
        rpa_use_grp_vec=bool(st["use_grp_vec"]),
        rpa_kv_pack_dense2=bool(kv_pack_dense2),
        rpa_kv_pack_bd4=bool(kv_pack_bd4),
        rpa_skip_kv_write=bool(skip_kv_write),
        interpret_mode=st["interpret_mode"],
        has_init_sv=st["has_init_sv"],
        has_kv_write_buf=st["has_kv_write_buf"],
    )
    st_full, gated_out, kv_out = pl.pallas_call(
        kernel_fn,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, st["pack2_cols"]), jnp.bfloat16),
            st["kv_out_shape"],
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(st["scalar_args"]),
            grid=(1,),
            in_specs=st["in_specs"],
            out_specs=(st["h_sp"], st["v_sp"], st["kv_out_spec"]),
            scratch_shapes=scratch_shapes,
        ),
        input_output_aliases=st["pallas_aliases"],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=56 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=st["interpret_arg"],
        name=(
            f"muse_w18_embed_l0_rpad_rep_b{rows}_tp{world_size}{st['pack_tag']}"
            f"_{'rope' if st['has_rope'] else 'nope'}"
        ),
    )(*st["call_args"])
    if (not skip_kv_write) and layer_idx is not None:
        kv_out = kv_out.reshape(st["kv_orig_shape"])
    return st_full, gated_out, kv_out
