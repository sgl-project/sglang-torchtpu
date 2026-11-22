# SPDX-License-Identifier: Apache-2.0
"""Chip-hierarchical Pallas attention o_proj all-reduce + norm/residual fusion for Muse Spark.

Exploits TPU7x's ``XOR = 1`` intra-chip die-to-die link (~733 GB/s bidirectional vs ~90 GB/s
inter-chip ICI) to cut inter-chip all-reduce traffic by 57% (``7/8 -> 3/8`` on TP8,
``15/16 -> 7/16`` on TP16) while fusing the entire serial post-attention boundary with
zero VREG spills (``<= 16`` live VREGs), incast-free cyclic ICI schedules, and
double-buffered HBM output streaming (zero post-kernel XLA copy ops):

  1. ``bridge_all_reduce_jax``: Standalone 2-stage ``XOR=1`` bridge RS + AG on ``[B, H] bf16``
     with overlapped HBM output write and optional 2-row-half pipelining (``pipe2``).
  2. ``tp_ar_res_norm_jax``: Fuses ``o_proj`` AR + ``post_attention_layernorm`` (weightless
     RMSNorm) + ``post_attention_residual_gate`` (``alpha * stream + beta * branch``) +
     ``pre_feedforward_layernorm`` + ``bfloat16`` cast, streaming ``(stream_f32, router_f32,
     hs_bf16)`` directly to HBM via a 3.5 MiB double-buffered VMEM ring (``flat``, ``bridge``,
     and ``bridge_pipe2`` modes).
  3. ``tp_attn_tail_ar_res_norm_jax``: Further fuses ``_per_head_normalize(attn_out)`` +
     ``sigmoid(gate) * attn_out`` + ``o_proj`` matmul (exact BF16 or weight-only FP8 K-major),
     computing the chipmate's column half first so ``o_proj``'s own-half MXU/HBM work overlaps
     the Stage-1 ``XOR=1`` bridge DMA and ``o_partial [B, H]`` never touches HBM.
  4. ``fp8_linear_bf16_jax`` / ``quantize_fp8_k_major``: Zero-spill streamed weight-only FP8
     linear (``1x`` HBM weight read, ``[32, 256]`` MXU sub-tiling) emitting ``bfloat16``.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from collections.abc import Callable
from threading import Lock

import torch

TPU7X_CHIPMATE_XOR = 1
_DEFAULT_COLLECTIVE_ID = 7

_OPS_LOCK = Lock()
_BRIDGE_AR_OPS: dict[int, Callable] = {}
_TP_AR_RES_NORM_OPS: dict[int, Callable] = {}
_ATTN_TAIL_BF16_OPS: dict[int, Callable] = {}
_ATTN_TAIL_FP8_OPS: dict[int, Callable] = {}
_FP8_LINEAR_BF16_OP: list[Callable] = []


# =============================================================================
# Shared Low-VREG Pallas Building Blocks (<= 16 live VREGs -> 0 / 0 spills)
# =============================================================================
def _entry_barrier(me, world_size: int) -> None:
    """Global entry barrier ensuring all peers' destination VMEM buffers are live."""
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=(jnp.mod(me + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)


def _sq_strip_lanes(x_f32, strip: int):
    """Reduce ``x_f32 [16, strip]`` into ``[16, 128]`` square sums using pure VALU ops."""
    sq = x_f32 * x_f32
    if strip == 256:
        return sq[:, 0:128] + sq[:, 128:256]
    return sq


def _accum_half_sq_vmem(
    buf_ref,
    sq_vmem,
    *,
    row_count: int,
    half_h: int,
    add_to_existing: bool,
) -> None:
    """Accumulate ``sum(x^2)`` over ``buf_ref [row_count, half_h]`` into ``sq_vmem [row_count, 128]``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    bt = 16
    strip = 256 if half_h % 256 == 0 else 128
    n_strips = half_h // strip
    n_tiles = row_count // bt

    def _row_body(t, _):
        r0 = pl.multiple_of(t * bt, bt)
        rs = pl.ds(r0, bt)

        def _inner(s, acc):
            c0 = pl.multiple_of(s * strip, strip)
            x = buf_ref[rs, pl.ds(c0, strip)].astype(jnp.float32)
            return acc + _sq_strip_lanes(x, strip)

        acc = lax.fori_loop(0, n_strips, _inner, jnp.zeros((bt, 128), dtype=jnp.float32))
        if add_to_existing:
            sq_vmem[rs, :] = sq_vmem[rs, :] + acc
        else:
            sq_vmem[rs, :] = acc
        return jnp.int32(0)

    if n_tiles == 1:
        _row_body(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_tiles, _row_body, jnp.int32(0))


def _prefetch_stream_initial(
    stream_hbm_ref,
    s_in_buf,
    s_in_sem,
    *,
    row_base: int,
    n_tiles: int,
    nbuf: int,
) -> None:
    """Kick off initial 1 or 2 HBM->VMEM ``stream`` tile reads (overlapped under Bridge AG)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    bt = 16
    pltpu.make_async_copy(
        stream_hbm_ref.at[pl.ds(row_base, bt)],
        s_in_buf.at[0],
        s_in_sem.at[0],
    ).start()
    if nbuf >= 2 and n_tiles >= 2:
        pltpu.make_async_copy(
            stream_hbm_ref.at[pl.ds(row_base + bt, bt)],
            s_in_buf.at[1],
            s_in_sem.at[1],
        ).start()


def _fused_dual_res_norm_hbm_pipelined(
    ag_half_buf,
    ag_mate_buf,
    own_c0,
    mate_c0,
    sq_vmem,
    stream_src_ref,
    alpha_ref,
    beta_ref,
    next_gain_ref,
    stream_out_hbm,
    router_f32_hbm,
    hs_bf16_hbm,
    s_in_buf,
    s_out_buf,
    r_out_buf,
    h_out_buf,
    s_in_sem,
    s_out_sem,
    r_out_sem,
    h_out_sem,
    *,
    row_base: int,
    row_count: int,
    hidden: int,
    branch_eps: float,
    next_eps: float,
    stream_in_hbm: bool,
    stream_already_prefetched: bool,
    nbuf: int,
) -> None:
    """Zero-spill double-buffered HBM ``post_attn_norm + gated_res + pre_ffn_norm + bf16``.

    Uses static slot indices ``0`` and ``1`` (paired ``lax.fori_loop``), ``strip = 256``
    (``<= 16`` live VREGs), and commutative ``next_own + next_mate`` so ``par=0`` and
    ``par=1`` execute the exact same code with bit-identical outputs and zero ``lax.cond``
    duplication.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    bt = 16
    half_h = hidden // 2
    strip = 256 if half_h % 256 == 0 else 128
    half_strips = half_h // strip
    full_strips = hidden // strip
    n_tiles = row_count // bt
    inv_h = 1.0 / float(hidden)

    if stream_in_hbm and not stream_already_prefetched:
        _prefetch_stream_initial(
            stream_src_ref,
            s_in_buf,
            s_in_sem,
            row_base=row_base,
            n_tiles=n_tiles,
            nbuf=nbuf,
        )

    def _run_tile(t, slot: int, wait_prev_out: bool, check_prefetch: bool) -> None:
        r_loc = pl.multiple_of(t * bt, bt)
        r_glb = pl.multiple_of(row_base + t * bt, bt)
        rs_loc = pl.ds(r_loc, bt)
        rs_glb = pl.ds(r_glb, bt)

        if stream_in_hbm:
            pltpu.make_async_copy(
                stream_src_ref.at[rs_glb],
                s_in_buf.at[slot],
                s_in_sem.at[slot],
            ).wait()

        if wait_prev_out:
            @pl.when(t >= 2)
            def _drain_slot():
                prev_rs = pl.ds(pl.multiple_of(row_base + (t - 2) * bt, bt), bt)
                pltpu.make_async_copy(
                    s_out_buf.at[slot], stream_out_hbm.at[prev_rs], s_out_sem.at[slot]
                ).wait()
                pltpu.make_async_copy(
                    r_out_buf.at[slot], router_f32_hbm.at[prev_rs], r_out_sem.at[slot]
                ).wait()
                pltpu.make_async_copy(
                    h_out_buf.at[slot], hs_bf16_hbm.at[prev_rs], h_out_sem.at[slot]
                ).wait()

        if sq_vmem is not None:
            sq_l = sq_vmem[rs_loc, :]
        else:
            def _sq_half(s, acc):
                lc = pl.ds(pl.multiple_of(s * strip, strip), strip)
                x0 = ag_half_buf[rs_loc, lc].astype(jnp.float32)
                x1 = ag_mate_buf[rs_loc, lc].astype(jnp.float32)
                # Parenthesize pair sum so IEEE-754 commutativity guarantees bit-identical par=0/1.
                return acc + (_sq_strip_lanes(x0, strip) + _sq_strip_lanes(x1, strip))

            sq_l = lax.fori_loop(0, half_strips, _sq_half, jnp.zeros((bt, 128), dtype=jnp.float32))

        inv_b = jax.lax.rsqrt(jnp.sum(sq_l, axis=-1, keepdims=True) * inv_h + branch_eps)

        def _pass1_own(s, acc):
            lc = pl.ds(pl.multiple_of(s * strip, strip), strip)
            gc = pl.ds(pl.multiple_of(own_c0 + s * strip, strip), strip)
            sc = s_in_buf[slot, :, gc] if stream_in_hbm else stream_src_ref[rs_glb, gc].astype(jnp.float32)
            bc = ag_half_buf[rs_loc, lc].astype(jnp.float32)
            new_sc = alpha_ref[0:1, gc] * sc + beta_ref[0:1, gc] * (bc * inv_b)
            s_out_buf[slot, :, gc] = new_sc
            return acc + _sq_strip_lanes(new_sc, strip)

        def _pass1_mate(s, acc):
            lc = pl.ds(pl.multiple_of(s * strip, strip), strip)
            gc = pl.ds(pl.multiple_of(mate_c0 + s * strip, strip), strip)
            sc = s_in_buf[slot, :, gc] if stream_in_hbm else stream_src_ref[rs_glb, gc].astype(jnp.float32)
            bc = ag_mate_buf[rs_loc, lc].astype(jnp.float32)
            new_sc = alpha_ref[0:1, gc] * sc + beta_ref[0:1, gc] * (bc * inv_b)
            s_out_buf[slot, :, gc] = new_sc
            return acc + _sq_strip_lanes(new_sc, strip)

        next_own = lax.fori_loop(0, half_strips, _pass1_own, jnp.zeros((bt, 128), dtype=jnp.float32))
        next_mate = lax.fori_loop(0, half_strips, _pass1_mate, jnp.zeros((bt, 128), dtype=jnp.float32))
        # IEEE-754 addition is commutative -> bit-identical across par=0 and par=1!
        inv_n = jax.lax.rsqrt(
            jnp.sum(next_own + next_mate, axis=-1, keepdims=True) * inv_h + next_eps
        )

        # s_in_buf[slot] is now dead; immediately prefetch tile t + 2 overlapped with Pass 2!
        if stream_in_hbm and check_prefetch:
            @pl.when(t + 2 < n_tiles)
            def _prefetch_next():
                nxt_rs = pl.ds(pl.multiple_of(row_base + (t + 2) * bt, bt), bt)
                pltpu.make_async_copy(
                    stream_src_ref.at[nxt_rs],
                    s_in_buf.at[slot],
                    s_in_sem.at[slot],
                ).start()

        def _pass2(s, _):
            gc = pl.ds(pl.multiple_of(s * strip, strip), strip)
            rn = s_out_buf[slot, :, gc] * inv_n * next_gain_ref[0:1, gc]
            r_out_buf[slot, :, gc] = rn
            h_out_buf[slot, :, gc] = rn.astype(jnp.bfloat16)
            return jnp.int32(0)

        lax.fori_loop(0, full_strips, _pass2, jnp.int32(0))

        # Stream all 3 outputs directly to HBM asynchronously!
        pltpu.make_async_copy(
            s_out_buf.at[slot], stream_out_hbm.at[rs_glb], s_out_sem.at[slot]
        ).start()
        pltpu.make_async_copy(
            r_out_buf.at[slot], router_f32_hbm.at[rs_glb], r_out_sem.at[slot]
        ).start()
        pltpu.make_async_copy(
            h_out_buf.at[slot], hs_bf16_hbm.at[rs_glb], h_out_sem.at[slot]
        ).start()

    if n_tiles == 1:
        _run_tile(jnp.int32(0), 0, wait_prev_out=False, check_prefetch=False)
        last0 = pl.ds(row_base, bt)
        pltpu.make_async_copy(s_out_buf.at[0], stream_out_hbm.at[last0], s_out_sem.at[0]).wait()
        pltpu.make_async_copy(r_out_buf.at[0], router_f32_hbm.at[last0], r_out_sem.at[0]).wait()
        pltpu.make_async_copy(h_out_buf.at[0], hs_bf16_hbm.at[last0], h_out_sem.at[0]).wait()
        return

    if n_tiles == 2:
        _run_tile(jnp.int32(0), 0, wait_prev_out=False, check_prefetch=False)
        _run_tile(jnp.int32(1), 1, wait_prev_out=False, check_prefetch=False)
    else:
        assert nbuf == 2

        def _pair_body(p, _):
            t0 = p * 2
            _run_tile(t0, 0, wait_prev_out=True, check_prefetch=True)
            _run_tile(t0 + 1, 1, wait_prev_out=True, check_prefetch=True)
            return jnp.int32(0)

        lax.fori_loop(0, n_tiles // 2, _pair_body, jnp.int32(0))
        if n_tiles % 2 == 1:
            _run_tile(jnp.int32(n_tiles - 1), 0, wait_prev_out=True, check_prefetch=False)

    if n_tiles % 2 == 1:
        last_even = pl.ds(row_base + (n_tiles - 1) * bt, bt)
        last_odd = pl.ds(row_base + (n_tiles - 2) * bt, bt)
    else:
        last_even = pl.ds(row_base + (n_tiles - 2) * bt, bt)
        last_odd = pl.ds(row_base + (n_tiles - 1) * bt, bt)
    pltpu.make_async_copy(s_out_buf.at[0], stream_out_hbm.at[last_even], s_out_sem.at[0]).wait()
    pltpu.make_async_copy(r_out_buf.at[0], router_f32_hbm.at[last_even], r_out_sem.at[0]).wait()
    pltpu.make_async_copy(h_out_buf.at[0], hs_bf16_hbm.at[last_even], h_out_sem.at[0]).wait()
    pltpu.make_async_copy(s_out_buf.at[1], stream_out_hbm.at[last_odd], s_out_sem.at[1]).wait()
    pltpu.make_async_copy(r_out_buf.at[1], router_f32_hbm.at[last_odd], r_out_sem.at[1]).wait()
    pltpu.make_async_copy(h_out_buf.at[1], hs_bf16_hbm.at[last_odd], h_out_sem.at[1]).wait()


def _bridge_start_ici_rs_cyclic(
    my_chip,
    my_par,
    own_in_fn,
    mate_rs_buf,
    rs_send_buf,
    rs_recv_buf,
    rs_send_sem,
    rs_recv_sem,
    *,
    row_count: int,
    num_chips: int,
    chunk: int,
    mate_row_base: int = 0,
    sem_offset: int = 0,
):
    """Intra-chip sum + incast-free cyclic ICI RS sends (1-to-1 permutation per step)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    bt = 16
    n_tiles = row_count // bt
    c_strip = 256 if chunk % 256 == 0 else 128
    c_strips = chunk // c_strip
    rs_dt = rs_send_buf.dtype

    def _sum_chunk_into_send(c_off):
        def _row(t, _):
            r0 = pl.multiple_of(t * bt, bt)
            rs = pl.ds(r0, bt)
            rs_m = pl.ds(pl.multiple_of(mate_row_base + r0, bt), bt)

            def _col(s, _2):
                lc = pl.multiple_of(s * c_strip, c_strip)
                c_sl = pl.ds(pl.multiple_of(c_off + lc, c_strip), c_strip)
                csum = own_in_fn(r0, bt, c_off + lc, c_strip).astype(jnp.float32) + mate_rs_buf[
                    rs_m, c_sl
                ].astype(jnp.float32)
                rs_send_buf[rs, c_sl] = csum.astype(rs_dt)
                return jnp.int32(0)

            lax.fori_loop(0, c_strips, _col, jnp.int32(0))
            return jnp.int32(0)

        if n_tiles == 1:
            _row(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_tiles, _row, jnp.int32(0))

    # Cyclic staggered sends: step=1..num_chips-1 forms a perfect 1-to-1 matching (zero incast!).
    rs_ops = []
    for step in range(1, num_chips):
        c = jnp.mod(my_chip + step, num_chips)
        c_off = pl.multiple_of(c * chunk, chunk)
        _sum_chunk_into_send(c_off)
        dest = jnp.int32(2) * c + my_par
        op = pltpu.make_async_remote_copy(
            rs_send_buf.at[:, pl.ds(c_off, chunk)],
            rs_recv_buf.at[my_chip],
            rs_send_sem.at[sem_offset + step - 1],
            rs_recv_sem,
            device_id=(dest,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        rs_ops.append(op)

    # Overlap our own chip's intra-chip sum directly into rs_recv_buf[my_chip] while ICI RS flies!
    my_c_off = pl.multiple_of(my_chip * chunk, chunk)

    def _own_row(t, _):
        r0 = pl.multiple_of(t * bt, bt)
        rs = pl.ds(r0, bt)
        rs_m = pl.ds(pl.multiple_of(mate_row_base + r0, bt), bt)

        def _own_col(s, _2):
            lc = pl.multiple_of(s * c_strip, c_strip)
            c_sl = pl.ds(pl.multiple_of(my_c_off + lc, c_strip), c_strip)
            csum = own_in_fn(r0, bt, my_c_off + lc, c_strip).astype(jnp.float32) + mate_rs_buf[
                rs_m, c_sl
            ].astype(jnp.float32)
            rs_recv_buf[my_chip, rs, pl.ds(lc, c_strip)] = csum.astype(rs_dt)
            return jnp.int32(0)

        lax.fori_loop(0, c_strips, _own_col, jnp.int32(0))
        return jnp.int32(0)

    if n_tiles == 1:
        _own_row(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_tiles, _own_row, jnp.int32(0))

    return rs_ops


def _bridge_finish_rs_and_start_ag_cyclic(
    my_chip,
    my_par,
    rs_recv_buf,
    ag_half_buf,
    rs_recv_sem,
    ag_send_sem,
    ag_recv_sem,
    *,
    row_count: int,
    num_chips: int,
    chunk: int,
    sem_offset: int = 0,
):
    """Wait ICI RS, run canonical chip-order f32 reduction, and fire incast-free cyclic ICI AG."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    bt = 16
    n_tiles = row_count // bt
    c_strip = 256 if chunk % 256 == 0 else 128
    c_strips = chunk // c_strip
    my_c_off = pl.multiple_of(my_chip * chunk, chunk)

    pltpu.make_async_copy(
        rs_recv_buf.at[pl.ds(0, num_chips - 1)],
        rs_recv_buf.at[pl.ds(0, num_chips - 1)],
        rs_recv_sem,
    ).wait()

    # Canonical chip-order f32 reduction -> bit-identical bf16 own shard across all ranks.
    def _red_row(t, _):
        rs = pl.ds(pl.multiple_of(t * bt, bt), bt)

        def _red_col(s, _2):
            lc = pl.multiple_of(s * c_strip, c_strip)
            sl_l = pl.ds(lc, c_strip)
            sl_g = pl.ds(pl.multiple_of(my_c_off + lc, c_strip), c_strip)
            tot = rs_recv_buf[0, rs, sl_l].astype(jnp.float32)
            for src_c in range(1, num_chips):
                tot = tot + rs_recv_buf[src_c, rs, sl_l].astype(jnp.float32)
            ag_half_buf[rs, sl_g] = tot.astype(jnp.bfloat16)
            return jnp.int32(0)

        lax.fori_loop(0, c_strips, _red_col, jnp.int32(0))
        return jnp.int32(0)

    if n_tiles == 1:
        _red_row(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_tiles, _red_row, jnp.int32(0))

    ag_ops = []
    for step in range(1, num_chips):
        dest_chip = jnp.mod(my_chip + step, num_chips)
        dest = jnp.int32(2) * dest_chip + my_par
        op = pltpu.make_async_remote_copy(
            ag_half_buf.at[:, pl.ds(my_c_off, chunk)],
            ag_half_buf.at[:, pl.ds(my_c_off, chunk)],
            ag_send_sem.at[sem_offset + step - 1],
            ag_recv_sem,
            device_id=(dest,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ag_ops.append(op)
    return ag_ops


def _wait_ag_recv(ag_half_buf, ag_recv_sem, *, num_chips: int, chunk: int) -> None:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    pltpu.make_async_copy(
        ag_half_buf.at[:, pl.ds(0, (num_chips - 1) * chunk)],
        ag_half_buf.at[:, pl.ds(0, (num_chips - 1) * chunk)],
        ag_recv_sem,
    ).wait()


# =============================================================================
# 1. Standalone 2-Stage Chipmate-Bridge All-Reduce (`[B, H] bf16 -> [B, H] bf16`)
# =============================================================================
def _bridge_ar_only_kernel(
    x_ref,
    out_hbm_ref,
    mate_rs_buf,
    rs_send_buf,
    rs_recv_buf,
    ag_half_buf,
    ag_mate_buf,
    br1_send_sem,
    br1_recv_sem,
    rs_send_sem,
    rs_recv_sem,
    ag_send_sem,
    ag_recv_sem,
    br2_send_sem,
    br2_recv_sem,
    out_sem,
    *,
    world_size: int,
    chipmate_xor: int,
) -> None:
    """Single-pass XOR=1 bridge AR writing directly to HBM (own half overlapped w/ Bridge AG)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    assert chipmate_xor == 1
    tokens, hidden = x_ref.shape
    num_chips = world_size // 2
    half_h = hidden // 2
    chunk = hidden // world_size

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier(me, world_size)
    my_chip = lax.shift_right_logical(me, jnp.int32(1))
    my_par = lax.bitwise_and(me, jnp.int32(1))
    mate = lax.bitwise_xor(me, jnp.int32(1))
    own_c0 = pl.multiple_of(my_par * half_h, half_h)
    mate_c0 = pl.multiple_of((jnp.int32(1) - my_par) * half_h, half_h)

    br1_op = pltpu.make_async_remote_copy(
        x_ref.at[:, pl.ds(mate_c0, half_h)],
        mate_rs_buf,
        br1_send_sem,
        br1_recv_sem,
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_op.start()
    pltpu.make_async_copy(mate_rs_buf, mate_rs_buf, br1_recv_sem).wait()

    def _own_in(r0, bt_r: int, c_off, c_sz: int):
        return x_ref[pl.ds(r0, bt_r), pl.ds(pl.multiple_of(own_c0 + c_off, 128), c_sz)]

    rs_ops = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        _own_in,
        mate_rs_buf,
        rs_send_buf,
        rs_recv_buf,
        rs_send_sem,
        rs_recv_sem,
        row_count=tokens,
        num_chips=num_chips,
        chunk=chunk,
    )
    ag_ops = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf,
        ag_half_buf,
        rs_recv_sem,
        ag_send_sem,
        ag_recv_sem,
        row_count=tokens,
        num_chips=num_chips,
        chunk=chunk,
    )
    _wait_ag_recv(ag_half_buf, ag_recv_sem, num_chips=num_chips, chunk=chunk)

    # Fire Stage-3b Bridge AG AND overlap own-half async HBM write immediately!
    br2_op = pltpu.make_async_remote_copy(
        ag_half_buf,
        ag_mate_buf,
        br2_send_sem,
        br2_recv_sem,
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_op.start()
    out_own_cp = pltpu.make_async_copy(
        ag_half_buf, out_hbm_ref.at[:, pl.ds(own_c0, half_h)], out_sem.at[0]
    )
    out_own_cp.start()

    pltpu.make_async_copy(ag_mate_buf, ag_mate_buf, br2_recv_sem).wait()
    out_mate_cp = pltpu.make_async_copy(
        ag_mate_buf, out_hbm_ref.at[:, pl.ds(mate_c0, half_h)], out_sem.at[1]
    )
    out_mate_cp.start()

    out_own_cp.wait()
    out_mate_cp.wait()
    br2_op.wait_send()
    for op in ag_ops:
        op.wait_send()
    for op in rs_ops:
        op.wait_send()
    br1_op.wait_send()


def _bridge_ar_pipe2_kernel(
    x_ref,
    out_hbm_ref,
    mate_rs_buf,
    rs_send_buf,
    rs_recv_buf,
    ag_half_buf,
    ag_mate_buf,
    br1_send_sem,
    br1_recv_sem,
    rs_send_sem,
    rs_recv_sem,
    ag_send_sem,
    ag_recv_sem,
    br2_send_sem,
    br2_recv_sem,
    out_sem,
    *,
    world_size: int,
    chipmate_xor: int,
) -> None:
    """2-row-half pipelined XOR=1 bridge AR overlapping H1 Bridge RS & H0 HBM write under ICI."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    assert chipmate_xor == 1
    tokens, hidden = x_ref.shape
    hb = tokens // 2
    num_chips = world_size // 2
    half_h = hidden // 2
    chunk = hidden // world_size

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier(me, world_size)
    my_chip = lax.shift_right_logical(me, jnp.int32(1))
    my_par = lax.bitwise_and(me, jnp.int32(1))
    mate = lax.bitwise_xor(me, jnp.int32(1))
    own_c0 = pl.multiple_of(my_par * half_h, half_h)
    mate_c0 = pl.multiple_of((jnp.int32(1) - my_par) * half_h, half_h)

    r_h0 = pl.ds(0, hb)
    r_h1 = pl.ds(hb, hb)

    # 1. Start H0 Bridge RS first (full 733 GB/s bridge to H0).
    br1_0 = pltpu.make_async_remote_copy(
        x_ref.at[r_h0, pl.ds(mate_c0, half_h)],
        mate_rs_buf.at[0],
        br1_send_sem.at[0],
        br1_recv_sem.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_0.start()
    pltpu.make_async_copy(mate_rs_buf.at[0], mate_rs_buf.at[0], br1_recv_sem.at[0]).wait()

    # 2. Immediately fire H1 Bridge RS overlapped under H0 ICI RS + AG!
    br1_1 = pltpu.make_async_remote_copy(
        x_ref.at[r_h1, pl.ds(mate_c0, half_h)],
        mate_rs_buf.at[1],
        br1_send_sem.at[1],
        br1_recv_sem.at[1],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_1.start()

    # 3. H0 cyclic ICI RS + AG.
    rs_ops_0 = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        lambda r0, bt_r, c_off, c_sz: x_ref[
            pl.ds(r0, bt_r), pl.ds(pl.multiple_of(own_c0 + c_off, 128), c_sz)
        ],
        mate_rs_buf.at[0],
        rs_send_buf.at[0],
        rs_recv_buf.at[0],
        rs_send_sem,
        rs_recv_sem.at[0],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=0,
    )
    ag_ops_0 = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf.at[0],
        ag_half_buf.at[0],
        rs_recv_sem.at[0],
        ag_send_sem,
        ag_recv_sem.at[0],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=0,
    )
    _wait_ag_recv(ag_half_buf.at[0], ag_recv_sem.at[0], num_chips=num_chips, chunk=chunk)

    # 4. Fire H0 Bridge AG + H0 own-half HBM write, then immediately start H1 ICI RS!
    br2_0 = pltpu.make_async_remote_copy(
        ag_half_buf.at[0],
        ag_mate_buf.at[0],
        br2_send_sem.at[0],
        br2_recv_sem.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_0.start()
    out_0_own = pltpu.make_async_copy(
        ag_half_buf.at[0], out_hbm_ref.at[r_h0, pl.ds(own_c0, half_h)], out_sem.at[0]
    )
    out_0_own.start()

    pltpu.make_async_copy(mate_rs_buf.at[1], mate_rs_buf.at[1], br1_recv_sem.at[1]).wait()
    rs_ops_1 = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        lambda r0, bt_r, c_off, c_sz: x_ref[
            pl.ds(pl.multiple_of(hb + r0, 16), bt_r),
            pl.ds(pl.multiple_of(own_c0 + c_off, 128), c_sz),
        ],
        mate_rs_buf.at[1],
        rs_send_buf.at[1],
        rs_recv_buf.at[1],
        rs_send_sem,
        rs_recv_sem.at[1],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=num_chips - 1,
    )

    # 5. Overlap H0 mate HBM write under H1 ICI RS!
    pltpu.make_async_copy(ag_mate_buf.at[0], ag_mate_buf.at[0], br2_recv_sem.at[0]).wait()
    out_0_mate = pltpu.make_async_copy(
        ag_mate_buf.at[0], out_hbm_ref.at[r_h0, pl.ds(mate_c0, half_h)], out_sem.at[1]
    )
    out_0_mate.start()

    # 6. Finish H1 ICI RS + AG + Bridge AG + HBM writes.
    ag_ops_1 = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf.at[1],
        ag_half_buf.at[1],
        rs_recv_sem.at[1],
        ag_send_sem,
        ag_recv_sem.at[1],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=num_chips - 1,
    )
    _wait_ag_recv(ag_half_buf.at[1], ag_recv_sem.at[1], num_chips=num_chips, chunk=chunk)

    br2_1 = pltpu.make_async_remote_copy(
        ag_half_buf.at[1],
        ag_mate_buf.at[1],
        br2_send_sem.at[1],
        br2_recv_sem.at[1],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_1.start()
    out_1_own = pltpu.make_async_copy(
        ag_half_buf.at[1], out_hbm_ref.at[r_h1, pl.ds(own_c0, half_h)], out_sem.at[2]
    )
    out_1_own.start()

    pltpu.make_async_copy(ag_mate_buf.at[1], ag_mate_buf.at[1], br2_recv_sem.at[1]).wait()
    out_1_mate = pltpu.make_async_copy(
        ag_mate_buf.at[1], out_hbm_ref.at[r_h1, pl.ds(mate_c0, half_h)], out_sem.at[3]
    )
    out_1_mate.start()

    out_0_own.wait()
    out_0_mate.wait()
    out_1_own.wait()
    out_1_mate.wait()
    br2_0.wait_send()
    br2_1.wait_send()
    for op in ag_ops_0:
        op.wait_send()
    for op in ag_ops_1:
        op.wait_send()
    for op in rs_ops_0:
        op.wait_send()
    for op in rs_ops_1:
        op.wait_send()
    br1_0.wait_send()
    br1_1.wait_send()


def bridge_all_reduce_jax(
    x,
    *,
    world_size: int,
    rs_wire_dtype: str = "bfloat16",
    pipe2: bool = False,
    chipmate_xor: int = TPU7X_CHIPMATE_XOR,
    collective_id: int = _DEFAULT_COLLECTIVE_ID,
):
    """Sum ``x [B, H]`` bf16 over ``model`` via 2-stage ``XOR=1`` chipmate bridge RS + AG."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = x.shape
    if world_size % 2 != 0 or world_size < 4:
        raise ValueError(f"bridge_all_reduce requires even world_size >= 4, got {world_size}")
    if tokens < 16 or tokens % 16 != 0 or hidden % (128 * world_size) != 0:
        raise ValueError(
            f"bridge_all_reduce requires B % 16 == 0 and H % (128*P) == 0, got {(tokens, hidden)}"
        )
    use_pipe2 = pipe2 and tokens >= 32 and (tokens % 32 == 0)
    num_chips = world_size // 2
    half_h = hidden // 2
    chunk = hidden // world_size
    rs_dt = jnp.float32 if rs_wire_dtype == "float32" else jnp.bfloat16
    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA

    if use_pipe2:
        hb = tokens // 2
        return pl.pallas_call(
            functools.partial(
                _bridge_ar_pipe2_kernel,
                world_size=world_size,
                chipmate_xor=chipmate_xor,
            ),
            out_shape=jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            in_specs=[pl.BlockSpec(memory_space=vmem)],
            out_specs=pl.BlockSpec(memory_space=hbm),
            scratch_shapes=[
                vmem((2, hb, half_h), jnp.bfloat16),
                vmem((2, hb, half_h), rs_dt),
                vmem((2, num_chips, hb, chunk), rs_dt),
                vmem((2, hb, half_h), jnp.bfloat16),
                vmem((2, hb, half_h), jnp.bfloat16),
                sem((2,)),
                sem((2,)),
                sem((2 * (num_chips - 1),)),
                sem((2,)),
                sem((2 * (num_chips - 1),)),
                sem((2,)),
                sem((2,)),
                sem((2,)),
                sem((4,)),
            ],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=(56 if tokens > 256 else 48) * 1024**2,
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            name=f"muse_bridge_ar_p2_b{tokens}_tp{world_size}_{'f32rs' if rs_wire_dtype == 'float32' else 'bf16rs'}",
        )(x.astype(jnp.bfloat16))

    return pl.pallas_call(
        functools.partial(
            _bridge_ar_only_kernel,
            world_size=world_size,
            chipmate_xor=chipmate_xor,
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        in_specs=[pl.BlockSpec(memory_space=vmem)],
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=[
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), rs_dt),
            vmem((num_chips, tokens, chunk), rs_dt),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            sem,
            sem,
            sem((num_chips - 1,)),
            sem,
            sem((num_chips - 1,)),
            sem,
            sem,
            sem,
            sem((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(56 if tokens > 256 else 48) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_bridge_ar_b{tokens}_tp{world_size}_{'f32rs' if rs_wire_dtype == 'float32' else 'bf16rs'}",
    )(x.astype(jnp.bfloat16))


# =============================================================================
# 2. Fused `o_proj` AR + `post_attn_norm` + `gated_res` + `pre_ffn_norm` (`mode=1`)
# =============================================================================
def _flat_ar_res_norm_kernel(
    stream_src_ref,
    o_part_ref,
    alpha_ref,
    beta_ref,
    next_gain_ref,
    stream_out_hbm,
    router_f32_hbm,
    hs_bf16_hbm,
    rs_recv_buf,
    own_buf,
    ag_buf,
    s_in_buf,
    s_out_buf,
    r_out_buf,
    h_out_buf,
    rs_send_sem,
    rs_recv_sem,
    ag_send_sem,
    ag_recv_sem,
    s_in_sem,
    s_out_sem,
    r_out_sem,
    h_out_sem,
    *,
    world_size: int,
    branch_eps: float,
    next_eps: float,
    stream_in_hbm: bool,
    nbuf: int,
) -> None:
    """1-hop cyclic RS + AG fused with zero-spill HBM-pipelined dual-output res+norm."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = o_part_ref.shape
    half_h = hidden // 2
    chunk = hidden // world_size
    bt = 16
    n_tiles = tokens // bt
    c_strip = 256 if chunk % 256 == 0 else 128
    c_strips = chunk // c_strip

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier(me, world_size)

    if stream_in_hbm:
        _prefetch_stream_initial(
            stream_src_ref, s_in_buf, s_in_sem, row_base=0, n_tiles=n_tiles, nbuf=nbuf
        )

    # Incast-free cyclic 1-hop RS.
    rs_ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        col = pl.ds(pl.multiple_of(peer * chunk, chunk), chunk)
        op = pltpu.make_async_remote_copy(
            o_part_ref.at[:, col],
            rs_recv_buf.at[me],
            rs_send_sem.at[step - 1],
            rs_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        rs_ops.append(op)

    me_c0 = pl.multiple_of(me * chunk, chunk)

    def _copy_own(t, _):
        rs = pl.ds(pl.multiple_of(t * bt, bt), bt)

        def _col(s, _2):
            lc = pl.multiple_of(s * c_strip, c_strip)
            rs_recv_buf[me, rs, pl.ds(lc, c_strip)] = o_part_ref[
                rs, pl.ds(pl.multiple_of(me_c0 + lc, c_strip), c_strip)
            ]
            return jnp.int32(0)

        lax.fori_loop(0, c_strips, _col, jnp.int32(0))
        return jnp.int32(0)

    if n_tiles == 1:
        _copy_own(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_tiles, _copy_own, jnp.int32(0))

    pltpu.make_async_copy(
        rs_recv_buf.at[pl.ds(0, world_size - 1)],
        rs_recv_buf.at[pl.ds(0, world_size - 1)],
        rs_recv_sem,
    ).wait()

    def _red_flat(t, _):
        rs = pl.ds(pl.multiple_of(t * bt, bt), bt)

        def _col(s, _2):
            lc = pl.multiple_of(s * c_strip, c_strip)
            sl = pl.ds(lc, c_strip)
            tot = rs_recv_buf[0, rs, sl].astype(jnp.float32)
            for p in range(1, world_size):
                tot = tot + rs_recv_buf[p, rs, sl].astype(jnp.float32)
            own_b = tot.astype(jnp.bfloat16)
            own_buf[rs, sl] = own_b
            ag_buf[rs, pl.ds(pl.multiple_of(me_c0 + lc, c_strip), c_strip)] = own_b
            return jnp.int32(0)

        lax.fori_loop(0, c_strips, _col, jnp.int32(0))
        return jnp.int32(0)

    if n_tiles == 1:
        _red_flat(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_tiles, _red_flat, jnp.int32(0))

    # Incast-free cyclic 1-hop AG.
    ag_ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        op = pltpu.make_async_remote_copy(
            own_buf,
            ag_buf.at[:, pl.ds(me_c0, chunk)],
            ag_send_sem.at[step - 1],
            ag_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        ag_ops.append(op)

    pltpu.make_async_copy(
        ag_buf.at[:, pl.ds(0, (world_size - 1) * chunk)],
        ag_buf.at[:, pl.ds(0, (world_size - 1) * chunk)],
        ag_recv_sem,
    ).wait()

    _fused_dual_res_norm_hbm_pipelined(
        ag_buf.at[:, pl.ds(0, half_h)],
        ag_buf.at[:, pl.ds(half_h, half_h)],
        jnp.int32(0),
        jnp.int32(half_h),
        None,
        stream_src_ref,
        alpha_ref,
        beta_ref,
        next_gain_ref,
        stream_out_hbm,
        router_f32_hbm,
        hs_bf16_hbm,
        s_in_buf,
        s_out_buf,
        r_out_buf,
        h_out_buf,
        s_in_sem,
        s_out_sem,
        r_out_sem,
        h_out_sem,
        row_base=0,
        row_count=tokens,
        hidden=hidden,
        branch_eps=branch_eps,
        next_eps=next_eps,
        stream_in_hbm=stream_in_hbm,
        stream_already_prefetched=stream_in_hbm,
        nbuf=nbuf,
    )

    for op in ag_ops:
        op.wait_send()
    for op in rs_ops:
        op.wait_send()


def _bridge_ar_res_norm_kernel(
    stream_src_ref,
    o_part_ref,
    alpha_ref,
    beta_ref,
    next_gain_ref,
    stream_out_hbm,
    router_f32_hbm,
    hs_bf16_hbm,
    mate_rs_buf,
    rs_send_buf,
    rs_recv_buf,
    ag_half_buf,
    ag_mate_buf,
    sq_vmem,
    s_in_buf,
    s_out_buf,
    r_out_buf,
    h_out_buf,
    br1_send_sem,
    br1_recv_sem,
    rs_send_sem,
    rs_recv_sem,
    ag_send_sem,
    ag_recv_sem,
    br2_send_sem,
    br2_recv_sem,
    s_in_sem,
    s_out_sem,
    r_out_sem,
    h_out_sem,
    *,
    world_size: int,
    branch_eps: float,
    next_eps: float,
    stream_in_hbm: bool,
    chipmate_xor: int,
    nbuf: int,
) -> None:
    """Single-pass XOR=1 bridge RS + AG fused with zero-spill HBM-pipelined dual res+norm."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    assert chipmate_xor == 1
    tokens, hidden = o_part_ref.shape
    num_chips = world_size // 2
    half_h = hidden // 2
    chunk = hidden // world_size
    n_tiles = tokens // 16

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier(me, world_size)
    my_chip = lax.shift_right_logical(me, jnp.int32(1))
    my_par = lax.bitwise_and(me, jnp.int32(1))
    mate = lax.bitwise_xor(me, jnp.int32(1))
    own_c0 = pl.multiple_of(my_par * half_h, half_h)
    mate_c0 = pl.multiple_of((jnp.int32(1) - my_par) * half_h, half_h)

    br1_op = pltpu.make_async_remote_copy(
        o_part_ref.at[:, pl.ds(mate_c0, half_h)],
        mate_rs_buf,
        br1_send_sem,
        br1_recv_sem,
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_op.start()
    if stream_in_hbm:
        _prefetch_stream_initial(
            stream_src_ref, s_in_buf, s_in_sem, row_base=0, n_tiles=n_tiles, nbuf=nbuf
        )
    pltpu.make_async_copy(mate_rs_buf, mate_rs_buf, br1_recv_sem).wait()

    def _own_in(r0, bt_r: int, c_off, c_sz: int):
        return o_part_ref[pl.ds(r0, bt_r), pl.ds(pl.multiple_of(own_c0 + c_off, 128), c_sz)]

    rs_ops = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        _own_in,
        mate_rs_buf,
        rs_send_buf,
        rs_recv_buf,
        rs_send_sem,
        rs_recv_sem,
        row_count=tokens,
        num_chips=num_chips,
        chunk=chunk,
    )
    ag_ops = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf,
        ag_half_buf,
        rs_recv_sem,
        ag_send_sem,
        ag_recv_sem,
        row_count=tokens,
        num_chips=num_chips,
        chunk=chunk,
    )
    _wait_ag_recv(ag_half_buf, ag_recv_sem, num_chips=num_chips, chunk=chunk)

    # Overlap own-half sq_sum in VMEM scratch under Stage-3b Bridge AG!
    br2_op = pltpu.make_async_remote_copy(
        ag_half_buf,
        ag_mate_buf,
        br2_send_sem,
        br2_recv_sem,
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_op.start()
    _accum_half_sq_vmem(ag_half_buf, sq_vmem, row_count=tokens, half_h=half_h, add_to_existing=False)
    pltpu.make_async_copy(ag_mate_buf, ag_mate_buf, br2_recv_sem).wait()
    _accum_half_sq_vmem(ag_mate_buf, sq_vmem, row_count=tokens, half_h=half_h, add_to_existing=True)

    _fused_dual_res_norm_hbm_pipelined(
        ag_half_buf,
        ag_mate_buf,
        own_c0,
        mate_c0,
        sq_vmem,
        stream_src_ref,
        alpha_ref,
        beta_ref,
        next_gain_ref,
        stream_out_hbm,
        router_f32_hbm,
        hs_bf16_hbm,
        s_in_buf,
        s_out_buf,
        r_out_buf,
        h_out_buf,
        s_in_sem,
        s_out_sem,
        r_out_sem,
        h_out_sem,
        row_base=0,
        row_count=tokens,
        hidden=hidden,
        branch_eps=branch_eps,
        next_eps=next_eps,
        stream_in_hbm=stream_in_hbm,
        stream_already_prefetched=stream_in_hbm,
        nbuf=nbuf,
    )

    br2_op.wait_send()
    for op in ag_ops:
        op.wait_send()
    for op in rs_ops:
        op.wait_send()
    br1_op.wait_send()


def _bridge_ar_res_norm_pipe2_kernel(
    stream_src_ref,
    o_part_ref,
    alpha_ref,
    beta_ref,
    next_gain_ref,
    stream_out_hbm,
    router_f32_hbm,
    hs_bf16_hbm,
    mate_rs_buf,
    rs_send_buf,
    rs_recv_buf,
    ag_half_buf,
    ag_mate_buf,
    sq_vmem,
    s_in_buf,
    s_out_buf,
    r_out_buf,
    h_out_buf,
    br1_send_sem,
    br1_recv_sem,
    rs_send_sem,
    rs_recv_sem,
    ag_send_sem,
    ag_recv_sem,
    br2_send_sem,
    br2_recv_sem,
    s_in_sem,
    s_out_sem,
    r_out_sem,
    h_out_sem,
    *,
    world_size: int,
    branch_eps: float,
    next_eps: float,
    stream_in_hbm: bool,
    chipmate_xor: int,
    nbuf: int,
) -> None:
    """2-row-half pipelined XOR=1 bridge AR + zero-spill HBM epilogue.

    Hides H1 Bridge RS under H0 ICI RS+AG, AND hides H0 Bridge AG + H0 complete res_norm +
    H0 3x HBM writes under H1 ICI RS!
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    assert chipmate_xor == 1
    tokens, hidden = o_part_ref.shape
    hb = tokens // 2
    hb_tiles = hb // 16
    num_chips = world_size // 2
    half_h = hidden // 2
    chunk = hidden // world_size

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier(me, world_size)
    my_chip = lax.shift_right_logical(me, jnp.int32(1))
    my_par = lax.bitwise_and(me, jnp.int32(1))
    mate = lax.bitwise_xor(me, jnp.int32(1))
    own_c0 = pl.multiple_of(my_par * half_h, half_h)
    mate_c0 = pl.multiple_of((jnp.int32(1) - my_par) * half_h, half_h)

    r_h0 = pl.ds(0, hb)
    r_h1 = pl.ds(hb, hb)

    # 1. Start H0 Bridge RS + prefetch H0 initial stream tiles!
    br1_0 = pltpu.make_async_remote_copy(
        o_part_ref.at[r_h0, pl.ds(mate_c0, half_h)],
        mate_rs_buf.at[0],
        br1_send_sem.at[0],
        br1_recv_sem.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_0.start()
    if stream_in_hbm:
        _prefetch_stream_initial(
            stream_src_ref, s_in_buf, s_in_sem, row_base=0, n_tiles=hb_tiles, nbuf=nbuf
        )
    pltpu.make_async_copy(mate_rs_buf.at[0], mate_rs_buf.at[0], br1_recv_sem.at[0]).wait()

    # 2. Fire H1 Bridge RS overlapped under H0 ICI RS + AG!
    br1_1 = pltpu.make_async_remote_copy(
        o_part_ref.at[r_h1, pl.ds(mate_c0, half_h)],
        mate_rs_buf.at[1],
        br1_send_sem.at[1],
        br1_recv_sem.at[1],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_1.start()

    # 3. H0 cyclic ICI RS + AG.
    rs_ops_0 = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        lambda r0, bt_r, c_off, c_sz: o_part_ref[
            pl.ds(r0, bt_r), pl.ds(pl.multiple_of(own_c0 + c_off, 128), c_sz)
        ],
        mate_rs_buf.at[0],
        rs_send_buf.at[0],
        rs_recv_buf.at[0],
        rs_send_sem,
        rs_recv_sem.at[0],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=0,
    )
    ag_ops_0 = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf.at[0],
        ag_half_buf.at[0],
        rs_recv_sem.at[0],
        ag_send_sem,
        ag_recv_sem.at[0],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=0,
    )
    _wait_ag_recv(ag_half_buf.at[0], ag_recv_sem.at[0], num_chips=num_chips, chunk=chunk)

    # 4. Fire H0 Bridge AG, wait H1 Bridge RS (already done!), and fire H1 cyclic ICI RS!
    br2_0 = pltpu.make_async_remote_copy(
        ag_half_buf.at[0],
        ag_mate_buf.at[0],
        br2_send_sem.at[0],
        br2_recv_sem.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_0.start()

    pltpu.make_async_copy(mate_rs_buf.at[1], mate_rs_buf.at[1], br1_recv_sem.at[1]).wait()
    rs_ops_1 = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        lambda r0, bt_r, c_off, c_sz: o_part_ref[
            pl.ds(pl.multiple_of(hb + r0, 16), bt_r),
            pl.ds(pl.multiple_of(own_c0 + c_off, 128), c_sz),
        ],
        mate_rs_buf.at[1],
        rs_send_buf.at[1],
        rs_recv_buf.at[1],
        rs_send_sem,
        rs_recv_sem.at[1],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=num_chips - 1,
    )

    # 5. WHILE H1 ICI RS IS IN FLIGHT: run H0 COMPLETE epilogue (res_norm + 3x HBM writes)!
    pltpu.make_async_copy(ag_mate_buf.at[0], ag_mate_buf.at[0], br2_recv_sem.at[0]).wait()
    _fused_dual_res_norm_hbm_pipelined(
        ag_half_buf.at[0],
        ag_mate_buf.at[0],
        own_c0,
        mate_c0,
        None,
        stream_src_ref,
        alpha_ref,
        beta_ref,
        next_gain_ref,
        stream_out_hbm,
        router_f32_hbm,
        hs_bf16_hbm,
        s_in_buf,
        s_out_buf,
        r_out_buf,
        h_out_buf,
        s_in_sem,
        s_out_sem,
        r_out_sem,
        h_out_sem,
        row_base=0,
        row_count=hb,
        hidden=hidden,
        branch_eps=branch_eps,
        next_eps=next_eps,
        stream_in_hbm=stream_in_hbm,
        stream_already_prefetched=stream_in_hbm,
        nbuf=nbuf,
    )

    # 6. Finish H1 ICI RS + AG + Bridge AG (overlap H1 stream prefetch & H1 sq_own!).
    if stream_in_hbm:
        _prefetch_stream_initial(
            stream_src_ref, s_in_buf, s_in_sem, row_base=hb, n_tiles=hb_tiles, nbuf=nbuf
        )
    ag_ops_1 = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf.at[1],
        ag_half_buf.at[1],
        rs_recv_sem.at[1],
        ag_send_sem,
        ag_recv_sem.at[1],
        row_count=hb,
        num_chips=num_chips,
        chunk=chunk,
        sem_offset=num_chips - 1,
    )
    _wait_ag_recv(ag_half_buf.at[1], ag_recv_sem.at[1], num_chips=num_chips, chunk=chunk)

    br2_1 = pltpu.make_async_remote_copy(
        ag_half_buf.at[1],
        ag_mate_buf.at[1],
        br2_send_sem.at[1],
        br2_recv_sem.at[1],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_1.start()
    _accum_half_sq_vmem(ag_half_buf.at[1], sq_vmem, row_count=hb, half_h=half_h, add_to_existing=False)
    pltpu.make_async_copy(ag_mate_buf.at[1], ag_mate_buf.at[1], br2_recv_sem.at[1]).wait()
    _accum_half_sq_vmem(ag_mate_buf.at[1], sq_vmem, row_count=hb, half_h=half_h, add_to_existing=True)

    _fused_dual_res_norm_hbm_pipelined(
        ag_half_buf.at[1],
        ag_mate_buf.at[1],
        own_c0,
        mate_c0,
        sq_vmem,
        stream_src_ref,
        alpha_ref,
        beta_ref,
        next_gain_ref,
        stream_out_hbm,
        router_f32_hbm,
        hs_bf16_hbm,
        s_in_buf,
        s_out_buf,
        r_out_buf,
        h_out_buf,
        s_in_sem,
        s_out_sem,
        r_out_sem,
        h_out_sem,
        row_base=hb,
        row_count=hb,
        hidden=hidden,
        branch_eps=branch_eps,
        next_eps=next_eps,
        stream_in_hbm=stream_in_hbm,
        stream_already_prefetched=stream_in_hbm,
        nbuf=nbuf,
    )

    br2_0.wait_send()
    br2_1.wait_send()
    for op in ag_ops_0:
        op.wait_send()
    for op in ag_ops_1:
        op.wait_send()
    for op in rs_ops_0:
        op.wait_send()
    for op in rs_ops_1:
        op.wait_send()
    br1_0.wait_send()
    br1_1.wait_send()


def tp_ar_res_norm_jax(
    stream,
    o_partial,
    alpha,
    beta,
    next_gain,
    *,
    world_size: int,
    branch_eps: float,
    next_eps: float,
    mode: str = "auto",
    rs_wire_dtype: str = "bfloat16",
    stream_in_hbm: bool = True,
    chipmate_xor: int = TPU7X_CHIPMATE_XOR,
    collective_id: int = _DEFAULT_COLLECTIVE_ID,
):
    """Fused ``o_proj`` AR + ``post_attn_norm`` + ``gated_res`` + ``pre_ffn_norm`` -> ``(f32, f32, bf16)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = o_partial.shape
    if stream.shape != (tokens, hidden):
        raise ValueError(f"stream shape {stream.shape} != o_partial {o_partial.shape}")
    if tokens < 16 or tokens % 16 != 0 or tokens > 384:
        raise ValueError(f"Expected 16 <= tokens <= 384 and tokens % 16 == 0, got {tokens}")
    if hidden % (128 * world_size) != 0:
        raise ValueError(f"Expected hidden={hidden} divisible by 128 * world_size={128 * world_size}")

    eff_mode = mode
    if eff_mode == "auto":
        if chipmate_xor != 1 or world_size % 2 != 0 or world_size < 4 or tokens <= 16:
            eff_mode = "flat"
        elif tokens >= 128 and tokens % 32 == 0:
            eff_mode = "bridge_pipe2"
        else:
            eff_mode = "bridge"
    if eff_mode == "bridge_pipe2" and (tokens < 32 or tokens % 32 != 0):
        eff_mode = "bridge"

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=hbm)
    s_in_spec = h_spec if stream_in_hbm else v_spec
    chunk = hidden // world_size
    half_h = hidden // 2

    out_shapes = (
        jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
        jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
    )
    in_args = (
        stream.astype(jnp.float32),
        o_partial.astype(jnp.bfloat16),
        alpha.reshape(1, hidden).astype(jnp.float32),
        beta.reshape(1, hidden).astype(jnp.float32),
        next_gain.reshape(1, hidden).astype(jnp.float32),
    )

    if eff_mode == "flat":
        nbuf = 1 if tokens == 16 else 2
        return pl.pallas_call(
            functools.partial(
                _flat_ar_res_norm_kernel,
                world_size=world_size,
                branch_eps=branch_eps,
                next_eps=next_eps,
                stream_in_hbm=stream_in_hbm,
                nbuf=nbuf,
            ),
            out_shape=out_shapes,
            in_specs=[s_in_spec, v_spec, v_spec, v_spec, v_spec],
            out_specs=(h_spec, h_spec, h_spec),
            scratch_shapes=[
                vmem((world_size, tokens, chunk), jnp.bfloat16),
                vmem((tokens, chunk), jnp.bfloat16),
                vmem((tokens, hidden), jnp.bfloat16),
                vmem((nbuf, 16, hidden), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.bfloat16),
                sem((world_size - 1,)),
                sem,
                sem((world_size - 1,)),
                sem,
                sem((nbuf,)),
                sem((nbuf,)),
                sem((nbuf,)),
                sem((nbuf,)),
            ],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=(56 if tokens > 256 else 48) * 1024**2,
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            name=f"muse_tpar_flat_res_norm_b{tokens}_tp{world_size}",
        )(*in_args)

    num_chips = world_size // 2
    rs_dt = jnp.float32 if rs_wire_dtype == "float32" else jnp.bfloat16

    if eff_mode == "bridge_pipe2":
        hb = tokens // 2
        nbuf = 1 if hb == 16 else 2
        return pl.pallas_call(
            functools.partial(
                _bridge_ar_res_norm_pipe2_kernel,
                world_size=world_size,
                branch_eps=branch_eps,
                next_eps=next_eps,
                stream_in_hbm=stream_in_hbm,
                chipmate_xor=chipmate_xor,
                nbuf=nbuf,
            ),
            out_shape=out_shapes,
            in_specs=[s_in_spec, v_spec, v_spec, v_spec, v_spec],
            out_specs=(h_spec, h_spec, h_spec),
            scratch_shapes=[
                vmem((2, hb, half_h), jnp.bfloat16),
                vmem((2, hb, half_h), rs_dt),
                vmem((2, num_chips, hb, chunk), rs_dt),
                vmem((2, hb, half_h), jnp.bfloat16),
                vmem((2, hb, half_h), jnp.bfloat16),
                vmem((hb, 128), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.float32),
                vmem((nbuf, 16, hidden), jnp.bfloat16),
                sem((2,)),
                sem((2,)),
                sem((2 * (num_chips - 1),)),
                sem((2,)),
                sem((2 * (num_chips - 1),)),
                sem((2,)),
                sem((2,)),
                sem((2,)),
                sem((nbuf,)),
                sem((nbuf,)),
                sem((nbuf,)),
                sem((nbuf,)),
            ],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=(56 if tokens > 256 else 48) * 1024**2,
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            name=(
                f"muse_tpar_bridge_p2_res_norm_b{tokens}_tp{world_size}"
                f"_{'f32rs' if rs_wire_dtype == 'float32' else 'bf16rs'}"
                f"{'_shbm' if stream_in_hbm else '_svmem'}"
            ),
        )(*in_args)

    if eff_mode != "bridge":
        raise ValueError(f"Unsupported tp_ar_res_norm mode: {mode!r}")

    nbuf = 1 if tokens == 16 else 2
    return pl.pallas_call(
        functools.partial(
            _bridge_ar_res_norm_kernel,
            world_size=world_size,
            branch_eps=branch_eps,
            next_eps=next_eps,
            stream_in_hbm=stream_in_hbm,
            chipmate_xor=chipmate_xor,
            nbuf=nbuf,
        ),
        out_shape=out_shapes,
        in_specs=[s_in_spec, v_spec, v_spec, v_spec, v_spec],
        out_specs=(h_spec, h_spec, h_spec),
        scratch_shapes=[
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), rs_dt),
            vmem((num_chips, tokens, chunk), rs_dt),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, 128), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.bfloat16),
            sem,
            sem,
            sem((num_chips - 1,)),
            sem,
            sem((num_chips - 1,)),
            sem,
            sem,
            sem,
            sem((nbuf,)),
            sem((nbuf,)),
            sem((nbuf,)),
            sem((nbuf,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(56 if tokens > 256 else 48) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_tpar_bridge_res_norm_b{tokens}_tp{world_size}"
            f"_{'f32rs' if rs_wire_dtype == 'float32' else 'bf16rs'}"
            f"{'_shbm' if stream_in_hbm else '_svmem'}"
        ),
    )(*in_args)


# =============================================================================
# 3. Full 8-Op Attention Tail Fusion (`head_norm + gate + o_proj + bridge_ar + res_norm`)
# =============================================================================
def _tail_head_norm_gate(
    attn_out_ref,
    gate_ref,
    gated_buf,
    *,
    tokens: int,
    k_local: int,
    head_dim: int,
    head_norm_eps: float,
) -> None:
    """Zero-spill fused per-head RMSNorm + ``sigmoid(gate) * norm_attn`` (<= 16 VREGs)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    bt = 16
    n_tiles = tokens // bt
    n_heads = k_local // head_dim
    h_blk = 4 if n_heads % 4 == 0 else (2 if n_heads % 2 == 0 else 1)
    c_blk = h_blk * head_dim
    n_hblks = k_local // c_blk

    def _row_body(t, _):
        rs = pl.ds(pl.multiple_of(t * bt, bt), bt)

        def _hb(h, _2):
            hc = pl.ds(pl.multiple_of(h * c_blk, c_blk), c_blk)
            ah = attn_out_ref[rs, hc].reshape(bt * h_blk, head_dim).astype(jnp.float32)
            inv_ah = jax.lax.rsqrt(jnp.mean(ah * ah, axis=-1, keepdims=True) + head_norm_eps)
            an = (ah * inv_ah).astype(jnp.bfloat16).reshape(bt, c_blk)
            gs = jax.nn.sigmoid(gate_ref[rs, hc].astype(jnp.float32)).astype(jnp.bfloat16)
            gated_buf[rs, hc] = gs * an
            return jnp.int32(0)

        if n_hblks == 1:
            _hb(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_hblks, _hb, jnp.int32(0))
        return jnp.int32(0)

    if n_tiles == 1:
        _row_body(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_tiles, _row_body, jnp.int32(0))


def _tail_dot_strip(
    gated_buf,
    wo_buf,
    wo_slot: int,
    wo_scale_ref,
    dst_buf,
    base_c,
    s_idx: int,
    *,
    tokens: int,
    k_local: int,
    tn: int,
    has_wo_scale: bool,
) -> None:
    """Zero-spill full-K MXU ``gated_buf [B, K] @ wo_buf[wo_slot, K, tn]`` (<= 16 VREGs)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    del k_local
    bm = min(tokens, 32)
    n_m = tokens // bm
    l_col = pl.ds(s_idx * tn, tn)

    def _m_body(im, _):
        r0 = pl.multiple_of(im * bm, bm)
        rs = pl.ds(r0, bm)
        acc = jnp.dot(
            gated_buf[rs, :],
            wo_buf[wo_slot, :, :],
            preferred_element_type=jnp.float32,
        )
        if has_wo_scale:
            g_col = pl.ds(pl.multiple_of(base_c + s_idx * tn, tn), tn)
            acc = acc * wo_scale_ref[0:1, g_col]
        dst_buf[rs, l_col] = acc.astype(jnp.bfloat16)
        return jnp.int32(0)

    if n_m == 1:
        _m_body(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_m, _m_body, jnp.int32(0))


def _attn_tail_bridge_ar_res_norm_kernel(
    *refs,
    world_size: int,
    head_dim: int,
    head_norm_eps: float,
    branch_eps: float,
    next_eps: float,
    has_wo_scale: bool,
    chipmate_xor: int,
    use_pipe2: bool,
    nbuf: int,
) -> None:
    """Zero-spill 8-op fused attention tail: ``head_norm + gate + o_proj + AR + res_norm``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    assert chipmate_xor == 1
    idx = 0
    stream_hbm_ref = refs[idx]; idx += 1
    attn_out_ref = refs[idx]; idx += 1
    gate_ref = refs[idx]; idx += 1
    wo_hbm_ref = refs[idx]; idx += 1
    wo_scale_ref = refs[idx] if has_wo_scale else None
    if has_wo_scale:
        idx += 1
    alpha_ref = refs[idx]; idx += 1
    beta_ref = refs[idx]; idx += 1
    next_gain_ref = refs[idx]; idx += 1

    stream_out_hbm = refs[idx]; idx += 1
    router_f32_hbm = refs[idx]; idx += 1
    hs_bf16_hbm = refs[idx]; idx += 1

    gated_buf = refs[idx]; idx += 1
    wo_buf = refs[idx]; idx += 1
    wo_sem = refs[idx]; idx += 1
    own_part_buf = refs[idx]; idx += 1
    mate_send_buf = refs[idx]; idx += 1
    mate_rs_buf = refs[idx]; idx += 1
    rs_send_buf = refs[idx]; idx += 1
    rs_recv_buf = refs[idx]; idx += 1
    ag_half_buf = refs[idx]; idx += 1
    ag_mate_buf = refs[idx]; idx += 1
    sq_vmem = refs[idx]; idx += 1
    s_in_buf = refs[idx]; idx += 1
    s_out_buf = refs[idx]; idx += 1
    r_out_buf = refs[idx]; idx += 1
    h_out_buf = refs[idx]; idx += 1

    br1_send_sem = refs[idx]; idx += 1
    br1_recv_sem = refs[idx]; idx += 1
    rs_send_sem = refs[idx]; idx += 1
    rs_recv_sem = refs[idx]; idx += 1
    ag_send_sem = refs[idx]; idx += 1
    ag_recv_sem = refs[idx]; idx += 1
    br2_send_sem = refs[idx]; idx += 1
    br2_recv_sem = refs[idx]; idx += 1
    s_in_sem = refs[idx]; idx += 1
    s_out_sem = refs[idx]; idx += 1
    r_out_sem = refs[idx]; idx += 1
    h_out_sem = refs[idx]; idx += 1

    tokens, k_local = attn_out_ref.shape
    hidden = wo_hbm_ref.shape[1]
    num_chips = world_size // 2
    half_h = hidden // 2
    chunk = hidden // world_size
    tn = wo_buf.shape[2]
    n_half_strips = half_h // tn

    me = lax.axis_index("model").astype(jnp.int32)
    my_chip = lax.shift_right_logical(me, jnp.int32(1))
    my_par = lax.bitwise_and(me, jnp.int32(1))
    mate = lax.bitwise_xor(me, jnp.int32(1))
    own_c0 = pl.multiple_of(my_par * half_h, half_h)
    mate_c0 = pl.multiple_of((jnp.int32(1) - my_par) * half_h, half_h)

    def _start_wo(slot: int, base_c, s_idx: int):
        c0 = pl.multiple_of(base_c + s_idx * tn, tn)
        pltpu.make_async_copy(
            wo_hbm_ref.at[:, pl.ds(c0, tn)], wo_buf.at[slot], wo_sem.at[slot]
        ).start()

    def _wait_wo(slot: int):
        pltpu.make_async_copy(wo_buf.at[slot], wo_buf.at[slot], wo_sem.at[slot]).wait()

    # Kick off HBM streaming of chipmate's column half at t=0 BEFORE barrier & head-norm!
    _start_wo(0, mate_c0, 0)
    if n_half_strips >= 2:
        _start_wo(1, mate_c0, 1)

    # Signal global entry barrier (wait deferred until right before Stage-1 remote send!).
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=(jnp.mod(me + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )

    _tail_head_norm_gate(
        attn_out_ref,
        gate_ref,
        gated_buf,
        tokens=tokens,
        k_local=k_local,
        head_dim=head_dim,
        head_norm_eps=head_norm_eps,
    )

    # Step 0a: Compute chipmate's column half first (1x HBM read, full 2-slot double-buffering).
    for s_idx in range(n_half_strips):
        slot = s_idx & 1
        _wait_wo(slot)
        _tail_dot_strip(
            gated_buf,
            wo_buf,
            slot,
            wo_scale_ref,
            mate_send_buf,
            mate_c0,
            s_idx,
            tokens=tokens,
            k_local=k_local,
            tn=tn,
            has_wo_scale=has_wo_scale,
        )
        if s_idx + 2 < n_half_strips:
            _start_wo(slot, mate_c0, s_idx + 2)
        elif (s_idx + 2 - n_half_strips) < n_half_strips:
            _start_wo(slot, own_c0, s_idx + 2 - n_half_strips)

    # Ensure peer is inside kernel, then FIRE Stage-1 XOR=1 Bridge RS + stream prefetch!
    pl.semaphore_wait(barrier, world_size - 1)
    br1_op = pltpu.make_async_remote_copy(
        mate_send_buf,
        mate_rs_buf,
        br1_send_sem,
        br1_recv_sem,
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br1_op.start()

    init_tiles = (tokens // 2) // 16 if use_pipe2 else (tokens // 16)
    _prefetch_stream_initial(
        stream_hbm_ref, s_in_buf, s_in_sem, row_base=0, n_tiles=init_tiles, nbuf=nbuf
    )

    # Step 0b: Compute OWN column half 100% overlapped with the in-flight XOR=1 bridge DMA!
    for s_idx in range(n_half_strips):
        slot = s_idx & 1
        _wait_wo(slot)
        _tail_dot_strip(
            gated_buf,
            wo_buf,
            slot,
            wo_scale_ref,
            own_part_buf,
            own_c0,
            s_idx,
            tokens=tokens,
            k_local=k_local,
            tn=tn,
            has_wo_scale=has_wo_scale,
        )
        if s_idx + 2 < n_half_strips:
            _start_wo(slot, own_c0, s_idx + 2)

    pltpu.make_async_copy(mate_rs_buf, mate_rs_buf, br1_recv_sem).wait()

    if use_pipe2:
        hb = tokens // 2
        hb_tiles = hb // 16

        # H0 cyclic ICI RS + AG.
        rs_ops_0 = _bridge_start_ici_rs_cyclic(
            my_chip,
            my_par,
            lambda r0, bt_r, c_off, c_sz: own_part_buf[
                pl.ds(r0, bt_r), pl.ds(pl.multiple_of(c_off, 128), c_sz)
            ],
            mate_rs_buf,
            rs_send_buf.at[0],
            rs_recv_buf.at[0],
            rs_send_sem,
            rs_recv_sem.at[0],
            row_count=hb,
            num_chips=num_chips,
            chunk=chunk,
            mate_row_base=0,
            sem_offset=0,
        )
        ag_ops_0 = _bridge_finish_rs_and_start_ag_cyclic(
            my_chip,
            my_par,
            rs_recv_buf.at[0],
            ag_half_buf.at[0],
            rs_recv_sem.at[0],
            ag_send_sem,
            ag_recv_sem.at[0],
            row_count=hb,
            num_chips=num_chips,
            chunk=chunk,
            sem_offset=0,
        )
        _wait_ag_recv(ag_half_buf.at[0], ag_recv_sem.at[0], num_chips=num_chips, chunk=chunk)

        # Fire H0 Bridge AG AND immediately launch H1 cyclic ICI RS!
        br2_0 = pltpu.make_async_remote_copy(
            ag_half_buf.at[0],
            ag_mate_buf.at[0],
            br2_send_sem.at[0],
            br2_recv_sem.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        br2_0.start()

        rs_ops_1 = _bridge_start_ici_rs_cyclic(
            my_chip,
            my_par,
            lambda r0, bt_r, c_off, c_sz: own_part_buf[
                pl.ds(pl.multiple_of(hb + r0, 16), bt_r),
                pl.ds(pl.multiple_of(c_off, 128), c_sz),
            ],
            mate_rs_buf,
            rs_send_buf.at[1],
            rs_recv_buf.at[1],
            rs_send_sem,
            rs_recv_sem.at[1],
            row_count=hb,
            num_chips=num_chips,
            chunk=chunk,
            mate_row_base=hb,
            sem_offset=num_chips - 1,
        )

        # Run H0 COMPLETE epilogue overlapped with H1 ICI RS!
        pltpu.make_async_copy(ag_mate_buf.at[0], ag_mate_buf.at[0], br2_recv_sem.at[0]).wait()
        _fused_dual_res_norm_hbm_pipelined(
            ag_half_buf.at[0],
            ag_mate_buf.at[0],
            own_c0,
            mate_c0,
            None,
            stream_hbm_ref,
            alpha_ref,
            beta_ref,
            next_gain_ref,
            stream_out_hbm,
            router_f32_hbm,
            hs_bf16_hbm,
            s_in_buf,
            s_out_buf,
            r_out_buf,
            h_out_buf,
            s_in_sem,
            s_out_sem,
            r_out_sem,
            h_out_sem,
            row_base=0,
            row_count=hb,
            hidden=hidden,
            branch_eps=branch_eps,
            next_eps=next_eps,
            stream_in_hbm=True,
            stream_already_prefetched=True,
            nbuf=nbuf,
        )

        # Prefetch H1 stream & finish H1 ICI RS + AG + Bridge AG + epilogue.
        _prefetch_stream_initial(
            stream_hbm_ref, s_in_buf, s_in_sem, row_base=hb, n_tiles=hb_tiles, nbuf=nbuf
        )
        ag_ops_1 = _bridge_finish_rs_and_start_ag_cyclic(
            my_chip,
            my_par,
            rs_recv_buf.at[1],
            ag_half_buf.at[1],
            rs_recv_sem.at[1],
            ag_send_sem,
            ag_recv_sem.at[1],
            row_count=hb,
            num_chips=num_chips,
            chunk=chunk,
            sem_offset=num_chips - 1,
        )
        _wait_ag_recv(ag_half_buf.at[1], ag_recv_sem.at[1], num_chips=num_chips, chunk=chunk)

        br2_1 = pltpu.make_async_remote_copy(
            ag_half_buf.at[1],
            ag_mate_buf.at[1],
            br2_send_sem.at[1],
            br2_recv_sem.at[1],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        br2_1.start()
        _accum_half_sq_vmem(
            ag_half_buf.at[1], sq_vmem, row_count=hb, half_h=half_h, add_to_existing=False
        )
        pltpu.make_async_copy(ag_mate_buf.at[1], ag_mate_buf.at[1], br2_recv_sem.at[1]).wait()
        _accum_half_sq_vmem(
            ag_mate_buf.at[1], sq_vmem, row_count=hb, half_h=half_h, add_to_existing=True
        )

        _fused_dual_res_norm_hbm_pipelined(
            ag_half_buf.at[1],
            ag_mate_buf.at[1],
            own_c0,
            mate_c0,
            sq_vmem,
            stream_hbm_ref,
            alpha_ref,
            beta_ref,
            next_gain_ref,
            stream_out_hbm,
            router_f32_hbm,
            hs_bf16_hbm,
            s_in_buf,
            s_out_buf,
            r_out_buf,
            h_out_buf,
            s_in_sem,
            s_out_sem,
            r_out_sem,
            h_out_sem,
            row_base=hb,
            row_count=hb,
            hidden=hidden,
            branch_eps=branch_eps,
            next_eps=next_eps,
            stream_in_hbm=True,
            stream_already_prefetched=True,
            nbuf=nbuf,
        )

        br2_0.wait_send()
        br2_1.wait_send()
        for op in ag_ops_0:
            op.wait_send()
        for op in ag_ops_1:
            op.wait_send()
        for op in rs_ops_0:
            op.wait_send()
        for op in rs_ops_1:
            op.wait_send()
        br1_op.wait_send()
        return

    # Single-pass cyclic ICI RS + AG + overlapped sq_own + HBM-pipelined epilogue.
    rs_ops = _bridge_start_ici_rs_cyclic(
        my_chip,
        my_par,
        lambda r0, bt_r, c_off, c_sz: own_part_buf[
            pl.ds(r0, bt_r), pl.ds(pl.multiple_of(c_off, 128), c_sz)
        ],
        mate_rs_buf,
        rs_send_buf,
        rs_recv_buf,
        rs_send_sem,
        rs_recv_sem,
        row_count=tokens,
        num_chips=num_chips,
        chunk=chunk,
    )
    ag_ops = _bridge_finish_rs_and_start_ag_cyclic(
        my_chip,
        my_par,
        rs_recv_buf,
        ag_half_buf,
        rs_recv_sem,
        ag_send_sem,
        ag_recv_sem,
        row_count=tokens,
        num_chips=num_chips,
        chunk=chunk,
    )
    _wait_ag_recv(ag_half_buf, ag_recv_sem, num_chips=num_chips, chunk=chunk)

    br2_op = pltpu.make_async_remote_copy(
        ag_half_buf,
        ag_mate_buf,
        br2_send_sem,
        br2_recv_sem,
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    br2_op.start()
    _accum_half_sq_vmem(ag_half_buf, sq_vmem, row_count=tokens, half_h=half_h, add_to_existing=False)
    pltpu.make_async_copy(ag_mate_buf, ag_mate_buf, br2_recv_sem).wait()
    _accum_half_sq_vmem(ag_mate_buf, sq_vmem, row_count=tokens, half_h=half_h, add_to_existing=True)

    _fused_dual_res_norm_hbm_pipelined(
        ag_half_buf,
        ag_mate_buf,
        own_c0,
        mate_c0,
        sq_vmem,
        stream_hbm_ref,
        alpha_ref,
        beta_ref,
        next_gain_ref,
        stream_out_hbm,
        router_f32_hbm,
        hs_bf16_hbm,
        s_in_buf,
        s_out_buf,
        r_out_buf,
        h_out_buf,
        s_in_sem,
        s_out_sem,
        r_out_sem,
        h_out_sem,
        row_base=0,
        row_count=tokens,
        hidden=hidden,
        branch_eps=branch_eps,
        next_eps=next_eps,
        stream_in_hbm=True,
        stream_already_prefetched=True,
        nbuf=nbuf,
    )

    br2_op.wait_send()
    for op in ag_ops:
        op.wait_send()
    for op in rs_ops:
        op.wait_send()
    br1_op.wait_send()


def tp_attn_tail_ar_res_norm_jax(
    stream,
    attn_out,
    gate,
    wo_kmajor,
    wo_scale,
    alpha,
    beta,
    next_gain,
    *,
    world_size: int,
    head_dim: int = 128,
    head_norm_eps: float = 1e-5,
    branch_eps: float = 1e-8,
    next_eps: float = 1e-5,
    rs_wire_dtype: str = "bfloat16",
    pipe2: bool | None = None,
    chipmate_xor: int = TPU7X_CHIPMATE_XOR,
    collective_id: int = _DEFAULT_COLLECTIVE_ID,
):
    """8-op fused attention tail: ``head_norm + gate + o_proj + bridge_ar + res_norm``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, k_local = attn_out.shape
    wk_k, hidden = wo_kmajor.shape
    if wk_k != k_local or gate.shape != (tokens, k_local) or stream.shape != (tokens, hidden):
        raise ValueError(
            f"Shape mismatch: stream={stream.shape}, attn_out={attn_out.shape}, "
            f"gate={gate.shape}, wo_kmajor={wo_kmajor.shape}"
        )
    if head_dim != 128 or k_local % 128 != 0:
        raise ValueError(f"Expected head_dim=128 and k_local%128==0, got {head_dim}, {k_local}")
    if tokens < 16 or tokens % 16 != 0 or tokens > 384:
        raise ValueError(f"Expected 16 <= tokens <= 384 and tokens % 16 == 0, got {tokens}")
    if world_size % 2 != 0 or world_size < 4 or hidden % (128 * world_size) != 0:
        raise ValueError(f"Unsupported world_size={world_size}, hidden={hidden}")

    use_pipe2 = (tokens >= 128 and tokens % 32 == 0) if pipe2 is None else (
        bool(pipe2) and tokens >= 32 and tokens % 32 == 0
    )
    half_h = hidden // 2
    # tn=512 keeps [bm<=32, tn=512] acc at <= 16 VREGs (0/0 spills) with full-K MXU pipelining!
    tn = 512 if half_h % 512 == 0 else (256 if half_h % 256 == 0 else 128)
    num_chips = world_size // 2
    chunk = hidden // world_size
    rs_dt = jnp.float32 if rs_wire_dtype == "float32" else jnp.bfloat16
    has_scale = wo_scale is not None

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=hbm)

    in_specs = [h_spec, v_spec, v_spec, h_spec]
    call_args = [
        stream.astype(jnp.float32),
        attn_out.astype(jnp.bfloat16),
        gate.astype(jnp.bfloat16),
        wo_kmajor,
    ]
    if has_scale:
        in_specs.append(v_spec)
        call_args.append(wo_scale.reshape(1, hidden).astype(jnp.float32))
    in_specs.extend([v_spec, v_spec, v_spec])
    call_args.extend([
        alpha.reshape(1, hidden).astype(jnp.float32),
        beta.reshape(1, hidden).astype(jnp.float32),
        next_gain.reshape(1, hidden).astype(jnp.float32),
    ])

    out_shapes = (
        jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
        jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
    )

    if use_pipe2:
        hb = tokens // 2
        nbuf = 1 if hb == 16 else 2
        scratch_shapes = [
            vmem((tokens, k_local), jnp.bfloat16),
            vmem((2, k_local, tn), wo_kmajor.dtype),
            sem((2,)),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((2, hb, half_h), rs_dt),
            vmem((2, num_chips, hb, chunk), rs_dt),
            vmem((2, hb, half_h), jnp.bfloat16),
            vmem((2, hb, half_h), jnp.bfloat16),
            vmem((hb, 128), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.bfloat16),
            sem,
            sem,
            sem((2 * (num_chips - 1),)),
            sem((2,)),
            sem((2 * (num_chips - 1),)),
            sem((2,)),
            sem((2,)),
            sem((2,)),
            sem((nbuf,)),
            sem((nbuf,)),
            sem((nbuf,)),
            sem((nbuf,)),
        ]
    else:
        nbuf = 1 if tokens == 16 else 2
        scratch_shapes = [
            vmem((tokens, k_local), jnp.bfloat16),
            vmem((2, k_local, tn), wo_kmajor.dtype),
            sem((2,)),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), rs_dt),
            vmem((num_chips, tokens, chunk), rs_dt),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, half_h), jnp.bfloat16),
            vmem((tokens, 128), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.float32),
            vmem((nbuf, 16, hidden), jnp.bfloat16),
            sem,
            sem,
            sem((num_chips - 1,)),
            sem,
            sem((num_chips - 1,)),
            sem,
            sem,
            sem,
            sem((nbuf,)),
            sem((nbuf,)),
            sem((nbuf,)),
            sem((nbuf,)),
        ]

    return pl.pallas_call(
        functools.partial(
            _attn_tail_bridge_ar_res_norm_kernel,
            world_size=world_size,
            head_dim=head_dim,
            head_norm_eps=head_norm_eps,
            branch_eps=branch_eps,
            next_eps=next_eps,
            has_wo_scale=has_scale,
            chipmate_xor=chipmate_xor,
            use_pipe2=use_pipe2,
            nbuf=nbuf,
        ),
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=(h_spec, h_spec, h_spec),
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=56 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_attn_tail_{'fp8' if has_scale else 'bf16'}"
            f"{'_p2' if use_pipe2 else ''}_ar_res_norm_b{tokens}_tp{world_size}"
        ),
    )(*call_args)


# =============================================================================
# 4. Zero-Spill Weight-Only FP8 K-Major Linear -> Direct BF16 (`B <= 256`)
# =============================================================================
def _pick_fp8_tile_n(n: int) -> int:
    # tn <= 512 keeps [bm<=32, tn] acc at <= 16 VREGs (0/0 spills) with full-K MXU pipelining.
    for tn in (512, 256, 128):
        if n % tn == 0:
            return tn
    return 256


def _fp8_linear_bf16_kernel(x_ref, w_ref, s_ref, o_ref) -> None:
    """Zero-spill 1x-weight-read FP8 linear using full-K MXU dots (<= 16 VREGs)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    m, _ = x_ref.shape
    bm = min(m, 32)
    while bm > 8 and m % bm != 0:
        bm //= 2
    n_m = m // bm

    def _m_body(im, _):
        rs = pl.ds(pl.multiple_of(im * bm, bm), bm)
        acc = jnp.dot(x_ref[rs, :], w_ref[...], preferred_element_type=jnp.float32)
        o_ref[rs, :] = (acc * s_ref[...]).astype(jnp.bfloat16)
        return jnp.int32(0)

    if n_m == 1:
        _m_body(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_m, _m_body, jnp.int32(0))


def fp8_linear_bf16_xla_jax(x, w8k, scale):
    """Native XLA mixed-precision ``bf16 @ fp8_e4m3fn * scale -> bf16`` (zero copy-done)."""
    import jax.numpy as jnp

    acc = jnp.dot(x.astype(jnp.bfloat16), w8k, preferred_element_type=jnp.float32)
    return (acc * scale.reshape(1, -1).astype(jnp.float32)).astype(jnp.bfloat16)


def fp8_linear_bf16_jax(x, w8k, scale):
    """Streamed K-major ``w8k [K, N]`` fp8_e4m3fn + ``scale [1, N]`` f32 -> ``[M, N]`` bf16."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m, k = x.shape
    wk, n = w8k.shape
    if wk != k or n % 128 != 0:
        raise ValueError(f"Expected w8k ({k}, 128*q), got {w8k.shape}")
    tn = _pick_fp8_tile_n(n)
    return pl.pallas_call(
        _fp8_linear_bf16_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(n // tn,),
        in_specs=[
            pl.BlockSpec((m, k), lambda j: (0, 0)),
            pl.BlockSpec((k, tn), lambda j: (0, j)),
            pl.BlockSpec((1, tn), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((m, tn), lambda j: (0, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=(48 if m > 256 else 32) * 1024**2,
            disable_bounds_checks=True,
        ),
        name=f"muse_fp8_linear_bf16_{m}x{k}x{n}_tn{tn}",
    )(x.astype(jnp.bfloat16), w8k, scale.reshape(1, n).astype(jnp.float32))


@torch.no_grad()
def quantize_fp8_k_major(weight_nk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``weight_nk [N, K]`` (``N % 128 == 0``) to fresh K-major ``(fp8 [K, N], f32 [1, N])``."""
    n, k = weight_nk.shape
    if n % 128 != 0:
        raise ValueError(f"Expected N divisible by 128 for FP8 K-major weight, got {(n, k)}")
    wk = weight_nk.detach().t().contiguous()
    cols, scales = [], []
    step = 2048
    for n0 in range(0, n, step):
        w = wk[:, n0 : n0 + step].float()
        s = (w.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
        cols.append((w / s).to(torch.float8_e4m3fn))
        scales.append(s.reshape(1, -1))
    w8k = torch.cat(cols, dim=1).contiguous()
    scale = torch.cat(scales, dim=1).contiguous()
    return w8k, scale


# =============================================================================
# 5. TorchTPU `cpu_export_jax_op` Registration Helpers
# =============================================================================
@functools.cache
def _get_tp_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def get_muse_decode_bridge_ar_op(world_size: int) -> Callable:
    op = _BRIDGE_AR_OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _BRIDGE_AR_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            sharded = jax.shard_map(
                functools.partial(bridge_all_reduce_jax, world_size=world_size),
                mesh=mesh,
                in_specs=(P("model"),),
                out_specs=P(),
                check_vma=False,
            )

            def _wrapped(x: jax.Array) -> jax.Array:
                return sharded(x)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_decode_bridge_ar_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=(P("model"),),
                output_partition_specs=(P(),),
            )
            _BRIDGE_AR_OPS[world_size] = op
    return op


def get_muse_decode_tp_ar_res_norm_op(world_size: int) -> Callable:
    op = _TP_AR_RES_NORM_OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _TP_AR_RES_NORM_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (P(), P("model"), P(), P(), P())
            in_specs = (*t_specs, None, None)
            out_specs = (P(), P(), P())

            def _wrapped(
                stream: jax.Array,
                o_partial: jax.Array,
                alpha: jax.Array,
                beta: jax.Array,
                next_gain: jax.Array,
                branch_eps: float,
                next_eps: float,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return jax.shard_map(
                    functools.partial(
                        tp_ar_res_norm_jax,
                        world_size=world_size,
                        branch_eps=float(branch_eps),
                        next_eps=float(next_eps),
                    ),
                    mesh=mesh,
                    in_specs=t_specs,
                    out_specs=out_specs,
                    check_vma=False,
                )(stream, o_partial, alpha, beta, next_gain)

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_decode_tp_ar_res_norm_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _TP_AR_RES_NORM_OPS[world_size] = op
    return op


def get_muse_decode_attn_tail_ar_res_norm_op(world_size: int, *, fp8_wo: bool = False) -> Callable:
    cache = _ATTN_TAIL_FP8_OPS if fp8_wo else _ATTN_TAIL_BF16_OPS
    op = cache.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = cache.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            out_specs = (P(), P(), P())
            if fp8_wo:
                t_specs = (P(), P("model"), P("model"), P("model"), P("model"), P(), P(), P())
                in_specs = (*t_specs, None, None, None)

                def _wrapped_fp8(
                    stream: jax.Array,
                    attn_out: jax.Array,
                    gate: jax.Array,
                    wo_w8k: jax.Array,
                    wo_scale: jax.Array,
                    alpha: jax.Array,
                    beta: jax.Array,
                    next_gain: jax.Array,
                    head_norm_eps: float,
                    branch_eps: float,
                    next_eps: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return jax.shard_map(
                        functools.partial(
                            tp_attn_tail_ar_res_norm_jax,
                            world_size=world_size,
                            head_norm_eps=float(head_norm_eps),
                            branch_eps=float(branch_eps),
                            next_eps=float(next_eps),
                        ),
                        mesh=mesh,
                        in_specs=t_specs,
                        out_specs=out_specs,
                        check_vma=False,
                    )(stream, attn_out, gate, wo_w8k, wo_scale, alpha, beta, next_gain)

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_decode_attn_tail_fp8_ar_res_norm_tp{world_size}",
                    _wrapped_fp8,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_specs,
                )
            else:
                t_specs = (P(), P("model"), P("model"), P("model"), P(), P(), P())
                in_specs = (*t_specs, None, None, None)

                def _wrapped_bf16(
                    stream: jax.Array,
                    attn_out: jax.Array,
                    gate: jax.Array,
                    wo_wk: jax.Array,
                    alpha: jax.Array,
                    beta: jax.Array,
                    next_gain: jax.Array,
                    head_norm_eps: float,
                    branch_eps: float,
                    next_eps: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return jax.shard_map(
                        functools.partial(
                            tp_attn_tail_ar_res_norm_jax,
                            world_size=world_size,
                            head_norm_eps=float(head_norm_eps),
                            branch_eps=float(branch_eps),
                            next_eps=float(next_eps),
                        ),
                        mesh=mesh,
                        in_specs=t_specs,
                        out_specs=out_specs,
                        check_vma=False,
                    )(stream, attn_out, gate, wo_wk, None, alpha, beta, next_gain)

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_decode_attn_tail_bf16_ar_res_norm_tp{world_size}",
                    _wrapped_bf16,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_specs,
                )
            cache[world_size] = op
    return op


def get_muse_decode_fp8_linear_bf16_op() -> Callable:
    if _FP8_LINEAR_BF16_OP:
        return _FP8_LINEAR_BF16_OP[0]
    with _OPS_LOCK:
        if not _FP8_LINEAR_BF16_OP:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(x: jax.Array, w8k: jax.Array, scale: jax.Array) -> jax.Array:
                return fp8_linear_bf16_xla_jax(x, w8k, scale)

            def _fake(x: torch.Tensor, w8k: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
                del scale
                return x.new_empty((x.shape[0], w8k.shape[1]), dtype=torch.bfloat16)

            op = cpu_export_jax_op("sglang_tpu_pallas::muse_decode_fp8_linear_bf16", _wrapped)
            op.register_fake(_fake)
            _FP8_LINEAR_BF16_OP.append(op)
    return _FP8_LINEAR_BF16_OP[0]
