# SPDX-License-Identifier: Apache-2.0
"""Zero-spill weight-only FP8 LM head and fused distributed argmax/top-k kernels for TPU7x.

Targets Muse Spark 816B's vocab-parallel LM head (`H=8192`, `V=202,048` -> `25,256` local
cols on TP8, `12,628` on TP16) across decode (`B=16..64`) and DFlash verify/draft
(`B=64..256`):

  1. ``fp8_lm_head_zsp_jax`` / ``fp8_lm_head_xla_jax``: Per-vocab-row K-major ``fp8_e4m3fn``
     LM head (``-50%`` HBM weight bytes: ``394.6 -> 197.3 MiB/rank``) using ``bm <= 128``
     (100% TPU7x MXU utilization at ``B=128, 256``) with ``0 / 0`` hardware LLO spills,
     fused ``logit_scale``, and selectable ``f32``/``bf16`` output.
  2. ``fp8_lm_head_argmax_jax``: Single-kernel fused ``FP8 lm_head + 128-lane online exact
     argmax + 1-hop cyclic VMEM all-gather``. ``[B, V_local]`` logits NEVER touch HBM;
     exact vocab-padding masking and lowest-global-ID tie-breaking match
     ``distributed_topk.py`` bit-for-bit on the FP8 logits.
  3. ``fp8_lm_head_fused_topk64_jax``: Single-kernel fused ``FP8 lm_head + zero-cross-lane
     vertical (64, 128) bitonic Top-64 + 1-hop cyclic VMEM all-gather + in-VMEM global
     rank merge``. ``[B, V_local]`` logits NEVER touch HBM, peak VREGs stay ``<= 18``
     (``0 / 0`` spills), and exact lowest-global-ID col-0 tie-breaking is tracked in
     1 VREG at zero extra cost.
  4. ``distributed_topk_vmem_ag_jax``: Lossless standalone drop-in replacement for
     ``distributed_topk.py`` using the same vertical ``(64, 128)`` bitonic Top-64 +
     1-hop cyclic VMEM all-gather + in-VMEM global merge for BF16/FP32 logits.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from collections.abc import Callable
from threading import Lock

import torch

_LM_HEAD_TN = 512
_QUANT_COLS = 4096
_DEFAULT_COLLECTIVE_ID = 11
SUPPORTED_WORLD_SIZES = frozenset((8, 16))
MAX_DISTRIBUTED_TOP_K = 64

_OPS_LOCK = Lock()
_FP8_LM_HEAD_OPS: dict[tuple[bool, bool], Callable] = {}
_FUSED_TOPK_OPS: dict[tuple[int, bool], Callable] = {}
_LOSSLESS_TOPK_OPS: dict[int, Callable] = {}
_GUMBEL_MAX_OPS: dict[int, Callable] = {}


def _ceil_multiple(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _pad_batch_pow2(tokens: int) -> int:
    if tokens <= 16:
        return 16
    if tokens <= 32:
        return 32
    if tokens <= 64:
        return 64
    if tokens <= 128:
        return 128
    if tokens <= 256:
        return 256
    if tokens <= 384:
        return 384
    raise ValueError(f"Expected 1 <= tokens <= 384, got {tokens}")


def _entry_barrier_signal(me, world_size: int) -> None:
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


def _entry_barrier_wait(world_size: int) -> None:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    pl.semaphore_wait(pltpu.get_barrier_semaphore(), world_size - 1)


def _map_local_to_global_id(
    local_idx,
    *,
    me,
    local_width: int,
    org_vocab_size: int,
    org_vocab_padded_count: int,
    added_width: int,
):
    import jax.numpy as jnp

    return jnp.where(
        local_idx < jnp.int32(org_vocab_padded_count),
        me * jnp.int32(org_vocab_padded_count) + local_idx,
        jnp.where(
            local_idx < jnp.int32(local_width),
            jnp.int32(org_vocab_size)
            + me * jnp.int32(added_width)
            + (local_idx - jnp.int32(org_vocab_padded_count)),
            jnp.int32(0x7FFFFFFF),
        ),
    )


# =============================================================================
# 0. Zero-Cross-Lane Vertical `(64, 128)` Bitonic Top-64 Primitives (8 VREGs/side)
# =============================================================================
def _v64_cas_pair(lv, li, rv, ri, *, desc: bool):
    import jax.numpy as jnp

    take_left = (lv > rv) if desc else (rv > lv)
    return (
        jnp.where(take_left, lv, rv),
        jnp.where(take_left, li, ri),
        jnp.where(take_left, rv, lv),
        jnp.where(take_left, ri, li),
    )


def _v64_cas_intra(v, i, sep: int, *, stage: int | None, tile_base: int):
    import jax.numpy as jnp
    from jax import lax

    s8 = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    perm = jnp.bitwise_xor(s8, jnp.int32(sep))
    right = (s8 & jnp.int32(sep)) > 0
    pv = jnp.take_along_axis(v, perm, axis=0)
    pi = jnp.take_along_axis(i, perm, axis=0)
    cl = jnp.where(right, pv, v)
    cr = jnp.where(right, v, pv)
    if stage is None:
        take_left = cl > cr
    elif stage >= 3:
        # Since sublane s in 0..7 never affects bit >= 3, direction is compile-time constant!
        take_left = (cl > cr) if bool(tile_base & (1 << stage)) else (cr > cl)
    else:
        # Exact distributed_topk.py pattern: pure i1 XOR on full (8, 128) i32-derived masks.
        bit = ((jnp.int32(tile_base) + s8) & jnp.int32(1 << stage)) > 0
        take_left = (cr > cl) ^ bit
    return jnp.where(take_left, v, pv), jnp.where(take_left, i, pi)


def _v64_substage_quad(vt, it, sub: int, *, stage: int | None, quad_base_tile: int):
    """Run one bitonic substage on 4 ``(8, 128)`` tiles (<= 8 data VREGs -> 0/0 spills!)."""
    sep = 1 << sub
    if sep < 8:
        ov, oi = [], []
        for t in range(4):
            v, i = _v64_cas_intra(
                vt[t], it[t], sep, stage=stage, tile_base=(quad_base_tile + t) * 8
            )
            ov.append(v)
            oi.append(i)
        return ov, oi
    tsep = sep // 8  # 1 or 2
    ov = [None] * 4
    oi = [None] * 4
    for p in range(2):
        left = (p // tsep) * 2 * tsep + (p % tsep)
        right = left + tsep
        desc = (
            True
            if stage is None
            else bool(((quad_base_tile + left) * 8) & (1 << stage))
        )
        ov[left], oi[left], ov[right], oi[right] = _v64_cas_pair(
            vt[left], it[left], vt[right], it[right], desc=desc
        )
    return ov, oi


def _v64_sort_quad_bitonic(vt, it, quad_idx: int):
    """Sort quad 0 (tiles 0..3) ASC or quad 1 (tiles 4..7) DESC in <= 8 data VREGs."""
    base_t = quad_idx * 4
    for stage in range(1, 6):
        for sub in range(stage - 1, -1, -1):
            vt, it = _v64_substage_quad(vt, it, sub, stage=stage, quad_base_tile=base_t)
    return vt, it


def _v64_finish_quad(vt, it, *, stage: int | None, quad_idx: int):
    """Finish bitonic substages 4..0 on 4 tiles (stage=6 ASC, stage=None DESC) in <= 8 VREGs."""
    base_t = quad_idx * 4
    for sub in (4, 3, 2, 1, 0):
        vt, it = _v64_substage_quad(vt, it, sub, stage=stage, quad_base_tile=base_t)
    return vt, it


def _v64_quad_sub5_pass(work_v_ref, work_i_ref, *, desc: bool):
    """Distance-32 compare-swap across tile pairs (0,4)..(3,7) in 4 data VREGs."""
    from jax.experimental import pallas as pl

    for p in range(4):
        sl_l = pl.ds(p * 8, 8)
        sl_r = pl.ds((p + 4) * 8, 8)
        lv, li = work_v_ref[sl_l, :], work_i_ref[sl_l, :]
        rv, ri = work_v_ref[sl_r, :], work_i_ref[sl_r, :]
        lv, li, rv, ri = _v64_cas_pair(lv, li, rv, ri, desc=desc)
        work_v_ref[sl_l, :], work_i_ref[sl_l, :] = lv, li
        work_v_ref[sl_r, :], work_i_ref[sl_r, :] = rv, ri


def _v64_quad_merge_work_to_top_desc(work_v_ref, work_i_ref, top_v_ref, top_i_ref, m_base):
    """Finish 64-element bitonic DESC merge from ``work_*`` -> ``top_*[m_base:m_base+64]`` (<= 8 VREGs)."""
    from jax.experimental import pallas as pl

    _v64_quad_sub5_pass(work_v_ref, work_i_ref, desc=True)
    for q in (0, 1):
        vt = [work_v_ref[pl.ds((q * 4 + k) * 8, 8), :] for k in range(4)]
        it = [work_i_ref[pl.ds((q * 4 + k) * 8, 8), :] for k in range(4)]
        vt, it = _v64_finish_quad(vt, it, stage=None, quad_idx=q)
        for k in range(4):
            sl_t = pl.ds(m_base + (q * 4 + k) * 8, 8)
            top_v_ref[sl_t, :] = vt[k]
            top_i_ref[sl_t, :] = it[k]


def _v64_quad_stream_merge_64(
    blk_t_buf,
    h_off,
    c0,
    top_v_buf,
    top_i_buf,
    m_base,
    work_v_buf,
    work_i_buf,
    arg_v_buf,
    arg_i_buf,
    a_base,
    *,
    org_count,
    added_count,
    org_vocab_padded_count: int,
    exact_tie_break: bool,
):
    """Zero-spill (<= 12 live VREGs) 64-col sort-ASC + half-cleaner + bitonic merge-DESC."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    sub_iota = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    # Step A & B: load/mask/sort quad 0 (ASC) and quad 1 (DESC) independently (<= 8 data VREGs).
    for q in (0, 1):
        vt, it = [], []
        if exact_tie_break:
            av = arg_v_buf[pl.ds(a_base, 8), :]
            ai = arg_i_buf[pl.ds(a_base, 8), :]
        for k in range(4):
            tk = q * 4 + k
            c_k = c0 + jnp.int32(tk * 8) + sub_iota
            add_p = c_k - jnp.int32(org_vocab_padded_count)
            valid = (c_k < org_count) | ((add_p >= 0) & (add_p < added_count))
            vk = jnp.where(valid, blk_t_buf[pl.ds(h_off + tk * 8, 8), :], -jnp.inf)
            ik = jnp.where(valid, c_k, jnp.int32(0x7FFFFFFF))
            if exact_tie_break:
                better = vk > av
                av = jnp.where(better, vk, av)
                ai = jnp.where(better, ik, ai)
            vt.append(vk)
            it.append(ik)
        if exact_tie_break:
            arg_v_buf[pl.ds(a_base, 8), :] = av
            arg_i_buf[pl.ds(a_base, 8), :] = ai
        vt, it = _v64_sort_quad_bitonic(vt, it, q)
        for k in range(4):
            sl_w = pl.ds((q * 4 + k) * 8, 8)
            work_v_buf[sl_w, :] = vt[k]
            work_i_buf[sl_w, :] = it[k]

    # Step C: stage-6 sub=5 ASC across pairs (0,4)..(3,7) (4 data VREGs).
    _v64_quad_sub5_pass(work_v_buf, work_i_buf, desc=False)

    # Step D & E: finish stage-6 sub=4..0 ASC per quad + half-cleaner max with top_* -> work_*.
    for q in (0, 1):
        vt = [work_v_buf[pl.ds((q * 4 + k) * 8, 8), :] for k in range(4)]
        it = [work_i_buf[pl.ds((q * 4 + k) * 8, 8), :] for k in range(4)]
        vt, it = _v64_finish_quad(vt, it, stage=6, quad_idx=q)
        for k in range(4):
            tk = q * 4 + k
            sl_w = pl.ds(tk * 8, 8)
            sl_t = pl.ds(m_base + tk * 8, 8)
            cv = top_v_buf[sl_t, :]
            ci = top_i_buf[sl_t, :]
            take = vt[k] > cv
            work_v_buf[sl_w, :] = jnp.where(take, vt[k], cv)
            work_i_buf[sl_w, :] = jnp.where(take, it[k], ci)

    # Step F: bitonic DESC merge from work_* -> top_* (<= 8 data VREGs).
    _v64_quad_merge_work_to_top_desc(work_v_buf, work_i_buf, top_v_buf, top_i_buf, m_base)


def _v64_quad_merge_rank_desc(recv_v_buf, recv_i_buf, r, top_v_buf, top_i_buf, m_base, work_v_buf, work_i_buf):
    """Zero-spill rank merge: reverse DESC ``recv[r]`` tile-by-tile -> half-cleaner -> merge DESC."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    sub_perm = jnp.bitwise_xor(lax.broadcasted_iota(jnp.int32, (8, 128), 0), jnp.int32(7))
    for k in range(8):
        sl_rev = pl.ds(m_base + (7 - k) * 8, 8)
        sl_t = pl.ds(m_base + k * 8, 8)
        sl_w = pl.ds(k * 8, 8)
        rv_asc = jnp.take_along_axis(recv_v_buf[r, sl_rev, :], sub_perm, axis=0)
        ri_asc = jnp.take_along_axis(recv_i_buf[r, sl_rev, :], sub_perm, axis=0)
        cv = top_v_buf[sl_t, :]
        ci = top_i_buf[sl_t, :]
        take = rv_asc > cv
        work_v_buf[sl_w, :] = jnp.where(take, rv_asc, cv)
        work_i_buf[sl_w, :] = jnp.where(take, ri_asc, ci)
    _v64_quad_merge_work_to_top_desc(work_v_buf, work_i_buf, top_v_buf, top_i_buf, m_base)


def _v64_quad_merge_butterfly_hop_desc(
    recv_v_buf,
    recv_i_buf,
    hop: int,
    me_lower,
    top_v_buf,
    top_i_buf,
    m_base,
    work_v_buf,
    work_i_buf,
):
    """Canonical symmetric bit-identical butterfly merge between ``me`` and ``peer``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    sub_perm = jnp.bitwise_xor(lax.broadcasted_iota(jnp.int32, (8, 128), 0), jnp.int32(7))
    for k in range(8):
        sl_rev = pl.ds(m_base + (7 - k) * 8, 8)
        sl_t = pl.ds(m_base + k * 8, 8)
        sl_w = pl.ds(k * 8, 8)
        tv_f = top_v_buf[sl_t, :]
        ti_f = top_i_buf[sl_t, :]
        pv_f = recv_v_buf[hop, sl_t, :]
        pi_f = recv_i_buf[hop, sl_t, :]
        tv_r = top_v_buf[sl_rev, :]
        ti_r = top_i_buf[sl_rev, :]
        pv_r = recv_v_buf[hop, sl_rev, :]
        pi_r = recv_i_buf[hop, sl_rev, :]

        base_v = lax.select(me_lower, tv_f, pv_f)
        base_i = lax.select(me_lower, ti_f, pi_f)
        inc_v_rev = lax.select(me_lower, pv_r, tv_r)
        inc_i_rev = lax.select(me_lower, pi_r, ti_r)

        rv_asc = jnp.take_along_axis(inc_v_rev, sub_perm, axis=0)
        ri_asc = jnp.take_along_axis(inc_i_rev, sub_perm, axis=0)
        take = rv_asc > base_v
        work_v_buf[sl_w, :] = jnp.where(take, rv_asc, base_v)
        work_i_buf[sl_w, :] = jnp.where(take, ri_asc, base_i)
    _v64_quad_merge_work_to_top_desc(work_v_buf, work_i_buf, top_v_buf, top_i_buf, m_base)


def _v64_sublane_exact_argmax(av, ai):
    """3-step butterfly on ``(8, 128)`` -> all 8 sublanes hold exact ``(max_val, min_id)``."""
    import jax.numpy as jnp
    from jax import lax

    s8 = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    for sep in (4, 2, 1):
        perm = jnp.bitwise_xor(s8, jnp.int32(sep))
        pv = jnp.take_along_axis(av, perm, axis=0)
        pi = jnp.take_along_axis(ai, perm, axis=0)
        take = (pv > av) | ((pv == av) & (pi < ai))
        av = jnp.where(take, pv, av)
        ai = jnp.where(take, pi, ai)
    return av, ai


def _v64_inject_exact_col0(top_i_ref, m_base, exact_id_8x128):
    """Swap exact lowest-global-ID argmax into slot 0 of ``top_i_ref[m_base:m_base+64]``."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    t0 = top_i_ref[pl.ds(m_base, 8), :]
    z_idx = jnp.zeros((8, 128), dtype=jnp.int32)
    old0 = jnp.take_along_axis(t0, z_idx, axis=0)
    is_sub0 = lax.broadcasted_iota(jnp.int32, (8, 128), 0) == jnp.int32(0)
    for k in range(8):
        sl = pl.ds(m_base + k * 8, 8)
        tk = top_i_ref[sl, :]
        tk = jnp.where(tk == exact_id_8x128, old0, tk)
        if k == 0:
            tk = jnp.where(is_sub0, exact_id_8x128, tk)
        top_i_ref[sl, :] = tk


# =============================================================================
# 1. Zero-Spill Weight-Only FP8 LM Head (Pallas bm<=128 + Native XLA)
# =============================================================================
def _fp8_lm_head_zsp_kernel(
    x_ref,
    w_ref,
    s_ref,
    o_ref,
    *,
    logit_scale: float,
) -> None:
    """Zero-spill K-major FP8 LM head tile (``bm <= 128`` for 100% TPU7x MXU utilization)."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    m, _ = x_ref.shape
    tn = w_ref.shape[1]
    if m <= 128:
        bm = m
    elif m % 128 == 0:
        bm = 128
    elif m % 64 == 0:
        bm = 64
    elif m % 32 == 0:
        bm = 32
    else:
        bm = 8
    n_m = m // bm
    scale_vec = s_ref[...] * jnp.float32(logit_scale) if logit_scale != 1.0 else s_ref[...]

    def _m_body(im, _):
        rs = pl.ds(pl.multiple_of(im * bm, bm), bm)
        if tn <= 512:
            acc = jnp.dot(x_ref[rs, :], w_ref[...], preferred_element_type=jnp.float32)
            o_ref[rs, :] = (acc * scale_vec).astype(o_ref.dtype)
        else:
            for h in range(tn // 512):
                cs = pl.ds(h * 512, 512)
                acc = jnp.dot(x_ref[rs, :], w_ref[:, cs], preferred_element_type=jnp.float32)
                o_ref[rs, cs] = (acc * scale_vec[:, cs]).astype(o_ref.dtype)
        return jnp.int32(0)

    if n_m == 1:
        _m_body(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_m, _m_body, jnp.int32(0))


def fp8_lm_head_zsp_jax(
    x,
    w8k,
    scale,
    *,
    out_bf16: bool = False,
    logit_scale: float = 1.0,
    tn: int = 1024,
):
    """Zero-spill Pallas ``x [M, H] @ w8k [H, Vp] fp8 * scale [1, Vp] -> [M, Vp]``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m, h = x.shape
    wk_h, vp = w8k.shape
    if wk_h != h or vp % 512 != 0:
        raise ValueError(f"Expected w8k ({h}, 512*q), got {w8k.shape}")
    eff_tn = tn if (vp % tn == 0) else 512
    m_pad = m if (m <= 128 or m % 8 == 0) else _ceil_multiple(m, 8)
    x_in = x.astype(jnp.bfloat16)
    if m_pad > m:
        x_in = jnp.pad(x_in, ((0, m_pad - m), (0, 0)))
    out_dt = jnp.bfloat16 if out_bf16 else jnp.float32
    out = pl.pallas_call(
        functools.partial(_fp8_lm_head_zsp_kernel, logit_scale=float(logit_scale)),
        out_shape=jax.ShapeDtypeStruct((m_pad, vp), out_dt),
        grid=(vp // eff_tn,),
        in_specs=[
            pl.BlockSpec((m_pad, h), lambda j: (0, 0)),
            pl.BlockSpec((h, eff_tn), lambda j: (0, j)),
            pl.BlockSpec((1, eff_tn), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((m_pad, eff_tn), lambda j: (0, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=(48 if m_pad > 256 else 32) * 1024**2,
            disable_bounds_checks=True,
        ),
        name=f"muse_fp8_lm_head_zsp_{m_pad}x{h}x{vp}_tn{eff_tn}_{'bf16' if out_bf16 else 'f32'}",
    )(x_in, w8k, scale.reshape(1, vp).astype(jnp.float32))
    return out[:m] if m_pad > m else out


def fp8_lm_head_xla_jax(
    x,
    w8k,
    scale,
    *,
    out_bf16: bool = False,
    logit_scale: float = 1.0,
):
    """Native XLA mixed-precision ``bf16 @ fp8_e4m3fn * scale`` LM head."""
    import jax.numpy as jnp

    acc = jnp.dot(x.astype(jnp.bfloat16), w8k, preferred_element_type=jnp.float32)
    s = scale.reshape(1, -1).astype(jnp.float32)
    if logit_scale != 1.0:
        s = s * jnp.float32(logit_scale)
    out = acc * s
    return out.astype(jnp.bfloat16) if out_bf16 else out


# =============================================================================
# 2. Single-Kernel Fused `FP8 LM_HEAD + 128-Lane Online Argmax + 1-Hop VMEM AG`
# =============================================================================
def _fp8_lm_head_argmax_kernel(
    x_ref,
    w8k_hbm,
    scale_ref,
    out_val_ref,
    out_id_ref,
    w_buf,
    tile_buf,
    best_val_buf,
    best_idx_buf,
    pack_tmp_v,
    pack_tmp_i,
    send_val_buf,
    send_id_buf,
    recv_val_buf,
    recv_id_buf,
    w_sem,
    ag_send_sem,
    ag_val_recv_sem,
    ag_id_recv_sem,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    logit_scale: float,
    tn: int,
    bm_mxu: int,
) -> None:
    """Zero-spill fused FP8 LM head + 128-lane online exact argmax + 1-hop VMEM AG.

    Uses ``bm_mxu = min(b_pad, 128)`` (100% MXU utilization at ``B=128, 256``), ``512``-col
    sub-tiles in ``tile_buf``, and ``32``-row VALU strips for both online argmax and final
    reduction (``0 / 0`` hardware spills across all ``B=16..256``).
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    b_pad, _ = x_ref.shape
    vp = w8k_hbm.shape[1]
    n_tiles = vp // tn
    n_m = b_pad // bm_mxu
    bm_v = min(bm_mxu, 32)
    n_v = bm_mxu // bm_v
    n_sub = tn // 512
    added_width = local_width - org_vocab_padded_count
    pack_rows = (b_pad // 128) * 8 if b_pad >= 128 else 8
    n_pack_chunks = pack_rows // 8

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier_signal(me, world_size)

    org_count = jnp.clip(
        jnp.int32(org_vocab_size) - me * jnp.int32(org_vocab_padded_count),
        jnp.int32(0),
        jnp.int32(org_vocab_padded_count),
    )
    added_start = jnp.int32(org_vocab_size) + me * jnp.int32(added_width)
    added_count = jnp.clip(
        jnp.int32(global_vocab_size) - added_start,
        jnp.int32(0),
        jnp.int32(added_width),
    )

    def _start_w(slot: int, tile_idx):
        c0 = pl.multiple_of(tile_idx * tn, tn)
        pltpu.make_async_copy(
            w8k_hbm.at[:, pl.ds(c0, tn)],
            w_buf.at[slot],
            w_sem.at[slot],
        ).start()

    def _wait_w(slot: int):
        pltpu.make_async_copy(w_buf.at[slot], w_buf.at[slot], w_sem.at[slot]).wait()

    _start_w(0, jnp.int32(0))
    if n_tiles >= 2:
        _start_w(1, jnp.int32(1))

    neg_inf = jnp.full((bm_v, 128), -jnp.inf, dtype=jnp.float32)
    max_i32 = jnp.full((bm_v, 128), 0x7FFFFFFF, dtype=jnp.int32)
    lane_iota = lax.broadcasted_iota(jnp.int32, (1, 128), 1)

    n_init_strips = b_pad // bm_v

    def _init_best(ist, _):
        rs = pl.ds(pl.multiple_of(ist * bm_v, bm_v), bm_v)
        best_val_buf[rs, :] = neg_inf
        best_idx_buf[rs, :] = max_i32
        return jnp.int32(0)

    if n_init_strips == 1:
        _init_best(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_init_strips, _init_best, jnp.int32(0))

    def _process_tile(tile_idx, slot: int, has_next_prefetch: bool) -> None:
        _wait_w(slot)
        c_tile = pl.multiple_of(tile_idx * tn, tn)

        def _sub_body(isub, _):
            c_sub = pl.multiple_of(isub * 512, 512)
            c_base = c_tile + c_sub
            s_vec = scale_ref[0:1, pl.ds(c_base, 512)]
            if logit_scale != 1.0:
                s_vec = s_vec * jnp.float32(logit_scale)

            def _m_step(im, _):
                r_m = pl.multiple_of(im * bm_mxu, bm_mxu)
                rs_m = pl.ds(r_m, bm_mxu)
                acc = jnp.dot(
                    x_ref[rs_m, :],
                    w_buf[slot, :, pl.ds(c_sub, 512)],
                    preferred_element_type=jnp.float32,
                )
                tile_buf[...] = acc * s_vec

                def _v_step(iv, _):
                    r_v = pl.multiple_of(iv * bm_v, bm_v)
                    rs_local = pl.ds(r_v, bm_v)
                    rs_glob = pl.ds(r_m + r_v, bm_v)
                    cur_v = best_val_buf[rs_glob, :]
                    cur_i = best_idx_buf[rs_glob, :]
                    for q in range(4):
                        c_q = c_base + jnp.int32(q * 128) + lane_iota
                        add_p = c_q - jnp.int32(org_vocab_padded_count)
                        valid = (c_q < org_count) | ((add_p >= 0) & (add_p < added_count))
                        v_q = jnp.where(valid, tile_buf[rs_local, pl.ds(q * 128, 128)], -jnp.inf)
                        better = v_q > cur_v
                        cur_v = jnp.where(better, v_q, cur_v)
                        cur_i = jnp.where(better, c_q, cur_i)
                    best_val_buf[rs_glob, :] = cur_v
                    best_idx_buf[rs_glob, :] = cur_i
                    return jnp.int32(0)

                if n_v == 1:
                    _v_step(jnp.int32(0), jnp.int32(0))
                else:
                    lax.fori_loop(0, n_v, _v_step, jnp.int32(0))
                return jnp.int32(0)

            if n_m == 1:
                _m_step(jnp.int32(0), jnp.int32(0))
            else:
                lax.fori_loop(0, n_m, _m_step, jnp.int32(0))
            return jnp.int32(0)

        if n_sub == 1:
            _sub_body(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_sub, _sub_body, jnp.int32(0))

        if has_next_prefetch:
            @pl.when(tile_idx + 2 < n_tiles)
            def _pf():
                _start_w(slot, tile_idx + 2)

    n_pairs = n_tiles // 2
    if n_pairs == 1:
        _process_tile(jnp.int32(0), 0, has_next_prefetch=(n_tiles > 2))
        _process_tile(jnp.int32(1), 1, has_next_prefetch=(n_tiles > 3))
    elif n_pairs > 1:
        def _pair_body(p, _):
            t0 = p * 2
            _process_tile(t0, 0, has_next_prefetch=True)
            _process_tile(t0 + 1, 1, has_next_prefetch=True)
            return jnp.int32(0)

        lax.fori_loop(0, n_pairs, _pair_body, jnp.int32(0))
    if n_tiles % 2 == 1:
        _process_tile(jnp.int32(n_tiles - 1), 0, has_next_prefetch=False)

    # Zero-spill strip-by-strip 128-lane reduction into pack_tmp (128, 8) -> native .T -> (8, 128)!
    chunk_b = min(b_pad, 128)
    n_st = chunk_b // bm_v
    for ch in range(n_pack_chunks):
        if chunk_b < 128:
            pack_tmp_v[...] = jnp.full((128, 8), -jnp.inf, dtype=jnp.float32)
            pack_tmp_i[...] = jnp.full((128, 8), 0x7FFFFFFF, dtype=jnp.int32)

        ch_base = ch * chunk_b

        def _pack_strip(ist, _):
            r_l = pl.multiple_of(ist * bm_v, bm_v)
            bv = best_val_buf[pl.ds(ch_base + r_l, bm_v), :]
            bi = best_idx_buf[pl.ds(ch_base + r_l, bm_v), :]
            gid = _map_local_to_global_id(
                bi,
                me=me,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                added_width=added_width,
            )
            rmax = jnp.max(bv, axis=-1, keepdims=True)
            rgid = jnp.min(
                jnp.where(bv == rmax, gid, jnp.int32(0x7FFFFFFF)),
                axis=-1,
                keepdims=True,
            )
            pack_tmp_v[pl.ds(r_l, bm_v), :] = jnp.broadcast_to(rmax, (bm_v, 8))
            pack_tmp_i[pl.ds(r_l, bm_v), :] = jnp.broadcast_to(rgid, (bm_v, 8))
            return jnp.int32(0)

        if n_st == 1:
            _pack_strip(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_st, _pack_strip, jnp.int32(0))

        v_t = pack_tmp_v[...].T
        i_t = pack_tmp_i[...].T
        p_sl = pl.ds(ch * 8, 8)
        send_val_buf[p_sl, :] = v_t
        send_id_buf[p_sl, :] = i_t
        recv_val_buf[me, p_sl, :] = v_t
        recv_id_buf[me, p_sl, :] = i_t

    _entry_barrier_wait(world_size)
    ag_ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        op_v = pltpu.make_async_remote_copy(
            send_val_buf,
            recv_val_buf.at[me],
            ag_send_sem.at[2 * (step - 1)],
            ag_val_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_i = pltpu.make_async_remote_copy(
            send_id_buf,
            recv_id_buf.at[me],
            ag_send_sem.at[2 * (step - 1) + 1],
            ag_id_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_v.start()
        op_i.start()
        ag_ops.append(op_v)
        ag_ops.append(op_i)

    pltpu.make_async_copy(
        recv_val_buf.at[pl.ds(0, world_size - 1)],
        recv_val_buf.at[pl.ds(0, world_size - 1)],
        ag_val_recv_sem,
    ).wait()
    pltpu.make_async_copy(
        recv_id_buf.at[pl.ds(0, world_size - 1)],
        recv_id_buf.at[pl.ds(0, world_size - 1)],
        ag_id_recv_sem,
    ).wait()

    for ch in range(n_pack_chunks):
        p_sl = pl.ds(ch * 8, 8)
        g_v = recv_val_buf[0, p_sl, :]
        g_i = recv_id_buf[0, p_sl, :]
        for r in range(1, world_size):
            rv = recv_val_buf[r, p_sl, :]
            ri = recv_id_buf[r, p_sl, :]
            take = (rv > g_v) | ((rv == g_v) & (ri < g_i))
            g_v = jnp.where(take, rv, g_v)
            g_i = jnp.where(take, ri, g_i)
        out_val_ref[p_sl, :] = g_v
        out_id_ref[p_sl, :] = g_i

    for op in ag_ops:
        op.wait_send()


def fp8_lm_head_argmax_jax(
    x,
    w8k,
    scale,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    logit_scale: float = 1.0,
    tn: int = 1024,
    collective_id: int = _DEFAULT_COLLECTIVE_ID,
):
    """Single-kernel fused FP8 LM head + global exact argmax (``k=1``) with zero HBM logits."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, h = x.shape
    wk_h, vp = w8k.shape
    if wk_h != h or vp % 512 != 0:
        raise ValueError(f"Expected w8k ({h}, 512*q), got {w8k.shape}")
    eff_tn = tn if (vp % tn == 0) else 512
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")

    b_pad = _pad_batch_pow2(tokens)
    bm_mxu = min(b_pad, 128)
    pack_rows = (b_pad // 128) * 8 if b_pad >= 128 else 8
    x_in = x.astype(jnp.bfloat16)
    if b_pad > tokens:
        x_in = jnp.pad(x_in, ((0, b_pad - tokens), (0, 0)))

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=hbm)

    packed_val, packed_id = pl.pallas_call(
        functools.partial(
            _fp8_lm_head_argmax_kernel,
            world_size=world_size,
            local_width=local_width,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            logit_scale=float(logit_scale),
            tn=eff_tn,
            bm_mxu=bm_mxu,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((pack_rows, 128), jnp.float32),
            jax.ShapeDtypeStruct((pack_rows, 128), jnp.int32),
        ),
        in_specs=[v_spec, h_spec, v_spec],
        out_specs=(v_spec, v_spec),
        scratch_shapes=[
            vmem((2, h, eff_tn), w8k.dtype),
            vmem((bm_mxu, 512), jnp.float32),
            vmem((b_pad, 128), jnp.float32),
            vmem((b_pad, 128), jnp.int32),
            vmem((128, 8), jnp.float32),
            vmem((128, 8), jnp.int32),
            vmem((pack_rows, 128), jnp.float32),
            vmem((pack_rows, 128), jnp.int32),
            vmem((world_size, pack_rows, 128), jnp.float32),
            vmem((world_size, pack_rows, 128), jnp.int32),
            sem((2,)),
            sem((2 * (world_size - 1),)),
            sem,
            sem,
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_fp8_lm_head_argmax_b{b_pad}_tp{world_size}_tn{eff_tn}",
    )(x_in, w8k, scale.reshape(1, vp).astype(jnp.float32))

    if pack_rows == 8:
        out_v = packed_val[0, :tokens].reshape(tokens, 1)
        out_i = packed_id[0, :tokens].reshape(tokens, 1)
    else:
        n_chunks = pack_rows // 8
        out_v = jnp.concatenate(
            [packed_val[ch * 8, :] for ch in range(n_chunks)], axis=0
        )[:tokens].reshape(tokens, 1)
        out_i = jnp.concatenate(
            [packed_id[ch * 8, :] for ch in range(n_chunks)], axis=0
        )[:tokens].reshape(tokens, 1)
    return out_v, out_i


# =============================================================================
# 3. Single-Kernel Fused `FP8 LM_HEAD + Vertical Bitonic Top-64 + 1-Hop VMEM AG`
# =============================================================================
def _fp8_lm_head_fused_topk64_kernel(
    x_ref,
    w8k_hbm,
    scale_ref,
    out_val_ref,
    out_id_ref,
    w_buf,
    tile_buf,
    blk_t_buf,
    blk_i_buf,
    work_v_buf,
    work_i_buf,
    top_v_buf,
    top_i_buf,
    arg_v_buf,
    arg_i_buf,
    send_v_buf,
    send_i_buf,
    recv_v_buf,
    recv_i_buf,
    w_sem,
    ag_send_sem,
    ag_val_recv_sem,
    ag_id_recv_sem,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    logit_scale: float,
    tn: int,
    bm_mxu: int,
    exact_tie_break: bool,
    use_xor_butterfly: bool = False,
    use_pack_fold: bool = True,
) -> None:
    """Zero-spill 1-kernel fused FP8 LM head + quad-decomposed bitonic Top-64 + VMEM AG.

    - ``[B, V_local]`` logits NEVER touch HBM.
    - Full ``bm_mxu = min(b_pad, 128)`` MXU utilization overlapped with double-buffered HBM DMA.
    - For ``b_pad <= 64``, Wave-19 batch-lane folding packs ``128 // bm_mxu`` 128-col chunks into
      each ``(128, 128)`` transpose and folds lanes across ``bm_mxu`` once at the end.
    - Quad-decomposed vertical bitonic Top-64 keeps ``<= 12`` live VREGs (``0 / 0`` spills).
    - Compact ``(n_m * 64, 128)`` VMEM all-gather + in-VMEM global rank merge returns final
      descending global Top-64 directly (zero post-kernel XLA ``top_k``!).
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    b_pad, _ = x_ref.shape
    vp = w8k_hbm.shape[1]
    n_full_tiles = vp // tn
    has_tail_512 = (vp % tn) == 512
    n_m = b_pad // bm_mxu
    added_width = local_width - org_vocab_padded_count
    n_grp = (128 // bm_mxu) if (use_pack_fold and bm_mxu < 128 and n_m == 1) else 1

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier_signal(me, world_size)

    org_count = jnp.clip(
        jnp.int32(org_vocab_size) - me * jnp.int32(org_vocab_padded_count),
        jnp.int32(0),
        jnp.int32(org_vocab_padded_count),
    )
    added_start = jnp.int32(org_vocab_size) + me * jnp.int32(added_width)
    added_count = jnp.clip(
        jnp.int32(global_vocab_size) - added_start,
        jnp.int32(0),
        jnp.int32(added_width),
    )

    def _start_w(slot: int, tile_idx, cols_w: int = tn):
        c0 = pl.multiple_of(tile_idx * tn, tn)
        if cols_w == tn:
            pltpu.make_async_copy(
                w8k_hbm.at[:, pl.ds(c0, tn)],
                w_buf.at[slot],
                w_sem.at[slot],
            ).start()
        else:
            pltpu.make_async_copy(
                w8k_hbm.at[:, pl.ds(c0, cols_w)],
                w_buf.at[slot, :, pl.ds(0, cols_w)],
                w_sem.at[slot],
            ).start()

    def _wait_w(slot: int, cols_w: int = tn):
        if cols_w == tn:
            pltpu.make_async_copy(w_buf.at[slot], w_buf.at[slot], w_sem.at[slot]).wait()
        else:
            pltpu.make_async_copy(
                w_buf.at[slot, :, pl.ds(0, cols_w)],
                w_buf.at[slot, :, pl.ds(0, cols_w)],
                w_sem.at[slot],
            ).wait()

    if n_full_tiles >= 1:
        _start_w(0, jnp.int32(0), tn)
    elif has_tail_512:
        _start_w(0, jnp.int32(0), 512)
    if n_full_tiles >= 2:
        _start_w(1, jnp.int32(1), tn)
    elif n_full_tiles == 1 and has_tail_512:
        _start_w(1, jnp.int32(1), 512)

    neg_inf_8 = jnp.full((8, 128), -jnp.inf, dtype=jnp.float32)
    max_i32_8 = jnp.full((8, 128), 0x7FFFFFFF, dtype=jnp.int32)

    for im in range(n_m):
        mb = im * 64
        for k in range(8):
            sl = pl.ds(mb + k * 8, 8)
            top_v_buf[sl, :] = neg_inf_8
            top_i_buf[sl, :] = max_i32_8
        arg_v_buf[pl.ds(im * 8, 8), :] = neg_inf_8
        arg_i_buf[pl.ds(im * 8, 8), :] = max_i32_8

    if n_grp > 1:
        bm_shift = bm_mxu.bit_length() - 1
        grp_col_off = lax.shift_left(
            lax.shift_right_logical(lax.broadcasted_iota(jnp.int32, (8, 128), 1), jnp.int32(bm_shift)),
            jnp.int32(7),
        )
    else:
        grp_col_off = jnp.int32(0)

    def _merge_128_folded(c_base_val) -> None:
        for hh in range(2):
            h_off = pl.multiple_of(hh * 64, 64)
            _v64_quad_stream_merge_64(
                blk_t_buf,
                h_off,
                c_base_val + grp_col_off + jnp.int32(h_off),
                top_v_buf,
                top_i_buf,
                0,
                work_v_buf,
                work_i_buf,
                arg_v_buf,
                arg_i_buf,
                0,
                org_count=org_count,
                added_count=added_count,
                org_vocab_padded_count=org_vocab_padded_count,
                exact_tie_break=exact_tie_break,
            )

    def _process_tile(tile_idx, slot: int, has_next_prefetch: bool, cols_w: int = tn) -> None:
        _wait_w(slot, cols_w)
        c_tile = pl.multiple_of(tile_idx * tn, tn)
        cur_subs = cols_w // 512

        if n_grp == 8 and cols_w == 1024:
            s0 = scale_ref[0:1, pl.ds(c_tile, 512)]
            s1 = scale_ref[0:1, pl.ds(c_tile + 512, 512)]
            if logit_scale != 1.0:
                s0 = s0 * jnp.float32(logit_scale)
                s1 = s1 * jnp.float32(logit_scale)
            a0 = jnp.dot(
                x_ref[...], w_buf[slot, :, pl.ds(0, 512)], preferred_element_type=jnp.float32
            ) * s0
            a1 = jnp.dot(
                x_ref[...], w_buf[slot, :, pl.ds(512, 512)], preferred_element_type=jnp.float32
            ) * s1
            stacked = jnp.concatenate(
                [a0[:, pl.ds(q * 128, 128)] for q in range(4)]
                + [a1[:, pl.ds(q * 128, 128)] for q in range(4)],
                axis=0,
            )
            blk_t_buf[...] = stacked.T
            if has_next_prefetch:
                @pl.when(tile_idx + 2 < n_full_tiles)
                def _pf():
                    _start_w(slot, tile_idx + 2, tn)
                if has_tail_512:
                    @pl.when(tile_idx + 2 == n_full_tiles)
                    def _pf_tail():
                        _start_w(slot, tile_idx + 2, 512)
            _merge_128_folded(c_tile)
            return

        def _sub_body(isub, _):
            c_sub = pl.multiple_of(isub * 512, 512)
            c_base = c_tile + c_sub
            s_vec = scale_ref[0:1, pl.ds(c_base, 512)]
            if logit_scale != 1.0:
                s_vec = s_vec * jnp.float32(logit_scale)

            def _m_step(im, _):
                r_m = pl.multiple_of(im * bm_mxu, bm_mxu)
                m_base = pl.multiple_of(im * 64, 64)
                a_base = pl.multiple_of(im * 8, 8)
                rs_m = pl.ds(r_m, bm_mxu)
                acc = jnp.dot(
                    x_ref[rs_m, :],
                    w_buf[slot, :, pl.ds(c_sub, 512)],
                    preferred_element_type=jnp.float32,
                )
                tile_buf[...] = acc * s_vec

                if n_grp > 1:
                    n_packs = 1 if n_grp >= 4 else 2
                    cols_per_pk = n_grp * 128 if n_grp <= 4 else 512
                    for pk in range(n_packs):
                        c_pk = c_base + pk * cols_per_pk
                        if n_grp == 8:
                            st = jnp.concatenate(
                                [tile_buf[:, pl.ds(q * 128, 128)] for q in range(4)], axis=0
                            )
                            st = jnp.pad(st, ((0, 64), (0, 0)), constant_values=-jnp.inf)
                        else:
                            st = jnp.concatenate(
                                [
                                    tile_buf[:, pl.ds(pk * cols_per_pk + q * 128, 128)]
                                    for q in range(n_grp)
                                ],
                                axis=0,
                            )
                        blk_t_buf[...] = st.T
                        _merge_128_folded(c_pk)
                else:
                    def _q_body(q, _):
                        q_off = pl.multiple_of(q * 128, 128)
                        blk = tile_buf[:, pl.ds(q_off, 128)]
                        if bm_mxu < 128:
                            blk = jnp.pad(
                                blk, ((0, 128 - bm_mxu), (0, 0)), constant_values=-jnp.inf
                            )
                        blk_t_buf[...] = blk.T

                        def _h_body(h, _):
                            h_off = pl.multiple_of(h * 64, 64)
                            c0 = c_base + q_off + h_off
                            _v64_quad_stream_merge_64(
                                blk_t_buf,
                                h_off,
                                c0,
                                top_v_buf,
                                top_i_buf,
                                m_base,
                                work_v_buf,
                                work_i_buf,
                                arg_v_buf,
                                arg_i_buf,
                                a_base,
                                org_count=org_count,
                                added_count=added_count,
                                org_vocab_padded_count=org_vocab_padded_count,
                                exact_tie_break=exact_tie_break,
                            )
                            return jnp.int32(0)

                        lax.fori_loop(0, 2, _h_body, jnp.int32(0))
                        return jnp.int32(0)

                    lax.fori_loop(0, 4, _q_body, jnp.int32(0))
                return jnp.int32(0)

            if n_m == 1:
                _m_step(jnp.int32(0), jnp.int32(0))
            else:
                lax.fori_loop(0, n_m, _m_step, jnp.int32(0))
            return jnp.int32(0)

        if cur_subs == 1:
            _sub_body(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, cur_subs, _sub_body, jnp.int32(0))

        if has_next_prefetch:
            @pl.when(tile_idx + 2 < n_full_tiles)
            def _pf():
                _start_w(slot, tile_idx + 2, tn)
            if has_tail_512:
                @pl.when(tile_idx + 2 == n_full_tiles)
                def _pf_tail():
                    _start_w(slot, tile_idx + 2, 512)

    n_pairs = n_full_tiles // 2
    if n_pairs == 1:
        _process_tile(
            jnp.int32(0), 0, has_next_prefetch=(n_full_tiles > 2 or has_tail_512)
        )
        _process_tile(
            jnp.int32(1),
            1,
            has_next_prefetch=(n_full_tiles > 3 or (n_full_tiles == 2 and has_tail_512)),
        )
    elif n_pairs > 1:
        def _pair_body(p, _):
            t0 = p * 2
            _process_tile(t0, 0, has_next_prefetch=True)
            _process_tile(t0 + 1, 1, has_next_prefetch=True)
            return jnp.int32(0)

        lax.fori_loop(0, n_pairs, _pair_body, jnp.int32(0))
    if n_full_tiles % 2 == 1:
        _process_tile(jnp.int32(n_full_tiles - 1), 0, has_next_prefetch=has_tail_512)
    if has_tail_512:
        _process_tile(
            jnp.int32(n_full_tiles),
            1 if (n_full_tiles % 2 == 1) else 0,
            has_next_prefetch=False,
            cols_w=512,
        )

    # Convert local indices -> global IDs, fold batch lanes (n_grp > 1), enforce exact min-ID col 0!
    for im in range(n_m):
        mb = im * 64
        for k in range(8):
            sl = pl.ds(mb + k * 8, 8)
            li = top_i_buf[sl, :]
            top_i_buf[sl, :] = _map_local_to_global_id(
                li,
                me=me,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                added_width=added_width,
            )
        if exact_tie_break:
            av = arg_v_buf[pl.ds(im * 8, 8), :]
            ai_g = _map_local_to_global_id(
                arg_i_buf[pl.ds(im * 8, 8), :],
                me=me,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                added_width=added_width,
            )
            av, exact_gid = _v64_sublane_exact_argmax(av, ai_g)
            arg_v_buf[pl.ds(im * 8, 8), :] = av
            arg_i_buf[pl.ds(im * 8, 8), :] = exact_gid

        if n_grp > 1:
            sub_perm = jnp.bitwise_xor(
                lax.broadcasted_iota(jnp.int32, (8, 128), 0), jnp.int32(7)
            )
            sh = 64
            while sh >= bm_mxu:
                for k in range(8):
                    sl_rev = pl.ds((7 - k) * 8, 8)
                    sl_t = pl.ds(k * 8, 8)
                    sl_w = pl.ds(k * 8, 8)
                    rv_rev = pltpu.roll(top_v_buf[sl_rev, :], 128 - sh, axis=1)
                    ri_rev = pltpu.roll(top_i_buf[sl_rev, :], 128 - sh, axis=1)
                    rv_asc = jnp.take_along_axis(rv_rev, sub_perm, axis=0)
                    ri_asc = jnp.take_along_axis(ri_rev, sub_perm, axis=0)
                    cv = top_v_buf[sl_t, :]
                    ci = top_i_buf[sl_t, :]
                    take = rv_asc > cv
                    work_v_buf[sl_w, :] = jnp.where(take, rv_asc, cv)
                    work_i_buf[sl_w, :] = jnp.where(take, ri_asc, ci)
                _v64_quad_merge_work_to_top_desc(
                    work_v_buf, work_i_buf, top_v_buf, top_i_buf, 0
                )
                if exact_tie_break:
                    av_p = pltpu.roll(arg_v_buf[pl.ds(0, 8), :], 128 - sh, axis=1)
                    ai_p = pltpu.roll(arg_i_buf[pl.ds(0, 8), :], 128 - sh, axis=1)
                    cur_av = arg_v_buf[pl.ds(0, 8), :]
                    cur_ai = arg_i_buf[pl.ds(0, 8), :]
                    take_a = (av_p > cur_av) | ((av_p == cur_av) & (ai_p < cur_ai))
                    arg_v_buf[pl.ds(0, 8), :] = jnp.where(take_a, av_p, cur_av)
                    arg_i_buf[pl.ds(0, 8), :] = jnp.where(take_a, ai_p, cur_ai)
                sh //= 2

        if exact_tie_break:
            _v64_inject_exact_col0(top_i_buf, mb, arg_i_buf[pl.ds(im * 8, 8), :])
    if use_xor_butterfly:
        n_hops = 3 if world_size == 8 else 4
        z_idx = jnp.zeros((8, 128), dtype=jnp.int32)
        _entry_barrier_wait(world_size)
        for hop in range(n_hops):
            peer = me ^ jnp.int32(1 << hop)
            me_lower = me < peer
            send_v_buf[...] = top_v_buf[...]
            send_i_buf[...] = top_i_buf[...]
            op_v = pltpu.make_async_remote_copy(
                send_v_buf,
                recv_v_buf.at[hop],
                ag_send_sem.at[2 * hop],
                ag_val_recv_sem.at[hop],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_i = pltpu.make_async_remote_copy(
                send_i_buf,
                recv_i_buf.at[hop],
                ag_send_sem.at[2 * hop + 1],
                ag_id_recv_sem.at[hop],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_v.start()
            op_i.start()
            op_v.wait()
            op_i.wait()
            for im in range(n_m):
                mb = im * 64
                if exact_tie_break:
                    gm = jnp.take_along_axis(top_v_buf[pl.ds(mb, 8), :], z_idx, axis=0)
                    gi = jnp.take_along_axis(top_i_buf[pl.ds(mb, 8), :], z_idx, axis=0)
                    pm = jnp.take_along_axis(recv_v_buf[hop, pl.ds(mb, 8), :], z_idx, axis=0)
                    pi = jnp.take_along_axis(recv_i_buf[hop, pl.ds(mb, 8), :], z_idx, axis=0)
                    take = (pm > gm) | ((pm == gm) & (pi < gi))
                    ggid = jnp.where(take, pi, gi)
                _v64_quad_merge_butterfly_hop_desc(
                    recv_v_buf, recv_i_buf, hop, me_lower,
                    top_v_buf, top_i_buf, mb, work_v_buf, work_i_buf,
                )
                if exact_tie_break:
                    _v64_inject_exact_col0(top_i_buf, mb, ggid)

        for im in range(n_m):
            mb = im * 64
            blk_t_buf[pl.ds(0, 64), :] = top_v_buf[pl.ds(mb, 64), :]
            blk_t_buf[pl.ds(64, 64), :] = jnp.full((64, 128), -jnp.inf, dtype=jnp.float32)
            out_v = blk_t_buf[...].T
            out_val_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_v[:bm_mxu, :]

            blk_i_buf[pl.ds(0, 64), :] = top_i_buf[pl.ds(mb, 64), :]
            blk_i_buf[pl.ds(64, 64), :] = jnp.full((64, 128), 0x7FFFFFFF, dtype=jnp.int32)
            out_i = blk_i_buf[...].T
            out_id_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_i[:bm_mxu, :]
        return

    send_v_buf[...] = top_v_buf[...]
    send_i_buf[...] = top_i_buf[...]
    recv_v_buf[me, :, :] = top_v_buf[...]
    recv_i_buf[me, :, :] = top_i_buf[...]

    # 1-hop cyclic VMEM all-gather of compact (n_m * 64, 128) sorted descending candidates!
    _entry_barrier_wait(world_size)
    ag_ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        op_v = pltpu.make_async_remote_copy(
            send_v_buf,
            recv_v_buf.at[me],
            ag_send_sem.at[2 * (step - 1)],
            ag_val_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_i = pltpu.make_async_remote_copy(
            send_i_buf,
            recv_i_buf.at[me],
            ag_send_sem.at[2 * (step - 1) + 1],
            ag_id_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_v.start()
        op_i.start()
        ag_ops.append(op_v)
        ag_ops.append(op_i)

    pltpu.make_async_copy(
        recv_v_buf.at[pl.ds(0, world_size - 1)],
        recv_v_buf.at[pl.ds(0, world_size - 1)],
        ag_val_recv_sem,
    ).wait()
    pltpu.make_async_copy(
        recv_i_buf.at[pl.ds(0, world_size - 1)],
        recv_i_buf.at[pl.ds(0, world_size - 1)],
        ag_id_recv_sem,
    ).wait()

    # Canonical rank-0..W-1 in-VMEM zero-spill quad merge -> exact global Top-64 on all ranks!
    z_idx = jnp.zeros((8, 128), dtype=jnp.int32)
    for im in range(n_m):
        mb = im * 64
        for k in range(8):
            sl = pl.ds(mb + k * 8, 8)
            top_v_buf[sl, :] = recv_v_buf[0, sl, :]
            top_i_buf[sl, :] = recv_i_buf[0, sl, :]
        gmax = jnp.take_along_axis(recv_v_buf[0, pl.ds(mb, 8), :], z_idx, axis=0)
        ggid = jnp.take_along_axis(recv_i_buf[0, pl.ds(mb, 8), :], z_idx, axis=0)

        def _r_merge(r, carry):
            gm, gi = carry
            if exact_tie_break:
                r0_v = jnp.take_along_axis(recv_v_buf[r, pl.ds(mb, 8), :], z_idx, axis=0)
                r0_i = jnp.take_along_axis(recv_i_buf[r, pl.ds(mb, 8), :], z_idx, axis=0)
                take = (r0_v > gm) | ((r0_v == gm) & (r0_i < gi))
                gm = jnp.where(take, r0_v, gm)
                gi = jnp.where(take, r0_i, gi)
            _v64_quad_merge_rank_desc(
                recv_v_buf, recv_i_buf, r, top_v_buf, top_i_buf, mb, work_v_buf, work_i_buf
            )
            return (gm, gi)

        gmax, ggid = lax.fori_loop(1, world_size, _r_merge, (gmax, ggid))
        if exact_tie_break:
            _v64_inject_exact_col0(top_i_buf, mb, ggid)

        blk_t_buf[pl.ds(0, 64), :] = top_v_buf[pl.ds(mb, 64), :]
        blk_t_buf[pl.ds(64, 64), :] = jnp.full((64, 128), -jnp.inf, dtype=jnp.float32)
        out_v = blk_t_buf[...].T
        out_val_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_v[:bm_mxu, :]

        blk_i_buf[pl.ds(0, 64), :] = top_i_buf[pl.ds(mb, 64), :]
        blk_i_buf[pl.ds(64, 64), :] = jnp.full((64, 128), 0x7FFFFFFF, dtype=jnp.int32)
        out_i = blk_i_buf[...].T
        out_id_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_i[:bm_mxu, :]

    for op in ag_ops:
        op.wait_send()


def fp8_lm_head_fused_topk64_jax(
    x,
    w8k,
    scale,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int = MAX_DISTRIBUTED_TOP_K,
    logit_scale: float = 1.0,
    tn: int = 512,
    exact_tie_break: bool = True,
    use_xor_butterfly: bool = False,
    use_pack_fold: bool = True,
    collective_id: int = _DEFAULT_COLLECTIVE_ID + 2,
):
    """1-kernel fused FP8 LM head + exact global Top-K (``1 <= k <= 64``) with zero HBM logits."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, h = x.shape
    wk_h, vp = w8k.shape
    if wk_h != h or vp % 512 != 0:
        raise ValueError(f"Expected w8k ({h}, 512*q), got {w8k.shape}")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    if not 1 <= k <= MAX_DISTRIBUTED_TOP_K:
        raise ValueError(f"Expected 1 <= k <= 64, got {k}")
    eff_tn = tn if (vp % tn == 0 or (vp >= tn and tn == 1024 and vp % 512 == 0)) else 512
    n_hops = 3 if world_size == 8 else 4
    n_recv = n_hops if use_xor_butterfly else world_size
    n_send_sems = (2 * n_hops) if use_xor_butterfly else (2 * (world_size - 1))

    b_pad = _pad_batch_pow2(tokens)
    bm_mxu = min(b_pad, 128)
    n_m = b_pad // bm_mxu
    eff_pack_fold = bool(use_pack_fold and bm_mxu < 128 and n_m == 1)
    x_in = x.astype(jnp.bfloat16)
    if b_pad > tokens:
        x_in = jnp.pad(x_in, ((0, b_pad - tokens), (0, 0)))

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=hbm)

    out_val, out_id = pl.pallas_call(
        functools.partial(
            _fp8_lm_head_fused_topk64_kernel,
            world_size=world_size,
            local_width=local_width,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            logit_scale=float(logit_scale),
            tn=eff_tn,
            bm_mxu=bm_mxu,
            exact_tie_break=bool(exact_tie_break),
            use_xor_butterfly=bool(use_xor_butterfly),
            use_pack_fold=eff_pack_fold,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((b_pad, 128), jnp.float32),
            jax.ShapeDtypeStruct((b_pad, 128), jnp.int32),
        ),
        in_specs=[v_spec, h_spec, v_spec],
        out_specs=(v_spec, v_spec),
        scratch_shapes=[
            vmem((2, h, eff_tn), w8k.dtype),
            vmem((bm_mxu, 512), jnp.float32),
            vmem((128, 128), jnp.float32),
            vmem((128, 128), jnp.int32),
            vmem((64, 128), jnp.float32),
            vmem((64, 128), jnp.int32),
            vmem((n_m * 64, 128), jnp.float32),
            vmem((n_m * 64, 128), jnp.int32),
            vmem((n_m * 8, 128), jnp.float32),
            vmem((n_m * 8, 128), jnp.int32),
            vmem((n_m * 64, 128), jnp.float32),
            vmem((n_m * 64, 128), jnp.int32),
            vmem((n_recv, n_m * 64, 128), jnp.float32),
            vmem((n_recv, n_m * 64, 128), jnp.int32),
            sem((2,)),
            sem((n_send_sems,)),
            sem((n_hops,)) if use_xor_butterfly else sem,
            sem((n_hops,)) if use_xor_butterfly else sem,
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024**2,
            collective_id=collective_id + (8 if use_xor_butterfly else 0),
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_fp8_lm_head_fused_topk64_b{b_pad}_tp{world_size}_tn{eff_tn}"
            f"{'_pf' if eff_pack_fold else ''}{'_xbf' if use_xor_butterfly else ''}"
        ),
    )(x_in, w8k, scale.reshape(1, vp).astype(jnp.float32))

    return out_val[:tokens, :k], out_id[:tokens, :k]


# =============================================================================
# 4. Lossless Standalone `Vertical Bitonic Top-64 + 1-Hop Cyclic VMEM AG`
# =============================================================================
def _distributed_topk_vmem_ag_kernel(
    logits_hbm,
    out_val_ref,
    out_id_ref,
    l_buf,
    blk_t_buf,
    blk_i_buf,
    work_v_buf,
    work_i_buf,
    top_v_buf,
    top_i_buf,
    arg_v_buf,
    arg_i_buf,
    send_v_buf,
    send_i_buf,
    recv_v_buf,
    recv_i_buf,
    l_sem,
    ag_send_sem,
    ag_val_recv_sem,
    ag_id_recv_sem,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    bm_mxu: int,
    exact_tie_break: bool,
    use_xor_butterfly: bool = False,
) -> None:
    """Lossless zero-spill standalone Top-64 + 1-hop VMEM AG replacing 3-hop HBM ring."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    b_pad, vp = logits_hbm.shape
    n_tiles = vp // 512
    n_m = b_pad // bm_mxu
    added_width = local_width - org_vocab_padded_count

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier_signal(me, world_size)

    org_count = jnp.clip(
        jnp.int32(org_vocab_size) - me * jnp.int32(org_vocab_padded_count),
        jnp.int32(0),
        jnp.int32(org_vocab_padded_count),
    )
    added_start = jnp.int32(org_vocab_size) + me * jnp.int32(added_width)
    added_count = jnp.clip(
        jnp.int32(global_vocab_size) - added_start,
        jnp.int32(0),
        jnp.int32(added_width),
    )

    def _start_l(slot: int, tile_idx):
        c0 = pl.multiple_of(tile_idx * 512, 512)
        pltpu.make_async_copy(
            logits_hbm.at[:, pl.ds(c0, 512)],
            l_buf.at[slot],
            l_sem.at[slot],
        ).start()

    def _wait_l(slot: int):
        pltpu.make_async_copy(l_buf.at[slot], l_buf.at[slot], l_sem.at[slot]).wait()

    _start_l(0, jnp.int32(0))
    if n_tiles >= 2:
        _start_l(1, jnp.int32(1))

    neg_inf_8 = jnp.full((8, 128), -jnp.inf, dtype=jnp.float32)
    max_i32_8 = jnp.full((8, 128), 0x7FFFFFFF, dtype=jnp.int32)

    for im in range(n_m):
        mb = im * 64
        for k in range(8):
            sl = pl.ds(mb + k * 8, 8)
            top_v_buf[sl, :] = neg_inf_8
            top_i_buf[sl, :] = max_i32_8
        arg_v_buf[pl.ds(im * 8, 8), :] = neg_inf_8
        arg_i_buf[pl.ds(im * 8, 8), :] = max_i32_8

    def _process_tile(tile_idx, slot: int, has_next_prefetch: bool) -> None:
        _wait_l(slot)
        c_base = pl.multiple_of(tile_idx * 512, 512)

        def _m_step(im, _):
            r_m = pl.multiple_of(im * bm_mxu, bm_mxu)
            m_base = pl.multiple_of(im * 64, 64)
            a_base = pl.multiple_of(im * 8, 8)
            rs_m = pl.ds(r_m, bm_mxu)

            def _q_body(q, _):
                q_off = pl.multiple_of(q * 128, 128)
                blk = l_buf[slot, rs_m, pl.ds(q_off, 128)].astype(jnp.float32)
                if bm_mxu < 128:
                    blk = jnp.pad(blk, ((0, 128 - bm_mxu), (0, 0)), constant_values=-jnp.inf)
                blk_t_buf[...] = blk.T

                def _h_body(h, _):
                    h_off = pl.multiple_of(h * 64, 64)
                    c0 = c_base + q_off + h_off
                    _v64_quad_stream_merge_64(
                        blk_t_buf,
                        h_off,
                        c0,
                        top_v_buf,
                        top_i_buf,
                        m_base,
                        work_v_buf,
                        work_i_buf,
                        arg_v_buf,
                        arg_i_buf,
                        a_base,
                        org_count=org_count,
                        added_count=added_count,
                        org_vocab_padded_count=org_vocab_padded_count,
                        exact_tie_break=exact_tie_break,
                    )
                    return jnp.int32(0)

                lax.fori_loop(0, 2, _h_body, jnp.int32(0))
                return jnp.int32(0)

            lax.fori_loop(0, 4, _q_body, jnp.int32(0))
            return jnp.int32(0)

        if n_m == 1:
            _m_step(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_m, _m_step, jnp.int32(0))

        if has_next_prefetch:
            @pl.when(tile_idx + 2 < n_tiles)
            def _pf():
                _start_l(slot, tile_idx + 2)

    n_pairs = n_tiles // 2
    if n_pairs == 1:
        _process_tile(jnp.int32(0), 0, has_next_prefetch=(n_tiles > 2))
        _process_tile(jnp.int32(1), 1, has_next_prefetch=(n_tiles > 3))
    elif n_pairs > 1:
        def _pair_body(p, _):
            t0 = p * 2
            _process_tile(t0, 0, has_next_prefetch=True)
            _process_tile(t0 + 1, 1, has_next_prefetch=True)
            return jnp.int32(0)

        lax.fori_loop(0, n_pairs, _pair_body, jnp.int32(0))
    if n_tiles % 2 == 1:
        _process_tile(jnp.int32(n_tiles - 1), 0, has_next_prefetch=False)

    for im in range(n_m):
        mb = im * 64
        for k in range(8):
            sl = pl.ds(mb + k * 8, 8)
            li = top_i_buf[sl, :]
            top_i_buf[sl, :] = _map_local_to_global_id(
                li,
                me=me,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                added_width=added_width,
            )
        if exact_tie_break:
            av = arg_v_buf[pl.ds(im * 8, 8), :]
            ai_g = _map_local_to_global_id(
                arg_i_buf[pl.ds(im * 8, 8), :],
                me=me,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                added_width=added_width,
            )
            _, exact_gid = _v64_sublane_exact_argmax(av, ai_g)
            _v64_inject_exact_col0(top_i_buf, mb, exact_gid)
    if use_xor_butterfly:
        n_hops = 3 if world_size == 8 else 4
        z_idx = jnp.zeros((8, 128), dtype=jnp.int32)
        _entry_barrier_wait(world_size)
        for hop in range(n_hops):
            peer = me ^ jnp.int32(1 << hop)
            me_lower = me < peer
            send_v_buf[...] = top_v_buf[...]
            send_i_buf[...] = top_i_buf[...]
            op_v = pltpu.make_async_remote_copy(
                send_v_buf,
                recv_v_buf.at[hop],
                ag_send_sem.at[2 * hop],
                ag_val_recv_sem.at[hop],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_i = pltpu.make_async_remote_copy(
                send_i_buf,
                recv_i_buf.at[hop],
                ag_send_sem.at[2 * hop + 1],
                ag_id_recv_sem.at[hop],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_v.start()
            op_i.start()
            op_v.wait()
            op_i.wait()
            for im in range(n_m):
                mb = im * 64
                if exact_tie_break:
                    gm = jnp.take_along_axis(top_v_buf[pl.ds(mb, 8), :], z_idx, axis=0)
                    gi = jnp.take_along_axis(top_i_buf[pl.ds(mb, 8), :], z_idx, axis=0)
                    pm = jnp.take_along_axis(recv_v_buf[hop, pl.ds(mb, 8), :], z_idx, axis=0)
                    pi = jnp.take_along_axis(recv_i_buf[hop, pl.ds(mb, 8), :], z_idx, axis=0)
                    take = (pm > gm) | ((pm == gm) & (pi < gi))
                    ggid = jnp.where(take, pi, gi)
                _v64_quad_merge_butterfly_hop_desc(
                    recv_v_buf, recv_i_buf, hop, me_lower,
                    top_v_buf, top_i_buf, mb, work_v_buf, work_i_buf,
                )
                if exact_tie_break:
                    _v64_inject_exact_col0(top_i_buf, mb, ggid)

        for im in range(n_m):
            mb = im * 64
            blk_t_buf[pl.ds(0, 64), :] = top_v_buf[pl.ds(mb, 64), :]
            blk_t_buf[pl.ds(64, 64), :] = jnp.full((64, 128), -jnp.inf, dtype=jnp.float32)
            out_v = blk_t_buf[...].T
            out_val_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_v[:bm_mxu, :]

            blk_i_buf[pl.ds(0, 64), :] = top_i_buf[pl.ds(mb, 64), :]
            blk_i_buf[pl.ds(64, 64), :] = jnp.full((64, 128), 0x7FFFFFFF, dtype=jnp.int32)
            out_i = blk_i_buf[...].T
            out_id_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_i[:bm_mxu, :]
        return

    send_v_buf[...] = top_v_buf[...]
    send_i_buf[...] = top_i_buf[...]
    recv_v_buf[me, :, :] = top_v_buf[...]
    recv_i_buf[me, :, :] = top_i_buf[...]

    _entry_barrier_wait(world_size)
    ag_ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        op_v = pltpu.make_async_remote_copy(
            send_v_buf,
            recv_v_buf.at[me],
            ag_send_sem.at[2 * (step - 1)],
            ag_val_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_i = pltpu.make_async_remote_copy(
            send_i_buf,
            recv_i_buf.at[me],
            ag_send_sem.at[2 * (step - 1) + 1],
            ag_id_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_v.start()
        op_i.start()
        ag_ops.append(op_v)
        ag_ops.append(op_i)

    pltpu.make_async_copy(
        recv_v_buf.at[pl.ds(0, world_size - 1)],
        recv_v_buf.at[pl.ds(0, world_size - 1)],
        ag_val_recv_sem,
    ).wait()
    pltpu.make_async_copy(
        recv_i_buf.at[pl.ds(0, world_size - 1)],
        recv_i_buf.at[pl.ds(0, world_size - 1)],
        ag_id_recv_sem,
    ).wait()

    z_idx = jnp.zeros((8, 128), dtype=jnp.int32)
    for im in range(n_m):
        mb = im * 64
        for k in range(8):
            sl = pl.ds(mb + k * 8, 8)
            top_v_buf[sl, :] = recv_v_buf[0, sl, :]
            top_i_buf[sl, :] = recv_i_buf[0, sl, :]
        gmax = jnp.take_along_axis(recv_v_buf[0, pl.ds(mb, 8), :], z_idx, axis=0)
        ggid = jnp.take_along_axis(recv_i_buf[0, pl.ds(mb, 8), :], z_idx, axis=0)

        def _r_merge(r, carry):
            gm, gi = carry
            if exact_tie_break:
                r0_v = jnp.take_along_axis(recv_v_buf[r, pl.ds(mb, 8), :], z_idx, axis=0)
                r0_i = jnp.take_along_axis(recv_i_buf[r, pl.ds(mb, 8), :], z_idx, axis=0)
                take = (r0_v > gm) | ((r0_v == gm) & (r0_i < gi))
                gm = jnp.where(take, r0_v, gm)
                gi = jnp.where(take, r0_i, gi)
            _v64_quad_merge_rank_desc(
                recv_v_buf, recv_i_buf, r, top_v_buf, top_i_buf, mb, work_v_buf, work_i_buf
            )
            return (gm, gi)

        gmax, ggid = lax.fori_loop(1, world_size, _r_merge, (gmax, ggid))
        if exact_tie_break:
            _v64_inject_exact_col0(top_i_buf, mb, ggid)

        blk_t_buf[pl.ds(0, 64), :] = top_v_buf[pl.ds(mb, 64), :]
        blk_t_buf[pl.ds(64, 64), :] = jnp.full((64, 128), -jnp.inf, dtype=jnp.float32)
        out_v = blk_t_buf[...].T
        out_val_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_v[:bm_mxu, :]

        blk_i_buf[pl.ds(0, 64), :] = top_i_buf[pl.ds(mb, 64), :]
        blk_i_buf[pl.ds(64, 64), :] = jnp.full((64, 128), 0x7FFFFFFF, dtype=jnp.int32)
        out_i = blk_i_buf[...].T
        out_id_ref[pl.ds(im * bm_mxu, bm_mxu), :] = out_i[:bm_mxu, :]

    for op in ag_ops:
        op.wait_send()


def distributed_topk_vmem_ag_jax(
    local_logits,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int = MAX_DISTRIBUTED_TOP_K,
    exact_tie_break: bool = True,
    use_xor_butterfly: bool = False,
    collective_id: int = _DEFAULT_COLLECTIVE_ID + 3,
):
    """Lossless standalone distributed Top-K (``1 <= k <= 64``) via vertical bitonic + VMEM AG."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = local_logits.shape
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    if not 1 <= k <= MAX_DISTRIBUTED_TOP_K:
        raise ValueError(f"Expected 1 <= k <= 64, got {k}")

    b_pad = _pad_batch_pow2(tokens)
    bm_mxu = min(b_pad, 128)
    n_m = b_pad // bm_mxu
    n_hops = 3 if world_size == 8 else 4
    n_recv = n_hops if use_xor_butterfly else world_size
    n_send_sems = (2 * n_hops) if use_xor_butterfly else (2 * (world_size - 1))
    vp = _ceil_multiple(width, 512)
    l_in = local_logits
    if b_pad > tokens or vp > width:
        l_in = jnp.pad(
            l_in,
            ((0, b_pad - tokens), (0, vp - width)),
            constant_values=-jnp.inf,
        )

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=hbm)

    out_val, out_id = pl.pallas_call(
        functools.partial(
            _distributed_topk_vmem_ag_kernel,
            world_size=world_size,
            local_width=local_width,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            bm_mxu=bm_mxu,
            exact_tie_break=bool(exact_tie_break),
            use_xor_butterfly=bool(use_xor_butterfly),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((b_pad, 128), jnp.float32),
            jax.ShapeDtypeStruct((b_pad, 128), jnp.int32),
        ),
        in_specs=[h_spec],
        out_specs=(v_spec, v_spec),
        scratch_shapes=[
            vmem((2, b_pad, 512), l_in.dtype),
            vmem((128, 128), jnp.float32),
            vmem((128, 128), jnp.int32),
            vmem((64, 128), jnp.float32),
            vmem((64, 128), jnp.int32),
            vmem((n_m * 64, 128), jnp.float32),
            vmem((n_m * 64, 128), jnp.int32),
            vmem((n_m * 8, 128), jnp.float32),
            vmem((n_m * 8, 128), jnp.int32),
            vmem((n_m * 64, 128), jnp.float32),
            vmem((n_m * 64, 128), jnp.int32),
            vmem((n_recv, n_m * 64, 128), jnp.float32),
            vmem((n_recv, n_m * 64, 128), jnp.int32),
            sem((2,)),
            sem((n_send_sems,)),
            sem((n_hops,)) if use_xor_butterfly else sem,
            sem((n_hops,)) if use_xor_butterfly else sem,
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024**2,
            collective_id=collective_id + (8 if use_xor_butterfly else 0),
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_dist_topk_vmem_ag_b{b_pad}_tp{world_size}"
            f"{'_xbf' if use_xor_butterfly else ''}"
        ),
    )(l_in)

    return out_val[:tokens, :k], out_id[:tokens, :k]


# =============================================================================
# 5. Fast 1-Hop Cyclic VMEM All-Gather Helper (For k==1 XLA-Dot Hybrid Path)
# =============================================================================
def _vmem_ag_topk_pairs_kernel(
    in_val_ref,
    in_id_ref,
    out_val_ref,
    out_id_ref,
    send_sem,
    val_recv_sem,
    id_recv_sem,
    *,
    world_size: int,
) -> None:
    """1-hop cyclic VMEM all-gather of ``[B_pad, 128]`` ``(val, id)`` candidate pairs."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier_signal(me, world_size)
    out_val_ref[me, :, :] = in_val_ref[...]
    out_id_ref[me, :, :] = in_id_ref[...]
    _entry_barrier_wait(world_size)

    ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        op_v = pltpu.make_async_remote_copy(
            in_val_ref,
            out_val_ref.at[me],
            send_sem.at[2 * (step - 1)],
            val_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_i = pltpu.make_async_remote_copy(
            in_id_ref,
            out_id_ref.at[me],
            send_sem.at[2 * (step - 1) + 1],
            id_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_v.start()
        op_i.start()
        ops.append(op_v)
        ops.append(op_i)

    pltpu.make_async_copy(
        out_val_ref.at[pl.ds(0, world_size - 1)],
        out_val_ref.at[pl.ds(0, world_size - 1)],
        val_recv_sem,
    ).wait()
    pltpu.make_async_copy(
        out_id_ref.at[pl.ds(0, world_size - 1)],
        out_id_ref.at[pl.ds(0, world_size - 1)],
        id_recv_sem,
    ).wait()

    for op in ops:
        op.wait_send()


def vmem_all_gather_topk_candidates_jax(
    local_vals,
    local_ids,
    *,
    world_size: int,
    collective_id: int = _DEFAULT_COLLECTIVE_ID + 1,
):
    """Gather ``[B, K]`` (``K <= 128``) candidate ``(vals, ids)`` across TP ranks in 1 VMEM hop."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, k_in = local_vals.shape
    b_pad = _ceil_multiple(tokens, 8)
    v_in = jnp.pad(
        local_vals.astype(jnp.float32),
        ((0, b_pad - tokens), (0, 128 - k_in)),
        constant_values=-jnp.inf,
    )
    i_in = jnp.pad(
        local_ids.astype(jnp.int32),
        ((0, b_pad - tokens), (0, 128 - k_in)),
        constant_values=0x7FFFFFFF,
    )
    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)

    g_vals, g_ids = pl.pallas_call(
        functools.partial(_vmem_ag_topk_pairs_kernel, world_size=world_size),
        out_shape=(
            jax.ShapeDtypeStruct((world_size, b_pad, 128), jnp.float32),
            jax.ShapeDtypeStruct((world_size, b_pad, 128), jnp.int32),
        ),
        in_specs=[v_spec, v_spec],
        out_specs=(v_spec, v_spec),
        scratch_shapes=[sem((2 * (world_size - 1),)), sem, sem],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=24 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_vmem_ag_topk_b{b_pad}_tp{world_size}",
    )(v_in, i_in)

    g_vals = jnp.transpose(g_vals[:, :tokens, :k_in], (1, 0, 2)).reshape(tokens, world_size * k_in)
    g_ids = jnp.transpose(g_ids[:, :tokens, :k_in], (1, 0, 2)).reshape(tokens, world_size * k_in)
    return g_vals, g_ids


def distributed_top_k_fast_jax(
    local_logits,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int,
    exact_tie_break: bool = True,
):
    """Exact distributed Top-K (``k==1`` via XLA argmax + VMEM AG; ``k>1`` via vertical bitonic)."""
    import jax.numpy as jnp
    from jax import lax

    if k > 1 and local_logits.shape[0] <= 384:
        return distributed_topk_vmem_ag_jax(
            local_logits,
            world_size=world_size,
            local_width=local_width,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            k=k,
            exact_tie_break=exact_tie_break,
        )

    tokens, width = local_logits.shape
    me = lax.axis_index("model").astype(jnp.int32)
    added_width = local_width - org_vocab_padded_count
    org_count = jnp.clip(
        jnp.int32(org_vocab_size) - me * jnp.int32(org_vocab_padded_count),
        jnp.int32(0),
        jnp.int32(org_vocab_padded_count),
    )
    added_start = jnp.int32(org_vocab_size) + me * jnp.int32(added_width)
    added_count = jnp.clip(
        jnp.int32(global_vocab_size) - added_start,
        jnp.int32(0),
        jnp.int32(added_width),
    )
    pos = jnp.arange(width, dtype=jnp.int32)[None, :]
    add_p = pos - jnp.int32(org_vocab_padded_count)
    valid = (pos < org_count) | ((add_p >= 0) & (add_p < added_count))
    masked = jnp.where(valid, local_logits.astype(jnp.float32), -jnp.inf)

    l_val = jnp.max(masked, axis=-1, keepdims=True)
    l_pos = jnp.argmax(masked, axis=-1).astype(jnp.int32)[:, None]
    l_gid = _map_local_to_global_id(
        l_pos,
        me=me,
        local_width=local_width,
        org_vocab_size=org_vocab_size,
        org_vocab_padded_count=org_vocab_padded_count,
        added_width=added_width,
    )
    g_vals, g_ids = vmem_all_gather_topk_candidates_jax(l_val, l_gid, world_size=world_size)
    row_max = jnp.max(g_vals, axis=1, keepdims=True)
    min_id = jnp.min(
        jnp.where(g_vals == row_max, g_ids, jnp.int32(0x7FFFFFFF)),
        axis=1,
        keepdims=True,
    )
    return row_max, min_id


def _fp8_lm_head_gumbel_max_kernel(
    x_ref,
    w8k_hbm,
    scale_ref,
    inv_temp_ref,
    gumbel_mask_ref,
    seed_ref,
    out_val_ref,
    out_id_ref,
    w_buf,
    tile_buf,
    best_val_buf,
    best_idx_buf,
    pack_tmp_v,
    pack_tmp_i,
    send_val_buf,
    send_id_buf,
    recv_val_buf,
    recv_id_buf,
    w_sem,
    ag_send_sem,
    ag_val_recv_sem,
    ag_id_recv_sem,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    logit_scale: float,
    tn: int,
    bm_mxu: int,
) -> None:
    """1-kernel fused FP8 LM head + stateless on-device Gumbel-max categorical sampling!

    - Generates i.i.d. Gumbel(0, 1) variates per ``(token, vocab_col)`` in VREGs via 2-round
      SplitMix32/Murmur3 bijective hashing on ``(seed_u32, me, row_id, col_id)`` underneath
      ``tn=1024`` MXU dots (zero HBM logits, zero Top-64 bitonic sort, zero rank quad-merge!).
    - Rows with ``T <= 1e-6`` set ``gumbel_mask == 0`` and ``inv_temp == 1``, falling back
      seamlessly to exact lowest-global-ID greedy argmax inside the same compiled kernel.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    b_pad, _ = x_ref.shape
    vp = w8k_hbm.shape[1]
    n_tiles = vp // tn
    n_m = b_pad // bm_mxu
    bm_v = min(bm_mxu, 32)
    n_v = bm_mxu // bm_v
    n_sub = tn // 512
    added_width = local_width - org_vocab_padded_count
    pack_rows = (b_pad // 128) * 8 if b_pad >= 128 else 8
    n_pack_chunks = pack_rows // 8

    me = lax.axis_index("model").astype(jnp.int32)
    _entry_barrier_signal(me, world_size)
    me_salt = me.astype(jnp.uint32) * jnp.uint32(0x9E3779B9)
    seed_row = lax.bitcast_convert_type(seed_ref[0:1, :], jnp.uint32) ^ me_salt

    org_count = jnp.clip(
        jnp.int32(org_vocab_size) - me * jnp.int32(org_vocab_padded_count),
        jnp.int32(0),
        jnp.int32(org_vocab_padded_count),
    )
    added_start = jnp.int32(org_vocab_size) + me * jnp.int32(added_width)
    added_count = jnp.clip(
        jnp.int32(global_vocab_size) - added_start,
        jnp.int32(0),
        jnp.int32(added_width),
    )

    def _start_w(slot: int, tile_idx):
        c0 = pl.multiple_of(tile_idx * tn, tn)
        pltpu.make_async_copy(
            w8k_hbm.at[:, pl.ds(c0, tn)],
            w_buf.at[slot],
            w_sem.at[slot],
        ).start()

    def _wait_w(slot: int):
        pltpu.make_async_copy(w_buf.at[slot], w_buf.at[slot], w_sem.at[slot]).wait()

    _start_w(0, jnp.int32(0))
    if n_tiles >= 2:
        _start_w(1, jnp.int32(1))

    neg_inf = jnp.full((bm_v, 128), -jnp.inf, dtype=jnp.float32)
    max_i32 = jnp.full((bm_v, 128), 0x7FFFFFFF, dtype=jnp.int32)
    lane_iota = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    row_sub_iota = lax.broadcasted_iota(jnp.int32, (bm_v, 128), 0)

    n_init_strips = b_pad // bm_v

    def _init_best(ist, _):
        rs = pl.ds(pl.multiple_of(ist * bm_v, bm_v), bm_v)
        best_val_buf[rs, :] = neg_inf
        best_idx_buf[rs, :] = max_i32
        return jnp.int32(0)

    if n_init_strips == 1:
        _init_best(jnp.int32(0), jnp.int32(0))
    else:
        lax.fori_loop(0, n_init_strips, _init_best, jnp.int32(0))

    def _process_tile(tile_idx, slot: int, has_next_prefetch: bool) -> None:
        _wait_w(slot)
        c_tile = pl.multiple_of(tile_idx * tn, tn)

        def _sub_body(isub, _):
            c_sub = pl.multiple_of(isub * 512, 512)
            c_base = c_tile + c_sub
            s_vec = scale_ref[0:1, pl.ds(c_base, 512)]
            if logit_scale != 1.0:
                s_vec = s_vec * jnp.float32(logit_scale)

            def _m_step(im, _):
                r_m = pl.multiple_of(im * bm_mxu, bm_mxu)
                rs_m = pl.ds(r_m, bm_mxu)
                acc = jnp.dot(
                    x_ref[rs_m, :],
                    w_buf[slot, :, pl.ds(c_sub, 512)],
                    preferred_element_type=jnp.float32,
                )
                tile_buf[...] = acc * s_vec

                def _v_step(iv, _):
                    r_v = pl.multiple_of(iv * bm_v, bm_v)
                    rs_local = pl.ds(r_v, bm_v)
                    rs_glob = pl.ds(r_m + r_v, bm_v)
                    cur_v = best_val_buf[rs_glob, :]
                    cur_i = best_idx_buf[rs_glob, :]
                    inv_t = inv_temp_ref[rs_glob, pl.ds(0, 128)][:, 0:1]
                    g_msk = gumbel_mask_ref[rs_glob, pl.ds(0, 128)][:, 0:1]
                    row_salt = (
                        (r_m + r_v + row_sub_iota).astype(jnp.uint32)
                        * jnp.uint32(0xBF58476D)
                    ) ^ seed_row
                    for q in range(4):
                        c_q = c_base + jnp.int32(q * 128) + lane_iota
                        add_p = c_q - jnp.int32(org_vocab_padded_count)
                        valid = (c_q < org_count) | ((add_p >= 0) & (add_p < added_count))
                        h_u32 = row_salt + c_q.astype(jnp.uint32) * jnp.uint32(0x94D049BB)
                        h_u32 = h_u32 ^ lax.shift_right_logical(h_u32, jnp.uint32(16))
                        h_u32 = h_u32 * jnp.uint32(0x85EBCA6B)
                        h_u32 = h_u32 ^ lax.shift_right_logical(h_u32, jnp.uint32(13))
                        h_u32 = h_u32 * jnp.uint32(0xC2B2AE35)
                        h_u32 = h_u32 ^ lax.shift_right_logical(h_u32, jnp.uint32(16))
                        u = (
                            lax.shift_right_logical(h_u32, jnp.uint32(8)).astype(jnp.float32)
                            * jnp.float32(1.0 / 16777216.0)
                            + jnp.float32(0.5 / 16777216.0)
                        )
                        gumbel = -jnp.log(-jnp.log(u))
                        raw_l = tile_buf[rs_local, pl.ds(q * 128, 128)]
                        score_q = raw_l * inv_t + g_msk * gumbel
                        v_q = jnp.where(valid, score_q, -jnp.inf)
                        better = v_q > cur_v
                        cur_v = jnp.where(better, v_q, cur_v)
                        cur_i = jnp.where(better, c_q, cur_i)
                    best_val_buf[rs_glob, :] = cur_v
                    best_idx_buf[rs_glob, :] = cur_i
                    return jnp.int32(0)

                if n_v == 1:
                    _v_step(jnp.int32(0), jnp.int32(0))
                else:
                    lax.fori_loop(0, n_v, _v_step, jnp.int32(0))
                return jnp.int32(0)

            if n_m == 1:
                _m_step(jnp.int32(0), jnp.int32(0))
            else:
                lax.fori_loop(0, n_m, _m_step, jnp.int32(0))
            return jnp.int32(0)

        if n_sub == 1:
            _sub_body(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_sub, _sub_body, jnp.int32(0))

        if has_next_prefetch:
            @pl.when(tile_idx + 2 < n_tiles)
            def _pf():
                _start_w(slot, tile_idx + 2)

    n_pairs = n_tiles // 2
    if n_pairs == 1:
        _process_tile(jnp.int32(0), 0, has_next_prefetch=(n_tiles > 2))
        _process_tile(jnp.int32(1), 1, has_next_prefetch=(n_tiles > 3))
    elif n_pairs > 1:
        def _pair_body(p, _):
            t0 = p * 2
            _process_tile(t0, 0, has_next_prefetch=True)
            _process_tile(t0 + 1, 1, has_next_prefetch=True)
            return jnp.int32(0)

        lax.fori_loop(0, n_pairs, _pair_body, jnp.int32(0))
    if n_tiles % 2 == 1:
        _process_tile(jnp.int32(n_tiles - 1), 0, has_next_prefetch=False)

    chunk_b = min(b_pad, 128)
    n_st = chunk_b // bm_v
    for ch in range(n_pack_chunks):
        if chunk_b < 128:
            pack_tmp_v[...] = jnp.full((128, 8), -jnp.inf, dtype=jnp.float32)
            pack_tmp_i[...] = jnp.full((128, 8), 0x7FFFFFFF, dtype=jnp.int32)

        ch_base = ch * chunk_b

        def _pack_strip(ist, _):
            r_l = pl.multiple_of(ist * bm_v, bm_v)
            bv = best_val_buf[pl.ds(ch_base + r_l, bm_v), :]
            bi = best_idx_buf[pl.ds(ch_base + r_l, bm_v), :]
            gid = _map_local_to_global_id(
                bi,
                me=me,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                added_width=added_width,
            )
            rmax = jnp.max(bv, axis=-1, keepdims=True)
            rgid = jnp.min(
                jnp.where(bv == rmax, gid, jnp.int32(0x7FFFFFFF)),
                axis=-1,
                keepdims=True,
            )
            pack_tmp_v[pl.ds(r_l, bm_v), :] = jnp.broadcast_to(rmax, (bm_v, 8))
            pack_tmp_i[pl.ds(r_l, bm_v), :] = jnp.broadcast_to(rgid, (bm_v, 8))
            return jnp.int32(0)

        if n_st == 1:
            _pack_strip(jnp.int32(0), jnp.int32(0))
        else:
            lax.fori_loop(0, n_st, _pack_strip, jnp.int32(0))

        v_t = pack_tmp_v[...].T
        i_t = pack_tmp_i[...].T
        p_sl = pl.ds(ch * 8, 8)
        send_val_buf[p_sl, :] = v_t
        send_id_buf[p_sl, :] = i_t
        recv_val_buf[me, p_sl, :] = v_t
        recv_id_buf[me, p_sl, :] = i_t

    _entry_barrier_wait(world_size)
    ag_ops = []
    for step in range(1, world_size):
        peer = jnp.mod(me + step, world_size)
        op_v = pltpu.make_async_remote_copy(
            send_val_buf,
            recv_val_buf.at[me],
            ag_send_sem.at[2 * (step - 1)],
            ag_val_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_i = pltpu.make_async_remote_copy(
            send_id_buf,
            recv_id_buf.at[me],
            ag_send_sem.at[2 * (step - 1) + 1],
            ag_id_recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_v.start()
        op_i.start()
        ag_ops.append(op_v)
        ag_ops.append(op_i)

    pltpu.make_async_copy(
        recv_val_buf.at[pl.ds(0, world_size - 1)],
        recv_val_buf.at[pl.ds(0, world_size - 1)],
        ag_val_recv_sem,
    ).wait()
    pltpu.make_async_copy(
        recv_id_buf.at[pl.ds(0, world_size - 1)],
        recv_id_buf.at[pl.ds(0, world_size - 1)],
        ag_id_recv_sem,
    ).wait()

    for ch in range(n_pack_chunks):
        p_sl = pl.ds(ch * 8, 8)
        g_v = recv_val_buf[0, p_sl, :]
        g_i = recv_id_buf[0, p_sl, :]
        for r in range(1, world_size):
            rv = recv_val_buf[r, p_sl, :]
            ri = recv_id_buf[r, p_sl, :]
            take = (rv > g_v) | ((rv == g_v) & (ri < g_i))
            g_v = jnp.where(take, rv, g_v)
            g_i = jnp.where(take, ri, g_i)
        out_val_ref[p_sl, :] = g_v
        out_id_ref[p_sl, :] = g_i

    for op in ag_ops:
        op.wait_send()


def fp8_lm_head_gumbel_max_jax(
    x,
    w8k,
    scale,
    temperatures,
    rng_seed,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    logit_scale: float = 1.0,
    tn: int = 1024,
    collective_id: int = _DEFAULT_COLLECTIVE_ID + 4,
):
    """Single-kernel fused FP8 LM head + exact on-device Gumbel-max categorical sampling."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, h = x.shape
    wk_h, vp = w8k.shape
    if wk_h != h or vp % 512 != 0:
        raise ValueError(f"Expected w8k ({h}, 512*q), got {w8k.shape}")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    eff_tn = tn if (vp % tn == 0) else 512

    b_pad = _pad_batch_pow2(tokens)
    bm_mxu = min(b_pad, 128)
    pack_rows = (b_pad // 128) * 8 if b_pad >= 128 else 8
    x_in = x.astype(jnp.bfloat16)
    t_col = temperatures.reshape(tokens, 1).astype(jnp.float32)
    if b_pad > tokens:
        x_in = jnp.pad(x_in, ((0, b_pad - tokens), (0, 0)))
        t_col = jnp.pad(t_col, ((0, b_pad - tokens), (0, 0)), constant_values=1.0)

    is_greedy = t_col <= jnp.float32(1e-6)
    inv_t = jnp.where(is_greedy, jnp.float32(1.0), jnp.float32(1.0) / jnp.maximum(t_col, jnp.float32(1e-6)))
    g_mask = jnp.where(is_greedy, jnp.float32(0.0), jnp.float32(1.0))
    inv_t_128 = jnp.broadcast_to(inv_t, (b_pad, 128))
    g_mask_128 = jnp.broadcast_to(g_mask, (b_pad, 128))
    seed_in = jnp.broadcast_to(rng_seed.reshape(1, 128).astype(jnp.float32), (8, 128))

    vmem = pltpu.VMEM
    hbm = pltpu.HBM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=hbm)

    packed_val, packed_id = pl.pallas_call(
        functools.partial(
            _fp8_lm_head_gumbel_max_kernel,
            world_size=world_size,
            local_width=local_width,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            logit_scale=float(logit_scale),
            tn=eff_tn,
            bm_mxu=bm_mxu,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((pack_rows, 128), jnp.float32),
            jax.ShapeDtypeStruct((pack_rows, 128), jnp.int32),
        ),
        in_specs=[v_spec, h_spec, v_spec, v_spec, v_spec, v_spec],
        out_specs=(v_spec, v_spec),
        scratch_shapes=[
            vmem((2, h, eff_tn), w8k.dtype),
            vmem((bm_mxu, 512), jnp.float32),
            vmem((b_pad, 128), jnp.float32),
            vmem((b_pad, 128), jnp.int32),
            vmem((128, 8), jnp.float32),
            vmem((128, 8), jnp.int32),
            vmem((pack_rows, 128), jnp.float32),
            vmem((pack_rows, 128), jnp.int32),
            vmem((world_size, pack_rows, 128), jnp.float32),
            vmem((world_size, pack_rows, 128), jnp.int32),
            sem((2,)),
            sem((2 * (world_size - 1),)),
            sem,
            sem,
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_fp8_lm_head_gumbel_max_b{b_pad}_tp{world_size}_tn{eff_tn}",
    )(x_in, w8k, scale.reshape(1, vp).astype(jnp.float32), inv_t_128, g_mask_128, seed_in)

    if pack_rows == 8:
        out_v = packed_val[0, :tokens].reshape(tokens, 1)
        out_i = packed_id[0, :tokens].reshape(tokens, 1)
    else:
        n_chunks = pack_rows // 8
        out_v = jnp.concatenate(
            [packed_val[ch * 8, :] for ch in range(n_chunks)], axis=0
        )[:tokens].reshape(tokens, 1)
        out_i = jnp.concatenate(
            [packed_id[ch * 8, :] for ch in range(n_chunks)], axis=0
        )[:tokens].reshape(tokens, 1)
    return out_v, out_i


def fp8_lm_head_distributed_topk_jax(
    x,
    w8k,
    scale,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int,
    logit_scale: float = 1.0,
    exact_tie_break: bool = True,
    tn: int | None = None,
    fast_topk: bool = False,
):
    """Single-kernel fused FP8 LM head + exact global Top-K (``k==1`` or ``1 < k <= 64``)."""
    if x.shape[0] <= 384:
        if k == 1:
            return fp8_lm_head_argmax_jax(
                x,
                w8k,
                scale,
                world_size=world_size,
                local_width=local_width,
                org_vocab_size=org_vocab_size,
                global_vocab_size=global_vocab_size,
                org_vocab_padded_count=org_vocab_padded_count,
                logit_scale=logit_scale,
                tn=1024 if tn is None else tn,
            )
        # Preserve exact lowest-ID col-0 tie-breaking unless fast_topk is explicitly enabled;
        # Wave-17/19 silicon: tn=1024 (+tail512 on TP=16) + batch-lane folding (B <= 64) cuts
        # TP=8 B=1..64 Top-64 -30.9%..-46.3% and TP=16 B=1..64 Top-64 -27.7%..-39.4% with id_diff=0.
        small_b_fold = x.shape[0] <= 64
        tp8_large_b = world_size == 8 and x.shape[0] > 32
        eff_tn = (1024 if (fast_topk or tp8_large_b or small_b_fold) else 512) if tn is None else tn
        eff_tie = False if (fast_topk and k > 1) else exact_tie_break
        eff_xor_bfly = bool(fast_topk or world_size == 16 or tp8_large_b)
        return fp8_lm_head_fused_topk64_jax(
            x,
            w8k,
            scale,
            world_size=world_size,
            local_width=local_width,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            k=k,
            logit_scale=logit_scale,
            tn=eff_tn,
            exact_tie_break=eff_tie,
            use_xor_butterfly=eff_xor_bfly,
            use_pack_fold=small_b_fold,
        )
    local_logits = fp8_lm_head_zsp_jax(x, w8k, scale, out_bf16=False, logit_scale=logit_scale)
    from sglang_tpu.kernels.distributed_topk import distributed_top_k as orig_dist_topk

    return orig_dist_topk(
        local_logits[:, :local_width],
        world_size=world_size,
        org_vocab_size=org_vocab_size,
        global_vocab_size=global_vocab_size,
        org_vocab_padded_count=org_vocab_padded_count,
        k=k,
    )


# =============================================================================
# 6. Load-Time K-Major FP8 Quantizer (Fresh Contiguous Tensors, Zero Views)
# =============================================================================
@torch.no_grad()
def quantize_lm_head_fp8_k_major(
    weight_vh: torch.Tensor,
    *,
    tn: int = _LM_HEAD_TN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``lm_head.weight [V, H]`` -> fresh contiguous ``(fp8 [H, Vp], f32 [1, Vp])``."""
    v, h = weight_vh.shape
    vp = _ceil_multiple(v, tn)
    wk = weight_vh.detach().t().contiguous()
    cols, scales = [], []
    for v0 in range(0, v, _QUANT_COLS):
        w = wk[:, v0 : v0 + _QUANT_COLS].float()
        s = (w.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
        cols.append((w / s).to(torch.float8_e4m3fn))
        scales.append(s.reshape(1, -1))
    if vp > v:
        cols.append(torch.zeros(h, vp - v, dtype=torch.float8_e4m3fn, device=weight_vh.device))
        scales.append(torch.zeros(1, vp - v, dtype=torch.float32, device=weight_vh.device))
    from sglang_tpu.fp8_moe import _own_storage

    w8k = torch.cat(cols, dim=1).contiguous()
    scale = _own_storage(torch.cat(scales, dim=1).contiguous())
    return w8k, scale


# =============================================================================
# 7. TorchTPU `cpu_export_jax_op` Registration Helpers
# =============================================================================
@functools.cache
def _get_tp_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def get_fp8_lm_head_zsp_op(*, out_bf16: bool = False, use_xla: bool = False) -> Callable:
    key = (bool(out_bf16), bool(use_xla))
    op = _FP8_LM_HEAD_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _FP8_LM_HEAD_OPS.get(key)
        if op is None:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                x: jax.Array,
                w8k: jax.Array,
                scale: jax.Array,
                logit_scale: float,
            ) -> jax.Array:
                if use_xla:
                    return fp8_lm_head_xla_jax(
                        x, w8k, scale, out_bf16=out_bf16, logit_scale=logit_scale
                    )
                return fp8_lm_head_zsp_jax(
                    x, w8k, scale, out_bf16=out_bf16, logit_scale=logit_scale
                )

            tag = f"{'xla' if use_xla else 'zsp'}_{'bf16' if out_bf16 else 'f32'}"
            op = cpu_export_jax_op(f"sglang_tpu_pallas::muse_fp8_lm_head_{tag}", _wrapped)
            _FP8_LM_HEAD_OPS[key] = op
    return op


def get_fp8_lm_head_distributed_topk_op(
    world_size: int, *, fast_topk: bool | None = None
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    if fast_topk is None:
        from sglang_tpu.envs import envs as tpu_envs

        fast_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_FAST_FUSED_LM_HEAD_TOPK", None)
        fast_topk = bool(fast_env is not None and fast_env.get())
    key = (int(world_size), bool(fast_topk))
    op = _FUSED_TOPK_OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _FUSED_TOPK_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            in_specs = (P(), P(None, "model"), P(None, "model"), None, None, None, None, None, None)
            out_specs = (P(), P())

            def _local(
                x: jax.Array,
                w8k: jax.Array,
                scale: jax.Array,
                local_width: int,
                org_vocab_size: int,
                global_vocab_size: int,
                org_vocab_padded_count: int,
                k: int,
                logit_scale: float,
            ) -> tuple[jax.Array, jax.Array]:
                return fp8_lm_head_distributed_topk_jax(
                    x,
                    w8k,
                    scale,
                    world_size=world_size,
                    local_width=local_width,
                    org_vocab_size=org_vocab_size,
                    global_vocab_size=global_vocab_size,
                    org_vocab_padded_count=org_vocab_padded_count,
                    k=k,
                    logit_scale=logit_scale,
                    fast_topk=bool(fast_topk),
                )

            sharded = jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )

            def _wrapped(
                x: jax.Array,
                w8k: jax.Array,
                scale: jax.Array,
                local_width: int,
                org_vocab_size: int,
                global_vocab_size: int,
                org_vocab_padded_count: int,
                k: int,
                logit_scale: float,
            ) -> tuple[jax.Array, jax.Array]:
                return sharded(
                    x,
                    w8k,
                    scale,
                    local_width,
                    org_vocab_size,
                    global_vocab_size,
                    org_vocab_padded_count,
                    k,
                    logit_scale,
                )

            sfx = "_fast" if fast_topk else ""
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_fp8_lm_head_dist_topk{sfx}_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _FUSED_TOPK_OPS[key] = op
    return op


def get_fp8_lm_head_gumbel_max_op(world_size: int) -> Callable:
    """Register 1-kernel fused FP8 LM head + on-device Gumbel-max categorical sampler."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    op = _GUMBEL_MAX_OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _GUMBEL_MAX_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            in_specs = (
                P(),
                P(None, "model"),
                P(None, "model"),
                P(),
                P(),
                None,
                None,
                None,
                None,
                None,
            )
            out_specs = (P(), P())

            def _local(
                x: jax.Array,
                w8k: jax.Array,
                scale: jax.Array,
                temperatures: jax.Array,
                rng_seed: jax.Array,
                local_width: int,
                org_vocab_size: int,
                global_vocab_size: int,
                org_vocab_padded_count: int,
                logit_scale: float,
            ) -> tuple[jax.Array, jax.Array]:
                return fp8_lm_head_gumbel_max_jax(
                    x,
                    w8k,
                    scale,
                    temperatures,
                    rng_seed,
                    world_size=world_size,
                    local_width=local_width,
                    org_vocab_size=org_vocab_size,
                    global_vocab_size=global_vocab_size,
                    org_vocab_padded_count=org_vocab_padded_count,
                    logit_scale=logit_scale,
                )

            sharded = jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )

            def _wrapped(
                x: jax.Array,
                w8k: jax.Array,
                scale: jax.Array,
                temperatures: jax.Array,
                rng_seed: jax.Array,
                local_width: int,
                org_vocab_size: int,
                global_vocab_size: int,
                org_vocab_padded_count: int,
                logit_scale: float,
            ) -> tuple[jax.Array, jax.Array]:
                return sharded(
                    x,
                    w8k,
                    scale,
                    temperatures,
                    rng_seed,
                    local_width,
                    org_vocab_size,
                    global_vocab_size,
                    org_vocab_padded_count,
                    logit_scale,
                )

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_fp8_lm_head_gumbel_max_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _GUMBEL_MAX_OPS[world_size] = op
    return op


def get_lossless_distributed_topk_vmem_ag_op(world_size: int) -> Callable:
    """Lazily register lossless vertical-bitonic + 1-hop VMEM AG distributed Top-K op."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    op = _LOSSLESS_TOPK_OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _LOSSLESS_TOPK_OPS.get(world_size)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            in_specs = (P(None, "model"), None, None, None, None)
            out_specs = (P(), P())

            def _local(
                local_logits: jax.Array,
                org_vocab_size: int,
                global_vocab_size: int,
                org_vocab_padded_count: int,
                k: int,
            ) -> tuple[jax.Array, jax.Array]:
                return distributed_topk_vmem_ag_jax(
                    local_logits,
                    world_size=world_size,
                    local_width=int(local_logits.shape[1]),
                    org_vocab_size=org_vocab_size,
                    global_vocab_size=global_vocab_size,
                    org_vocab_padded_count=org_vocab_padded_count,
                    k=k,
                )

            sharded = jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )

            def _wrapped(
                local_logits: jax.Array,
                org_vocab_size: int,
                global_vocab_size: int,
                org_vocab_padded_count: int,
                k: int,
            ) -> tuple[jax.Array, jax.Array]:
                return sharded(
                    local_logits,
                    org_vocab_size,
                    global_vocab_size,
                    org_vocab_padded_count,
                    k,
                )

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_lossless_dist_topk_vmem_ag_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _LOSSLESS_TOPK_OPS[world_size] = op
    return op
