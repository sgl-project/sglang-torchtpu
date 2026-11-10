# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prefill tiles with transposed scores: S^T = K Q^T, O^T = V P^T.

The standard tile computes scores [(token, q head) rows, keys] and is bound by
MXU result pops: every f32 [8, 256] half of a QK or PV matmul is popped, the
PV result is as wide as a vreg although the head is 64 lanes, and the running
max / sum need cross-lane reductions and lane-replicated carries. With keys
along sublanes and (token, q head) rows along lanes:

* the column max and sum are elementwise across vregs (no cross-lane work),
* m, l and alpha are one lane vector per 128 rows instead of [rows, 128],
* O^T = V^T P^T is [head_dim, rows], 4x fewer MXU rows and result pops than
  P V, and a block of ones rows in V^T yields l for free,
* K streams through the MXU as the LHS; with fp8 K and fp8 q it takes one
  MXU op per 32 keys. q is quantized to fp8 here (a bf16 q would be upcast
  with K and double the MXU issue), so this path is not bit-exact with the
  standard tile.

One compute block is bq_c tokens x q heads = 128k lanes; scratch carries are
m [KV, 1, rows] and acc [KV, head_dim + 16, rows] in f32, and Q^T is built
once per (sequence, q block) in VMEM.
"""

import math
import os
from collections.abc import Callable

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.batched_rpa import configs, utils

LOG2E = 1.0 / math.log(2.0)

# Minimax cubic for 2^f on [0, 1): max rel. error ~1e-4 (below bf16's 2^-8).
_EXP2_C = (1.0, 0.69583356, 0.22606716, 0.07802460)


def _exp2_valu(s: jax.Array) -> jax.Array:
    """exp2 on the VALU (no EUP): 2^floor(x) via the exponent bits times a cubic.

    Lets a share of the softmax exponentials bypass the single EUP, which is the
    binding unit of the unshifted streaming tile.
    """
    x = jnp.maximum(s.astype(jnp.float32), -126.0)
    xi = jnp.floor(x)
    f = x - xi
    c0, c1, c2, c3 = _EXP2_C
    poly = c0 + f * (c1 + f * (c2 + f * c3))
    scale = lax.bitcast_convert_type((xi.astype(jnp.int32) + 127) << 23, jnp.float32)
    return (poly * scale).astype(s.dtype)


def refresh_qt(
    load_q: Callable[[], jax.Array],  # -> [1, KV, rows, aligned_q_head_dim]
    qt_ref: jax.Ref,  # [KV, aligned_kv_head_dim, rows]
    qkey_ref: jax.Ref,  # SMEM [2]: (s_idx, q_idx) the cached Q^T belongs to
    s_idx: jax.Array,
    q_idx: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    """Rebuilds the fp8 Q^T when the step moved to another (sequence, q block)."""
    changed = jnp.logical_or(s_idx != qkey_ref[0], q_idx != qkey_ref[1])

    @pl.when(changed)
    def _():
        q = load_q()
        scale = cfgs.model.sm_scale * LOG2E
        hd = cfgs.aligned_kv_head_dim
        for kv_head in range(cfgs.model.num_kv_heads):
            qh = q[0, kv_head].astype(jnp.float32) * scale  # [rows, aligned_q_head_dim]
            qt = qh.T[:hd]  # [hd, rows]
            info = jnp.finfo(qt_ref.dtype)
            qt = jnp.clip(qt, float(info.min), float(info.max))
            qt_ref[kv_head] = qt.astype(qt_ref.dtype)
        qkey_ref[0] = s_idx
        qkey_ref[1] = q_idx


def skip_group_tokens(cfgs: configs.RpaConfigs) -> int:
    """Query tokens per partial-block loop iteration: the configured group rounded down
    to whole compute blocks, and to a divisor of the q block (the whole block if none)."""
    per = max(1, cfgs.serve.skip_pad_q_group // cfgs.bq_c_sz)
    n_blocks = cfgs.bq_sz // cfgs.bq_c_sz
    while n_blocks % per:
        per -= 1
    return per * cfgs.bq_c_sz


def store_tile(
    step: jax.Array,
    schedule_ref,
    acc_ref: jax.Ref,
    o_vref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Normalizes and writes the completed q block on its final k step.

    Factored out so ``kernel.py`` emits ``store()`` once after the
    ``full_tile`` / ``partial_q`` dispatch instead of duplicating it in up to
    four branches (keeping IMEM well inside Percale's 4 MiB limit at bq=1024).
    """
    heads = cfgs.aligned_num_q_heads_per_kv_head
    hd = cfgs.aligned_kv_head_dim
    rows = cfgs.bq_sz * heads
    no_max = cfgs.serve.no_max
    is_last_k = schedule_ref.is_last_k[step, 0] == 1

    def store():
        outs = []
        for kv_head in range(cfgs.model.num_kv_heads):
            acc = acc_ref[kv_head]  # [acc rows, rows]
            l = acc[hd : hd + 1]
            if no_max and cfgs.serve.skip_pad_q:
                # Skipped groups keep acc == 0 (l == 0): output 0 there, not NaN.
                # Exact for real rows, whose l >= 2^-64 (a query sees its own key).
                l = jnp.maximum(l, 1e-30)
            o_t = acc[:hd] * pl.reciprocal(l, approx=True)
            if cfgs.serve.out_head_norm_eps is not None:
                o_t = utils.head_norm(
                    o_t.astype(cfgs.serve.dtype_out),
                    0,
                    cfgs.model.head_dim,
                    cfgs.serve.out_head_norm_eps,
                )
            pad = cfgs.aligned_q_head_dim - hd
            if pad:
                o_t = jnp.concatenate([o_t, jnp.zeros((pad, rows), o_t.dtype)], axis=0)
            outs.append(
                o_t.T.astype(cfgs.serve.dtype_out)
            )  # [rows, aligned_q_head_dim]
        out = jnp.stack(outs, axis=0)
        o_u32_vref = o_vref.at[0].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)
        if no_max:
            # No alpha == 0 restart without a running max: clear for the next block.
            acc_ref[...] = jnp.zeros_like(acc_ref)

    lax.cond(is_last_k, store, lambda: None)


def tile(
    step: jax.Array,
    k: jax.Array,  # [1, KV, aligned_kv_head_dim, bkv] fp8
    v: jax.Array,  # [1, KV, aligned_kv_head_dim, bkv] fp8
    schedule_ref,
    m_ref: jax.Ref,  # [KV, 1, rows] f32, base-2 running max
    acc_ref: jax.Ref,  # [KV, acc_rows, rows] f32: O^T, then l in row aligned_kv_head_dim
    qt_ref: jax.Ref,  # [KV, aligned_kv_head_dim, rows] fp8
    o_vref: jax.Ref,
    *,
    processed_q_len: jax.Array,
    processed_kv_len: jax.Array,
    apply_mask: bool,
    cfgs: configs.RpaConfigs,
    q_rows: jax.Array | None = None,
    emit_store: bool = True,
):
    """``q_rows``: the q block's real queries when it may be partial. Its compute blocks
    then run in a loop over skip_group_tokens groups up to the last query. With
    no_max this is the unshifted streaming tile (skipped groups keep acc == 0, which
    store_tile() clears after every q block); else ``_tile_shifted``, where the groups
    past the last query only set m to -inf, so the next q block starts clean."""
    heads = cfgs.aligned_num_q_heads_per_kv_head
    hd = cfgs.aligned_kv_head_dim
    bkv = cfgs.bkv_sz
    lanes = cfgs.bq_c_sz * heads
    offset = processed_kv_len - processed_q_len  # kv index of the first query
    # QK-normed q and k bound |s| (log2 units) far inside exp2's range.
    no_max = cfgs.serve.no_max

    if apply_mask:
        kv_iota = lax.broadcasted_iota(jnp.int32, (bkv, lanes), 0)
        q_iota = lax.broadcasted_iota(jnp.int32, (bkv, lanes), 1) // heads
        q_kv_diff = q_iota - kv_iota

    # Partial q blocks (skip_pad_q): groups of compute blocks covering
    # skip_group_tokens tokens; only the groups up to the last real query run.
    group_tokens = skip_group_tokens(cfgs)
    per_group = group_tokens // cfgs.bq_c_sz
    n_groups = cfgs.bq_sz // group_tokens
    grouped = q_rows is not None and n_groups > 1
    live_groups = (
        jnp.clip((q_rows + group_tokens - 1) // group_tokens, 1, n_groups)
        if grouped
        else None
    )

    if no_max:
        # Unshifted softmax (QK-normed models: |base-2 score| <= ~64, so exp2
        # cannot overflow bf16 / the f32 sums): stream the tile in key chunks so a
        # chunk's scores and probabilities never leave vregs, and drop the running
        # max, its reduction and the alpha rescale. acc is cleared after each
        # finished q block in store_tile(), so skipped groups keep acc == 0.
        kc_cap = cfgs.serve.transposed_key_chunk or min(bkv, 1536)
        kc = math.gcd(bkv, min(bkv, kc_cap))
        chunk_vt = cfgs.serve.transposed_chunk_vt
        mxu_split = int(os.environ.get("RPA_EXP_MXU", "0"))
        for kv_head in range(cfgs.model.num_kv_heads):
            kh = k[0, kv_head]  # [hd, bkv] fp8
            ones_rows = cfgs.acct_scratch_shape[1] - hd
            vt = None
            if not chunk_vt:
                vt = jnp.concatenate(
                    [
                        v[0, kv_head].astype(jnp.bfloat16),
                        jnp.ones((ones_rows, bkv), jnp.bfloat16),
                    ],
                    axis=0,
                )  # [acc rows, bkv]

            if mxu_split:
                kht = _padded_kt(kh)

                def q_chunk(bq_start, cs, par=0, defer=False, kht=kht, vt=vt, kv_head=kv_head):
                    mask = None
                    if apply_mask:
                        lo = offset - bq_start
                        hi = None if cfgs.model.sliding_window is None else (
                            cfgs.model.sliding_window + lo)
                        mask = (lo, hi)
                    pend = _q_chunk_mxu(kht, vt, qt_ref, acc_ref, kv_head, cs, par,
                                        q_kv_diff if apply_mask else None, mask, cfgs)
                    if not defer:
                        _q_chunk_finish(pend[0], *pend[1])
                        return None
                    return pend

            def q_chunk(
                bq_start, cs, par=0, defer=False, kh=kh, vt=vt, kv_head=kv_head, ones_rows=ones_rows,
                _mxu=q_chunk if mxu_split else None,
            ):
                if _mxu is not None:
                    return _mxu(bq_start, cs, par, defer)
                qt = qt_ref[kv_head, :, cs]  # [hd, lanes] fp8
                o_t = None
                for c0 in range(0, bkv, kc):
                    s = lax.dot_general(
                        kh[:, c0 : c0 + kc],
                        qt,
                        (((0,), (0,)), ((), ())),
                        preferred_element_type=(
                            jnp.bfloat16
                            if cfgs.serve.transposed_s_bf16
                            else jnp.float32
                        ),
                    ).astype(jnp.bfloat16)  # [kc, lanes]
                    if apply_mask:
                        diff = q_kv_diff[c0 : c0 + kc]
                        mask = diff >= offset - bq_start
                        if (window := cfgs.model.sliding_window) is not None:
                            mask = jnp.logical_and(
                                mask, diff < window + offset - bq_start
                            )
                        s = jnp.where(mask, s, cfgs.model.mask_value)
                    every = cfgs.serve.transposed_valu_exp2_every
                    if every and (c0 // kc) % every == every - 1:
                        p = _exp2_valu(s)  # this chunk's exponentials on the VALU
                    else:
                        p = jnp.exp2(s)  # [kc, lanes] bf16 on the EUP
                    if chunk_vt:
                        # Only this chunk's V^T (+ ones rows) is live: fewer vregs held
                        # across the q-chunk loop (the whole [acc rows, bkv] spills).
                        vt_c = jnp.concatenate(
                            [
                                v[0, kv_head, :, c0 : c0 + kc].astype(jnp.bfloat16),
                                jnp.ones((ones_rows, kc), jnp.bfloat16),
                            ],
                            axis=0,
                        )
                    else:
                        vt_c = vt[:, c0 : c0 + kc]
                    part = lax.dot_general(
                        vt_c,
                        p,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )  # [acc rows, lanes]
                    o_t = part if o_t is None else o_t + part
                acc_ref[kv_head, :, cs] = acc_ref[kv_head, :, cs] + o_t

            n_qc = cfgs.bq_sz // cfgs.bq_c_sz
            loop_unroll = int(os.environ.get("SGLANG_TPU_RPA_QCHUNK_LOOP", "0"))
            # Masked boundary tiles run at most 1-2 times per q block on global
            # layers; rolling them when total compute blocks > 32 keeps hot
            # full_tile 100% unrolled while holding kernel IMEM well inside
            # Percale's 4 MiB budget at bq=1024.
            mask_loop_unroll = (
                4
                if (
                    apply_mask
                    and not loop_unroll
                    and n_qc * cfgs.model.num_kv_heads > 32
                )
                else loop_unroll
            )
            if grouped:
                sub = math.gcd(per_group, 2 if apply_mask else 4)
                steps_per_group = per_group // sub
                sub_tokens = sub * cfgs.bq_c_sz
                sub_lanes = sub * lanes

                def run_group(
                    s_idx,
                    carry,
                    q_chunk=q_chunk,
                    sub=sub,
                    sub_tokens=sub_tokens,
                    sub_lanes=sub_lanes,
                ):
                    for j in range(sub):
                        bq_start = s_idx * sub_tokens + j * cfgs.bq_c_sz
                        lane0 = pl.multiple_of(s_idx * sub_lanes + j * lanes, lanes)
                        q_chunk(bq_start, pl.ds(lane0, lanes), j % 2)
                    return carry

                lax.fori_loop(0, live_groups * steps_per_group, run_group, 0)
            elif mask_loop_unroll and n_qc > 1:

                def body(i, carry, q_chunk=q_chunk):
                    bq_start = i * cfgs.bq_c_sz
                    cs = pl.ds(pl.multiple_of(i * lanes, lanes), lanes)
                    q_chunk(bq_start, cs)
                    return carry

                lax.fori_loop(0, n_qc, body, 0, unroll=min(mask_loop_unroll, n_qc))
            elif mxu_split and os.environ.get("RPA_EXP_FUSED"):
                _streams_fused(kht, vt, qt_ref, acc_ref, kv_head,
                               [slice(b * heads, b * heads + lanes) for b in range(0, cfgs.bq_sz, cfgs.bq_c_sz)], cfgs)
            elif mxu_split and os.environ.get("RPA_EXP_STREAMS"):
                depth = int(os.environ["RPA_EXP_STREAMS"])
                pend = []
                for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
                    cs_b = slice(bq_start * heads, bq_start * heads + lanes)
                    pend.append((_streams_issue(kht, qt_ref, kv_head, cs_b, cfgs), cs_b))
                    if len(pend) > depth:
                        iss, c_ = pend.pop(0)
                        _streams_finish(iss, vt, acc_ref, kv_head, c_)
                for iss, c_ in pend:
                    _streams_finish(iss, vt, acc_ref, kv_head, c_)
            elif mxu_split and os.environ.get("RPA_EXP_GRP"):
                g = int(os.environ["RPA_EXP_GRP"])
                starts = list(range(0, cfgs.bq_sz, cfgs.bq_c_sz))
                for g0 in range(0, len(starts), g):
                    grp_s = starts[g0:g0 + g]
                    def cs_of(b):
                        return slice(b * heads, b * heads + lanes)
                    vp = q_chunk(grp_s[-1], cs_of(grp_s[-1]), grp_s[-1] // cfgs.bq_c_sz, True)
                    for b in grp_s[:-1]:
                        ep = q_chunk(b, cs_of(b), b // cfgs.bq_c_sz, True)
                        _q_chunk_finish(ep[0], *ep[1])
                    _q_chunk_finish(vp[0], *vp[1])
            elif mxu_split and os.environ.get("RPA_EXP_SWP"):
                depth = int(os.environ["RPA_EXP_SWP"])
                pend = []
                for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
                    pend.append(q_chunk(bq_start, slice(bq_start * heads, bq_start * heads + lanes), bq_start // cfgs.bq_c_sz, True))
                    if len(pend) > depth:
                        p0 = pend.pop(0)
                        _q_chunk_finish(p0[0], *p0[1])
                for p0 in pend:
                    _q_chunk_finish(p0[0], *p0[1])
            else:
                for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
                    q_chunk(
                        bq_start,
                        slice(bq_start * heads, bq_start * heads + lanes),
                        bq_start // cfgs.bq_c_sz,
                    )
    else:
        _tile_shifted(
            step,
            k,
            v,
            schedule_ref,
            m_ref,
            acc_ref,
            qt_ref,
            q_kv_diff=q_kv_diff if apply_mask else None,
            offset=offset,
            apply_mask=apply_mask,
            cfgs=cfgs,
            grouped=grouped,
            live_groups=live_groups,
        )

    if emit_store:
        store_tile(step, schedule_ref, acc_ref, o_vref, cfgs=cfgs)


def _tile_shifted(
    step,
    k,
    v,
    schedule_ref,
    m_ref,
    acc_ref,
    qt_ref,
    *,
    q_kv_diff,
    offset,
    apply_mask,
    cfgs,
    grouped=False,
    live_groups=None,
):
    """The online-softmax transposed tile (running base-2 max).

    With ``grouped`` (skip_pad_q on a partial q block) the compute blocks run in a
    loop over the ``live_groups`` groups up to the last query; the groups past it
    only set m to -inf, so the next q block starts clean (acc stays finite).
    """
    heads = cfgs.aligned_num_q_heads_per_kv_head
    hd = cfgs.aligned_kv_head_dim
    bkv = cfgs.bkv_sz
    lanes = cfgs.bq_c_sz * heads
    group_tokens = skip_group_tokens(cfgs)
    per_group = group_tokens // cfgs.bq_c_sz
    n_groups = cfgs.bq_sz // group_tokens
    group_lanes = group_tokens * heads

    def block(kv_head, kh, vt, bq_start, cs):
        """One compute block: queries [bq_start, bq_start + bq_c_sz) of the q block,
        lanes ``cs`` (a static slice, or a pl.ds at a traced lane offset)."""
        qt = qt_ref[kv_head, :, cs]  # [hd, lanes] fp8
        s = lax.dot_general(
            kh, qt, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )  # [bkv, lanes]
        s = s.astype(jnp.bfloat16)
        if apply_mask:
            # key index kv_iota + processed_kv_len <= query index
            # q_iota + bq_start + processed_q_len, i.e. q_kv_diff >= offset - bq_start.
            mask = q_kv_diff >= offset - bq_start
            if (window := cfgs.model.sliding_window) is not None:
                mask = jnp.logical_and(mask, q_kv_diff < window + offset - bq_start)
            s = jnp.where(mask, s, cfgs.model.mask_value)
        m_prev = m_ref[kv_head, :, cs]  # [1, lanes]
        m_cur = jnp.max(s, axis=0, keepdims=True).astype(jnp.float32)
        m_new = jnp.maximum(m_prev, m_cur)
        alpha = jnp.exp2(m_prev - m_new)
        p = jnp.exp2(s - m_new.astype(jnp.bfloat16))  # [bkv, lanes] bf16
        if cfgs.serve.transposed_fp8_p:
            # p in (0, 1] and V fp8: one 8-bit MXU pass for O^T (and l).
            o_t = lax.dot_general(
                vt.astype(jnp.float8_e4m3fn),
                p.astype(jnp.float8_e4m3fn),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            o_t = lax.dot_general(
                vt, p, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )  # [acc rows, lanes]: O^T rows, then the row sums
        acc_ref[kv_head, :, cs] = alpha * acc_ref[kv_head, :, cs] + o_t
        # A finished q block starts its next one from -inf (alpha == 0 then
        # clears acc and l).
        m_ref[kv_head, :, cs] = jnp.where(
            schedule_ref.is_last_k[step, 0], -jnp.inf, m_new
        )

    for kv_head in range(cfgs.model.num_kv_heads):
        kh = k[0, kv_head]  # [hd, bkv] fp8, contracted on dim 0: the MXU LHS
        ones_rows = cfgs.acct_scratch_shape[1] - hd
        vt = jnp.concatenate(
            [
                v[0, kv_head].astype(jnp.bfloat16),
                jnp.ones((ones_rows, bkv), jnp.bfloat16),
            ],
            axis=0,
        )  # [acc rows, bkv]
        if not grouped:
            for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
                cs = slice(bq_start * heads, bq_start * heads + lanes)
                block(kv_head, kh, vt, bq_start, cs)
            continue

        def run_group(g, carry, kv_head=kv_head, kh=kh, vt=vt):
            for j in range(per_group):
                bq_start = g * group_tokens + j * cfgs.bq_c_sz
                lane0 = pl.multiple_of(g * group_lanes + j * lanes, lanes)
                block(kv_head, kh, vt, bq_start, pl.ds(lane0, lanes))
            return carry

        def reset_group(g, carry, kv_head=kv_head):
            lane0 = pl.multiple_of(g * group_lanes, group_lanes)
            m_ref[kv_head, :, pl.ds(lane0, group_lanes)] = jnp.full(
                (1, group_lanes), -jnp.inf, jnp.float32
            )
            return carry

        lax.fori_loop(0, live_groups, run_group, 0)
        lax.fori_loop(live_groups, n_groups, reset_group, 0)


_PIECE = 128  # keys per MXU result pop group


def _padded_kt(kh: jax.Array) -> jax.Array:
    """K^T with the head padded to the 256-deep MXU contraction: [bkv, 256] fp8."""
    hd, bkv = kh.shape
    kt = kh.astype(jnp.bfloat16).T
    kt = jnp.concatenate([kt, jnp.zeros((bkv, 256 - hd), jnp.bfloat16)], axis=1)
    return kt.astype(kh.dtype)


class _MrbRing:
    """Static ring allocation of one MXU's accumulators (units of 8 result rows)."""

    def __init__(self, base: int, size: int):
        self.base, self.size, self.next = base, size, 0

    def take(self, n: int) -> int:
        if self.next + n > self.size:
            self.next = 0
        a = self.base + self.next
        self.next += n
        return a


_RINGS: dict[int, _MrbRing] = {}


def _q_chunk_mxu(kht, vt, qt_ref, acc_ref, kv_head, cs, par, q_kv_diff, mask, cfgs):
    """One compute block on explicitly placed MXUs: QK keys [0, k1) on MXU 0, keys
    [k1, bkv) and PV on MXU 1, result buffers as rings; the last valu_keys keys' exp2
    on the VALU (unmasked tiles)."""
    k1 = int(os.environ.get("RPA_EXP_MXU", "0"))
    piece = int(os.environ.get("RPA_EXP_PIECE", "32"))
    valu_keys = 0 if mask is not None else int(os.environ.get("RPA_EXP_VALU_KEYS", "0"))
    vchunk = int(os.environ.get("RPA_EXP_VCHUNK", "0"))
    if vchunk and mask is None and par % vchunk == vchunk - 1:
        k1, valu_keys = 0, kht.shape[0]
    hd = cfgs.aligned_kv_head_dim
    bkv, lanes = kht.shape[0], 256
    acc_rows = vt.shape[0]
    pv_accs = acc_rows // 4  # result buffer entries are [8, 128] f32
    if 0 not in _RINGS:
        _RINGS[0] = _MrbRing(0, 256)
        _RINGS[1] = _MrbRing(0, 256 - 2 * pv_accs)
        _RINGS[2] = _MrbRing(256 - 2 * pv_accs, 2 * pv_accs)
    qt = qt_ref[kv_head, :, cs]
    assert qt.shape[1] == lanes and bkv % 256 == 0 and k1 % piece == 0
    qt = jnp.concatenate([qt, jnp.zeros((256 - hd, lanes), qt.dtype)], axis=0)
    grp = int(os.environ.get("RPA_EXP_GRP", "0"))
    if grp:
        is_v = mask is None and par % grp == grp - 1
        m = 1 if is_v else 0
        for mm in (0, 1):
            if ("q", mm) not in _RINGS:
                _RINGS[("q", mm)] = _MrbRing(0, 256 - 2 * pv_accs)
                _RINGS[("p", mm)] = _MrbRing(256 - 2 * pv_accs, 2 * pv_accs)
        pltpu.matmul_push_rhs(qt, staging_register=0, mxu_index=m)
        addr = []
        for j in range(0, bkv, piece):
            a = _RINGS[("q", m)].take(piece // 4)
            pltpu.matmul_acc_lhs(a, kht[j:j + piece], mxu_index=m,
                                 load_staged_rhs=0 if j == 0 else None)
            addr.append((a, m))
        return addr, (kht, vt, acc_ref, kv_head, cs, q_kv_diff, mask, cfgs, k1, piece, bkv if is_v else 0, pv_accs, acc_rows, lanes, 1)
    vsplit = int(os.environ.get("RPA_EXP_VSPLIT", "0"))
    if os.environ.get("RPA_EXP_ALT") == "1" or vsplit:
        m = par % 2
        if vsplit:
            m = 1 if (mask is None and par % vsplit == vsplit - 1) else 0
            valu_keys = bkv if m == 1 else 0
        for mm in (0, 1):
            if ("q", mm) not in _RINGS:
                _RINGS[("q", mm)] = _MrbRing(0, 256 - 2 * pv_accs)
                _RINGS[("p", mm)] = _MrbRing(256 - 2 * pv_accs, 2 * pv_accs)
        pltpu.matmul_push_rhs(qt, staging_register=0, mxu_index=m)
        addr = []
        for j in range(0, bkv, piece):
            a = _RINGS[("q", m)].take(piece // 4)
            pltpu.matmul_acc_lhs(a, kht[j:j + piece], mxu_index=m,
                                 load_staged_rhs=0 if j == 0 else None)
            addr.append((a, m))
        return addr, (kht, vt, acc_ref, kv_head, cs, q_kv_diff, mask, cfgs, k1, piece, valu_keys, pv_accs, acc_rows, lanes, m)
    return _qk_issue(kht, qt, k1, piece, bkv), (kht, vt, acc_ref, kv_head, cs, q_kv_diff, mask, cfgs, k1, piece, valu_keys, pv_accs, acc_rows, lanes, 1)


def _qk_issue(kht, qt, k1, piece, bkv):
    if k1:
        pltpu.matmul_push_rhs(qt, staging_register=0, mxu_index=0)
    if bkv - k1:
        pltpu.matmul_push_rhs(qt, staging_register=0, mxu_index=1)
    addr = []
    m1_every = int(os.environ.get("RPA_EXP_M1_EVERY", "0"))
    loaded = set()
    for i, j in enumerate(range(0, bkv, piece)):
        if m1_every:
            mxu = 1 if i % m1_every == m1_every - 1 else 0
        else:
            mxu = 0 if j < k1 else 1
        a = _RINGS[mxu].take(piece // 4)
        pltpu.matmul_acc_lhs(a, kht[j:j + piece], mxu_index=mxu,
                             load_staged_rhs=None if mxu in loaded else 0)
        loaded.add(mxu)
        addr.append((a, mxu))
    return addr


def _q_chunk_finish(addr, kht, vt, acc_ref, kv_head, cs, q_kv_diff, mask, cfgs, k1, piece, valu_keys, pv_accs, acc_rows, lanes, pm):
    bkv = kht.shape[0]
    pa = _RINGS[("p", pm)].take(pv_accs) if ("p", pm) in _RINGS else _RINGS[2].take(pv_accs)
    ps = []
    for (a, mxu), j in zip(addr, range(0, bkv, piece)):
        sj = pltpu.matmul_pop(a, (piece, lanes), jnp.float32, mxu_index=mxu)
        sj = sj.astype(jnp.bfloat16)
        if mask is not None:
            lo, hi = mask
            diff = q_kv_diff[j:j + piece]
            m = diff >= lo
            if hi is not None:
                m = jnp.logical_and(m, diff < hi)
            sj = jnp.where(m, sj, cfgs.model.mask_value)
        ve = int(os.environ.get("RPA_EXP_VALU_EVERY", "0"))
        on_valu = (ve and mask is None and (j // piece) % ve == ve - 1) or j >= bkv - valu_keys
        ps.append(_exp2_valu_bf16(sj) if on_valu else jnp.exp2(sj))
        if (j + piece) % 256 == 0:
            t = j // 256
            tile = jnp.concatenate(ps, axis=0)
            ps = []
            reg = 1 - t % 2
            pltpu.matmul_push_rhs(tile, staging_register=reg, mxu_index=pm)
            pltpu.matmul_acc_lhs(pa, vt[:, 256 * t:256 * t + 256], mxu_index=pm,
                                 load_staged_rhs=reg)
    o_t = pltpu.matmul_pop(pa, (acc_rows, lanes), jnp.float32, mxu_index=pm)
    acc_ref[kv_head, :, cs] = acc_ref[kv_head, :, cs] + o_t


def _streams_issue(kht, qt_ref, kv_head, cs, cfgs):
    """Two independent streams: keys [0, bkv/2) on MXU 0, [bkv/2, bkv) on MXU 1."""
    piece = int(os.environ.get("RPA_EXP_PIECE", "32"))
    hd = cfgs.aligned_kv_head_dim
    bkv = kht.shape[0]
    half = bkv // 2
    qt = qt_ref[kv_head, :, cs]
    qt = jnp.concatenate([qt, jnp.zeros((256 - hd, qt.shape[1]), qt.dtype)], axis=0)
    for m in (0, 1):
        if ("sq", m) not in _RINGS:
            _RINGS[("sq", m)] = _MrbRing(0, 216)
            _RINGS[("sp", m)] = _MrbRing(216, 40)
    addr = [[], []]
    for m in (0, 1):
        pltpu.matmul_push_rhs(qt, staging_register=0, mxu_index=m)
    for j in range(0, half, piece):
        for m in (0, 1):
            a = _RINGS[("sq", m)].take(piece // 4)
            pltpu.matmul_acc_lhs(a, kht[m * half + j : m * half + j + piece], mxu_index=m,
                                 load_staged_rhs=0 if j == 0 else None)
            addr[m].append(a)
    return addr, piece, half


def _streams_finish(issued, vt, acc_ref, kv_head, cs):
    addr, piece, half = issued
    lanes = 256
    acc_rows = vt.shape[0]
    ve = int(os.environ.get("RPA_EXP_VALU_EVERY", "0"))
    pa = [_RINGS[("sp", m)].take(acc_rows // 4) for m in (0, 1)]
    ps = [[], []]
    tiles = [0, 0]
    n = len(addr[0])
    for i in range(n):
        for m in (0, 1):
            sj = pltpu.matmul_pop(addr[m][i], (piece, lanes), jnp.float32, mxu_index=m)
            sj = sj.astype(jnp.bfloat16)
            on_valu = ve and (i * 2 + m) % ve == ve - 1
            ps[m].append(_exp2_valu_bf16(sj) if on_valu else jnp.exp2(sj))
            keys = len(ps[m]) * piece
            if keys == 256 or i == n - 1:
                t = tiles[m]
                k0 = m * half + t * 256
                tile = jnp.concatenate(ps[m], axis=0)
                lhs = vt[:, k0 : k0 + keys]
                if keys < 256:
                    tile = jnp.concatenate([tile, jnp.zeros((256 - keys, lanes), tile.dtype)], 0)
                    lhs = jnp.concatenate([lhs, jnp.zeros((acc_rows, 256 - keys), lhs.dtype)], 1)
                reg = 1 - t % 2
                pltpu.matmul_push_rhs(tile, staging_register=reg, mxu_index=m)
                pltpu.matmul_acc_lhs(pa[m], lhs, mxu_index=m, load_staged_rhs=reg)
                ps[m] = []
                tiles[m] += 1
    o_t = sum(pltpu.matmul_pop(pa[m], (acc_rows, lanes), jnp.float32, mxu_index=m) for m in (0, 1))
    acc_ref[kv_head, :, cs] = acc_ref[kv_head, :, cs] + o_t


def _streams_fused(kht, vt, qt_ref, acc_ref, kv_head, chunks, cfgs):
    """Two independent key-half streams (MXU 0: keys [0, bkv/2), MXU 1: the rest), each
    MXU's ops in one interleaved program order: pop chunk c's piece i, then issue chunk
    c + 1's piece i; a PV tile reloads the array, so Q^T is pushed again after it."""
    piece = int(os.environ.get("RPA_EXP_PIECE", "32"))
    ve = int(os.environ.get("RPA_EXP_VALU_EVERY", "0"))
    hd = cfgs.aligned_kv_head_dim
    bkv, lanes = kht.shape[0], 256
    half = bkv // 2
    acc_rows = vt.shape[0]
    n = half // piece
    for m in (0, 1):
        if ("fq", m) not in _RINGS:
            _RINGS[("fq", m)] = _MrbRing(0, 216)
            _RINGS[("fp", m)] = _MrbRing(216, 40)

    def qt_of(cs):
        qt = qt_ref[kv_head, :, cs]
        return jnp.concatenate([qt, jnp.zeros((256 - hd, lanes), qt.dtype)], axis=0)

    def issue_piece(m, qt, i, need_load):
        a = _RINGS[("fq", m)].take(piece // 4)
        if need_load:
            pltpu.matmul_push_rhs(qt, staging_register=0, mxu_index=m)
        pltpu.matmul_acc_lhs(a, kht[m * half + i * piece : m * half + (i + 1) * piece],
                             mxu_index=m, load_staged_rhs=0 if need_load else None)
        return a

    cur = None  # (addrs[m][i], cs)
    qts = [qt_of(cs) for cs in chunks]
    for c in range(len(chunks) + 1):
        nxt = [[], []] if c < len(chunks) else None
        pa = [_RINGS[("fp", m)].take(acc_rows // 4) for m in (0, 1)] if cur else None
        ps, tiles, reload = [[], []], [0, 0], [True, True]
        for i in range(n):
            for m in (0, 1):
                if cur is not None:
                    sj = pltpu.matmul_pop(cur[0][m][i], (piece, lanes), jnp.float32, mxu_index=m)
                    sj = sj.astype(jnp.bfloat16)
                    on_valu = ve and (i * 2 + m) % ve == ve - 1
                    ps[m].append(_exp2_valu_bf16(sj) if on_valu else jnp.exp2(sj))
                    keys = len(ps[m]) * piece
                    if keys == 256 or i == n - 1:
                        t = tiles[m]
                        k0 = m * half + t * 256
                        tile = jnp.concatenate(ps[m], axis=0)
                        lhs = vt[:, k0 : k0 + keys]
                        if keys < 256:
                            tile = jnp.concatenate([tile, jnp.zeros((256 - keys, lanes), tile.dtype)], 0)
                            lhs = jnp.concatenate([lhs, jnp.zeros((acc_rows, 256 - keys), lhs.dtype)], 1)
                        pltpu.matmul_push_rhs(tile, staging_register=1, mxu_index=m)
                        pltpu.matmul_acc_lhs(pa[m], lhs, mxu_index=m, load_staged_rhs=1)
                        ps[m] = []
                        tiles[m] += 1
                        reload[m] = True
                if nxt is not None:
                    nxt[m].append(issue_piece(m, qts[c], i, reload[m]))
                    reload[m] = False
        if cur is not None:
            o_t = sum(pltpu.matmul_pop(pa[m], (acc_rows, lanes), jnp.float32, mxu_index=m) for m in (0, 1))
            acc_ref[kv_head, :, cur[1]] = acc_ref[kv_head, :, cur[1]] + o_t
        cur = (nxt, chunks[c]) if nxt is not None else None


_EXP2_C_HALF = (1.0, 0.69314718, 0.24022651, 0.05550411)


def _exp2_valu_bf16(x: jax.Array) -> jax.Array:
    """2^x for bf16 x (|x| < 64) on the VALU: rounding by the 1.5 * 2^7 magic, a cubic
    on [-0.5, 0.5] in bf16, 2^n from the magic sum's low mantissa bits."""
    magic = jnp.bfloat16(192.0)
    t = x + magic
    f = x - (t - magic)
    c0, c1, c2, c3 = _EXP2_C_HALF
    poly = c0 + f * (c1 + f * (c2 + f * c3))
    u = pltpu.bitcast(t, jnp.uint32)
    bits = ((u & 0x007F007F) + 0x003F003F) << 7
    return poly * pltpu.bitcast(bits, jnp.bfloat16)
