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
  MXU op per 32 keys. A bf16 q would upcast K and cost 3-4x, so Q^T is stored
  as two fp8 terms hi + lo (about bf16 precision) and S^T = [K; K]^T [hi; lo]
  is one fp8 matmul whose contraction 2 * head_dim still fits one MXU pass.
  The fp8 MXU result itself is only about bf16 precise (not f32 accumulated),
  so scores stay noisier than the standard tile's.

One compute block is bq_c tokens x q heads = 128k lanes; scratch carries are
m [KV, 1, rows] and acc [KV, head_dim + 16, rows] in f32, and Q^T is built
once per (sequence, q block) in VMEM.
"""

import functools
import math
from collections.abc import Callable

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.batched_rpa import configs, utils

LOG2E = 1.0 / math.log(2.0)


def refresh_qt(
    load_q: Callable[[], jax.Array],  # -> [1, KV, rows, aligned_q_head_dim]
    qt_ref: jax.Ref,  # [KV, 2 * aligned_kv_head_dim, rows]: hi, then lo
    qkey_ref: jax.Ref,  # SMEM [2]: (s_idx, q_idx) the cached Q^T belongs to
    s_idx: jax.Array,
    q_idx: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    """Rebuilds Q^T = [hi; lo] (fp8) when the step moved to another (sequence, q block)."""
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
            # Scores reach ~44 in log2 units: a lone e4m3 q (3 mantissa bits) puts
            # whole units of error into exp2; lo brings q to ~2^-8.
            hi = qt.astype(qt_ref.dtype)
            lo = (qt - hi.astype(jnp.float32)).astype(qt_ref.dtype)
            qt_ref[kv_head] = jnp.concatenate([hi, lo], axis=0)
        qkey_ref[0] = s_idx
        qkey_ref[1] = q_idx


def skip_group_tokens(cfgs: configs.RpaConfigs) -> int:
    """Query tokens per partial-block loop iteration: skip_pad_q_group rounded down
    to whole compute blocks dividing the q block."""
    per = max(1, cfgs.serve.skip_pad_q_group // cfgs.bq_c_sz)
    n_blocks = cfgs.bq_sz // cfgs.bq_c_sz
    while n_blocks % per:
        per -= 1
    return per * cfgs.bq_c_sz


def tile(
    step: jax.Array,
    k: jax.Array,  # [1, KV, aligned_kv_head_dim, bkv] fp8
    v: jax.Array,  # [1, KV, aligned_kv_head_dim, bkv] fp8
    schedule_ref,
    m_ref: jax.Ref,  # [KV, 1, rows] f32, base-2 running max
    acc_ref: jax.Ref,  # [KV, acc_rows, rows] f32: O^T, then l in row aligned_kv_head_dim
    qt_ref: jax.Ref,  # [KV, 2 * aligned_kv_head_dim, rows] fp8: hi, then lo
    *,
    processed_q_len: jax.Array,
    processed_kv_len: jax.Array,
    apply_mask: bool,
    cfgs: configs.RpaConfigs,
    q_rows: jax.Array | None = None,
):
    """One k step of the q block. With q_rows (skip_pad_q, a partial q block) only
    the groups of compute blocks up to the last real query run.

    With no_max this is the unshifted streaming tile (no running max, no alpha;
    store_tile() clears acc after each q block); else the online-softmax tile.
    """
    heads = cfgs.aligned_num_q_heads_per_kv_head
    hd = cfgs.aligned_kv_head_dim
    bkv = cfgs.bkv_sz
    lanes = cfgs.bq_c_sz * heads
    offset = processed_kv_len - processed_q_len  # kv index of the first query

    if apply_mask:
        kv_iota = lax.broadcasted_iota(jnp.int32, (bkv, lanes), 0)
        q_iota = lax.broadcasted_iota(jnp.int32, (bkv, lanes), 1) // heads
        q_kv_diff = q_iota - kv_iota

    def mask_scores(s, diff, bq_start):
        # key index kv_iota + processed_kv_len <= query index
        # q_iota + bq_start + processed_q_len, i.e. q_kv_diff >= offset - bq_start.
        mask = diff >= offset - bq_start
        if (window := cfgs.model.sliding_window) is not None:
            mask = jnp.logical_and(mask, diff < window + offset - bq_start)
        return jnp.where(mask, s, cfgs.model.mask_value)

    group_tokens = skip_group_tokens(cfgs)
    per_group = group_tokens // cfgs.bq_c_sz
    n_groups = cfgs.bq_sz // group_tokens
    group_lanes = group_tokens * heads
    grouped = q_rows is not None and n_groups > 1
    live_groups = (
        jnp.clip((q_rows + group_tokens - 1) // group_tokens, 1, n_groups)
        if grouped
        else None
    )
    n_qc = cfgs.bq_sz // cfgs.bq_c_sz

    def block_shifted(kv_head, kh2, vt, bq_start, cs):
        """One compute block: queries [bq_start, bq_start + bq_c_sz) of the q block."""
        qt = qt_ref[kv_head, :, cs]  # [2 * hd, lanes] fp8
        s = lax.dot_general(
            kh2, qt, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )  # [bkv, lanes]: K^T hi + K^T lo
        s = s.astype(jnp.bfloat16)
        if apply_mask:
            s = mask_scores(s, q_kv_diff, bq_start)
        m_prev = m_ref[kv_head, :, cs]  # [1, lanes]
        m_cur = jnp.max(s, axis=0, keepdims=True).astype(jnp.float32)
        m_new = jnp.maximum(m_prev, m_cur)
        alpha = jnp.exp2(m_prev - m_new)
        p = jnp.exp2(s - m_new.astype(jnp.bfloat16))  # [bkv, lanes] bf16
        o_t = lax.dot_general(
            vt, p, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )  # [acc rows, lanes]: O^T rows, then the row sums
        acc_ref[kv_head, :, cs] = alpha * acc_ref[kv_head, :, cs] + o_t
        # A finished q block starts its next one from -inf (alpha == 0 then
        # clears acc and l).
        m_ref[kv_head, :, cs] = jnp.where(
            schedule_ref.is_last_k[step, 0], -jnp.inf, m_new
        )

    # Key chunks whose scores and probabilities stay in vregs.
    kc = math.gcd(bkv, min(bkv, 1536))

    def block_streaming(kv_head, kh2, vt, bq_start, cs):
        qt = qt_ref[kv_head, :, cs]  # [2 * hd, lanes] fp8
        o_t = None
        for c0 in range(0, bkv, kc):
            s = lax.dot_general(
                kh2[:, c0 : c0 + kc],
                qt,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)  # [kc, lanes]
            if apply_mask:
                s = mask_scores(s, q_kv_diff[c0 : c0 + kc], bq_start)
            # Bounded scores (QK-normed): exp2 needs no max shift.
            p = jnp.exp2(s)
            part = lax.dot_general(
                vt[:, c0 : c0 + kc],
                p,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [acc rows, lanes]
            o_t = part if o_t is None else o_t + part
        acc_ref[kv_head, :, cs] = acc_ref[kv_head, :, cs] + o_t

    block = block_streaming if cfgs.serve.no_max else block_shifted

    for kv_head in range(cfgs.model.num_kv_heads):
        kh = k[0, kv_head]  # [hd, bkv] fp8, contracted on dim 0: the MXU LHS
        kh2 = jnp.concatenate([kh, kh], axis=0)  # [2 * hd, bkv]
        ones_rows = cfgs.acct_scratch_shape[1] - hd
        vt = jnp.concatenate(
            [
                v[0, kv_head].astype(jnp.bfloat16),
                jnp.ones((ones_rows, bkv), jnp.bfloat16),
            ],
            axis=0,
        )  # [acc rows, bkv]
        run = functools.partial(block, kv_head, kh2, vt)

        if grouped:
            # Streaming: a few compute blocks per iteration; shifted: one group.
            sub = math.gcd(per_group, 2 if apply_mask else 4)
            sub = sub if cfgs.serve.no_max else per_group

            def run_sub(i, carry, run=run, sub=sub):
                for j in range(sub):
                    bq_start = i * sub * cfgs.bq_c_sz + j * cfgs.bq_c_sz
                    lane0 = pl.multiple_of((i * sub + j) * lanes, lanes)
                    run(bq_start, pl.ds(lane0, lanes))
                return carry

            lax.fori_loop(0, live_groups * (per_group // sub), run_sub, 0)
            if not cfgs.serve.no_max:
                # Skipped groups start the next q block from -inf.
                def reset_group(g, carry, kv_head=kv_head):
                    lane0 = pl.multiple_of(g * group_lanes, group_lanes)
                    m_ref[kv_head, :, pl.ds(lane0, group_lanes)] = jnp.full(
                        (1, group_lanes), -jnp.inf, jnp.float32
                    )
                    return carry

                lax.fori_loop(live_groups, n_groups, reset_group, 0)
        elif (
            cfgs.serve.no_max
            and apply_mask
            and n_qc > 1
            and n_qc * cfgs.model.num_kv_heads > 32
        ):
            # Masked tiles run once or twice per q block: rolled, they keep IMEM
            # in budget at bq=1024 while the hot full tile stays unrolled.
            def body(i, carry, run=run):
                run(i * cfgs.bq_c_sz, pl.ds(pl.multiple_of(i * lanes, lanes), lanes))
                return carry

            lax.fori_loop(0, n_qc, body, 0, unroll=min(4, n_qc))
        else:
            for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
                run(bq_start, slice(bq_start * heads, bq_start * heads + lanes))


def store_tile(
    step: jax.Array,
    schedule_ref,
    acc_ref: jax.Ref,
    o_vref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Normalizes and writes the completed q block on its final k step.

    Emitted once after the full / masked tile dispatch, not per branch, to keep
    IMEM in budget at bq=1024.
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
                # Skipped groups keep l == 0: output 0, not NaN. Real rows see
                # their own key, so l >= 2^-64.
                l = jnp.maximum(l, 1e-30)
            o_t = acc[:hd] * pltpu.reciprocal(l, approx=True)
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
            # No alpha == 0 restart without a running max.
            acc_ref[...] = jnp.zeros_like(acc_ref)

    lax.cond(is_last_k, store, lambda: None)
