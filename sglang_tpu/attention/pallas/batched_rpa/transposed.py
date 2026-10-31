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

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.batched_rpa import configs, utils

LOG2E = 1.0 / math.log(2.0)


def refresh_qt(
    q: jax.Array,  # [1, KV, rows, aligned_q_head_dim]
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
):
    heads = cfgs.aligned_num_q_heads_per_kv_head
    hd = cfgs.aligned_kv_head_dim
    bkv = cfgs.bkv_sz
    lanes = cfgs.bq_c_sz * heads
    rows = cfgs.bq_sz * heads
    offset = processed_kv_len - processed_q_len  # kv index of the first query

    if apply_mask:
        kv_iota = lax.broadcasted_iota(jnp.int32, (bkv, lanes), 0)
        q_iota = lax.broadcasted_iota(jnp.int32, (bkv, lanes), 1) // heads
        q_kv_diff = q_iota - kv_iota

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
        for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
            cs = slice(bq_start * heads, bq_start * heads + lanes)
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
            o_t = lax.dot_general(
                vt, p, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )  # [acc rows, lanes]: O^T rows, then the row sums
            acc_ref[kv_head, :, cs] = alpha * acc_ref[kv_head, :, cs] + o_t
            # A finished q block starts its next one from -inf (alpha == 0 then
            # clears acc and l).
            m_ref[kv_head, :, cs] = jnp.where(
                schedule_ref.is_last_k[step, 0], -jnp.inf, m_new
            )

    is_last_k = schedule_ref.is_last_k[step, 0] == 1

    def store():
        outs = []
        for kv_head in range(cfgs.model.num_kv_heads):
            acc = acc_ref[kv_head]  # [acc rows, rows]
            l = acc[hd : hd + 1]
            o_t = acc[:hd] * pl.reciprocal(l, approx=True)
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

    lax.cond(is_last_k, store, lambda: None)
