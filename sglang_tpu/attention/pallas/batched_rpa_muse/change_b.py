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
"""Wave-22 prefill step body ("Change B-5D"), ported from batched_rpa_v22.

Runs one 512 x 512 (q block, k block) step of a batch_size=1 MIXED launch when
`RpaConfigs.change_b_enabled`. Q, O, m, l and acc keep tokens on lanes:

    q_vref, o_vref:  [1, KV, G, 64, 512]
    m, l scratch:    [KV, G, 8, 512] f32 (m holds bf16 row maxima, bitcast)
    acc scratch:     [KV, G, 64, 512] f32

Each step is split into 256-key x 256-query tiles computed as K.Q^T and
V^T.P, so no Q/P/O transpose is needed. A step takes one of four branches:

    0  fully visible, first k block: sets m from a "pilot" maximum over the
       first 32 keys of the tile and starts l and acc.
    1  fully visible, later k block: accumulates against the stored m.
    2  diagonal block with q and k starts aligned (causal offset 0): three
       tiles, the all-masked (keys 256:512, queries 0:256) tile skipped.
    3  any other partially visible block: all four tiles, all masked.

Numerics: m is set once per q block from the first k block and never
rescaled, so a later score more than 88 (bf16 KV; log2 units) above it is
clipped, and when P is cast to e4m3 (no clip), a later score more than
log2(448) ~ 8.8 above it overflows P. With fp8 KV, Q and P are
cast to e4m3 for the MXU only under SGLANG_TPU_MUSE_RPA_CHANGE_B_FP8_QP=1
(v22's choice); by default fp8 K/V are upcast to bf16 and Q/P stay bf16, as in
the standard kernel, since e4m3 Q/P cost about 6e-4 of output cosine.
"""

import os

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.batched_rpa_muse import configs, utils

_LOG2_E = 1.4426950408889634
_TILE = 256
_PILOT_KEYS = 32
_PIPE_DEPTH = 2


def fp8_qp_enabled() -> bool:
    """Whether fp8-KV steps run Q.K and P.V on the MXU in e4m3 (v22)."""
    return os.environ.get("SGLANG_TPU_MUSE_RPA_CHANGE_B_FP8_QP") == "1"


def change_b_attention(
    step: jax.Array,
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    o_vref: jax.Ref,
    schedule_ref,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    causal_offset: jax.Array,
    is_first_k: jax.Array,
    load_kv_head,
    cfgs: configs.RpaConfigs,
):
    """One change_b step; see the module docstring.

    causal_offset: first key position minus first query position of the step.
    is_first_k: whether this is the q block's first k block.
    load_kv_head: kv_head -> (k [1, 512, 64], v^T [1, 64, 512]) in KV dtype.
    """
    assert cfgs.change_b_enabled
    assert cfgs.bq_sz == cfgs.bkv_sz == 2 * _TILE
    num_kv_heads = cfgs.model.num_kv_heads
    num_g = cfgs.model.num_q_heads_per_kv_head
    num_g_pairs = num_g // 2
    head_dim = cfgs.model.head_dim
    dtype_out = cfgs.serve.dtype_out  # bf16, so 16 m rows bitcast to 8 f32 rows
    rows = _TILE // 16

    qk_scale_val = cfgs.model.sm_scale * _LOG2_E
    if cfgs.serve.scale_k is not None:
        qk_scale_val *= cfgs.serve.scale_k
    v_out_scale = 1.0 if cfgs.serve.scale_v is None else float(cfgs.serve.scale_v)
    is_fp8_kv = cfgs.serve.dtype_kv == jnp.float8_e4m3fn
    fp8_kq = fp8_vp = is_fp8_kv and fp8_qp_enabled()
    # bf16 keeps the first tile's l in VMEM ("nodict"); fp8 keeps it in vregs.
    use_nodict = not fp8_vp
    kq_dtype = jnp.float8_e4m3fn if fp8_kq else dtype_out
    v_dtype = jnp.float8_e4m3fn if fp8_vp else dtype_out
    mask_value = jnp.asarray(cfgs.model.mask_value, dtype_out)

    def _load_kv(kv_head):
        k_3d, v_t_3d = load_kv_head(kv_head)
        return k_3d.astype(kq_dtype), v_t_3d.astype(v_dtype)

    def _prescale_q(q_t: jax.Array) -> jax.Array:
        if qk_scale_val == 1.0:
            return q_t.astype(kq_dtype)
        if fp8_kq:
            return (q_t * jnp.bfloat16(qk_scale_val)).astype(kq_dtype)
        return (q_t.astype(jnp.float32) * qk_scale_val).astype(kq_dtype)

    def _load_q(kv_head, g, t_sl):
        return _prescale_q(q_vref[0, kv_head, g : g + 1, :, t_sl])

    def _kq_f32(k_mat: jax.Array, q_t: jax.Array) -> jax.Array:
        # [1, keys, d] x [1, d, queries] -> [1, keys, queries]
        return lax.dot(
            k_mat,
            q_t,
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )

    def _exp2_rowsum(diff: jax.Array):
        if not fp8_vp:
            diff = jnp.minimum(diff, jnp.bfloat16(88.0))
        p = jnp.exp2(diff)
        # Two-stage key reduction: 16 rows in bf16, the rest in f32.
        p_sum16 = jnp.sum(p.reshape(1, 16, rows, _TILE), axis=1, dtype=dtype_out)
        p_sum = jnp.broadcast_to(
            jnp.sum(p_sum16, axis=-2, keepdims=True, dtype=jnp.float32),
            (1, 8, _TILE),
        )
        if fp8_vp:
            p = p.astype(jnp.float8_e4m3fn)
        return p, p_sum

    def _sub_m(qk: jax.Array, m_bf16: jax.Array) -> jax.Array:
        return (qk.reshape(1, 16, rows, _TILE) - m_bf16[:, None, :, :]).reshape(
            1, _TILE, _TILE
        )

    def _softmax_accum(qk_f32, m_bf16, mask=None):
        qk = qk_f32.astype(dtype_out)
        if mask is not None:
            qk = jnp.where(mask, qk, mask_value)
        return _exp2_rowsum(_sub_m(qk, m_bf16))

    def _softmax_pilot(qk_f32, mb_in, l_in, first, mask=None, first_k=None):
        """first=True: unconditional first tile; else select on first_k."""
        qk = qk_f32.astype(dtype_out)
        if mask is not None:
            qk = jnp.where(mask, qk, mask_value)
        m_pilot = jnp.broadcast_to(
            jnp.max(qk[:, :_PILOT_KEYS, :], axis=-2, keepdims=True), (1, 16, _TILE)
        )
        if first:
            p, l_next = _exp2_rowsum(_sub_m(qk, m_pilot))
            return p, m_pilot, pltpu.bitcast(m_pilot, jnp.float32), l_next
        m_next = jnp.where(first_k, m_pilot, mb_in)
        p, p_sum = _exp2_rowsum(_sub_m(qk, m_next))
        l_next = jnp.where(first_k, p_sum, l_in + p_sum)
        return p, m_next, pltpu.bitcast(m_next, jnp.float32), l_next

    def _vp(p, v_t, o_in=None, first_k=None):
        vp = lax.dot(
            v_t,
            p,
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
        if o_in is None:
            return vp
        if first_k is None:
            return o_in + vp
        return jnp.where(first_k, vp, o_in + vp)

    def _m_ref(kv_h, g, t_sl):
        return m_scratch_ref.at[kv_h : kv_h + 1, g, :, t_sl]

    def _l_ref(kv_h, g, t_sl):
        return l_scratch_ref.at[kv_h : kv_h + 1, g, :, t_sl]

    def _acc_ref(kv_h, g, t_sl):
        return acc_scratch_ref.at[kv_h : kv_h + 1, g, :, t_sl]

    def _read_m(kv_h, g, t_sl):
        return pltpu.bitcast(_m_ref(kv_h, g, t_sl)[...], dtype_out)

    def _drain(mrb_q, advance, flush, state):
        while mrb_q:
            if state["ready"] is not None:
                flush()
            advance()
        if state["ready"] is not None:
            flush()

    def _run_unmasked_first():
        mrb_q = []
        state = {"ready": None}
        mb_reg = {}
        l_reg = {}

        def advance():
            qk0, qk1, kv_h, t_idx, t_sl, g0, g1, s_idx, vt_s = mrb_q.pop(0)
            outs = []
            for g, qk in ((g0, qk0), (g1, qk1)):
                if s_idx == 0:
                    p, mb, m_f32, l0 = _softmax_pilot(qk, None, None, first=True)
                    _m_ref(kv_h, g, t_sl)[...] = m_f32
                    if use_nodict:
                        _l_ref(kv_h, g, t_sl)[...] = l0
                    else:
                        mb_reg[(kv_h, t_idx, g)] = mb
                        l_reg[(kv_h, t_idx, g)] = l0
                elif use_nodict:
                    p, ps = _softmax_accum(qk, _read_m(kv_h, g, t_sl))
                    _l_ref(kv_h, g, t_sl)[...] = _l_ref(kv_h, g, t_sl)[...] + ps
                else:
                    p, ps = _softmax_accum(qk, mb_reg.pop((kv_h, t_idx, g)))
                    _l_ref(kv_h, g, t_sl)[...] = l_reg.pop((kv_h, t_idx, g)) + ps
                outs.append(p)
            state["ready"] = (outs, kv_h, t_sl, (g0, g1), s_idx, vt_s)

        def flush():
            outs, kv_h, t_sl, gs, s_idx, vt_s = state["ready"]
            for g, p in zip(gs, outs):
                acc = _acc_ref(kv_h, g, t_sl)
                acc[...] = _vp(p, vt_s) if s_idx == 0 else _vp(p, vt_s, acc[...])
            state["ready"] = None

        for kv_head in range(num_kv_heads):
            k_3d, v_t_3d = _load_kv(kv_head)
            for t_idx in range(cfgs.bq_sz // _TILE):
                t_sl = slice(t_idx * _TILE, (t_idx + 1) * _TILE)
                for gp in range(num_g_pairs):
                    g0, g1 = 2 * gp, 2 * gp + 1
                    q_t0 = _load_q(kv_head, g0, t_sl)
                    q_t1 = _load_q(kv_head, g1, t_sl)
                    for s_idx in range(cfgs.bkv_sz // _TILE):
                        k_sub = k_3d[:, s_idx * _TILE : (s_idx + 1) * _TILE, :]
                        v_t_sub = v_t_3d[:, :, s_idx * _TILE : (s_idx + 1) * _TILE]
                        qk0 = _kq_f32(k_sub, q_t0)
                        qk1 = _kq_f32(k_sub, q_t1)
                        if state["ready"] is not None:
                            flush()
                        mrb_q.append(
                            (qk0, qk1, kv_head, t_idx, t_sl, g0, g1, s_idx, v_t_sub)
                        )
                        if len(mrb_q) >= _PIPE_DEPTH:
                            advance()
        _drain(mrb_q, advance, flush, state)

    def _run_unmasked_accum():
        mrb_q = []
        state = {"ready": None}

        def advance():
            qk0, qk1, mb0, mb1, kv_h, t_sl, g0, g1, vt_s = mrb_q.pop(0)
            outs = []
            for g, qk, mb in ((g0, qk0, mb0), (g1, qk1, mb1)):
                p, ps = _softmax_accum(qk, mb)
                _l_ref(kv_h, g, t_sl)[...] = _l_ref(kv_h, g, t_sl)[...] + ps
                outs.append(p)
            state["ready"] = (outs, kv_h, t_sl, (g0, g1), vt_s)

        def flush():
            outs, kv_h, t_sl, gs, vt_s = state["ready"]
            for g, p in zip(gs, outs):
                acc = _acc_ref(kv_h, g, t_sl)
                acc[...] = _vp(p, vt_s, acc[...])
            state["ready"] = None

        for kv_head in range(num_kv_heads):
            k_3d, v_t_3d = _load_kv(kv_head)
            for t_idx in range(cfgs.bq_sz // _TILE):
                t_sl = slice(t_idx * _TILE, (t_idx + 1) * _TILE)
                for gp in range(num_g_pairs):
                    g0, g1 = 2 * gp, 2 * gp + 1
                    q_t0 = _load_q(kv_head, g0, t_sl)
                    q_t1 = _load_q(kv_head, g1, t_sl)
                    mb0 = _read_m(kv_head, g0, t_sl)
                    mb1 = _read_m(kv_head, g1, t_sl)
                    for s_idx in range(cfgs.bkv_sz // _TILE):
                        k_sub = k_3d[:, s_idx * _TILE : (s_idx + 1) * _TILE, :]
                        v_t_sub = v_t_3d[:, :, s_idx * _TILE : (s_idx + 1) * _TILE]
                        qk0 = _kq_f32(k_sub, q_t0)
                        qk1 = _kq_f32(k_sub, q_t1)
                        if state["ready"] is not None:
                            flush()
                        mrb_q.append(
                            (qk0, qk1, mb0, mb1, kv_head, t_sl, g0, g1, v_t_sub)
                        )
                        if len(mrb_q) >= _PIPE_DEPTH:
                            advance()
        _drain(mrb_q, advance, flush, state)

    def _tile_mask(s_idx, t_idx):
        # Key 256*s + a is visible to query 256*t + b iff
        # causal_offset + 256*s + a <= 256*t + b.
        k_iota = lax.broadcasted_iota(jnp.int32, (1, _TILE, _TILE), 1)
        q_iota = lax.broadcasted_iota(jnp.int32, (1, _TILE, _TILE), 2)
        return k_iota <= q_iota - (causal_offset + _TILE * (s_idx - t_idx))

    def _run_masked(tiles, use_dict):
        """tiles: (s_idx, t_idx, masked) in an order where (0, t) precedes (1, t)."""
        mrb_q = []
        state = {"ready": None}
        mb_reg = {}
        l_reg = {}
        masks = {}
        # Columns whose (1, t) tile reads the (0, t) tile's m and l from vregs.
        followed = {t for s, t, _ in tiles if s == 1}

        def advance():
            qk0, qk1, kv_h, s_idx, t_idx, t_sl, g0, g1, vt_s = mrb_q.pop(0)
            mask = masks[(s_idx, t_idx)]
            outs = []
            for g, qk in ((g0, qk0), (g1, qk1)):
                if s_idx == 0:
                    p, mb, m_f32, l_next = _softmax_pilot(
                        qk,
                        _read_m(kv_h, g, t_sl),
                        _l_ref(kv_h, g, t_sl)[...],
                        first=False,
                        mask=mask,
                        first_k=is_first_k,
                    )
                    _m_ref(kv_h, g, t_sl)[...] = m_f32
                    if use_dict and t_idx in followed:
                        mb_reg[(kv_h, t_idx, g)] = mb
                        l_reg[(kv_h, t_idx, g)] = l_next
                    else:
                        _l_ref(kv_h, g, t_sl)[...] = l_next
                    mode = "pilot"
                elif use_dict:
                    p, ps = _softmax_accum(qk, mb_reg.pop((kv_h, t_idx, g)), mask)
                    _l_ref(kv_h, g, t_sl)[...] = l_reg.pop((kv_h, t_idx, g)) + ps
                    mode = "accum"
                else:
                    p, ps = _softmax_accum(qk, _read_m(kv_h, g, t_sl), mask)
                    _l_ref(kv_h, g, t_sl)[...] = _l_ref(kv_h, g, t_sl)[...] + ps
                    mode = "accum"
                outs.append(p)
            state["ready"] = (outs, kv_h, t_sl, (g0, g1), vt_s, mode)

        def flush():
            outs, kv_h, t_sl, gs, vt_s, mode = state["ready"]
            for g, p in zip(gs, outs):
                acc = _acc_ref(kv_h, g, t_sl)
                first_k = is_first_k if mode == "pilot" else None
                acc[...] = _vp(p, vt_s, acc[...], first_k)
            state["ready"] = None

        for s_idx, t_idx, masked in tiles:
            masks[(s_idx, t_idx)] = _tile_mask(s_idx, t_idx) if masked else None
        for kv_head in range(num_kv_heads):
            k_3d, v_t_3d = _load_kv(kv_head)
            for s_idx, t_idx, _ in tiles:
                t_sl = slice(t_idx * _TILE, (t_idx + 1) * _TILE)
                k_sub = k_3d[:, s_idx * _TILE : (s_idx + 1) * _TILE, :]
                v_t_sub = v_t_3d[:, :, s_idx * _TILE : (s_idx + 1) * _TILE]
                for gp in range(num_g_pairs):
                    g0, g1 = 2 * gp, 2 * gp + 1
                    qk0 = _kq_f32(k_sub, _load_q(kv_head, g0, t_sl))
                    qk1 = _kq_f32(k_sub, _load_q(kv_head, g1, t_sl))
                    if state["ready"] is not None:
                        flush()
                    mrb_q.append(
                        (qk0, qk1, kv_head, s_idx, t_idx, t_sl, g0, g1, v_t_sub)
                    )
                    if len(mrb_q) >= _PIPE_DEPTH:
                        advance()
        _drain(mrb_q, advance, flush, state)

    def _run_masked_diag():
        # causal_offset == 0: keys 256:512 never reach queries 0:256, and keys
        # 0:256 reach all of queries 256:512.
        _run_masked(((0, 0, True), (0, 1, False), (1, 1, True)), not use_nodict)

    def _run_masked_any():
        _run_masked(
            ((0, 0, True), (0, 1, True), (1, 0, True), (1, 1, True)), not use_nodict
        )

    fully_visible = causal_offset + (cfgs.bkv_sz - 1) <= 0
    branch_idx = jnp.where(
        fully_visible,
        jnp.where(is_first_k, 0, 1),
        jnp.where(causal_offset == 0, 2, 3),
    ).astype(jnp.int32)
    lax.switch(
        branch_idx,
        [_run_unmasked_first, _run_unmasked_accum, _run_masked_diag, _run_masked_any],
    )

    @pl.when(schedule_ref.is_last_k[step, 0] == 1)
    @jax.named_scope("store_out_chunks")
    def _store_out_chunks():
        for kv_head in range(num_kv_heads):
            for t_idx in range(cfgs.bq_sz // _TILE):
                t_sl = slice(t_idx * _TILE, (t_idx + 1) * _TILE)
                for g in range(num_g):
                    acc_g = acc_scratch_ref[kv_head, g, :, t_sl]
                    l_g = l_scratch_ref[kv_head, g, :, t_sl]
                    l_safe = jnp.where(l_g > 0.0, l_g, 1.0)
                    inv_l = pl.reciprocal(l_safe[None, :, :], approx=True)
                    if v_out_scale != 1.0:
                        inv_l = inv_l * v_out_scale
                    out_g = (acc_g.reshape(head_dim // 8, 8, _TILE) * inv_l).reshape(
                        head_dim, _TILE
                    )
                    o_vref[0, kv_head, g, :, t_sl] = out_g.astype(dtype_out)


def load_kv_head_head_along_sublane(kv_in_vref, strided_load_bkv, cfgs):
    """kv_head -> (k [1, bkv, d], v^T [1, d, bkv]) from the token-major buffer."""
    heads_per_load = cfgs.serve.packing_kv // 2

    def load(kv_head):
        start = (kv_head // heads_per_load) * heads_per_load * 2
        k_raw, v_raw = strided_load_bkv(kv_in_vref, 0, start, cfgs=cfgs)[
            kv_head % heads_per_load
        ]
        d = cfgs.model.head_dim
        return k_raw[None, :, :d], jnp.swapaxes(v_raw[:, :d], 0, 1)[None]

    return load


def load_kv_head_seq_along_lane(kv_in_vref, cfgs):
    """kv_head -> (k [1, bkv, d], v^T [1, d, bkv]) from the lane-major buffer."""
    pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv
    lanes = cfgs.bkv_sz + 2 * cfgs.serve.page_size
    d = cfgs.model.head_dim

    def load_one(kv_row):
        ref = kv_in_vref.at[0, kv_row].bitcast(jnp.uint32).reshape(-1, lanes)
        x = utils.strided_load(ref, 0, pack_dim, 1, dtype=cfgs.serve.dtype_kv)
        return x[:, : cfgs.bkv_sz].reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz)[:d]

    def load(kv_head):
        k_t = load_one(kv_head * 2)
        v_t = load_one(kv_head * 2 + 1)
        return jnp.swapaxes(k_t, 0, 1)[None], v_t[None]

    return load
