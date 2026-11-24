# SPDX-License-Identifier: Apache-2.0
"""The per-device body of the SP + TP prefill program (prototype, not wired into serving).

Dense path as prefill/spep (rank m owns rows [m * Ts, (m + 1) * Ts), Ts = rows / tp), MoE as
prefill/tp (expert-TP fused-RS, every rank 1/tp of every expert's intermediate). Per layer:

    own rows --AG (e4m3 + row scale)--> qkvg and attention on every row with this rank's heads
      -> gated output --a2a by row (e4m3 + row scale)--> o proj on own rows (replicated wo)
      -> attention residual, router + replicated pool + pre-expert norm on own rows
      -> AG of the expert input (e4m3 + row scale) and routing: every row
      -> fused-RS expert-TP MoE: this rank's rows of the combined output
      -> replicated unpool and FFN residual on own rows

Weights: wqkvg this rank's heads (as both programs); wo / wpool / wunpool replicated as
prefill/spep; experts in prefill/tp's layout (expert units, W4A16 layers).
"""

import os

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel_v2.common import plan
from sglang_tpu.muse_megakernel_v2.common.config import PFConfig
from sglang_tpu.muse_megakernel_v2.common.consts import LANE, small_row
from sglang_tpu.muse_megakernel_v2.common.numerics import _rms, heads_rms_xla
from sglang_tpu.muse_megakernel_v2.common.qkv import qkv_from_proj
from sglang_tpu.muse_megakernel_v2.common.trace import _scope
from sglang_tpu.muse_megakernel_v2.prefill.spep.program import (
    _BF16,
    _F32,
    _bf,
    _fp8_rows,
    ag,
    attention,
    dense,
    residual,
    route,
    to_owner,
)
from sglang_tpu.muse_megakernel_v2.prefill.sptp.qkvg import (
    _heads_rms_mxu,
    dequant_rows_call,
    qkvg_call,
)
from sglang_tpu.muse_megakernel_v2.prefill.sptp.rpa import rpa_prepared
from sglang_tpu.muse_megakernel_v2.prefill.tp.program import (
    _layer_experts,
    fused_rs_moe,
)

# Numerics debugging only: bf16 activations (scale 1) where the program quantizes rows to e4m3,
# to separate layout bugs from e4m3 rounding. The recipe sets it at build time.
DEBUG_BF16_ACTIVATIONS = False
# row tile of the Pallas qkvg card
QKVG_TILE = 256
# x_norm AG chunks overlapped with the chunked qkvg card, and that card's row tile
XAG_CHUNKS = 4
XAG_TILE = 128


def _bf16_rows(x):
    return x.astype(_BF16), jnp.ones((x.shape[0], 1), _F32)


def route_top_k(c: PFConfig, ffn, wrouter, sm):
    """prefill/spep's route with XLA's TopK instead of top_k argmax rounds: the same picks (ties
    to the lower id) and order."""
    ep = wrouter.shape[1]
    sc = jax.nn.sigmoid(
        jnp.dot(ffn, wrouter, preferred_element_type=_F32) * c.router_mult
    )
    _, ids = jax.lax.top_k(sc + small_row(sm, "expert_bias")[:, :ep], c.top_k)
    w = jnp.take_along_axis(sc, ids, axis=1)
    return ids, w / (jnp.sum(w, axis=1, keepdims=True) + c.router_eps) * c.route_scale


def _rms_fp8_rows(x, eps: float, w):
    """_fp8_rows(_rms(x, eps) * w) with the row amax taken in the rms pass: rms scales a row by
    one positive factor, so amax = rsqrt(ms + eps) * max|x * w| (two passes over x, not three)."""
    x32 = x.astype(_F32)
    r = jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    m = jnp.max(jnp.abs(x32 * w), axis=-1, keepdims=True)
    s = jnp.maximum(m * r, 1e-30) / 448.0
    return (x32 * (r / s) * w).astype(jnp.float8_e4m3fn), s


def n_expert_refs(c: PFConfig) -> int:
    """prefill/tp's expert operands: (w13, s13, w2, s2) per expert unit."""
    return plan.n_expert_refs(c)


def build_forward(c: PFConfig, axis: str = "model"):
    """prefill/tp's (meta, pos, kv_lens, cu_q, dist, pf, psw, emb, *weights, *experts, *kv) ->
    (hidden, aux, *kv), dense path sequence-parallel, MoE expert-TP."""
    c.validate()
    if c.sp or c.fp8_gather:
        raise ValueError("SP + TP takes prefill/tp's config without sp / fp8_gather")
    L, T, tp, hd = c.n_layers, c.rows, c.tp, c.head_dim
    Ts = T // tp
    H, He, qd = c.hidden, c.expert_hidden, c.q_dim
    if Ts % 8:
        raise ValueError(f"{T} rows: rows / tp must be a multiple of 8")
    ne = n_expert_refs(c)
    layer_experts = _layer_experts(c)
    q8 = _bf16_rows if DEBUG_BF16_ACTIVATIONS else _fp8_rows
    # SPTP_RMS_QUANT_3PASS=1: rms, then a separate amax pass, then the cast
    fused_rq = (
        not DEBUG_BF16_ACTIVATIONS
        and os.environ.get("SPTP_RMS_QUANT_3PASS", "0") != "1"
    )

    def rms_q8(x, w):
        return (
            _rms_fp8_rows(x, c.rms_eps, w) if fused_rq else q8(_rms(x, c.rms_eps) * w)
        )

    # SPTP_OGATE_MXU_NORM=1: the o gate head norm as MXU dots (measured 0.8 ms / chunk slower
    # than XLA's 3-D norm with its relayout copy)
    ogate_mxu = os.environ.get("SPTP_OGATE_MXU_NORM", "0") == "1"
    # SPTP_A2A_U8=1: the gated output a2a as bytes (prefill/spep) instead of e4m3 itself
    a2a_u8 = os.environ.get("SPTP_A2A_U8", "0") == "1"

    def to_owner_x(x):
        if a2a_u8 or x.dtype != jnp.float8_e4m3fn:
            return to_owner(x, tp, axis)
        b = jax.lax.all_to_all(x, axis, 0, 0, tiled=True).reshape(tp, Ts, -1)
        return b.transpose(1, 0, 2)

    # SPTP_ROUTE_TOPK=1: XLA TopK (same picks; measured 1.6 ms / chunk slower than the rounds)
    router = route_top_k if os.environ.get("SPTP_ROUTE_TOPK", "0") == "1" else route
    # prototype A/B knobs, read at build time (anvil --env): SPTP_XLA_QKVG=1 the XLA qkvg card
    xla_qkvg = DEBUG_BF16_ACTIVATIONS or os.environ.get("SPTP_XLA_QKVG", "0") == "1"
    # SPTP_QKVG_SLICE_NORM=1: per-head lane-slice norms (common/qkv) in the Pallas card
    mxu_norm = os.environ.get("SPTP_QKVG_SLICE_NORM", "0") != "1"
    qkvg_tile = int(os.environ.get("SPTP_QKVG_TILE", str(QKVG_TILE)))
    kc = int(os.environ.get("SPTP_QKVG_KC", "512"))
    # SPTP_RPA_WRAPPER=1: q [T, H, D] through batched_rpa.wrapper (its relayouts) instead of
    # the qkvg card writing the kernel's query layout and reading its raw output (sptp/rpa.py)
    q_rpa = not xla_qkvg and os.environ.get("SPTP_RPA_WRAPPER", "0") != "1"
    qkvg_k = {r: qkvg_call(c, qkvg_tile, r, mxu_norm, kc, q_rpa) for r in (True, False)}
    # SPTP_XAG_CHUNKS=n: the x_norm AG in n row chunks, each chunk's qkvg as it lands (the
    # next chunk's AG in flight meanwhile); 1 = one AG, then qkvg
    n_xag = int(os.environ.get("SPTP_XAG_CHUNKS", str(XAG_CHUNKS)))
    xag_tile = int(os.environ.get("SPTP_XAG_TILE", str(XAG_TILE)))
    qkvg_c = {r: qkvg_call(c, xag_tile, r, mxu_norm, kc, q_rpa) for r in (True, False)}
    gq = c.q_heads // c.kv_heads
    # SPTP_XE_XLA_DEQUANT=1: XLA's xe dequant (transposed layout + a conversion copy)
    xe_pallas = (
        not DEBUG_BF16_ACTIVATIONS and os.environ.get("SPTP_XE_XLA_DEQUANT", "0") != "1"
    )
    dequant = dequant_rows_call(T, He)

    def forward(
        meta,
        pos,
        kv_lens,
        cu_q,
        dist,
        pf,
        psw,
        emb,
        rope,
        final_norm,
        smalls,
        wqkvg,
        wo,
        wpool,
        wrouter,
        wunpool,
        *rest,
    ):
        experts, kv = rest[:ne], list(rest[ne:])
        # the TP rank from meta: axis_index lowers to partition-id, which this export rejects
        me = meta[1]
        # expert TP: this rank's slices are virtual experts [rank * E, (rank + 1) * E)
        group_offset = (me * c.n_experts).reshape(1).astype(jnp.int32)
        rope_rows = jnp.take(rope, pos, axis=0)
        own = jax.lax.dynamic_slice_in_dim(emb, me * Ts, Ts, axis=0)
        # the native decoder streams the bf16-rounded normalized embedding
        stream = _bf(_rms(own, c.rms_eps))
        aux = []
        for l in range(L):
            sm = smalls[l]
            with _scope("sp_in_norm"):
                a8, a_s = rms_q8(stream, small_row(sm, "pre_attn_w"))
            if n_xag > 1 and not xla_qkvg:
                with _scope("sp_xag_qkvg"):
                    sub = Ts // n_xag
                    out = None
                    # every row's scale in one small AG (a scale gather per chunk paid a wait each)
                    s_all = ag(a_s, axis)
                    for i in range(n_xag):
                        rs = slice(i * sub, (i + 1) * sub)
                        out = qkvg_c[c.use_rope(l)].chunk(
                            i,
                            n_xag,
                            ag(a8[rs], axis),
                            s_all,
                            wqkvg[l],
                            sm,
                            rope_rows,
                            out,
                        )
                    q, kvs, gate = out
            else:
                with _scope("sp_x_ag"):
                    a8, a_s = ag(a8, axis), ag(a_s, axis)
            with _scope("sp_qkvg"):
                if n_xag > 1 and not xla_qkvg:
                    pass
                elif xla_qkvg:
                    qkvg = dense(a8, a_s, wqkvg[l], sm, "wqkvg", c.qkvg_dim)
                    q, kvs, gate = qkv_from_proj(
                        c, qkvg, rope_rows if c.use_rope(l) else None, xla=True
                    )
                else:
                    q, kvs, gate = qkvg_k[c.use_rope(l)](
                        a8, a_s, wqkvg[l], sm, rope_rows
                    )
            if q_rpa:
                kvd = c.kv_dim
                is_global = l in c.global_layers
                o, kv[l] = rpa_prepared(
                    q.reshape(c.kv_heads, T, gq // 2, 2, 128),
                    kvs[:, :kvd].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
                    kvs[:, kvd:].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
                    kv[l],
                    kv_lens,
                    pf if is_global else psw,
                    cu_q,
                    dist,
                    num_q_heads=c.q_heads,
                    sm_scale=c.attn_scale,
                    sliding_window=None if is_global else c.window,
                )
                with _scope("sp_o_gate"):
                    # o, gate in the kernel's [Hkv, T, G, d] head layout; rows quantized per token
                    o4 = o.reshape(c.kv_heads, T, gq, 128)[..., :hd].astype(_F32)
                    o4 = o4 * jax.lax.rsqrt(
                        jnp.mean(o4 * o4, axis=-1, keepdims=True) + c.rms_eps
                    )
                    g4 = gate.reshape(T, c.kv_heads, gq, hd).transpose(1, 0, 2, 3)
                    a4 = jax.nn.sigmoid(g4.astype(_F32)) * o4
                    g_s = (
                        jnp.maximum(
                            jnp.max(jnp.abs(a4), axis=(0, 2, 3)), 1e-30
                        ).reshape(T, 1)
                        / 448.0
                    )
                    g8 = (a4 / g_s.reshape(1, T, 1, 1)).astype(jnp.float8_e4m3fn)
                with _scope("sp_o_a2a"):
                    g8 = jax.lax.all_to_all(
                        g8.reshape(c.kv_heads, tp, Ts, gq, hd), axis, 1, 1, tiled=True
                    )
                    g_s = to_owner(g_s, tp, axis)
                with _scope("sp_o_proj"):
                    # [Hkv, src, Ts, G, d] -> own rows x (src rank, head, d): wo's row order
                    a = g8.astype(_F32).transpose(2, 1, 0, 3, 4) * g_s.reshape(
                        Ts, tp, 1, 1, 1
                    )
                    g8, g_s = q8(a.reshape(Ts, tp * qd))
                    o_own = dense(g8, g_s, wo[l], sm, "wo", H)
            else:
                o, kv[l] = attention(c, l, q, kvs, kv[l], kv_lens, pf, psw, cu_q, dist)
                with _scope("sp_o_gate"):
                    if ogate_mxu:
                        attn = _heads_rms_mxu(
                            o.reshape(T, qd).astype(_F32), c.q_heads, hd, c.rms_eps
                        )
                    else:
                        attn = heads_rms_xla(o.reshape(T, qd), c.q_heads, hd, c.rms_eps)
                    g8, g_s = q8(jax.nn.sigmoid(gate.astype(_F32)) * attn)
                with _scope("sp_o_a2a"):
                    g8, g_s = to_owner_x(g8), to_owner(g_s, tp, axis)
                # own rows' every head, then the o proj on the whole wo: no partial sums
                with _scope("sp_o_proj"):
                    g8, g_s = q8((g8.astype(_F32) * g_s).reshape(Ts, tp * qd))
                    o_own = dense(g8, g_s, wo[l], sm, "wo", H)
            with _scope("sp_attn_res"):
                stream = residual(c, stream, sm, "attn", o_own)
            with _scope("sp_router"):
                ffn = _rms(stream, c.rms_eps) * small_row(sm, "pre_ffn_w")
                ids, tw = router(c, ffn.astype(_BF16), wrouter[l], sm)
            with _scope("sp_pool"):
                if fused_rq:
                    f8, f_s = rms_q8(stream, small_row(sm, "pre_ffn_w"))
                else:
                    f8, f_s = q8(ffn)
                pooled = dense(f8, f_s, wpool[l], sm, "wpool", He)
                xe8, xs = rms_q8(pooled, small_row(sm, "pre_expert_w")[:, :He])
            with _scope("sp_xe_ag"):
                if xe_pallas:
                    xe = dequant(ag(xe8, axis), ag(xs, axis))
                else:
                    xe = (ag(xe8, axis).astype(_F32) * ag(xs, axis)).astype(_BF16)
                ids_all, tw_all = ag(ids, axis), ag(tw, axis)
            with _scope("sp_moe"):
                y = fused_rs_moe(
                    c,
                    layer_experts,
                    l,
                    xe,
                    ids_all,
                    tw_all,
                    experts,
                    group_offset,
                    sm,
                    axis,
                    # ids / tw are all-gathered: equal on every rank, no rank-0 broadcast
                    skip_routing_psum=True,
                )
            with _scope("sp_unpool"):
                u8, u_s = q8(y.astype(_F32))
                # bf16: the unpool output feeds the post-FFN norm
                f = dense(u8, u_s, wunpool[l], sm, "wunpool", H).astype(_BF16)
            with _scope("sp_ffn_res"):
                stream = residual(
                    c, stream, sm, "ffn", f.astype(_F32) * small_row(sm, "post_ffn_w")
                )
            if l in c.capture_layers:
                aux.append(stream.astype(_BF16))
        hidden = (_rms(stream.astype(_BF16), c.rms_eps) * final_norm[0]).astype(_BF16)
        aux_out = jnp.concatenate(aux, axis=1) if aux else jnp.zeros((Ts, LANE), _BF16)
        return (ag(hidden, axis), ag(aux_out, axis), *kv)

    return forward
