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
from sglang_tpu.muse_megakernel_v2.prefill.sptp.qkvg import qkvg_call
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
    # prototype A/B knobs, read at build time (anvil --env): SPTP_XLA_QKVG=1 the XLA qkvg card
    xla_qkvg = DEBUG_BF16_ACTIVATIONS or os.environ.get("SPTP_XLA_QKVG", "0") == "1"
    # SPTP_QKVG_SLICE_NORM=1: per-head lane-slice norms (common/qkv) in the Pallas card
    mxu_norm = os.environ.get("SPTP_QKVG_SLICE_NORM", "0") != "1"
    qkvg_k = {r: qkvg_call(c, QKVG_TILE, r, mxu_norm) for r in (True, False)}
    # SPTP_XAG_CHUNKS=n: the x_norm AG in n row chunks, each chunk's qkvg as it lands (the
    # next chunk's AG in flight meanwhile); 1 = one AG, then qkvg
    n_xag = int(os.environ.get("SPTP_XAG_CHUNKS", str(XAG_CHUNKS)))
    qkvg_c = {r: qkvg_call(c, XAG_TILE, r, mxu_norm) for r in (True, False)}

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
                a8, a_s = q8(_rms(stream, c.rms_eps) * small_row(sm, "pre_attn_w"))
            if n_xag > 1 and not xla_qkvg:
                with _scope("sp_xag_qkvg"):
                    sub = Ts // n_xag
                    out = None
                    for i in range(n_xag):
                        rs = slice(i * sub, (i + 1) * sub)
                        out = qkvg_c[c.use_rope(l)].chunk(
                            i,
                            n_xag,
                            ag(a8[rs], axis),
                            ag(a_s[rs], axis),
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
            o, kv[l] = attention(c, l, q, kvs, kv[l], kv_lens, pf, psw, cu_q, dist)
            with _scope("sp_o_gate"):
                attn = heads_rms_xla(o.reshape(T, qd), c.q_heads, hd, c.rms_eps)
                g8, g_s = q8(jax.nn.sigmoid(gate.astype(_F32)) * attn)
            with _scope("sp_o_a2a"):
                g8, g_s = to_owner(g8, tp, axis), to_owner(g_s, tp, axis)
            # own rows' every head, then the o proj on the whole wo: no partial sums
            with _scope("sp_o_proj"):
                g8, g_s = q8((g8.astype(_F32) * g_s).reshape(Ts, tp * qd))
                o_own = dense(g8, g_s, wo[l], sm, "wo", H)
            with _scope("sp_attn_res"):
                stream = residual(c, stream, sm, "attn", o_own)
            with _scope("sp_router"):
                ffn = _rms(stream, c.rms_eps) * small_row(sm, "pre_ffn_w")
                ids, tw = route(c, ffn.astype(_BF16), wrouter[l], sm)
            with _scope("sp_pool"):
                f8, f_s = q8(ffn)
                pooled = dense(f8, f_s, wpool[l], sm, "wpool", He)
                xe8, xs = q8(
                    _rms(pooled, c.rms_eps) * small_row(sm, "pre_expert_w")[:, :He]
                )
            with _scope("sp_xe_ag"):
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
