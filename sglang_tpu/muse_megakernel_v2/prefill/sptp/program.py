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
from sglang_tpu.muse_megakernel_v2.prefill.tp.program import (
    _layer_experts,
    fused_rs_moe,
)

# Numerics debugging only: bf16 activations (scale 1) where the program quantizes rows to e4m3,
# to separate layout bugs from e4m3 rounding. The recipe sets it at build time.
DEBUG_BF16_ACTIVATIONS = False


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
            with _scope("sp_x_ag"):
                a8, a_s = ag(a8, axis), ag(a_s, axis)
            with _scope("sp_qkvg"):
                qkvg = dense(a8, a_s, wqkvg[l], sm, "wqkvg", c.qkvg_dim)
                q, kvs, gate = qkv_from_proj(
                    c, qkvg, rope_rows if c.use_rope(l) else None, xla=True
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
