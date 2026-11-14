# SPDX-License-Identifier: Apache-2.0
"""The per-device body of the prefill program: every decoder layer of one extend chunk."""

import jax.numpy as jnp

from sglang_tpu.muse_megakernel.mk.consts import LANE, SMALL
from sglang_tpu.muse_megakernel.mk.numerics import _rms
from sglang_tpu.muse_megakernel.mk.plan import expert_units, n_expert_refs
from sglang_tpu.muse_megakernel.prefill.attn_out import attn_out_call
from sglang_tpu.muse_megakernel.prefill.config import (
    ATTN_OUT_TILE,
    POST_MOE_TILE,
    PFConfig,
)
from sglang_tpu.muse_megakernel.prefill.post_moe import attention_inputs, post_moe_call

# Leading operands before the weights: meta, pos, kv_lens, cu_q, dist, pf, psw, emb.
N_STEP_ARGS = 8


def _layer_experts(c: PFConfig) -> list:
    """Per layer: (expert unit index, layer index in its stacked group or None)."""
    out: list = [None] * c.n_layers
    for u, unit in enumerate(expert_units(c)):
        if unit[0] == "group":
            for li in range(unit[1], unit[2]):
                out[li] = (u, li - unit[1])
        else:
            out[unit[1]] = (u, None)
    return out


def build_forward(c: PFConfig, axis: str = "model"):
    """(meta, pos, kv_lens, cu_q, dist, pf, psw, emb, *weights in kernel.WEIGHT_NAMES order,
    *experts, *kv) -> (hidden, aux, *kv). meta is [live rows, TP rank]; smalls rows hold the
    norms' gains (zero-centered weights + 1), as for the decode megakernel."""
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import ragged_paged_attention
    from sglang_tpu.moe.pallas.fused_rs.fused_moe import moe_gmm_local_rs_nodedup

    c.validate()
    L, T, hd = c.n_layers, c.rows, c.head_dim
    qd, kvd, He = c.q_dim, c.kv_dim, c.expert_hidden
    ne = n_expert_refs(c)
    layer_experts = _layer_experts(c)
    attn_out = attn_out_call(c, ATTN_OUT_TILE, axis)
    post_moe = {
        r: post_moe_call(c, POST_MOE_TILE, next_rope=r, axis=axis)
        for r in (True, False, None)
    }

    def attention(l, q, kvs, kv_pool, kv_lens, pf, psw, cu_q, dist):
        is_global = l in c.global_layers
        return ragged_paged_attention(
            q.reshape(T, c.q_heads, hd),
            kvs[:, :kvd].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
            kvs[:, kvd:].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
            kv_pool,
            kv_lens,
            pf if is_global else psw,
            cu_q,
            dist,
            sm_scale=c.attn_scale,
            sliding_window=None if is_global else c.window,
            k_scale=1.0,
            v_scale=1.0,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    def moe(l, xe, ids, tw, experts, group_offset, smalls_l):
        u, in_group = layer_experts[l]
        w13, s13, w2, s2 = experts[4 * u : 4 * u + 4]
        go = (
            group_offset
            if in_group is None
            else jnp.stack([group_offset[0], jnp.int32(in_group)])
        )
        return moe_gmm_local_rs_nodedup(
            xe,
            w13,
            s13,
            None,
            w2,
            s2,
            None,
            None,
            None,
            go,
            tw[:, : c.top_k].astype(jnp.bfloat16),
            ids[:, : c.top_k],
            # fused-RS applies (w + 1): hand it the zero-centered weight
            (smalls_l[SMALL["post_expert_w"], :He] - 1.0).astype(jnp.bfloat16),
            activation="silu",
            topk=c.top_k,
            ep_size=c.tp,
            ep_axis_name=axis,
            has_post_norm=True,
            # the output stays token-sharded: post_moe gathers it
            sp_enabled=False,
            expert_tp=True,
            fp4_w4a16=l in c.w4a16_layers,
        )

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
        # expert TP: this rank's slices are virtual experts [rank * E, (rank + 1) * E)
        group_offset = (meta[1] * c.n_experts).reshape(1).astype(jnp.int32)
        rope_rows = jnp.take(rope, pos, axis=0)
        # the native decoder streams the bf16-rounded normalized embedding
        x = _rms(emb, c.rms_eps).astype(jnp.bfloat16).astype(jnp.float32)
        q, kvs, gate = attention_inputs(
            c, x, smalls[0], wqkvg[0], rope_rows if c.use_rope(0) else None
        )
        aux = []
        for l in range(L):
            o, kv[l] = attention(l, q, kvs, kv[l], kv_lens, pf, psw, cu_q, dist)
            lyr = jnp.full((1,), l, jnp.int32)
            x, xe, ids, tw = attn_out(
                lyr, o.reshape(T, qd), gate, x, wo, wrouter, wpool, smalls
            )
            y = moe(l, xe, ids, tw, experts, group_offset, smalls[l])
            nxt = c.use_rope(l + 1) if l + 1 < L else None
            x, q, kvs, gate = post_moe[nxt](
                lyr, y, x, rope_rows, wunpool, smalls, wqkvg
            )
            if l in c.capture_layers:
                aux.append(x.astype(jnp.bfloat16))
        hidden = (_rms(x.astype(jnp.bfloat16), c.rms_eps) * final_norm[0]).astype(
            jnp.bfloat16
        )
        aux_out = (
            jnp.concatenate(aux, axis=1) if aux else jnp.zeros((T, LANE), jnp.bfloat16)
        )
        return (hidden, aux_out, *kv)

    return forward
