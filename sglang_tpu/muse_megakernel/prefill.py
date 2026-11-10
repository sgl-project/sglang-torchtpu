# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill (extend) program: the whole decoder as one jitted shard_map per chunk.

Reads the decode megakernel's weights (fp8 K-major dense copies, bf16 router, smalls) and
the sglang-owned experts and KV pools. A layer is batched RPA attention, the fused
attention-output kernel, the fused-RS MoE (expert TP or EP, PFConfig.moe_mode) and the fused
post-MoE kernel, which also produces the next layer's attention inputs (prefill_kernels.py).
"""

import os

import jax.numpy as jnp
import msgspec

from sglang_tpu.muse_megakernel.kernel import SMALL, _rms, expert_units, n_expert_refs
from sglang_tpu.muse_megakernel.moe_mode import MoeMode, check_no_legacy_env


class PFConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static description of one compiled prefill program."""

    hidden: int
    head_dim: int
    q_heads: int
    kv_heads: int
    n_layers: int
    n_experts: int
    top_k: int
    expert_hidden: int
    tp: int
    rms_eps: float
    post_eps: float
    scale_query_by: float
    attn_scale: float
    router_mult: float
    route_scale: float
    router_eps: float
    window: int
    global_layers: tuple
    capture_layers: tuple
    # padded token rows of the chunk
    rows: int
    # the experts' layout (the decode megakernel's)
    moe_mode: MoeMode
    # rows per tile of the attention-output and post-MoE kernels
    attn_out_tile: int = 64
    post_moe_tile: int = 128
    # the expert-input and post-expert-projection gathers carry e4m3 blocks with one scale
    fp8_gather: bool = False
    # [lo, hi) stacked expert slab groups (shared with the decode megakernel)
    expert_groups: tuple = ()

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkvg_dim(self) -> int:
        return 2 * self.q_dim + 2 * self.kv_dim

    def use_rope(self, layer: int) -> bool:
        return layer not in self.global_layers


def _first_attention_inputs(c: PFConfig, x, sm, wqkvg, rope):
    """Layer 0's (q, kv, gate) with XLA ops; later layers get them from post_moe_call."""
    from sglang_tpu.muse_megakernel.kernel import _heads_rms, _rope_roll

    hd, qd, kvd = c.head_dim, c.q_dim, c.kv_dim
    row = lambda name: sm[SMALL[name]]
    a_in = (_rms(x, c.rms_eps) * row("pre_attn_w")).astype(jnp.bfloat16)
    qkvg = (
        jnp.dot(a_in, wqkvg, preferred_element_type=jnp.float32)
        * row("wqkvg_s")[: c.qkvg_dim]
    )
    q = _heads_rms(qkvg[:, :qd], c.q_heads, hd, c.rms_eps, gain=c.scale_query_by)
    k = _heads_rms(qkvg[:, qd : qd + kvd], c.kv_heads, hd, c.rms_eps)
    if c.use_rope(0):
        q = _rope_roll(q, c.q_heads, hd, rope[:, :hd], rope[:, hd:])
        k = _rope_roll(k, c.kv_heads, hd, rope[:, :hd], rope[:, hd:])
    kv = jnp.concatenate([k, qkvg[:, qd + kvd : qd + 2 * kvd]], axis=1)
    return (
        q.astype(jnp.bfloat16),
        kv.astype(jnp.bfloat16),
        qkvg[:, qd + 2 * kvd :].astype(jnp.bfloat16),
    )


def build_forward(c: PFConfig, axis: str = "model"):
    """Per-device body: (meta, pos, kv_lens, cu_q, dist, pf, psw, emb, *weights in
    kernel.WEIGHT_NAMES order, *experts, *kv) -> (hidden, aux, *kv). Smalls rows hold the
    norms' gains (zero-centered weights + 1), as for the decode megakernel."""
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import ragged_paged_attention
    from sglang_tpu.moe.pallas.fused_rs.fused_moe import moe_gmm_local_rs_nodedup
    from sglang_tpu.muse_megakernel.prefill_kernels import attn_out_call, post_moe_call

    check_no_legacy_env()
    L, T, hd = c.n_layers, c.rows, c.head_dim
    qd, kvd, He = c.q_dim, c.kv_dim, c.expert_hidden
    ne = n_expert_refs(c)
    layer_unit: list[tuple[int, int | None]] = [(0, None)] * L
    for u, unit in enumerate(expert_units(c)):
        if unit[0] == "group":
            _, lo, hi = unit
            for li in range(lo, hi):
                layer_unit[li] = (u, li - lo)
        else:
            layer_unit[unit[1]] = (u, None)
    attn_out = attn_out_call(c, c.attn_out_tile, axis)
    post_moe = {
        r: post_moe_call(c, c.post_moe_tile, next_rope=r, axis=axis)
        for r in (True, False, None)
    }

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
        # ep: whole experts, E / tp per rank; tp: 1/tp of every expert. Both keep the
        # token-sharded (sequence-parallel) fused-RS output.
        moe_ep = c.moe_mode == "ep"
        local_experts = c.n_experts // c.tp if moe_ep else c.n_experts
        group_offset = (meta[1] * local_experts).reshape(1).astype(jnp.int32)
        rope_rows = jnp.take(rope, pos, axis=0)
        # the native decoder streams the bf16-rounded normalized embedding
        x = _rms(emb, c.rms_eps).astype(jnp.bfloat16).astype(jnp.float32)
        q, kvs, gate = _first_attention_inputs(c, x, smalls[0], wqkvg[0], rope_rows)
        aux = []
        for l in range(L):
            s = smalls[l]
            is_global = l in c.global_layers
            o, kv[l] = ragged_paged_attention(
                q.reshape(T, c.q_heads, hd),
                kvs[:, :kvd].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
                kvs[:, kvd:].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
                kv[l],
                kv_lens,
                pf if is_global else psw,
                cu_q,
                dist,
                sm_scale=c.attn_scale,
                sliding_window=None if is_global else c.window,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                # attn_out normalizes per head on-chip; never double-normalize inside RPA.
                out_head_norm_eps=None,
            )
            lyr = jnp.full((1,), l, jnp.int32)
            x, xe, ids, tw = attn_out(
                lyr, o.reshape(T, qd), gate, x, wo, wrouter, wpool, smalls
            )
            u_idx, layer_idx = layer_unit[l]
            w13, s13, w2, s2 = experts[4 * u_idx : 4 * u_idx + 4]
            layer_go = (
                group_offset
                if layer_idx is None
                else jnp.stack([group_offset[0], jnp.int32(layer_idx)])
            )
            y = moe_gmm_local_rs_nodedup(
                xe,
                w13,
                s13,
                None,
                w2,
                s2,
                None,
                None,
                None,
                layer_go,
                tw[:, : c.top_k].astype(jnp.bfloat16),
                ids[:, : c.top_k],
                # fused-RS applies (w + 1): hand it the zero-centered weight
                (s[SMALL["post_expert_w"], :He] - 1.0).astype(jnp.bfloat16),
                activation="silu",
                topk=c.top_k,
                ep_size=c.tp,
                ep_axis_name=axis,
                has_post_norm=True,
                sp_enabled=False,
                expert_tp=not moe_ep,
                # Tuning: e4m3 row writes (half the output / ICI bytes, adds a
                # rounding of the per-slice partial rows).
                fp8_direct_write=os.environ.get("MUSE_PF_MOE_FP8_WRITE", "0") == "1",
            )
            nxt = l + 1 if l + 1 < L else None
            x, q, kvs, gate = post_moe[c.use_rope(nxt) if nxt is not None else None](
                lyr, y, x, rope_rows, wunpool, smalls, wqkvg
            )
            if l in c.capture_layers:
                aux.append(x.astype(jnp.bfloat16))
        hidden = (_rms(x.astype(jnp.bfloat16), c.rms_eps) * final_norm[0]).astype(
            jnp.bfloat16
        )
        aux_out = (
            jnp.concatenate(aux, axis=1) if aux else jnp.zeros((T, 128), jnp.bfloat16)
        )
        return (hidden, aux_out, *kv)

    return forward
