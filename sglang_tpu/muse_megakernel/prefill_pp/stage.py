# SPDX-License-Identifier: Apache-2.0
"""One PP prefill stage: per layer slot, attn_in, attention, route, experts, combine."""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.prefill_pp import attention as attn
from sglang_tpu.muse_megakernel.prefill_pp import moe
from sglang_tpu.muse_megakernel.prefill_pp.config import (
    F_FP8,
    HDR,
    NF,
    PPConfig,
    skipped_phases,
)
from sglang_tpu.muse_megakernel.prefill_pp.dense import stage_calls


def stage_forward(
    c: PPConfig,
    ctl,
    slots,
    x,
    rope,
    weights,
    experts_fp4,
    experts_fp8,
    pools,
    kv_lens,
    page_tables,
    cu_q,
    dist,
    scratch: bool = False,
):
    """One stage. ctl [2] i32 = [cached tokens, active stage or -1]; slots: the stage's slot
    table; x [T, H] f32 stream; weights: the dense stacks in WEIGHT_ORDER; experts_fp4: per slot
    (w13, w13_scale, w2, w2_scale); experts_fp8: the stage's fp8 set; pools: one per slot
    (global slots' as wide as the page table); page_tables: (global, sliding window).
    Returns (x, pools), and with scratch each slot's (ids, routing weights)."""
    skip = skipped_phases()
    attn_in_call, route_call, combine_call = stage_calls(c)
    T, K = c.rows, c.top_k
    pools, routes = list(pools), []
    for s in range(c.slots):
        ctl_s = jnp.concatenate([ctl, jnp.full((1,), s, jnp.int32)])
        is_g = s in c.global_slots
        with jax.named_scope(f"slot{s}/attn_in"):
            q, k, v, g = attn_in_call(ctl_s, slots, x, rope, *weights)
        with jax.named_scope(f"slot{s}/attention"):
            if "attention" in skip:
                o, pools[s] = attn.no_attention(c, pools[s])
            else:
                o, pools[s] = attn.attention(
                    c,
                    is_g,
                    q,
                    k,
                    v,
                    pools[s],
                    kv_lens,
                    page_tables[0 if is_g else 1],
                    cu_q,
                    dist,
                )
        with jax.named_scope(f"slot{s}/route"):
            x, xe, ids, tw = route_call(ctl_s, slots, x, o, g, *weights)
        with jax.named_scope(f"slot{s}/experts"):
            if "experts" in skip:
                y = moe.no_experts(c)
            else:
                is_fp8 = slots[HDR + s * NF + F_FP8] != 0
                y = moe.slot_experts(
                    c, s, is_fp8, xe, ids, tw, experts_fp4[s], experts_fp8
                )
        with jax.named_scope(f"slot{s}/combine"):
            (x,) = combine_call(
                ctl_s, slots, x, y.reshape(T * K, c.expert_hidden), tw, *weights
            )
        routes.append((ids, tw))
    return (x, pools, routes) if scratch else (x, pools)
