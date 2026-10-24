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

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.batched_rpa import configs, utils


def flash_attention_qk_softmax(
    step: jax.Array,
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array,  # [KV, TQ, 128]
    l_prev: jax.Array,  # [KV, TQ, 128]
    is_last_k: jax.Ref,  # [B]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    cfgs: configs.RpaConfigs,
    bq_start: int,
    kv_len_left: list[jax.Array] | None = None,  # [B]
    global_q_k_gap: list[jax.Array] | None = None,  # [B]
):
    """Flash attention kernel."""
    b, k_heads, tq, h_size = q.shape

    if cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        s = k.shape[-1]
        qk = lax.dot(
            q.reshape(-1, tq, h_size),
            k.reshape(-1, h_size, s),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    else:
        s = k.shape[-2]
        qk = lax.dot(
            q.reshape(-1, tq, h_size),
            k.reshape(-1, s, h_size),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    qk = qk.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, s)

    qk *= cfgs.model.sm_scale
    if cfgs.serve.scale_k is not None:
        qk *= cfgs.serve.scale_k
    if cfgs.serve.scale_q is not None:
        qk *= cfgs.serve.scale_q

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    qk_masked = []
    masks = []

    kv_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 2)
    q_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
    q_iota //= cfgs.aligned_num_q_heads_per_kv_head
    q_kv_diff = q_iota - kv_iota
    if global_q_k_gap is not None:
        # Page-striped cache: local key j holds global position
        # j + (j // page_size) * (cp_group_size - 1) * page_size + rank * page_size.
        # The rank term is scalar and lives in global_q_k_gap; the local-index
        # part is computed per batch element below from the block's first key.
        lp = cfgs.serve.page_size_log2
        stripe_gap = (cfgs.serve.cp_group_size - 1) << lp

    for b_idx in range(cfgs.block.batch_size):
        # NOTE: Goal is to compute q_len >= kv_len. But we want to utilize scalar
        # compute as much as possible before involving vector compute. Therefore, we
        # break down a computational steps into following equations to separate out
        # scalar and vector compute.
        # q_len = q_iota + (bq_start + processed_q_len)
        # kv_len = kv_iota + processed_kv_len
        # Step 1: q_len >= kv_len:
        # Step 2:
        #   q_iota + (bq_start + processed_q_len)
        #   >= kv_iota + processed_kv_len
        # Step 3:
        #   q_iota - kv_iota
        #   >= processed_kv_len - (bq_start + processed_q_len)
        # Step 4:
        #   q_kv_delta = q_iota - kv_iota
        #   offset = processed_kv_len - (bq_start + processed_q_len)
        offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
        if kv_len_left is None:
            mask_b = q_kv_diff >= offset
        else:
            # Every cached key precedes every query; keys past kv_len are stale.
            mask_b = kv_iota < kv_len_left[b_idx]

        if (sliding_window := cfgs.model.sliding_window) is not None:
            # NOTE: Goal is to compute q_len < sliding_window + kv_len. And similar
            # to above, we want to minimize vector compute. Therefore, we break
            # down computes into following steps.
            # Step 1: q_len < sliding_window + kv_len
            # Step 2:
            #   q_iota + (bq_start + processed_q_len)
            #   < sliding_window + kv_iota + processed_kv_len
            # Step 3:
            #   q_iota - kv_iota
            #   < sliding_window + processed_kv_len - (bq_start + processed_q_len)
            # Step 4:
            #   q_kv_diff < sliding_window + offset
            if global_q_k_gap is None:
                mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)
            else:
                # Same in global positions: gap = q_global - (k_id + rank * page)
                # at q_iota = kv_iota = 0 of the block; the stripe term adds
                # (cp - 1) pages per local page of the key's local index.
                local_key = kv_iota + processed_kv_len[b_idx]
                q_kv_diff_global = q_kv_diff - (local_key >> lp) * stripe_gap
                window_offset = sliding_window - (bq_start + global_q_k_gap[b_idx])
                mask_b = jnp.logical_and(mask_b, q_kv_diff_global < window_offset)

        masks.append(mask_b)
        qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
    qk = jnp.stack(qk_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)

    alpha_list = []
    m_next_list = []

    for b_idx in range(cfgs.block.batch_size):
        m_curr_b = m_curr[b_idx]
        m_next_b = jnp.maximum(m_prev, m_curr_b)
        alpha_b = jnp.exp(m_prev - m_next_b)
        alpha_list.append(alpha_b)
        m_next_list.append(m_next_b)
        m_prev = jnp.where(is_last_k[step, b_idx], -jnp.inf, m_next_b)

    m_next = jnp.stack(m_next_list, axis=0)
    p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
    if kv_len_left is not None:
        # A row with no key in range keeps l == 0 rather than weighting masked
        # keys by exp(mask_value - mask_value).
        p = jnp.where(jnp.stack(masks, axis=0), p, 0)
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

    l_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        l_next_b = alpha_list[b_idx] * l_prev + p_rowsum[b_idx]
        l_next_list.append(l_next_b)
        l_prev = l_next_b

    l_next = jnp.stack(l_next_list, axis=0)

    # m_prev is the carry into the next step (reset after a last k block);
    # m_next is each lane's running max, used by the sink term and the LSE.
    return p, alpha_list, m_prev, l_next, m_next


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha_list: list[jax.Array],  # B * [KV, TQ, 128]
    o_prev: jax.Array,  # [KV, TQ, H]
    cfgs: configs.RpaConfigs,
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        h_size = v.shape[-2]
        pv = lax.dot(
            p.reshape(-1, tq, s),
            v.reshape(-1, h_size, s),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    else:
        h_size = v.shape[-1]
        pv = lax.dot(
            p.reshape(-1, tq, s),
            v.reshape(-1, s, h_size),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    pv = pv.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, h_size)

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
        o_next_b = alpha_b * o_prev + pv[b_idx]
        o_next_list.append(o_next_b)
        o_prev = o_next_b
    o_next = jnp.stack(o_next_list, axis=0)

    return o_next
