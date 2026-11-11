# SPDX-License-Identifier: Apache-2.0
"""The PP prefill stage's attention: sglang_tpu's batched RPA prefill kernel with main's tiles."""

import jax.numpy as jnp

from sglang_tpu.attention.pallas.batched_rpa import configs
from sglang_tpu.attention.pallas.batched_rpa.wrapper import ragged_paged_attention
from sglang_tpu.muse_megakernel.prefill_pp.config import BF16, FP8, PPConfig


def attention(c: PPConfig, is_global: bool, q, k, v, pool, kv_lens, page_table, cu_q):
    """One slot's attention: the chunk's K/V (fp8) into pool, attention over the pool (cached
    context and the causal chunk; the window on sliding-window slots), each head's output
    RMS-normalized. q [T, q_dim] bf16, k / v [T, kv_dim] bf16 -> (o [T, q_dim] bf16, pool)."""
    T = q.shape[0]
    n = cu_q.shape[0] - 1
    mixed = jnp.full((3,), n, jnp.int32)  # every sequence on the MIXED pass: [0, n)
    o, pool = ragged_paged_attention(
        q.reshape(T, c.q_heads, c.head_dim),
        k.astype(FP8).reshape(T, c.kv_heads, c.head_dim),
        v.astype(FP8).reshape(T, c.kv_heads, c.head_dim),
        pool, kv_lens, page_table, cu_q, mixed,
        sm_scale=c.attn_scale, sliding_window=None if is_global else c.window,
        k_scale=1.0, v_scale=1.0, kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        out_head_norm_eps=c.rms_eps,
    )  # fmt: skip
    return o.reshape(T, c.q_dim).astype(BF16), pool


def no_attention(c: PPConfig, pool):
    """MUSE_PP_SKIP=attention: zeros (timing ablation only)."""
    return jnp.zeros((c.rows, c.q_dim), BF16), pool
