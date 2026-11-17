# SPDX-License-Identifier: Apache-2.0
"""Attention inputs of the prefill programs: pre-attention norm, qkvg projection, per-head RMS
and rope, split into (q, kv, gate) rows."""

import jax.numpy as jnp

from sglang_tpu.muse_megakernel.common.config import PFConfig
from sglang_tpu.muse_megakernel.common.consts import small_row
from sglang_tpu.muse_megakernel.common.numerics import _heads_rms, _rms, _rope_roll


def attention_inputs(c: PFConfig, x, smn, wq, rope):
    """A layer's (q, kv, gate) rows from its input stream x; rope None: a NoPE layer."""
    return qkv_from_input(c, attention_input(c, x, smn), smn, wq, rope)


def attention_input(c: PFConfig, x, smn):
    """The pre-attention norm of stream rows x, bf16."""
    return (_rms(x, c.rms_eps) * small_row(smn, "pre_attn_w")).astype(jnp.bfloat16)


def qkv_from_input(c: PFConfig, a_in, smn, wq, rope):
    """(q, kv, gate) rows from pre-attention-normed rows a_in; rope None: a NoPE layer."""
    qkvg = jnp.dot(a_in, wq, preferred_element_type=jnp.float32)
    return qkv_from_proj(c, qkvg * small_row(smn, "wqkvg_s")[:, : c.qkvg_dim], rope)


def qkv_from_proj(c: PFConfig, qkvg, rope):
    """(q, kv, gate) rows (bf16) from the f32 qkvg projection; rope None: a NoPE layer."""
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    q = _heads_rms(qkvg[:, :qd], c.q_heads, hd, c.rms_eps, gain=c.scale_query_by)
    k = _heads_rms(qkvg[:, qd : qd + kvd], c.kv_heads, hd, c.rms_eps)
    if rope is not None:
        q = _rope_roll(q, c.q_heads, hd, rope[:, :hd], rope[:, hd:])
        k = _rope_roll(k, c.kv_heads, hd, rope[:, :hd], rope[:, hd:])
    kv = jnp.concatenate([k, qkvg[:, qd + kvd : qd + 2 * kvd]], axis=1)
    return (
        q.astype(jnp.bfloat16),
        kv.astype(jnp.bfloat16),
        qkvg[:, qd + 2 * kvd :].astype(jnp.bfloat16),
    )
