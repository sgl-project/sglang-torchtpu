# SPDX-License-Identifier: Apache-2.0
"""Packed gated delta net: the engine's fused kernel on its per-token recurrence."""

# No ``from __future__ import annotations``: the Pallas bridge reads the annotations.

import jax
import jax.numpy as jnp

from sglang_tpu.rl.true_on_policy.common import register

# Tokens per prefill tile; sets the speed only.
GDN_PREFILL_TILE = 16


def bf16_rounded(x: jax.Array) -> jax.Array:
    """A_log or dt_bias as the engine serves it: rounded to bf16, kept in its dtype (Qwen3.5 holds A_log in fp32)."""
    return x.astype(jnp.bfloat16).astype(x.dtype)


def _jax_packed_gated_delta_net(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cu_seqlens: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> jax.Array:
    """Gated delta net over packed sequences, each from a zero state.

    qkv [T, 2 * n_kq * d_k + n_v * d_v], b and a [T, n_v] bf16; conv_weight [dim, 1, kernel], no bias;
    a_log and dt_bias [n_v], read as their bf16 rounding; cu_seqlens [n + 1] int32 sequence starts (rows past
    cu_seqlens[-1] are padding). Returns [T, n_v, d_v].
    """
    from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    n = cu_seqlens.shape[0] - 1
    kernel_size = conv_weight.shape[-1]
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).astype(jnp.int32)
    # Slot 0 is the null block; sequence i owns slot i + 1.
    slots = jnp.where(lens > 0, jnp.arange(1, n + 1, dtype=jnp.int32), 0)
    conv_state = jnp.zeros((n + 1, kernel_size - 1, 1, qkv.shape[1]), jnp.float32)
    recurrent_state = jnp.zeros((n + 1, n_v, d_k, d_v), jnp.float32)
    with tpu_info_for_cpu_export():
        _, out = fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            None,
            bf16_rounded(a_log),
            bf16_rounded(dt_bias),
            cu_seqlens.astype(jnp.int32),
            slots,
            # No decode sequences: every sequence is a prefill.
            jnp.array([0, 0, n], jnp.int32),
            lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            prefill_only=True,
            recurrent_scan=True,
            mixed_tile_size=GDN_PREFILL_TILE,
        )
    return out.reshape(-1, n_v, d_v)


packed_gated_delta_net = register("packed_gated_delta_net", _jax_packed_gated_delta_net)
