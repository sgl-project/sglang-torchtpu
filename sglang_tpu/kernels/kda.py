# SPDX-License-Identifier: Apache-2.0
"""Reference ragged KDA for TPU bring-up.

Recurrence adapted from sglang-jax/srt/kernels/kda/naive.py (originally
primatrix/pallas-kernel, rev 3c691ad3). A lax.scan keeps compilation bounded
for prefill. This deliberately favors correctness over throughput.
"""

import jax
import jax.numpy as jnp


def recurrent_kda_step(q, k, v, g, beta, state):
    """All arithmetic/state is fp32; state is [heads, key_dim, value_dim]."""
    state = state * jnp.exp(g)[..., None]
    residual = v - jnp.sum(k[..., None] * state, axis=-2)
    state = state + (beta[..., None] * k)[..., None] * residual[..., None, :]
    output = jnp.sum((q * q.shape[-1] ** -0.5)[..., None] * state, axis=-2)
    return output, state


def ragged_kda(
    qkv: jax.Array,
    a: jax.Array,
    b: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    num_heads: int,
    head_dim: int,
    lower_bound: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused causal depthwise conv, gate and KDA over packed request tokens.

    Slot zero and padding do not mutate state. A request with zero prefix starts
    with zero state even if its pool slot belonged to an earlier request.
    Conv state is [slots, kernel_size-1, channels], SSM is [slots,H,K,V].
    lower_bound=0 selects the unbounded -exp(A)*softplus gate (Kimi Linear).
    """
    tokens = qkv.shape[0]
    a = a.reshape(tokens, num_heads, head_dim).astype(jnp.float32)
    b = b.reshape(tokens, num_heads).astype(jnp.float32)
    weight = conv_weight.reshape(3 * num_heads * head_dim, -1).T.astype(jnp.float32)
    decay = jnp.exp(a_log.reshape(num_heads, 1).astype(jnp.float32))
    bias = dt_bias.reshape(num_heads, head_dim).astype(jnp.float32)
    # Repeated start offsets for padded empty requests are safe: only tokens
    # strictly before query_start_loc[-1] are live.
    req_ids = jnp.sum(
        jnp.arange(tokens)[:, None] >= query_start_loc[1:][None, :], axis=1
    )
    req_ids = jnp.minimum(req_ids, state_indices.shape[0] - 1)

    def step(carry, inputs):
        conv_pool, ssm_pool = carry
        token, req, raw, gate, beta = inputs
        slot = state_indices[req]
        active = (token < query_start_loc[-1]) & (slot > 0)
        start = query_start_loc[req]
        end = query_start_loc[req + 1]
        reset = (token == start) & (seq_lens[req] == end - start)
        conv = jnp.where(reset, 0.0, conv_pool[slot])
        state = jnp.where(reset, 0.0, ssm_pool[slot])
        window = jnp.concatenate([conv, raw.astype(jnp.float32)[None, :]], axis=0)
        conv_out = jax.nn.silu(jnp.sum(window * weight, axis=0)).astype(qkv.dtype)
        q, k, v = jnp.split(
            conv_out.reshape(3, num_heads, head_dim).astype(jnp.float32), 3, axis=0
        )
        q, k, v = q[0], k[0], v[0]
        # FLA uses rsqrt(sum(x*x)+eps), not 1/max(norm,eps).
        q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        if lower_bound < 0:
            g = lower_bound * jax.nn.sigmoid(decay * (gate + bias))
        else:
            g = -decay * jax.nn.softplus(gate + bias)
        out, state = recurrent_kda_step(q, k, v, g, jax.nn.sigmoid(beta), state)
        conv_pool = conv_pool.at[slot].set(
            jnp.where(active, window[1:], conv_pool[slot])
        )
        ssm_pool = ssm_pool.at[slot].set(jnp.where(active, state, ssm_pool[slot]))
        return (conv_pool, ssm_pool), jnp.where(active, out, 0).astype(qkv.dtype)

    (conv_state, recurrent_state), output = jax.lax.scan(
        step, (conv_state, recurrent_state), (jnp.arange(tokens), req_ids, qkv, a, b)
    )
    return conv_state, recurrent_state, output
