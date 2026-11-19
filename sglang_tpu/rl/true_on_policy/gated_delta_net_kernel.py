# SPDX-License-Identifier: Apache-2.0
"""Packed gated delta net: the engine's fused conv1d + gated-delta-rule kernel, pinned to its per-token recurrence.

A decode step always runs the recurrence; a chunked prefill solves the same tokens with other algebra, and gives
other bytes.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here: the torch_tpu
# Pallas bridge reads the wrapped functions' annotations (``jax.Array``).

import jax
import jax.numpy as jnp

from sglang_tpu.rl.true_on_policy.common import register

# The recurrence gives a token the same bytes in any tile, so the tile only sets the speed (measured on v7x:
# 8 to 64 tokens per tile all take about 60% of the 1-token tile's time).
GDN_PREFILL_TILE = 16


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
    """Gated delta net over packed sequences, each from a zero state, as the engine computes it.

    qkv [T, 2 * n_kq * d_k + n_v * d_v], b and a [T, n_v] bf16; conv_weight [dim, 1, kernel] with no bias;
    cu_seqlens [n + 1] int32 sequence starts (rows past cu_seqlens[-1] are padding). Returns [T, n_v, d_v].
    """
    from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    n = cu_seqlens.shape[0] - 1
    kernel_size = conv_weight.shape[-1]
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).astype(jnp.int32)
    # Slot 0 is the kernel's null block: sequence i owns slot i + 1, an empty sequence the null block.
    slots = jnp.where(lens > 0, jnp.arange(1, n + 1, dtype=jnp.int32), 0)
    conv_state = jnp.zeros((n + 1, kernel_size - 1, 1, qkv.shape[1]), jnp.float32)
    recurrent_state = jnp.zeros((n + 1, n_v, d_k, d_v), jnp.float32)
    # The kernel sizes its tiles from the physical TPU's metadata while it is traced, as in the engine's op.
    with tpu_info_for_cpu_export():
        _, out = fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            None,
            a_log,
            dt_bias,
            cu_seqlens.astype(jnp.int32),
            slots,
            # No decode sequences: every sequence is a prefill, and one as long as its query carries no state in.
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
    # The kernel returns [T, n_v * d_v]; the engine's op hands the model the per-head layout.
    return out.reshape(-1, n_v, d_v)


packed_gated_delta_net = register("packed_gated_delta_net", _jax_packed_gated_delta_net)
