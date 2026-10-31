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
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/gdn/v3/tiling.py
"""VMEM estimate of the fused Conv1D-GDN kernel, and the prefill tile that fits.

The two estimate functions are upstream's, with the KDA terms. Upstream also
searches decode and prefill tile sizes; here the wrapper keeps its own sizes
and fit_mixed_tile_size only shrinks a prefill tile that would not fit.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl

from sglang_tpu.attention.pallas.gdn import compute_gdn, config


def align_to(x: int, alignment: int) -> int:
    """Aligns an integer upward to the nearest multiple of alignment."""
    return pl.cdiv(x, alignment) * alignment


def get_vmem_declared_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_out_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
    carry_scratch_bytes: int = 4,
    window_size: int = 1,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
) -> int:
    """Declared on-chip VMEM footprint in bytes (create_allocs + scratch shapes)."""
    aligned_num_v_heads = align_to(n_v, num_lanes)
    dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
    aligned_out_dim = align_to(n_v * d_v, num_lanes)
    is_kda = attention_mode == config.AttentionMode.KDA
    aligned_gate_dim = align_to(n_v * d_k if is_kda else n_v, num_lanes)

    # 1. Double-buffered input activation buffers (QKV, A, B), upcast to float32.
    qkv_bytes = 2 * (tile_b * chunk_sz * dim_size * carry_scratch_bytes)
    b_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes)
    a_bytes = 2 * (tile_b * chunk_sz * aligned_gate_dim * carry_scratch_bytes)

    # 2. Double-buffered state cache buffers (convolution and recurrent states).
    # Input state windows hold 1 state per sequence; output state windows hold
    # `window_size` checkpoints per sequence, giving `1 + window_size` state
    # slots per sequence per buffer.
    state_slots_per_seq = 1 + window_size
    conv_state_buffer_bytes = 2 * (
        tile_b
        * state_slots_per_seq
        * max(0, kernel_size - 1)
        * conv_state_dim_size
        * carry_scratch_bytes
    )
    recurrent_state_buffer_bytes = 2 * (
        tile_b * state_slots_per_seq * n_v * d_v * d_k * rec_state_bytes
    )

    # 3. Double-buffered output activation buffer.
    out_bytes = 2 * (tile_b * chunk_sz * aligned_out_dim * act_out_bytes)

    # 4. Working state. Prefill keeps f32 copies of convolution and recurrent
    # carries for inter-tile chunking; decode mode is batched with no carries.
    if is_decode:
        scratch_conv_bytes = 0
        scratch_recurrent_bytes = 0
    else:
        scratch_conv_bytes = (
            tile_b * max(0, kernel_size - 1) * dim_size * carry_scratch_bytes
        )
        scratch_recurrent_bytes = tile_b * n_v * d_v * d_k * carry_scratch_bytes

    # 5. Static weight cache references in on-chip memory.
    # Conv weight [K, dim] plus conv bias [dim] in f32; a_log and dt_bias.
    weights_bytes = (kernel_size + 1) * dim_size * carry_scratch_bytes + (
        aligned_num_v_heads + aligned_gate_dim
    ) * carry_scratch_bytes

    return (
        qkv_bytes
        + b_bytes
        + a_bytes
        + conv_state_buffer_bytes
        + recurrent_state_buffer_bytes
        + out_bytes
        + scratch_conv_bytes
        + scratch_recurrent_bytes
        + weights_bytes
    )


def get_vmem_estimate_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_out_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
    carry_scratch_bytes: int = 4,
    window_size: int = 1,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
) -> int:
    """Estimates total on-chip VMEM footprint in bytes for a GDN tile."""
    aligned_d_k = align_to(d_k, num_lanes)
    aligned_d_v = align_to(d_v, num_lanes)
    declared_bytes = get_vmem_declared_bytes(
        tile_b=tile_b,
        chunk_sz=chunk_sz,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_out_bytes=act_out_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        is_decode=is_decode,
        carry_scratch_bytes=carry_scratch_bytes,
        window_size=window_size,
        attention_mode=attention_mode,
    )

    # Working state during decode: one f32 copy of the recurrent state
    # alongside the double-buffered state windows.
    decode_working_recurrent_bytes = (
        tile_b * n_v * d_v * d_k * carry_scratch_bytes if is_decode else 0
    )

    # 6. Working memory for intra-chunk recurrence and per-head projections
    # (chunk_sz is the verify window in decode) plus per-slot vector
    # temporaries.
    kda_decay_elems = (
        3 * chunk_sz * aligned_d_k if attention_mode == config.AttentionMode.KDA else 0
    )
    intermediate_bytes = (
        tile_b
        * (
            n_v
            * (
                5 * chunk_sz * chunk_sz
                + 3 * chunk_sz * (aligned_d_v + aligned_d_k)
                + kda_decay_elems
            )
            + 8 * 8 * num_lanes
        )
        * carry_scratch_bytes
    )

    return declared_bytes + decode_working_recurrent_bytes + intermediate_bytes


def fit_mixed_tile_size(
    mixed_tile_size: int,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_limit_bytes: int,
    attention_mode: config.AttentionMode,
) -> int:
    """Halve the prefill tile until its VMEM estimate fits the limit.

    A tile that already fits is returned unchanged, so this only changes
    shapes that would not compile, such as GLM-5.3-Flash's 64 KDA heads on
    one chip at the default 64-token tile. The tile stays a whole number of
    the triangular inverse's blocks.
    """
    while mixed_tile_size > compute_gdn.INVERT_BLOCK:
        estimate = get_vmem_estimate_bytes(
            tile_b=1,
            chunk_sz=mixed_tile_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=jnp.dtype(act_out_dtype).itemsize,
            rec_state_bytes=jnp.dtype(recurrent_state_dtype).itemsize,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            attention_mode=attention_mode,
        )
        if estimate <= vmem_limit_bytes:
            break
        mixed_tile_size = align_to(mixed_tile_size // 2, compute_gdn.INVERT_BLOCK)
    return mixed_tile_size
