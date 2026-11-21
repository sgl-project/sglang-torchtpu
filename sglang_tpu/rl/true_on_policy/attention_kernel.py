# SPDX-License-Identifier: Apache-2.0
"""Packed causal attention: the engine's batched RPA kernel with its KV block pinned to the page."""

# No ``from __future__ import annotations``: the Pallas bridge reads the annotations.

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from sglang_tpu.rl.true_on_policy.common import refuse_flags, register

PAGE_SIZE = 256


def pin_kv_block(decode, prefill, serve_cfgs):
    """One KV block, the page, for decode and prefill, on a bf16 SEQ_ALONG_LANE cache."""
    from sglang_tpu.attention.pallas.batched_rpa import configs

    cache = (serve_cfgs.page_size, serve_cfgs.kv_layout, serve_cfgs.dtype_kv)
    if cache != (PAGE_SIZE, configs.KVLayout.SEQ_ALONG_LANE, jnp.bfloat16):
        raise ValueError(
            f"True on-policy attention runs a bf16 SEQ_ALONG_LANE KV cache with {PAGE_SIZE}-token pages "
            f"(SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1, --page-size {PAGE_SIZE}), got {cache}"
        )
    return dataclasses.replace(decode, bkv_sz=PAGE_SIZE), dataclasses.replace(
        prefill, bkv_sz=PAGE_SIZE
    )


def refuse_rpa_rounding_flags():
    """Refuse the RPA flags that change rounding."""
    flags = (
        "SGLANG_TPU_RPA_EXP2_SOFTMAX",
        "SGLANG_TPU_RPA_ONES_V",
        "SGLANG_TPU_RPA_TRANSPOSED",
    )
    refuse_flags(flags, "attention")


@contextlib.contextmanager
def _pinned_kv_blocks():
    """Pin the KV block while the attention is traced."""
    from sglang_tpu.attention.pallas.batched_rpa import wrapper

    original = wrapper.calculate_block_sizes

    def pinned(model_cfgs, serve_cfgs, vmem_limit_bytes):
        return pin_kv_block(
            *original(model_cfgs, serve_cfgs, vmem_limit_bytes), serve_cfgs
        )

    wrapper.calculate_block_sizes = pinned
    try:
        yield
    finally:
        wrapper.calculate_block_sizes = original


def _jax_packed_causal_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, cu_seqlens: jax.Array, sm_scale: float
) -> jax.Array:
    """Causal attention within each packed sequence.

    q [T, Hq, D], k and v [T, Hkv, D] bf16; cu_seqlens [n + 1] int32 sequence starts (rows past cu_seqlens[-1] are
    padding); sm_scale is static. No sliding window, logit cap or attention sink.
    """
    from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    refuse_rpa_rounding_flags()
    tokens, _, head_dim = q.shape
    n = cu_seqlens.shape[0] - 1
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pages_per_seq = pl.cdiv(tokens, PAGE_SIZE)
    num_pages = pages_per_seq + n
    seq_pages = (lens + PAGE_SIZE - 1) // PAGE_SIZE
    first_page = jnp.cumsum(seq_pages) - seq_pages
    page_indices = jnp.minimum(
        first_page[:, None] + jnp.arange(pages_per_seq)[None, :], num_pages - 1
    )
    # One idle sequence slot past the batch.
    page_indices = jnp.concatenate(
        [page_indices, jnp.zeros((1, pages_per_seq), page_indices.dtype)]
    ).reshape(-1)
    kv_lens = jnp.concatenate([lens, jnp.zeros((1,), lens.dtype)]).astype(jnp.int32)
    cu_q_lens = jnp.concatenate([cu_seqlens, cu_seqlens[-1:]]).astype(jnp.int32)
    layout = configs.KVLayout.SEQ_ALONG_LANE
    distribution = jnp.full((3,), n, jnp.int32)  # every sequence is an extend
    with tpu_info_for_cpu_export(), _pinned_kv_blocks():
        kv_cache = jnp.zeros(
            wrapper.get_kv_cache_shape(
                num_pages, PAGE_SIZE, k.shape[1], head_dim, k.dtype, kv_layout=layout
            ),
            k.dtype,
        )
        out, _ = wrapper.ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices.astype(jnp.int32),
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            kv_layout=layout,
        )
    return out


packed_causal_attention = register(
    "packed_causal_attention", _jax_packed_causal_attention
)
