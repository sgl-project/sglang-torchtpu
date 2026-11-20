# SPDX-License-Identifier: Apache-2.0
"""Packed causal attention: the engine's batched RPA kernel with its KV block pinned to the page.

Decode, prefill, chunked prefill and a packed training batch then walk a row's keys in the same blocks. The
SEQ_ALONG_LANE layout and the page size are part of the contract: the two layouts round differently.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here: the torch_tpu
# Pallas bridge reads the wrapped functions' annotations (``jax.Array``).

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from sglang_tpu.rl.true_on_policy.common import refuse_flags, register

PAGE_SIZE = 256


def pin_kv_block(decode, prefill, page_size):
    """One KV block, the page, for the decode and the prefill launches."""
    if page_size != PAGE_SIZE:
        raise ValueError(
            f"True on-policy attention runs {PAGE_SIZE}-token pages, got {page_size}"
        )
    return dataclasses.replace(decode, bkv_sz=page_size), dataclasses.replace(
        prefill, bkv_sz=page_size
    )


def refuse_rpa_rounding_flags():
    """The RPA tile flags that change rounding must be unset in every process that traces this attention."""
    flags = (
        "SGLANG_TPU_RPA_EXP2_SOFTMAX",
        "SGLANG_TPU_RPA_ONES_V",
        "SGLANG_TPU_RPA_TRANSPOSED",
    )
    refuse_flags(flags, "attention")


@contextlib.contextmanager
def _pinned_kv_blocks():
    """Pin the KV block while the trainer's attention is traced (the engine pins it with a hook)."""
    from sglang_tpu.attention.pallas.batched_rpa import wrapper

    original = wrapper.calculate_block_sizes

    def pinned(model_cfgs, serve_cfgs, vmem_limit_bytes):
        return pin_kv_block(
            *original(model_cfgs, serve_cfgs, vmem_limit_bytes), serve_cfgs.page_size
        )

    wrapper.calculate_block_sizes = pinned
    try:
        yield
    finally:
        wrapper.calculate_block_sizes = original


def _jax_packed_causal_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, cu_seqlens: jax.Array, sm_scale: float
) -> jax.Array:
    """Causal attention within each packed sequence, as the engine computes it.

    q [T, Hq, D], k and v [T, Hkv, D] bf16; cu_seqlens [n + 1] int32 sequence starts (rows past cu_seqlens[-1] are
    padding); sm_scale is the layer's softmax scale (a static argument). Each sequence gets its own pages; the
    kernel writes its K/V there and attends, as an engine extend with no cached prefix does. Plain causal attention
    over a bf16 cache only: no sliding window, logit cap or attention sink.
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
    # One idle sequence slot past the batch, as the serving backend pads it.
    page_indices = jnp.concatenate(
        [page_indices, jnp.zeros((1, pages_per_seq), page_indices.dtype)]
    ).reshape(-1)
    kv_lens = jnp.concatenate([lens, jnp.zeros((1,), lens.dtype)]).astype(jnp.int32)
    cu_q_lens = jnp.concatenate([cu_seqlens, cu_seqlens[-1:]]).astype(jnp.int32)
    layout = configs.KVLayout.SEQ_ALONG_LANE
    distribution = jnp.full((3,), n, jnp.int32)  # every sequence is an extend
    # The kernel sizes its cache and tiles from the physical TPU's metadata while it is traced, as in the engine.
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
