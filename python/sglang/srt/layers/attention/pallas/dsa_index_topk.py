# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's device step: quantise, store the new index keys, select top-k.

One JAX function behind one custom op, so a layer's indexer costs a single
launch per forward. The keys go into a paged cache of their own, addressed by
the same page table as the latent cache; the selection is the StreamIndex
kernel over that cache.

Queries and keys are quantised to FP8 here, each vector with one power-of-two
scale (UE8M0). A key's scale is stored next to it as one byte; a query's scale
is folded into its head weight, which is how the kernel expects both.

The key write is an XLA scatter.
TODO(sglang-torchtpu#1072): move to the Pallas key writer and the 132-byte
sequence-along-lane record once that writer is ported.
"""

import jax
import jax.numpy as jnp

from sglang.srt.hardware_backend.tpu.dsv4.streamindex_topk import (
    KVLayout,
    streamindex_topk,
)
from sglang.srt.hardware_backend.tpu.dsv4.streamindex_topk.streamindex_topk import (
    get_kv_cache_shape,
)
from sglang.srt.layers.attention.pallas.mla_sparse import kv_cache_utils

# The only index key width the DSA checkpoints use.
INDEX_HEAD_DIM = 128
# A stored key is 128 FP8 bytes and one UE8M0 scale byte; the cache pads it to 256.
INDEX_KEY_PACKED_BYTES = INDEX_HEAD_DIM + 1

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
# UE8M0 stores a scale as its exponent biased by 127, one byte; float32 uses the
# same bias, and its normal range bounds the exponents used here.
_EXPONENT_BIAS = 127
_MIN_EXPONENT = -126
_MAX_EXPONENT = 127
_F32_MANTISSA_BITS = 23

# (decode, prefill, mixed) block sizes the kernel was tuned with upstream.
_NUM_KV_PAGES_PER_BLOCK = (2, 2, 2)
_NUM_QUERIES_PER_BLOCK = (1, 128, 128)


def index_key_cache_shape(num_pages: int, page_size: int) -> tuple[int, ...]:
    """Shape of one layer's index-key cache: ``uint8[pages, page_size // 4, 4, 256]``."""
    return get_kv_cache_shape(
        num_pages, page_size, INDEX_HEAD_DIM, KVLayout.HEAD_ALONG_SUBLANE
    )


def quantize_fp8_ue8m0(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FP8 over the last axis with one power-of-two scale per vector.

    Returns the FP8 values, the scale as ``f32[..., 1]`` and its biased
    exponent as ``u8[..., 1]``.
    """
    x = x.astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    # Round the scale up to a power of two, so the largest element still fits.
    # Read the exponent off the float32 bits: log2 and exp2 are not exact on TPU.
    bits = jax.lax.bitcast_convert_type(abs_max / _FP8_MAX, jnp.int32)
    has_fraction = (bits & ((1 << _F32_MANTISSA_BITS) - 1)) != 0
    exponent = (bits >> _F32_MANTISSA_BITS) - _EXPONENT_BIAS + has_fraction
    # An all-zero vector lands on the smallest exponent and stays exact zeros.
    exponent = jnp.clip(exponent, _MIN_EXPONENT, _MAX_EXPONENT)
    scale = jax.lax.bitcast_convert_type(
        (exponent + _EXPONENT_BIAS) << _F32_MANTISSA_BITS, jnp.float32
    )
    values = jnp.clip(x / scale, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn)
    return values, scale, (exponent + _EXPONENT_BIAS).astype(jnp.uint8)


def dsa_indexer_step(
    query: jax.Array,  # float[max_num_tokens, index_n_heads, 128]
    key: jax.Array,  # float[max_num_tokens, 128]
    head_weights: jax.Array,  # float[max_num_tokens, index_n_heads]
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, 256]
    kv_lens: jax.Array,  # i32[max_num_seqs], includes this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the active sequence count
    *,
    weight_scale: float,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Quantise this step's queries and keys, then run ``dsa_index_topk``.

    ``weight_scale`` is the constant factor on the head weights, the indexer's
    softmax scale times ``index_n_heads ** -0.5``.
    """
    q_fp8, q_scale, _ = quantize_fp8_ue8m0(query)
    k_fp8, _, k_scale_byte = quantize_fp8_ue8m0(key)
    weights = head_weights.astype(jnp.float32) * q_scale[..., 0] * weight_scale
    k_packed = jnp.concatenate(
        [jax.lax.bitcast_convert_type(k_fp8, jnp.uint8), k_scale_byte], axis=-1
    )
    return dsa_index_topk(
        q_fp8,
        weights,
        k_packed,
        cache_k,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        topk=topk,
    )


def dsa_index_topk(
    q: jax.Array,  # fp8[max_num_tokens, index_n_heads, 128]
    weights: jax.Array,  # f32[max_num_tokens, index_n_heads], q scale folded in
    k_packed: jax.Array,  # u8[max_num_tokens, 129]: FP8 key bytes, then the scale
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, 256]
    kv_lens: jax.Array,  # i32[max_num_seqs], includes this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the active sequence count
    *,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Returns ``i32[max_num_tokens, topk]`` and the updated key cache.

    A row holds positions within the token's own sequence, selected ones first
    and in no promised order, then ``-1``. A token with at most ``topk`` causal
    keys gets all of them.
    """
    if k_packed.shape[-1] != INDEX_KEY_PACKED_BYTES:
        raise ValueError(
            f"index keys must be {INDEX_KEY_PACKED_BYTES} packed bytes, "
            f"got {k_packed.shape[-1]}."
        )
    num_pages, rows_per_page, kv_packing, record_bytes = cache_k.shape
    page_size = rows_per_page * kv_packing
    # The pool allocates this buffer without importing the kernel.
    if cache_k.shape != index_key_cache_shape(num_pages, page_size):
        raise ValueError(
            f"index-key cache is {cache_k.shape}, the kernel reads "
            f"{index_key_cache_shape(num_pages, page_size)}."
        )

    # Padded batch rows carry a query token as well; only active ones are stored.
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    # The latent writer's position rule, so both caches agree on a token's slot.
    page, slot = kv_cache_utils.get_page_and_slot(
        k_packed.shape[0], kv_lens, page_indices, active_cu_q_lens, page_size
    )
    records = jnp.pad(k_packed, ((0, 0), (0, record_bytes - INDEX_KEY_PACKED_BYTES)))
    # Inactive tokens carry an out-of-range page, which the scatter drops.
    cache_k = cache_k.at[page, slot // kv_packing, slot % kv_packing].set(
        records, mode="drop"
    )

    # No early exit. Lowered into the compiled serving graph, its conditional
    # corrupts the logits of the sampled position, measured on GLM-5.2 at TP8.
    # TODO(sglang-torchtpu#1072): turn it on once that is understood.
    topk_indices = streamindex_topk(
        q,
        weights,
        cache_k,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=topk,
        compression_ratio=1,
        num_kv_pages_per_block=_NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    )
    return topk_indices, cache_k
