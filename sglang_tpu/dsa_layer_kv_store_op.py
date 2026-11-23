# SPDX-License-Identifier: Apache-2.0
"""Torch custom ops for the cache-only store of a DSA layer.

One op per layer and step updates all of the layer's caches in place: the
latent NoPE / RoPE buffers and the index-key cache (or, for the pooled
indexer, its raw and pooled caches). The JAX side is
``attention.pallas.dsa_layer_kv_store``. Ops are built eagerly and called by
name, so a compiled caller stays one graph (see ``kv_fused_rope_write``).
"""

import functools
from typing import Callable

import torch

_PLAIN_OP = "sglang_tpu_pallas::dsa_layer_kv_store"
_POOLED_OP = "sglang_tpu_pallas::dsa_layer_kv_store_pooled"


def _plain_op_name(key_store: str, write_rope: bool) -> str:
    return f"{_PLAIN_OP}_{key_store}" + ("" if write_rope else "_zero_rope")


def _pooled_op_name(
    kpool: int, max_query_length: int, raw_ring_pages_cnt: int, write_rope: bool
) -> str:
    return f"{_POOLED_OP}_k{kpool}_q{max_query_length}_r{raw_ring_pages_cnt}" + (
        "" if write_rope else "_zero_rope"
    )


def _jax_kernel(name: str, store, cache_names: tuple[str, ...]) -> Callable[..., None]:
    """``store`` (positional tensors, returns the caches in order) as a call
    that updates the ``cache_names`` arguments in place."""
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.pallas_export import argnums, tpu_info_for_cpu_export

    caches = argnums(store, *cache_names)
    jax_kernel = _torch_pallas.custom_jax_kernel(
        store,
        name=f"jax_{name.split('::')[1]}",
        donate_argnums=caches,
        input_output_aliases={arg: out for out, arg in enumerate(caches)},
        warn_deprecated=False,
    )

    def run(*args: torch.Tensor) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(*args)

    return run


@functools.lru_cache(maxsize=None)
def _plain_op(key_store: str, write_rope: bool) -> Callable[..., None]:
    from sglang_tpu.attention.pallas.dsa_layer_kv_store import store_dsa_layer_kv

    # Tensors only, unannotated (see kv_fused_rope_write); the static choices
    # are closed over and named in the op.
    def store(
        new_kv_c,
        new_k_pe,
        index_keys,
        cache_nope,
        cache_rope,
        cache_k,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        return store_dsa_layer_kv(
            cache_nope,
            cache_rope,
            cache_k,
            new_kv_c,
            new_k_pe,
            index_keys,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            key_store=key_store,
            write_rope=write_rope,
        )

    name = _plain_op_name(key_store, write_rope)
    run = _jax_kernel(name, store, ("cache_nope", "cache_rope", "cache_k"))

    @torch.library.custom_op(
        name, mutates_args=("cache_nope", "cache_rope", "cache_k"), device_types=["tpu"]
    )
    def op(
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        index_keys: torch.Tensor,
        cache_nope: torch.Tensor,
        cache_rope: torch.Tensor,
        cache_k: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
    ) -> None:
        run(
            new_kv_c,
            new_k_pe,
            index_keys,
            cache_nope,
            cache_rope,
            cache_k,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

    @op.register_fake
    def _(*args: torch.Tensor) -> None:
        return None

    return op


@functools.lru_cache(maxsize=None)
def _pooled_op(
    kpool: int, max_query_length: int, raw_ring_pages_cnt: int, write_rope: bool
) -> Callable[..., None]:
    from sglang_tpu.attention.pallas.dsa_layer_kv_store import (
        store_dsa_layer_kv_pooled,
    )

    def store(
        new_kv_c,
        new_k_pe,
        keys,
        gates,
        ape,
        cache_nope,
        cache_rope,
        raw_cache,
        pooled_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        raw_page_base,
    ):
        return store_dsa_layer_kv_pooled(
            cache_nope,
            cache_rope,
            raw_cache,
            pooled_cache,
            new_kv_c,
            new_k_pe,
            keys,
            gates,
            ape,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            raw_page_base,
            kpool=kpool,
            max_query_length=max_query_length,
            raw_ring_pages_cnt=raw_ring_pages_cnt,
            write_rope=write_rope,
        )

    name = _pooled_op_name(kpool, max_query_length, raw_ring_pages_cnt, write_rope)
    caches = ("cache_nope", "cache_rope", "raw_cache", "pooled_cache")
    run = _jax_kernel(name, store, caches)

    @torch.library.custom_op(name, mutates_args=caches, device_types=["tpu"])
    def op(
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        keys: torch.Tensor,
        gates: torch.Tensor,
        ape: torch.Tensor,
        cache_nope: torch.Tensor,
        cache_rope: torch.Tensor,
        raw_cache: torch.Tensor,
        pooled_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        raw_page_base: torch.Tensor,
    ) -> None:
        run(
            new_kv_c,
            new_k_pe,
            keys,
            gates,
            ape,
            cache_nope,
            cache_rope,
            raw_cache,
            pooled_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            raw_page_base,
        )

    @op.register_fake
    def _(*args: torch.Tensor) -> None:
        return None

    return op


@torch._dynamo.assume_constant_result
def plain_store_op_name(key_store: str, write_rope: bool) -> str:
    """Build the plain-DSA store op eagerly, even under dynamo; its name."""
    _plain_op(key_store, write_rope)
    return _plain_op_name(key_store, write_rope)


@torch._dynamo.assume_constant_result
def pooled_store_op_name(
    kpool: int, max_query_length: int, raw_ring_pages_cnt: int, write_rope: bool
) -> str:
    """Build the pooled-indexer store op eagerly, even under dynamo; its name."""
    _pooled_op(kpool, max_query_length, raw_ring_pages_cnt, write_rope)
    return _pooled_op_name(kpool, max_query_length, raw_ring_pages_cnt, write_rope)


def call_op(name: str, *args: torch.Tensor) -> None:
    """Call a registered op by its ``namespace::name``."""
    namespace, op = name.split("::")
    getattr(getattr(torch.ops, namespace), op)(*args)
