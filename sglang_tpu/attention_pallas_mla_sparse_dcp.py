# SPDX-License-Identifier: Apache-2.0
"""Decode-context-parallel for the TPU sparse MLA backend.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from typing import Optional

import torch

from sglang_tpu.pallas_export import argnums, tpu_info_for_cpu_export

__all__ = [
    "build_sparse_mla_dcp_op",
    "index_page_indices",
]

_OPS: dict = {}


def build_sparse_mla_dcp_op(
    *, dcp_size: int, zero_rope: bool, gather_chunk_tokens: Optional[int] = None
):
    """Register `sparse_mla_attention_dcp` as a torch custom op, once per key.
    """
    key = (dcp_size, zero_rope, gather_chunk_tokens)
    if key in _OPS:
        return _OPS[key]

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.mla_sparse_dcp import sparse_mla_attention_dcp

    op_name = f"sglang_tpu_pallas::sparse_mla_attention_dcp{dcp_size}"
    chunk_kwargs = {}
    if gather_chunk_tokens is not None:
        op_name += f"_g{gather_chunk_tokens}"
        chunk_kwargs["gather_chunk_tokens"] = gather_chunk_tokens
    if zero_rope:
        op_name += "_zero_rope"

    def _wrapped(
        ql_nope: jax.Array,
        q_pe: jax.Array,
        new_kv_c: jax.Array,
        new_k_pe: jax.Array,
        cache_nope: jax.Array,
        cache_rope: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float,
        k_scale: Optional[float],
        v_scale: Optional[float],
        topk_indices: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return sparse_mla_attention_dcp(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_nope,
            cache_rope,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            topk_indices,
            dcp_rank,
            sm_scale=sm_scale,
            k_scale=1.0 if k_scale is None else k_scale,
            dcp_size=dcp_size,
            zero_rope=zero_rope,
            **chunk_kwargs,
        )

    nope_argnum, rope_argnum = argnums(_wrapped, "cache_nope", "cache_rope")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "sm_scale", "k_scale", "v_scale"),
        donate_argnums=(nope_argnum, rope_argnum),
        input_output_aliases={nope_argnum: 2, rope_argnum: 3},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("cache_nope", "cache_rope"),
        device_types=["tpu"],
    )
    def custom_op(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        cache_nope: torch.Tensor,
        cache_rope: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        sm_scale: float,
        k_scale: Optional[float],
        v_scale: Optional[float],
        topk_indices: torch.Tensor,
        dcp_rank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with tpu_info_for_cpu_export():
            output, lse, _nope, _rope = jax_kernel(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache_nope,
                cache_rope,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                sm_scale,
                k_scale,
                v_scale,
                topk_indices,
                dcp_rank,
            )
        return output, lse

    def _fake_impl(ql_nope: torch.Tensor, *args, **kwargs):
        return (
            torch.empty_like(ql_nope),
            ql_nope.new_empty(ql_nope.shape[:2], dtype=torch.float32),
        )

    custom_op.register_fake(_fake_impl)

    _OPS[key] = custom_op
    return custom_op


def index_page_indices(page_indices: torch.Tensor, *, num_seqs: int, dcp: int):
    """
    This function computes the page indices for indxer cache.

    Since latent cache is sharded on DCP axis and indexer is replicated,
    each indexer cache page spans page_size * dcp tokens.
    
    If we view indexer cache in page_size granularity, we then need this function 
    to compute the mappins.

    E.g. page_indices [1, 4] with dcp=4 -> [4, 5, 6, 7, 16, 17, 18, 19].
    """
    stripe = page_indices.view(num_seqs, -1, 1).to(torch.int32)
    offsets = torch.arange(dcp, dtype=torch.int32, device=page_indices.device)
    return (stripe * dcp + offsets).reshape(-1).contiguous()
