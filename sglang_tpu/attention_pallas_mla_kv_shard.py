# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA with the latent cache sharded by page across a group of ranks.

The ops behind ``TpuPallasSparseMlaAttnBackend`` when
``SGLANG_TPU_DSA_MLA_KV_SHARDS`` > 1 (layout and math: attention/pallas/
mla_sparse/kv_shard.py). Each rank of a group gathers the group's queries,
keeps the top-k candidates on its own pages, attends them for every head of
the group with the log-sum-exp, and the group merges the partials in FP32,
each rank keeping its own heads.
"""

# custom_jax_kernel reads the op wrappers' jax.Array annotations as objects: no
# ``from __future__ import annotations`` here (it turns them into strings).
import torch

from sglang_tpu.pallas_export import argnums, tpu_info_for_cpu_export

_OPS: dict = {}


def build_kv_shard_attention_op(*, zero_rope: bool, gather_chunk_tokens: int):
    """Torch op: store this step's tokens on their owners' pages, then attend
    this rank's candidates of the indexer's rows; returns (output
    bf16[T, Hg, 512], lse f32[T, Hg])."""
    key = ("attention", zero_rope, gather_chunk_tokens)
    if key in _OPS:
        return _OPS[key]

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.mla_sparse_fused import (
        sparse_mla_attention_kv_shard,
    )

    op_name = f"sglang_tpu_pallas::sparse_mla_kv_shard_g{gather_chunk_tokens}"
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
        distribution: jax.Array,
        topk_indices: jax.Array,
        sm_scale: float,
        k_scale: float,
        shards: int,
        role: int,
        kpool: int,
    ):
        return sparse_mla_attention_kv_shard(
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
            sm_scale=sm_scale,
            k_scale=k_scale,
            shards=shards,
            role=role,
            kpool=kpool,
            zero_rope=zero_rope,
            gather_chunk_tokens=gather_chunk_tokens,
        )

    nope_argnum, rope_argnum = argnums(_wrapped, "cache_nope", "cache_rope")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(
            _wrapped, "sm_scale", "k_scale", "shards", "role", "kpool"
        ),
        donate_argnums=(nope_argnum, rope_argnum),
        input_output_aliases={nope_argnum: 2, rope_argnum: 3},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name, mutates_args=("cache_nope", "cache_rope"), device_types=["tpu"]
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
        distribution: torch.Tensor,
        topk_indices: torch.Tensor,
        sm_scale: float,
        k_scale: float,
        shards: int,
        role: int,
        kpool: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with tpu_info_for_cpu_export():
            out, lse, _nope, _rope = jax_kernel(
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
                sm_scale,
                k_scale,
                shards,
                role,
                kpool,
            )
        return out, lse

    def _fake_impl(ql_nope: torch.Tensor, *args, **kwargs):
        num_tokens, heads, _ = ql_nope.shape
        return (
            torch.empty_like(ql_nope),
            ql_nope.new_empty((num_tokens, heads), dtype=torch.float32),
        )

    custom_op.register_fake(_fake_impl)
    _OPS[key] = custom_op
    return custom_op


def merge_kv_shards(out: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    """Merge the group's partials (each rank: ``out`` [T, Hg, D] for all group
    heads, ``lse`` f32[T, Hg], -inf where it held no candidate) into this rank's
    heads, [T, Hg / shards, D]."""
    shards = group.world_size
    num_tokens, group_heads, dim = out.shape
    heads = group_heads // shards
    lse_all = group.all_gather(lse.unsqueeze(0), dim=0)  # [shards, T, Hg]
    m = lse_all.amax(dim=0)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    weights = torch.exp(lse_all - m)  # 0 for a rank with no candidate
    mine = weights[group.rank_in_group] / weights.sum(dim=0).clamp_min(1e-30)
    held = torch.isfinite(lse).unsqueeze(-1)
    partial = torch.where(held, out.float(), torch.zeros((), device=out.device))
    partial = partial * mine.unsqueeze(-1)
    # [shards, T, heads, D]: chunk r is rank r's heads, summed over the group.
    partial = partial.view(num_tokens, shards, heads, dim).transpose(0, 1).contiguous()
    merged = partial.new_empty((1, num_tokens, heads, dim))
    group.reduce_scatter_tensor(merged, partial)
    return merged[0].to(out.dtype)
