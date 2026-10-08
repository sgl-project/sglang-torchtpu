# SPDX-License-Identifier: Apache-2.0
"""Direct TPU correctness coverage for batched-RPA with Context Parallelism (DCP and PCP)."""

from __future__ import annotations

import dataclasses
import functools
import os
import unittest
from typing import Any

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

os.environ.setdefault("JAX_PLATFORMS", "tpu")


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


if _has_jax_tpu():
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
else:
    jax = None
    lax = None
    jnp = None
    Mesh = Any
    NamedSharding = Any
    P = Any


try:
    from absl.testing import parameterized

    _HAS_PARAMETERIZED = True
except Exception:
    _HAS_PARAMETERIZED = False


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclasses.dataclass(frozen=True)
class ModelParams:
    """Unsharded parameters for LLM models."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_devices: list[int]
    hidden_size: int | None = None

    def get_sharded_heads(self, num_devices: int) -> tuple[int, int]:
        assert self.num_q_heads % num_devices == 0
        sharded_kv_heads = self.num_kv_heads // num_devices
        if sharded_kv_heads == 0:
            assert num_devices % self.num_kv_heads == 0
            sharded_kv_heads = 1
        return self.num_q_heads // num_devices, sharded_kv_heads

    def __getitem__(self, key: str):
        return getattr(self, key)


MODEL_CONFIGS = {
    "Qwen3-32B": ModelParams(
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[2],
    ),
    "Qwen3.5": ModelParams(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        num_devices=[8],
        hidden_size=4096,
    ),
    "Qwen3-Coder-480B": ModelParams(
        num_q_heads=96,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[8],
        hidden_size=6144,
    ),
    "Gemma4-31B": ModelParams(
        num_q_heads=32,
        num_kv_heads=16,
        head_dim=256,
        num_devices=[8],
        hidden_size=5376,
    ),
}


def merge_kv(
    k: jax.Array,
    v: jax.Array,
):
    from sglang.srt.layers.attention.pallas.batched_rpa import utils

    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = utils.get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = utils.align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = utils.align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1).reshape(
            max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
        ),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def ref_ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    from sglang.srt.layers.attention.pallas.batched_rpa import utils

    if out_dtype is None:
        out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        mask_value = -float(jnp.finfo(out_dtype).max)

    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    merged_kv = merge_kv(keys, values)
    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = kv_cache.shape
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert utils.get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == utils.align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    outputs = []

    for i in range(int(distribution[-1])):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(int(kv_len), page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :actual_head_dim]

        # Update the kv cache.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len - q_len : kv_len].set(
            merged_kv[q_start:q_end]
        )
        kv_cache = kv_cache.at[indices].set(gathered_kv.reshape(gathered_shape))

        kv = gathered_kv.reshape(-1, num_kv_heads_x2, head_dim)[
            :, : actual_num_kv_heads * 2, :
        ].reshape(-1, actual_num_kv_heads, head_dim * 2)
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        attn = jnp.einsum(
            "qhd,khd->hqk", q, k, preferred_element_type=jnp.float32
        ).astype(out_dtype)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if use_causal_mask:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1
            )
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    return result, kv_cache


@dataclasses.dataclass(frozen=True)
class PCPMetadata:
    query_start_loc: jax.Array
    kv_cache_lens: jax.Array
    q_pos_offsets: jax.Array
    new_kv_page_indices: jax.Array | None = None
    update_kv_cache: jax.Array | None = None
    cache_pages: int = 0


@dataclasses.dataclass(frozen=True)
class AttentionMetadata:
    input_positions: jax.Array
    seq_lens: jax.Array
    block_tables: jax.Array
    request_distribution: jax.Array
    pcp: PCPMetadata | None = None


def merge_attn_states(
    cache_out: jax.Array,
    cache_lse: jax.Array,
    query_out: jax.Array,
    query_lse: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """LSE-weighted merge of two disjoint attention spans."""
    max_lse = jnp.maximum(cache_lse, query_lse)
    max_lse_safe = jnp.where(jnp.isinf(max_lse), 0.0, max_lse)
    exp_cache = jnp.exp(cache_lse - max_lse_safe)
    exp_query = jnp.exp(query_lse - max_lse_safe)
    sum_exp = exp_cache + exp_query
    denom = jnp.where(sum_exp == 0.0, 1.0, sum_exp)
    merged_out = (
        cache_out * exp_cache[..., None] + query_out * exp_query[..., None]
    ) / denom[..., None]
    merged_lse = jnp.where(sum_exp == 0.0, -jnp.inf, max_lse_safe + jnp.log(denom))
    return merged_out, merged_lse


def pcp_forward_batched(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    md: AttentionMetadata,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    use_causal_mask: bool = True,
    kv_layout: Any = None,
) -> tuple[jax.Array, jax.Array]:
    from sglang.srt.layers.attention.pallas.batched_rpa import configs, wrapper

    if kv_layout is None:
        kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE

    pcp_axis = "pcp"
    pcp_size = mesh.shape[pcp_axis]
    num_seqs = md.seq_lens.shape[0]
    if (
        md.pcp is None
        or md.pcp.new_kv_page_indices is None
        or md.pcp.update_kv_cache is None
    ):
        raise ValueError(
            "pcp_forward_batched needs pcp.new_kv_page_indices and pcp.update_kv_cache."
        )

    q_spec = P(pcp_axis, "model", None)
    kv_spec = P(pcp_axis, "model", None)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        kv_cache_spec = P(None, "model", None, None, pcp_axis)
    else:
        kv_cache_spec = P(None, pcp_axis, "model", None, None)

    common = dict(
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=pcp_size,
        kv_layout=kv_layout,
        return_lse=True,
    )

    cache_pages = md.pcp.cache_pages

    def _shard_fn(
        q_local,
        k_local,
        v_local,
        kv_cache_local,
        kv_lens_local,
        kv_cache_lens_local,
        page_indices_local,
        distribution_local,
        new_kv_page_indices,
        pcp_cu_q_lens_local,
        pcp_q_pos_offsets_local,
        pcp_update_kv_cache_local,
    ):
        cp_rank = jnp.reshape(lax.axis_index(pcp_axis), (1,)).astype(jnp.int32)
        kv_new_lens = kv_lens_local - kv_cache_lens_local

        def all_gather_tokens(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        def _per_request(x):
            heads = x[::2]
            pad = [(0, num_seqs - heads.shape[0])] + [(0, 0)] * (x.ndim - 1)
            return jnp.pad(heads, pad)

        cu_cache = pcp_cu_q_lens_local[0][::2]
        cu_cache = jnp.pad(cu_cache, (0, num_seqs + 1 - cu_cache.shape[0]), mode="edge")
        dist_cache = (
            jnp.array([0, 0, 0], jnp.int32).at[2].set(distribution_local[2] // 2)
        )

        # ---- Cache phase --------------------------------------------------
        if cache_pages == 0:
            context_out = context_lse = None
        else:
            context_out, _, context_lse = wrapper.ragged_paged_attention(
                q_local,
                k_local,
                v_local,
                kv_cache_local,
                _per_request(kv_lens_local),
                _per_request(page_indices_local.reshape(num_seqs, -1)).reshape(-1),
                cu_cache,
                dist_cache,
                cp_rank=cp_rank,
                kv_new_lens=_per_request(kv_new_lens),
                attention_scope=configs.AttentionScope.CACHE_ONLY,
                pcp_ring_axis_name=pcp_axis,
                pcp_ring_mesh_axis_names=tuple(mesh.axis_names),
                update_kv_cache=False,
                **common,
            )
            context_out = context_out.astype(jnp.float32)
            context_lse = context_lse.astype(jnp.float32)

        # ---- Current phase ------------------------------------------------
        q_positions = kv_cache_lens_local + pcp_q_pos_offsets_local[0]
        k_curr = all_gather_tokens(k_local)
        v_curr = all_gather_tokens(v_local)
        curr_out, kv_cache_updated, curr_lse = wrapper.ragged_paged_attention(
            q_local,
            k_curr,
            v_curr,
            kv_cache_local,
            kv_lens_local,
            page_indices_local,
            pcp_cu_q_lens_local[0],
            distribution_local,
            cp_rank=cp_rank,
            kv_new_lens=kv_new_lens,
            q_positions=q_positions,
            update_kv_cache=pcp_update_kv_cache_local[0],
            new_kv_page_indices=new_kv_page_indices,
            attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
            use_causal_mask=use_causal_mask,
            **common,
        )

        if context_out is None:
            out = curr_out
        else:
            out, _ = merge_attn_states(
                context_out,
                context_lse,
                curr_out.astype(jnp.float32),
                curr_lse.astype(jnp.float32),
            )
        return kv_cache_updated, out.astype(q_local.dtype)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(),
            P(),
            P(),
            P(),
            P(),
            P(pcp_axis, None),
            P(pcp_axis, None),
            P(pcp_axis, None),
        ),
        out_specs=(kv_cache_spec, q_spec),
        check_vma=False,
    )(
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.pcp.kv_cache_lens,
        md.block_tables,
        md.request_distribution,
        md.pcp.new_kv_page_indices,
        md.pcp.query_start_loc,
        md.pcp.q_pos_offsets,
        md.pcp.update_kv_cache,
    )


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaDecodeContextParallelism(
    parameterized.TestCase if _HAS_PARAMETERIZED else unittest.TestCase
):
    """Direct validation of 2-phase mixed decode/prefill CP on batched RPA."""

    def _test_two_phase_attention_mixed_engine(
        self,
        seq_lens: list[tuple[int, int]],
        cp_group_size: int,
        num_heads: tuple[int, int] = (16, 2),
        head_dim: int = 128,
        kv_layout: Any = None,
        rtol: float = 1e-1,
        atol: float = 1e-1,
    ):
        from sglang.srt.layers.attention.pallas.batched_rpa import (
            configs,
            utils,
            wrapper,
        )

        if kv_layout is None:
            kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE

        use_seq_on_lane = kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        _new_tokens_only_kwargs = {
            "attention_scope": configs.AttentionScope.NEW_TOKENS_ONLY,
            "kv_layout": kv_layout,
        }
        _cache_only_kwargs = {
            "attention_scope": configs.AttentionScope.CACHE_ONLY,
            "kv_layout": kv_layout,
        }

        max_num_batched_tokens = 512
        max_num_seq = 8
        q_dtype = jnp.bfloat16
        kv_dtype = jnp.bfloat16

        local_page_size = 128
        global_page_size = local_page_size * cp_group_size
        page_size = global_page_size

        cu_q_lens = [0]
        kv_lens_list = []
        for q_len, kv_len in seq_lens:
            assert q_len <= kv_len
            cu_q_lens.append(cu_q_lens[-1] + q_len)
            kv_lens_list.append(kv_len)
        max_num_batched_tokens = max(
            utils.align_to(cu_q_lens[-1], 128), max_num_batched_tokens
        )
        max_num_seq = max(utils.align_to(len(seq_lens), 8), max_num_seq)
        max_kv_len = max(kv_lens_list)
        pages_per_seq = cdiv(max_kv_len, page_size)
        num_q_heads, num_kv_heads = num_heads
        page_cnt = 0
        page_indices_list = []
        kv_packing = utils.get_dtype_packing(kv_dtype)
        padded_head_dim = utils.align_to(head_dim, 128)
        num_kv_heads_x2 = utils.align_to(num_kv_heads * 2, kv_packing)
        for kv_len in kv_lens_list:
            num_pages_for_seq = cdiv(kv_len, page_size)
            indices = page_cnt + jnp.arange(num_pages_for_seq, dtype=jnp.int32)
            indices = jnp.pad(
                indices,
                ((0, pages_per_seq - indices.shape[0]),),
                constant_values=0,
            )
            page_indices_list.append(indices)
            page_cnt += num_pages_for_seq
        num_pages = max(page_cnt + 8, 16)
        kv_cache_shape = (
            num_pages,
            page_size,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            padded_head_dim,
        )

        @jax.jit
        def _init_tensors(key):
            k1, k2, k3, k4 = jax.random.split(key, 4)
            q = (
                jax.random.uniform(
                    k1,
                    (max_num_batched_tokens, num_q_heads, head_dim),
                    dtype=jnp.float32,
                    minval=-0.5,
                    maxval=0.5,
                )
                * 0.1
            ).astype(q_dtype)
            k = (
                jax.random.uniform(
                    k2,
                    (max_num_batched_tokens, num_kv_heads, head_dim),
                    dtype=jnp.float32,
                    minval=-0.5,
                    maxval=0.5,
                )
                * 0.1
            ).astype(kv_dtype)
            v = (
                jax.random.uniform(
                    k3,
                    (max_num_batched_tokens, num_kv_heads, head_dim),
                    dtype=jnp.float32,
                    minval=-0.5,
                    maxval=0.5,
                )
                * 0.1
            ).astype(kv_dtype)
            cache = (
                jax.random.uniform(
                    k4,
                    kv_cache_shape,
                    dtype=jnp.float32,
                    minval=-0.5,
                    maxval=0.5,
                )
                * 0.1
            ).astype(kv_dtype)
            return q, k, v, cache

        q, k, v, kv_cache = _init_tensors(jax.random.key(1234))
        page_indices = jnp.stack(page_indices_list, axis=0)
        page_indices = jnp.pad(
            page_indices,
            ((0, max_num_seq - page_indices.shape[0]), (0, 0)),
            constant_values=0,
        )
        page_indices = page_indices.reshape(-1)
        cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seq + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
        kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

        if use_seq_on_lane:
            kv_cache_sal = (
                kv_cache.reshape(num_pages, page_size, num_kv_heads_x2, padded_head_dim)
                .transpose(0, 2, 3, 1)
                .reshape(
                    num_pages,
                    num_kv_heads_x2,
                    padded_head_dim // kv_packing,
                    kv_packing,
                    page_size,
                )
            )

        def _sal_to_has(sal_cache):
            lp = local_page_size
            return (
                sal_cache.reshape(num_pages, num_kv_heads_x2, padded_head_dim, lp)
                .transpose(0, 3, 1, 2)
                .reshape(
                    num_pages,
                    lp,
                    num_kv_heads_x2 // kv_packing,
                    kv_packing,
                    padded_head_dim,
                )
            )

        def get_kv_cache_for_rank(rank):
            lp = local_page_size
            if use_seq_on_lane:
                return kv_cache_sal[:, :, :, :, rank * lp : (rank + 1) * lp]
            else:
                return kv_cache[:, rank * lp : (rank + 1) * lp, ...]

        kwargs = {"use_causal_mask": True}
        expected_out, expected_kv_cache = ref_ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            **kwargs,
        )
        query_outs = []
        query_lses = []
        context_outs = []
        context_lses = []
        for rank in range(cp_group_size):
            query_out, updated_kv_cache, query_lse = wrapper.ragged_paged_attention(
                q,
                k,
                v,
                get_kv_cache_for_rank(rank),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                cp_rank=jnp.array([rank], dtype=jnp.int32),
                cp_group_size=cp_group_size,
                return_lse=True,
                **_new_tokens_only_kwargs,
                **kwargs,
            )
            query_outs.append(query_out)
            query_lses.append(query_lse)

            context_out, final_kv_cache, context_lse = wrapper.ragged_paged_attention(
                q,
                k,
                v,
                updated_kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                cp_rank=jnp.array([rank], dtype=jnp.int32),
                cp_group_size=cp_group_size,
                return_lse=True,
                **_cache_only_kwargs,
                **kwargs,
            )
            context_outs.append(context_out)
            context_lses.append(context_lse)

        outs_to_merge = [query_outs[0].astype(jnp.float32)]
        lses_to_merge = [query_lses[0].astype(jnp.float32)]
        for rank in range(cp_group_size):
            outs_to_merge.append(context_outs[rank].astype(jnp.float32))
            lses_to_merge.append(context_lses[rank].astype(jnp.float32))

        stacked_lses = jnp.stack(lses_to_merge, axis=0)
        max_lse = jnp.max(stacked_lses, axis=0)
        exp_sums = jnp.zeros_like(max_lse)
        weighted_outs = jnp.zeros_like(outs_to_merge[0])

        for out, lse in zip(outs_to_merge, lses_to_merge):
            exp_val = jnp.exp(lse - max_lse)
            exp_sums += exp_val
            weighted_outs += out * exp_val[..., None]

        merged_out = (weighted_outs / exp_sums[..., None]).astype(expected_out.dtype)
        np.testing.assert_allclose(
            np.asarray(merged_out[: cu_q_lens[distribution[-1]]]),
            np.asarray(expected_out),
            rtol=rtol,
            atol=atol,
        )

    def test_two_phase_attention_head_along_sublane(self):
        from sglang.srt.layers.attention.pallas.batched_rpa import configs

        seq_lens = [
            (1, 51),
            (1, 181),
            (1, 257),
            (3, 1024),
            (128, 512),
            (1, 1),
            (128, 128),
        ]
        self._test_two_phase_attention_mixed_engine(
            seq_lens=seq_lens,
            cp_group_size=2,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        )

    def test_two_phase_attention_seq_along_lane(self):
        from sglang.srt.layers.attention.pallas.batched_rpa import configs

        seq_lens = [
            (1, 51),
            (1, 181),
            (1, 257),
            (3, 1024),
            (128, 512),
            (1, 1),
            (128, 128),
        ]
        self._test_two_phase_attention_mixed_engine(
            seq_lens=seq_lens,
            cp_group_size=2,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaPCP(unittest.TestCase):
    """Validation of multi-device Ring Attention Prefill CP (PCP)."""

    def _test_pcp_correctness(
        self,
        pcp: int = 2,
        num_reqs: int = 1,
        kv_layout: Any = None,
        nchunk: int = 2,
        num_heads: tuple[int, int] = (16, 2),
        head_dim: int = 128,
    ):
        from sglang.srt.layers.attention.pallas.batched_rpa import (
            configs,
            wrapper,
        )

        if kv_layout is None:
            kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE

        if len(jax.devices()) < pcp:
            self.skipTest(f"needs >= {pcp} devices")

        NQ, NKV = num_heads
        HD = head_dim
        dtype = jnp.bfloat16
        kv_dtype = jnp.bfloat16
        sm_scale = 1.0 / (HD**0.5)
        page = 128
        two_p = 2 * pcp
        C = 128
        req_chunk = two_p * C
        gpage = page * pcp
        max_seq = max(8, 2 * num_reqs)
        max_ctx = nchunk * req_chunk
        pages_per_seq = max(cdiv(max_ctx, gpage), 1)
        npages = pages_per_seq * num_reqs

        mesh_axis_names = ("pcp", "model")
        mesh = Mesh(np.array(jax.devices()[:pcp]).reshape((pcp, 1)), mesh_axis_names)

        per_rank = wrapper.get_kv_cache_shape(
            npages, page, NKV, HD, kv_dtype, kv_layout=kv_layout
        )
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            cache_shape = (npages, per_rank[1], per_rank[2], per_rank[3], gpage)
            cache_spec = P(None, "model", None, None, "pcp")
        else:
            cache_shape = (npages, gpage, per_rank[2], per_rank[3], per_rank[4])
            cache_spec = P(None, "pcp", "model", None, None)

        def put(x, s):
            return jax.device_put(x, NamedSharding(mesh, s))

        q_spec = P("pcp", "model", None)
        kv_spec = P("pcp", "model", None)

        _pi = np.zeros((max_seq, pages_per_seq), np.int32)
        for i in range(num_reqs):
            _pi[2 * i] = _pi[2 * i + 1] = np.arange(
                i * pages_per_seq, (i + 1) * pages_per_seq
            )
        pi = jnp.asarray(_pi.reshape(-1))
        dist = jnp.array([0, 0, 2 * num_reqs], jnp.int32)

        pcp_cu = np.zeros((pcp, max_seq + 1), np.int32)
        pcp_qp = np.zeros((pcp, max_seq), np.int32)
        for r in range(pcp):
            run = 0
            for i in range(num_reqs):
                toff = (two_p - 1 - r) * C
                treal = int(np.clip(req_chunk - toff, 0, C))
                run += C
                pcp_cu[r, 2 * i + 1] = run
                run += treal if i == num_reqs - 1 else C
                pcp_cu[r, 2 * i + 2 :] = run
                pcp_qp[r, 2 * i] = r * C
                pcp_qp[r, 2 * i + 1] = toff

        pcp_spec = P("pcp", None)
        pcp_write = np.zeros((pcp, max_seq), bool)
        pcp_write[:, 1 : 2 * num_reqs : 2] = True

        req_pages = C // page
        local_len = 2 * C * num_reqs
        _table = np.zeros((max_seq, two_p * req_pages), np.int32)
        for i in range(num_reqs):
            for c in range(two_p):
                rank = c if c < pcp else two_p - 1 - c
                base = (rank * local_len + 2 * C * i + (C if c >= pcp else 0)) // page
                cols = slice(c * req_pages, (c + 1) * req_pages)
                _table[2 * i, cols] = base + np.arange(req_pages)
                _table[2 * i + 1, cols] = base + np.arange(req_pages)
        pcp_table = jnp.asarray(_table.reshape(-1))
        pcp_write = put(jnp.asarray(pcp_write), pcp_spec)
        pcp_cu = put(jnp.asarray(pcp_cu), pcp_spec)
        pcp_qp = put(jnp.asarray(pcp_qp), pcp_spec)

        fns = {}

        def fn_for(cache_pages):
            if cache_pages not in fns:

                @functools.partial(jax.jit, donate_argnums=(0,))
                def fn(cache, q, k, v, kvl, kvcl, _cp=cache_pages):
                    md = AttentionMetadata(
                        input_positions=jnp.zeros(1, jnp.int32),
                        seq_lens=kvl,
                        block_tables=pi,
                        request_distribution=dist,
                        pcp=PCPMetadata(
                            query_start_loc=pcp_cu,
                            kv_cache_lens=kvcl,
                            q_pos_offsets=pcp_qp,
                            new_kv_page_indices=pcp_table,
                            update_kv_cache=pcp_write,
                            cache_pages=_cp,
                        ),
                    )
                    cache, out = pcp_forward_batched(
                        mesh,
                        q,
                        k,
                        v,
                        cache,
                        md,
                        sm_scale=sm_scale,
                        use_causal_mask=True,
                        kv_layout=kv_layout,
                    )
                    return out, cache

                fns[cache_pages] = fn
            return fns[cache_pages]

        def cache_pages_for(computed):
            if computed <= 0:
                return 0
            live = cdiv(computed, gpage)
            return min(1 << max(live - 1, 0).bit_length(), npages)

        def seq_lens_for(req_ctx):
            kvl = np.zeros((max_seq,), np.int32)
            kvcl = np.zeros((max_seq,), np.int32)
            for i in range(num_reqs):
                kvl[2 * i : 2 * i + 2] = req_ctx
                kvcl[2 * i : 2 * i + 2] = max(req_ctx - req_chunk, 0)
            return jnp.asarray(kvl), jnp.asarray(kvcl)

        def to_rank_order(per_req):
            blocks = []
            for r in range(pcp):
                for x in per_req:
                    xc = np.asarray(x).reshape(two_p, C, *x.shape[1:])
                    blocks.append(xc[r])
                    blocks.append(xc[two_p - 1 - r])
            return np.concatenate(blocks, 0)

        def from_rank_order(y, i):
            local = 2 * C * num_reqs
            chunks = [None] * two_p
            for r in range(pcp):
                base = r * local + 2 * C * i
                chunks[r] = y[base : base + C]
                chunks[two_p - 1 - r] = y[base + C : base + 2 * C]
            return np.concatenate(chunks, 0)

        def gen(i, j, which, heads, dt):
            g = np.random.default_rng((0x9E3779B9, i, j, which))
            x = g.standard_normal((req_chunk, heads, HD), np.float32) * 0.5
            return jnp.asarray(x).astype(dt)

        def q_of(i, j):
            return gen(i, j, 0, NQ, dtype)

        def k_of(i, j):
            return gen(i, j, 1, NKV, kv_dtype)

        def v_of(i, j):
            return gen(i, j, 2, NKV, kv_dtype)

        grp = NQ // NKV

        def reference(i, j):
            qq = np.asarray(q_of(i, j), np.float32)
            n = qq.shape[0]
            qg = qq.reshape(n, NKV, grp, HD)
            m = np.full((n, NKV, grp), -np.inf, np.float32)
            den = np.zeros((n, NKV, grp), np.float32)
            acc = np.zeros((n, NKV, grp, HD), np.float32)
            last = np.arange(req_chunk)
            for jj in range(j + 1):
                kk = np.asarray(k_of(i, jj), np.float32)
                vv = np.asarray(v_of(i, jj), np.float32)
                sc = np.einsum("sngd,rnd->sngr", qg, kk) * sm_scale
                if jj == j:
                    keep = np.arange(req_chunk)[None, :] <= last[:, None]
                    sc = np.where(keep[:, None, None, :], sc, -np.inf)
                mn = np.maximum(m, sc.max(-1))
                corr = np.exp(m - mn)
                p = np.exp(sc - mn[..., None])
                den = den * corr + p.sum(-1)
                acc = acc * corr[..., None] + np.einsum("sngr,rnd->sngd", p, vv)
                m = mn
            return (acc / den[..., None]).reshape(n, NQ, HD)

        cache = jax.device_put(
            jnp.zeros(cache_shape, kv_dtype), NamedSharding(mesh, cache_spec)
        )

        for j in range(nchunk):
            req_ctx = (j + 1) * req_chunk
            kvl, kvcl = seq_lens_for(req_ctx)
            fn = fn_for(cache_pages_for(j * req_chunk))
            q_batch = put(
                jnp.asarray(to_rank_order([q_of(i, j) for i in range(num_reqs)])),
                q_spec,
            )
            k_batch = put(
                jnp.asarray(to_rank_order([k_of(i, j) for i in range(num_reqs)])),
                kv_spec,
            )
            v_batch = put(
                jnp.asarray(to_rank_order([v_of(i, j) for i in range(num_reqs)])),
                kv_spec,
            )
            out, cache = fn(cache, q_batch, k_batch, v_batch, kvl, kvcl)
            out = np.asarray(jax.block_until_ready(out), np.float32)
            self.assertTrue(
                np.all(np.isfinite(out)), f"NaN or inf in output at chunk {j}"
            )
            for i in range(num_reqs):
                got = from_rank_order(out, i)
                ref = reference(i, j)
                err = np.abs(got - ref).max()
                rel = float(err / max(np.abs(ref).max(), 1e-6))
                self.assertLess(
                    rel,
                    0.05,
                    f"PCP mismatch for req {i} chunk {j} (kv_layout={kv_layout}):"
                    f" rel_err={rel}, max_abs_err={err}",
                )

    def test_pcp_correctness_head_along_sublane(self):
        from sglang.srt.layers.attention.pallas.batched_rpa import configs

        self._test_pcp_correctness(
            pcp=2,
            num_reqs=1,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            nchunk=2,
        )

    def test_pcp_correctness_seq_along_lane(self):
        from sglang.srt.layers.attention.pallas.batched_rpa import configs

        self._test_pcp_correctness(
            pcp=2,
            num_reqs=1,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            nchunk=2,
        )


if __name__ == "__main__":
    unittest.main()
