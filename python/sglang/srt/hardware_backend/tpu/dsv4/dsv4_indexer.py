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
# Source: src/vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_indexer.py
"""TPU DeepSeek-V4 CSA Lightning Indexer custom op and backend mixin.

IMPORTANT: Do NOT add ``from __future__ import annotations`` to this module.
``torch_tpu._internal.pallas.custom_jax_kernel`` inspects function parameter
annotations at registration time with ``param.annotation is jax.Array``, which
requires evaluated type objects rather than stringified annotations.
"""

from typing import List, Optional

import jax
import jax.numpy as jnp
import torch

from sglang.srt.hardware_backend.tpu.dsv4.dsv4_memory_pool import (
    DSV4TPUTokenToKVPool,
)
from sglang.srt.hardware_backend.tpu.dsv4.rope import (
    get_or_build_cos_sin_cache,
    rope_quant,
)
from sglang.srt.hardware_backend.tpu.dsv4.streamindex_topk import streamindex_topk
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer

_STREAMINDEX_TOPK_OP_NAME = "sglang_tpu_pallas::dsv4_streamindex_topk"
_streamindex_topk_custom_op = None


def ensure_tpu_indexer_caches(c4_indexer) -> None:
    """Lazily prepare and cache the ``float32`` ``[max_pos, rope_head_dim]``
    ``[cos, sin]`` table on ``c4_indexer``."""
    if hasattr(c4_indexer, "_tpu_caches_ready") and c4_indexer._tpu_caches_ready:
        return
    if c4_indexer.use_fp4_indexer:
        raise NotImplementedError(
            "DeepSeek-V4 on TPU does not support FP4 indexer; "
            "the TPU streamindex_topk kernel consumes FP8 queries and KV cache."
        )
    cos_sin_cache = (
        get_or_build_cos_sin_cache(c4_indexer.freqs_cis)
        if c4_indexer.freqs_cis.is_complex()
        else c4_indexer.freqs_cis
    )
    if isinstance(c4_indexer, torch.nn.Module):
        c4_indexer.register_buffer(
            "_tpu_cos_sin_cache", cos_sin_cache, persistent=False
        )
    else:
        c4_indexer._tpu_cos_sin_cache = cos_sin_cache
    c4_indexer._tpu_caches_ready = True


def _register_streamindex_topk_op():
    global _streamindex_topk_custom_op
    if _streamindex_topk_custom_op is not None:
        return _streamindex_topk_custom_op

    from torch_tpu._internal import pallas as _torch_pallas

    _exported_shapes = set()

    def _wrapped_indexer(
        q: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        indexer_weights: jax.Array,
        cache_kv: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        k: int,
        compression_ratio: int,
        softmax_scale: float,
        n_head: int,
    ) -> jax.Array:
        with tpu_info_for_cpu_export():
            if cache_kv.shape[0] == 0:
                return jnp.zeros((q.shape[0], int(k)), dtype=jnp.int32)

            q_quant, q_scales = rope_quant(
                q, positions, cos_sin_cache, quant_dtype=jnp.float8_e4m3fn
            )
            weights = (
                indexer_weights.astype(jnp.float32)
                * float(softmax_scale)
                * (int(n_head) ** -0.5)
                * q_scales
            )
            _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
            c4_page_size = page_size_per_kv_packing * kv_packing
            pages_per_128 = max(1, 128 // c4_page_size)
            return streamindex_topk(
                q=q_quant,
                indexer_weights=weights,
                cache_kv=cache_kv,
                seq_lens=seq_lens,
                page_indices=page_indices.reshape(-1),
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                k=int(k),
                compression_ratio=int(compression_ratio),
                num_kv_pages_per_block=(
                    3 * pages_per_128,
                    2 * pages_per_128,
                    2 * pages_per_128,
                ),
                num_queries_per_block=(1, 128, 128),
            )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_indexer,
        name="jax_dsv4_streamindex_topk",
        static_argnums=(9, 10, 11, 12),
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _STREAMINDEX_TOPK_OP_NAME,
        mutates_args=(),
        device_types=["tpu"],
    )
    def custom_op(
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        indexer_weights: torch.Tensor,
        cache_kv: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        k: int,
        compression_ratio: int,
        softmax_scale: float,
        n_head: int,
    ) -> torch.Tensor:
        key = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            seq_lens.shape[0],
            page_indices.numel(),
            k,
            compression_ratio,
            softmax_scale,
            n_head,
        )
        if key in _exported_shapes:
            return jax_kernel(
                q,
                positions,
                cos_sin_cache,
                indexer_weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k,
                compression_ratio,
                softmax_scale,
                n_head,
            )
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            out = jax_kernel(
                q,
                positions,
                cos_sin_cache,
                indexer_weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k,
                compression_ratio,
                softmax_scale,
                n_head,
            )
        _exported_shapes.add(key)
        return out

    @custom_op.register_fake
    def _(
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        indexer_weights: torch.Tensor,
        cache_kv: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        k: int,
        compression_ratio: int,
        softmax_scale: float,
        n_head: int,
    ) -> torch.Tensor:
        return torch.empty((q.shape[0], int(k)), dtype=torch.int32, device=q.device)

    _streamindex_topk_custom_op = custom_op
    return _streamindex_topk_custom_op


class C4IndexerTPUBackendMixin:
    """Mixin for :class:`DSV4TPUAttnBackend` providing the TPU Pallas
    ``rope_quant`` + ``streamindex_topk`` Lightning Indexer pipeline."""

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer,
        forward_batch,
        alt_streams: Optional[List] = None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
        skip_compressor: bool = False,
    ) -> Optional[torch.Tensor]:
        del alt_streams, enable_multi_stream, q_lora_ready
        if forward_batch.forward_mode.is_idle() or x.shape[0] == 0:
            return None

        ensure_tpu_indexer_caches(c4_indexer)

        if not skip_compressor:
            self.forward_indexer_compressor(
                x=x,
                forward_batch=forward_batch,
                layer_id=c4_indexer.layer_id,
                compressor=c4_indexer.compressor,
            )

        pool: DSV4TPUTokenToKVPool = self.token_to_kv_pool
        fm = self.forward_metadata

        positions = fm.positions
        if positions is None:
            positions = forward_batch.positions.to(torch.int32)

        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        q, _ = c4_indexer.wq_b(q_lora)
        q = q.reshape(num_queries, c4_indexer.n_local_heads, c4_indexer.head_dim)
        if not q.is_contiguous():
            q = q.contiguous()

        indexer_weights, _ = c4_indexer.weights_proj(x)
        indexer_weights = indexer_weights.reshape(num_queries, c4_indexer.n_local_heads)
        if not indexer_weights.is_contiguous():
            indexer_weights = indexer_weights.contiguous()

        cache_kv = pool.get_indexer_k_buffer(c4_indexer.layer_id)
        seq_lens = fm.seq_lens
        query_start_loc = fm.query_start_loc
        request_distribution = fm.distribution

        k_block_tables = fm.c4_block_tables_flat
        assert k_block_tables is not None, (
            "Missing compressed KV block_tables in forward_metadata for C4 indexer."
        )

        op = _register_streamindex_topk_op()
        topk_indices = op(
            q,
            positions,
            c4_indexer._tpu_cos_sin_cache,
            indexer_weights,
            cache_kv,
            seq_lens,
            k_block_tables,
            query_start_loc,
            request_distribution,
            int(c4_indexer.index_topk),
            int(c4_indexer.compressor.ratio),
            float(c4_indexer.softmax_scale),
            int(c4_indexer.n_local_heads),
        )

        fm.c4_topk_indices = topk_indices

        if not torch.compiler.is_compiling():
            indexer_capturer = get_global_indexer_capturer()
            if indexer_capturer is not None:
                compress_layer_id = pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                indexer_capturer.capture(compress_layer_id, topk_indices)

        return topk_indices
