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
# Source: src/vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py
#         src/vllm_torchtpu/models/vllm/deepseek_v4/attention.py
"""TPU Pallas Attention Backend for DeepSeek-V4 (SWA, CSA, HCA).

IMPORTANT: Do NOT add ``from __future__ import annotations`` to this module.
``torch_tpu._internal.pallas.custom_jax_kernel`` inspects function parameter
annotations at registration time with ``param.annotation is jax.Array``, which
requires evaluated type objects rather than stringified annotations.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang_tpu.layers.attention.dsv4.dsv4_compressor import (
    CompressorTPUBackendMixin,
    _register_csa_compressor_op,
    _register_hca_compressor_op,
    _register_indexer_compressor_op,
    ensure_tpu_compressor_caches,
)
from sglang_tpu.layers.attention.dsv4.dsv4_indexer import (
    C4IndexerTPUBackendMixin,
    _register_streamindex_topk_op,
    ensure_tpu_indexer_caches,
)
from sglang_tpu.layers.attention.dsv4.dsv4_memory_pool import (
    DSV4TPUTokenToKVPool,
    get_or_build_cos_sin_cache,
)
from sglang_tpu.utils.pallas_export import tpu_info_for_cpu_export

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)

_LOGICAL_PAGE_SIZE = 256

_ROPE_OP_NAME = "sglang_tpu_pallas::dsv4_rope"
_QNORM_ROPE_OP_NAME = "sglang_tpu_pallas::dsv4_qnorm_rope"
_ATTN_SWA_OP_NAME = "sglang_tpu_pallas::dsv4_attention_swa"
_ATTN_CSA_OP_NAME = "sglang_tpu_pallas::dsv4_attention_csa"
_ATTN_HCA_OP_NAME = "sglang_tpu_pallas::dsv4_attention_hca"

_rope_custom_op = None
_qnorm_rope_custom_op = None
_attn_swa_custom_op = None
_attn_csa_custom_op = None
_attn_hca_custom_op = None


# =============================================================================
# 1. Pallas RoPE & Fused QNorm+RoPE Custom Ops
# =============================================================================


def _register_rope_op():
    global _rope_custom_op
    if _rope_custom_op is not None:
        return _rope_custom_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.layers.attention.dsv4.rope import rope

    _exported_shapes = set()

    def _wrapped_rope(
        x: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        inverse: int,
    ) -> jax.Array:
        with tpu_info_for_cpu_export():
            return rope(x, positions, cos_sin_cache, inverse=bool(inverse))

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_rope,
        name="jax_dsv4_rope",
        static_argnums=(3,),
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _ROPE_OP_NAME,
        mutates_args=(),
        device_types=["tpu"],
    )
    def custom_op(
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        inverse: int,
    ) -> torch.Tensor:
        key = (
            tuple(x.shape),
            x.dtype,
            tuple(positions.shape),
            positions.dtype,
            tuple(cos_sin_cache.shape),
            cos_sin_cache.dtype,
            inverse,
        )
        if key in _exported_shapes:
            return jax_kernel(x, positions, cos_sin_cache, inverse)
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            out = jax_kernel(x, positions, cos_sin_cache, inverse)
        _exported_shapes.add(key)
        return out

    @custom_op.register_fake
    def _(
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        inverse: int,
    ) -> torch.Tensor:
        return x.new_empty(x.shape)

    _rope_custom_op = custom_op
    return _rope_custom_op


def apply_tpu_dsv4_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs_cis_or_cos_sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x
    cos_sin_cache = (
        get_or_build_cos_sin_cache(freqs_cis_or_cos_sin)
        if freqs_cis_or_cos_sin.is_complex()
        else freqs_cis_or_cos_sin
    )
    op = _register_rope_op()
    return op(
        x.contiguous(),
        positions.to(torch.int32).contiguous(),
        cos_sin_cache,
        int(bool(inverse)),
    )


def _register_qnorm_rope_op():
    global _qnorm_rope_custom_op
    if _qnorm_rope_custom_op is not None:
        return _qnorm_rope_custom_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.layers.attention.dsv4.rope import qnorm_rope

    _exported_shapes = set()

    def _wrapped_qnorm_rope(
        q: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        eps: float,
    ) -> jax.Array:
        with tpu_info_for_cpu_export():
            return qnorm_rope(q, positions, cos_sin_cache, eps=float(eps))

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_qnorm_rope,
        name="jax_dsv4_qnorm_rope",
        static_argnums=(3,),
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _QNORM_ROPE_OP_NAME,
        mutates_args=(),
        device_types=["tpu"],
    )
    def custom_op(
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        key = (
            tuple(q.shape),
            q.dtype,
            tuple(positions.shape),
            positions.dtype,
            tuple(cos_sin_cache.shape),
            cos_sin_cache.dtype,
            eps,
        )
        if key in _exported_shapes:
            return jax_kernel(q, positions, cos_sin_cache, eps)
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            out = jax_kernel(q, positions, cos_sin_cache, eps)
        _exported_shapes.add(key)
        return out

    @custom_op.register_fake
    def _(
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return q.new_empty(q.shape)

    _qnorm_rope_custom_op = custom_op
    return _qnorm_rope_custom_op


def apply_tpu_dsv4_qnorm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    freqs_cis_or_cos_sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if q.shape[0] == 0:
        return q
    cos_sin_cache = (
        get_or_build_cos_sin_cache(freqs_cis_or_cos_sin)
        if freqs_cis_or_cos_sin.is_complex()
        else freqs_cis_or_cos_sin
    )
    op = _register_qnorm_rope_op()
    return op(
        q.contiguous(),
        positions.to(torch.int32).contiguous(),
        cos_sin_cache,
        float(eps),
    )


# =============================================================================
# 2. Pallas Core Attention Custom Ops (SWA)
# =============================================================================


def _register_attention_swa_op():
    global _attn_swa_custom_op
    if _attn_swa_custom_op is not None:
        return _attn_swa_custom_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.layers.attention.dsv4.core_attention import (
        mla_sliding_window_ragged_paged_attention,
    )

    _exported_shapes = set()

    def _wrapped_swa(
        q: jax.Array,
        new_kv: jax.Array,
        sw_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        attention_sinks: jax.Array,
        sm_scale: float,
        sliding_window: int,
        logical_page_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            # Ensure bkv_sz >= sliding_window (128) so single_bkv_block=True in VMEM.
            swa_kv_pages = 1 if int(logical_page_size) >= 256 else 2
            output, updated_sw_cache, _, _ = mla_sliding_window_ragged_paged_attention(
                q=q,
                new_kv=new_kv,
                cache_kv=sw_cache,
                kv_lens=kv_lens,
                page_indices=page_indices.reshape(-1),
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                attention_sinks=attention_sinks,
                sm_scale=float(sm_scale),
                sliding_window=int(sliding_window),
                logical_page_size=int(logical_page_size),
                num_kv_pages_per_block=(
                    swa_kv_pages,
                    swa_kv_pages,
                    swa_kv_pages,
                ),
                num_queries_per_block=(1, 32, 32),
                q_compute_block_size=4,
                unnormalized_output=False,
            )
            return output, updated_sw_cache

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_swa,
        name="jax_dsv4_attention_swa",
        static_argnums=(8, 9, 10),
        donate_argnums=(2,),
        input_output_aliases={2: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _ATTN_SWA_OP_NAME,
        mutates_args=("sw_cache",),
        device_types=["tpu"],
    )
    def custom_op(
        q: torch.Tensor,
        new_kv: torch.Tensor,
        sw_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sinks: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        logical_page_size: int,
    ) -> torch.Tensor:
        key = (
            tuple(q.shape),
            q.dtype,
            tuple(new_kv.shape),
            tuple(sw_cache.shape),
            tuple(kv_lens.shape),
            tuple(page_indices.shape),
            tuple(cu_q_lens.shape),
            tuple(attention_sinks.shape),
            sm_scale,
            sliding_window,
            logical_page_size,
        )
        if key in _exported_shapes:
            output, _ = jax_kernel(
                q,
                new_kv,
                sw_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sm_scale,
                sliding_window,
                logical_page_size,
            )
            return output
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            output, _ = jax_kernel(
                q,
                new_kv,
                sw_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sm_scale,
                sliding_window,
                logical_page_size,
            )
        _exported_shapes.add(key)
        return output

    @custom_op.register_fake
    def _(
        q: torch.Tensor,
        new_kv: torch.Tensor,
        sw_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sinks: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        logical_page_size: int,
    ) -> torch.Tensor:
        return q.new_empty(q.shape)

    _attn_swa_custom_op = custom_op
    return _attn_swa_custom_op


def _register_attention_csa_op():
    global _attn_csa_custom_op
    if _attn_csa_custom_op is not None:
        return _attn_csa_custom_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.layers.attention.dsv4.core_attention import (
        mla_sliding_window_ragged_paged_attention,
        sparse_ragged_paged_attention,
    )

    _exported_shapes = set()

    def _wrapped_csa(
        q: jax.Array,
        new_kv: jax.Array,
        sw_cache: jax.Array,
        swa_kv_lens: jax.Array,
        swa_page_indices: jax.Array,
        csa_cache_nope: jax.Array,
        csa_cache_rope: jax.Array,
        topk_indices: jax.Array,
        csa_page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        attention_sinks: jax.Array,
        sm_scale: float,
        sliding_window: int,
        swa_logical_page_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            swa_kv_pages = 1 if int(swa_logical_page_size) >= 256 else 2
            swa_output, updated_sw_cache, swa_l, swa_m = (
                mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=new_kv,
                    cache_kv=sw_cache,
                    kv_lens=swa_kv_lens,
                    page_indices=swa_page_indices.reshape(-1),
                    cu_q_lens=cu_q_lens,
                    distribution=distribution,
                    attention_sinks=attention_sinks,
                    sm_scale=float(sm_scale),
                    sliding_window=int(sliding_window),
                    logical_page_size=int(swa_logical_page_size),
                    num_kv_pages_per_block=(
                        swa_kv_pages,
                        swa_kv_pages,
                        swa_kv_pages,
                    ),
                    num_queries_per_block=(1, 32, 32),
                    q_compute_block_size=4,
                    unnormalized_output=True,
                )
            )
            output = sparse_ragged_paged_attention(
                q=q,
                cache_kv_nope=csa_cache_nope,
                cache_kv_rope=csa_cache_rope,
                topk_indices=topk_indices,
                page_indices=csa_page_indices.reshape(-1),
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                attention_sinks=attention_sinks,
                swa_accumution=swa_output,
                swa_l=swa_l,
                swa_m=swa_m,
                sm_scale=float(sm_scale),
                # Each chunk unrolls a SparseCore csa_gather + TensorCore SparseMLA
                # kernel pair. Using 1024 (instead of 64) keeps num_chunks=1 up to
                # 1024 tokens and 4 at 4096 tokens, cutting 4096-token CSA compile
                # time ~12x and code size ~10x while preserving SparseCore/TensorCore
                # overlap across chunks at large prefill lengths.
                gather_and_attention_chunk_size=1024,
                attention_kernel_batch_size=16,
            )
            return output, updated_sw_cache

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_csa,
        name="jax_dsv4_attention_csa",
        static_argnums=(12, 13, 14),
        donate_argnums=(2,),
        input_output_aliases={2: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _ATTN_CSA_OP_NAME,
        mutates_args=("sw_cache",),
        device_types=["tpu"],
    )
    def custom_op(
        q: torch.Tensor,
        new_kv: torch.Tensor,
        sw_cache: torch.Tensor,
        swa_kv_lens: torch.Tensor,
        swa_page_indices: torch.Tensor,
        csa_cache_nope: torch.Tensor,
        csa_cache_rope: torch.Tensor,
        topk_indices: torch.Tensor,
        csa_page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sinks: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        swa_logical_page_size: int,
    ) -> torch.Tensor:
        key = (
            tuple(q.shape),
            q.dtype,
            tuple(new_kv.shape),
            tuple(sw_cache.shape),
            tuple(swa_kv_lens.shape),
            tuple(swa_page_indices.shape),
            tuple(csa_cache_nope.shape),
            tuple(csa_cache_rope.shape),
            tuple(topk_indices.shape),
            tuple(csa_page_indices.shape),
            tuple(cu_q_lens.shape),
            tuple(attention_sinks.shape),
            sm_scale,
            sliding_window,
            swa_logical_page_size,
        )
        if key in _exported_shapes:
            output, _ = jax_kernel(
                q,
                new_kv,
                sw_cache,
                swa_kv_lens,
                swa_page_indices,
                csa_cache_nope,
                csa_cache_rope,
                topk_indices,
                csa_page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sm_scale,
                sliding_window,
                swa_logical_page_size,
            )
            return output
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            output, _ = jax_kernel(
                q,
                new_kv,
                sw_cache,
                swa_kv_lens,
                swa_page_indices,
                csa_cache_nope,
                csa_cache_rope,
                topk_indices,
                csa_page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sm_scale,
                sliding_window,
                swa_logical_page_size,
            )
        _exported_shapes.add(key)
        return output

    @custom_op.register_fake
    def _(
        q: torch.Tensor,
        new_kv: torch.Tensor,
        sw_cache: torch.Tensor,
        swa_kv_lens: torch.Tensor,
        swa_page_indices: torch.Tensor,
        csa_cache_nope: torch.Tensor,
        csa_cache_rope: torch.Tensor,
        topk_indices: torch.Tensor,
        csa_page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sinks: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        swa_logical_page_size: int,
    ) -> torch.Tensor:
        return q.new_empty(q.shape)

    _attn_csa_custom_op = custom_op
    return _attn_csa_custom_op


def _register_attention_hca_op():
    global _attn_hca_custom_op
    if _attn_hca_custom_op is not None:
        return _attn_hca_custom_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.layers.attention.dsv4.core_attention import (
        mla_ragged_paged_attention,
        mla_sliding_window_ragged_paged_attention,
    )

    _exported_shapes = set()

    def _wrapped_hca(
        q: jax.Array,
        new_kv: jax.Array,
        sw_cache: jax.Array,
        swa_kv_lens: jax.Array,
        swa_page_indices: jax.Array,
        hca_cache_kv: jax.Array,
        hca_kv_lens: jax.Array,
        hca_kv_lens_to_attend: jax.Array,
        hca_page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        attention_sinks: jax.Array,
        sm_scale: float,
        sliding_window: int,
        swa_logical_page_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            swa_kv_pages = 1 if int(swa_logical_page_size) >= 256 else 2
            swa_output, updated_sw_cache, swa_l, swa_m = (
                mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=new_kv,
                    cache_kv=sw_cache,
                    kv_lens=swa_kv_lens,
                    page_indices=swa_page_indices.reshape(-1),
                    cu_q_lens=cu_q_lens,
                    distribution=distribution,
                    attention_sinks=attention_sinks,
                    sm_scale=float(sm_scale),
                    sliding_window=int(sliding_window),
                    logical_page_size=int(swa_logical_page_size),
                    num_kv_pages_per_block=(
                        swa_kv_pages,
                        swa_kv_pages,
                        swa_kv_pages,
                    ),
                    num_queries_per_block=(1, 32, 32),
                    q_compute_block_size=4,
                    unnormalized_output=True,
                )
            )
            output = mla_ragged_paged_attention(
                q=q,
                cache_kv=hca_cache_kv,
                kv_lens=hca_kv_lens,
                kv_lens_to_attend=hca_kv_lens_to_attend,
                page_indices=hca_page_indices.reshape(-1),
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                attention_sinks=attention_sinks,
                swa_accumution=swa_output,
                swa_l=swa_l,
                swa_m=swa_m,
                sm_scale=float(sm_scale),
                num_kv_pages_per_block=(16, 16, 16),
                num_queries_per_block=(1, 32, 32),
            )
            return output, updated_sw_cache

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_hca,
        name="jax_dsv4_attention_hca",
        static_argnums=(12, 13, 14),
        donate_argnums=(2,),
        input_output_aliases={2: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _ATTN_HCA_OP_NAME,
        mutates_args=("sw_cache",),
        device_types=["tpu"],
    )
    def custom_op(
        q: torch.Tensor,
        new_kv: torch.Tensor,
        sw_cache: torch.Tensor,
        swa_kv_lens: torch.Tensor,
        swa_page_indices: torch.Tensor,
        hca_cache_kv: torch.Tensor,
        hca_kv_lens: torch.Tensor,
        hca_kv_lens_to_attend: torch.Tensor,
        hca_page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sinks: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        swa_logical_page_size: int,
    ) -> torch.Tensor:
        key = (
            tuple(q.shape),
            q.dtype,
            tuple(new_kv.shape),
            tuple(sw_cache.shape),
            tuple(swa_kv_lens.shape),
            tuple(swa_page_indices.shape),
            tuple(hca_cache_kv.shape),
            tuple(hca_kv_lens.shape),
            tuple(hca_kv_lens_to_attend.shape),
            tuple(hca_page_indices.shape),
            tuple(cu_q_lens.shape),
            tuple(attention_sinks.shape),
            sm_scale,
            sliding_window,
            swa_logical_page_size,
        )
        if key in _exported_shapes:
            output, _ = jax_kernel(
                q,
                new_kv,
                sw_cache,
                swa_kv_lens,
                swa_page_indices,
                hca_cache_kv,
                hca_kv_lens,
                hca_kv_lens_to_attend,
                hca_page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sm_scale,
                sliding_window,
                swa_logical_page_size,
            )
            return output
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            output, _ = jax_kernel(
                q,
                new_kv,
                sw_cache,
                swa_kv_lens,
                swa_page_indices,
                hca_cache_kv,
                hca_kv_lens,
                hca_kv_lens_to_attend,
                hca_page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sm_scale,
                sliding_window,
                swa_logical_page_size,
            )
        _exported_shapes.add(key)
        return output

    @custom_op.register_fake
    def _(
        q: torch.Tensor,
        new_kv: torch.Tensor,
        sw_cache: torch.Tensor,
        swa_kv_lens: torch.Tensor,
        swa_page_indices: torch.Tensor,
        hca_cache_kv: torch.Tensor,
        hca_kv_lens: torch.Tensor,
        hca_kv_lens_to_attend: torch.Tensor,
        hca_page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sinks: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        swa_logical_page_size: int,
    ) -> torch.Tensor:
        return q.new_empty(q.shape)

    _attn_hca_custom_op = custom_op
    return _attn_hca_custom_op


# =============================================================================
# 3. Forward Metadata & Attention Backend
# =============================================================================


@dataclass
class DSV4TPUForwardMetadata:
    """Per-forward-batch metadata shared across DSV4 compressor, indexer, and attention."""

    # Common ragged batching metadata
    positions: Optional[torch.Tensor] = None  # int32[num_tokens]
    seq_lens: Optional[torch.Tensor] = None  # int32[bs]
    kv_lens: Optional[torch.Tensor] = None  # int32[bs] (alias of seq_lens)
    query_start_loc: Optional[torch.Tensor] = None  # int32[bs + 1]
    cu_seqlens_q: Optional[torch.Tensor] = (
        None  # int32[bs + 1] (alias of query_start_loc)
    )
    distribution: Optional[torch.Tensor] = None  # int32[3]

    # SWA metadata
    swa_block_tables_flat: Optional[torch.Tensor] = None  # int32[bs * pages_per_seq]

    # C4 (CSA + Indexer) metadata
    c4_block_tables_flat: Optional[torch.Tensor] = None  # int32[bs * pages_per_seq]
    c4_state_block_tables_flat: Optional[torch.Tensor] = None  # int32[bs * 6]

    # C128 (HCA) metadata
    c128_block_tables_flat: Optional[torch.Tensor] = None  # int32[bs * pages_per_seq]
    c128_state_block_tables_flat: Optional[torch.Tensor] = None  # int32[bs * 6]
    c128_kv_lens: Optional[torch.Tensor] = None  # int32[bs]
    c128_kv_lens_to_attend: Optional[torch.Tensor] = None  # int32[num_tokens]

    # Ephemeral per-step Top-K output from LightningIndexer for current CSA layer
    c4_topk_indices: Optional[torch.Tensor] = None  # int32[num_tokens, index_topk]


class DSV4TPUAttnBackend(
    CompressorTPUBackendMixin, C4IndexerTPUBackendMixin, AttentionBackend
):
    """TPU Pallas Attention Backend for DeepSeek-V4 (SWA, CSA, HCA)."""

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self.model_runner = model_runner
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool: DSV4TPUTokenToKVPool = model_runner.token_to_kv_pool
        self.max_context_len = model_runner.model_config.context_len
        self.page_size = _LOGICAL_PAGE_SIZE

        hf_config = model_runner.model_config.hf_config
        self.sliding_window: int = int(hf_config.sliding_window)
        self.index_topk: int = int(hf_config.index_topk)

        self.forward_metadata: Optional[DSV4TPUForwardMetadata] = None

        # Eagerly register Pallas custom ops and pre-warm model caches so
        # inspect.signature / custom_op registration and complex64 RoPE table
        # conversion never happen inside a torch.compile traced region.
        _register_rope_op()
        _register_qnorm_rope_op()
        _register_attention_swa_op()
        _register_attention_csa_op()
        _register_attention_hca_op()
        _register_csa_compressor_op()
        _register_hca_compressor_op()
        _register_indexer_compressor_op()
        _register_streamindex_topk_op()
        self._prewarm_model_caches()

    def support_triton(self) -> bool:
        return False

    def _prewarm_model_caches(self) -> None:
        model = self.model_runner.model
        if model is None:
            return
        for module in model.modules():
            if (
                hasattr(module, "wkv_gate")
                and hasattr(module, "ape")
                and hasattr(module, "norm")
                and hasattr(module, "freqs_cis")
            ):
                ensure_tpu_compressor_caches(module)
            if (
                hasattr(module, "wq_b")
                and hasattr(module, "weights_proj")
                and hasattr(module, "compressor")
                and hasattr(module, "freqs_cis")
            ):
                ensure_tpu_indexer_caches(module)
            if hasattr(module, "_local_attn_sink") and hasattr(
                module, "_kernel_num_heads"
            ):
                module._local_attn_sink(module._kernel_num_heads(1))
        for module in model.modules():
            if hasattr(module, "freqs_cis"):
                freqs_cis = module.freqs_cis
                if isinstance(freqs_cis, torch.Tensor) and freqs_cis.is_complex():
                    cos_sin = get_or_build_cos_sin_cache(freqs_cis)
                    if isinstance(module, torch.nn.Module) and (
                        "freqs_cis" in module._buffers
                    ):
                        module.register_buffer("freqs_cis", cos_sin, persistent=False)
                    else:
                        module.freqs_cis = cos_sin

    def get_or_create_metadata(
        self, forward_batch: ForwardBatch
    ) -> DSV4TPUForwardMetadata:
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)
        return self.forward_metadata

    def _compute_metadata(
        self,
        forward_batch: ForwardBatch,
        seq_lens_override: Optional[torch.Tensor] = None,
        seq_lens_cpu_override: Optional[torch.Tensor] = None,
        positions_override: Optional[torch.Tensor] = None,
        num_tokens_per_bs_override: Optional[int] = None,
    ) -> DSV4TPUForwardMetadata:
        bs = forward_batch.batch_size
        if bs == 0:
            return DSV4TPUForwardMetadata()

        device = self.device
        pool = self.token_to_kv_pool

        seq_lens = (
            seq_lens_override
            if seq_lens_override is not None
            else forward_batch.seq_lens
        )
        seq_lens_cpu = (
            seq_lens_cpu_override
            if seq_lens_cpu_override is not None
            else forward_batch.seq_lens_cpu
        )
        positions = (
            positions_override
            if positions_override is not None
            else forward_batch.positions
        )

        # 1. Determine non_empty_bs from TPU bucket padding and DP zero-length rows
        real_bs = int(getattr(forward_batch, "_tpu_real_b", bs))
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            non_empty_bs = int(torch.count_nonzero(seq_lens_cpu[:real_bs]).item())
        else:
            non_empty_bs = int(torch.count_nonzero(seq_lens[:real_bs]).item())

        # 2. Build cu_seqlens_q (query_start_loc) and distribution on host
        forward_mode = forward_batch.forward_mode
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if forward_mode.is_decode_or_idle():
            cu_seqlens_q_cpu = torch.arange(0, bs + 1, dtype=torch.int32)
            dist_cpu = torch.tensor(
                [non_empty_bs, non_empty_bs, non_empty_bs], dtype=torch.int32
            )
            extend_seq_lens_cpu = [1] * bs
        elif (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or num_tokens_per_bs_override is not None
        ):
            if num_tokens_per_bs_override is not None:
                tokens_per_bs = num_tokens_per_bs_override
            elif forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                tokens_per_bs = int(forward_batch.spec_info.draft_token_num)
            elif extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0:
                tokens_per_bs = int(extend_seq_lens_cpu[0])
            else:
                tokens_per_bs = max(1, int(positions.shape[0] // max(1, bs)))
            cu_seqlens_q_cpu = (
                torch.arange(0, bs + 1, dtype=torch.int32) * tokens_per_bs
            )
            dist_cpu = torch.tensor([0, 0, non_empty_bs], dtype=torch.int32)
            extend_seq_lens_cpu = [tokens_per_bs] * bs
        else:
            # Extend / Prefill: compute cumsum on host with explicit int32 dtype
            # so torch_tpu never JIT-compiles an int64 cumsum + slice-scatter on TPU.
            if extend_seq_lens_cpu is not None:
                if isinstance(extend_seq_lens_cpu, torch.Tensor):
                    extend_lens_cpu = extend_seq_lens_cpu[:bs].to(
                        dtype=torch.int32, device="cpu"
                    )
                else:
                    extend_lens_cpu = torch.tensor(
                        extend_seq_lens_cpu[:bs], dtype=torch.int32
                    )
            elif forward_batch.extend_seq_lens is not None:
                extend_lens_cpu = forward_batch.extend_seq_lens[:bs].to(
                    dtype=torch.int32, device="cpu"
                )
            elif seq_lens_cpu is not None:
                if isinstance(seq_lens_cpu, torch.Tensor):
                    extend_lens_cpu = seq_lens_cpu[:bs].to(
                        dtype=torch.int32, device="cpu"
                    )
                else:
                    extend_lens_cpu = torch.tensor(seq_lens_cpu[:bs], dtype=torch.int32)
            else:
                extend_lens_cpu = seq_lens[:bs].to(dtype=torch.int32, device="cpu")
            if extend_lens_cpu.shape[0] < bs:
                extend_lens_cpu = torch.nn.functional.pad(
                    extend_lens_cpu, (0, bs - extend_lens_cpu.shape[0])
                )
            cu_seqlens_q_cpu = torch.zeros(bs + 1, dtype=torch.int32)
            cu_seqlens_q_cpu[1:] = torch.cumsum(
                extend_lens_cpu, dim=0, dtype=torch.int32
            )
            dist_cpu = torch.tensor([0, 0, non_empty_bs], dtype=torch.int32)

        cu_seqlens_q = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
        distribution = dist_cpu.to(device=device, non_blocking=True)
        kv_lens = seq_lens[:bs].to(device=device, dtype=torch.int32)
        if kv_lens.shape[0] < bs:
            kv_lens = torch.nn.functional.pad(kv_lens, (0, bs - kv_lens.shape[0]))

        # 3. Use fixed max_context_len for pages_per_seq so block tables keep a
        # compile-invariant shape (bs * pages_per_seq) across warmup and serving.
        pages_per_seq = max(
            1, (self.max_context_len + self.page_size - 1) // self.page_size
        )

        # 4. Build page tables (SWA, C4, C128) on host and transfer once
        req_pool_indices = forward_batch.req_pool_indices
        req_pool_indices_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)
        req_to_token = self.req_to_token
        if req_to_token.device.type == "cpu":
            req_indices_cpu = (
                req_pool_indices_cpu[:bs].to(torch.int64)
                if req_pool_indices_cpu is not None
                else req_pool_indices[:bs].cpu().to(torch.int64)
            )
            if req_indices_cpu.shape[0] < bs:
                req_indices_cpu = torch.nn.functional.pad(
                    req_indices_cpu, (0, bs - req_indices_cpu.shape[0])
                )
            page_cols = torch.arange(pages_per_seq, dtype=torch.int64) * self.page_size
            if page_cols[-1] >= req_to_token.shape[1]:
                page_cols = page_cols.clamp(max=req_to_token.shape[1] - 1)
            per_seq_tokens = req_to_token[req_indices_cpu[:, None], page_cols[None, :]]
            full_page_indices_cpu = (
                (per_seq_tokens // self.page_size)
                .to(torch.int32)
                .reshape(bs * pages_per_seq)
                .contiguous()
            )
            full_token_starts_cpu = (
                full_page_indices_cpu.to(torch.int64) * self.page_size
            )
            if pool.full_to_swa_index_mapping is not None:
                swa_tokens_cpu = pool.full_to_swa_index_mapping[
                    full_token_starts_cpu.clamp(
                        min=0,
                        max=pool.full_to_swa_index_mapping.shape[0] - 1,
                    )
                ]
                swa_page_indices_cpu = (
                    (swa_tokens_cpu // self.page_size)
                    .clamp(min=0)
                    .to(torch.int32)
                    .contiguous()
                )
            else:
                swa_page_indices_cpu = full_page_indices_cpu

            c4_block_tables_flat = full_page_indices_cpu.to(
                device=device, non_blocking=True
            )
            c128_block_tables_flat = c4_block_tables_flat
            swa_block_tables_flat = swa_page_indices_cpu.to(
                device=device, non_blocking=True
            )
        else:
            req_indices_dev = req_pool_indices[:bs].to(
                device=req_to_token.device, dtype=torch.int64
            )
            if req_indices_dev.shape[0] < bs:
                req_indices_dev = torch.nn.functional.pad(
                    req_indices_dev, (0, bs - req_indices_dev.shape[0])
                )
            per_seq_tokens = req_to_token[req_indices_dev]
            c4_block_tables_flat = (
                (
                    per_seq_tokens[:, : pages_per_seq * self.page_size : self.page_size]
                    // self.page_size
                )
                .to(torch.int32)
                .reshape(bs * pages_per_seq)
                .contiguous()
                .to(device=device)
            )
            c128_block_tables_flat = c4_block_tables_flat
            full_token_starts_cpu = (
                c4_block_tables_flat.cpu().to(torch.int64) * self.page_size
            )
            if pool.full_to_swa_index_mapping is not None:
                swa_tokens_cpu = pool.full_to_swa_index_mapping[
                    full_token_starts_cpu.clamp(
                        min=0,
                        max=pool.full_to_swa_index_mapping.shape[0] - 1,
                    )
                ]
                swa_block_tables_flat = (
                    (swa_tokens_cpu // self.page_size)
                    .clamp(min=0)
                    .to(torch.int32)
                    .contiguous()
                    .to(device=device, non_blocking=True)
                )
            else:
                swa_block_tables_flat = c4_block_tables_flat
            req_indices_cpu = (
                req_pool_indices_cpu
                if req_pool_indices_cpu is not None
                else req_pool_indices.cpu()
            )

        # 5. Build Compressor Ring State Block Tables (6 slots per request)
        is_decode = forward_mode.is_decode_or_idle()
        c4_state_bt = pool.build_state_block_table(
            ratio=4,
            req_pool_indices_cpu=req_indices_cpu,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            is_decode=is_decode,
            bs=bs,
            non_empty_bs=non_empty_bs,
            device=device,
        )
        c128_state_bt = pool.build_state_block_table(
            ratio=128,
            req_pool_indices_cpu=req_indices_cpu,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            is_decode=is_decode,
            bs=bs,
            non_empty_bs=non_empty_bs,
            device=device,
        )
        c4_state_block_tables_flat = (
            c4_state_bt.reshape(-1).contiguous() if c4_state_bt is not None else None
        )
        c128_state_block_tables_flat = (
            c128_state_bt.reshape(-1).contiguous()
            if c128_state_bt is not None
            else None
        )

        # 6. Build HCA (C128) compressed lengths on host; defer per-token
        # causal attend lengths on TPU to the compiled graph to avoid eager JIT.
        # Note: seq_lens_cpu is intentionally left unpadded by _pad_forward_batch_inputs
        # (it has length real_bs), so zero-pad kv_lens_cpu on host to bs so
        # c128_kv_lens always has compile-invariant shape (bs,).
        if seq_lens_cpu is not None:
            if isinstance(seq_lens_cpu, torch.Tensor):
                kv_lens_cpu = seq_lens_cpu[:bs].to(dtype=torch.int32, device="cpu")
            else:
                kv_lens_cpu = torch.tensor(seq_lens_cpu[:bs], dtype=torch.int32)
        else:
            kv_lens_cpu = seq_lens[:bs].to(dtype=torch.int32, device="cpu")
        if kv_lens_cpu.shape[0] < bs:
            kv_lens_cpu = torch.nn.functional.pad(
                kv_lens_cpu, (0, bs - kv_lens_cpu.shape[0])
            )
        c128_kv_lens = (
            torch.div(kv_lens_cpu, 128, rounding_mode="floor")
            .to(torch.int32)
            .to(device=device, non_blocking=True)
        )

        if positions.device.type == "cpu":
            pos_i32 = positions.to(device=device, dtype=torch.int32)
            c128_kv_lens_to_attend = torch.div(
                pos_i32 + 1, 128, rounding_mode="floor"
            ).to(torch.int32)
        else:
            pos_i32 = (
                positions_override.to(device=device, dtype=torch.int32)
                if positions_override is not None
                else None
            )
            c128_kv_lens_to_attend = None

        return DSV4TPUForwardMetadata(
            positions=pos_i32,
            seq_lens=kv_lens,
            kv_lens=kv_lens,
            query_start_loc=cu_seqlens_q,
            cu_seqlens_q=cu_seqlens_q,
            distribution=distribution,
            swa_block_tables_flat=swa_block_tables_flat,
            c4_block_tables_flat=c4_block_tables_flat,
            c4_state_block_tables_flat=c4_state_block_tables_flat,
            c128_block_tables_flat=c128_block_tables_flat,
            c128_state_block_tables_flat=c128_state_block_tables_flat,
            c128_kv_lens=c128_kv_lens,
            c128_kv_lens_to_attend=c128_kv_lens_to_attend,
            c4_topk_indices=None,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        self.forward_metadata = self._compute_metadata(forward_batch)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["EagleDraftInput | EagleVerifyInput"],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        if forward_batch is not None:
            self.forward_metadata = self._compute_metadata(
                forward_batch,
                seq_lens_override=seq_lens,
                seq_lens_cpu_override=seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def store_cache(
        self,
        layer_id: int,
        swa_k: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        **kwargs,
    ) -> None:
        # SWA KV cache updates are fused directly into the Pallas SWA kernel
        # (dsv4_attention_swa / dsv4_attention_csa / dsv4_attention_hca).
        pass

    def prepare_mqa_qkv(
        self,
        layer,
        x_linear: torch.Tensor,
        q_lora: torch.Tensor,
        qkv_a: Optional[torch.Tensor],
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_lora = layer.q_norm(q_lora)
        q, _ = layer.wq_b(q_lora)
        q = q.view(-1, layer.n_local_heads, layer.head_dim)
        if layer.q_head_norm:
            q = apply_tpu_dsv4_qnorm_rope(q, positions, layer.freqs_cis, eps=layer.eps)
        else:
            q = apply_tpu_dsv4_rope(q, positions, layer.freqs_cis, inverse=False)
        if qkv_a is not None:
            kv = qkv_a[..., layer.q_lora_rank :]
        else:
            kv, _ = layer.wkv(x_linear)
        kv = layer.kv_norm(kv)
        kv = apply_tpu_dsv4_rope(kv, positions, layer.freqs_cis, inverse=False)
        if q_out is not None:
            q_out.copy_(q)
        return q_lora, q, kv

    def apply_mqa_inv_rope(
        self,
        layer,
        o: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return apply_tpu_dsv4_rope(o, positions, layer.freqs_cis, inverse=True)

    # =========================================================================
    # Core Attention Forward (Extend & Decode unified in Pallas kernels)
    # =========================================================================

    def _run_dsv4_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        compress_ratio: Optional[int] = None,
        attn_sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.shape[0] == 0:
            return q.new_empty(q.shape)

        metadata = self.get_or_create_metadata(forward_batch)
        pool: DSV4TPUTokenToKVPool = self.token_to_kv_pool
        layer_id = layer.layer_id
        sm_scale = float(layer.scaling)

        q_contig = q.contiguous()
        new_kv_contig = k.contiguous()
        if new_kv_contig.ndim == 3 and new_kv_contig.shape[1] == 1:
            new_kv_contig = new_kv_contig.squeeze(1)

        if attn_sink is None:
            sinks_f32 = torch.zeros(
                q_contig.shape[1], dtype=torch.float32, device=q_contig.device
            )
        else:
            sinks_f32 = attn_sink[: q_contig.shape[1]].to(torch.float32).contiguous()

        sw_cache = pool.get_swa_buffer(layer_id)

        if compress_ratio is None or compress_ratio == 0:
            # Pure SWA layer (Layers 0, 1, 43)
            op = _register_attention_swa_op()
            return op(
                q_contig,
                new_kv_contig,
                sw_cache,
                metadata.kv_lens,
                metadata.swa_block_tables_flat,
                metadata.cu_seqlens_q,
                metadata.distribution,
                sinks_f32,
                sm_scale,
                self.sliding_window,
                self.page_size,
            )
        elif compress_ratio == 4:
            # CSA (C4 Sparse MLA + SWA branch) layer
            csa_cache_nope, csa_cache_rope = pool.get_csa_kv_buffers(layer_id)
            topk_indices = metadata.c4_topk_indices
            metadata.c4_topk_indices = None
            assert topk_indices is not None, (
                f"CSA layer {layer_id} requires metadata.c4_topk_indices from "
                "C4Indexer before running core attention."
            )
            op = _register_attention_csa_op()
            return op(
                q_contig,
                new_kv_contig,
                sw_cache,
                metadata.kv_lens,
                metadata.swa_block_tables_flat,
                csa_cache_nope,
                csa_cache_rope,
                topk_indices,
                metadata.c4_block_tables_flat,
                metadata.cu_seqlens_q,
                metadata.distribution,
                sinks_f32,
                sm_scale,
                self.sliding_window,
                self.page_size,
            )
        elif compress_ratio == 128:
            # HCA (C128 Dense MLA + SWA branch) layer
            hca_cache_kv = pool.get_hca_buffer(layer_id)
            c128_kv_lens = (
                metadata.c128_kv_lens
                if metadata.c128_kv_lens is not None
                else torch.div(metadata.kv_lens, 128, rounding_mode="floor").to(
                    torch.int32
                )
            )
            c128_kv_lens_to_attend = metadata.c128_kv_lens_to_attend
            if c128_kv_lens_to_attend is None:
                pos_i32 = (
                    metadata.positions
                    if metadata.positions is not None
                    else forward_batch.positions.to(torch.int32)
                )
                if pos_i32.shape[0] != q_contig.shape[0]:
                    pos_i32 = pos_i32[: q_contig.shape[0]]
                c128_kv_lens_to_attend = torch.div(
                    pos_i32 + 1, 128, rounding_mode="floor"
                ).to(torch.int32)
            op = _register_attention_hca_op()
            return op(
                q_contig,
                new_kv_contig,
                sw_cache,
                metadata.kv_lens,
                metadata.swa_block_tables_flat,
                hca_cache_kv,
                c128_kv_lens,
                c128_kv_lens_to_attend,
                metadata.c128_block_tables_flat,
                metadata.cu_seqlens_q,
                metadata.distribution,
                sinks_f32,
                sm_scale,
                self.sliding_window,
                self.page_size,
            )
        else:
            raise ValueError(f"Unsupported DSV4 compress_ratio: {compress_ratio}")

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        compress_ratio: Optional[int] = None,
        attn_sink: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._run_dsv4_attention(
            q=q,
            k=k,
            layer=layer,
            forward_batch=forward_batch,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        compress_ratio: Optional[int] = None,
        attn_sink: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._run_dsv4_attention(
            q=q,
            k=k,
            layer=layer,
            forward_batch=forward_batch,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
        )


class DSV4TPUMultiStepBackend:
    """Multi-step speculative decoding wrapper for DSV4TPUAttnBackend."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: list[DSV4TPUAttnBackend] = [
            DSV4TPUAttnBackend(model_runner) for _ in range(self.speculative_num_steps)
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        base_seq_lens = forward_batch.seq_lens
        base_seq_lens_cpu = forward_batch.seq_lens_cpu
        base_positions = forward_batch.positions

        for i in range(self.speculative_num_steps):
            seq_lens_i = base_seq_lens + i if base_seq_lens is not None else None
            seq_lens_cpu_i = (
                base_seq_lens_cpu + i if base_seq_lens_cpu is not None else None
            )
            positions_i = (
                base_positions + i
                if base_positions is not None
                else torch.zeros(bs, dtype=torch.int32, device=self.model_runner.device)
            )
            self.attn_backends[i].forward_metadata = self.attn_backends[
                i
            ]._compute_metadata(
                forward_batch,
                seq_lens_override=seq_lens_i,
                seq_lens_cpu_override=seq_lens_cpu_i,
                positions_override=positions_i,
                num_tokens_per_bs_override=1,
            )
