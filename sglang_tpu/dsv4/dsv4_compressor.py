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
# Source: src/vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_compressor.py
"""TPU DeepSeek-V4 KV/score compressor custom ops and backend mixin.

IMPORTANT: Do NOT add ``from __future__ import annotations`` to this module.
``torch_tpu._internal.pallas.custom_jax_kernel`` inspects function parameter
annotations at registration time with ``param.annotation is jax.Array``, which
requires evaluated type objects rather than stringified annotations.
"""

from typing import Optional

import torch

from sglang.srt.runtime_context import get_exec
from sglang_tpu.dsv4.dsv4_memory_pool import (
    DSV4TPUTokenToKVPool,
    TPUCompressStatePool,
    get_or_build_cos_sin_cache,
)
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_CSA_OP_NAME = "sglang_tpu_pallas::dsv4_compressor_csa"
_HCA_OP_NAME = "sglang_tpu_pallas::dsv4_compressor_hca"
_INDEXER_OP_NAME = "sglang_tpu_pallas::dsv4_compressor_indexer"

_csa_custom_op = None
_hca_custom_op = None
_indexer_custom_op = None


def ensure_tpu_compressor_caches(compressor) -> None:
    """Lazily prepare and cache transposed ``wkv_gate`` weight, ``float32``
    ``norm.weight``, ``float32`` ``ape``, and ``float32`` ``cos_sin_cache`` on
    ``compressor`` so no transpose or dtype conversion runs on the hot path."""
    if hasattr(compressor, "_tpu_caches_ready") and compressor._tpu_caches_ready:
        return
    if compressor.is_in_indexer and get_exec().kernel.enable_deepseek_v4_fp4_indexer:
        raise NotImplementedError(
            "DeepSeek-V4 on TPU does not support FP4 indexer KV cache; "
            "the TPU compressor kernel emits the FP8/UE8M0 cache layout."
        )
    w = compressor.wkv_gate.weight.data
    wkv_wgate = w.t().contiguous()
    ape = compressor.ape.data.to(torch.float32).contiguous()
    norm_weight = compressor.norm.weight.data.to(torch.float32).contiguous()
    cos_sin_cache = (
        get_or_build_cos_sin_cache(compressor.freqs_cis)
        if compressor.freqs_cis.is_complex()
        else compressor.freqs_cis
    )
    if isinstance(compressor, torch.nn.Module):
        compressor.register_buffer("_tpu_wkv_wgate", wkv_wgate, persistent=False)
        compressor.register_buffer("_tpu_ape", ape, persistent=False)
        compressor.register_buffer("_tpu_norm_weight", norm_weight, persistent=False)
        compressor.register_buffer(
            "_tpu_cos_sin_cache", cos_sin_cache, persistent=False
        )
    else:
        compressor._tpu_wkv_wgate = wkv_wgate
        compressor._tpu_ape = ape
        compressor._tpu_norm_weight = norm_weight
        compressor._tpu_cos_sin_cache = cos_sin_cache
    compressor._tpu_rms_eps = float(compressor.norm.variance_epsilon)
    compressor._tpu_caches_ready = True


def _register_csa_compressor_op():
    global _csa_custom_op
    if _csa_custom_op is not None:
        return _csa_custom_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.dsv4.compress_and_store.compressor_v1 import (
        compressor_forward,
    )

    _exported_shapes = set()

    def _wrapped_csa(
        hidden_states: jax.Array,
        wkv_wgate: jax.Array,
        ape: jax.Array,
        norm_weight: jax.Array,
        cos_sin_cache: jax.Array,
        positions: jax.Array,
        state_block_tables: jax.Array,
        query_start_loc: jax.Array,
        k_block_tables: jax.Array,
        request_distribution: jax.Array,
        cache: jax.Array,
        rope_cache: jax.Array,
        state_cache: jax.Array,
        state_block_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_eps: float,
        quant_block: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            new_cache, new_rope_cache, new_state_cache = compressor_forward(
                hidden_states=hidden_states,
                wkv_wgate=wkv_wgate,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                block_table=state_block_tables,
                query_start_loc=query_start_loc,
                kv_block_table=k_block_tables,
                cache=cache,
                rope_cache=rope_cache,
                distribution=request_distribution,
                state_block_size=int(state_block_size),
                head_dim=int(head_dim),
                compress_ratio=int(compress_ratio),
                overlap=bool(overlap),
                rms_eps=float(rms_eps),
                quant_block=int(quant_block),
                state_cache=state_cache,
            )
        assert new_rope_cache is not None
        return new_cache, new_rope_cache, new_state_cache

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped_csa,
        name="jax_dsv4_compressor_csa",
        static_argnums=(13, 14, 15, 16, 17, 18),
        donate_argnums=(10, 11, 12),
        input_output_aliases={10: 0, 11: 1, 12: 2},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _CSA_OP_NAME,
        mutates_args=("cache", "rope_cache", "state_cache"),
        device_types=["tpu"],
    )
    def custom_op(
        hidden_states: torch.Tensor,
        wkv_wgate: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        state_block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        k_block_tables: torch.Tensor,
        request_distribution: torch.Tensor,
        cache: torch.Tensor,
        rope_cache: torch.Tensor,
        state_cache: torch.Tensor,
        state_block_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_eps: float,
        quant_block: int,
    ) -> None:
        key = (
            hidden_states.shape[0],
            state_block_tables.shape[0],
            k_block_tables.shape[0],
            state_block_size,
            head_dim,
            compress_ratio,
            overlap,
            rms_eps,
            quant_block,
        )
        if key in _exported_shapes:
            jax_kernel(
                hidden_states,
                wkv_wgate,
                ape,
                norm_weight,
                cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                rope_cache,
                state_cache,
                state_block_size,
                head_dim,
                compress_ratio,
                overlap,
                rms_eps,
                quant_block,
            )
        else:
            with (
                tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                jax_kernel(
                    hidden_states,
                    wkv_wgate,
                    ape,
                    norm_weight,
                    cos_sin_cache,
                    positions,
                    state_block_tables,
                    query_start_loc,
                    k_block_tables,
                    request_distribution,
                    cache,
                    rope_cache,
                    state_cache,
                    state_block_size,
                    head_dim,
                    compress_ratio,
                    overlap,
                    rms_eps,
                    quant_block,
                )
            _exported_shapes.add(key)

    @custom_op.register_fake
    def _(
        hidden_states: torch.Tensor,
        wkv_wgate: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        state_block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        k_block_tables: torch.Tensor,
        request_distribution: torch.Tensor,
        cache: torch.Tensor,
        rope_cache: torch.Tensor,
        state_cache: torch.Tensor,
        state_block_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_eps: float,
        quant_block: int,
    ) -> None:
        return None

    _csa_custom_op = custom_op
    return _csa_custom_op


def _make_two_buffer_compressor_op(op_name: str, jax_name: str):
    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.dsv4.compress_and_store.compressor_v1 import (
        compressor_forward,
    )

    _exported_shapes = set()

    def _wrapped(
        hidden_states: jax.Array,
        wkv_wgate: jax.Array,
        ape: jax.Array,
        norm_weight: jax.Array,
        cos_sin_cache: jax.Array,
        positions: jax.Array,
        state_block_tables: jax.Array,
        query_start_loc: jax.Array,
        k_block_tables: jax.Array,
        request_distribution: jax.Array,
        cache: jax.Array,
        state_cache: jax.Array,
        state_block_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_eps: float,
        quant_block: int,
    ) -> tuple[jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            new_cache, _, new_state_cache = compressor_forward(
                hidden_states=hidden_states,
                wkv_wgate=wkv_wgate,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                block_table=state_block_tables,
                query_start_loc=query_start_loc,
                kv_block_table=k_block_tables,
                cache=cache,
                rope_cache=None,
                distribution=request_distribution,
                state_block_size=int(state_block_size),
                head_dim=int(head_dim),
                compress_ratio=int(compress_ratio),
                overlap=bool(overlap),
                rms_eps=float(rms_eps),
                quant_block=int(quant_block),
                state_cache=state_cache,
            )
        return new_cache, new_state_cache

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=jax_name,
        static_argnums=(12, 13, 14, 15, 16, 17),
        donate_argnums=(10, 11),
        input_output_aliases={10: 0, 11: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("cache", "state_cache"),
        device_types=["tpu"],
    )
    def custom_op(
        hidden_states: torch.Tensor,
        wkv_wgate: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        state_block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        k_block_tables: torch.Tensor,
        request_distribution: torch.Tensor,
        cache: torch.Tensor,
        state_cache: torch.Tensor,
        state_block_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_eps: float,
        quant_block: int,
    ) -> None:
        key = (
            hidden_states.shape[0],
            state_block_tables.shape[0],
            k_block_tables.shape[0],
            state_block_size,
            head_dim,
            compress_ratio,
            overlap,
            rms_eps,
            quant_block,
        )
        if key in _exported_shapes:
            jax_kernel(
                hidden_states,
                wkv_wgate,
                ape,
                norm_weight,
                cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                state_cache,
                state_block_size,
                head_dim,
                compress_ratio,
                overlap,
                rms_eps,
                quant_block,
            )
        else:
            with (
                tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                jax_kernel(
                    hidden_states,
                    wkv_wgate,
                    ape,
                    norm_weight,
                    cos_sin_cache,
                    positions,
                    state_block_tables,
                    query_start_loc,
                    k_block_tables,
                    request_distribution,
                    cache,
                    state_cache,
                    state_block_size,
                    head_dim,
                    compress_ratio,
                    overlap,
                    rms_eps,
                    quant_block,
                )
            _exported_shapes.add(key)

    @custom_op.register_fake
    def _(
        hidden_states: torch.Tensor,
        wkv_wgate: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        state_block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        k_block_tables: torch.Tensor,
        request_distribution: torch.Tensor,
        cache: torch.Tensor,
        state_cache: torch.Tensor,
        state_block_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool,
        rms_eps: float,
        quant_block: int,
    ) -> None:
        return None

    return custom_op


def _register_hca_compressor_op():
    global _hca_custom_op
    if _hca_custom_op is None:
        _hca_custom_op = _make_two_buffer_compressor_op(
            _HCA_OP_NAME, "jax_dsv4_compressor_hca"
        )
    return _hca_custom_op


def _register_indexer_compressor_op():
    global _indexer_custom_op
    if _indexer_custom_op is None:
        _indexer_custom_op = _make_two_buffer_compressor_op(
            _INDEXER_OP_NAME, "jax_dsv4_compressor_indexer"
        )
    return _indexer_custom_op


class CompressorTPUBackendMixin:
    """Mixin for :class:`DSV4TPUAttnBackend` providing fused Pallas compressor
    dispatch for CSA (4x), HCA (128x), and CSA Lightning Indexer (4x)."""

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch,
        layer_id: int,
        compressor,
    ) -> None:
        del layer_id
        if forward_batch.forward_mode.is_idle():
            return
        compressor(x, forward_batch, attn_backend=self)

    forward_indexer_compressor = forward_core_compressor

    def forward_compress(
        self,
        compressor,
        x: torch.Tensor,
        forward_batch,
    ) -> Optional[torch.Tensor]:
        if forward_batch.forward_mode.is_idle() or x.shape[0] == 0:
            return None

        ensure_tpu_compressor_caches(compressor)

        pool: DSV4TPUTokenToKVPool = self.token_to_kv_pool
        fm = self.forward_metadata
        layer_id: int = compressor.layer_id
        ratio: int = compressor.ratio
        head_dim: int = compressor.head_dim
        overlap: bool = bool(compressor.overlap)

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        positions = fm.positions
        if positions is None:
            positions = forward_batch.positions.to(torch.int32)
        if positions.shape[0] != x_2d.shape[0]:
            positions = positions[: x_2d.shape[0]]

        query_start_loc = fm.query_start_loc
        request_distribution = fm.distribution

        # Compressed KV page table (1D flattened `[bs * max_pages_per_req]`).
        k_block_tables = (
            fm.c4_block_tables_flat if ratio == 4 else fm.c128_block_tables_flat
        )
        assert k_block_tables is not None, (
            f"Missing compressed KV block_tables in forward_metadata for ratio={ratio}."
        )

        # Compact `[bs * 6]` state block table (1D flattened).
        state_block_tables = (
            fm.c4_state_block_tables_flat
            if ratio == 4
            else fm.c128_state_block_tables_flat
        )
        assert state_block_tables is not None, (
            f"Missing c{ratio}_state_block_tables in forward_metadata."
        )

        state_pool = (
            pool.get_indexer_compress_states(layer_id)
            if compressor.is_in_indexer
            else pool.get_attention_compress_states(layer_id)
        )
        assert isinstance(state_pool, TPUCompressStatePool)
        state_cache = state_pool.state_buffer
        state_block_size = int(state_pool.state_block_size)

        if compressor.is_in_indexer:
            cache = pool.get_indexer_k_buffer(layer_id)
            op = _register_indexer_compressor_op()
            op(
                x_2d,
                compressor._tpu_wkv_wgate,
                compressor._tpu_ape,
                compressor._tpu_norm_weight,
                compressor._tpu_cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                state_cache,
                state_block_size,
                head_dim,
                ratio,
                overlap,
                compressor._tpu_rms_eps,
                128,  # quant_block for Mode.CSA_INDEXER
            )
        elif ratio == 4:
            cache, rope_cache = pool.get_csa_kv_buffers(layer_id)
            op = _register_csa_compressor_op()
            op(
                x_2d,
                compressor._tpu_wkv_wgate,
                compressor._tpu_ape,
                compressor._tpu_norm_weight,
                compressor._tpu_cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                rope_cache,
                state_cache,
                state_block_size,
                head_dim,
                ratio,
                overlap,
                compressor._tpu_rms_eps,
                64,  # quant_block for Mode.CSA
            )
        elif ratio == 128:
            cache = pool.get_hca_buffer(layer_id)
            op = _register_hca_compressor_op()
            op(
                x_2d,
                compressor._tpu_wkv_wgate,
                compressor._tpu_ape,
                compressor._tpu_norm_weight,
                compressor._tpu_cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                state_cache,
                state_block_size,
                head_dim,
                ratio,
                overlap,
                compressor._tpu_rms_eps,
                0,  # quant_block for Mode.HCA (unquantized bf16)
            )
        else:
            raise ValueError(f"Unsupported DeepSeek-V4 compress_ratio={ratio}")

        return None
