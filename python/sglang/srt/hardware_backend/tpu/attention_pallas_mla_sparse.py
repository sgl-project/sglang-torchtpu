# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA backend for the TPU torch_tpu platform.

Selected via ``--attention-backend pallas_tpu_mla_sparse``. It runs the sparse
MLA kernels (``sglang.srt.layers.attention.pallas.mla_sparse``) in place of the
dense MLA v2 kernel, and is otherwise ``pallas_tpu_mla``: the same absorbed
call form, per-forward metadata and fused latent pool.

The top-k rows come from the DSA indexer, which this backend also runs: one op
per indexer layer stores that layer's index keys and selects (see
``dsa_indexer_tpu``).

Two things bound what it serves today.

* The context is capped at ``index_topk``. Nothing here depends on that; a
  longer context is simply not validated yet.
* No pool allocates the split cache the kernels read, so the op derives it
  from the fused FP8 buffer and merges it back, copying the pool each step
  (see ``mla_sparse_fused``).

TODO(sglang-torchtpu#1072): lift the context cap, then move to a split pool.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    dsa_context_fits_index_topk,
    is_glm_moe_dsa,
)
from sglang.srt.hardware_backend.tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
)
from sglang.srt.hardware_backend.tpu.dsa_indexer_tpu import build_dsa_index_topk_op
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

_PALLAS_SPARSE_MLA_OP = None


def _build_pallas_sparse_mla_kernel_op():
    """Lazily register the fused-cache sparse MLA step as a torch custom op.

    The op takes ``pallas_tpu_mla``'s arguments and then the indexer's top-k
    rows. ``cu_kv_lens`` and ``v_scale`` are accepted and unused: the sparse
    kernels address pages by a fixed stride and dequantize keys and values
    with one scale.
    """
    global _PALLAS_SPARSE_MLA_OP
    if _PALLAS_SPARSE_MLA_OP is not None:
        return _PALLAS_SPARSE_MLA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.attention.pallas.mla_sparse_fused import (
        sparse_mla_attention_fused,
    )

    op_name = "sglang_tpu_pallas::sparse_mla_attention_fused"

    def _wrapped(
        ql_nope: jax.Array,
        q_pe: jax.Array,
        new_kv_c: jax.Array,
        new_k_pe: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float,
        k_scale: Optional[float],
        v_scale: Optional[float],
        topk_indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return sparse_mla_attention_fused(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            topk_indices,
            sm_scale=sm_scale,
            k_scale=k_scale,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(10, 11, 12),
        donate_argnums=(4,),
        input_output_aliases={4: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("kv_cache_fused",),
        device_types=["tpu"],
    )
    def custom_op(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        sm_scale: float,
        k_scale: Optional[float],
        v_scale: Optional[float],
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        # JAX is CPU-only here, so Mosaic needs the TPU generation patched in.
        with tpu_info_for_cpu_export():
            output, _updated_kv = jax_kernel(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                sm_scale,
                k_scale,
                v_scale,
                topk_indices,
            )
        return output

    def _fake_impl(ql_nope: torch.Tensor, *args, **kwargs):
        return torch.empty_like(ql_nope)

    custom_op.register_fake(_fake_impl)

    _PALLAS_SPARSE_MLA_OP = custom_op
    return custom_op


class TpuPallasSparseMlaAttnBackend(TpuPallasMlaAttnBackend):
    """``pallas_tpu_mla`` with the sparse MLA kernels, and the DSA indexer's host."""

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        model_config = model_runner.model_config
        hf_config = model_config.hf_config
        if not is_glm_moe_dsa(hf_config):
            # TODO(sglang-torchtpu#1072): other DSA checkpoints are not validated.
            raise ValueError(
                "--attention-backend pallas_tpu_mla_sparse serves GlmMoeDsa "
                f"checkpoints only, got {hf_config.architectures}."
            )
        if not dsa_context_fits_index_topk(hf_config, model_config.context_len):
            # TODO(sglang-torchtpu#1072): a longer context is not validated yet.
            raise ValueError(
                "--attention-backend pallas_tpu_mla_sparse serves a "
                "--context-length of at most index_topk for now; got context "
                f"length {model_config.context_len}."
            )
        if model_runner.kv_cache_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "--attention-backend pallas_tpu_mla_sparse reads an FP8 latent "
                "cache only; pass --kv-cache-dtype fp8_e4m3 (or leave it unset), "
                f"got {model_runner.kv_cache_dtype}."
            )
        self._topk = hf_config.index_topk
        super().__init__(model_runner, speculative_step_id)
        self._index_topk_op = build_dsa_index_topk_op()

    def _build_kernel_op(self):
        return _build_pallas_sparse_mla_kernel_op()

    def index_topk(
        self,
        *,
        layer_id: int,
        query: torch.Tensor,  # [num_tokens, index_n_heads, index_head_dim]
        key: torch.Tensor,  # [num_tokens, index_head_dim]
        head_weights: torch.Tensor,  # [num_tokens, index_n_heads]
        weight_scale: float,
    ) -> torch.Tensor:
        """Store one layer's index keys and return its top-k rows.

        ``i32[num_tokens, index_topk]``: positions within the token's own
        sequence, selected ones first, then ``-1``.
        """
        metadata = self.forward_metadata
        # The torch->JAX bridge hands over raw buffers, so every input must be dense.
        return self._index_topk_op(
            query.contiguous(),
            key.contiguous(),
            head_weights.contiguous(),
            self.token_to_kv_pool.get_index_key_cache(layer_id),
            metadata.kv_lens,
            metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
            weight_scale,
            self._topk,
        )

    def _forward_mla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Every layer has rows: its own indexer's, or the previous layer's.
        assert topk_indices is not None, (
            f"layer {layer.layer_id} reached pallas_tpu_mla_sparse without the "
            "indexer's top-k rows."
        )
        args = self._mla_kernel_args(
            q=q, k=k, layer=layer, q_rope=q_rope, k_rope=k_rope
        )
        out = self._op(*args, topk_indices)
        return out.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
