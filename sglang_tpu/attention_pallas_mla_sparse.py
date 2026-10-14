# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention backend for TPU (``--attention-backend pallas_tpu_mla_sparse``).

This backend serves models with DeepSeek Sparse Attention (DSA). Each layer
attends only over the keys the DSA indexer picked for it:

* The indexer runs here too. One op per indexer layer stores that layer's
  index keys and returns the top-k rows (see ``dsa_indexer_tpu``).
* The attention op takes those rows and runs the sparse MLA kernels
  (``sglang_tpu.attention.pallas.mla_sparse``).

Everything else is inherited from ``pallas_tpu_mla``: how the model calls the
backend, and the metadata built for each forward pass.

The KV cache is a ``TpuSplitMLATokenToKVPool``: two buffers per layer, in the
layout the sparse kernels read.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import get_dsa_index_topk
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang_tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
)
from sglang_tpu.dsa_indexer_tpu import build_dsa_index_topk_op, indexer_forward
from sglang_tpu.pallas_export import (
    argnums,
    tpu_info_for_cpu_export,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

_PALLAS_SPARSE_MLA_OP = None


def _build_pallas_sparse_mla_kernel_op():
    """Register the sparse MLA step as a torch custom op, once per process.

    The op takes the same arguments as the ``pallas_tpu_mla`` op, with the two
    cache buffers in place of the fused one, followed by the indexer's top-k
    rows. ``cu_kv_lens`` and ``v_scale`` are accepted but not used: the sparse
    kernels find pages with a fixed stride, and they use one scale for both
    keys and values.
    """
    global _PALLAS_SPARSE_MLA_OP
    if _PALLAS_SPARSE_MLA_OP is not None:
        return _PALLAS_SPARSE_MLA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.mla_sparse_fused import (
        sparse_mla_attention_fused,
    )

    op_name = "sglang_tpu_pallas::sparse_mla_attention_fused"

    # custom_jax_kernel reads this function's signature: every tensor has to
    # be a positional parameter annotated ``jax.Array``. The JAX step cannot be
    # passed in directly, because it has keyword-only arguments and does not
    # take cu_kv_lens or v_scale. This wrapper adapts one to the other.
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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return sparse_mla_attention_fused(
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
        )

    nope_argnum, rope_argnum = argnums(_wrapped, "cache_nope", "cache_rope")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "sm_scale", "k_scale", "v_scale"),
        # The kernel updates both caches in place: its outputs 1 and 2 are the
        # same buffers as the cache_nope and cache_rope inputs. This only works
        # when each cache is a whole tensor, not a view of a larger one.
        donate_argnums=(nope_argnum, rope_argnum),
        input_output_aliases={nope_argnum: 1, rope_argnum: 2},
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
    ) -> torch.Tensor:
        # The first call with a new shape compiles the kernel. In this process
        # JAX has no TPU device of its own (torch_tpu owns the device), so
        # Pallas cannot find out which TPU it is compiling for. This context
        # gives it the answer.
        with tpu_info_for_cpu_export():
            output, _updated_nope, _updated_rope = jax_kernel(
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
            )
        return output

    # Shape-only version for torch.compile tracing: same shape as the queries.
    def _fake_impl(ql_nope: torch.Tensor, *args, **kwargs):
        return torch.empty_like(ql_nope)

    custom_op.register_fake(_fake_impl)

    _PALLAS_SPARSE_MLA_OP = custom_op
    return custom_op


class TpuPallasSparseMlaAttnBackend(TpuPallasMlaAttnBackend):
    """``pallas_tpu_mla`` with sparse attention: it runs the DSA indexer too."""

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        super().__init__(model_runner, speculative_step_id)
        self._topk = get_dsa_index_topk(model_runner.model_config.hf_config)
        self._index_topk_op = build_dsa_index_topk_op()

    def _build_kernel_op(self):
        # The base class constructor calls this to set self._op.
        return _build_pallas_sparse_mla_kernel_op()

    def forward_indexer(
        self,
        indexer: "Indexer",
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Run one layer's indexer. ``Indexer.forward_tpu`` calls this."""
        return indexer_forward(
            indexer,
            backend=self,
            x=x,
            q_lora=q_lora,
            positions=positions,
            layer_id=layer_id,
        )

    def index_topk(
        self,
        *,
        layer_id: int,
        query: torch.Tensor,  # [num_tokens, index_n_heads, index_head_dim]
        key: torch.Tensor,  # [num_tokens, index_head_dim]
        head_weights: torch.Tensor,  # [num_tokens, index_n_heads]
        weight_scale: float,
    ) -> torch.Tensor:
        """Run one layer's indexer: store its index keys, return its top-k rows.

        Returns ``i32[num_tokens, index_topk]``. Each row lists the chosen key
        positions, counted from the start of the token's own sequence. The
        chosen positions come first and the rest of the row is ``-1``.
        """
        metadata = self.forward_metadata
        # The torch-to-JAX bridge passes raw buffers, so a tensor with strides
        # would be read wrongly. contiguous() makes each input dense.
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
        # Every layer gets rows: from its own indexer, or from the layer before.
        assert topk_indices is not None, (
            f"layer {layer.layer_id} reached pallas_tpu_mla_sparse without the "
            "indexer's top-k rows."
        )
        args = self._mla_kernel_args(
            q=q, k=k, layer=layer, q_rope=q_rope, k_rope=k_rope
        )
        out = self._op(*args, topk_indices)
        # TODO(sglang-torchtpu#1072): avoid this reshape, as in pallas_tpu_mla.
        return out.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _kv_cache_buffers(self, layer_id: int) -> tuple:
        # This backend's pool has two buffers per layer where the fused pool
        # has one. The op takes them in this order.
        pool = self.token_to_kv_pool
        return (pool.get_nope_cache(layer_id), pool.get_rope_cache(layer_id))
