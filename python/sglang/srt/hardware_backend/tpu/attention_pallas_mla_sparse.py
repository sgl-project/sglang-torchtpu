# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA backend for the TPU torch_tpu platform.

Selected via ``--attention-backend pallas_tpu_mla_sparse``. It runs the sparse
MLA kernels (``sglang.srt.layers.attention.pallas.mla_sparse``) in place of the
dense MLA v2 kernel, and is otherwise ``pallas_tpu_mla``: the same absorbed
call form, per-forward metadata and fused latent pool.

Two things are not there yet, and both bound what this backend serves today.

* No indexer runs on TPU, so the top-k rows are built in the op: every causal
  position. The result is dense attention, exact only while the context is at
  most ``index_topk``, which the constructor enforces.
* No pool allocates the split cache the kernels read, so the op derives it
  from the fused FP8 buffer and merges it back, copying the pool each step
  (see ``mla_sparse_fused``).

TODO(sglang-torchtpu#1072): take the indexer's rows, then move to a split pool.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import dsa_context_fits_index_topk
from sglang.srt.hardware_backend.tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
)
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_PALLAS_SPARSE_MLA_OP = None


def _build_pallas_sparse_mla_kernel_op():
    """Lazily register the fused-cache sparse MLA step as a torch custom op.

    The op takes ``pallas_tpu_mla``'s arguments plus ``topk``, so the dense
    backend's forward drives it unchanged. ``cu_kv_lens`` and ``v_scale`` are
    accepted and unused: the sparse kernels address pages by a fixed stride
    and dequantize keys and values with one scale.
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
        sm_scale: float = 1.0,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        topk: int = 0,
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
            sm_scale=sm_scale,
            k_scale=k_scale,
            topk=topk,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(10, 11, 12, 13),
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
        sm_scale: float = 1.0,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        topk: int = 0,
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
                topk,
            )
        return output

    def _fake_impl(ql_nope: torch.Tensor, *args, **kwargs):
        return torch.empty_like(ql_nope)

    custom_op.register_fake(_fake_impl)

    _PALLAS_SPARSE_MLA_OP = custom_op
    return custom_op


class TpuPallasSparseMlaAttnBackend(TpuPallasMlaAttnBackend):
    """``pallas_tpu_mla`` with the sparse MLA kernels behind the same forward."""

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        model_config = model_runner.model_config
        if not dsa_context_fits_index_topk(
            model_config.hf_config, model_config.context_len
        ):
            raise ValueError(
                "--attention-backend pallas_tpu_mla_sparse builds its own top-k "
                "rows until the indexer runs on TPU (sglang-torchtpu#1072), so it "
                "serves GlmMoeDsa checkpoints with --context-length at most "
                f"index_topk; got context length {model_config.context_len}."
            )
        if model_runner.kv_cache_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "--attention-backend pallas_tpu_mla_sparse reads an FP8 latent "
                "cache only; pass --kv-cache-dtype fp8_e4m3 (or leave it unset), "
                f"got {model_runner.kv_cache_dtype}."
            )
        # Read by _build_kernel_op, which the base constructor calls.
        self._topk = model_config.hf_config.index_topk
        super().__init__(model_runner, speculative_step_id)

    def _build_kernel_op(self):
        op = _build_pallas_sparse_mla_kernel_op()
        topk = self._topk

        def invoke(*args):
            return op(*args, topk)

        return invoke
