# SPDX-License-Identifier: Apache-2.0
"""Pallas v3 ragged-paged-attention backend for the TPU torch_tpu platform.

Thin wrapper over :class:`BaseTpuPallasRpaAttnBackend` (see
``attention_pallas_rpa_base``) selecting the JAX/Pallas v3 kernel
(``sglang.srt.layers.attention.pallas.ragged_paged_attention_v3``). Selected via
``--attention-backend pallas_tpu_rpa_v3`` (the TPU platform default). Supports
attention sinks, custom masks, and EAGLE speculative decoding (TARGET_VERIFY,
DRAFT_EXTEND_V2, multi-step draft decode).
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from typing import Optional

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    BaseTpuPallasRpaMultiStepDraftBackend,
    PallasForwardMetadata,
    RpaKernelSpec,
)

__all__ = [
    "PallasForwardMetadata",
    "TpuPallasRpaV3AttnBackend",
    "TpuPallasRpaV3MultiStepDraftBackend",
]


def _import_v3_kernel():
    from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def _invoke_v3_kernel(
    jax_rpa,
    *,
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    attention_sink,
    sm_scale,
    sliding_window,
    soft_cap,
    k_scale,
    v_scale,
):
    return jax_rpa(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        None,  # custom_mask
        attention_sink,
        causal=1,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
    )


RPA_V3_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention",
    import_kernel=_import_v3_kernel,
    invoke_kernel=_invoke_v3_kernel,
    supports_attention_sink=True,
    extend_distribution_mask=(0, 1, 1),  # [0, n, n]
    uses_draft_extend_v2=True,
    min_page_size=None,
)


class TpuPallasRpaV3AttnBackend(BaseTpuPallasRpaAttnBackend):
    """RPA v3 attention backend (``--attention-backend pallas_tpu_rpa_v3``)."""

    kernel_spec = RPA_V3_SPEC

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int] = None
    ):
        # RPA-v3 rebuilds verify metadata and has no graph-capture buffers.
        pass


class TpuPallasRpaV3MultiStepDraftBackend(BaseTpuPallasRpaMultiStepDraftBackend):
    """One RPA-v3 backend per EAGLE draft decode step."""

    attn_backend_cls = TpuPallasRpaV3AttnBackend
