# SPDX-License-Identifier: Apache-2.0
"""Batched Pallas ragged-paged-attention backend for the TPU torch_tpu platform."""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    BaseTpuPallasRpaMultiStepDraftBackend,
    PallasForwardMetadata,
    RpaKernelSpec,
)

__all__ = [
    "PallasForwardMetadata",
    "TpuPallasRpaBatchedAttnBackend",
    "TpuPallasRpaBatchedMultiStepDraftBackend",
]


def _import_batched_kernel():
    from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def _invoke_batched_kernel(
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
    stacked,
):
    # cu_kv_lens and attention_sink are not consumed by the batched kernel.
    del cu_kv_lens, attention_sink
    return jax_rpa(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        use_causal_mask=True,
        update_kv_cache=True,
        stacked=stacked,
    )


RPA_BATCHED_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_batched",
    import_kernel=_import_batched_kernel,
    invoke_kernel=_invoke_batched_kernel,
    supports_attention_sink=False,
    extend_distribution_mask=(0, 0, 1),  # [0, 0, n] (MIXED-only)
    uses_draft_extend_v2=False,
    supports_stacked_decode=True,
    min_page_size=256,
)


class TpuPallasRpaBatchedAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Batched RPA attention backend (``--attention-backend pallas_tpu_rpa_batched``)."""

    kernel_spec = RPA_BATCHED_SPEC


class TpuPallasRpaBatchedMultiStepDraftBackend(BaseTpuPallasRpaMultiStepDraftBackend):
    """One batched-RPA backend per EAGLE draft decode step."""

    attn_backend_cls = TpuPallasRpaBatchedAttnBackend
