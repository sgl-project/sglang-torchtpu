# SPDX-License-Identifier: Apache-2.0
"""Batched Pallas ragged-paged-attention backend for the TPU torch_tpu platform.

The kernel reads either of two KV cache layouts, chosen once here from
``SGLANG_TPU_USE_SEQ_ALONG_LANE_KV``:

``HEAD_ALONG_SUBLANE`` (default)
    Token-major fused 5D buffer, ``num_kv_heads * 2`` padded up to
    ``kv_packing``.
``SEQ_ALONG_LANE``
    Token index on the lane axis and ``num_kv_heads * 2`` unpadded, so at one
    KV head per rank an fp8 cache is 256 bytes per token instead of 512. That
    halving is what lets long context and large batches fit; it needs
    ``TpuSeqAlongLaneKVPool`` (selected by the platform from the same flag) and
    a page size of at least 256.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import torch

from sglang_tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    PallasForwardMetadata,
    RpaKernelSpec,
)
from sglang_tpu.cp_forward import get_tpu_cp_layout
from sglang_tpu.envs import envs

__all__ = [
    "PallasForwardMetadata",
    "TpuPallasRpaBatchedAttnBackend",
    "kv_layout",
    "uses_seq_along_lane_kv",
]


def uses_seq_along_lane_kv() -> bool:
    """Whether the KV cache is in the seq-along-lane layout.

    The kernel, the pool that allocates the buffer and the page-size floor all
    have to agree, so they all ask here instead of reading the flag separately.
    """
    return envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.get()


def kv_layout():
    """The layout as the kernel's own enum.

    Read at trace time, not per token, so a plain lookup is fine here.
    """
    from sglang_tpu.attention.pallas.batched_rpa import configs

    if uses_seq_along_lane_kv():
        return configs.KVLayout.SEQ_ALONG_LANE
    return configs.KVLayout.HEAD_ALONG_SUBLANE


def _import_batched_kernel():
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
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
    decode_query_size=1,
    use_causal_mask=True,
):
    if not use_causal_mask:
        raise NotImplementedError(
            "pallas_tpu_rpa_batched supports only causal attention; "
            "use pallas_tpu_rpa_v3."
        )
    # cu_kv_lens is not consumed by the batched kernel.
    # It builds its schedule inline and has no per-mode dispatch to skip, so
    # the argument the shared shell passes for kernels that do is
    # discarded here. The base never widens decode for this kernel
    # (``supports_decode_query_size`` is off), so the width is always 1.
    del cu_kv_lens, decode_query_size
    return jax_rpa(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink=attention_sink,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        use_causal_mask=True,
        # The kernel owns the KV write; the backend never writes out of band.
        skip_kv_update=False,
        kv_layout=kv_layout(),
    )


# SEQ_ALONG_LANE tiles the page axis onto lanes, so a page has to be a multiple
# of the 128-lane width, and 256 is what the benchmarks run at. The
# token-major layout has no such constraint but keeps the same floor, because a
# smaller page turns each KV fetch into many tiny DMAs.
_MIN_PAGE_SIZE = 256

RPA_BATCHED_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_batched",
    import_kernel=_import_batched_kernel,
    invoke_kernel=_invoke_batched_kernel,
    supports_attention_sink=True,
    uses_cu_kv_lens=False,
    is_batched=True,
    min_page_size=_MIN_PAGE_SIZE,
)


class TpuPallasRpaBatchedAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Batched RPA attention backend (``--attention-backend pallas_tpu_rpa_batched``)."""

    kernel_spec = RPA_BATCHED_SPEC

    def __init__(self, model_runner, speculative_step_id=0):
        super().__init__(model_runner, speculative_step_id)
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        self._pp_schedule_runner = model_runner if parallel.pp_size > 1 else None
        self._cp_schedule_runner = model_runner if parallel.attn_cp_size > 1 else None

        self._cp_op = None
        from sglang_tpu.batched_rpa_cp import PcpBatchedMetadata

        self.pcp_forward_metadata = PcpBatchedMetadata.create_empty(self.device)
        if parallel.attn_cp_size > 1:
            from sglang.srt.layers.cp.zigzag import set_zigzag_tpu_layout
            from sglang_tpu.batched_rpa_cp import (
                build_batched_rpa_cp_op,
                build_batched_rpa_pcp_ring_op,
                build_cp_all_gather_op,
            )
            from sglang_tpu.envs import envs as tpu_envs

            self._cp_op = build_batched_rpa_cp_op()
            if tpu_envs.SGLANG_TPU_CP_PAGE_LAYOUT.get():
                replicated = tpu_envs.SGLANG_TPU_PCP_REPLICATED_PREFILL.get()
                pool_align = tpu_envs.SGLANG_TPU_PCP_POOL_ALIGN.get()
                set_zigzag_tpu_layout(
                    page_align=self.page_size,
                    balance_pairs=True,
                    pool_align=replicated and pool_align,
                )
                if replicated:
                    from sglang_tpu.kv_row_write import ensure_kv_row_ops

                    ensure_kv_row_ops()
                else:
                    build_batched_rpa_pcp_ring_op(
                        parallel.attn_cp_size, parallel.attn_tp_size
                    )
                build_cp_all_gather_op(parallel.attn_cp_size, parallel.attn_tp_size)

    def init_forward_metadata(self, forward_batch):
        cp_layout = get_tpu_cp_layout(forward_batch)
        if cp_layout is not None:
            from sglang_tpu.pp_attention_capacity import (
                check_cp_schedule,
            )

            # The kernel runs the context-parallel rows, not the global lengths.
            check_cp_schedule(self._cp_schedule_runner, self, cp_layout.host)
        elif self._pp_schedule_runner is not None:
            from sglang_tpu.pp_attention_capacity import (
                check_batch_schedule,
            )

            check_batch_schedule(self._pp_schedule_runner, self, forward_batch)
        super().init_forward_metadata(forward_batch)

        from sglang.srt.layers.cp.utils import is_cp_active
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_cp_size > 1 and is_cp_active(forward_batch):
            from sglang_tpu.batched_rpa_cp import (
                build_pcp_batched_metadata,
            )

            self.pcp_forward_metadata = build_pcp_batched_metadata(
                seq_lens=self.forward_metadata.seq_lens,
                prefix_lens=self.forward_metadata.prefix_lens,
                page_indices=self.forward_metadata.page_indices,
                page_size=self.page_size,
                cp_size=get_parallel().attn_cp_size,
                cp_rank=get_parallel().attn_cp_rank,
                device=self.device,
                forward_batch=forward_batch,
            )
        else:
            self.pcp_forward_metadata = None

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.cp.utils import is_cp_active
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_cp_size > 1 and is_cp_active(forward_batch):
            return self._forward_cp_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                pcp_meta=self.pcp_forward_metadata,
                **kwargs,
            )
        return super().forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def _forward_cp_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        pcp_meta=None,
        **kwargs,
    ) -> torch.Tensor:
        from sglang_tpu.batched_rpa_cp import batched_rpa_pcp_forward

        num_tokens = q.shape[0]
        q = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(num_tokens, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(num_tokens, layer.tp_v_head_num, layer.v_head_dim)
        if layer.v_head_dim < layer.qk_head_dim:
            v = torch.nn.functional.pad(v, (0, layer.qk_head_dim - layer.v_head_dim))
        kv_cache = self.token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

        if self.kv_cache_quant_dtype is not None:
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
            from sglang_tpu.attention_pallas_rpa_base import _quantize_kv_to_fp8

            fp8_dtype = self.kv_cache_quant_dtype
            quantize_kv = lambda k_, v_: (
                _quantize_kv_to_fp8(k_, k_scale, fp8_dtype),
                _quantize_kv_to_fp8(v_, v_scale, fp8_dtype),
            )
        else:
            k_scale = None
            v_scale = None
            quantize_kv = None

        pcp_meta = (
            pcp_meta
            or self.pcp_forward_metadata
            or getattr(self.forward_metadata, "pcp_metadata", None)
        )
        out = batched_rpa_pcp_forward(
            self._cp_op,
            q,
            k,
            v,
            kv_cache,
            layer,
            forward_batch,
            pcp_meta,
            self.forward_metadata,
            k_scale=k_scale,
            v_scale=v_scale,
            sinks=kwargs.get("sinks"),
            quantize_kv=quantize_kv,
        )
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)[..., : layer.v_head_dim]
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
