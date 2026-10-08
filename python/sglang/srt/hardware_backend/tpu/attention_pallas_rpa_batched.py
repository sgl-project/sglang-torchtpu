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

from sglang.srt.environ import envs
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
    from sglang.srt.layers.attention.pallas.batched_rpa import configs

    if uses_seq_along_lane_kv():
        return configs.KVLayout.SEQ_ALONG_LANE
    return configs.KVLayout.HEAD_ALONG_SUBLANE


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
    decode_query_size=1,
    use_causal_mask=True,
):
    if not use_causal_mask:
        raise NotImplementedError(
            "pallas_tpu_rpa_batched supports only causal attention; "
            "use pallas_tpu_rpa_v3."
        )
    # cu_kv_lens and attention_sink are not consumed by the batched kernel.
    # It builds its schedule inline and has no per-mode dispatch to skip, so
    # the two arguments the shared shell passes for kernels that do are
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
        update_kv_cache=True,
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
    extend_distribution_mask=(0, 0, 1),  # [0, 0, n] (MIXED-only)
    min_page_size=_MIN_PAGE_SIZE,
)


class TpuPallasRpaBatchedAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Batched RPA attention backend (``--attention-backend pallas_tpu_rpa_batched``)."""

    kernel_spec = RPA_BATCHED_SPEC

    def __init__(self, model_runner, speculative_step_id: int = 0) -> None:
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        from sglang.srt.runtime_context import get_parallel

        self._pp_schedule_runner = model_runner if get_parallel().pp_size > 1 else None
        self._cp_op = None
        self.pcp_forward_metadata = None
        self._cp_reqs: set[int] = set()
        if get_parallel().attn_cp_size > 1 or get_parallel().attn_dcp_size > 1:
            from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
                build_batched_rpa_cp_op,
            )

            self._cp_op = build_batched_rpa_cp_op()

    def init_forward_metadata(self, forward_batch):
        if self._pp_schedule_runner is not None:
            from sglang.srt.hardware_backend.tpu.pp_attention_capacity import (
                check_batch_schedule,
            )

            check_batch_schedule(self._pp_schedule_runner, self, forward_batch)
        super().init_forward_metadata(forward_batch)
        from sglang.srt.layers.cp.utils import is_cp_active
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_cp_size > 1 and is_cp_active(forward_batch):
            from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
                build_pcp_batched_metadata,
            )

            cp_size = get_parallel().attn_cp_size
            cp_rank = get_parallel().attn_cp_rank
            prefix_lens = getattr(forward_batch, "prefix_lens", None)
            if prefix_lens is None:
                prefix_lens = torch.zeros_like(forward_batch.seq_lens)
            num_local_tokens = None
            if getattr(forward_batch, "attn_cp_metadata", None) is not None:
                per_rank = forward_batch.attn_cp_metadata.per_rank_actual_token
                if per_rank is not None and len(per_rank) > cp_rank:
                    num_local_tokens = per_rank[cp_rank]
            self.pcp_forward_metadata = build_pcp_batched_metadata(
                seq_lens=forward_batch.seq_lens,
                prefix_lens=prefix_lens,
                page_indices=self.forward_metadata.page_indices,
                page_size=self.page_size,
                cp_size=cp_size,
                cp_rank=cp_rank,
                device=self.device,
                forward_batch=forward_batch,
                num_local_tokens=num_local_tokens,
            )
            req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
            if req_pool_indices is not None:
                for r_idx in req_pool_indices.tolist():
                    self._cp_reqs.add(r_idx)
        else:
            self.pcp_forward_metadata = None
            forward_mode = getattr(forward_batch, "forward_mode", None)
            if forward_mode is not None and forward_mode.is_extend():
                req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
                if req_pool_indices is not None:
                    for r_idx in req_pool_indices.tolist():
                        self._cp_reqs.discard(r_idx)

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
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_cp_size > 1 and self.pcp_forward_metadata is not None:
            return self._forward_cp_extend(q, k, v, layer, forward_batch, **kwargs)
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
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
            batched_rpa_pcp_forward,
        )

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
            from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
                _quantize_kv_to_fp8,
            )

            k = _quantize_kv_to_fp8(k, k_scale, self.kv_cache_quant_dtype)
            v = _quantize_kv_to_fp8(v, v_scale, self.kv_cache_quant_dtype)
        else:
            k_scale = None
            v_scale = None

        out = batched_rpa_pcp_forward(
            self._cp_op,
            q,
            k,
            v,
            kv_cache,
            layer,
            forward_batch,
            self.pcp_forward_metadata,
            self.forward_metadata,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)[..., : layer.v_head_dim]
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def _is_cp_decode(self, forward_batch) -> bool:
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            return False
        reqs = req_pool_indices.tolist()
        return len(reqs) > 0 and all(r in self._cp_reqs for r in reqs)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.runtime_context import get_parallel

        if not forward_batch.forward_mode.is_idle() and (
            get_parallel().attn_cp_size > 1 and self._is_cp_decode(forward_batch)
        ):
            return self._forward_cp_decode(q, k, v, layer, forward_batch, **kwargs)
        return super().forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def _forward_cp_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
            batched_rpa_dcp_forward,
        )

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
            from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
                _quantize_kv_to_fp8,
            )

            k = _quantize_kv_to_fp8(k, k_scale, self.kv_cache_quant_dtype)
            v = _quantize_kv_to_fp8(v, v_scale, self.kv_cache_quant_dtype)
        else:
            k_scale = None
            v_scale = None

        out = batched_rpa_dcp_forward(
            self._cp_op,
            q,
            k,
            v,
            kv_cache,
            layer,
            forward_batch,
            self.forward_metadata,
            k_scale=k_scale,
            v_scale=v_scale,
            sinks=kwargs.get("sinks"),
        )
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)[..., : layer.v_head_dim]
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)


class TpuPallasRpaBatchedMultiStepDraftBackend(BaseTpuPallasRpaMultiStepDraftBackend):
    """One batched-RPA backend per EAGLE draft decode step."""

    attn_backend_cls = TpuPallasRpaBatchedAttnBackend
