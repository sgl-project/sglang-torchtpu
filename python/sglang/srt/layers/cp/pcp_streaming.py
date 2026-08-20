# SPDX-License-Identifier: Apache-2.0
"""Prefill-Context-Parallel (PCP) streaming strategy + driver.

Wraps the PCP streaming RPA kernel
(``sglang.srt.layers.attention.pallas.pcp_streaming_rpa``) as an sglang
CP-v2 strategy. Per attention layer the kernel runs a short fixed
sequence of Pallas calls — a current-chunk causal pass, the fused KV
writeback of the rows that pass captured off the ring, then a
history no-causal pass plus an output finalize — streaming KV pages
through the CP ring via ``pltpu.make_async_remote_copy`` and running
online-softmax on the locally-owned rank-major Q rows. Replaces the DCP
two-phase driver: no caller-side all-to-all, no LSE-weighted merge
outside the kernel, no separate context-vs-query phases, and no
pre-attention KV scatter (the cache update rides the ring).

Selected via ``--cp-strategy pcp`` and the attention backend
``pallas_tpu_rpa_v3_cp`` (auto-swapped from ``pallas_tpu_rpa_v3`` in
``TpuSRTPlatform.apply_server_args_defaults`` when ``attn_cp_size > 1``).

Rank-major compact Q layout note
--------------------------------
The PCP kernel expects the Q tensor rank-packed by ``interleave_size``.
Query ownership is BATCH-FLAT: the batch's request-major query rows are
one flat stream, split into interleave chunks, and rank ``r`` owns flat
indices ``[k*cycle + r*interleave, k*cycle + (r+1)*interleave)`` where
``cycle = pcp_size * interleave``. The owner coordinate of a request is
therefore ``cu_q_lens[req]``, not its request-absolute start. Causal
positions and KV-cache slots stay request-absolute and are derived on
device from ``kv_lens - q_lens``. The rank-major reordering is planned
per-forward on the host by :func:`plan_pcp_sequence_layout` (see
:mod:`sglang.srt.layers.cp.pcp_sequence_layout`), applied to the hidden
states and positions by ``shard_hidden_states`` / ``shard_position_ids``,
and reversed by ``gather_hidden_states`` before the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class PCPStreamingContextParallelMetadata(BaseContextParallelMetadata):
    """Per-forward metadata carried on ``forward_batch.attn_cp_metadata``
    for the PCP streaming path.

    Populated by :meth:`PCPStreamingCPStrategy.build_metadata` at the
    eager boundary and finalized by
    :meth:`PCPStreamingCPStrategy.build_extra_metadata`. The
    compiled attention forward does NOT read this struct — msgspec
    attribute access under torch.compile silently corrupts the trace on
    torch_tpu (see the M5 investigation cited in
    ``layers/communicator.py``). Instead, the eager runner mirrors the
    load-bearing scalars onto ``ForwardBatch`` (``pcp_attn_active``,
    ``moe_cp_active``, ``moe_cp_pad_max_tokens``, ``moe_cp_local_tokens``)
    and the compiled forward reads only those.

    The remaining consumer is the eager MoE-CP communicator
    (``_gather_hidden_states_and_residual_moe`` /
    ``_scatter_hidden_states_moe``), which reads
    ``per_rank_actual_token`` as a fallback when the plain-scalar mirror
    isn't populated (Zigzag / Interleave strategies).

    Inherits ``total_seq_lens`` / ``bs`` from
    :class:`~sglang.srt.layers.cp.base.BaseContextParallelMetadata`.
    """

    cp_group_size: int = 1
    cp_rank: int = 0
    interleave_size: int = 1
    # local_num_tokens / global_num_tokens: filled by the sequence-layout
    # planner. Kept for eager introspection; the compiled forward reads
    # the corresponding scalars off ForwardBatch instead.
    local_num_tokens: int = 0
    global_num_tokens: int = 0
    # per_rank_actual_token[r] is the local token count rank r sees at the
    # MoE all-gather boundary (:mod:`sglang.srt.layers.communicator`). PCP
    # is rank-balanced by construction — each rank's tensor has exactly
    # ``local_padded_num_tokens`` slots — so all entries are equal. The
    # non-PCP CP strategies (Zigzag, Interleave) use unequal shards and
    # rely on this field to pad the ``moe_cp`` all-gather.
    per_rank_actual_token: List[int] = field(default_factory=list)
    # Written by the shared ``pad_logical_token_to_physical`` pass in
    # layers/cp/padding.py, which every CP strategy goes through.
    per_rank_logical_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None


class PCPStreamingCPStrategy(ContextParallelStrategy):
    """TPU Prefill-Context-Parallel via the PCP streaming kernel.

    Both prefill and decode fire through the same path — the
    ``distribution=[decode_end, prefill_end, mixed_end]`` metadata lets
    the kernel handle mixed batches. What the kernel does NOT support:
    sliding-window attention, attention sinks, soft cap,
    ``head_dim == 64``. ``_validate_pcp_platform_config`` in
    ``hardware_backend/tpu/tpu_platform.py`` screens the server-arg-level
    combinations it can see; the model-level ones (sinks, soft cap,
    head_dim) are rejected when the op is built at warm time or when the
    kernel traces. Qwen3-Coder-480B satisfies all of them.
    """

    name = "pcp"
    kind = ContextParallelStrategyKind.PCP_STREAMING

    def __init__(self, cp_size: int):
        super().__init__(cp_size=cp_size)
        # Local Q-head count, published by the attention backend once the
        # model config is known. It sizes the kernel's ring tile, which is
        # bounded by VMEM and register spills (both scale with
        # tile * local_q_heads). Under PCP attn_tp_size == 1, so this is the
        # model's full attention-head count.
        self._local_q_heads: Optional[int] = None

    def set_local_q_heads(self, num_q_heads: int) -> None:
        self._local_q_heads = int(num_q_heads)

    @property
    def local_q_heads(self) -> int:
        if self._local_q_heads is None:
            raise RuntimeError(
                "PCP local Q-head count was never published. The attention "
                "backend calls set_local_q_heads() at construction; this "
                "means the CP strategy was used before the backend was built."
            )
        return self._local_q_heads

    def get_supported_attention_backend(self) -> List[CPAttentionBackendKind]:
        return [CPAttentionBackendKind.PALLAS_TPU_RPA_V3_CP]

    @property
    def supports_decode(self) -> bool:
        # Prefill and decode share one kernel; can_apply() accepts both.
        return True

    @property
    def local_compile_token_multiple(self) -> int:
        """Per-rank Q-tile floor of the PCP streaming Pallas kernel.

        This is the kernel's ring tile: the number of local Q rows that share
        one trip around the ring, which is also the compile-bucket floor
        because the kernel needs the global padded token count to be a
        multiple of ``pcp_size * ring_tile``.

        Exposed on the strategy so generic CP callers (and the TPU compile
        policy) can round padded_num_tokens without importing from
        ``layers/attention/pallas/...`` (which would be a layering violation).
        """
        from sglang.srt.layers.cp.pcp_sequence_layout import (
            resolve_pcp_ring_tile_tokens,
        )

        return resolve_pcp_ring_tile_tokens(self.local_q_heads)

    def can_apply(self, num_tokens: int, forward_batch: ForwardBatch) -> bool:
        """PCP is applicable whenever CP-v2 is enabled and there are tokens
        to shard. Unlike Zigzag (which required ``extend_len >= cp_size * 2``
        per seq), PCP's compact-Q layout has no minimum per-seq requirement;
        it just requires the total padded token count to be divisible by
        ``pcp_size * interleave_size``, which the platform's bucket
        calculator enforces.
        """
        if self.cp_size <= 1:
            return False
        forward_mode = forward_batch.forward_mode
        return forward_mode.is_context_parallel_extend() or forward_mode.is_decode()

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> PCPStreamingContextParallelMetadata:
        from sglang.srt.layers.cp.pcp_sequence_layout import (
            resolve_pcp_interleave_size,
        )
        from sglang.srt.runtime_context import get_server_args

        interleave_size = resolve_pcp_interleave_size(get_server_args().page_size)
        # seqs_len == [] is a caller-side bug (empty batch dispatched into
        # build_metadata); the len/sum fallbacks would silently produce
        # bs=0 / total_seq_lens=num_tokens. Fail loudly.
        if seqs_len is not None and len(seqs_len) == 0:
            raise ValueError(
                "PCPStreamingCPStrategy.build_metadata received an empty "
                "seqs_len list — caller dispatched an empty batch."
            )
        bs = len(seqs_len) if seqs_len is not None else 1
        total_seq_lens = int(sum(seqs_len)) if seqs_len else int(num_tokens)
        return PCPStreamingContextParallelMetadata(
            total_seq_lens=total_seq_lens,
            bs=bs,
            cp_group_size=int(self.cp_size),
            cp_rank=int(self.cp_rank),
            interleave_size=int(interleave_size),
            # local_num_tokens gets overwritten in build_extra_metadata
            # once the plan is materialised; leave 0 so any caller that
            # reads it before the plan lands sees an obviously-wrong count
            # instead of a stale value.
            local_num_tokens=0,
            global_num_tokens=int(num_tokens),
        )

    def shard_hidden_states(self, x: Any, forward_batch: ForwardBatch) -> Any:
        """Rank-major shard of hidden states. Reads the planner-computed
        reorder from ``forward_batch.pcp_layout_plan``; raises if that
        wasn't set (indicates planner integration is missing)."""
        from sglang.srt.layers.cp.pcp_sequence_layout import (
            apply_pcp_local_shard,
        )

        return apply_pcp_local_shard(x, forward_batch, axis=0)

    def shard_position_ids(self, positions: Any, forward_batch: ForwardBatch) -> Any:
        from sglang.srt.layers.cp.pcp_sequence_layout import (
            apply_pcp_local_shard,
        )

        return apply_pcp_local_shard(positions, forward_batch, axis=0)

    def gather_hidden_states(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-major-packed hidden states across the CP ring and
        rewrite back into request-major order for the sampler."""
        from sglang.srt.layers.cp.pcp_sequence_layout import (
            gather_and_unpack_pcp,
        )

        return gather_and_unpack_pcp(x, forward_batch)

    # ------------------------------------------------------------------
    # Base-class hooks PCP does not use. PCP keeps the KV cache sharded by
    # page across the ring and never materialises a full-layout copy: the
    # kernel streams remote pages instead. The attention backend
    # (TpuPallasRpaV3CpAttnBackend) drives the kernel directly, so the
    # run_attention dispatch hook is unused too.
    # ------------------------------------------------------------------

    def gather_kv_cache(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} streams KV pages around the ring inside the kernel; "
            "there is no host-side KV gather."
        )

    def run_attention(
        self,
        q: Any,
        forward_batch: ForwardBatch,
        device: Any,
        attn_fn: Any,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} runs attention through TpuPallasRpaV3CpAttnBackend, "
            "not through the CP run_attention dispatch hook."
        )

    def materialize_full_kv(
        self,
        forward_batch: ForwardBatch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} keeps the KV cache page-striped across ranks; "
            "full-layout K/V is never materialised."
        )

    def materialize_full_mla_kv(
        self,
        forward_batch: ForwardBatch,
        layer: Any,
        k_nope: Any,
        k_rope: Any,
    ) -> Any:
        raise NotImplementedError(f"{self.name} does not support MLA models.")

    def build_extra_metadata(
        self,
        forward_batch: ForwardBatch,
        seq_lens_cpu: Optional[List[int]],
        extend_lens_cpu: Optional[List[int]],
    ) -> None:
        """Populate ``forward_batch.pcp_layout_plan`` + Dynamo-safe mirrors.

        The PCP kernel expects Q in rank-major-packed order; the
        planner produces the reorder indices consumed by
        ``shard_hidden_states`` / ``shard_position_ids`` and read again in
        ``pcp_streaming_attention`` to populate ``local_num_tokens``. Also
        mirrors the load-bearing counts as plain scalars on ForwardBatch —
        the compiled forward can't read the msgspec metadata (M5).
        """
        import numpy as np

        from sglang.srt.layers.cp.pcp_sequence_layout import (
            plan_pcp_sequence_layout,
            resolve_pcp_interleave_size,
        )
        from sglang.srt.runtime_context import get_server_args

        num_tokens = len(forward_batch.input_ids)
        pcp_size = int(self.cp_size)
        pcp_rank = int(self.cp_rank)
        interleave_size = resolve_pcp_interleave_size(get_server_args().page_size)

        # Chunked prefill emits pure-prefill chunks (never a mix within one
        # request), so prompt_tokens = computed + scheduled satisfies the
        # boundary-crossing check even without a real per-request
        # prompt-vs-decode distinction.
        if extend_lens_cpu is None:
            bs = (
                int(forward_batch.seq_lens.shape[0])
                if forward_batch.seq_lens is not None
                else 1
            )
            scheduled = np.ones(bs, dtype=np.int64)
        else:
            scheduled = np.asarray(extend_lens_cpu, dtype=np.int64)
        raw_prefix = forward_batch.extend_prefix_lens_cpu
        prefix_lens_cpu = (
            [int(x) for x in raw_prefix] if raw_prefix is not None else None
        )
        if prefix_lens_cpu is not None:
            computed = np.asarray(prefix_lens_cpu, dtype=np.int64)
        elif seq_lens_cpu is not None:
            computed = np.asarray(seq_lens_cpu, dtype=np.int64) - scheduled
        else:
            computed = np.zeros_like(scheduled)
        prompt_tokens = computed + scheduled

        # Round up to pcp_size * q_block_size (kernel lockstep floor).
        # Rounding only to pcp_size would let a bucket like 1032 (=8*129)
        # pass and blow up at Pallas trace time.
        kernel_bucket = pcp_size * int(self.local_compile_token_multiple)
        padded_num_tokens = (
            (num_tokens + kernel_bucket - 1) // kernel_bucket
        ) * kernel_bucket
        if padded_num_tokens == 0:
            padded_num_tokens = kernel_bucket

        plan = plan_pcp_sequence_layout(
            num_scheduled_tokens_per_req=scheduled,
            prompt_tokens_per_req=prompt_tokens,
            computed_tokens_per_req=computed,
            padded_num_tokens=padded_num_tokens,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        forward_batch.pcp_layout_plan = plan

        meta = forward_batch.attn_cp_metadata
        assert isinstance(meta, PCPStreamingContextParallelMetadata)
        meta.local_num_tokens = plan.local_num_tokens
        meta.global_num_tokens = plan.global_num_tokens
        # PCP pads to local_padded_num_tokens per rank inside the planner,
        # so all ranks contribute the same padded count — no extra pad
        # needed at the moe_cp allgather boundary.
        meta.per_rank_actual_token = [plan.local_padded_num_tokens] * pcp_size

        # Plain-scalar mirrors: msgspec attribute access + IntEnum method
        # calls inside a torch.compile trace corrupt output on torch_tpu
        # (M5). The communicator + attention backend read these instead.
        forward_batch.moe_cp_pad_max_tokens = int(plan.local_padded_num_tokens)
        forward_batch.moe_cp_local_tokens = int(plan.local_padded_num_tokens)
        forward_batch.moe_cp_active = True
        forward_batch.pcp_attn_active = True


def pcp_streaming_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: ForwardBatch,
    token_to_kv_pool: Any,
    forward_metadata: Any,
    kv_cache_quant_dtype: Optional[torch.dtype],
    pcp_op: Any,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PCP streaming attention for one layer.

    Called by :meth:`TpuPallasRpaV3CpAttnBackend._forward_pcp`. Packages
    only the standard RPA tensor metadata (``kv_lens``, ``page_indices``,
    ``cu_q_lens``, ``distribution``) and invokes the kernel's
    torch custom op. All PCP-specific scalars (``interleave_size``,
    ``sm_scale``, ``sliding_window``, ``soft_cap``, ``k_scale``,
    ``v_scale``) are baked into the pre-built op at
    warm time by
    :func:`sglang.srt.layers.cp.pcp_streaming_op_registry.get_or_register_pcp_streaming_op`;
    the call site is tensor-args-only. (That split matters — reading
    those scalars off ``forward_batch.attn_cp_metadata`` inside the
    compiled forward silently corrupts the trace on torch_tpu, so the
    only place they can be sourced is the eager warm path.)

    This function does NOT write K/V into the cache before calling the
    kernel: the current-chunk pass captures the fresh rows as they
    circulate the ring and a fused writeback lands them on their
    request-absolute cache owner before the history pass runs. K and V
    are passed in purely as kernel inputs.

    The kernel returns ``(new_kv_cache, output)``. new_kv_cache aliases
    the donated storage byte-identically (kv_cache is arg 0 with
    ``donate_argnums=(0,)``), so callers must ``kv_cache.copy_(new_kv_cache)``
    under torch.compile to preserve XLA's aliasing chain and avoid a
    preserving copy of the full KV buffer.

    The rank-major Q layout the kernel expects is produced host-side by
    :func:`sglang.srt.layers.cp.pcp_sequence_layout.plan_pcp_sequence_layout`
    (called from :meth:`PCPStreamingCPStrategy.build_extra_metadata`)
    and the per-layer JAX op is prebuilt from the attention backend init
    (see ``_warm_pcp_streaming_op_registry`` in
    ``hardware_backend/tpu/attention_pallas_rpa_v3_cp.py``).
    """
    if sinks is not None:
        raise NotImplementedError(
            "PCP streaming does not support attention sinks. A model with "
            "sinks should never reach this backend — reject it in "
            "TpuSRTPlatform.apply_server_args_defaults / "
            "_validate_pcp_platform_config."
        )

    # Import inside the function so the kernel module is only
    # loaded on the CP path (non-CP forwards should not pay for its
    # heavy JAX imports).
    from sglang.srt.hardware_backend.tpu.attention_pallas_pcp_streaming_op import (
        invoke_pcp_streaming_op,
    )

    # NOTE: intentionally do NOT access forward_batch.attn_cp_metadata
    # (msgspec.Struct) inside this function — msgspec attribute reads may
    # produce spurious Dynamo specializations on the struct's Python identity
    # under torch.compile, breaking parity with the eager path. All PCP
    # scalars the kernel needs (interleave_size, per-rank counts) were baked
    # into the pre-built op at warm time; the kernel derives per-request
    # positions from ``seq_lens`` + ``cu_q_lens``.

    num_tokens = q.shape[0]
    q = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
    k = k.view(num_tokens, layer.tp_k_head_num, layer.qk_head_dim)
    v = v.view(num_tokens, layer.tp_v_head_num, layer.v_head_dim)

    # Asymmetric K/V head dims (rare on Qwen3-Coder; see base ``_forward``
    # for the padding rationale).
    if layer.v_head_dim < layer.qk_head_dim:
        v = torch.nn.functional.pad(v, (0, layer.qk_head_dim - layer.v_head_dim))

    # fp8 KV: quantize before invoking the kernel (matches non-CP path).
    from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
        _quantize_kv_to_fp8,
    )

    if kv_cache_quant_dtype is not None:
        k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
        k = _quantize_kv_to_fp8(k, k_scale, kv_cache_quant_dtype)
        v = _quantize_kv_to_fp8(v, v_scale, kv_cache_quant_dtype)

    kv_cache = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

    # ``invoke_pcp_streaming_op`` does the op call + the
    # ``kv_cache.copy_(new_kv_cache)`` writeback + returns only ``output``.
    # See its docstring for the frame-layout rationale (moving the
    # copy_ up here triggered warm-probe recompiles on the sglang stack).
    output = invoke_pcp_streaming_op(
        pcp_op,
        kv_cache=kv_cache,
        query=q,
        key=k,
        value=v,
        seq_lens=forward_metadata.kv_lens,
        block_tables=forward_metadata.page_indices,
        query_start_loc=forward_metadata.cu_q_lens,
        request_distribution=forward_metadata.distribution,
    )

    # Same padded-qk head_dim slice as base ``_forward``.
    out = output.reshape(num_tokens, layer.tp_q_head_num, -1)[..., : layer.v_head_dim]
    return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
