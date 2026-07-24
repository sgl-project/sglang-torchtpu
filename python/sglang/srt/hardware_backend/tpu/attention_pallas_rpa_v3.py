# SPDX-License-Identifier: Apache-2.0
"""Pallas ragged-paged-attention backend for the TPU torch_tpu platform.

Implements the Ragged Paged Attention (RPA) algorithm
(https://arxiv.org/abs/2604.15464), adapted from vllm/tpu-inference
(https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/kernels/).

This backend wraps the JAX/Pallas v3 ragged_paged_attention kernel
(``sglang.srt.layers.attention.pallas.ragged_paged_attention_v3``) and
exposes it as a PyTorch custom op so it can be invoked from PyTorch and
traced through ``torch.compile`` with ``backend="tpu"``.

Design notes
------------

* The kernel writes K/V into the fused KV cache itself. The backend therefore
  ignores ``save_kv_cache``: the cache is always updated.
* The fused KV cache must be allocated by
  :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`.
* All attention metadata (cu_q_lens, cu_kv_lens, page_indices, distribution)
  is built once per forward pass in :meth:`init_forward_metadata` and reused
  by every layer.
* Only Qwen3-dense-style models are exercised today (causal, GQA, no
  SWA). The kernel itself supports more, so enabling extra features is
  a matter of plumbing flags through the ``_KernelHandle`` cache key.
* EAGLE speculative decoding is supported: TARGET_VERIFY, DRAFT_EXTEND,
  and multi-step draft decode with ``speculative_step_id``.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here.
# ``torch_tpu._internal.pallas.jax_op`` inspects the wrapped function's
# annotations at registration time (``param.annotation is jax.Array``); under
# PEP 563 those would be plain strings and validation would reject them.
# Forward-reference types used only at type-check time are spelled as
# explicit string literals (e.g. ``"RadixAttention"``).

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class PallasForwardMetadata:
    """Per-forward-pass metadata shared by every attention layer."""

    # i32[max_num_seqs]  total KV length per sequence (after appending new K/V)
    kv_lens: torch.Tensor
    # i32[max_num_seqs + 1]  cumulative sum of query lengths (starts at 0)
    cu_q_lens: torch.Tensor
    # i32[max_num_seqs + 1]  cumulative sum of page-aligned KV lengths
    cu_kv_lens: torch.Tensor
    # i32[max_num_seqs * pages_per_seq]  flat page-indices look-up table
    page_indices: torch.Tensor
    # i32[3]  (decode_end, prefill_end, mixed_end) sequence-index ranges
    distribution: torch.Tensor
    # i32[max_num_seqs * pages_per_seq]  page indices
    swa_page_indices: Optional[torch.Tensor] = None


_PALLAS_RPA_OP = None


def _quantize_kv_to_fp8(
    t: torch.Tensor, scale: float, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """Static per-tensor quantization of K or V into ``fp8_dtype``.

    Mirrors the GPU path's ``set_kv_buffer`` quantization (``cache_k.div_(scale)
    .to(fp8)``) and vllm/tpu-inference's ``quantize_kv``
    (https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/kernels/):
    divide by the per-tensor scale, clamp to the
    fp8 representable range, then cast. The kernel undoes
    this by multiplying the attention scores / output by the same scale.
    """
    finfo = torch.finfo(fp8_dtype)
    scaled = t.to(torch.float32) / scale
    scaled = torch.clamp(scaled, finfo.min, finfo.max)
    return scaled.to(fp8_dtype)


def _build_pallas_kernel_op():
    """Lazily register the ragged_paged_attention kernel as a torch custom op.

    Only one custom op is registered per process; the kernel's
    Pallas-internal block-size selection is shape-aware, so shape-dependent
    specialisation happens inside the kernel rather than at the
    ``torch.library`` boundary. We override ``register_fake`` to return
    shape-matching empties without re-tracing the JAX function — this is
    what lets the dynamic token dimension flow through ``torch.compile``.
    """
    global _PALLAS_RPA_OP
    if _PALLAS_RPA_OP is not None:
        return _PALLAS_RPA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention as _jax_rpa,
    )

    op_name = "sglang_tpu_pallas::ragged_paged_attention"

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        attention_sink: Optional[jax.Array] = None,
        sm_scale: float = 1.0,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> tuple[jax.Array, jax.Array]:
        return _jax_rpa(
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
            # For fp8 KV cache: K/V are quantized to fp8 before this call, so
            # the kernel dequantizes on read by multiplying the attention
            # scores / output by these per-tensor scales. ``q_scale`` is left
            # unset (None) so queries stay bf16 — only K/V are quantized,
            # matching the GPU backends and torchtpu-vllm.
            k_scale=k_scale,
            v_scale=v_scale,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(10, 11, 12, 13, 14),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("kv_cache_fused",),
        device_types=["tpu"],
    )
    def custom_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        attention_sink: Optional[torch.Tensor] = None,
        sm_scale: float = 1.0,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.hardware_backend.tpu.pallas_export import (
            tpu_info_for_cpu_export,
        )

        with tpu_info_for_cpu_export():
            output, _updated_kv = jax_kernel(
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
            )
        return output

    def _fake_impl(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.empty_like(queries)

    custom_op.register_fake(_fake_impl)

    _PALLAS_RPA_OP = custom_op
    return custom_op


class TpuPallasRpaV3AttnBackend(AttentionBackend):
    """SGLang attention backend backed by the Pallas v3 ragged-paged-attention kernel.

    Sibling to :class:`TpuPallasAttnBackend` (``attention_pallas.py``), which
    wraps the Pallas flash + paged-attention kernels. Selected via
    ``--attention-backend pallas_tpu_rpa_v3`` (the TPU platform default).
    Requires the fused KV layout supplied by
    :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`,
    which ``TpuSRTPlatform.get_mha_kv_pool_cls`` returns for this backend and
    the model_runner passes as ``token_to_kv_pool_class``.
    """

    # The torch custom op exposes KV updates as an in-place mutation, so syncing
    # model outputs during TPU precompile is enough to execute the graph.
    requires_kv_sync_after_compile = False

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        super().__init__()
        self.device = model_runner.device
        self.speculative_step_id = speculative_step_id
        self.page_size = model_runner.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool

        # Hybrid-SWA models keep two KV sub-pools and need a
        # second page_indices tensor with indices translated into the SWA
        # sub-pool. ``token_to_kv_pool`` carries the full→swa index mapping.
        self._is_hybrid_swa = bool(getattr(model_runner, "is_hybrid_swa", False))
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # fp8 KV cache: when the resolved cache dtype is fp8, K/V are quantized
        # before the kernel call and the kernel dequantizes on read via
        # ``k_scale``/``v_scale``. ``None`` for bf16 caches (the default path).
        # e4m3 is the default/tested fp8 storage format (TPU's natural one);
        # e5m2 is accepted for parity. With per-tensor scaling, e4m3's extra
        # mantissa bit gives better KV-cache accuracy.
        kv_cache_dtype = getattr(model_runner, "kv_cache_dtype", None)
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        self.kv_cache_quant_dtype: Optional[torch.dtype] = (
            kv_cache_dtype if kv_cache_dtype in fp8_dtypes else None
        )

        self.forward_metadata: Optional[PallasForwardMetadata] = None
        # Cached ``distribution`` tensors on TPU, keyed by
        # (is_decode, real_batch, device) -- it depends only on those, so a
        # cached buffer is reused every step (see init_forward_metadata).
        self._distribution_cache: dict[tuple[bool, int, torch.device], torch.Tensor] = (
            {}
        )
        # Register the custom op eagerly so the (Python-introspection-heavy)
        # ``jax_op`` registration does not happen inside a ``torch.compile``
        # traced region — dynamo can't handle ``inspect.signature`` on the
        # wrapped JAX function. The forward path then just looks the op up.
        self._op = _build_pallas_kernel_op()

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int] = None
    ):
        # RPA-v3 rebuilds verify metadata and has no graph-capture buffers.
        pass

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Land deferred HiCache reload commits into the KV buffers before any
        # layer reads KV; this hook runs per forward in eager AND compiled.
        # Use the backend's cached pool: upstream moved token_to_kv_pool off
        # ForwardBatch onto the attn backend (see self.token_to_kv_pool above).
        self.token_to_kv_pool.tpu_commit_pending_loads()

        # Look up the cached RPA ``distribution`` on TPU, so we don't build it,
        # don't H2D it per step, and don't need the barrier that ensured the RPA
        # kernel launched only after that H2D finished.
        device = forward_batch.seq_lens.device
        non_empty_bs = self._non_empty_batch_size(forward_batch)
        is_decode = forward_batch.forward_mode.is_decode()
        distribution = self._get_cached_rpa_distribution(
            is_decode, non_empty_bs, device
        )
        self.forward_metadata = self._build_metadata(forward_batch, distribution)

    @staticmethod
    def _pre_bucket_padded_batch_size(forward_batch: ForwardBatch) -> int:
        """Scheduler row count before compile-bucket padding. Includes the
        zero-length rows the DP MLP sync appends, so it is not the
        token-carrying count (see _non_empty_batch_size)."""
        return int(
            getattr(forward_batch, "_tpu_real_b", forward_batch.seq_lens.shape[0])
        )

    def _non_empty_batch_size(self, forward_batch: ForwardBatch) -> int:
        """Token-carrying row count fed to the RPA distribution. A zero-length
        row marked active makes the kernel attend an empty KV range and stall,
        so clip the padded count to the live prefix (DP sync zero-pads the
        tail). Prefers seq_lens_cpu, falls back to device seq_lens, never the
        padded count; interior zeros raise rather than hang."""
        bs = self._pre_bucket_padded_batch_size(forward_batch)
        if bs == 0:
            return bs
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        seq_lens = seq_lens_cpu if seq_lens_cpu is not None else forward_batch.seq_lens
        live = int(torch.count_nonzero(seq_lens[:bs]).item())
        if live == bs:
            return bs
        if torch.any(seq_lens[:live] == 0):
            raise ValueError(
                "RPA active-row clip: zero-length rows are not tail-contiguous "
                f"(live={live}, batch={bs}); batch layout violates the DP "
                "sync-padding invariant."
            )
        return live

    def _get_cached_rpa_distribution(
        self, is_decode: bool, non_empty_bs: int, device
    ) -> torch.Tensor:
        """Cached RPA ``distribution`` on TPU for (is_decode, non_empty_bs, device)."""
        device = torch.device(device)
        key = (is_decode, non_empty_bs, device)
        cached = self._distribution_cache.get(key)
        if cached is not None:
            return cached

        dist = self._make_rpa_distribution_cpu(is_decode, non_empty_bs).to(device)
        dist.tolist()  # one-time blocking sync: wait until the H2D completes
        self._distribution_cache[key] = dist
        return dist

    @staticmethod
    def _make_rpa_distribution_cpu(is_decode: bool, non_empty_bs: int) -> torch.Tensor:
        num_active = torch.tensor(non_empty_bs, dtype=torch.int32)
        if is_decode:
            return num_active.repeat(3)
        zero = torch.zeros((), dtype=torch.int32)
        return torch.stack([zero, num_active, num_active])

    def _build_metadata(
        self, forward_batch: ForwardBatch, distribution: torch.Tensor
    ) -> PallasForwardMetadata:
        """Build per-forward-pass metadata for the Pallas kernel.

        Layout choice
        -------------
        The kernel uses ``cu_kv_lens[i]`` only as a *page-indices offset*
        (``start_kv_page_idx = cdiv(cu_kv_lens[i], page_size)``). It uses
        ``kv_lens[i]`` for the actual sequence length, and ``page_indices``
        is read as a flat 1D table of size
        ``sum_i cdiv(kv_lens[i], page_size)`` in the *ragged* world.

        We use a *rectangular* layout: every sequence reserves
        ``pages_per_seq`` slots in ``page_indices``, and we set
        ``cu_kv_lens[i] = i * pages_per_seq * page_size``. The kernel reads
        ``page_indices[i * pages_per_seq + kv_p_start]`` which is exactly
        the page for token ``kv_p_start`` of sequence ``i``.
        """
        page_size = self.page_size
        device = forward_batch.seq_lens.device
        bs = int(forward_batch.seq_lens.shape[0])
        req_to_token = self.req_to_token_pool.req_to_token
        max_context_len = req_to_token.shape[1]
        pages_per_seq = max_context_len // page_size

        # Host-resident fast path: when req_to_token is on CPU, build the
        # metadata on host (one H2D) instead of the on-device gather/reshape
        # that recompiles per shape.
        if req_to_token.device.type == "cpu":
            return self._build_metadata_host(
                forward_batch,
                page_size,
                device,
                bs,
                req_to_token,
                pages_per_seq,
                distribution,
            )

        # -------------------------------------------------------------- #
        # Speculative decoding: adjust seq_lens / extend metadata
        # -------------------------------------------------------------- #
        if forward_batch.forward_mode.is_target_verify():
            # TARGET_VERIFY: the target model verifies draft_token_num
            # tokens per request.  The KV cache already includes the
            # draft tokens, so bump seq_lens accordingly.
            assert forward_batch.spec_info is not None
            draft_token_num = forward_batch.spec_info.draft_token_num
            seq_lens = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
            extend_seq_lens = torch.full(
                (bs,), draft_token_num, dtype=torch.int32, device=device
            )
        elif forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND: batch.prefix_lens may be stale from the
            # initial prefill; seq_lens and extend_seq_lens are current.
            seq_lens = forward_batch.seq_lens.to(torch.int32)
            assert forward_batch.extend_seq_lens is not None
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        elif forward_batch.forward_mode.is_extend():
            assert forward_batch.extend_seq_lens is not None, (
                "extend_seq_lens is required for the Pallas attention backend in "
                "extend mode."
            )
            seq_lens = forward_batch.seq_lens.to(torch.int32)
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        else:
            # Decode (including draft decode).
            seq_lens = forward_batch.seq_lens.to(torch.int32)
            if (
                forward_batch.forward_mode.is_decode()
                and forward_batch.spec_info is not None
            ):
                # Multi-step draft decode: each step's backend has a
                # different speculative_step_id.  Bump seq_lens so the
                # kernel sees KV entries from prior draft steps.
                seq_lens = seq_lens + self.speculative_step_id + 1
            extend_seq_lens = torch.ones(bs, dtype=torch.int32, device=device)

        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)

        cu_q_lens = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        cu_q_lens[1:] = torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)

        # cu_kv_lens[i] = i * pages_per_seq * page_size so the kernel addresses
        # ``page_indices[i * pages_per_seq + j]`` for token j of seq i.
        cu_kv_lens = torch.arange(bs + 1, dtype=torch.int32, device=device) * (
            pages_per_seq * page_size
        )

        per_seq = req_to_token[req_pool_indices]  # [bs, max_ctx]
        # ``req_to_token.shape[1]`` may not be an exact multiple of
        # ``page_size`` (SGLang adds extra slack tokens), in which case
        # ``per_seq[:, ::page_size]`` would emit one extra trailing column.
        # Trim to exactly ``pages_per_seq`` entries per row so the flat
        # ``page_indices`` shape matches the rectangular layout the kernel
        # expects.
        page_indices = (
            per_seq[:, : pages_per_seq * page_size : page_size] // page_size
        ).to(torch.int32)
        page_indices = page_indices.reshape(bs * pages_per_seq).contiguous()

        swa_page_indices: Optional[torch.Tensor] = None
        if self._is_hybrid_swa:
            # ``page_indices`` are page-level indices into the full pool. To get
            # the SWA-pool variant, look up the full→swa token mapping at the
            # first token of each page and divide by page_size again.
            # ``full_to_swa_index_mapping`` is registered by
            # ``SWATokenToKVPoolAllocator`` at init time.
            full_to_swa = self.token_to_kv_pool.full_to_swa_index_mapping
            assert full_to_swa is not None, (
                "Hybrid-SWA model is configured but full_to_swa_index_mapping "
                "has not been registered yet — did the allocator finish init?"
            )
            # First token index of each full-pool page (page_indices * page_size).
            # full_to_swa is host-resident (CPU) on TPU: gather on CPU (where the
            # table lives), then move the small int32 result to the batch device.
            full_token_starts = page_indices.to(torch.int64) * page_size
            swa_page_indices = (
                (full_to_swa[full_token_starts.cpu()] // page_size)
                .to(torch.int32)
                .to(device)
            )

        return PallasForwardMetadata(
            kv_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            page_indices=page_indices,
            distribution=distribution,
            swa_page_indices=swa_page_indices,
        )

    def _build_metadata_host(
        self,
        forward_batch,
        page_size,
        device,
        bs,
        req_to_token,
        pages_per_seq,
        distribution: torch.Tensor,
    ) -> PallasForwardMetadata:
        """Host-resident metadata build (gated in ``_build_metadata``).

        req_to_token is on CPU, so page_indices / cu-lens are built with
        torch-CPU ops (never traced -> no per-shape recompile) and H2D'd once.
        kv_lens stays on device; ``distribution`` is the caller's cached buffer.
        """
        if forward_batch.forward_mode.is_target_verify():
            # TARGET_VERIFY: the KV cache already includes the draft tokens,
            # so bump seq_lens by draft_token_num (shape-stable device add).
            assert forward_batch.spec_info is not None
            draft_token_num = forward_batch.spec_info.draft_token_num
            kv_lens = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
            extend_seq_lens = torch.full((bs,), draft_token_num, dtype=torch.int32)
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            # EXTEND and DRAFT_EXTEND_V2 use current lengths.
            assert forward_batch.extend_seq_lens is not None
            kv_lens = forward_batch.seq_lens.to(torch.int32)
            extend_seq_lens = forward_batch.extend_seq_lens.to("cpu", torch.int32)
        else:
            kv_lens = forward_batch.seq_lens.to(torch.int32)
            if (
                forward_batch.forward_mode.is_decode()
                and forward_batch.spec_info is not None
            ):
                # Multi-step draft decode: each step's backend has a different
                # speculative_step_id.  Bump seq_lens so the kernel sees KV
                # entries from prior draft steps.
                kv_lens = kv_lens + self.speculative_step_id + 1
            extend_seq_lens = torch.ones(bs, dtype=torch.int32)

        # Small per-batch tensors -> host.
        req_pool_indices = forward_batch.req_pool_indices.to("cpu", torch.int64)

        cu_q_lens = torch.zeros(bs + 1, dtype=torch.int32)
        cu_q_lens[1:] = torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)
        cu_kv_lens = torch.arange(bs + 1, dtype=torch.int32) * (
            pages_per_seq * page_size
        )

        # Gather only the strided page-slot columns [bs, pages_per_seq] (one
        # slot per page), not the full [bs, max_ctx] row. page_cols is constant,
        # so cache it.
        page_cols = getattr(self, "_host_page_cols", None)
        if page_cols is None or page_cols.numel() != pages_per_seq:
            page_cols = torch.arange(
                0, pages_per_seq * page_size, page_size, dtype=torch.long
            )
            self._host_page_cols = page_cols
        per_seq = req_to_token[req_pool_indices[:, None], page_cols[None, :]]
        page_indices = (per_seq // page_size).to(torch.int32)
        page_indices = page_indices.reshape(bs * pages_per_seq).contiguous()

        # Hybrid-SWA: translate full-pool pages to SWA-pool pages. The host path
        # gates only on req_to_token residency, so this must mirror the device
        # path or SWA layers would address the full pool.
        # full_to_swa_index_mapping is host-resident (CPU) on TPU, so do the
        # lookup here on the CPU page_indices and H2D the result once — keeping
        # this builder free of on-device gathers.
        swa_page_indices: Optional[torch.Tensor] = None
        if self._is_hybrid_swa:
            full_to_swa = self.token_to_kv_pool.full_to_swa_index_mapping
            assert full_to_swa is not None, (
                "Hybrid-SWA model is configured but full_to_swa_index_mapping "
                "has not been registered yet — did the allocator finish init?"
            )
            # First token index of each full-pool page (page_indices * page_size).
            # page_indices and full_to_swa are both CPU here, so gather on CPU and
            # H2D the small int32 result once.
            full_token_starts = page_indices.to(torch.int64) * page_size
            swa_page_indices = (
                (full_to_swa[full_token_starts] // page_size)
                .to(torch.int32)
                .to(device, non_blocking=True)
            )

        # H2D for host-built cu-lens / page_indices; kv_lens already on device,
        # distribution is the caller's cached buffer.
        return PallasForwardMetadata(
            kv_lens=kv_lens,
            cu_q_lens=cu_q_lens.to(device, non_blocking=True),
            cu_kv_lens=cu_kv_lens.to(device, non_blocking=True),
            page_indices=page_indices.to(device, non_blocking=True),
            distribution=distribution,
            swa_page_indices=swa_page_indices,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch, sinks=kwargs.get("sinks"))

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # During torch.compile, rotary_emb may produce a 3D query; flatten so
        # the view below matches the kernel's expected layout. This mirrors
        # TorchNativeAttnBackend.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        return self._forward(q, k, v, layer, forward_batch, sinks=kwargs.get("sinks"))

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.forward_metadata is not None, (
            "Pallas attention backend was not initialised for this forward pass; "
            "did you forget to call init_forward_metadata()?"
        )

        num_tokens = q.shape[0]
        q = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(num_tokens, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(num_tokens, layer.tp_v_head_num, layer.v_head_dim)

        # Asymmetric K/V head dims (e.g. MiMo / DeepSeek-style qk=192, v=128):
        # the fused KV buffer stores one shared head_dim, and the Pallas kernel
        # reads K and V from it at the same width. Pad V up to the qk head_dim
        # with zeros so it occupies the qk-width slot; the padded value columns
        # contribute 0 to the attention output, which is sliced back to
        # v_head_dim after the kernel. ``F.pad`` is a no-op when qk == v.
        #
        # This is a correctness stopgap, not the end goal: it wastes
        # (qk - v)/qk of the V cache and kernel V-bandwidth on zero columns. The
        # goal is a kernel that reads V at its native v_head_dim so no padding
        # (or output slicing) is needed; until then this keeps K==V width.
        if layer.v_head_dim < layer.qk_head_dim:
            v = torch.nn.functional.pad(v, (0, layer.qk_head_dim - layer.v_head_dim))

        kv_cache = self.token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

        op = self._op

        sliding_window = getattr(layer, "sliding_window_size", None)
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None

        # For hybrid-SWA models, SWA layers index into the SWA sub-pool;
        # full-attention layers (or non-hybrid models) use the full-pool indices.
        is_swa_layer = sliding_window is not None
        if is_swa_layer and self.forward_metadata.swa_page_indices is not None:
            page_indices = self.forward_metadata.swa_page_indices
        else:
            page_indices = self.forward_metadata.page_indices

        soft_cap = layer.logit_cap if getattr(layer, "logit_cap", 0) else None

        # fp8 KV cache: quantize new K/V to fp8 here (the kernel writes them to
        # the fp8 cache itself and reads them back for the current tokens), and
        # hand the per-tensor scales to the kernel for dequant-on-read. The
        # scales come from the same ``k_scale_float``/``v_scale_float`` the GPU
        # backends use; they default to 1.0 when the checkpoint carries no KV
        # scales (e.g. a bf16-weight model run with --kv-cache-dtype fp8_e4m3).
        if self.kv_cache_quant_dtype is not None:
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
            k = _quantize_kv_to_fp8(k, k_scale, self.kv_cache_quant_dtype)
            v = _quantize_kv_to_fp8(v, v_scale, self.kv_cache_quant_dtype)
        else:
            k_scale = None
            v_scale = None

        out = op(
            q,
            k,
            v,
            kv_cache,
            self.forward_metadata.kv_lens,
            page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            sinks,
            float(layer.scaling),
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
        )

        # The kernel emits output at the (padded) qk head_dim. For asymmetric
        # K/V the trailing (qk - v) columns come from the zero-padded V slots and
        # are dropped here so o_proj receives exactly tp_q_head_num * v_head_dim.
        # When qk == v this view+slice is a no-op reshape.
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)[..., : layer.v_head_dim]
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------ #
    # Capability flags
    # ------------------------------------------------------------------ #

    def support_triton(self) -> bool:
        return False

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0


class TpuPallasRpaV3MultiStepDraftBackend:
    """Wrap one RPA-v3 backend per EAGLE draft decode step.

    Each step's backend carries a different ``speculative_step_id`` so
    ``_build_metadata`` bumps ``seq_lens`` by the right offset,
    mirroring ``TpuPallasMultiStepDraftBackend`` for the flash backend.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            TpuPallasRpaV3AttnBackend(model_runner, speculative_step_id=i)
            for i in range(self.speculative_num_steps - 1)
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert forward_batch.spec_info is not None
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        pass

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        self.init_forward_metadata(forward_batch)

    def init_forward_metadata_replay_cuda_graph(self, forward_batch: ForwardBatch, bs):
        self.init_forward_metadata(forward_batch)
