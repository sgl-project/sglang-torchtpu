# SPDX-License-Identifier: Apache-2.0
"""Shared base for the TPU Pallas ragged-paged-attention backends.

Implements the Ragged Paged Attention (RPA) algorithm
(https://arxiv.org/abs/2604.15464), adapted from vllm/tpu-inference
(https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/kernels/).

Both ``attention_pallas_rpa_v3`` and ``attention_pallas_rpa_batched`` are thin
subclasses of :class:`BaseTpuPallasRpaAttnBackend` that differ only in a small
:class:`RpaKernelSpec` (which kernel to import, how to invoke it, the extend
``distribution`` encoding, and the draft-extend forward-mode variant). The
shared logic — metadata building, fp8 KV quantization, the torch custom-op
shell, ``forward_extend`` / ``forward_decode`` / ``_forward`` — lives here.

Design notes
------------

* The kernel writes K/V into the fused KV cache itself. The backend therefore
  ignores ``save_kv_cache``: the cache is always updated.
* The fused KV cache must be allocated by
  :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`.
* All attention metadata (cu_q_lens, cu_kv_lens, page_indices, distribution)
  is built once per forward pass in :meth:`init_forward_metadata` and reused
  by every layer.
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.host_metadata_cache import HostMetadataCache
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.speculative.spec_info import SpecInputType

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _get_pages_per_seq(max_context_len: int, page_size: int) -> int:
    return (max_context_len + page_size - 1) // page_size


# Scoped SMEM available to one Pallas kernel invocation on TPU.
_SCOPED_SMEM_BYTES = 1 << 20


def _page_table_smem_bytes(max_num_seqs: int, pages_per_seq: int) -> int:
    """Bytes of SMEM the kernel's page table occupies for ``max_num_seqs``.

    The page table is a scalar-prefetch operand: ``max_num_seqs * pages_per_seq``
    int32 entries, padded to a multiple of 128 entries.
    """
    entries = max_num_seqs * pages_per_seq
    return ((entries + 127) // 128) * 128 * 4


def _validate_kv_len_capacity(max_kv_len: int, capacity: int) -> None:
    if max_kv_len > capacity:
        raise ValueError(
            "RPA KV length exceeds the per-request page-table capacity: "
            f"required={max_kv_len}, available={capacity}."
        )


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
    # Scalar i32 query-row count covered by ``distribution``.
    active_num_tokens: torch.Tensor
    # i32[max_num_seqs * pages_per_seq]  page indices
    swa_page_indices: torch.Tensor | None = None
    # Static query rows per DECODE sequence for this forward (1 = ordinary
    # decode); above 1 the active rows take the kernel's DECODE range.
    decode_query_size: int = 1
    # Owned scalar i32: -1=native fallback, 0=static base, 1024/2048=promotion.
    # Backends without runtime tile selection leave this unset.
    prefill_kv_tile_hint: torch.Tensor | None = None


def _make_active_num_tokens(cu_q_lens: torch.Tensor, non_empty_bs: int) -> torch.Tensor:
    """Return an owned scalar whose compiled input layout is batch-independent."""
    # TorchTPU compiled execution binds a view's contiguous base buffer. Without
    # the clone, changing non_empty_bs changes that base from [old_bs + 1] to
    # [new_bs + 1] even though the scalar's logical shape remains unchanged.
    # Compiler-level tracking: https://github.com/sgl-project/sglang-torchtpu/issues/565
    return cu_q_lens[non_empty_bs].clone()


def _is_fixed_width_eagle_refresh(forward_batch) -> bool:
    spec_info = forward_batch.spec_info
    return (
        forward_batch.forward_mode.is_draft_extend_v2()
        and spec_info is not None
        and spec_info.spec_input_type == SpecInputType.EAGLE_DRAFT_EXTEND
        and spec_info.num_tokens_per_req > 0
    )


def _zero_inactive_rpa_rows(
    output: torch.Tensor, active_num_tokens: torch.Tensor
) -> torch.Tensor:
    """Zero output rows that the RPA distribution does not execute."""
    row_ids = torch.arange(output.shape[0], device=output.device)
    active = row_ids < active_num_tokens
    active = active.reshape((output.shape[0],) + (1,) * (output.ndim - 1))
    zero = torch.zeros((), dtype=output.dtype, device=output.device)
    return torch.where(active, output, zero)


class InvokeKernelFn(Protocol):
    """Calls the JAX ragged_paged_attention kernel with a variant's arg layout.

    Made explicit (rather than ``Callable[..., tuple]``) so a new variant's
    invoker is checked against the contract the base's custom-op shell relies on.
    Returns ``(output, updated_kv_cache)``.
    """

    def __call__(
        self,
        jax_rpa: Callable,
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
        decode_query_size: int = 1,
        use_causal_mask: bool = True,
    ) -> tuple: ...


@dataclass(frozen=True)
class RpaKernelSpec:
    """Declares how one RPA kernel variant differs from the shared base.

    Every field maps to a concrete code path that actually differs between the
    v3 and batched backends — nothing speculative.
    """

    # torch custom-op name (must be unique per variant; the compile cache keys
    # on the FX graph = op name + shapes, so bump the suffix when the vendored
    # kernel source changes to force a real recompile).
    op_name: str
    # Returns the JAX ``ragged_paged_attention`` callable (imported lazily so
    # JAX-less hosts can still import this module).
    import_kernel: Callable[[], Callable]
    # Invokes ``import_kernel()`` with this variant's argument layout.
    invoke_kernel: InvokeKernelFn
    # Whether the kernel consumes ``attention_sink``. When False the base raises
    # rather than silently dropping a passed sink.
    supports_attention_sink: bool
    # Whether the JAX kernel consumes ``cu_kv_lens``. Keep unused operands out
    # of the exported JAX function: StableHLO drops them, while torch_tpu still
    # tries to refine every declared dynamic operand.
    uses_cu_kv_lens: bool
    # Per-slot on/off mask selecting which of the three extend/prefill
    # ``distribution`` slots get the active-sequence count: v3 = (0, 1, 1)
    # ([0, n, n]); batched = (0, 0, 1) ([0, 0, n], MIXED-only — the batched
    # kernel never runs the pure-prefill range). Decode is always [n, n, n].
    extend_distribution_mask: tuple[int, int, int]
    # Minimum KV page size required by the backend.
    min_page_size: int | None = None
    # Whether the kernel's DECODE range accepts ``decode_query_size`` query rows
    # per sequence, so fixed-width target verification and decode-phase draft
    # extension run as decode.
    supports_decode_query_size: bool = False


# torch custom ops are process-global; cache one per op_name so repeated backend
# construction (e.g. one backend per draft step) reuses the registered op.
_OP_CACHE: dict = {}


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


def _resolve_decode_query_size(
    *, supports_decode_query_size: bool, spec_enabled: bool
) -> int:
    """Static query rows per DECODE sequence for one backend instance.

    With speculative decoding both fixed-width phases carry this many query
    rows per request: the target's verification and the draft's extension
    over the verified window. Without it every phase is single-token decode
    or ragged extend.
    """
    if not supports_decode_query_size or not spec_enabled:
        return 1
    return get_spec().speculative_num_draft_tokens or 1


def _build_pallas_kernel_op(spec: RpaKernelSpec):
    """Lazily register ``spec``'s ragged_paged_attention kernel as a torch custom op.

    One custom op is registered per ``spec.op_name`` (cached in ``_OP_CACHE``);
    the kernel's Pallas-internal block-size selection is shape-aware, so
    shape-dependent specialisation happens inside the kernel rather than at the
    ``torch.library`` boundary. We override ``register_fake`` to return
    shape-matching empties without re-tracing the JAX function — this is what
    lets the dynamic token dimension flow through ``torch.compile``.
    """
    cached = _OP_CACHE.get(spec.op_name)
    if cached is not None:
        return cached

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    _jax_rpa = spec.import_kernel()
    op_name = spec.op_name

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array | None,
        distribution: jax.Array,
        attention_sink: jax.Array | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        decode_query_size: int = 1,
        prefill_kv_tile_hint: jax.Array | None = None,
        use_causal_mask: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        if not spec.supports_attention_sink and attention_sink is not None:
            raise NotImplementedError(f"{op_name} does not support attention sinks")
        # For fp8 KV cache: K/V are quantized to fp8 before this call, so the
        # kernel dequantizes on read by multiplying the attention scores /
        # output by these per-tensor scales. ``q_scale`` is left unset so
        # queries stay bf16 — only K/V are quantized, matching the GPU backends.
        hint_options = (
            {}
            if prefill_kv_tile_hint is None
            else {"prefill_kv_tile_hint": prefill_kv_tile_hint}
        )
        return spec.invoke_kernel(
            _jax_rpa,
            queries=queries,
            keys=keys,
            values=values,
            kv_cache_fused=kv_cache_fused,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            distribution=distribution,
            attention_sink=attention_sink,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            decode_query_size=decode_query_size,
            use_causal_mask=use_causal_mask,
            **hint_options,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        # Static (non-traced) args by _wrapped position: 10=sm_scale,
        # 11=sliding_window, 12=soft_cap, 13=k_scale, 14=v_scale,
        # 15=decode_query_size, 17=use_causal_mask.
        static_argnums=(10, 11, 12, 13, 14, 15, 17),
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
        attention_sink: torch.Tensor | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        decode_query_size: int = 1,
        prefill_kv_tile_hint: torch.Tensor | None = None,
        use_causal_mask: bool = True,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            output, _updated_kv = jax_kernel(
                queries,
                keys,
                values,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens if spec.uses_cu_kv_lens else None,
                distribution,
                attention_sink,
                sm_scale,
                sliding_window,
                soft_cap,
                k_scale,
                v_scale,
                decode_query_size,
                prefill_kv_tile_hint,
                use_causal_mask,
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

    _OP_CACHE[op_name] = custom_op
    return custom_op


class BaseTpuPallasRpaAttnBackend(AttentionBackend):
    """Shared SGLang attention backend for the Pallas ragged-paged-attention kernels.

    Subclasses set :attr:`kernel_spec` to a :class:`RpaKernelSpec` and inherit
    everything else. Requires a KV pool in the layout its kernel reads:
    :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`
    or a subclass (batched's SEQ_ALONG_LANE layout uses
    ``TpuSeqAlongLaneKVPool``). ``tpu_platform._mha_kv_pool_cls`` maps each
    backend to its pool, and the model_runner passes it as
    ``token_to_kv_pool_class``.
    """

    # Set by each subclass to select the kernel variant; None on the base only.
    kernel_spec: ClassVar[RpaKernelSpec | None] = None

    # The torch custom op exposes KV updates as an in-place mutation, so syncing
    # model outputs during TPU precompile is enough to execute the graph.
    requires_kv_sync_after_compile = False

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        if self.kernel_spec is None:
            raise ValueError(
                f"{type(self).__name__} must set a class-level kernel_spec; "
                "instantiate a concrete RPA backend subclass, not the base."
            )
        super().__init__()
        self.device = model_runner.device
        self._host_metadata_cache = (
            HostMetadataCache()
            if envs.SGLANG_TPU_ENABLE_ATTN_METADATA_CACHE.get()
            else None
        )
        self.speculative_step_id = speculative_step_id
        self.page_size = model_runner.page_size
        self.max_context_len = (
            model_runner.model_config.context_len + get_req_to_token_extra_context_len()
        )
        self.req_to_token_pool = model_runner.req_to_token_pool
        req_to_token_width = self.req_to_token_pool.req_to_token.shape[1]
        if self.max_context_len > req_to_token_width:
            raise ValueError(
                "RPA context capacity exceeds the request-to-token table width: "
                f"required={self.max_context_len}, available={req_to_token_width}."
            )
        max_bs = get_exec().graph.torch_compile_max_bs
        if max_bs:
            pages_per_seq = _get_pages_per_seq(self.max_context_len, self.page_size)
            table_bytes = _page_table_smem_bytes(max_bs, pages_per_seq)
            if table_bytes > _SCOPED_SMEM_BYTES:
                logger.warning(
                    "RPA page table for %d sequences x %d pages needs %.2f MiB of "
                    "SMEM, above the %.0f MiB scoped budget; the kernel compile "
                    "will fail. Lower --torch-compile-max-bs (<= %d at this "
                    "context length) or raise --page-size.",
                    max_bs,
                    pages_per_seq,
                    table_bytes / 2**20,
                    _SCOPED_SMEM_BYTES / 2**20,
                    max(1, (_SCOPED_SMEM_BYTES // 4 - 128) // pages_per_seq),
                )

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
        self.kv_cache_quant_dtype: torch.dtype | None = (
            kv_cache_dtype if kv_cache_dtype in fp8_dtypes else None
        )

        # Query rows per DECODE sequence. Target verification and the draft's
        # decode-phase extension both carry a fixed number of draft tokens per
        # request and run on the decode schedule; non-speculative servers keep
        # single-token decode.
        self.decode_query_size = _resolve_decode_query_size(
            supports_decode_query_size=self.kernel_spec.supports_decode_query_size,
            spec_enabled=not model_runner.spec_algorithm.is_none(),
        )

        self.forward_metadata: PallasForwardMetadata | None = None
        # Cached ``distribution`` tensors on TPU, keyed by
        # (is_decode, real_batch, device) -- it depends only on those, so a
        # cached buffer is reused every step (see init_forward_metadata).
        self._distribution_cache: dict[
            tuple[bool, int, torch.device], torch.Tensor
        ] = {}
        # Register the custom op eagerly so the (Python-introspection-heavy)
        # ``jax_op`` registration does not happen inside a ``torch.compile``
        # traced region — dynamo can't handle ``inspect.signature`` on the
        # wrapped JAX function. The forward path then just looks the op up.
        self._op = self._build_kernel_op()

    def _build_kernel_op(self):
        """Build the Pallas custom op stored as ``self._op``.

        Overridable hook: variants whose kernel does not fit the shared
        ``(queries, keys, values)`` op shell — e.g. the MLA backend, whose
        kernel takes four inputs and aliases the donated cache at a different
        position — return their own op instead of an ``RpaKernelSpec``-built one.
        """
        return _build_pallas_kernel_op(self.kernel_spec)

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def _is_draft_extend(self, forward_mode) -> bool:
        """Return whether this is the backend's Spec V2 draft-extend mode."""
        return forward_mode.is_draft_extend_v2()

    def _is_extend_including_draft(self, forward_mode) -> bool:
        """Extend (including Spec V2 draft-extend) — for the host path."""
        return forward_mode.is_extend(include_draft_extend_v2=True)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Land deferred HiCache reload commits into the KV buffers before any
        # layer reads KV; this hook runs per forward in eager AND compiled.
        # Use the backend's cached pool: upstream moved token_to_kv_pool off
        # ForwardBatch onto the attn backend (see self.token_to_kv_pool above).
        self.token_to_kv_pool.tpu_commit_pending_loads()

        # Build metadata and its cached distribution from the same active width.
        non_empty_bs = self._non_empty_batch_size(forward_batch)
        self.forward_metadata = self._build_metadata(forward_batch, non_empty_bs)

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

    def _forward_decode_query_size(self, forward_batch: ForwardBatch) -> int:
        """Static DECODE query width for this forward, or 1.

        The kernel's decode loop bounds are static, so the width the batch
        carries has to be the one this backend was built for. Target
        verification publishes it on ``spec_info``; the draft's decode-phase
        extension carries it as a uniform ``extend_seq_lens``. A widened DP
        window or the ragged prompt-phase extension stays on MIXED.
        """
        width = self.decode_query_size
        if width == 1:
            return 1
        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            return width if forward_batch.spec_info.num_tokens_per_req == width else 1
        if self._is_draft_extend(mode):
            lens = forward_batch.extend_seq_lens_cpu
            return width if lens and all(n == width for n in lens) else 1
        return 1

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

    def _make_rpa_distribution_cpu(
        self, is_decode: bool, non_empty_bs: int
    ) -> torch.Tensor:
        num_active = torch.tensor(non_empty_bs, dtype=torch.int32)
        if is_decode:
            return num_active.repeat(3)
        # Extend/prefill: each slot is num_active or 0 per the variant's mask
        # (v3 [0, n, n]; batched [0, 0, n]).
        zero = torch.zeros((), dtype=torch.int32)
        return torch.stack(
            [
                num_active if slot else zero
                for slot in self.kernel_spec.extend_distribution_mask
            ]
        )

    def _build_metadata(
        self,
        forward_batch: ForwardBatch,
        non_empty_bs: int,
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
        decode_query_size = self._forward_decode_query_size(forward_batch)
        distribution = self._get_cached_rpa_distribution(
            forward_batch.forward_mode.is_decode() or decode_query_size > 1,
            non_empty_bs,
            device,
        )
        bs = int(forward_batch.seq_lens.shape[0])
        req_to_token = self.req_to_token_pool.req_to_token
        pages_per_seq = _get_pages_per_seq(self.max_context_len, page_size)

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
                non_empty_bs,
                decode_query_size,
            )

        # -------------------------------------------------------------- #
        # Speculative decoding: adjust seq_lens / extend metadata
        # -------------------------------------------------------------- #
        kv_len_bump = 0
        if forward_batch.forward_mode.is_target_verify():
            # EAGLE3 and DFlash expose their verification width through
            # num_tokens_per_req, which can differ from draft_token_num.
            assert forward_batch.spec_info is not None
            verify_width = forward_batch.spec_info.num_tokens_per_req
            kv_len_bump = verify_width
            seq_lens = (forward_batch.seq_lens + verify_width).to(torch.int32)
            extend_seq_lens = torch.full(
                (bs,), verify_width, dtype=torch.int32, device=device
            )
        elif self._is_draft_extend(forward_batch.forward_mode):
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
                kv_len_bump = self.speculative_step_id + 1
                seq_lens = seq_lens + kv_len_bump
            extend_seq_lens = torch.ones(bs, dtype=torch.int32, device=device)

        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and bs > 0:
            max_kv_len = int(seq_lens_cpu[:bs].max().item()) + kv_len_bump
            _validate_kv_len_capacity(
                max_kv_len,
                pages_per_seq * page_size,
            )

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

        swa_page_indices: torch.Tensor | None = None
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
            active_num_tokens=_make_active_num_tokens(cu_q_lens, non_empty_bs),
            swa_page_indices=swa_page_indices,
            decode_query_size=decode_query_size,
        )

    def _get_query_metadata(
        self,
        *,
        bs: int,
        kv_stride: int,
        non_empty_bs: int,
        device,
        query_width: int | None = None,
        query_lens: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Device offsets for one fixed ``query_width`` or per-row ``query_lens``."""
        assert (query_width is None) != (query_lens is None)
        if query_width is not None:
            # Decode uses one token per row; target verify uses query_width.
            # Padded rows have the same width, so (bs, query_width) fixes the offsets.
            cu_q_lens = self._layout_metadata(
                key=("cu_q_lens", bs, query_width),
                build=lambda: torch.arange(bs + 1, dtype=torch.int32) * query_width,
                device=device,
            )
            active_tokens = non_empty_bs * query_width
        else:
            # Extend query lengths can differ across requests, even at the same bs.
            # Include padded rows in the offsets, but exclude them from active_tokens.
            assert len(query_lens) == bs
            cu_q_lens_cpu = torch.zeros(bs + 1, dtype=torch.int32)
            cu_q_lens_cpu[1:] = torch.cumsum(
                torch.tensor(query_lens, dtype=torch.int32), dim=0, dtype=torch.int32
            )
            # One value slot, so varying prefills cannot grow the cache.
            cu_q_lens = self._value_metadata(
                name="cu_q_lens", value=cu_q_lens_cpu, device=device
            )
            active_tokens = sum(query_lens[:non_empty_bs])
        cu_kv_lens = self._layout_metadata(
            key=("cu_kv_lens", bs, kv_stride),
            build=lambda: torch.arange(bs + 1, dtype=torch.int32) * kv_stride,
            device=device,
        )
        # Own the scalar's storage so its compiled input layout stays the same
        # across batch sizes (see _make_active_num_tokens).
        active_num_tokens = self._layout_metadata(
            key=("active_num_tokens", active_tokens),
            build=lambda: torch.tensor(active_tokens, dtype=torch.int32),
            device=device,
        )
        return cu_q_lens, cu_kv_lens, active_num_tokens

    def _build_metadata_host(
        self,
        forward_batch,
        page_size,
        device,
        bs,
        req_to_token,
        pages_per_seq,
        distribution: torch.Tensor,
        non_empty_bs: int,
        decode_query_size: int = 1,
    ) -> PallasForwardMetadata:
        """Build metadata on the CPU and reuse cached device copies when possible.

        CPU operations stay outside tracing, avoiding shape-specific recompiles.
        kv_lens stays on device; distribution is the caller's cached buffer.
        """
        kv_len_bump = 0
        query_width = None
        query_lens = None
        if forward_batch.forward_mode.is_target_verify():
            assert forward_batch.spec_info is not None
            verify_width = forward_batch.spec_info.num_tokens_per_req
            kv_len_bump = verify_width
            kv_lens = (forward_batch.seq_lens + verify_width).to(torch.int32)
            query_width = verify_width
        elif self._is_extend_including_draft(forward_batch.forward_mode):
            # EXTEND and DRAFT_EXTEND use current lengths.
            assert forward_batch.extend_seq_lens is not None
            kv_lens = forward_batch.seq_lens.to(torch.int32)
            query_lens = self._extend_query_lens(forward_batch, bs)
        else:
            kv_lens = forward_batch.seq_lens.to(torch.int32)
            if (
                forward_batch.forward_mode.is_decode()
                and forward_batch.spec_info is not None
            ):
                # Multi-step draft decode: each step's backend has a different
                # speculative_step_id.  Bump seq_lens so the kernel sees KV
                # entries from prior draft steps.
                kv_len_bump = self.speculative_step_id + 1
                kv_lens = kv_lens + kv_len_bump
            query_width = 1

        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and bs > 0:
            if isinstance(seq_lens_cpu, list):
                slice_cpu = seq_lens_cpu[:bs]
                if slice_cpu:
                    max_kv_len = max(slice_cpu) + kv_len_bump
                    _validate_kv_len_capacity(max_kv_len, pages_per_seq * page_size)
            elif isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.numel() > 0:
                slice_cpu = seq_lens_cpu[:bs]
                if slice_cpu.numel() > 0:
                    max_kv_len = int(slice_cpu.max().item()) + kv_len_bump
                    _validate_kv_len_capacity(max_kv_len, pages_per_seq * page_size)
            else:
                # Empty seq_lens_cpu (e.g. idle rank forced into decode for DP
                # collectives) or unhandled type.
                pass

        req_pool_indices = forward_batch.req_pool_indices_cpu
        assert req_pool_indices is not None
        assert req_pool_indices.device.type == "cpu"
        assert req_pool_indices.shape[0] == bs
        req_pool_indices = req_pool_indices.to(torch.int64)

        cu_q_lens, cu_kv_lens, active_num_tokens = self._get_query_metadata(
            bs=bs,
            kv_stride=pages_per_seq * page_size,
            non_empty_bs=non_empty_bs,
            device=device,
            query_width=query_width,
            query_lens=query_lens,
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

        # Sliding-window attention uses a separate KV pool. Translate full-pool
        # page indices through the CPU mapping before uploading or reusing them.
        swa_page_indices: torch.Tensor | None = None
        if self._is_hybrid_swa:
            full_to_swa = self.token_to_kv_pool.full_to_swa_index_mapping
            assert full_to_swa is not None, (
                "Hybrid-SWA model is configured but full_to_swa_index_mapping "
                "has not been registered yet — did the allocator finish init?"
            )
            # The mapping uses token slots, so look up each page's first token.
            full_token_starts = page_indices.to(torch.int64) * page_size
            swa_page_indices = self._value_metadata(
                name="swa_page_indices",
                value=(full_to_swa[full_token_starts] // page_size).to(torch.int32),
                device=device,
            )

        # Page tables change with KV growth, so they are value-checked.
        return PallasForwardMetadata(
            kv_lens=kv_lens,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            page_indices=self._value_metadata(
                name="page_indices", value=page_indices, device=device
            ),
            distribution=distribution,
            active_num_tokens=active_num_tokens,
            swa_page_indices=swa_page_indices,
            decode_query_size=decode_query_size,
        )

    def _extend_query_lens(self, forward_batch, bs: int) -> tuple[int, ...]:
        """Per-request query lengths, read from the host mirror when complete."""
        host_lens = forward_batch.extend_seq_lens_cpu
        if isinstance(host_lens, torch.Tensor):
            assert host_lens.device.type == "cpu"
            host_lens = host_lens.tolist() if host_lens.dim() == 1 else None
        if not envs.SGLANG_TPU_ENABLE_HOST_QUERY_LENS.get():
            host_lens = None
        elif _is_fixed_width_eagle_refresh(forward_batch):
            # prepare_for_draft_extend supplies these counts; padding keeps them
            # complete. Missing entries indicate a producer or padding bug.
            assert forward_batch.extend_seq_lens.shape == (bs,)
            assert host_lens is not None and len(host_lens) == bs
        if host_lens is None or len(host_lens) != bs:
            # Padding clears incomplete mirrors instead of inventing counts.
            host_lens = forward_batch.extend_seq_lens.to("cpu", torch.int32).tolist()
        return tuple(host_lens)

    def _layout_metadata(self, *, key, build, device) -> torch.Tensor:
        cache = self._host_metadata_cache
        if cache is None:
            return build().to(device, non_blocking=True)
        return cache.get_by_layout(key=key, build=build, device=device)

    def _value_metadata(self, *, name, value, device) -> torch.Tensor:
        cache = self._host_metadata_cache
        if cache is None:
            return value.to(device, non_blocking=True)
        return cache.get_by_value(name=name, value=value, device=device)

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
        sinks: torch.Tensor | None = None,
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
        elif sliding_window is not None:
            # SGLang stores the left span, while RPA expects the total number
            # of attended tokens, including the current token.
            sliding_window += 1

        # For hybrid-SWA models, SWA layers index into the SWA sub-pool;
        # full-attention layers (or non-hybrid models) use the full-pool indices.
        is_swa_layer = sliding_window is not None
        if is_swa_layer and self.forward_metadata.swa_page_indices is not None:
            page_indices = self.forward_metadata.swa_page_indices
        else:
            page_indices = self.forward_metadata.page_indices

        soft_cap = layer.logit_cap if getattr(layer, "logit_cap", 0) else None
        # Same rule as the other attention backends; DFlash draft layers are
        # ENCODER_ONLY and attend bidirectionally within the block.
        use_causal_mask = not (
            layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY
        )

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

        draft_kv_lens = getattr(forward_batch.spec_info, "rpa_draft_kv_lens", None)
        kv_lens = (
            draft_kv_lens
            if draft_kv_lens is not None
            else self.forward_metadata.kv_lens
        )
        out = op(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
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
            self.forward_metadata.decode_query_size,
            use_causal_mask=use_causal_mask,
        )

        capture_mode = getattr(forward_batch, "capture_hidden_mode", None)
        # FULL capture retains padded rows for later EAGLE steps.
        if capture_mode is not None and capture_mode.is_full():
            out = _zero_inactive_rpa_rows(out, self.forward_metadata.active_num_tokens)

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


class BaseTpuPallasRpaMultiStepDraftBackend:
    """Wrap one RPA backend per EAGLE draft decode step.

    Each step's backend carries a different ``speculative_step_id`` so
    ``_build_metadata`` bumps ``seq_lens`` by the right offset, mirroring
    ``TpuPallasMultiStepDraftBackend`` for the flash backend. Subclasses set
    :attr:`attn_backend_cls` to the per-step backend to instantiate.
    """

    # Set by each subclass to the concrete per-step attention backend class.
    attn_backend_cls: ClassVar[type | None] = None

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
    ):
        if self.attn_backend_cls is None:
            raise ValueError(
                f"{type(self).__name__} must set a class-level attn_backend_cls."
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            self.attn_backend_cls(model_runner, speculative_step_id=i)
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
