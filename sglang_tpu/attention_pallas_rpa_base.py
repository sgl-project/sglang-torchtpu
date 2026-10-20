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
  :class:`sglang_tpu.kv_pool.TpuFusedMHATokenToKVPool`.
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
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
from sglang.srt.speculative.spec_info import SpecInputType
from sglang_tpu.attention_pallas_rpa_dcp import (
    DcpWriteSpans,
    broadcast_own_kv_heads,
    build_dcp_kernel_op,
    build_owner_write_spans,
    fused_kv_rows,
    gather_heads,
    merge_across_ranks,
    merge_lse,
    sink_fold,
    write_owned_kv,
)
from sglang_tpu.cp_forward import (
    TpuCpBatchLayout,
    all_gather_and_write_kv,
    get_tpu_cp_layout,
)
from sglang_tpu.cp_layout import TpuCpHostLayout
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.host_metadata_cache import HostMetadataCache
from sglang_tpu.kv_row_write import (
    get_write_rows_op,
    kv_write_segments,
    write_run_bound,
)
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

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
    # Kernel passes ``distribution`` can populate ("all" | "decode" | "mixed"),
    # chosen with it in _build_metadata. Only the MLA v2 kernel reads it.
    segments: str = "all"
    # Prefill context parallelism only (None otherwise). The kernel then runs
    # two rows per request (one per owned zigzag block) on this rank's local
    # token rows; see ``cp_layout``.
    # i32[3, write_run_bound(cp_size * local_tokens)]  (source row, KV slot,
    # length) runs of the rank-major all-gathered K/V rows, and i32[1] their count
    cp_write_segments: torch.Tensor | None = None
    cp_write_num_segments: torch.Tensor | None = None
    # The same rows as runs of SWA sub-pool slots (hybrid sliding-window
    # models only).
    cp_swa_write_segments: torch.Tensor | None = None
    cp_swa_write_num_segments: torch.Tensor | None = None
    # i64[max_num_seqs]  local row of each request's last token (0 when this
    # rank does not own it)
    cp_last_rows: torch.Tensor | None = None
    # i32[max_num_seqs]  1 where this rank owns the request's last token
    cp_last_owner_mask: torch.Tensor | None = None
    # Owned scalar i32: -1=native fallback, 0=static base, 1024/2048=promotion.
    # Backends without runtime tile selection leave this unset.
    prefill_kv_tile_hint: torch.Tensor | None = None
    # Decode context parallelism: all three are None unless the DCP group size
    # is > 1, in which case ``kv_lens`` holds this rank's shard of each sequence.
    # i32[max_num_seqs]  unsharded KV length per sequence
    dcp_kv_lens_global: torch.Tensor | None = None
    # i32[max_num_seqs]  this rank's shard of the KV cached before this step
    dcp_prefix_kv_lens: torch.Tensor | None = None
    # i32[1]  this rank's index inside its DCP group
    dcp_cp_rank: torch.Tensor | None = None
    # Compaction index and DMA spans of the rows this rank writes into the
    # full pool this step; under DCP the backend and not the kernel writes
    # the step's K/V.
    dcp_write: DcpWriteSpans | None = None
    # The same targets in the SWA sub-pool's slot space, set only for
    # hybrid-SWA models.
    dcp_swa_write: DcpWriteSpans | None = None


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


def _pool_kv_pack_heads(pool) -> int:
    """Heads the fused pool stores side by side in one row (``kv_pack_heads``), or 1."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    pools = (
        (pool.full_kv_pool, pool.swa_kv_pool)
        if isinstance(pool, SWAKVPool)
        else (pool,)
    )
    packs = {
        p.kv_pack_heads if isinstance(p, TpuFusedMHATokenToKVPool) else 1 for p in pools
    }
    if len(packs) != 1:
        raise NotImplementedError(f"SWA sub-pools pack KV heads differently: {packs}")
    return packs.pop()


def _pack_qkv_heads(q, k, v, pack: int):
    """q ``[T, Hq, d]``, k/v ``[T, pack, d]`` -> one kv head of width ``pack * d``.

    K/V rows are the heads side by side; each query head carries its q in its kv
    head's lanes and zeros elsewhere, so it scores only its own head's keys and its
    output lands in those lanes (``_unpack_out_heads``). Built with a one-hot
    multiply: torch_tpu returns wrong values for strided head slices joined by ``cat``.
    """
    T, hq, d = q.shape
    eye = torch.eye(pack, dtype=q.dtype, device=q.device).view(1, pack, 1, pack, 1)
    q = (q.reshape(T, pack, hq // pack, 1, d) * eye).reshape(T, hq, pack * d)
    return q, k.reshape(T, 1, pack * d), v.reshape(T, 1, pack * d)


def _unpack_out_heads(out, pack: int, d: int):
    """``[T, Hq, pack * d]`` -> ``[T, Hq, d]``: each head's own lanes (one-hot sum, exact)."""
    T, hq, _ = out.shape
    eye = torch.eye(pack, dtype=out.dtype, device=out.device).view(1, pack, 1, pack, 1)
    return (
        (out.reshape(T, pack, hq // pack, pack, d) * eye).sum(dim=3).reshape(T, hq, d)
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


def _import_dcp_helpers():
    """Return ``get_dcp_lens``, imported on first use.

    ``layers.dcp`` reaches back into ``distributed`` and ``runtime_context``;
    importing it from inside the DCP branch of ``__init__`` keeps this module's
    import graph the same whether or not DCP is configured.
    """
    from sglang.srt.layers.dcp.layout import get_dcp_lens

    return get_dcp_lens


def _check_dcp_kv_heads(*, layer: "RadixAttention", pool_kv_heads: int) -> None:
    """Refuse a layer whose K/V heads do not fit decode context parallelism.

    Every rank of a DCP group must compute the KV heads its pool holds, and the
    chunk pass groups the rank's own query heads over those KV heads.
    """
    if layer.tp_k_head_num != pool_kv_heads:
        raise ValueError(
            f"Layer {layer.layer_id} computes {layer.tp_k_head_num} KV heads per "
            f"rank, but its decode-context-parallel KV pool holds {pool_kv_heads}: "
            "the model must shard K/V over tp_size // dcp_size ranks, as gpt-oss "
            "and Qwen3.5 do; a model that shards K/V over all tp_size ranks runs "
            "only with --dcp-size <= tp_size // num_key_value_heads."
        )
    if layer.tp_q_head_num % layer.tp_k_head_num:
        raise ValueError(
            f"Layer {layer.layer_id} has {layer.tp_q_head_num} query heads per "
            f"rank over {layer.tp_k_head_num} KV heads: decode context parallelism "
            "needs a whole number of query heads per KV head on every rank; use a "
            "smaller --dcp-size."
        )


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
    :class:`sglang_tpu.kv_pool.TpuFusedMHATokenToKVPool`
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
        # Decode context parallelism (DCP): a group of ranks splits every
        # sequence's KV by absolute position (pos % dcp == rank), and the
        # allocator hands out slots from a pool widened by dcp, so one page
        # spans page_size * dcp logical tokens over page_size physical slots.
        parallel = get_parallel()
        self._dcp = parallel.attn_dcp_size
        self._dcp_rank = parallel.attn_dcp_rank if self._dcp > 1 else 0
        self._dcp_group = parallel.dcp_group if self._dcp > 1 else None
        self._page_stride = self.page_size * self._dcp
        self._cp_rank_t: torch.Tensor | None = None
        self._cp_rank_index: torch.Tensor | None = None
        self._cp_rank_mask: torch.Tensor | None = None
        self._dcp_op = None
        self._kv_write_op = None
        self._get_dcp_lens = None
        if self._dcp > 1:
            if self.req_to_token_pool.req_to_token.device.type != "cpu":
                raise ValueError(
                    "Decode context parallelism needs the host-resident "
                    "request-to-token table to build per-rank page tables; set "
                    "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1 (the default)."
                )
            cls = type(self)
            if (
                cls.forward_decode is not BaseTpuPallasRpaAttnBackend.forward_decode
                or cls.forward_extend is not BaseTpuPallasRpaAttnBackend.forward_extend
            ):
                raise ValueError(
                    f"{cls.__name__} runs its own attention forward, which reads "
                    "the DCP-sharded kv_lens and page table as if they were "
                    "unsharded; decode context parallelism needs "
                    "--attention-backend pallas_tpu_rpa_batched."
                )
            self._get_dcp_lens = _import_dcp_helpers()
            self._cp_rank_t = torch.tensor(
                [self._dcp_rank], dtype=torch.int32, device=self.device
            )
            # The rank enters the compiled forward as data (kernel scalar,
            # one-hot mask, index) so every rank traces one program.
            self._cp_rank_index = torch.tensor(
                [self._dcp_rank], dtype=torch.int64, device=self.device
            )
            self._cp_rank_mask = (
                torch.arange(self._dcp, dtype=torch.int64, device=self.device)
                == self._dcp_rank
            )
            self._dcp_op = build_dcp_kernel_op(self.kernel_spec)
            from sglang_tpu.kv_span_write import (
                get_kv_span_write_op,
            )

            self._kv_write_op = get_kv_span_write_op()

        max_bs = get_exec().graph.torch_compile_max_bs
        if max_bs:
            pages_per_seq = _get_pages_per_seq(self.max_context_len, self._page_stride)
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
        # Host req_to_token page-slot columns, built by _host_page_columns.
        self._host_page_cols: torch.Tensor | None = None
        # Register the custom op eagerly so the (Python-introspection-heavy)
        # ``jax_op`` registration does not happen inside a ``torch.compile``
        # traced region — dynamo can't handle ``inspect.signature`` on the
        # wrapped JAX function. The forward path then just looks the op up.
        self._op = self._build_kernel_op()
        if get_parallel().attn_cp_size > 1 and model_runner.device == "tpu":
            get_write_rows_op()

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

    def _dcp_local_kv_len(self, length: int) -> int:
        """Scalar form of ``layers.dcp.layout.get_dcp_lens``; identity without DCP.

        Monotone in ``length``, so the batch's longest sequence also holds the
        longest shard and the page-table capacity check can use it.
        """
        return length // self._dcp + (self._dcp_rank < length % self._dcp)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Land deferred HiCache reload commits into the KV buffers before any
        # layer reads KV; this hook runs per forward in eager AND compiled.
        # Use the backend's cached pool: upstream moved token_to_kv_pool off
        # ForwardBatch onto the attn backend (see self.token_to_kv_pool above).
        self.token_to_kv_pool.tpu_commit_pending_loads()

        cp_layout = get_tpu_cp_layout(forward_batch)
        if cp_layout is not None:
            self.forward_metadata = self._build_cp_metadata(forward_batch, cp_layout)
            return

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

    def warmup_distributions(self, max_live: int, device) -> None:
        """Warm fixed-shape metadata; live row counts are tensor values."""
        # Serving keys use tensor.device (e.g. tpu:0), not the runner's "tpu".
        device = torch.empty(0, device=device).device
        for is_decode in (False, True):
            self._get_cached_rpa_distribution(is_decode, max_live, device)

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
        if not tpu_envs.SGLANG_TPU_HOST_PADDING.get():
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

    def _kernel_segments(self, forward_batch: ForwardBatch, is_decode: bool) -> str:
        """Kernel passes the ``distribution`` built from ``is_decode`` can
        populate. A static kernel arg, so it may only depend on values fixed
        per compiled graph."""
        return "all"

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
        is_decode = forward_batch.forward_mode.is_decode() or decode_query_size > 1
        distribution = self._get_cached_rpa_distribution(
            is_decode, non_empty_bs, device
        )
        segments = self._kernel_segments(forward_batch, is_decode)
        bs = int(forward_batch.seq_lens.shape[0])
        req_to_token = self.req_to_token_pool.req_to_token
        pages_per_seq = _get_pages_per_seq(self.max_context_len, self._page_stride)

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
                segments,
            )

        if self._dcp > 1:
            raise ValueError(
                "Decode context parallelism needs the host-resident "
                "request-to-token table: the on-device metadata path reads the "
                "page table at the unsharded page stride. Set "
                "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1 (the default)."
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
            segments=segments,
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
        segments: str = "all",
    ) -> PallasForwardMetadata:
        """Build metadata on the CPU and reuse cached device copies when possible.

        CPU operations stay outside tracing, avoiding shape-specific recompiles.
        kv_lens stays on device; distribution is the caller's cached buffer.

        Under DCP ``kv_lens`` becomes this rank's shard of each sequence and the
        page table is read at the widened logical page stride.
        """
        if self._dcp > 1 and forward_batch.forward_mode not in (
            ForwardMode.DECODE,
            ForwardMode.EXTEND,
        ):
            raise ValueError(
                "Decode context parallelism supports plain decode and extend "
                f"batches only; got {forward_batch.forward_mode.name}. "
                "Speculative decoding, mixed batches and split prefill are not "
                "supported on the TPU DCP path."
            )

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
                    max_kv_len = self._dcp_local_kv_len(max(slice_cpu) + kv_len_bump)
                    _validate_kv_len_capacity(max_kv_len, pages_per_seq * page_size)
            elif isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.numel() > 0:
                slice_cpu = seq_lens_cpu[:bs]
                if slice_cpu.numel() > 0:
                    max_kv_len = self._dcp_local_kv_len(
                        int(slice_cpu.max().item()) + kv_len_bump
                    )
                    _validate_kv_len_capacity(max_kv_len, pages_per_seq * page_size)
            else:
                # Empty seq_lens_cpu (e.g. idle rank forced into decode for DP
                # collectives) or unhandled type.
                pass

        dcp_kv_lens_global: torch.Tensor | None = None
        dcp_prefix_kv_lens: torch.Tensor | None = None
        dcp_cp_rank: torch.Tensor | None = None
        seq_lens_host: torch.Tensor | None = None
        ext_host: torch.Tensor | None = None
        if self._dcp > 1:
            # The kernel derives the global query position of a local key from
            # the unsharded lengths, so it needs them alongside the shard. The
            # prefix shard excludes this step's own tokens: the extend path
            # attends those in a separate chunk-local pass. seq_lens_cpu holds
            # the live rows only; the bucket's padding rows stay at length 0.
            seq_lens_host = torch.as_tensor(
                list(seq_lens_cpu) if isinstance(seq_lens_cpu, list) else seq_lens_cpu,
                dtype=torch.int64,
            )[:bs]
            live = seq_lens_host.numel()
            ext_host = (
                torch.tensor(query_lens[:live], dtype=torch.int64)
                if query_lens is not None
                else torch.full((live,), query_width, dtype=torch.int64)
            )
            if forward_batch.forward_mode.is_decode():
                prefix_host = seq_lens_host
            else:
                prefix_host = seq_lens_host - ext_host

            def _padded(values: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(bs, dtype=torch.int32)
                out[:live] = values.to(torch.int32)
                return out.to(device, non_blocking=True)

            kv_lens = _padded(
                self._get_dcp_lens(seq_lens_host, self._dcp, self._dcp_rank)
            )
            dcp_kv_lens_global = _padded(seq_lens_host)
            dcp_prefix_kv_lens = _padded(
                self._get_dcp_lens(prefix_host, self._dcp, self._dcp_rank)
            )
            dcp_cp_rank = self._cp_rank_t

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
        # slot per page), not the full [bs, max_ctx] row.
        page_cols = self._host_page_columns(pages_per_seq, self._page_stride)
        per_seq = req_to_token[req_pool_indices[:, None], page_cols[None, :]]
        page_indices = (per_seq // self._page_stride).to(torch.int32)
        page_indices = page_indices.reshape(bs * pages_per_seq).contiguous()

        # Sliding-window attention uses a separate KV pool. Translate full-pool
        # page indices through the CPU mapping before uploading or reusing them.
        swa_page_indices: torch.Tensor | None = None
        write_spans: DcpWriteSpans | None = None
        swa_write_spans: DcpWriteSpans | None = None
        step_slots: torch.Tensor | None = None
        step_positions: torch.Tensor | None = None
        num_tokens = 0
        if self._dcp > 1:
            # The slots and positions of this step's tokens come from the host
            # request-to-token table, so the owner-masked write targets are
            # built here and shipped with the rest of the metadata, padded to
            # the token bucket like the batch's own token fields.
            starts = (seq_lens_host - ext_host).tolist()
            counts = ext_host.tolist()
            step_positions = torch.cat(
                [
                    torch.arange(s, s + n, dtype=torch.int64)
                    for s, n in zip(starts, counts)
                ]
                + [torch.zeros(0, dtype=torch.int64)]
            )
            rows = torch.repeat_interleave(
                req_pool_indices[: ext_host.numel()], ext_host
            )
            step_slots = req_to_token[rows, step_positions].to(torch.int64)
            padded = getattr(forward_batch, "out_cache_loc", None)
            num_tokens = (
                padded.shape[0] if padded is not None else step_positions.numel()
            )
            write_spans = build_owner_write_spans(
                step_positions,
                step_slots,
                self._dcp,
                self._dcp_rank,
                page_size,
                num_tokens,
                bs,
            ).to_device(device)
        if self._is_hybrid_swa:
            full_to_swa = self.token_to_kv_pool.full_to_swa_index_mapping
            assert full_to_swa is not None, (
                "Hybrid-SWA model is configured but full_to_swa_index_mapping "
                "has not been registered yet — did the allocator finish init?"
            )
            # The mapping uses token slots, so look up each page's first token.
            full_token_starts = page_indices.to(torch.int64) * self._page_stride
            swa_page_indices = self._value_metadata(
                name="swa_page_indices",
                value=(full_to_swa[full_token_starts] // self._page_stride).to(
                    torch.int32
                ),
                device=device,
            )
            if self._dcp > 1:
                # SWA layers write into the SWA sub-pool, whose slots the
                # mapping gives for the same tokens.
                swa_slots = full_to_swa[step_slots.to(full_to_swa.device)].to(
                    torch.int64
                )
                swa_write_spans = build_owner_write_spans(
                    step_positions,
                    swa_slots,
                    self._dcp,
                    self._dcp_rank,
                    page_size,
                    num_tokens,
                    bs,
                ).to_device(device)

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
            dcp_kv_lens_global=dcp_kv_lens_global,
            dcp_prefix_kv_lens=dcp_prefix_kv_lens,
            dcp_cp_rank=dcp_cp_rank,
            dcp_write=write_spans,
            dcp_swa_write=swa_write_spans,
            decode_query_size=decode_query_size,
            segments=segments,
        )

    def _host_page_columns(self, pages_per_seq: int, page_size: int) -> torch.Tensor:
        """Cached ``[pages_per_seq]`` req_to_token columns, one per page."""
        page_cols = self._host_page_cols
        if page_cols is None or page_cols.numel() != pages_per_seq:
            page_cols = torch.arange(
                0, pages_per_seq * page_size, page_size, dtype=torch.long
            )
            self._host_page_cols = page_cols
        return page_cols

    def _build_cp_metadata(
        self, forward_batch: ForwardBatch, cp_layout: TpuCpBatchLayout
    ) -> PallasForwardMetadata:
        """Metadata for a context-parallel extend: a row per non-empty owned block.

        Each request's early zigzag block and then its late block, each with
        ``kv_len`` = the block's absolute end and ``q_len`` = the block length
        over the request's page row, so the kernel reads the cached prefix and
        earlier blocks and writes the block's own K/V at its positions. The
        remaining rows are zero rows at the tail, outside ``distribution``.
        Built on the host from the CPU lengths and the host-resident
        req_to_token, then uploaded once.
        """
        req_to_token = self.req_to_token_pool.req_to_token
        if req_to_token.device.type != "cpu":
            raise RuntimeError(
                "Prefill context parallelism on TPU needs a host-resident "
                "req_to_token (SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1)."
            )
        host = cp_layout.host
        page_size = self.page_size
        pages_per_seq = _get_pages_per_seq(self.max_context_len, page_size)
        num_rows = 2 * host.pad_b
        table_bytes = _page_table_smem_bytes(num_rows, pages_per_seq)
        if table_bytes > _SCOPED_SMEM_BYTES:
            raise ValueError(
                f"RPA page table for {num_rows} context-parallel rows x "
                f"{pages_per_seq} pages needs {table_bytes / 2**20:.2f} MiB of "
                f"SMEM, above the {_SCOPED_SMEM_BYTES / 2**20:.0f} MiB budget; "
                "lower --prefill-max-requests or raise --page-size."
            )
        _validate_kv_len_capacity(
            int(host.pseudo_kv_lens.max()), pages_per_seq * page_size
        )
        device = forward_batch.seq_lens.device

        kv_lens = torch.from_numpy(host.pseudo_kv_lens).to(torch.int32)
        cu_q_lens = torch.zeros(num_rows + 1, dtype=torch.int32)
        cu_q_lens[1:] = torch.cumsum(
            torch.from_numpy(host.pseudo_q_lens), dim=0, dtype=torch.int32
        )
        cu_kv_lens = torch.arange(num_rows + 1, dtype=torch.int32) * (
            pages_per_seq * page_size
        )

        req_pool_indices = forward_batch.req_pool_indices_cpu
        assert req_pool_indices is not None
        assert req_pool_indices.device.type == "cpu"
        assert req_pool_indices.shape[0] == host.pad_b
        req_pool_indices = req_pool_indices.to(torch.int64)
        page_cols = self._host_page_columns(pages_per_seq, page_size)
        per_seq = req_to_token[req_pool_indices[:, None], page_cols[None, :]]
        page_indices = (per_seq // page_size).to(torch.int32)
        page_indices = (
            page_indices[torch.from_numpy(host.row_reqs)]
            .reshape(num_rows * pages_per_seq)
            .contiguous()
        )

        write_segments, write_num_segments, dest_slots, valid = self._cp_write_runs(
            host=host, req_to_token=req_to_token, req_pool_indices=req_pool_indices
        )
        swa_page_indices, swa_write_segments, swa_write_num_segments = (
            self._cp_swa_write_runs(
                page_indices=page_indices,
                dest_slots=dest_slots,
                valid=valid,
                max_segments=write_segments.shape[1],
                device=device,
            )
        )

        distribution = self._get_cached_rpa_distribution(False, host.num_rows, device)
        cu_q_lens = cu_q_lens.to(device, non_blocking=True)
        return PallasForwardMetadata(
            kv_lens=kv_lens.to(device, non_blocking=True),
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens.to(device, non_blocking=True),
            page_indices=page_indices.to(device, non_blocking=True),
            distribution=distribution,
            active_num_tokens=_make_active_num_tokens(cu_q_lens, host.num_rows),
            swa_page_indices=swa_page_indices,
            cp_write_segments=write_segments.to(device, non_blocking=True),
            cp_write_num_segments=write_num_segments.to(device, non_blocking=True),
            cp_swa_write_segments=(
                None
                if swa_write_segments is None
                else swa_write_segments.to(device, non_blocking=True)
            ),
            cp_swa_write_num_segments=(
                None
                if swa_write_num_segments is None
                else swa_write_num_segments.to(device, non_blocking=True)
            ),
            cp_last_rows=torch.from_numpy(host.last_rows).to(device, non_blocking=True),
            cp_last_owner_mask=torch.from_numpy(host.last_owner_mask)
            .to(torch.int32)
            .to(device, non_blocking=True),
        )

    def _cp_write_runs(
        self,
        *,
        host: TpuCpHostLayout,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-pool write runs of the rank-major all-gathered rows and their
        i32[1] count, plus each row's KV slot (0 for padding) and validity."""
        # KV slot of every all-gathered row: the request's req_to_token entry
        # at prefix + offset; padding rows are not written.
        gathered_req = torch.from_numpy(host.gathered_req)
        gathered_pos = torch.from_numpy(host.gathered_pos)
        valid = gathered_req >= 0
        req = gathered_req.clamp(min=0)
        prefix = torch.from_numpy(host.prefix_lens)[req]
        slots = req_to_token[req_pool_indices[req], prefix + gathered_pos.clamp(min=0)]
        dest_slots = torch.where(valid, slots.to(torch.int64), torch.zeros_like(req))
        # Padding rows point at slot 0, the reserved page, and are not written.
        # The run table is prefetched into SMEM, so it is sized for a served
        # batch (a run per page and per zigzag block) rather than per row.
        max_segments = write_run_bound(
            slots.shape[0],
            page_size=self.page_size,
            blocks=2 * host.cp_size * host.pad_b,
        )
        write_segments, num_segments = kv_write_segments(
            dest_slots=slots.numpy(),
            valid=(valid & (slots > 0)).numpy(),
            max_segments=max_segments,
        )
        write_num_segments = torch.tensor([num_segments], dtype=torch.int32)
        return write_segments, write_num_segments, dest_slots, valid

    def _cp_swa_write_runs(
        self,
        *,
        page_indices: torch.Tensor,
        dest_slots: torch.Tensor,
        valid: torch.Tensor,
        max_segments: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """SWA page rows, write runs and i32[1] run count; all None unless hybrid-SWA.

        Sliding-window layers read and write the SWA sub-pool, so page rows and
        write slots go through the host-resident full-to-SWA mapping.
        """
        if not self._is_hybrid_swa:
            return None, None, None
        page_size = self.page_size
        full_to_swa = self.token_to_kv_pool.full_to_swa_index_mapping
        assert full_to_swa is not None, (
            "Hybrid-SWA model is configured but full_to_swa_index_mapping "
            "has not been registered yet; did the allocator finish init?"
        )
        full_token_starts = page_indices.to(torch.int64) * page_size
        swa_page_indices = (
            (full_to_swa[full_token_starts] // page_size)
            .to(torch.int32)
            .to(device, non_blocking=True)
        )
        translated = full_to_swa[dest_slots].to(torch.int64)
        # An unmapped entry reads 0, the reserved padding slot; warmup batches
        # point every row at full slot 0 and have nothing mapped.
        if bool((translated[valid & (dest_slots > 0)] <= 0).any()):
            raise RuntimeError(
                "context-parallel extend rows without an SWA slot: the "
                "sliding-window allocator did not map every token of the chunk."
            )
        swa_write_segments, swa_num_segments = kv_write_segments(
            dest_slots=translated.numpy(),
            valid=(valid & (translated > 0)).numpy(),
            max_segments=max_segments,
        )
        swa_write_num_segments = torch.tensor([swa_num_segments], dtype=torch.int32)
        return swa_page_indices, swa_write_segments, swa_write_num_segments

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

    def _kv_lens(self, forward_batch: ForwardBatch) -> torch.Tensor:
        # The fused draft loop runs every step on step 0's metadata and passes
        # each step's KV length in-graph.
        draft_kv_lens = getattr(forward_batch.spec_info, "rpa_draft_kv_lens", None)
        if draft_kv_lens is not None:
            return draft_kv_lens
        return self.forward_metadata.kv_lens

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
        # Packed pools (SGLANG_TPU_KV_PACK_HEADS) hold this rank's kv heads side by side.
        pack = _pool_kv_pack_heads(self.token_to_kv_pool)
        if pack > 1:
            if self._dcp != 1 or pack != layer.tp_k_head_num:
                raise NotImplementedError(
                    "packed KV heads need dcp 1 and all local heads packed"
                )
            q, k, v = _pack_qkv_heads(q, k, v, pack)

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
        uses_swa_pool = (
            is_swa_layer and self.forward_metadata.swa_page_indices is not None
        )
        if uses_swa_pool:
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
        quantize_kv = None
        if self.kv_cache_quant_dtype is not None:
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
            quant_dtype = self.kv_cache_quant_dtype

            def quantize_kv(k_in, v_in):
                return (
                    _quantize_kv_to_fp8(k_in, k_scale, quant_dtype),
                    _quantize_kv_to_fp8(v_in, v_scale, quant_dtype),
                )

            k_compute, v_compute = k, v
            k, v = quantize_kv(k, v)
        else:
            k_scale = None
            v_scale = None

        # Prefill context parallelism: every rank writes the whole sequence's
        # K/V for this layer into its own pool before the kernel reads the
        # earlier blocks through the page table.
        cp_write_segments = self.forward_metadata.cp_write_segments
        cp_write_num_segments = self.forward_metadata.cp_write_num_segments
        if is_swa_layer and self.forward_metadata.cp_swa_write_segments is not None:
            cp_write_segments = self.forward_metadata.cp_swa_write_segments
            cp_write_num_segments = self.forward_metadata.cp_swa_write_num_segments
        if cp_write_segments is not None:
            all_gather_and_write_kv(
                kv_cache=kv_cache,
                k=k if quantize_kv is None else k_compute,
                v=v if quantize_kv is None else v_compute,
                write_segments=cp_write_segments,
                write_num_segments=cp_write_num_segments,
                group=get_parallel().attn_cp_group,
                quantize=quantize_kv,
            )

        kv_lens = self._kv_lens(forward_batch)
        if self._dcp > 1:
            if not use_causal_mask:
                raise NotImplementedError(
                    "Decode context parallelism supports only causal attention."
                )
            pool = self.token_to_kv_pool
            _check_dcp_kv_heads(
                layer=layer,
                pool_kv_heads=(pool.swa_kv_pool if uses_swa_pool else pool).head_num,
            )
            # SWA layers of a hybrid model write into the SWA sub-pool, whose
            # slots the metadata builder translated from the full-pool ones.
            spans = (
                self.forward_metadata.dcp_swa_write
                if uses_swa_pool
                else self.forward_metadata.dcp_write
            )
            out = self._forward_dcp(
                q,
                k,
                v,
                kv_cache,
                page_indices,
                spans,
                forward_batch,
                sinks,
                float(layer.scaling),
                sliding_window,
                soft_cap,
                k_scale,
                v_scale,
            )
        else:
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
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)
        if pack > 1:
            out = _unpack_out_heads(out, pack, layer.qk_head_dim)
        out = out[..., : layer.v_head_dim]
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        page_indices: torch.Tensor,
        spans: DcpWriteSpans,
        forward_batch: ForwardBatch,
        sinks: torch.Tensor | None,
        sm_scale: float,
        sliding_window: int | None,
        soft_cap: float | None,
        k_scale: float | None,
        v_scale: float | None,
    ) -> torch.Tensor:
        """One layer of attention over a position-sharded KV cache, ``[T, H, D]``.

        The rank writes only the tokens it owns and the kernel is told to leave
        the cache alone, so the step's K/V reaches exactly one shard. The kernel
        then runs on this rank's shard with the whole group's query heads, and
        ``merge_across_ranks`` merges the partial softmaxes and hands each rank
        back its own heads.

        Extend takes a second pass because the chunk's own tokens are spread
        across the group yet have to be attended causally: the chunk is attended
        locally, uncached, and merged into the prefix result by log-sum-exp. The
        sink belongs to whichever pass covers every key, so it is applied once.

        The cached passes hand the kernel the group's gathered query heads
        against the full replicated KV block. The chunk pass keeps this rank's
        query heads, whose KV is one slice of that block, so it takes the block
        rebuilt out of that slice alone.
        """
        md = self.forward_metadata
        cp = self._dcp
        group = self._dcp_group

        write_owned_kv(
            self._kv_write_op, kv_cache, fused_kv_rows(k, v, kv_cache), spans
        )

        q_all = gather_heads(q.contiguous(), group, cp, self._cp_rank_mask)
        if forward_batch.forward_mode.is_decode():
            out_p, lse_p = self._dcp_op(
                q_all,
                k,
                v,
                kv_cache,
                md.kv_lens,
                page_indices,
                md.cu_q_lens,
                md.distribution,
                md.dcp_kv_lens_global,
                md.dcp_cp_rank,
                None,  # attention_sink: folded in after the merge
                sm_scale,
                sliding_window,
                soft_cap,
                k_scale,
                v_scale,
                True,  # cache_only
                True,  # skip_kv_write
                cp,  # cp_group_size
            )
            out, lse = merge_across_ranks(
                out_p.float(),
                lse_p,
                group,
                cp,
                self._cp_rank_mask,
                self._cp_rank_index,
            )
            if sinks is not None:
                out = sink_fold(out, lse, sinks)
            return out.to(q.dtype)

        out_prefix, lse_prefix = self._dcp_op(
            q_all,
            k,
            v,
            kv_cache,
            md.dcp_prefix_kv_lens,
            page_indices,
            md.cu_q_lens,
            md.distribution,
            md.dcp_kv_lens_global,
            md.dcp_cp_rank,
            None,  # attention_sink: folded in by the chunk pass below
            sm_scale,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
            True,  # cache_only
            True,  # skip_kv_write
            cp,  # cp_group_size
        )
        out_prefix, lse_prefix = merge_across_ranks(
            out_prefix.float(),
            lse_prefix,
            group,
            cp,
            self._cp_rank_mask,
            self._cp_rank_index,
        )

        # kv_len == q_len makes the kernel take every key from the keys/values
        # operands, so the page table is never followed and zeros will do.
        q_lens = md.cu_q_lens[1:] - md.cu_q_lens[:-1]
        k_own, v_own = broadcast_own_kv_heads(k, v, cp, self._cp_rank_index)
        out_chunk, lse_chunk = self._dcp_op(
            q,
            k_own,
            v_own,
            kv_cache,
            q_lens,
            torch.zeros_like(page_indices),
            md.cu_q_lens,
            md.distribution,
            md.dcp_kv_lens_global,  # unused: this pass is not context-parallel
            md.dcp_cp_rank,
            sinks,
            sm_scale,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
            False,  # cache_only
            True,  # skip_kv_write
            0,  # cp_group_size
        )
        out, _lse = merge_lse(out_prefix, lse_prefix, out_chunk.float(), lse_chunk)
        return out.to(q.dtype)

    # ------------------------------------------------------------------ #
    # Capability flags
    # ------------------------------------------------------------------ #

    def support_triton(self) -> bool:
        return False

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0


class TpuPallasRpaMultiStepDraftBackend:
    """Wrap one attention backend per EAGLE draft decode step.

    Each step's backend carries a different ``speculative_step_id`` so
    ``_build_metadata`` bumps ``seq_lens`` by the right offset.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
        attn_backend_cls: type[BaseTpuPallasRpaAttnBackend],
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            attn_backend_cls(model_runner, speculative_step_id=i)
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
