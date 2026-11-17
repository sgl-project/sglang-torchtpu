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

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_kpool,
    get_dsa_index_topk,
    get_dsa_mtp_topk_width,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
)
from sglang_tpu.attention_pallas_rpa_base import _get_pages_per_seq
from sglang_tpu.attention_pallas_rpa_dcp import (
    _all_reduce_gather,
    gather_heads,
    merge_across_ranks,
)
from sglang_tpu.dcp_prefill_scratch import SlotState, plan_scratch_slots
from sglang_tpu.dsa_dcp_ops import (
    build_dsa_index_topk_dcp_op,
    build_dsa_topk_merge_dcp_op,
    build_index_key_store_gather_dcp_op,
    build_sparse_mla_attention_dcp_op,
    build_sparse_mla_kv_store_gather_dcp_op,
)
from sglang_tpu.dsa_indexer_tpu import (
    build_dsa_index_topk_op,
    build_dsa_index_topk_split_op,
    build_dsa_kpool_expand_op,
    build_dsa_kpool_topk_op,
    build_dsa_kpool_topk_split_op,
    indexer_forward,
    kpool_indexer_forward,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_pool import _zeros_kv_buffer
from sglang_tpu.oneshot_collectives import get_dcp_group_all_gather_op
from sglang_tpu.pallas_export import (
    argnums,
    tpu_info_for_cpu_export,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


_PALLAS_SPARSE_MLA_OPS = {}


def topk_row_width(hf_config) -> int:
    """Columns of the top-k rows index_topk / index_topk_kpool return."""
    if get_dsa_index_kpool(hf_config) == 1:
        return get_dsa_index_topk(hf_config)
    # Pooled tokens plus the open pool's tail, padded with -1 to whole lanes for
    # the sparse kernel's gather (SparseCore RoPE packing).
    return -(-get_dsa_mtp_topk_width(hf_config) // 128) * 128


def _build_pallas_sparse_mla_kernel_op(
    qblock_kpool: int = 0,
    gather_chunk_tokens: Optional[int] = None,
    zero_rope: bool = False,
    window: bool = False,
):
    """Register the sparse MLA step as a torch custom op, once per process.

    The op takes the same arguments as the ``pallas_tpu_mla`` op, with the two
    cache buffers in place of the fused one, followed by the indexer's top-k
    rows. ``cu_kv_lens`` and ``v_scale`` are accepted but not used: the sparse
    kernels find pages with a fixed stride, and they use one scale for both
    keys and values.

    ``zero_rope`` neither writes nor reads the rope cache, for a model whose
    rope lane is identically zero (see ``sparse_mla_attention_fused``).

    ``window`` (SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT) builds
    ``sparse_mla_prefill_window`` instead: the queries and rows are one window
    of the batch, and the ``cu_kv_lens`` slot carries the window's cu_q_lens.
    """
    key = (qblock_kpool, gather_chunk_tokens, zero_rope, window)
    if key in _PALLAS_SPARSE_MLA_OPS:
        return _PALLAS_SPARSE_MLA_OPS[key]

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.mla_sparse_fused import (
        sparse_mla_attention_fused,
        sparse_mla_prefill_fused_tokens,
        sparse_mla_prefill_window,
    )

    op_name = "sglang_tpu_pallas::sparse_mla_attention_fused"
    if window:
        op_name = "sglang_tpu_pallas::sparse_mla_prefill_window"
    chunk_kwargs = {}
    if gather_chunk_tokens is not None:
        # The chunk size changes the op body; naming it keeps cached graphs apart.
        op_name += f"_g{gather_chunk_tokens}"
        chunk_kwargs["gather_chunk_tokens"] = gather_chunk_tokens
    if qblock_kpool:
        # qblock_kpool > 0 sends prefill batches through the query-blocked kernel
        # (sparse_mla_attention_fused); it changes the body, so it is in the name.
        # qblock_kpool == 1 (plain DSA) also takes and returns head-major
        # queries and output, [heads, tokens, ...] (_forward_mla).
        op_name += f"_qblock{qblock_kpool}"
    if zero_rope:
        # No rope cache write or read: a different body, so a different name.
        op_name += "_zero_rope"
        chunk_kwargs["zero_rope"] = True

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
        if window:
            return sparse_mla_prefill_window(
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
                topk_indices,
                sm_scale=sm_scale,
                k_scale=k_scale,
            )
        if qblock_kpool == 1:
            return sparse_mla_prefill_fused_tokens(
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
            qblock_kpool=qblock_kpool,
            **chunk_kwargs,
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

    _PALLAS_SPARSE_MLA_OPS[key] = custom_op
    return custom_op


def _has_zero_rope(hf_config) -> bool:
    """Whether the model has no rope (``qk_rope_head_dim == 0``, GLM-5.3-Flash).

    Such a model attends with an identically-zero rope lane
    (``_mla_kernel_args``), so the rope cache, allocated zeroed, never needs
    writing or reading.
    """
    rope_dim = getattr(hf_config, "qk_rope_head_dim", None)
    if rope_dim is None:
        text_config = getattr(hf_config, "text_config", None) or hf_config
        rope_dim = getattr(text_config, "qk_rope_head_dim", None)
    return rope_dim == 0


def _kpool_query_bound(metadata, *, num_tokens: int) -> int:
    """Static per-request new-token bound for fused kpool pooling.

    Decode carries one row per request. Verify and draft extend carry exactly
    ``decode_query_size`` rows when it is above 1: _forward_decode_query_size
    only reports it for uniform batches, and it is already a compiled-graph
    constant for the attention kernel. Ragged extend keeps the token bucket.
    """
    if metadata.decode_query_size > 1 or metadata.segments == "decode":
        return min(metadata.decode_query_size, num_tokens)
    return num_tokens


# Token split: fewer rows per rank than this would make the SparseCore top-k
# pick a different partition than the unsplit call (_pick_partition depends
# on the row count below 32), so rows could be ordered or tie-broken
# differently. The split is skipped there.
_TOKEN_SPLIT_MIN_ROWS_PER_RANK = 32


def _assemble_split_rows(
    local_rows: torch.Tensor, rank: torch.Tensor, world: int, all_reduce
) -> torch.Tensor:
    """Put the ranks' ``i32[R, topk]`` row slices together in rank order.

    An all-reduce (sum), not an all-gather: each rank writes its slice plus 1
    (the -1 padding becomes 0) into its own slot of a zero ``[world, R, topk]``
    buffer, and the sum minus 1 is every rank's slice in rank order. Exact on
    int32. XLA's all-gather of these rows (32 MiB at T=4096) in the compiled
    prefill graph delivered wrong rows on TPU (13L TP8 and 93L TP32), with and
    without the SparseCore all-gather offload; this all-reduce was bit-exact
    against the unsplit indexer on both.
    """
    mine = (torch.arange(world, device=local_rows.device) == rank).view(world, 1, 1)
    buf = torch.where(
        mine, (local_rows + 1).unsqueeze(0), torch.zeros_like(local_rows).unsqueeze(0)
    ).to(torch.int32)
    buf = all_reduce(buf)
    return (buf - 1).reshape(world * local_rows.shape[0], local_rows.shape[1])


def _split_pool(pool):
    """The TPU split MLA pool behind a hybrid linear-attention model's pool."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    return pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool


# dcp_gather_buckets() entry that warms the scratch prefill instead of a gather.
_DCP_SCRATCH_BUCKET = -1


class TpuPallasSparseMlaAttnBackend(TpuPallasMlaAttnBackend):
    """``pallas_tpu_mla`` with sparse attention: it runs the DSA indexer too."""

    # Set by _build_kernel_op under the q-block opt-in.
    _prefill_op = None
    # Set by the compile policy to warm one dcp_gather_buckets() length.
    _dcp_gather_bucket_override: Optional[int] = None
    # Under DCP this backend shards the latents and index keys itself and
    # merges the partial attention (_forward_mla_dcp).
    _dcp_owns_forward: ClassVar[bool] = True

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        hf_config = model_runner.model_config.hf_config
        # Set before super().__init__(), which calls _build_kernel_op().
        self._kpool = get_dsa_index_kpool(hf_config)
        self._zero_rope = _has_zero_rope(hf_config)
        super().__init__(model_runner, speculative_step_id)
        self._topk = get_dsa_index_topk(hf_config)
        self.raw_ring_pages_cnt = 0
        if self._kpool > 1:
            self._kpool_width = topk_row_width(hf_config)
            self._index_topk_op = build_dsa_kpool_topk_op()
            pool = _split_pool(self.token_to_kv_pool)
            self.raw_ring_pages_cnt = pool.raw_ring_pages_cnt
            pool.attach_raw_ring(self.req_to_token_pool.req_to_token.shape[0])
        else:
            self._index_topk_op = build_dsa_index_topk_op()
        # Prefill under DCP: gather an unsharded cache, or the decode path.
        self._dcp_gathers_prefill = (
            self._dcp > 1 and not tpu_envs.SGLANG_TPU_DCP_PREFILL_HEAD_PARALLEL.get()
        )
        self._dcp_all_gather = (
            self._dcp_gathers_prefill
            and tpu_envs.SGLANG_TPU_DCP_PREFILL_ALL_GATHER.get()
        )
        # The DMA all-gather addresses peers by position on the attention-TP mesh.
        # Built here: the op registry takes a lock, which traced code cannot.
        self._dcp_all_gather_op = (
            get_dcp_group_all_gather_op(get_parallel().attn_tp_size, self._dcp)
            if self._dcp_all_gather
            else None
        )
        self._dcp_scratch_slots = (
            tpu_envs.SGLANG_TPU_DCP_PREFILL_SCRATCH.get()
            if self._dcp_gathers_prefill
            else 0
        )
        self._dcp_scratch_state: SlotState = [None] * self._dcp_scratch_slots
        # layer id -> (NoPE, RoPE, index keys or None), unsharded, in place.
        self._dcp_scratch: dict[int, tuple] = {}
        if self._dcp > 1:
            self._init_dcp_ops()
        # The head-parallel DCP prefill runs the sharded decode ops: no token split.
        self._init_indexer_token_split(
            enabled=not self._dcp > 1 or self._dcp_gathers_prefill
        )
        self._init_attention_token_split()

    def _init_attention_token_split(self) -> None:
        """Set up SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT (on the indexer split)."""
        self._attn_split_world = 0
        if not tpu_envs.SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT.get():
            return
        import torch.distributed as dist

        from sglang.srt.runtime_context import get_parallel

        # The all-to-all addresses peers by mesh position, which is the global
        # rank (get_pallas_mesh is rank-aligned): the attention-TP group must be
        # the whole world, in order. The pooled indexer has no local rows.
        world = dist.get_world_size()
        if (
            self._kpool > 1
            or self._idx_split_group is None
            or self._idx_split_size != world
            or get_parallel().attn_tp_rank != dist.get_rank()
        ):
            raise RuntimeError(
                "SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT needs plain DSA "
                "(index kpool 1) and an attention-TP group that spans the "
                "world in rank order."
            )
        from sglang_tpu.oneshot_collectives import get_oneshot_all_to_all_op

        get_oneshot_all_to_all_op(world)  # build before torch.compile traces
        self._window_op = _build_pallas_sparse_mla_kernel_op(window=True)
        self._attn_split_world = world

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.raw_ring_pages_cnt:
            self.forward_metadata.raw_page_base = self._raw_page_base(forward_batch)
        if self._dcp_gathers_prefill and not forward_batch.forward_mode.is_decode():
            self._add_dcp_gather_metadata(forward_batch)

    def _raw_page_base(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """i32[rows]: each row's first raw-ring page, its request slot times the ring."""
        from sglang_tpu.cp_forward import get_tpu_cp_layout

        if get_tpu_cp_layout(forward_batch) is not None:
            raise NotImplementedError(
                "The pooled indexer's raw ring addresses one row per request; "
                "prefill context parallelism runs two pseudo-rows per request."
            )
        rows = forward_batch.req_pool_indices_cpu
        if rows is None:
            rows = forward_batch.req_pool_indices
        base = rows.to(torch.int32) * self.raw_ring_pages_cnt
        return base.to(forward_batch.seq_lens.device, non_blocking=True)

    def _raw_ring_args(self) -> tuple[torch.Tensor, int]:
        """(raw_page_base, raw_ring_pages_cnt) for the pooled indexer ops. Without a
        ring the op ignores the base, so kv_lens stands in as an i32[rows] input."""
        metadata = self.forward_metadata
        if self.raw_ring_pages_cnt:
            return metadata.raw_page_base, self.raw_ring_pages_cnt
        return metadata.kv_lens, 0

    def _init_dcp_ops(self) -> None:
        """Ops of the DCP path. The latents and index keys are page-striped:
        rank r holds the r-th page_size run of every widened page."""
        if self._kpool > 1:
            raise ValueError(
                "Decode context parallelism supports plain DSA only, not the "
                f"pooled indexer (index_kpool={self._kpool})."
            )
        # The DCP path scores only the shard; the token split has nothing to do.
        self._idx_split_group = None
        self._dcp_interleave = self.page_size
        self._index_topk_dcp_op = build_dsa_index_topk_dcp_op(
            self._dcp, self._dcp_interleave
        )
        self._topk_merge_dcp_op = build_dsa_topk_merge_dcp_op(
            self._dcp, self._dcp_interleave
        )
        self._attention_dcp_op = build_sparse_mla_attention_dcp_op(
            self._dcp, self._dcp_interleave
        )
        self._index_key_store_gather_op = build_index_key_store_gather_dcp_op(
            self._dcp, self._dcp_interleave
        )
        self._kv_store_gather_op = build_sparse_mla_kv_store_gather_dcp_op(
            self._dcp, self._dcp_interleave
        )

    def _init_indexer_token_split(self, *, enabled: bool) -> None:
        """Set up the DSA indexer token split (on with attn-TP > 1 unless disabled)."""
        self._idx_split_group = None
        self._idx_split_size = 1
        self._idx_split_min_tokens = 0
        self._index_topk_split_op = None
        self._kpool_expand_op = None
        if not enabled:
            return
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        group = parallel.attn_tp_group
        if group.world_size <= 1:
            return
        # Each rank scores the rows of slot attn_tp_rank, and the all-reduce
        # lays the slots out in group-rank order: the two must agree.
        if group.rank_in_group != parallel.attn_tp_rank:
            raise RuntimeError(
                "DSA indexer token split: attention-TP rank "
                f"{parallel.attn_tp_rank} is rank {group.rank_in_group} of its "
                "group; the gathered rows would be out of order."
            )
        self._idx_split_group = group
        self._idx_split_size = group.world_size
        self._idx_split_min_tokens = max(
            tpu_envs.SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT_MIN_TOKENS.get(),
            _TOKEN_SPLIT_MIN_ROWS_PER_RANK * group.world_size,
        )
        if self._kpool > 1:
            # Pooled indexer: the ranks exchange picked pools (topk // kpool
            # per row), not token rows, and expand them after the all-reduce.
            self._index_topk_split_op = build_dsa_kpool_topk_split_op()
            self._kpool_expand_op = build_dsa_kpool_expand_op()
        else:
            self._index_topk_split_op = build_dsa_index_topk_split_op()

    def _indexer_split_rows(self, forward_batch: ForwardBatch, num_tokens: int) -> int:
        """Rows per rank for the indexer token split, or 0 to run unsplit.

        A trace-time (static) decision: every rank sees the same token bucket,
        forward mode and metadata, so all ranks take the same branch and the
        all-reduce always has every rank.
        """
        # getattr: tests build the backend without __init__.
        if getattr(self, "_idx_split_group", None) is None:
            return 0
        if not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            return 0  # decode, idle, target verify, draft extend
        if self.forward_metadata.decode_query_size != 1:
            return 0
        if num_tokens < self._idx_split_min_tokens:
            return 0
        tp = self._idx_split_size
        # Multiple of 8: whole sublanes, and a multiple of the decode pass's
        # 4-sequence group so a window never cuts one (MIXED batches).
        rows = -(-num_tokens // tp)  # ceil
        rows = -(-rows // 8) * 8
        if rows < _TOKEN_SPLIT_MIN_ROWS_PER_RANK:
            return 0
        return rows

    def _attention_split_rows(
        self, forward_batch: ForwardBatch, num_tokens: int
    ) -> int:
        """Rows per rank for the attention token split, or 0 to attend unsplit.

        Static per trace, like _indexer_split_rows: prefill batches the
        indexer splits and the per-query kernel would attend. Batches the
        query-blocked kernel takes keep it.
        """
        # getattr: tests build the backend without __init__.
        if not getattr(self, "_attn_split_world", 0):
            return 0
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return 0
        # Rows that leave the layer must be the whole batch's: MTP top-k sharing
        # (IndexTopKShareState publishes them) and the indexer top-k capturer.
        from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer

        spec_info = getattr(forward_batch, "spec_info", None)
        if (
            getattr(forward_batch, "reuse_dsa_topk_indices", False)
            or getattr(spec_info, "dsa_seed_topk_capture", None) is not None
            or get_global_indexer_capturer() is not None
        ):
            return 0
        rows = self._indexer_split_rows(forward_batch, num_tokens)
        if not rows or self._prefill_takes_qblock(num_tokens):
            return 0
        return rows

    def _prefill_takes_qblock(self, num_tokens: int) -> bool:
        """Whether this prefill batch goes through the query-blocked kernel."""
        if self._prefill_op is None:
            return False
        from sglang_tpu.attention.pallas.mla_sparse_fused import (
            qblock_tokens_supported,
        )

        metadata = self.forward_metadata
        pages_per_seq = metadata.page_indices.shape[0] // (
            metadata.cu_q_lens.shape[0] - 1
        )
        return qblock_tokens_supported(num_tokens, pages_per_seq, self.page_size)

    def _build_kernel_op(self):
        # The base class constructor calls this to set self._op. The q-block
        # opt-in adds a prefill op: decode, verify and draft extend keep the
        # per-query kernel.
        self._prefill_op = None
        zero_rope = self._zero_rope
        if (
            not tpu_envs.SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL.get()
            or self._dcp > 1
        ):
            return _build_pallas_sparse_mla_kernel_op(zero_rope=zero_rope)
        if self._kpool == 1:
            # Plain DSA: token-bitmap q-block prefill. Its fallback (short or
            # long-context batches) and the decode op keep the default chunk.
            self._prefill_op = _build_pallas_sparse_mla_kernel_op(1)
            return _build_pallas_sparse_mla_kernel_op(zero_rope=zero_rope)
        from sglang_tpu.attention.pallas.mla_sparse_fused import (
            QBLOCK_GATHER_CHUNK_TOKENS,
        )

        self._prefill_op = _build_pallas_sparse_mla_kernel_op(
            self._kpool, QBLOCK_GATHER_CHUNK_TOKENS, zero_rope=zero_rope
        )
        return _build_pallas_sparse_mla_kernel_op(
            0, QBLOCK_GATHER_CHUNK_TOKENS, zero_rope=zero_rope
        )

    def forward_indexer(
        self,
        indexer: "Indexer",
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        """Run one layer's indexer. ``Indexer.forward_tpu`` calls this."""
        if self._kpool > 1:
            return kpool_indexer_forward(
                indexer,
                backend=self,
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                layer_id=layer_id,
            )
        return indexer_forward(
            indexer,
            backend=self,
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
        )

    def index_topk(
        self,
        *,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [num_tokens, index_n_heads, index_head_dim]
        key: torch.Tensor,  # [num_tokens, index_head_dim]
        head_weights: torch.Tensor,  # [num_tokens, index_n_heads]
        weight_scale: float,
        key_store: str = "xla",
    ) -> torch.Tensor:
        """Run one layer's indexer: store its index keys, return its top-k rows.

        Returns ``i32[num_tokens, index_topk]``, or only this rank's window of
        rows under the attention token split. Each row lists the chosen key
        positions, counted from the start of the token's own sequence. The
        chosen positions come first and the rest of the row is ``-1``.
        """
        if self._dcp_gathers_prefill and not forward_batch.forward_mode.is_decode():
            return self._index_topk_dcp_dense(
                forward_batch=forward_batch,
                seq_lens=self._dcp_seq_lens(forward_batch),
                layer_id=layer_id,
                query=query,
                key=key,
                head_weights=head_weights,
                weight_scale=weight_scale,
            )
        if self._dcp > 1:
            return self._index_topk_dcp(
                seq_lens=self._dcp_seq_lens(forward_batch),
                layer_id=layer_id,
                query=query,
                key=key,
                head_weights=head_weights,
                weight_scale=weight_scale,
            )
        metadata = self.forward_metadata
        num_tokens = query.shape[0]
        split_rows = self._indexer_split_rows(forward_batch, num_tokens)
        if split_rows:
            return self._index_topk_token_split(
                forward_batch=forward_batch,
                layer_id=layer_id,
                query=query,
                key=key,
                head_weights=head_weights,
                weight_scale=weight_scale,
                key_store=key_store,
                split_rows=split_rows,
            )
        # The torch-to-JAX bridge passes raw buffers, so a tensor with strides
        # would be read wrongly. contiguous() makes each input dense.
        return self._index_topk_op(
            query.contiguous(),
            key.contiguous(),
            head_weights.contiguous(),
            self.token_to_kv_pool.get_index_key_cache(layer_id),
            self._kv_lens(forward_batch),
            metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
            weight_scale,
            self._topk,
            metadata.decode_query_size,
            metadata.segments,
            key_store,
        )

    def dcp_gather_buckets(self, batch_size: int) -> list[int]:
        """Lengths ``dcp_gather_index`` takes for prefill batches of
        ``batch_size`` rows; each is a compiled prefill variant."""
        if not self._dcp_gathers_prefill:
            return []
        cap = batch_size * _get_pages_per_seq(self.max_context_len, self._page_stride)
        # Powers of 4: half the compiled prefill variants of powers of 2 (each
        # a cold compile), for at most 4x the gathered pages.
        buckets = [4**i for i in range(cap.bit_length()) if 4**i < cap]
        scratch = [_DCP_SCRATCH_BUCKET] if batch_size <= self._dcp_scratch_slots else []
        return scratch + buckets + [cap]

    def _dcp_scratch_plan(
        self, forward_batch: ForwardBatch, need: list[int], seq_lens: list[int]
    ) -> Optional[list[int]]:
        """Scratch slot of each row (plan_scratch_slots), or None when this
        batch gathers; the scratch warmup takes slot ``row``."""
        state = self._dcp_scratch_state
        if self._dcp_gather_bucket_override == _DCP_SCRATCH_BUCKET:
            return list(range(len(need))) if len(need) <= len(state) else None
        owners = forward_batch.cache_request_handles
        prefix = forward_batch.extend_prefix_lens_cpu
        if (
            not state
            or self._dcp_gather_bucket_override is not None
            or owners is None
            or prefix is None
        ):
            return None
        rows = len(owners)
        # Rows past the requests are padding and must read nothing. Warmup
        # compiles the scratch variant only for padded batches that fit the
        # slots (dcp_gather_buckets); a wider one gathers.
        if rows > len(need) or any(need[rows:]) or len(need) > len(state):
            return None
        slots = plan_scratch_slots(
            state,
            owners=owners,
            prefix_lens=[int(n) for n in prefix[:rows]],
            seq_lens=seq_lens[:rows],
        )
        return None if slots is None else slots + [0] * (len(need) - rows)

    def _ensure_dcp_scratch(self, pages_per_slot: int) -> None:
        """Allocate the scratch once: every full-attention layer's buffers with
        ``slots * pages_per_slot`` unsharded pages."""
        if self._dcp_scratch:
            return
        pool = self.token_to_kv_pool
        if isinstance(pool, HybridLinearKVPool):
            layer_ids = sorted(pool.full_attention_layer_id_mapping)
        else:
            layer_ids = range(pool.start_layer, pool.start_layer + pool.layer_num)
        pages = self._dcp_scratch_slots * pages_per_slot

        def like(buf: torch.Tensor) -> torch.Tensor:
            return _zeros_kv_buffer((pages, *buf.shape[1:]), buf.dtype, buf.device)

        for layer_id in layer_ids:
            try:
                index_keys = like(pool.get_index_key_cache(layer_id))
            except ValueError:  # the layer reuses the previous layer's top-k
                index_keys = None
            self._dcp_scratch[layer_id] = (
                like(pool.get_nope_cache(layer_id)),
                like(pool.get_rope_cache(layer_id)),
                index_keys,
            )

    def _add_dcp_gather_metadata(self, forward_batch: ForwardBatch) -> None:
        """Gather list and dense page table for this prefill: only the live
        sequences' widened pages, padded to a bucket (page 0 of row 0)."""
        md = self.forward_metadata
        bs = md.cu_q_lens.shape[0] - 1
        pages_per_seq = md.page_indices.shape[0] // bs
        seq_lens = forward_batch.seq_lens_cpu
        seq_lens = (
            list(seq_lens) if isinstance(seq_lens, list) else seq_lens.tolist()
        )[:bs]
        need = [-(-int(n) // self._page_stride) for n in seq_lens]
        device = md.page_indices.device
        slots = self._dcp_scratch_plan(forward_batch, need, seq_lens)
        if slots is not None:
            # No gather: one dummy page keeps the store op's signature.
            slot_pages = pages_per_seq * self._dcp
            self._ensure_dcp_scratch(slot_pages)
            dense = torch.zeros(bs, slot_pages, dtype=torch.int32)
            for row, n in enumerate(need):
                n = min(n, pages_per_seq) * self._dcp
                dense[row, :n] = slots[row] * slot_pages + torch.arange(
                    n, dtype=torch.int32
                )
            md.dcp_gather_index = self._value_metadata(
                name="dcp_gather_index",
                value=torch.zeros(1, dtype=torch.int64),
                device=device,
            )
            md.dcp_dense_pages = self._value_metadata(
                name="dcp_dense_pages", value=dense.reshape(-1), device=device
            )
            md.dcp_scratch = True
            return
        total = sum(need)
        size = self._dcp_gather_bucket_override or next(
            b for b in self.dcp_gather_buckets(bs) if b >= total
        )
        gather = torch.zeros(size, dtype=torch.int64)
        dense = torch.zeros(bs, pages_per_seq * self._dcp, dtype=torch.int32)
        offset = 0
        for row, n in enumerate(need):
            # Only a warmup bucket below the need truncates (its rows are dummies).
            n = max(0, min(n, size - offset))
            gather[offset : offset + n] = row * pages_per_seq + torch.arange(n)
            dense[row, : n * self._dcp] = self._dcp_dense_page_ids(
                first=offset, count=n, gathered=size
            )
            offset += n
        md.dcp_gather_index = self._value_metadata(
            name="dcp_gather_index", value=gather, device=device
        )
        md.dcp_dense_pages = self._value_metadata(
            name="dcp_dense_pages", value=dense.reshape(-1), device=device
        )

    @staticmethod
    def _dcp_seq_lens(forward_batch: ForwardBatch) -> torch.Tensor:
        """Unsharded KV length per sequence, counting this step's tokens: the
        kv_lens of an unsharded forward (DCP rejects speculative modes)."""
        return forward_batch.seq_lens.to(torch.int32)

    def _dcp_gather_page_ids(self) -> torch.Tensor:
        """i32[G]: this rank's physical page of each widened page listed in
        ``dcp_gather_index``."""
        md = self.forward_metadata
        return md.page_indices.index_select(0, md.dcp_gather_index).to(torch.int32)

    def _dcp_dense_page_ids(
        self, *, first: int, count: int, gathered: int
    ) -> torch.Tensor:
        """i32[count * dcp]: where _dcp_dense_cache puts the unsharded pages of
        gathered widened pages ``[first, first + count)``, in position order
        (widened page, then rank). The all-reduce gather is page-major, the DMA
        all-gather member-major over all ``gathered`` listed pages."""
        j = torch.arange(count * self._dcp, dtype=torch.int32)
        if self._dcp_all_gather:
            return (j % self._dcp) * gathered + first + j // self._dcp
        return first * self._dcp + j

    def _dcp_dense_cache(self, pages: torch.Tensor) -> torch.Tensor:
        """The group's pages put together unsharded, at the page table
        ``dcp_dense_pages`` (_dcp_dense_page_ids): rank r's ``pages[g]`` is rank
        r's run of the g-th listed widened page."""
        if self._dcp_all_gather:
            # Rank-major [dcp, G, ...]: every page buffer is whole 128-byte lanes.
            gathered = self._dcp_all_gather_op(pages.reshape(-1, 128), self._cp_rank_t)
            return gathered.reshape(self._dcp * pages.shape[0], *pages.shape[1:])
        # Zero-padded int32 sums are exact; bytes are summed as words.
        words = pages.view(torch.int32)
        gathered = _all_reduce_gather(
            words, self._dcp_group, self._dcp, self._cp_rank_mask, dim=1
        )
        return (
            gathered.reshape(pages.shape[0] * self._dcp, *words.shape[1:])
            .view(pages.dtype)
            .contiguous()
        )

    def _index_topk_dcp_dense(
        self,
        *,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        weight_scale: float,
    ) -> torch.Tensor:
        """index_topk under DCP for prefill: the unsharded indexer over the
        gathered index keys (rows identical to an unsharded server), and the
        owned keys stored in this rank's shard."""
        md = self.forward_metadata
        shard = self.token_to_kv_pool.get_index_key_cache(layer_id)
        # One op stores the owned keys and reads the listed pages back: a
        # second op on the donated shard made XLA copy the whole shard.
        pages = self._index_key_store_gather_op(
            key.contiguous(),
            shard,
            seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.distribution,
            self._cp_rank_t,
            self._dcp_gather_page_ids(),
        )
        # The scratch is every rank's unsharded copy; the ops below store this
        # chunk's keys into it, as they do into an unsharded pool.
        dense = (
            self._dcp_scratch[layer_id][2]
            if md.dcp_scratch
            else self._dcp_dense_cache(pages)
        )
        # Every attention-TP rank now holds the unsharded cache, as without
        # DCP, so the token split applies unchanged: each rank scores a slice
        # of the rows instead of all of them against the whole history.
        split_rows = self._indexer_split_rows(forward_batch, query.shape[0])
        if split_rows:
            return self._index_topk_token_split(
                forward_batch=forward_batch,
                layer_id=layer_id,
                query=query,
                key=key,
                head_weights=head_weights,
                weight_scale=weight_scale,
                key_store="xla",
                split_rows=split_rows,
                index_key_cache=dense,
                page_indices=md.dcp_dense_pages,
                kv_lens=seq_lens,
            )
        return self._index_topk_op(
            query.contiguous(),
            key.contiguous(),
            head_weights.contiguous(),
            dense,
            seq_lens,
            self.forward_metadata.dcp_dense_pages,
            md.cu_q_lens,
            md.distribution,
            weight_scale,
            self._topk,
            md.decode_query_size,
            md.segments,
            "xla",
        )

    def _index_topk_dcp(
        self,
        *,
        seq_lens: torch.Tensor,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        weight_scale: float,
    ) -> torch.Tensor:
        """index_topk under DCP: this rank's rows of the global top-k, as
        rank-local positions (what _forward_mla_dcp hands the kernel)."""
        metadata = self.forward_metadata
        indices, scores = self._index_topk_dcp_op(
            query.contiguous(),
            key.contiguous(),
            head_weights.contiguous(),
            self.token_to_kv_pool.get_index_key_cache(layer_id),
            seq_lens,
            metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
            self._cp_rank_t,
            weight_scale,
            self._topk,
            metadata.segments,
        )
        # One gather for both: int32 sums of zero-padded blocks are exact.
        candidates = _all_reduce_gather(
            torch.stack([indices, scores]).unsqueeze(0),
            self._dcp_group,
            self._dcp,
            self._cp_rank_mask,
            dim=0,
        )  # i32[dcp, 2, num_tokens, topk]
        return self._topk_merge_dcp_op(
            candidates[:, 0].contiguous(),
            candidates[:, 1].contiguous(),
            self._cp_rank_t,
            self._topk,
        )

    def _index_topk_token_split(
        self,
        *,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        weight_scale: float,
        key_store: str,
        split_rows: int,
        index_key_cache: Optional[torch.Tensor] = None,
        page_indices: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """index_topk with the DSA indexer token split.

        Every rank stores the index keys of all tokens (each keeps the full
        index-key cache) and selects the rows of its own token slice
        ``[rank * split_rows, (rank + 1) * split_rows)``; ``_assemble_split_rows``
        puts the slices of the attention-TP group together. Under the attention
        token split, each rank attends its own slice and gets ``i32[split_rows,
        topk]`` back instead. The rank enters as a device tensor, so all ranks
        run one compiled program.

        The gathered DCP prefill passes the unsharded cache it gathered, its
        page table and the unsharded lengths; by default they are this layer's
        pool cache, the batch's page table and kv lengths.
        """
        from sglang.srt.layers.dp_attention import _rank_tensor

        metadata = self.forward_metadata
        num_tokens = query.shape[0]
        rank = _rank_tensor("attn_tp", query.device)
        row_start = (rank * split_rows).to(torch.int32).reshape(1)
        local_rows = self._index_topk_split_op(
            query.contiguous(),
            key.contiguous(),
            head_weights.contiguous(),
            (
                index_key_cache
                if index_key_cache is not None
                else self.token_to_kv_pool.get_index_key_cache(layer_id)
            ),
            kv_lens if kv_lens is not None else self._kv_lens(forward_batch),
            page_indices if page_indices is not None else metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
            row_start,
            weight_scale,
            self._topk,
            metadata.decode_query_size,
            metadata.segments,
            key_store,
            split_rows,
        )  # i32[split_rows, topk]
        if self._attention_split_rows(forward_batch, num_tokens):
            # The attention token split attends this rank's window with these
            # rows; no other rank needs them.
            return local_rows
        rows = _assemble_split_rows(
            local_rows, rank, self._idx_split_size, self._idx_split_group.all_reduce
        )
        if rows.shape[0] != num_tokens:
            # Padding rows of the last slices (all -1) are dropped.
            rows = rows[:num_tokens].contiguous()
        return rows

    def index_topk_kpool(
        self,
        *,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [num_tokens, index_n_heads, index_head_dim]
        key: torch.Tensor,  # [num_tokens, index_head_dim]
        gate: torch.Tensor,  # [num_tokens, index_head_dim]
        head_weights: torch.Tensor,  # [num_tokens, index_n_heads], f32
        ape: torch.Tensor,  # [kpool, index_head_dim], f32
        weight_scale: float,
        kpool: int,
    ) -> torch.Tensor:
        """Pooled indexer (GLM-5.3-Flash): store raw keys, close pools, return
        ``i32[num_tokens, width]`` rows in the format of index_topk."""
        metadata = self.forward_metadata
        pool = self.token_to_kv_pool
        split_rows = self._indexer_split_rows(forward_batch, query.shape[0])
        if split_rows:
            return self._index_topk_kpool_token_split(
                forward_batch=forward_batch,
                layer_id=layer_id,
                query=query,
                key=key,
                gate=gate,
                head_weights=head_weights,
                ape=ape,
                weight_scale=weight_scale,
                kpool=kpool,
                split_rows=split_rows,
            )
        raw_page_base, raw_ring_pages_cnt = self._raw_ring_args()
        return self._index_topk_op(
            query.contiguous(),
            key.contiguous(),
            gate.contiguous(),
            head_weights.contiguous(),
            ape.contiguous(),
            pool.get_kpool_raw_cache(layer_id),
            pool.get_index_key_cache(layer_id),
            self._kv_lens(forward_batch),
            metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
            raw_page_base,
            weight_scale,
            self._topk,
            kpool,
            self._kpool_width,
            metadata.decode_query_size,
            _kpool_query_bound(metadata, num_tokens=query.shape[0]),
            raw_ring_pages_cnt,
        )

    def _index_topk_kpool_token_split(
        self,
        *,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        head_weights: torch.Tensor,
        ape: torch.Tensor,
        weight_scale: float,
        kpool: int,
        split_rows: int,
    ) -> torch.Tensor:
        """index_topk_kpool with the DSA indexer token split.

        Every rank stores all tokens' raw keys and closes all pools (each
        keeps the full caches), and scores only its own token slice
        ``[rank * split_rows, (rank + 1) * split_rows)``. Scoring every query
        against every closed pool is the part that grows with the context, and
        unsplit, all attention-TP ranks repeat it for the whole batch. The
        ranks' picked pools, ``i32[split_rows, topk // kpool]``, are put
        together by ``_assemble_split_rows`` and expanded to token rows.
        """
        from sglang.srt.layers.dp_attention import _rank_tensor

        metadata = self.forward_metadata
        pool = self.token_to_kv_pool
        num_tokens = query.shape[0]
        kv_lens = self._kv_lens(forward_batch)
        rank = _rank_tensor("attn_tp", query.device)
        row_start = (rank * split_rows).to(torch.int32).reshape(1)
        raw_page_base, raw_ring_pages_cnt = self._raw_ring_args()
        local_picked = self._index_topk_split_op(
            query.contiguous(),
            key.contiguous(),
            gate.contiguous(),
            head_weights.contiguous(),
            ape.contiguous(),
            pool.get_kpool_raw_cache(layer_id),
            pool.get_index_key_cache(layer_id),
            kv_lens,
            metadata.page_indices,
            metadata.cu_q_lens,
            metadata.distribution,
            row_start,
            raw_page_base,
            weight_scale,
            self._topk,
            kpool,
            _kpool_query_bound(metadata, num_tokens=num_tokens),
            split_rows,
            raw_ring_pages_cnt,
        )  # i32[split_rows, topk // kpool]
        picked = _assemble_split_rows(
            local_picked, rank, self._idx_split_size, self._idx_split_group.all_reduce
        )
        if picked.shape[0] != num_tokens:
            # Padding rows of the last slices (all -1) are dropped.
            picked = picked[:num_tokens].contiguous()
        return self._kpool_expand_op(
            picked,
            kv_lens,
            metadata.cu_q_lens,
            metadata.distribution,
            self._topk,
            kpool,
            self._kpool_width,
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
        if self._zero_rope:
            # The zero-rope ops are exact only for a zero rope lane, which
            # _mla_kernel_args builds when the layer has no rope dims.
            assert layer.qk_head_dim == layer.v_head_dim, (
                f"layer {layer.layer_id} has rope dims, but the sparse MLA ops "
                "were built for a model without rope."
            )
        args = self._mla_kernel_args(
            q=q,
            k=k,
            layer=layer,
            forward_batch=forward_batch,
            q_rope=q_rope,
            k_rope=k_rope,
        )
        if self._dcp > 1:
            seq_lens = self._dcp_seq_lens(forward_batch)
            if self._dcp_gathers_prefill and not forward_batch.forward_mode.is_decode():
                out = self._forward_mla_dcp_dense(
                    args, topk_indices, seq_lens, layer.layer_id
                )
            else:
                out = self._forward_mla_dcp(args, topk_indices, seq_lens)
            return out.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        split_rows = self._attention_split_rows(forward_batch, q.shape[0])
        if split_rows:
            out = self._forward_mla_token_split(args, topk_indices, split_rows)
        elif (
            self._prefill_op is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            if self._kpool == 1:
                # The plain-DSA prefill op is head-major: MLA's absorbed query
                # is a per-head matmul output, so the transposes cancel in XLA
                # with the ones that made it token-major.
                ql_nope, q_pe, *rest = args
                out = self._prefill_op(
                    ql_nope.transpose(0, 1).contiguous(),
                    q_pe.transpose(0, 1).contiguous(),
                    *rest,
                    topk_indices,
                ).transpose(0, 1)
            else:
                out = self._prefill_op(*args, topk_indices)
        else:
            out = self._op(*args, topk_indices)
        # TODO(sglang-torchtpu#1072): avoid this reshape, as in pallas_tpu_mla.
        return out.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _forward_mla_token_split(
        self, args: tuple, topk_indices: torch.Tensor, rows: int
    ) -> torch.Tensor:
        """Prefill attention with SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT.

        Each rank holds its heads' queries for all T tokens and the indexer
        rows of its window ``[rank * rows, (rank + 1) * rows)``. An all-to-all
        gives each rank its window's queries for every head; the window op
        stores the step's latents (every token, on every rank) and attends the
        window; a second all-to-all returns each rank's heads for every token.
        Returns ``[T, heads, 512]``, as the per-query op does.
        """
        from sglang.srt.layers.dp_attention import _rank_tensor
        from sglang_tpu.oneshot_collectives import oneshot_all_to_all

        ql_nope, q_pe, cu_q_lens = args[0], args[1], args[8]
        num_tokens, heads, nope = ql_nope.shape
        world = self._attn_split_world
        # Send block j = window j's tokens: [world, rows * heads * (512 + 64)].
        q = torch.cat([ql_nope, q_pe], dim=-1)
        if world * rows != num_tokens:
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, world * rows - num_tokens))
        q = oneshot_all_to_all(q.reshape(world, -1), world_size=world)
        # Received block i = rank i's heads: [rows, world * heads, 512 + 64].
        q = q.view(world, rows, heads, -1).transpose(0, 1)
        q = q.reshape(rows, world * heads, -1)
        # The window's own cu_q_lens: each request's tokens inside the window.
        rank = _rank_tensor("attn_tp", q.device)
        window_cu_q_lens = (cu_q_lens - rank * rows).clamp(0, rows).to(torch.int32)
        out = self._window_op(
            q[..., :nope].contiguous(),
            q[..., nope:].contiguous(),
            *args[2:9],
            window_cu_q_lens,  # in the cu_kv_lens slot
            *args[10:],
            topk_indices.contiguous(),
        )  # [rows, world * heads, 512]
        # Send block i = this window for rank i's heads; receive every window.
        out = out.view(rows, world, heads, nope).transpose(0, 1)
        out = oneshot_all_to_all(out.reshape(world, -1), world_size=world)
        return out.view(world * rows, heads, nope)[:num_tokens]

    def _forward_mla_dcp_dense(
        self,
        args: tuple,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Prefill under DCP: the unsharded op, with this rank's heads, over the
        gathered latents; the owned latents stored in this rank's shards.

        ``topk_indices`` are unsharded rows from ``_index_topk_dcp_dense``.
        """
        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_nope,
            cache_rope,
            _kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale,
            k_scale,
            v_scale,
        ) = args
        # Store and read back in one op (see _index_topk_dcp_dense).
        nope_pages, rope_pages = self._kv_store_gather_op(
            new_kv_c,
            new_k_pe,
            cache_nope,
            cache_rope,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            self._cp_rank_t,
            self._dcp_gather_page_ids(),
        )
        if self.forward_metadata.dcp_scratch:
            dense_nope, dense_rope, _ = self._dcp_scratch[layer_id]
        else:
            dense_nope = self._dcp_dense_cache(nope_pages)
            dense_rope = self._dcp_dense_cache(rope_pages)
        return self._op(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            dense_nope,
            dense_rope,
            seq_lens,
            self.forward_metadata.dcp_dense_pages,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale,
            k_scale,
            v_scale,
            topk_indices,
        )

    def _forward_mla_dcp(
        self, args: tuple, topk_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Attend with the DCP group's heads over this rank's winners, then
        merge the group's partials by log-sum-exp and keep this rank's heads.

        ``topk_indices`` are rank-local rows from ``_index_topk_dcp``.
        """
        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_nope,
            cache_rope,
            _kv_lens,
            page_indices,
            cu_q_lens,
            _cu_kv_lens,
            distribution,
            sm_scale,
            k_scale,
            _v_scale,
        ) = args
        md = self.forward_metadata
        group, cp, rank_mask = self._dcp_group, self._dcp, self._cp_rank_mask
        nope_dim = ql_nope.shape[-1]
        q_all = gather_heads(
            torch.cat([ql_nope, q_pe], dim=-1), group, cp, rank_mask
        )  # [T, heads * dcp, 576]
        out, lse = self._attention_dcp_op(
            q_all[..., :nope_dim].contiguous(),
            q_all[..., nope_dim:].contiguous(),
            new_kv_c,
            new_k_pe,
            cache_nope,
            cache_rope,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            self._cp_rank_t,
            topk_indices,
            sm_scale,
            k_scale,
        )
        # A row with no owned winner (or a padding row) has l = 0: its lse is
        # -inf or NaN. NaN would poison the group-wide max, so pin it to -inf.
        lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float("-inf")))
        merged, _ = merge_across_ranks(
            out.float(), lse, group, cp, rank_mask, self._cp_rank_index
        )
        return merged.to(ql_nope.dtype)

    def _kv_cache_buffers(self, layer_id: int) -> tuple:
        # This backend's pool has two buffers per layer where the fused pool
        # has one. The op takes them in this order.
        pool = self.token_to_kv_pool
        return (pool.get_nope_cache(layer_id), pool.get_rope_cache(layer_id))
