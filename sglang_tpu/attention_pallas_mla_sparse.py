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

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_kpool,
    get_dsa_index_topk,
    get_dsa_mtp_topk_width,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang_tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
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


class TpuPallasSparseMlaAttnBackend(TpuPallasMlaAttnBackend):
    """``pallas_tpu_mla`` with sparse attention: it runs the DSA indexer too."""

    # Set by _build_kernel_op under the q-block opt-in.
    _prefill_op = None

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        hf_config = model_runner.model_config.hf_config
        # Set before super().__init__(), which calls _build_kernel_op().
        self._kpool = get_dsa_index_kpool(hf_config)
        self._zero_rope = _has_zero_rope(hf_config)
        super().__init__(model_runner, speculative_step_id)
        self._topk = get_dsa_index_topk(hf_config)
        if self._kpool > 1:
            self._kpool_width = topk_row_width(hf_config)
            self._index_topk_op = build_dsa_kpool_topk_op()
        else:
            self._index_topk_op = build_dsa_index_topk_op()
        self._init_indexer_token_split()
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

    def _init_indexer_token_split(self) -> None:
        """Set up the DSA indexer token split (always on with attn-TP > 1)."""
        self._idx_split_group = None
        self._idx_split_size = 1
        self._idx_split_min_tokens = 0
        self._index_topk_split_op = None
        self._kpool_expand_op = None
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
        if not tpu_envs.SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL.get():
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
    ) -> torch.Tensor:
        """index_topk with the DSA indexer token split.

        Every rank stores the index keys of all tokens (each keeps the full
        index-key cache) and selects the rows of its own token slice
        ``[rank * split_rows, (rank + 1) * split_rows)``; ``_assemble_split_rows``
        puts the slices of the attention-TP group together. Under the attention
        token split, each rank attends its own slice and gets ``i32[split_rows,
        topk]`` back instead. The rank enters as a device tensor, so all ranks
        run one compiled program.
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
            self.token_to_kv_pool.get_index_key_cache(layer_id),
            self._kv_lens(forward_batch),
            metadata.page_indices,
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
            weight_scale,
            self._topk,
            kpool,
            self._kpool_width,
            metadata.decode_query_size,
            _kpool_query_bound(metadata, num_tokens=query.shape[0]),
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
            weight_scale,
            self._topk,
            kpool,
            _kpool_query_bound(metadata, num_tokens=num_tokens),
            split_rows,
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

    def _kv_cache_buffers(self, layer_id: int) -> tuple:
        # This backend's pool has two buffers per layer where the fused pool
        # has one. The op takes them in this order.
        pool = self.token_to_kv_pool
        return (pool.get_nope_cache(layer_id), pool.get_rope_cache(layer_id))
