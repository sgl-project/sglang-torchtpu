# SPDX-License-Identifier: Apache-2.0
"""Ring-mode prefill-context-parallel (PCP) op of the TPU batched RPA backend.

Used only under ``SGLANG_TPU_PCP_MODE=ring``

KV layout: each sequence's KV cache is split across the context-parallel ranks
page by page, as in decode context parallelism. With ``P`` ranks, rank ``r``
stores pages ``r, r + P, r + 2P, ...`` (its "stripe"), and all ranks share one
page table.

Prefill: every layer runs one fused kernel on every rank, in two phases.

1. Cached prefix: the stripes are passed around the ring (remote DMA between
   ranks), so each rank's queries see the whole prefix while each rank only
   ever stores its own stripe.
2. New tokens: the same queries then attend the chunk's own K/V, which the
   caller has all-gathered across the group
   (``cp_forward.all_gather_and_write_kv``).

Read-only KV pool: the op never writes the pool. XLA copies any input that a
remote-DMA kernel writes in place, which for the pool would mean copying all of
it every layer. Instead, the caller writes this rank's new pages beforehand
with a local in-place write (``cp_forward.all_gather_and_write_kv``).
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import logging
from dataclasses import dataclass, fields

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import cdiv
from sglang_tpu.kv_row_write import kv_write_segments, write_run_bound
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

logger = logging.getLogger(__name__)

__all__ = [
    "PCP_RING_AXIS",
    "PCP_RING_CP_AXIS",
    "PCP_RING_OP_NAME",
    "PcpRingMetadata",
    "build_pcp_ring_metadata",
    "build_pcp_ring_op",
    "check_pcp_ring_layer",
    "check_pcp_ring_schedule",
    "dcp_stripe_group",
    "forward_pcp_ring",
    "init_pcp_ring_op",
    "pcp_kv_layout",
    "pcp_ring_mesh_axes",
    "pcp_ring_partition_specs",
]


@dataclass
class PcpRingMetadata:
    """Device inputs of the ring kernel for one extend batch, in the op's
    argument order: ``cp_layout.RingLayout`` uploaded, plus the page tables.

    Lengths and positions count tokens of the whole (unsplit) sequence. Only
    ``q_positions`` and ``cp_rank`` differ between ranks.
    """

    # Phase 1, cached prefix: one row per request.
    # i32[pad_b]  cached prefix + the row's query length
    cache_kv_lens: torch.Tensor
    # i32[pad_b * pages_per_seq]  page table of the row's request
    cache_page_indices: torch.Tensor
    # i32[pad_b + 1]  cumulative query lengths
    cache_cu_q_lens: torch.Tensor
    # i32[3]  RPA row distribution
    cache_distribution: torch.Tensor
    # Phase 2, new tokens: two rows per request (one per owned block).
    # i32[2 * pad_b]  prefix + extend length of the row's request
    new_kv_lens: torch.Tensor
    # i32[2 * pad_b * pages_per_seq]  page table of the row's request
    new_page_indices: torch.Tensor
    # i32[2 * pad_b + 1]  cumulative query lengths
    new_cu_q_lens: torch.Tensor
    # i32[3]  RPA row distribution
    new_distribution: torch.Tensor
    # i32[2 * pad_b * gathered_pages]  where each page of the request's new
    # tokens sits in the all-gathered K/V buffer
    new_kv_page_indices: torch.Tensor
    # i32[2 * pad_b]  position of the row's first query in its sequence
    q_positions: torch.Tensor
    # i32[2 * pad_b]  cached prefix length of the row's request
    kv_cache_lens: torch.Tensor
    # i32[1]  this rank's index in the context-parallel group
    cp_rank: torch.Tensor


def dcp_stripe_group(parallel, *, pcp_ring: bool):
    """Which ranks share one sequence's page-striped KV cache.

    Returns ``(this rank's stripe index, the group, head blocks)``.

    * Plain DCP uses SGLang's DCP group: consecutive ranks in the TP group.
    * Ring mode uses the attention-CP group instead. With attention TP ``t``
      that is ranks ``r, r + t, r + 2t, ...``: they hold the same heads and
      differ only in which tokens they store.

    ``head blocks`` is how many different query-head blocks the group's ranks
    hold: 1 if they all hold the same heads (attention TP 1, or ring mode),
    else the group size (plain DCP at attention TP > 1).
    """
    if parallel.attn_dcp_size <= 1:
        return 0, None, 1
    if pcp_ring:
        return parallel.attn_cp_rank, parallel.attn_cp_group, 1
    head_blocks = 1 if parallel.attn_tp_size == 1 else parallel.attn_dcp_size
    return parallel.attn_dcp_rank, parallel.dcp_group, head_blocks


def init_pcp_ring_op(backend, device: str):
    """Build the ring op for ``backend``, or return None off TPU.

    Fails for hybrid sliding-window KV pools, which the kernel does not support.
    """
    parallel = get_parallel()
    if backend._is_hybrid_swa:
        raise NotImplementedError(
            "Ring-mode prefill context parallelism does not support hybrid "
            "sliding-window KV pools."
        )
    return build_pcp_ring_op(parallel.attn_cp_size, attn_tp_size=parallel.attn_tp_size)


def build_pcp_ring_metadata(backend, forward_batch, host):
    """Attention metadata for one ring-mode extend batch.

    Built on the host and uploaded once per batch. As under DCP, the page table
    is read in logical pages of ``page_size * cp`` tokens, so it is the same on
    every rank, and every length passed to the kernel counts the whole
    sequence.

    The ring kernel's inputs go in ``cp_ring``; the plain RPA fields describe
    the new-token rows; ``cp_write_segments`` lists only the rows whose pages
    this rank stores.
    """
    from sglang_tpu.attention_pallas_rpa_base import (
        _SCOPED_SMEM_BYTES,
        PallasForwardMetadata,
        _get_pages_per_seq,
        _make_active_num_tokens,
        _page_table_smem_bytes,
        _validate_kv_len_capacity,
    )

    ring = host.ring
    page_size = backend.page_size
    stride = backend._page_stride
    pages_per_seq = _get_pages_per_seq(backend.max_context_len, stride)
    new_rows = 2 * host.pad_b
    # All three tables are scalar-prefetch operands of one kernel launch.
    table_bytes = (
        _page_table_smem_bytes(new_rows, pages_per_seq)
        + _page_table_smem_bytes(host.pad_b, pages_per_seq)
        + _page_table_smem_bytes(new_rows, ring.gathered_pages)
    )
    if table_bytes > _SCOPED_SMEM_BYTES:
        raise ValueError(
            f"Ring-mode context-parallel page tables ({new_rows} rows x "
            f"{pages_per_seq} cache pages and x {ring.gathered_pages} chunk "
            f"pages) need {table_bytes / 2**20:.2f} MiB of SMEM, above the "
            f"{_SCOPED_SMEM_BYTES / 2**20:.0f} MiB budget; lower "
            "--prefill-max-requests or the prefill token buckets."
        )
    _validate_kv_len_capacity(
        backend._dcp_local_kv_len(int(host.pseudo_kv_lens.max())),
        pages_per_seq * page_size,
    )
    device = forward_batch.seq_lens.device

    req_pool_indices = forward_batch.req_pool_indices_cpu
    assert req_pool_indices is not None and req_pool_indices.device.type == "cpu"
    assert req_pool_indices.shape[0] == host.pad_b
    req_pool_indices = req_pool_indices.to(torch.int64)
    req_to_token = backend.req_to_token_pool.req_to_token
    assert req_to_token.device.type == "cpu"  # pinned at start-up for PCP
    page_cols = backend._host_page_columns(pages_per_seq, stride)
    per_seq = req_to_token[req_pool_indices[:, None], page_cols[None, :]]
    req_pages = (per_seq // stride).to(torch.int32)

    def _i32(values) -> torch.Tensor:
        return torch.from_numpy(values).to(torch.int32)

    def _cu(lens) -> torch.Tensor:
        out = torch.zeros(lens.shape[0] + 1, dtype=torch.int32)
        out[1:] = torch.cumsum(torch.from_numpy(lens), dim=0, dtype=torch.int32)
        return out

    def _dev(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=True)

    write_segments, write_num_segments = _ring_write_runs(
        host,
        req_to_token,
        req_pool_indices,
        page_size=page_size,
        stride=stride,
        rank=backend._dcp_rank,
    )
    row_reqs = torch.from_numpy(host.row_reqs)
    ring_md = PcpRingMetadata(
        cache_kv_lens=_dev(_i32(ring.cache_kv_lens)),
        cache_page_indices=_dev(req_pages.reshape(-1).contiguous()),
        cache_cu_q_lens=_dev(_cu(ring.cache_q_lens)),
        cache_distribution=backend._get_cached_rpa_distribution(
            False, host.num_reqs, device
        ),
        new_kv_lens=_dev(_i32(host.pseudo_kv_lens)),
        new_page_indices=_dev(req_pages[row_reqs].reshape(-1).contiguous()),
        new_cu_q_lens=_dev(_cu(host.pseudo_q_lens)),
        new_distribution=backend._get_cached_rpa_distribution(
            False, host.num_rows, device
        ),
        new_kv_page_indices=_dev(_i32(ring.new_kv_page_indices).reshape(-1)),
        q_positions=_dev(_i32(ring.q_positions)),
        kv_cache_lens=_dev(_i32(ring.kv_cache_lens)),
        cp_rank=backend._cp_rank_t,
    )
    cu_kv_lens = torch.arange(new_rows + 1, dtype=torch.int32) * (
        pages_per_seq * page_size
    )
    return PallasForwardMetadata(
        kv_lens=ring_md.new_kv_lens,
        cu_q_lens=ring_md.new_cu_q_lens,
        cu_kv_lens=_dev(cu_kv_lens),
        page_indices=ring_md.new_page_indices,
        distribution=ring_md.new_distribution,
        active_num_tokens=_make_active_num_tokens(ring_md.new_cu_q_lens, host.num_rows),
        cp_write_segments=_dev(write_segments),
        cp_write_num_segments=_dev(write_num_segments),
        cp_last_rows=_dev(torch.from_numpy(host.last_rows)),
        cp_last_owner_mask=_dev(_i32(host.last_owner_mask)),
        cp_ring=ring_md,
    )


def _ring_write_runs(host, req_to_token, req_pool_indices, *, page_size, stride, rank):
    """Which all-gathered K/V rows this rank writes into its pool, and where.

    Returns write runs (``kv_write_segments``) and their count as i32[1].

    As under DCP, the allocator's locations are widened by the group size: a
    logical page is ``stride = page_size * cp`` locations, and rank ``r``
    stores its ``r``-th ``page_size`` slice. So location ``loc`` belongs to
    rank ``(loc % stride) // page_size`` and goes to row
    ``(loc // stride) * page_size + loc % page_size`` of that rank's pool.

    Skipped: rows of other ranks' pages, padding rows, and the reserved page 0
    (used by warmup batches). The table has the same shape on every rank; only
    its contents differ.
    """
    gathered_req = torch.from_numpy(host.gathered_req)
    gathered_pos = torch.from_numpy(host.gathered_pos)
    valid = gathered_req >= 0
    req = gathered_req.clamp(min=0)
    pos = torch.from_numpy(host.prefix_lens)[req] + gathered_pos.clamp(min=0)
    locs = req_to_token[req_pool_indices[req], pos].to(torch.int64)
    allocated = valid & (locs >= stride)
    # The kernel derives a key's rank and row from its position, so the
    # allocator's location must sit at the same offset in its logical page.
    if bool((locs[allocated] % stride != pos[allocated] % stride).any()):
        raise RuntimeError(
            "ring-mode context parallelism: a KV location is not aligned with "
            "its token position inside the logical page; the page stripes of "
            "the cache would not match the kernel's."
        )
    owned = allocated & ((locs % stride) // page_size == rank)
    slots = (locs // stride) * page_size + locs % page_size
    max_segments = write_run_bound(
        slots.shape[0], page_size=page_size, blocks=2 * host.cp_size * host.pad_b
    )
    write_segments, num_segments = kv_write_segments(
        dest_slots=slots.numpy(), valid=owned.numpy(), max_segments=max_segments
    )
    return write_segments, torch.tensor([num_segments], dtype=torch.int32)


def check_pcp_ring_layer(layer, *, sinks, sliding_window, soft_cap, causal, pack):
    """Refuse a layer the fused ring kernel cannot run."""
    if sinks is not None or sliding_window is not None or soft_cap is not None:
        raise NotImplementedError(
            "Ring-mode prefill context parallelism supports plain causal "
            "attention only (no attention sink, sliding window or logit "
            f"cap); layer {layer.layer_id} needs one."
        )
    if not causal or pack > 1:
        raise NotImplementedError(
            "Ring-mode prefill context parallelism supports only causal "
            "attention over an unpacked KV pool."
        )


def forward_pcp_ring(
    md,
    op,
    layer,
    q,
    k_all,
    v_all,
    kv_cache,
    *,
    sinks,
    sliding_window,
    soft_cap,
    causal,
    pack,
    k_scale,
    v_scale,
) -> torch.Tensor:
    """Attention for one layer of a ring-mode extend; returns ``[T, H, D]``.

    ``q`` holds this rank's zigzag rows. ``k_all`` / ``v_all`` are the chunk's
    K/V gathered from all ranks by ``all_gather_and_write_kv`` (quantized for an
    fp8 pool). That call has already written this rank's new pages into the
    pool; writing them here, after the kernel reads the pool, made XLA copy the
    whole pool each layer.
    """
    check_pcp_ring_layer(
        layer,
        sinks=sinks,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        causal=causal,
        pack=pack,
    )
    ring = md.cp_ring
    operands = [getattr(ring, field.name) for field in fields(ring)]
    sm_scale = float(layer.scaling)
    return op(q, k_all, v_all, kv_cache, *operands, sm_scale, k_scale, v_scale)


PCP_RING_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_pcp_ring"

# Attention TP 1: a 1-D mesh over all ranks, and the ring runs along it.
PCP_RING_AXIS = "model"
# Attention TP > 1: a ``(cp, attn_tp)`` mesh with axes ``("pcp", "model")``;
# the ring runs along ``"pcp"``.
PCP_RING_CP_AXIS = "pcp"

# Tensor arguments of the op, in order. Each rank passes its own copy, split
# along the first dimension, so no rank relies on another holding equal values.
_TENSOR_ARGS = (
    "queries",
    "keys",
    "values",
    "kv_cache_fused",
    *(field.name for field in fields(PcpRingMetadata)),
)
# Plain Python arguments after the tensors. They change the generated kernel,
# so each distinct value gets its own export.
_STATIC_ARGS = ("sm_scale", "k_scale", "v_scale")

# Torch custom ops are process-global, so register once per
# (CP size, attention TP size) and reuse it across backend instances.
_OP_CACHE: dict = {}


def pcp_kv_layout():
    """The KV layout, as the ring kernel package's own enum.

    Same answer as ``attention_pallas_rpa_batched.kv_layout``, but that one
    returns the production package's enum type.
    """
    from sglang_tpu.attention.pallas.batched_rpa_pcp import configs
    from sglang_tpu.attention_pallas_rpa_batched import uses_seq_along_lane_kv

    if uses_seq_along_lane_kv():
        return configs.KVLayout.SEQ_ALONG_LANE
    return configs.KVLayout.HEAD_ALONG_SUBLANE


def pcp_ring_mesh_axes(attn_tp_size: int = 1) -> tuple[str, ...]:
    """Mesh axis names, ring axis first: ``("model",)`` at attention TP 1,
    else ``("pcp", "model")``."""
    if attn_tp_size == 1:
        return (PCP_RING_AXIS,)
    return (PCP_RING_CP_AXIS, PCP_RING_AXIS)


def pcp_ring_partition_specs(attn_tp_size: int = 1):
    """``(input specs, output specs)`` for the exported function.

    Every tensor is split along its first dimension over all mesh axes, so
    each rank's shard is exactly the tensor it passed. The static arguments
    get an empty spec.
    """
    from jax.sharding import PartitionSpec as P

    axes = pcp_ring_mesh_axes(attn_tp_size)
    shard = P(axes[0]) if len(axes) == 1 else P(axes)
    in_specs = (shard,) * len(_TENSOR_ARGS) + (P(),) * len(_STATIC_ARGS)
    out_specs = (shard,)
    return in_specs, out_specs


# Shapes whose block sizes have been logged already (once per process).
_LOGGED_BLOCK_SIZES: set = set()


def _log_block_sizes_once(
    *,
    cp_size,
    mesh_axes,
    queries,
    keys,
    kv_cache_fused,
    cache_kv_lens,
    cache_page_indices,
    new_kv_lens,
    new_page_indices,
    k_scale,
    v_scale,
):
    """Log the kernel's block sizes once per shape, for debugging.

    Runs at trace time and never raises.
    """
    try:
        key = (
            tuple(queries.shape),
            tuple(keys.shape),
            tuple(kv_cache_fused.shape),
            tuple(cache_page_indices.shape),
            tuple(new_page_indices.shape),
            cp_size,
        )
        if key in _LOGGED_BLOCK_SIZES:
            return
        _LOGGED_BLOCK_SIZES.add(key)
        from sglang_tpu.attention.pallas.batched_rpa_pcp.wrapper import (
            pcp_fused_configs,
        )

        num_cache_rows = cache_kv_lens.shape[0]
        num_new_rows = new_kv_lens.shape[0]
        fused = pcp_fused_configs(
            num_q_heads=queries.shape[1],
            num_kv_heads=keys.shape[1],
            head_dim=queries.shape[2],
            dtype_q=queries.dtype,
            dtype_kv=kv_cache_fused.dtype,
            page_size=kv_cache_fused.shape[1],
            cp_group_size=cp_size,
            total_q_tokens=queries.shape[0],
            num_cache_rows=num_cache_rows,
            cache_pages_per_row=cache_page_indices.shape[0] // num_cache_rows,
            num_new_rows=num_new_rows,
            new_pages_per_row=new_page_indices.shape[0] // num_new_rows,
            ring_axis_name=mesh_axes[0],
            ring_mesh_axis_names=mesh_axes,
            k_scale=k_scale,
            v_scale=v_scale,
            cache_lens_cp_page_aligned=True,
        )
        c, n = fused.cfgs_cache.block, fused.cfgs_new.block
        logger.info(
            "PCP ring kernel blocks: q=%s kv=%s cache_rows=%d new_rows=%d "
            "cache_pages/row=%d | cache phase bq=%d bq_c=%d bkv=%d batch=%d | "
            "new-token phase bq=%d bq_c=%d bkv=%d batch=%d",
            tuple(queries.shape),
            tuple(keys.shape),
            num_cache_rows,
            num_new_rows,
            cache_page_indices.shape[0] // num_cache_rows,
            c.bq_sz,
            c.bq_c_sz,
            c.bkv_sz,
            c.batch_size,
            n.bq_sz,
            n.bq_c_sz,
            n.bkv_sz,
            n.batch_size,
        )
    except Exception as e:  # noqa: BLE001 -- diagnostics only
        logger.debug("PCP ring block-size log skipped: %s", e)


def _invoke_pcp_kernel(
    kernel,
    *,
    cp_size: int,
    mesh_axes: tuple[str, ...] = (PCP_RING_AXIS,),
    queries,
    keys,
    values,
    kv_cache_fused,
    cache_kv_lens,
    cache_page_indices,
    cache_cu_q_lens,
    cache_distribution,
    new_kv_lens,
    new_page_indices,
    new_cu_q_lens,
    new_distribution,
    new_kv_page_indices,
    q_positions,
    kv_cache_lens,
    cp_rank,
    sm_scale,
    k_scale,
    v_scale,
):
    """Run the ring kernel on one rank's inputs (called inside ``shard_map``).

    Returns this rank's attention output and leaves the pool untouched. The
    same compiled program handles requests with and without a cached prefix;
    one without a prefix just skips the ring.
    """
    _log_block_sizes_once(
        cp_size=cp_size,
        mesh_axes=mesh_axes,
        queries=queries,
        keys=keys,
        kv_cache_fused=kv_cache_fused,
        cache_kv_lens=cache_kv_lens,
        cache_page_indices=cache_page_indices,
        new_kv_lens=new_kv_lens,
        new_page_indices=new_page_indices,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return kernel(
        queries,
        keys,
        values,
        kv_cache_fused,
        cache_kv_lens=cache_kv_lens,
        cache_page_indices=cache_page_indices,
        cache_cu_q_lens=cache_cu_q_lens,
        cache_distribution=cache_distribution,
        new_kv_lens=new_kv_lens,
        new_page_indices=new_page_indices,
        new_cu_q_lens=new_cu_q_lens,
        new_distribution=new_distribution,
        new_kv_page_indices=new_kv_page_indices,
        q_positions=q_positions,
        kv_cache_lens=kv_cache_lens,
        cp_rank=cp_rank,
        cp_group_size=cp_size,
        ring_axis_name=mesh_axes[0],
        ring_mesh_axis_names=mesh_axes,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_layout=pcp_kv_layout(),
        update_kv_cache=False,
        # Cached prefixes are whole logical pages (page_size * cp tokens),
        # since that is how the allocator and chunking cut them. Checked on
        # the host by `check_pcp_ring_schedule`.
        cache_lens_cp_page_aligned=True,
    )


def build_pcp_ring_op(cp_size: int, *, attn_tp_size: int = 1, mesh=None, kernel=None):
    """Wrap the ring kernel as a torch custom op.

    There are ``attn_tp_size`` rings (one per attention-TP head shard) of
    ``cp_size`` ranks each. The op takes this rank's inputs, returns this
    rank's attention output, and modifies nothing.

    Call it from the backend constructor: registration can't run inside
    ``torch.compile`` (dynamo cannot trace the signature inspection), and the
    mesh needs ``torch.distributed`` to be initialized.

    ``mesh`` and ``kernel`` are for tests: a prebuilt mesh and a stand-in
    kernel.
    """
    cache_key = (cp_size, attn_tp_size)
    cached = _OP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import jax
    from jax.sharding import NamedSharding
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

    mesh_axes = pcp_ring_mesh_axes(attn_tp_size)
    if mesh is None:
        from torch_tpu._internal.pallas import get_pallas_mesh

        shape = (cp_size,) if attn_tp_size == 1 else (cp_size, attn_tp_size)
        mesh = get_pallas_mesh(mesh_axes, shape)
    if kernel is None:
        from sglang_tpu.attention.pallas.batched_rpa_pcp.wrapper import (
            ragged_paged_attention_pcp_fused as kernel,
        )
    in_specs, out_specs = pcp_ring_partition_specs(attn_tp_size)

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        cache_kv_lens: jax.Array,
        cache_page_indices: jax.Array,
        cache_cu_q_lens: jax.Array,
        cache_distribution: jax.Array,
        new_kv_lens: jax.Array,
        new_page_indices: jax.Array,
        new_cu_q_lens: jax.Array,
        new_distribution: jax.Array,
        new_kv_page_indices: jax.Array,
        q_positions: jax.Array,
        kv_cache_lens: jax.Array,
        cp_rank: jax.Array,
        sm_scale: float = 1.0,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> jax.Array:
        def _per_rank(*operands):
            return _invoke_pcp_kernel(
                kernel,
                cp_size=cp_size,
                mesh_axes=mesh_axes,
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                **dict(zip(_TENSOR_ARGS, operands)),
            )

        return jax.shard_map(
            _per_rank,
            mesh=mesh,
            in_specs=in_specs[: len(_TENSOR_ARGS)],
            out_specs=out_specs[0],
            check_vma=False,
        )(
            queries,
            keys,
            values,
            kv_cache_fused,
            cache_kv_lens,
            cache_page_indices,
            cache_cu_q_lens,
            cache_distribution,
            new_kv_lens,
            new_page_indices,
            new_cu_q_lens,
            new_distribution,
            new_kv_page_indices,
            q_positions,
            kv_cache_lens,
            cp_rank,
        )

    static_argnums = tuple(
        range(len(_TENSOR_ARGS), len(_TENSOR_ARGS) + len(_STATIC_ARGS))
    )
    out_sharding = NamedSharding(mesh, out_specs[0])

    jit_fn = jax.jit(
        _wrapped,
        static_argnums=static_argnums,
        keep_unused=True,
        out_shardings=out_sharding,
    )
    # The name is unique in the process and the same on every rank and across
    # restarts; the exported program is also fingerprinted.
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"jax_{PCP_RING_OP_NAME}",
        [],
        {
            "cp_size": cp_size,
            "attn_tp_size": attn_tp_size,
            "static_argnums": static_argnums,
        },
    )
    jax_kernel = _CpuExportJaxCallable(
        name=f"jax_{PCP_RING_OP_NAME}_cp{cp_size}"
        + (f"_tp{attn_tp_size}" if attn_tp_size > 1 else ""),
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=in_specs,
        static_argnums=static_argnums,
    )
    jax_kernel.sglang_output_shardings = (out_sharding,)

    custom_op = _register_torch_op(jax_kernel)
    _OP_CACHE[cache_key] = custom_op
    return custom_op


def _register_torch_op(jax_kernel):
    """Register ``jax_kernel`` as the torch custom op. Separate so tests can
    pass a stand-in kernel."""

    @torch.library.custom_op(PCP_RING_OP_NAME, mutates_args=(), device_types=["tpu"])
    def custom_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        cache_kv_lens: torch.Tensor,
        cache_page_indices: torch.Tensor,
        cache_cu_q_lens: torch.Tensor,
        cache_distribution: torch.Tensor,
        new_kv_lens: torch.Tensor,
        new_page_indices: torch.Tensor,
        new_cu_q_lens: torch.Tensor,
        new_distribution: torch.Tensor,
        new_kv_page_indices: torch.Tensor,
        q_positions: torch.Tensor,
        kv_cache_lens: torch.Tensor,
        cp_rank: torch.Tensor,
        sm_scale: float = 1.0,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            return jax_kernel(
                queries,
                keys,
                values,
                kv_cache_fused,
                cache_kv_lens,
                cache_page_indices,
                cache_cu_q_lens,
                cache_distribution,
                new_kv_lens,
                new_page_indices,
                new_cu_q_lens,
                new_distribution,
                new_kv_page_indices,
                q_positions,
                kv_cache_lens,
                cp_rank,
                sm_scale,
                k_scale,
                v_scale,
            )

    custom_op.register_fake(_fake_impl)
    return custom_op


def _fake_impl(queries: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # Shape-only version for torch.compile: same shape as the queries, without
    # tracing JAX.
    return torch.empty_like(queries)


def check_pcp_ring_schedule(runner, backend, host):
    """Check on the host that a ring-mode extend batch fits the kernel's
    schedule buffers, before any rank launches it.

    Only the kernel package knows its block sizes, so this passes it the row
    lengths and lets it do the counting. All ranks pass the same cache-phase
    rows, so they accept or reject a batch together.
    """
    from sglang_tpu.attention.pallas.batched_rpa_pcp.wrapper import (
        check_pcp_fused_schedule,
    )

    ring = host.ring
    config = runner.model_config
    attn_tp_size = get_parallel().attn_tp_size
    stride = backend.page_size * host.cp_size
    pages_per_seq = cdiv(backend.max_context_len, stride)
    with tpu_info_for_cpu_export():
        check_pcp_fused_schedule(
            num_q_heads=config.get_num_attention_heads(attn_tp_size),
            num_kv_heads=config.get_num_kv_heads(attn_tp_size),
            head_dim=config.head_dim,
            dtype_q=str(config.dtype).removeprefix("torch."),
            dtype_kv=str(runner.token_to_kv_pool.dtype).removeprefix("torch."),
            page_size=backend.page_size,
            kv_layout=pcp_kv_layout(),
            cp_group_size=host.cp_size,
            total_q_tokens=host.local_rows_padded,
            total_kv_tokens=host.cp_size * host.local_rows_padded,
            cache_cu_q_lens=[0, *ring.cache_q_lens.cumsum().tolist()],
            cache_kv_lens=ring.cache_kv_lens.tolist(),
            cache_pages_per_row=pages_per_seq,
            new_cu_q_lens=[0, *host.pseudo_q_lens.cumsum().tolist()],
            new_kv_lens=host.pseudo_kv_lens.tolist(),
            kv_cache_lens=ring.kv_cache_lens.tolist(),
            q_positions=ring.q_positions.tolist(),
            new_pages_per_row=pages_per_seq,
            new_kv_pages_per_row=ring.gathered_pages,
            num_cache_rows_live=host.num_reqs,
            num_new_rows_live=host.num_rows,
            cache_lens_cp_page_aligned=True,
        )
