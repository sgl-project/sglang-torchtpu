# SPDX-License-Identifier: Apache-2.0
"""Ring-mode prefill-context-parallel (PCP) op of the TPU batched RPA backend.

Under ``SGLANG_TPU_PCP_MODE=ring`` the KV cache of a sequence is split across
the ranks of the context-parallel group by page, exactly as under decode
context parallelism: rank ``r`` stores global page ``l * P + r`` and every rank
reads the same page table. A zigzag prefill then runs one fused kernel per
layer on every rank:

* cache phase: each rank attends its own queries against the whole cached
  prefix by rotating the KV blocks of the stripes around the ring (remote DMA
  from rank to rank), so no rank ever holds more than its stripe;
* new-token phase: the same queries continue the online softmax over the
  chunk's own K/V, which the caller all-gathered across the group
  (``cp_forward.all_gather_and_write_kv``) and the kernel reads by page through
  ``new_kv_page_indices``.

Because the ring uses remote DMA, the op is a mesh kernel: it is exported as a
``jax.shard_map`` over the whole world with partition specs, every rank passes
its own shard, and all ranks must launch it in the same order with the same
static shapes. With attention TP 1 the mesh is one axis over the world; with
attention TP ``t > 1`` it is ``(cp, t)`` in torch-rank order (rank
``cp_rank * t + tp_rank``) and the ring runs along the ``cp`` axis, so the
``t`` rings of one host (one per attention-TP head shard) run side by side. Everything rank-dependent (``cp_rank``, ``q_positions``) is a
traced tensor, so all ranks trace one program.

The op reads the KV pool and never writes it: XLA copies every operand that a
kernel containing a remote DMA aliases (see ``muse_megakernel/kernel.py`` and
``glm_flash_megakernel/kernel/decode.py``), which for a KV pool would be a
whole-pool copy per layer. The caller writes the new pages this rank owns with
the local in-place row writer instead (``cp_forward.all_gather_and_write_kv``).

``attention_pallas_rpa_base`` drives the op; gather-mode PCP never calls in
here. The host-side row layout is ``cp_layout.RingLayout``.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import logging
from dataclasses import dataclass, fields

import torch

from sglang.srt.runtime_context import get_parallel
from sglang_tpu.cp_forward import all_gather_and_write_kv
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
    """Operands of the fused ring kernel for one ring-mode context-parallel
    extend (``cp_layout.RingLayout`` on the device, plus the page tables), in
    the op's order.

    Lengths and positions are global token counts. Everything but
    ``q_positions`` and ``cp_rank`` has the same value on every rank.
    """

    # Cache (ring) phase rows, one per request.
    # i32[pad_b]  cached prefix + the row's query length
    cache_kv_lens: torch.Tensor
    # i32[pad_b * pages_per_seq]  striped page table of each row's request
    cache_page_indices: torch.Tensor
    # i32[pad_b + 1]
    cache_cu_q_lens: torch.Tensor
    # i32[3]
    cache_distribution: torch.Tensor
    # New-token phase rows (two per request).
    # i32[2 * pad_b]  prefix + extend length of the row's request
    new_kv_lens: torch.Tensor
    # i32[2 * pad_b * pages_per_seq]
    new_page_indices: torch.Tensor
    # i32[2 * pad_b + 1]
    new_cu_q_lens: torch.Tensor
    # i32[3]
    new_distribution: torch.Tensor
    # i32[2 * pad_b * gathered_pages]  page of the rank-major gathered K/V
    # buffer holding each page of the row's chunk
    new_kv_page_indices: torch.Tensor
    # i32[2 * pad_b]  global position of each row's first query
    q_positions: torch.Tensor
    # i32[2 * pad_b]  cached prefix of the row's request
    kv_cache_lens: torch.Tensor
    # i32[1]  this rank's index in the context-parallel group
    cp_rank: torch.Tensor


def dcp_stripe_group(parallel, *, pcp_ring: bool):
    """``(stripe rank, stripe group, replicates_qo)`` of the page-striped KV cache.

    Plain DCP stripes over SGLang's DCP group (consecutive ranks inside the
    TP group). Ring-mode prefill CP stripes over the attention-CP group
    instead: with attention TP ``t`` that is ranks ``r, r + t, ...``, which
    share one attention-TP head shard and differ only in tokens, so the
    attention-CP rank owns the stripe and decode merges over that group.
    ``replicates_qo``: q and o_proj heads are replicated across the group
    (attention TP 1, or ring mode at any attention TP; every rank computes the
    same q heads and needs them all for its o_proj), so the striped kernels run
    on this rank's own heads, q is not gathered and the merge does not crop.
    TODO: replace with ``--dcp-replicate-q-proj`` plus a new
    ``--dcp-replicate-o-proj`` once both handle a DCP group that spans the
    attention-CP axis.
    """
    if parallel.attn_dcp_size <= 1:
        return 0, None, False
    if pcp_ring:
        return parallel.attn_cp_rank, parallel.attn_cp_group, True
    return parallel.attn_dcp_rank, parallel.dcp_group, parallel.attn_tp_size == 1


def init_pcp_ring_op(backend, device: str):
    """Register the fused ring op for ``backend`` (``None`` off TPU).
    ``--dcp-size == --attn-cp-size`` is enforced by server-args validation."""
    parallel = get_parallel()
    if backend._is_hybrid_swa:
        raise NotImplementedError(
            "Ring-mode prefill context parallelism does not support hybrid "
            "sliding-window KV pools."
        )
    if device != "tpu":
        return None
    return build_pcp_ring_op(
        backend.kernel_spec, parallel.attn_cp_size, attn_tp_size=parallel.attn_tp_size
    )


def build_pcp_ring_metadata(backend, forward_batch, host):
    """``PallasForwardMetadata`` of a ring-mode context-parallel extend.

    The KV cache is page-striped (rank ``r`` holds global page
    ``l * cp + r``), so the page table is read at the logical page stride
    ``page_size * cp`` and is identical on every rank, exactly as under
    DCP; all lengths handed to the kernel are global. Built on the host and
    uploaded once. ``cp_write_segments`` holds only the runs of the pages this
    rank owns, and the plain fields describe the new-token rows.
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
    """Write runs of the rank-major all-gathered rows ``rank`` owns, as slots
    of its page stripe, and their i32[1] count.

    The allocator hands out locations in a space widened by the group size
    (as under DCP): a logical page spans ``page_size * cp`` locations and
    maps to one physical page id on every rank, whose rank-``r`` copy holds
    the ``r``-th ``page_size`` run. So location ``loc`` belongs to rank
    ``(loc % stride) // page_size`` and lands in that rank's pool at row
    ``(loc // stride) * page_size + loc % page_size``. Rows of every other
    rank's pages, padding rows and the reserved page 0 (warmup batches)
    are left out. Which rows are written is data, not program structure:
    the table has the same shape on every rank.
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
    md, op, q, k, v, kv_cache, *, quantize_kv, sm_scale, k_scale, v_scale
) -> torch.Tensor:
    """One layer of a ring-mode context-parallel extend, ``[T, H, D]``.

    ``q``/``k``/``v`` are this rank's zigzag rows in the compute dtype.
    The chunk's K/V are all-gathered over the group and, for an fp8 pool,
    quantized after the gather, as in gather mode. The fused kernel attends
    the striped cached prefix around the ring and then the gathered chunk;
    it only reads the pool. The new pages this rank owns are written from
    the gathered rows by the local in-place writer *before* the kernel:
    the kernel reads cached positions only, so the order does not change
    its result, and writing after a read of the same buffer made XLA copy
    the whole per-layer pool ahead of the in-place write (one pool-sized
    HBM copy per layer, growing with --max-total-tokens).
    """
    k_all, v_all = all_gather_and_write_kv(
        kv_cache=kv_cache,
        k=k,
        v=v,
        write_segments=md.cp_write_segments,
        write_num_segments=md.cp_write_num_segments,
        group=get_parallel().attn_cp_group,
        quantize=quantize_kv,
    )
    ring = md.cp_ring
    operands = [getattr(ring, field.name) for field in fields(ring)]
    return op(q, k_all, v_all, kv_cache, *operands, sm_scale, k_scale, v_scale)


# torch custom-op name. Distinct from the plain batched op and the DCP op so
# all three can be registered in one process and are keyed separately by the
# FX-graph compile cache; bump the suffix when the kernel source changes.
PCP_RING_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_pcp_ring_v1"

# Mesh axis the ring runs along with attention TP 1. The mesh is then
# one-dimensional over the whole world, in torch-rank order
# (``torch_tpu.get_pallas_mesh``).
PCP_RING_AXIS = "model"
# With attention TP > 1 the mesh is ``(PCP_RING_CP_AXIS, PCP_RING_AXIS)`` of
# shape ``(cp, attn_tp)``; the ring runs along the first axis.
PCP_RING_CP_AXIS = "pcp"

# Tensor operands of the op, in order. Every one is passed per rank and
# partitioned along its first dimension, so nothing relies on two ranks holding
# equal values.
_TENSOR_ARGS = (
    "queries",
    "keys",
    "values",
    "kv_cache_fused",
    *(field.name for field in fields(PcpRingMetadata)),
)
# Static (non-traced) arguments after the tensors; they select the generated
# kernel, so they key the export.
_STATIC_ARGS = ("sm_scale", "k_scale", "v_scale")

# torch custom ops are process-global; cache the registration per
# (CP size, attention TP size) so repeated backend construction reuses it.
_OP_CACHE: dict = {}


def pcp_kv_layout():
    """The KV layout as the experimental ``batched_rpa_pcp`` kernel's own enum
    (the production package's ``attention_pallas_rpa_batched.kv_layout`` returns
    the production enum). Read at trace / check time."""
    from sglang_tpu.attention.pallas.batched_rpa_pcp import configs
    from sglang_tpu.attention_pallas_rpa_batched import uses_seq_along_lane_kv

    if uses_seq_along_lane_kv():
        return configs.KVLayout.SEQ_ALONG_LANE
    return configs.KVLayout.HEAD_ALONG_SUBLANE


def pcp_ring_mesh_axes(attn_tp_size: int = 1) -> tuple[str, ...]:
    """Mesh axis names, ring axis first: ``("model",)`` with attention TP 1,
    else ``("pcp", "model")`` for a ``(cp, attn_tp)`` mesh."""
    if attn_tp_size == 1:
        return (PCP_RING_AXIS,)
    return (PCP_RING_CP_AXIS, PCP_RING_AXIS)


def pcp_ring_partition_specs(attn_tp_size: int = 1):
    """``(input specs, output specs)`` of the exported mesh function.

    Inputs: one spec per positional argument of the wrapped function (the
    static ones included; they carry no array). Output: the attention output.
    Every tensor is split along its first dimension over all mesh axes, in
    torch-rank order, so each rank's shard is exactly what it passed.
    """
    from jax.sharding import PartitionSpec as P

    axes = pcp_ring_mesh_axes(attn_tp_size)
    shard = P(axes[0]) if len(axes) == 1 else P(axes)
    in_specs = (shard,) * len(_TENSOR_ARGS) + (P(),) * len(_STATIC_ARGS)
    out_specs = (shard,)
    return in_specs, out_specs


# Static shapes whose block sizes were already logged (trace time, per process).
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
    """Log, once per static shape, the block sizes the fused kernel resolves
    (``pcp_fused_configs`` is what the kernel itself calls). Runs at trace time;
    never fails the trace."""
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
    """Call the fused kernel for one rank (inside ``shard_map``).

    Returns this rank's attention output. The pool is read-only
    (``update_kv_cache=False``): the kernel neither aliases nor returns it.
    One program covers batches with and without a cached prefix (a request
    without one gets no ring steps).
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
        # Ring-mode cached prefixes are whole stripes: the allocator page and
        # the prefill chunk boundaries are page_size * cp tokens (checked on
        # the host by `check_pcp_ring_schedule`).
        cache_lens_cp_page_aligned=True,
    )


def build_pcp_ring_op(
    spec, cp_size: int, *, attn_tp_size: int = 1, mesh=None, kernel=None
):
    """Register the fused ring kernel as a torch custom op for ``cp_size`` ranks
    per ring and ``attn_tp_size`` rings (one per attention-TP head shard).

    Must be called from the backend constructor, never inside a traced region
    (the registration inspects signatures, which dynamo cannot trace), and
    after ``torch.distributed`` is initialized (the mesh is built from the
    process group).

    The op takes this rank's operands and returns this rank's attention
    output; it mutates nothing (the KV pool is a read-only operand).

    ``mesh`` and ``kernel`` exist for tests: a prebuilt JAX mesh and a stand-in
    for ``ragged_paged_attention_pcp_fused``.
    """
    from sglang_tpu.attention_pallas_rpa_batched import RPA_BATCHED_SPEC

    if spec is not RPA_BATCHED_SPEC:
        raise NotImplementedError(
            f"{spec.op_name} is not the batched RPA kernel; ring-mode prefill "
            "context parallelism on TPU requires --attention-backend "
            "pallas_tpu_rpa_batched."
        )
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
    # No donation and no aliasing: see the module docstring.
    jit_fn = jax.jit(
        _wrapped,
        static_argnums=static_argnums,
        keep_unused=True,
        out_shardings=out_sharding,
    )
    # The qualified name is unique within the process and stable across ranks
    # and restarts; the callable additionally fingerprints the exported MLIR.
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
    """The torch custom op around ``jax_kernel`` (split out so CPU tests can
    exercise the argument plumbing with a stand-in callable)."""

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
    # Shapes only, without tracing JAX: this is what lets the dynamic token
    # dimension flow through torch.compile.
    return torch.empty_like(queries)


def check_pcp_ring_schedule(runner, backend, host):
    """Check a ring-mode context-parallel extend against the fused kernel's
    schedule buffers before any rank launches it.

    The fused kernel runs two schedules (the ring over the cached prefix and
    the new-token phase) with block sizes only the kernel package knows, so the
    step accounting lives there; this hands it the host lengths of the rows
    this rank will run. Every rank passes the same cache-phase rows, so the
    ranks accept or refuse a batch together.
    """
    from sglang_tpu.attention.pallas.batched_rpa_pcp.wrapper import (
        check_pcp_fused_schedule,
    )

    ring = host.ring
    config = runner.model_config
    attn_tp_size = get_parallel().attn_tp_size
    stride = backend.page_size * host.cp_size
    pages_per_seq = -(-backend.max_context_len // stride)
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
