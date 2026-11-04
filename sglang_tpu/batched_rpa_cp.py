# SPDX-License-Identifier: Apache-2.0
"""Context Parallelism (CP) extensions for the TPU Pallas Batched RPA backend.

Supports:
1. Prefill Context Parallelism (PCP) via ICI Ring Attention:
   - Cache Phase (CACHE_ONLY): KV cache is page-striped across CP ranks (1/CP memory).
     KV blocks are rotated around the TPU mesh ring in VMEM via Pallas async remote copy
     (pltpu.make_async_remote_copy), accumulating attention in a single online softmax pass.
   - Current Phase (NEW_TOKENS_ONLY): Current chunk tokens are all-gathered across the CP
     group, attended causally, and selectively written back to the rank's owned pages.
   - Softmax state merge: Combines Cache Phase and Current Phase outputs using Log-Sum-Exp.
2. Decode Context Parallelism (DCP) via position-sharded KV cache:
   - Evaluates NEW_TOKENS_ONLY and CACHE_ONLY with rank-selective writeback, followed
     by group LSE reduction.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import torch

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_CP_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_cp"
_PCP_DEBUG_DUMP = bool(int(__import__("os").environ.get("SGLANG_PCP_DEBUG_DUMP", "0")))
_PCP_DEBUG_CALLS: dict[int, int] = {}


_GATHER_BUFFERS: dict = {}
_AG_OP_NAME = "sglang_tpu_pallas::pcp_all_gather_rows"
_AG_OP_CACHE: dict = {}
_ORDER_CACHE: dict = {}
_GLOBAL_AG_OP = None
_GLOBAL_RING_OP = None
# Row granularity of the exported all-gather: bounds the number of per-shape
# exports (one per padded row count) and keeps the gathered chunks tile aligned.
_AG_ROW_BUCKET = 16
_PCP_GATHER_IMPL = __import__("os").environ.get("SGLANG_PCP_GATHER_IMPL", "jax")
# Device order of the exported SPMD meshes: "native" (torch_tpu's
# get_pallas_mesh; falls back to "rank" when the build lacks it) or "rank"
# (jax devices listed in torch-rank order).
_PCP_MESH_ORDER = __import__("os").environ.get("SGLANG_PCP_MESH_ORDER", "native")


_MESH_DEVICE_IDS: list | None = None


def _rank_ordered_device_ids(world: int) -> list[int]:
    """Physical TPU device id of every torch rank (index = rank).

    The exported SPMD programs assign partition p to `mesh.devices.flat[p]`,
    and the CP axis of the mesh must coincide with the attn_cp process groups
    (built over torch ranks), so the mesh has to list devices in rank order.
    """
    global _MESH_DEVICE_IDS
    if _MESH_DEVICE_IDS is not None and len(_MESH_DEVICE_IDS) >= world:
        return _MESH_DEVICE_IDS[:world]
    import torch.distributed as dist

    ids = list(range(world))
    if dist.is_initialized() and dist.get_world_size() >= world:
        try:
            from torch_tpu._internal.distributed import tpu_distributed

            mine = int(tpu_distributed.global_device_id())
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, mine)
            ids = [int(i) for i in gathered[:world]]
        except Exception:  # noqa: BLE001 - fall back to identity mapping
            pass
    _MESH_DEVICE_IDS = ids
    return ids


def _get_pallas_mesh(axis_names, mesh_shape, order: str | None = None):
    order = order or _PCP_MESH_ORDER
    if order == "native":
        try:
            from torch_tpu._internal.pallas import get_pallas_mesh

            return get_pallas_mesh(axis_names, mesh_shape)
        except ImportError:
            order = "rank"
    # torch_tpu builds without get_pallas_mesh: build the mesh from the JAX
    # view of the local TPU devices, ordered by torch rank.
    import math

    import jax

    world = math.prod(mesh_shape)
    by_id = {int(d.id): d for d in jax.devices()}
    ids = _rank_ordered_device_ids(world)
    devices = np.array([by_id[i] for i in ids]).reshape(mesh_shape)
    return jax.sharding.Mesh(devices, axis_names)


def _call_custom_kernel_supports_device_assignment() -> bool:
    try:
        from torch_tpu._internal.pallas import pallas as pallas_impl

        doc = (
            getattr(pallas_impl.tpu_torch_pallas.call_custom_kernel, "__doc__", "")
            or ""
        )
        return "device_assignment" in doc
    except Exception:  # noqa: BLE001
        return False


def _export_mesh_op(name, fn, mesh, input_partition_specs, output_partition_specs):
    """Export an SPMD JAX op over `mesh` as a torch custom op.

    Newer torch_tpu builds take the mesh's device assignment through
    `call_custom_kernel(device_assignment=...)` (sglang's `cpu_export_jax_op`);
    older ones only offer `torch_tpu._internal.pallas.jax_op(mesh=...)`.
    """
    if _call_custom_kernel_supports_device_assignment():
        try:
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op
        except ImportError:
            from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        return cpu_export_jax_op(
            name,
            fn,
            mesh=mesh,
            input_partition_specs=input_partition_specs,
            output_partition_specs=output_partition_specs,
        )
    from torch_tpu._internal.pallas import jax_op

    return jax_op(name, fn, mesh=mesh, input_partition_specs=input_partition_specs)


def build_cp_all_gather_op(
    cp_size: int, attn_tp_size: int, mesh_order: str | None = None
):
    """Export a `lax.all_gather` over the CP axis of the (pcp, model) mesh.

    The eager torch collective (`all_gather_into_tensor` on the attn_cp group)
    is not ordered against the Pallas/XLA programs the prefill path launches
    around it, and in the PCP forward the remote ranks' rows intermittently
    arrived as NaN/zero. An XLA all-gather inside the same program stream has no
    such race. Rows are gathered along dim 0: rank r's block lands at
    [r * rows, (r + 1) * rows). Mesh position (c, t) is torch rank
    c * attn_tp_size + t, matching the attn_cp groups built in parallel_state.
    """
    global _GLOBAL_AG_OP
    mesh_order = mesh_order or _PCP_MESH_ORDER
    key = (cp_size, attn_tp_size, mesh_order)
    cached = _AG_OP_CACHE.get(key)
    if cached is not None:
        _GLOBAL_AG_OP = cached
        return cached

    import jax
    from jax.sharding import PartitionSpec as P

    mesh = _get_pallas_mesh(("pcp", "model"), (cp_size, attn_tp_size), order=mesh_order)
    import jax.numpy as jnp

    # Every rank passes its own [rows, cols] block as a "replicated" P() array
    # (the torch_tpu runtime specializes the program on the local shapes). The
    # gather is cp-1 collective-permutes along the CP axis, as in pp_wave:
    # output slot s holds the block of CP rank (self - s) % cp (slot 0 = own
    # block); the caller maps slots back to ranks. (A plain `all_gather` of the
    # P() input is folded into copies of the local block, and `axis_index`
    # lowers to partition-id, which the runtime's SPMD compile rejects.)
    def _local(a: jax.Array) -> jax.Array:
        blocks = [a]
        for s in range(1, cp_size):
            perm = [(i, (i + s) % cp_size) for i in range(cp_size)]
            blocks.append(jax.lax.ppermute(a, "pcp", perm))
        return jnp.stack(blocks, axis=0)

    def _wrapped(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(),
            check_vma=False,
        )(x)

    op = _export_mesh_op(
        f"{_AG_OP_NAME}_cp{cp_size}_tp{attn_tp_size}_{mesh_order}",
        _wrapped,
        mesh,
        (P(),),
        (P(),),
    )

    def _ag_fake(src):
        return src.new_empty((cp_size, src.shape[0], src.shape[1]))

    try:
        op.register_fake(_ag_fake)
    except Exception:
        pass

    target_op = getattr(op, "_opoverload", op)
    _AG_OP_CACHE[key] = target_op
    _GLOBAL_AG_OP = target_op
    return target_op


def cp_all_gather_padded(
    x: torch.Tensor, local_len: int, max_len: int, cp_size: int, group=None, impl=None
) -> torch.Tensor:
    """All-gather `x[:local_len]` (zero-padded to `max_len` rows) across the CP
    group. Returns `[cp_size, max_len, cols]`; entry r holds rank r's rows.

    impl="jax" (default) runs the exported XLA all-gather. impl="torch"
    (SGLANG_PCP_GATHER_IMPL=torch) uses the eager `all_gather_into_tensor`,
    which on torch_tpu returned wrong rows for the remote ranks whenever the
    per-rank shard lengths differ (uneven zigzag chunks); kept for A/B only.
    """
    impl = impl or _PCP_GATHER_IMPL
    cols = x.shape[1]
    if impl == "jax":
        rows = -(-max_len // _AG_ROW_BUCKET) * _AG_ROW_BUCKET
        src = x[:local_len]
        if rows > local_len:
            src = torch.nn.functional.pad(src, (0, 0, 0, rows - local_len))
        src = src.contiguous()
        parallel = get_parallel()
        op = _GLOBAL_AG_OP
        if op is None:
            op = build_cp_all_gather_op(cp_size, parallel.attn_tp_size)
        out = op(src).view(cp_size, rows, cols)
        # slot s holds rank (cp_rank - s) % cp_size -> rank r is slot (cp_rank - r) % cp_size
        cp_rank = parallel.attn_cp_rank
        _order_key = (cp_rank, cp_size, str(out.device))
        order = _ORDER_CACHE.get(_order_key)
        if order is None:
            order = torch.tensor(
                [(cp_rank - r) % cp_size for r in range(cp_size)], device=out.device
            )
            _ORDER_CACHE[_order_key] = order
        return out.index_select(0, order)[:, :max_len]

    group = group or get_parallel().attn_cp_group
    key = ("src", max_len, cols, x.dtype, x.device)
    src = _GATHER_BUFFERS.get(key)
    if src is None:
        src = torch.zeros((max_len, cols), dtype=x.dtype, device=x.device)
        _GATHER_BUFFERS[key] = src
    src[:local_len].copy_(x[:local_len])
    if max_len > local_len:
        src[local_len:].zero_()
    okey = ("dst", max_len * cp_size, cols, x.dtype, x.device)
    dst = _GATHER_BUFFERS.get(okey)
    if dst is None:
        dst = torch.zeros((max_len * cp_size, cols), dtype=x.dtype, device=x.device)
        _GATHER_BUFFERS[okey] = dst
    torch.accelerator.synchronize()
    group.all_gather_into_tensor(dst, src)
    torch.accelerator.synchronize()
    return dst.view(cp_size, max_len, cols)


def cp_all_gather_rows(x: torch.Tensor, forward_batch, group=None) -> torch.Tensor:
    """All-gather a rank-local 2-D row block into global zigzag token order.

    Every rank holds the same number of (padded) local rows, so the gather is
    one in-graph ``all_gather_into_tensor`` over the attention-CP group
    (rank-major result, ``[cp * rows, cols]``) followed by one row gather with
    the host-built ``gather_reorder_index``. ``SGLANG_PCP_GATHER_IMPL=ppermute``
    keeps the older exported ``cp_size - 1`` collective-permute program (slot
    order + ``direct_reorder_index``) for A/B comparison.
    """
    gather_idx = getattr(forward_batch, "gather_reorder_index", None)
    if (
        gather_idx is None
        and getattr(forward_batch, "attn_cp_metadata", None) is not None
    ):
        gather_idx = getattr(
            forward_batch.attn_cp_metadata, "gather_reorder_index", None
        )
    if gather_idx is not None and _PCP_GATHER_IMPL != "ppermute":
        cp_size = get_parallel().attn_cp_size
        group = group or get_parallel().attn_cp_group
        x = x.contiguous()
        out = x.new_empty((cp_size * x.shape[0], x.shape[1]))
        group.all_gather_into_tensor(out, x)
        if gather_idx.device != out.device:
            gather_idx = gather_idx.to(out.device)
        return out.index_select(0, gather_idx)

    direct_idx = getattr(forward_batch, "direct_reorder_index", None)
    if (
        direct_idx is None
        and getattr(forward_batch, "attn_cp_metadata", None) is not None
    ):
        direct_idx = getattr(
            forward_batch.attn_cp_metadata, "direct_reorder_index", None
        )
    if direct_idx is not None:
        cp_size = get_parallel().attn_cp_size
        op = _GLOBAL_AG_OP
        if op is None:
            op = build_cp_all_gather_op(cp_size, get_parallel().attn_tp_size)
        out = op(x.contiguous())
        if direct_idx.device != out.device:
            direct_idx = direct_idx.to(out.device)
        return out.reshape(-1, x.shape[1]).index_select(0, direct_idx)

    meta = (
        getattr(forward_batch, "attn_cp_metadata", None)
        if forward_batch is not None
        else None
    )
    per_rank = meta.per_rank_logical_token or meta.per_rank_actual_token
    cp_size = len(per_rank)
    cp_rank = get_parallel().attn_cp_rank
    max_len = max(per_rank)
    local_len = per_rank[cp_rank]
    assert x.dim() == 2 and x.shape[0] >= local_len

    chunks = cp_all_gather_padded(x, local_len, max_len, cp_size, group=group)
    gathered = torch.cat([chunks[r][: per_rank[r]] for r in range(cp_size)], dim=0)
    reorder_index = getattr(meta, "reorder_index_tensor", None)
    if reorder_index is not None:
        if reorder_index.device != gathered.device:
            reorder_index = reorder_index.to(gathered.device)
        return gathered.index_select(0, reorder_index)
    parts = torch.split(gathered, meta.reverse_split_len, dim=0)
    return torch.cat([parts[i] for i in meta.cp_reverse_index], dim=0)


def cp_last_token_rows(
    hidden_states: torch.Tensor,
    *,
    last_rows: torch.Tensor,
    last_owner_mask: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """[pad_b, H] last-token hidden states, identical on every rank.

    Each rank contributes the rows it owns (the owner of each request's last
    non-empty block) and zeros elsewhere; a sum all-reduce over the group
    broadcasts them.
    """
    if group is None:
        group = get_parallel().attn_cp_group
    if last_rows.device != hidden_states.device:
        last_rows = last_rows.to(hidden_states.device)
    if last_owner_mask.device != hidden_states.device:
        last_owner_mask = last_owner_mask.to(hidden_states.device)
    rows = hidden_states.index_select(0, last_rows)
    zero = torch.zeros((), dtype=rows.dtype, device=rows.device)
    rows = torch.where(last_owner_mask.to(torch.bool)[:, None], rows, zero)
    res = group.all_reduce(rows)
    return res if res is not None else rows


def _pcp_debug_dump(
    layer,
    cp_rank,
    q,
    k,
    v,
    k_reordered,
    v_reordered,
    kv_cache,
    kv_lens,
    page_indices,
    pcp_meta,
    distribution,
    update_kv,
    ctx_out,
    ctx_lse,
    cur_out,
    cur_lse,
    sliding_window,
    sm_scale,
    cp_size,
    forward_batch,
):
    """Debug aid (SGLANG_PCP_DEBUG_DUMP=1): per-layer NaN statistics of the
    PCP attention inputs/outputs."""
    layer_id = int(getattr(layer, "layer_id", -1))
    call = _PCP_DEBUG_CALLS.get(layer_id, 0)
    _PCP_DEBUG_CALLS[layer_id] = call + 1

    def nan_rows(t):
        if t is None:
            return "-"
        bad = torch.isnan(t.float()).flatten(1).any(1)
        idx = torch.nonzero(bad).flatten().tolist()
        return f"{len(idx)}:{idx[:12]}"

    cu = pcp_meta.pcp_cu_q_lens.tolist()
    print(
        f"[PCP-DBG r{cp_rank} L{layer_id} call{call}] q={tuple(q.shape)} k_loc={tuple(k.shape)} "
        f"k_glob={tuple(k_reordered.shape)} cu={cu[:9]} qpos={pcp_meta.q_positions.tolist()[:8]} "
        f"kvlens={kv_lens.tolist()[:8]} kvnew={pcp_meta.kv_new_lens.tolist()[:8]} dist={distribution.tolist()} "
        f"upd={update_kv.tolist()[:8]} sw={sliding_window} | nan q={nan_rows(q)} k={nan_rows(k)} v={nan_rows(v)} "
        f"kglob={nan_rows(k_reordered)} vglob={nan_rows(v_reordered)} cur_out={nan_rows(cur_out)} cur_lse={nan_rows(cur_lse)} "
        f"ctx_out={nan_rows(ctx_out)} ctx_lse={nan_rows(ctx_lse)}",
        flush=True,
    )


_OP_CACHE: dict = {}
_CP_RANK_TENSORS: dict[tuple[torch.device, int], torch.Tensor] = {}


def ensure_cp_rank_tensor(
    device: torch.device, cp_rank: int, ref_tensor: torch.Tensor | None = None
) -> torch.Tensor:
    """Create a persistent device tensor for cp_rank to avoid graph recompilations."""
    if ref_tensor is not None and ref_tensor.numel() > 0:
        return (ref_tensor[:1] * 0 + cp_rank).to(torch.int32)
    device = torch.device(device)
    key = (device, cp_rank)
    tensor = _CP_RANK_TENSORS.get(key)
    if tensor is None:
        tensor = torch.tensor([cp_rank], dtype=torch.int32, device=device)
        _CP_RANK_TENSORS[key] = tensor
    return tensor


@dataclass
class PcpBatchedMetadata:
    """Metadata for Batched RPA Prefill Context Parallelism."""

    pcp_cu_q_lens: torch.Tensor
    q_positions: torch.Tensor
    new_kv_page_indices: Optional[torch.Tensor]
    update_kv_cache: torch.Tensor
    kv_new_lens: torch.Tensor
    striped_page_indices: torch.Tensor
    cache_pages: int
    num_reqs: int
    pcp_cache_lens: torch.Tensor
    c_per_req: int = 1
    is_active: bool = True
    # Any request in the batch has cached (prefix) tokens: selects the
    # program that runs the ring / cache phase. Host value (trace-time).
    has_prefix: bool = False
    # [2 * num_reqs] largest head / tail chunk of each request over the CP
    # group: the ring walks that many Q blocks on every rank so the hops stay
    # in lockstep when the page-granular split gives ranks unequal chunks.
    ring_q_lens: Optional[torch.Tensor] = None
    # Replicated-pool prefill (see `batched_rpa_pcp_forward_replicated`), set
    # when the page-aligned layout is active and the batch's prefix fits:
    #   kv_row_dst   [cp * pad_t] row inside the touched-pages buffer
    #                (touched index * page + offset) of every row of the
    #                rank-major all-gathered new K/V (0 = padding dump);
    #   touched_slots [P] pool slots of the pages the new tokens land in
    #                (index 0 is the dump page, slot 0; 0-padded);
    #   prefix_send  [R] this rank's owned prefix-page slots, 0-padded;
    #   prefix_recv  [cp * R] slot each gathered prefix page is written to;
    #   kv_lens_full [2 * num_reqs] prefix + new tokens per virtual sequence;
    #   full_page_indices [2 * num_reqs * pages_per_seq] the request's whole
    #                (unstriped) page table per virtual sequence.
    kv_row_dst: Optional[torch.Tensor] = None
    touched_slots: Optional[torch.Tensor] = None
    # Page-granular write (pool-aligned layout): gathered page feeding each
    # touched page, and per request the touched index / leading rows of a
    # page shared with the prefix (0 = none).
    page_src: Optional[torch.Tensor] = None
    boundary_pos: Optional[torch.Tensor] = None
    boundary_lead: Optional[torch.Tensor] = None
    prefix_send: Optional[torch.Tensor] = None
    prefix_recv: Optional[torch.Tensor] = None
    kv_lens_full: Optional[torch.Tensor] = None
    full_page_indices: Optional[torch.Tensor] = None
    # [2 * num_reqs] end position (prefix + chunk end) of each virtual
    # sequence, for a kernel without q_positions (reference kernel).
    kv_lens_ends: Optional[torch.Tensor] = None

    @classmethod
    def create_empty(cls, device=None) -> "PcpBatchedMetadata":
        if device is None:
            device = torch.device("cpu")
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        return cls(
            pcp_cu_q_lens=empty_i32,
            q_positions=empty_i32,
            new_kv_page_indices=None,
            update_kv_cache=empty_i32,
            kv_new_lens=empty_i32,
            striped_page_indices=empty_i32,
            cache_pages=0,
            num_reqs=0,
            pcp_cache_lens=empty_i32,
            c_per_req=1,
            is_active=False,
        )


_RING_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_pcp_ring"


def build_batched_rpa_pcp_ring_op(cp_size: int, attn_tp_size: int):
    """Register the PCP cache-phase (ring attention) op over the CP x TP mesh.

    The ring rotates KV blocks with mesh-addressed remote DMAs and reads its
    rank from `lax.axis_index("pcp")`, so the kernel has to run inside a
    `shard_map` over a ("pcp", "model") mesh that spans the whole TP group:
    device `d` is CP rank `d // attn_tp_size`, matching the attn_cp groups
    built in parallel_state. Every rank passes its own local tensors; they are
    declared replicated (P()) so local and global shapes coincide, as in
    `fused_moe_rs_op`.
    """
    global _GLOBAL_RING_OP
    key = (_RING_OP_NAME, cp_size, attn_tp_size)
    cached = _OP_CACHE.get(key)
    if cached is not None:
        _GLOBAL_RING_OP = cached
        return cached

    import functools

    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.attention_pallas_rpa_batched import kv_layout
    from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

    mesh = _get_pallas_mesh(("pcp", "model"), (cp_size, attn_tp_size))
    num_tensor_args = 13

    def _local(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        q_positions,
        kv_new_lens,
        cp_rank,
        update_kv_cache,
        ring_q_lens,
        *,
        sm_scale,
        sliding_window,
        soft_cap,
        k_scale,
        v_scale,
    ):
        out, _, lse = wrapper.ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache_fused,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=True,
            attention_scope=configs.AttentionScope.CACHE_ONLY,
            cp_group_size=cp_size,
            cp_rank=cp_rank,
            cp_ring_q_lens=ring_q_lens,
            pcp_ring_axis_name="pcp",
            pcp_ring_mesh_axis_names=("pcp", "model"),
            q_positions=q_positions,
            kv_new_lens=kv_new_lens,
            new_kv_page_indices=None,
            update_kv_cache=update_kv_cache,
            return_lse=True,
            kv_layout=kv_layout(),
            # The cache phase only reads the pool. Without this the kernel's
            # kv in/out alias makes the (non-donating) exported program copy
            # the whole layer pool into a fresh output buffer on every call.
            alias_kv_cache=False,
        )
        return out, lse

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        q_positions: jax.Array,
        kv_new_lens: jax.Array,
        cp_rank: jax.Array,
        update_kv_cache: jax.Array,
        ring_q_lens: jax.Array,
        sm_scale: float,
        sliding_window: int | None,
        soft_cap: float | None,
        k_scale: float | None,
        v_scale: float | None,
    ) -> tuple[jax.Array, jax.Array]:
        fn = functools.partial(
            _local,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return jax.shard_map(
            fn,
            mesh=mesh,
            in_specs=(P(),) * num_tensor_args,
            out_specs=(P(), P()),
            check_vma=False,
        )(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            q_positions,
            kv_new_lens,
            cp_rank,
            update_kv_cache,
            ring_q_lens,
        )

    op = _export_mesh_op(
        f"{_RING_OP_NAME}_cp{cp_size}_tp{attn_tp_size}",
        _wrapped,
        mesh,
        (P(),) * num_tensor_args + (None,) * 5,
        (P(), P()),
    )

    def _ring_fake(queries, *args, **kwargs):
        lse = queries.new_empty(queries.shape[0], queries.shape[1], dtype=torch.float32)
        return torch.empty_like(queries), lse

    try:
        op.register_fake(_ring_fake)
    except Exception:
        pass

    target_op = getattr(op, "_opoverload", op)
    _OP_CACHE[key] = target_op
    _GLOBAL_RING_OP = target_op
    return target_op


_REF_FULL_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_ref_full"


def build_batched_rpa_ref_full_op():
    """FULL-scope attention through the merge-base (reference) batched RPA
    kernel, for A/B against this branch's kernel (SGLANG_TPU_PCP_REF_KERNEL=1).

    The reference kernel has no q_positions: a virtual sequence's Q rows are
    its last `q_len` positions, read from `keys`/`values`, and `kv_lens` is
    its end position.
    """
    cached = _OP_CACHE.get(_REF_FULL_OP_NAME)
    if cached is not None:
        return cached

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.batched_rpa import configs as rconfigs
    from sglang_tpu.attention.pallas.batched_rpa import wrapper as rwrapper

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        out, kv_updated = rwrapper.ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache_fused,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=True,
            skip_kv_update=True,
            kv_layout=rconfigs.KVLayout.HEAD_ALONG_SUBLANE,
        )
        return out, kv_updated

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{_REF_FULL_OP_NAME}",
        static_argnums=(8, 9, 10, 11, 12),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _REF_FULL_OP_NAME, mutates_args=("kv_cache_fused",), device_types=["tpu"]
    )
    def custom_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        out, _ = jax_kernel(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
        )
        return out

    @custom_op.register_fake
    def _fake_impl(queries, keys, values, kv_cache_fused, *args, **kwargs):
        return torch.empty_like(queries)

    _OP_CACHE[_REF_FULL_OP_NAME] = custom_op
    return custom_op


_USE_REF_KERNEL = os.environ.get("SGLANG_TPU_PCP_REF_KERNEL", "0") == "1"


def build_batched_rpa_cp_op():
    """Register the CP-aware custom op for batched RPA."""
    cached = _OP_CACHE.get(_CP_OP_NAME)
    if cached is not None:
        return cached

    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention_pallas_rpa_batched import kv_layout
    from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        q_positions: jax.Array,
        kv_new_lens: jax.Array,
        cp_rank: jax.Array,
        new_kv_page_indices: jax.Array | None = None,
        update_kv_cache: jax.Array | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        attention_scope: int = 0,
        cp_group_size: int = 1,
        pcp_ring_axis_name: str | None = None,
        return_lse: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(attention_scope, int):
            _scopes = [
                configs.AttentionScope.FULL,
                configs.AttentionScope.CACHE_ONLY,
                configs.AttentionScope.NEW_TOKENS_ONLY,
                configs.AttentionScope.STRIPED,
            ]
            scope_enum = _scopes[attention_scope]
        elif isinstance(attention_scope, str):
            scope_enum = configs.AttentionScope(attention_scope.lower())
        else:
            scope_enum = attention_scope
        cp_size_arg = cp_group_size if cp_group_size > 1 else None
        ring_mesh = ("pcp", "model") if pcp_ring_axis_name is not None else None
        new_pages = (
            None
            if (
                scope_enum == configs.AttentionScope.CACHE_ONLY
                or new_kv_page_indices is None
            )
            else new_kv_page_indices
        )

        res = wrapper.ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache_fused,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=True,
            attention_scope=scope_enum,
            cp_group_size=cp_size_arg,
            cp_rank=cp_rank,
            pcp_ring_axis_name=pcp_ring_axis_name,
            pcp_ring_mesh_axis_names=ring_mesh,
            q_positions=q_positions,
            kv_new_lens=kv_new_lens,
            new_kv_page_indices=new_pages,
            update_kv_cache=update_kv_cache,
            return_lse=return_lse,
            kv_layout=kv_layout(),
        )
        if return_lse:
            out, kv_updated, lse = res
            return out, kv_updated, lse
        else:
            # No LSE requested (single-phase prefill, nothing to merge): the
            # kernel skips the LSE buffers and writeback. Keep a fixed-shape
            # placeholder so the op's signature does not change.
            out, kv_updated = res
            return out, kv_updated, jnp.zeros((1, 1), jnp.float32)

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{_CP_OP_NAME}",
        # Static non-traced arguments
        static_argnums=(13, 14, 15, 16, 17, 18, 19, 20, 21),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _CP_OP_NAME,
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
        distribution: torch.Tensor,
        q_positions: torch.Tensor,
        kv_new_lens: torch.Tensor,
        cp_rank: torch.Tensor,
        new_kv_page_indices: torch.Tensor | None = None,
        update_kv_cache: torch.Tensor | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        attention_scope: int = 0,
        cp_group_size: int = 1,
        pcp_ring_axis_name: str | None = None,
        return_lse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, _, lse = jax_kernel(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            q_positions,
            kv_new_lens,
            cp_rank,
            new_kv_page_indices,
            update_kv_cache,
            sm_scale,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
            attention_scope,
            cp_group_size,
            pcp_ring_axis_name,
            return_lse,
        )
        return out, lse

    @custom_op.register_fake
    def _fake_impl(queries, keys, values, kv_cache_fused, *args, **kwargs):
        # `return_lse` is positional index 21 of the op (17 after kv_cache_fused).
        return_lse = kwargs.get("return_lse", args[17] if len(args) > 17 else True)
        if return_lse:
            lse = queries.new_empty(
                queries.shape[0], queries.shape[1], dtype=torch.float32
            )
        else:
            lse = queries.new_empty(1, 1, dtype=torch.float32)
        return torch.empty_like(queries), lse

    _OP_CACHE[_CP_OP_NAME] = custom_op
    return custom_op


def merge_attn_states(
    context_out: torch.Tensor,
    context_lse: torch.Tensor,
    curr_out: torch.Tensor,
    curr_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention states across cache phase and current phase using Log-Sum-Exp.

    Args:
        context_out: [num_tokens, num_heads, head_dim]
        context_lse: [num_tokens, num_heads]
        curr_out: [num_tokens, num_heads, head_dim]
        curr_lse: [num_tokens, num_heads]
    Returns:
        merged_out: [num_tokens, num_heads, head_dim]
        merged_lse: [num_tokens, num_heads]
    """
    out_dtype = context_out.dtype
    context_lse = context_lse.to(torch.float32)
    curr_lse = curr_lse.to(torch.float32)
    max_lse = torch.maximum(context_lse, curr_lse)
    # A phase that attended nothing reports -inf; keep the subtraction finite.
    max_lse_safe = torch.where(torch.isinf(max_lse), torch.zeros_like(max_lse), max_lse)
    exp_context = torch.exp(context_lse - max_lse_safe)
    exp_curr = torch.exp(curr_lse - max_lse_safe)
    exp_sum = exp_context + exp_curr
    exp_sum_safe = torch.where(exp_sum == 0.0, torch.ones_like(exp_sum), exp_sum)

    w_context = (exp_context / exp_sum_safe).unsqueeze(-1)
    w_curr = (exp_curr / exp_sum_safe).unsqueeze(-1)

    merged_out = (
        torch.nan_to_num(context_out.to(torch.float32), nan=0.0) * w_context
        + torch.nan_to_num(curr_out.to(torch.float32), nan=0.0) * w_curr
    )
    merged_lse = max_lse_safe + torch.log(exp_sum_safe)
    return merged_out.to(out_dtype), merged_lse


def build_pcp_batched_metadata(
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    forward_batch: Optional["ForwardBatch"] = None,
    num_local_tokens: Optional[int] = None,
) -> PcpBatchedMetadata:
    """Builds PCP metadata for the 2-phase batched RPA kernel."""
    num_reqs = len(seq_lens)
    two_p = 2 * cp_size

    # Extend tokens per request
    extend_lens = (
        getattr(forward_batch, "extend_seq_lens_cpu", None)
        if forward_batch is not None
        else None
    )
    if extend_lens is None:
        extend_lens = (
            getattr(forward_batch, "seq_lens_cpu", None)
            if forward_batch is not None
            else None
        )
    if extend_lens is None:
        try:
            extend_lens = (seq_lens - prefix_lens).cpu().numpy()
        except Exception:
            extend_lens = [256] * num_reqs

    prefix_lens_cpu = (
        getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if forward_batch is not None
        else None
    )
    if prefix_lens_cpu is None:
        prefix_lens_cpu = (
            getattr(forward_batch, "prefix_lens_cpu", None)
            if forward_batch is not None
            else None
        )
    if prefix_lens_cpu is not None:
        prefix_lens_np = prefix_lens_cpu
    else:
        try:
            prefix_lens_np = prefix_lens.cpu().numpy()
        except Exception:
            prefix_lens_np = [0] * num_reqs

    meta = (
        getattr(forward_batch, "attn_cp_metadata", None)
        if forward_batch is not None
        else None
    )
    if meta is not None and getattr(meta, "actual_seq_q_prev_list", None) is not None:
        pcp_cu = [0]
        q_pos_offsets = []
        pcp_write = []
        kv_new_lens = []
        ring_q_lens = []
        run = 0
        real_b = getattr(
            forward_batch, "_real_b", getattr(forward_batch, "_tpu_real_b", None)
        )
        meta_num_reqs = len(meta.actual_seq_q_prev_list)
        max_prev = (
            getattr(meta, "max_seq_q_prev_list", None) or meta.actual_seq_q_prev_list
        )
        max_next = (
            getattr(meta, "max_seq_q_next_list", None) or meta.actual_seq_q_next_list
        )
        for i in range(num_reqs):
            is_valid_req = (i < real_b) if real_b is not None else (i < meta_num_reqs)
            if i < meta_num_reqs:
                c_head = meta.actual_seq_q_prev_list[i]
                c_tail = meta.actual_seq_q_next_list[i]
                toff_head = meta.kv_len_prev_list[i] - c_head
                toff_tail = meta.kv_len_next_list[i] - c_tail
                ext_len = (
                    int(extend_lens[i])
                    if (extend_lens is not None and i < len(extend_lens))
                    else (c_head + c_tail) * two_p
                )
                m_head = max_prev[i]
                m_tail = max_next[i]
            else:
                c_head = 0
                c_tail = 0
                toff_head = 0
                toff_tail = 0
                ext_len = 0
                m_head = m_tail = 0

            # Head chunk (sequence 2*i)
            run += c_head
            pcp_cu.append(run)
            q_pos_offsets.append(toff_head)
            pcp_write.append(False)  # Head chunk does not commit cache
            kv_new_lens.append(ext_len)
            ring_q_lens.append(m_head)

            # Tail chunk (sequence 2*i + 1)
            run += c_tail
            pcp_cu.append(run)
            q_pos_offsets.append(toff_tail)
            pcp_write.append(is_valid_req)  # Tail chunk commits cache
            kv_new_lens.append(ext_len)
            ring_q_lens.append(m_tail)

        c_per_req = (
            meta.actual_seq_q_prev_list[0]
            if len(meta.actual_seq_q_prev_list) > 0
            else 1
        )
    else:
        real_b = getattr(
            forward_batch, "_real_b", getattr(forward_batch, "_tpu_real_b", None)
        )
        c_sizes = (
            [max(1, int(l) // two_p) for l in extend_lens]
            if extend_lens is not None
            else []
        )
        c_per_req = c_sizes[0] if len(c_sizes) > 0 else 1
        pcp_cu = [0]
        q_pos_offsets = []
        pcp_write = []
        kv_new_lens = []
        ring_q_lens = []
        run = 0
        for i in range(num_reqs):
            is_valid_req = (i < real_b) if real_b is not None else (i < len(c_sizes))
            c_len = c_sizes[i] if i < len(c_sizes) else 0
            prefix_len = prefix_lens_np[i] if i < len(prefix_lens_np) else 0
            ext_len = (
                int(extend_lens[i])
                if (extend_lens is not None and i < len(extend_lens))
                else 0
            )
            toff_head = prefix_len + cp_rank * c_len
            run += c_len
            pcp_cu.append(run)
            q_pos_offsets.append(toff_head)
            pcp_write.append(False)
            kv_new_lens.append(ext_len)

            toff_tail = prefix_len + (two_p - 1 - cp_rank) * c_len
            run += c_len
            pcp_cu.append(run)
            q_pos_offsets.append(toff_tail)
            pcp_write.append(is_valid_req)
            kv_new_lens.append(ext_len)
            ring_q_lens.extend((c_len, c_len))

    pcp_cu_t = torch.tensor(pcp_cu, dtype=torch.int32, device=device)
    ring_q_lens_t = torch.tensor(ring_q_lens, dtype=torch.int32, device=device)
    q_pos_t = torch.tensor(q_pos_offsets, dtype=torch.int32, device=device)
    pcp_write_t = torch.tensor(pcp_write, dtype=torch.bool, device=device)
    kv_new_lens_t = torch.tensor(kv_new_lens, dtype=torch.int32, device=device)

    # Cache phase: page-striped page table
    pages_per_seq = page_indices.shape[0] // num_reqs
    pages_2d = page_indices.view(num_reqs, pages_per_seq)
    owned_pages = pages_2d[:, cp_rank::cp_size]
    pad_len = pages_per_seq - owned_pages.shape[1]
    if pad_len > 0:
        striped_page_indices = torch.nn.functional.pad(
            owned_pages, (0, pad_len), value=0
        ).reshape(-1)
    else:
        striped_page_indices = owned_pages.reshape(-1)

    prefix_list = [int(x) for x in prefix_lens_np] if prefix_lens_np is not None else []
    has_prefix = any(x > 0 for x in prefix_list)
    total_cache_pages = sum(prefix_list) // page_size
    pcp_cache_lens_t = prefix_lens.to(dtype=torch.int32, device=device)

    replicated = _build_replicated_maps(
        meta,
        forward_batch,
        pages_2d,
        prefix_list,
        extend_lens,
        num_reqs,
        page_size,
        cp_size,
        cp_rank,
        device,
    )

    return PcpBatchedMetadata(
        pcp_cu_q_lens=pcp_cu_t,
        q_positions=q_pos_t,
        new_kv_page_indices=None,
        update_kv_cache=pcp_write_t,
        kv_new_lens=kv_new_lens_t,
        striped_page_indices=striped_page_indices,
        cache_pages=total_cache_pages,
        num_reqs=num_reqs,
        pcp_cache_lens=pcp_cache_lens_t,
        c_per_req=c_per_req,
        has_prefix=has_prefix,
        ring_q_lens=ring_q_lens_t,
        **replicated,
    )


_REPLICATED_PREFILL = os.environ.get("SGLANG_TPU_PCP_REPLICATED_PREFILL", "1") != "0"


def _zigzag_layout() -> dict:
    from sglang.srt.layers.cp.zigzag import _TPU_LAYOUT

    return _TPU_LAYOUT


def _gather_prefix_pages() -> bool:
    """Whether a prefix may live only on its owner ranks' stripes.

    Pages written by this path are replicated on every rank; pages written by
    decode (DCP) are not. Without the radix cache a request's prefix is only
    its own earlier chunks, so the gather can be skipped. Forced with
    SGLANG_TPU_PCP_GATHER_PREFIX=1/0.
    """
    env = os.environ.get("SGLANG_TPU_PCP_GATHER_PREFIX")
    if env is not None:
        return env != "0"
    try:
        from sglang.srt.runtime_context import get_server_args

        return not bool(get_server_args().disable_radix_cache)
    except Exception:
        return True


def _build_replicated_maps(
    meta,
    forward_batch,
    pages_2d: torch.Tensor,
    prefix_list: List[int],
    extend_lens,
    num_reqs: int,
    page: int,
    cp: int,
    cp_rank: int,
    device: torch.device,
) -> dict:
    """Host maps for the replicated-pool prefill, or {} when it does not apply.

    Every rank allocates the same pool slots for every page (the allocator is
    replicated) but decode only writes and reads its own stripe. For a
    prefill batch we write the whole new K/V of its requests into those slots
    on every rank, and gather the striped prefix pages the same way, so the
    attention is one FULL-scope launch over the request's ordinary page table
    -- no ring, no new-token phase, no merge.
    """
    if (
        not _REPLICATED_PREFILL
        or meta is None
        or getattr(meta, "new_kv_page_rank", None) is None
        or not getattr(meta, "layout_page", 0)
        or forward_batch is None
    ):
        return {}
    pad_t = int(forward_batch.input_ids.shape[0])
    table = getattr(forward_batch, "cp_new_kv_page_indices", None)
    if table is not None and num_reqs:
        # The shim sized the gather table from the padded local row count.
        pad_t = int(table.shape[0]) // (2 * num_reqs) * page // cp
    pages_per_seq = pages_2d.shape[1]
    n_meta = int(meta.new_kv_page_rank.shape[0])
    rank_of = meta.new_kv_page_rank
    row_of = meta.new_kv_page_row
    n_pages = meta.new_kv_num_pages
    ext = np.asarray(
        [int(extend_lens[i]) if i < len(extend_lens) else 0 for i in range(n_meta)]
    )
    pre = np.asarray(
        [prefix_list[i] if i < len(prefix_list) else 0 for i in range(n_meta)]
    )

    # --- new K/V rows: gathered row (rank-major, pad_t rows per rank) -> (req, abs pos)
    b_idx, p_idx = np.nonzero(n_pages[:, None] > np.arange(rank_of.shape[1])[None, :])
    b_idx = b_idx[b_idx < num_reqs]
    p_idx = p_idx[: len(b_idx)]
    t0 = p_idx * page
    n_tok = np.clip(ext[b_idx] - t0, 0, page)
    keep = n_tok > 0
    b_idx, p_idx, t0, n_tok = b_idx[keep], p_idx[keep], t0[keep], n_tok[keep]
    src0 = rank_of[b_idx, p_idx].astype(np.int64) * pad_t + row_of[b_idx, p_idx]
    total = int(n_tok.sum())
    rows = cp * pad_t
    row_req = np.zeros((rows,), dtype=np.int64)
    row_pos = np.zeros((rows,), dtype=np.int64)
    row_valid = np.zeros((rows,), dtype=bool)
    if total:
        ends = np.cumsum(n_tok)
        begins = ends - n_tok
        idx = np.arange(total)
        which = np.searchsorted(ends, idx, side="right")
        j = idx - begins[which]
        src = src0[which] + j
        row_req[src] = b_idx[which]
        row_pos[src] = pre[b_idx[which]] + t0[which] + j
        row_valid[src] = True
    if int(row_pos.max(initial=0)) >= pages_per_seq * page:
        return {}  # beyond the page table (should not happen)
    leads = getattr(meta, "layout_leads", None)
    pool_aligned = bool(_zigzag_layout().get("pool_align")) and leads is not None
    page_src = boundary_pos = boundary_lead = None
    if pool_aligned:
        # Layout page p of request b is pool page pre // page + p.
        lead = np.asarray([leads[i] if i < len(leads) else 0 for i in range(n_meta)])
        p_max = rows // page + 1
        pb_, pp_ = np.nonzero(n_pages[:, None] > np.arange(rank_of.shape[1])[None, :])
        pb_ = pb_[pb_ < num_reqs]
        pp_ = pp_[: len(pb_)]
        if len(pb_) + 1 > p_max:
            return {}
        t_req = np.zeros((p_max,), np.int64)
        t_page = np.zeros((p_max,), np.int64)
        t_valid = np.zeros((p_max,), bool)
        t_src = np.zeros((p_max,), np.int64)
        t_req[1 : 1 + len(pb_)] = pb_
        t_page[1 : 1 + len(pb_)] = pre[pb_] // page + pp_
        t_valid[1 : 1 + len(pb_)] = True
        t_src[1 : 1 + len(pb_)] = (
            rank_of[pb_, pp_].astype(np.int64) * pad_t + row_of[pb_, pp_]
        ) // page
        b_pos = np.zeros((num_reqs,), np.int64)
        b_lead = np.zeros((num_reqs,), np.int64)
        first = (pp_ == 0) & (lead[pb_] > 0)
        b_pos[pb_[first]] = np.nonzero(first)[0] + 1
        b_lead[pb_[first]] = lead[pb_[first]]
        if int(t_page.max(initial=0)) >= pages_per_seq:
            return {}
    if pool_aligned:
        tidx = np.zeros((rows,), np.int64)  # row scatter unused
    else:
        # Pages the new tokens land in, in (req, page) order; index 0 = dump page.
        p_max = rows // page + num_reqs + 1
        key = np.where(row_valid, row_req * pages_per_seq + row_pos // page, -1)
        uniq, inverse = np.unique(key, return_inverse=True)
        # np.unique sorts; -1 (padding) is first when present.
        has_pad = bool(len(uniq)) and uniq[0] == -1
        touched = uniq[1:] if has_pad else uniq
        if len(touched) + 1 > p_max:
            return {}
        tidx = inverse - (1 if has_pad else 0) + 1  # 1-based touched index, pad -> 0
        tidx = np.where(row_valid, tidx, 0)
        t_req = np.zeros((p_max,), np.int64)
        t_page = np.zeros((p_max,), np.int64)
        t_valid = np.zeros((p_max,), bool)
        t_req[1 : 1 + len(touched)] = touched // pages_per_seq
        t_page[1 : 1 + len(touched)] = touched % pages_per_seq
        t_valid[1 : 1 + len(touched)] = True

    # --- prefix pages: owner rank r holds pages p % cp == r of each request's prefix.
    n_pref = (pre + page - 1) // page
    pb, pp = np.nonzero(
        n_pref[:, None] > np.arange(max(int(n_pref.max(initial=0)), 1))[None, :]
    )
    pb = pb[pb < num_reqs]
    pp = pp[: len(pb)]
    r_max = -(-pages_per_seq // cp)
    counts = (
        np.bincount(pp % cp, minlength=cp) if len(pb) else np.zeros(cp, dtype=np.int64)
    )
    if counts.max(initial=0) > r_max:
        return {}  # too much prefix for the static gather; ring fallback
    send_b = np.zeros((r_max,), np.int64)
    send_p = np.zeros((r_max,), np.int64)
    send_valid = np.zeros((r_max,), bool)
    recv_b = np.zeros((cp * r_max,), np.int64)
    recv_p = np.zeros((cp * r_max,), np.int64)
    recv_valid = np.zeros((cp * r_max,), bool)
    for r in range(cp):
        sel = pp % cp == r
        k = int(sel.sum())
        recv_b[r * r_max : r * r_max + k] = pb[sel]
        recv_p[r * r_max : r * r_max + k] = pp[sel]
        recv_valid[r * r_max : r * r_max + k] = True
        if r == cp_rank:
            send_b[:k] = pb[sel]
            send_p[:k] = pp[sel]
            send_valid[:k] = True

    def _t(a, dt=torch.int64):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)

    pages_i64 = pages_2d.to(torch.int64)
    kv_row_dst = _t(np.where(row_valid, tidx * page + row_pos % page, 0))
    touched_slots = torch.where(
        _t(t_valid, torch.bool),
        pages_i64[_t(t_req), _t(t_page)],
        torch.zeros((p_max,), dtype=torch.int64, device=device),
    )
    if pool_aligned:
        page_src = _t(t_src, torch.int32)
        boundary_pos = _t(b_pos, torch.int32)
        boundary_lead = _t(b_lead, torch.int32)
    send_slot = torch.where(
        _t(send_valid, torch.bool),
        pages_i64[_t(send_b), _t(send_p)],
        torch.zeros((r_max,), dtype=torch.int64, device=device),
    )
    recv_slot = torch.where(
        _t(recv_valid, torch.bool),
        pages_i64[_t(recv_b), _t(recv_p)],
        torch.zeros((cp * r_max,), dtype=torch.int64, device=device),
    )
    kv_lens_full = torch.from_numpy(
        np.repeat((pre[:num_reqs] + ext[:num_reqs]).astype(np.int32), 2)
    ).to(device)
    if kv_lens_full.shape[0] < 2 * num_reqs:
        kv_lens_full = torch.nn.functional.pad(
            kv_lens_full, (0, 2 * num_reqs - kv_lens_full.shape[0])
        )
    full_page_indices = pages_2d.repeat_interleave(2, dim=0).reshape(-1)
    ends = np.zeros((2 * num_reqs,), dtype=np.int32)
    kp = getattr(meta, "kv_len_prev_list", None) or []
    kn = getattr(meta, "kv_len_next_list", None) or []
    for i in range(min(num_reqs, len(kp))):
        ends[2 * i] = kp[i]
        ends[2 * i + 1] = kn[i]
    kv_lens_ends = torch.from_numpy(ends).to(device)
    return dict(
        kv_lens_ends=kv_lens_ends,
        page_src=page_src,
        boundary_pos=boundary_pos,
        boundary_lead=boundary_lead,
        kv_row_dst=kv_row_dst.to(torch.int32),
        touched_slots=touched_slots.to(torch.int32),
        prefix_send=send_slot.to(torch.int32),
        prefix_recv=recv_slot.to(torch.int32),
        kv_lens_full=kv_lens_full,
        full_page_indices=full_page_indices,
    )


def batched_rpa_pcp_forward(
    cp_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    layer: "RadixAttention",
    forward_batch: "ForwardBatch",
    pcp_meta: PcpBatchedMetadata,
    base_meta,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sinks: torch.Tensor | None = None,
    quantize_kv=None,
) -> torch.Tensor:
    """Executes the two-phase PCP forward pass on Batched RPA.

    1. Cache Phase: Attends sharded KV cache via ICI Ring Attention (CACHE_ONLY).
    2. Current Phase: All-gathers current tokens and attends causally (NEW_TOKENS_ONLY).
    3. Merges softmax states using Log-Sum-Exp.

    ``quantize_kv(k, v) -> (k, v)`` casts K/V into the (fp8) pool dtype. It is
    applied after the all-gather, which must move the compute-dtype rows: an
    8-bit all-gather returns wrong remote rows on libtpu 0.0.47.
    """
    if getattr(pcp_meta, "kv_row_dst", None) is not None:
        return batched_rpa_pcp_forward_replicated(
            cp_op,
            q,
            k,
            v,
            kv_cache,
            layer,
            forward_batch,
            pcp_meta,
            base_meta,
            k_scale=k_scale,
            v_scale=v_scale,
            sinks=sinks,
            quantize_kv=quantize_kv,
        )

    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    device = q.device
    cp_rank_tensor = ensure_cp_rank_tensor(
        device, cp_rank, ref_tensor=base_meta.kv_lens
    )
    num_reqs = pcp_meta.num_reqs

    from sglang.srt.layers.cp.utils import get_cp_strategy

    strategy = get_cp_strategy()

    q_op = q
    k_logical = k
    v_logical = v

    sliding_window = getattr(layer, "sliding_window_size", None)
    if sliding_window is not None and sliding_window <= 0:
        sliding_window = None
    elif sliding_window is not None:
        sliding_window += 1

    is_swa_layer = sliding_window is not None
    if is_swa_layer and getattr(base_meta, "swa_page_indices", None) is not None:
        raw_pages = base_meta.swa_page_indices
    else:
        raw_pages = base_meta.page_indices

    # Both phases see the same local Q layout: per request, a head chunk
    # (virtual sequence 2*i) followed by a tail chunk (2*i + 1). All per-sequence
    # metadata is therefore sized 2 * num_reqs, and every virtual sequence
    # carries its request's page-striped page table (rank r owns pages
    # p % cp_size == r in local slot p // cp_size, for cache and new KV alike).
    pages_per_seq = raw_pages.shape[0] // num_reqs
    pages_2d = raw_pages.view(num_reqs, pages_per_seq)
    owned_pages = pages_2d[:, cp_rank::cp_size]
    pad_len = pages_per_seq - owned_pages.shape[1]
    if pad_len > 0:
        striped_curr_pages = torch.nn.functional.pad(owned_pages, (0, pad_len), value=0)
    else:
        striped_curr_pages = owned_pages
    curr_page_indices = striped_curr_pages.repeat_interleave(2, dim=0).reshape(-1)
    curr_distribution = base_meta.distribution * 2
    cache_lens_2x = pcp_meta.pcp_cache_lens.repeat_interleave(2)
    zeros_2x = torch.zeros_like(cache_lens_2x, dtype=torch.int32)
    ring_q_lens = getattr(pcp_meta, "ring_q_lens", None)
    if ring_q_lens is None:
        # Equal chunks on every rank: this rank's own Q lengths.
        ring_q_lens = pcp_meta.pcp_cu_q_lens[1:] - pcp_meta.pcp_cu_q_lens[:-1]
    ring_q_lens = ring_q_lens.to(dtype=torch.int32, device=device)

    # 2. Current Phase (All-Gather current chunk tokens)
    new_kv_page_indices = getattr(forward_batch, "cp_new_kv_page_indices", None)
    if (
        strategy is not None
        and getattr(forward_batch, "attn_cp_metadata", None) is not None
    ):
        # Gather K and V as one flat [T, 2 * heads * head_dim] tensor in the
        # compute dtype. On TPU a [T, heads, head_dim] gather with small minor
        # dims (e.g. (2, 64) bf16) came back with NaN rows in a tiled pattern;
        # a lane-dense 2-D tensor (like the hidden-state gathers) is moved
        # correctly. The gather must not run on fp8 data (see `quantize_kv`).
        num_local = k.shape[0]
        kd = k.shape[1] * k.shape[2]
        vd = v.shape[1] * v.shape[2]
        kv_local = torch.cat(
            [k.reshape(num_local, kd), v.reshape(num_local, vd)], dim=-1
        ).contiguous()
        if new_kv_page_indices is not None:
            # Page-aligned zigzag layout: the kernel addresses the rank-major
            # gathered buffer through `new_kv_page_indices`; no row reorder.
            group = get_parallel().attn_cp_group
            kv_glob = kv_local.new_empty((cp_size * num_local, kv_local.shape[1]))
            group.all_gather_into_tensor(kv_glob, kv_local)
        else:
            kv_glob = cp_all_gather_rows(kv_local, forward_batch)
        num_glob = kv_glob.shape[0]
        k_reordered = (
            kv_glob[:, :kd].reshape(num_glob, k.shape[1], k.shape[2]).contiguous()
        )
        v_reordered = (
            kv_glob[:, kd:].reshape(num_glob, v.shape[1], v.shape[2]).contiguous()
        )
    else:
        k_curr = get_parallel().attn_cp_group.all_gather(k, dim=0)
        v_curr = get_parallel().attn_cp_group.all_gather(v, dim=0)
        c_per_req = max(1, q.shape[0] // (2 * num_reqs))
        k_curr_5d = k_curr.view(cp_size, num_reqs, 2, c_per_req, k.shape[1], k.shape[2])
        head_k = k_curr_5d[:, :, 0].permute(1, 0, 2, 3, 4)
        tail_k = k_curr_5d[:, :, 1].flip(0).permute(1, 0, 2, 3, 4)
        k_reordered = torch.cat([head_k, tail_k], dim=1).reshape(
            -1, k.shape[1], k.shape[2]
        )

        v_curr_5d = v_curr.view(cp_size, num_reqs, 2, c_per_req, v.shape[1], v.shape[2])
        head_v = v_curr_5d[:, :, 0].permute(1, 0, 2, 3, 4)
        tail_v = v_curr_5d[:, :, 1].flip(0).permute(1, 0, 2, 3, 4)
        v_reordered = torch.cat([head_v, tail_v], dim=1).reshape(
            -1, v.shape[1], v.shape[2]
        )

    if quantize_kv is not None:
        k_reordered, v_reordered = quantize_kv(k_reordered, v_reordered)
        k_logical, v_logical = quantize_kv(k_logical, v_logical)
        k_logical = k_logical.contiguous()
        v_logical = v_logical.contiguous()
        k_reordered = k_reordered.contiguous()
        v_reordered = v_reordered.contiguous()

    # 1. Cache Phase (Ring Attention over ICI), only when some request has a
    # cached prefix. `cache_pages` is a host int, so this is a trace-time
    # branch; the compile policy keys the program on `cp_has_prefix`
    # (decoder_compile_policy._batch_has_prefix) and warms both variants, so
    # the two shapes never share a Dynamo cache entry.
    has_prefix = bool(getattr(pcp_meta, "has_prefix", True))
    context_out = context_lse = None
    if has_prefix:
        ring_op = _GLOBAL_RING_OP
        if ring_op is None:
            ring_op = build_batched_rpa_pcp_ring_op(
                cp_size, get_parallel().attn_tp_size
            )
        context_out, context_lse = ring_op(
            q_op,
            k_logical,
            v_logical,
            kv_cache,
            cache_lens_2x,
            curr_page_indices,
            pcp_meta.pcp_cu_q_lens,
            curr_distribution,
            pcp_meta.q_positions,
            zeros_2x,
            cp_rank_tensor,
            zeros_2x,
            ring_q_lens,
            float(layer.scaling),
            sliding_window,
            float(layer.logit_cap) if getattr(layer, "logit_cap", 0) else None,
            k_scale,
            v_scale,
        )

    # Each virtual sequence's KV is the cached prefix plus the whole
    # all-gathered new KV of its request; causality is enforced by q_positions.
    curr_kv_lens = cache_lens_2x + pcp_meta.kv_new_lens

    if k_reordered.shape[0] < q_op.shape[0]:
        k_reordered = torch.nn.functional.pad(
            k_reordered, (0, 0, 0, 0, 0, q_op.shape[0] - k_reordered.shape[0])
        )
        v_reordered = torch.nn.functional.pad(
            v_reordered, (0, 0, 0, 0, 0, q_op.shape[0] - v_reordered.shape[0])
        )

    curr_update_kv = pcp_meta.update_kv_cache.to(dtype=torch.int32, device=device)
    if sinks is None:
        sinks = getattr(layer, "sinks", None)
    # The LSE is only needed to merge with the cache phase (or for attention
    # sinks); without a prefix the new-token phase is the whole result and the
    # kernel can skip the LSE buffers (8 rows x 128 lanes x f32 per token and
    # KV head) and their writeback.
    need_lse = bool(has_prefix) or sinks is not None
    curr_out, curr_lse = cp_op(
        q_op,
        k_reordered,
        v_reordered,
        kv_cache,
        curr_kv_lens,
        curr_page_indices,
        pcp_meta.pcp_cu_q_lens,
        curr_distribution,
        pcp_meta.q_positions,
        pcp_meta.kv_new_lens,
        cp_rank_tensor,
        new_kv_page_indices,
        curr_update_kv,
        float(layer.scaling),
        sliding_window,
        layer.logit_cap if getattr(layer, "logit_cap", 0) else None,
        k_scale,
        v_scale,
        2,  # NEW_TOKENS_ONLY
        cp_size,
        None,
        need_lse,
    )

    if _PCP_DEBUG_DUMP:
        _pcp_debug_dump(
            layer,
            cp_rank,
            q_op,
            k,
            v,
            k_reordered,
            v_reordered,
            kv_cache,
            curr_kv_lens,
            curr_page_indices,
            pcp_meta,
            curr_distribution,
            curr_update_kv,
            context_out,
            context_lse,
            curr_out,
            curr_lse,
            sliding_window,
            float(layer.scaling),
            cp_size,
            forward_batch,
        )

    # 3. Softmax State Merge (no prefix: the new-token phase is the whole result)
    if has_prefix:
        out, lse = merge_attn_states(context_out, context_lse, curr_out, curr_lse)
    else:
        out, lse = curr_out, curr_lse

    # Fold in attention sinks if present (e.g. GPT-OSS-120B)
    if sinks is not None:
        out = out * torch.sigmoid(lse - sinks.to(torch.float32))[..., None]
        out = out.to(q.dtype)

    return out


def batched_rpa_pcp_forward_replicated(
    cp_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    layer: "RadixAttention",
    forward_batch: "ForwardBatch",
    pcp_meta: PcpBatchedMetadata,
    base_meta,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sinks: torch.Tensor | None = None,
    quantize_kv=None,
) -> torch.Tensor:
    """PCP prefill with the batch's K/V replicated into every rank's pool.

    Per layer: quantize this rank's new K/V, all-gather it (as int32 words)
    and scatter every row into its pool slot on this rank (owned pages and
    the other ranks' pages alike -- the slots exist on every rank); gather
    the striped prefix pages the same way; then run one FULL-scope launch of
    the plain (non-CP) kernel over the request's ordinary page table. Decode
    keeps reading its own stripe, which this path also writes.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    device = q.device
    group = get_parallel().attn_cp_group
    cp_rank_tensor = ensure_cp_rank_tensor(
        device, cp_rank, ref_tensor=base_meta.kv_lens
    )

    sliding_window = getattr(layer, "sliding_window_size", None)
    if sliding_window is not None and sliding_window <= 0:
        sliding_window = None
    elif sliding_window is not None:
        sliding_window += 1

    # Built at backend init (building them under Dynamo is not traceable).
    # The pool is only touched through the aliased page kernels and the rows
    # travel in bf16 / the pool dtype: a torch dtype view of fp8, a whole-pool
    # view or an 8-bit bitcast on TPU each cost 100+ ms per step.
    read_pages = torch.ops.sglang_tpu.kv_read_pages
    write_pages = torch.ops.sglang_tpu.kv_write_pages
    pool_dtype = kv_cache.dtype

    # 1. Prefix pages: each rank contributes its owned pages, every rank
    #    writes all of them into their slots (padding -> the dump slot 0).
    #    Moved as bf16 (exact for fp8; an 8-bit all-gather is unreliable on
    #    libtpu 0.0.47). Skipped when every prefix is this request's own
    #    earlier chunks, which this path already replicated.
    if bool(getattr(pcp_meta, "has_prefix", True)) and _gather_prefix_pages():
        send = read_pages(pcp_meta.prefix_send, kv_cache)
        send_bf = send.to(torch.bfloat16).contiguous()
        recv_bf = send_bf.new_empty((cp_size * send_bf.shape[0], *send_bf.shape[1:]))
        group.all_gather_into_tensor(recv_bf, send_bf)
        write_pages(pcp_meta.prefix_recv, recv_bf.to(pool_dtype).contiguous(), kv_cache)

    # 2. New K/V: all-gather in the compute dtype, quantize, and scatter the
    #    rows into the pages they land in (read back first, so a page shared
    #    with the prefix keeps its prefix rows). Row layout matches the
    #    kernel's new-KV packing: [k_h, v_h] per head.
    num_local = k.shape[0]
    kd = k.shape[1] * k.shape[2]
    vd = v.shape[1] * v.shape[2]
    kv_local = torch.cat(
        [k.reshape(num_local, kd), v.reshape(num_local, vd)], dim=-1
    ).contiguous()
    kv_all = kv_local.new_empty((cp_size * num_local, kv_local.shape[1]))
    group.all_gather_into_tensor(kv_all, kv_local)
    k_all = kv_all[:, :kd].reshape(-1, k.shape[1], k.shape[2])
    v_all = kv_all[:, kd:].reshape(-1, v.shape[1], v.shape[2])
    if quantize_kv is not None:
        k_all, v_all = quantize_kv(k_all, v_all)
        kq, vq = quantize_kv(k, v)
    else:
        kq, vq = k, v
    rows = torch.cat([k_all, v_all], dim=-1).reshape(kv_all.shape[0], -1).contiguous()
    if getattr(pcp_meta, "page_src", None) is not None:
        # Pool-aligned layout: every layout page is one pool page.
        torch.ops.sglang_tpu.kv_write_pages_from_rows(
            pcp_meta.touched_slots,
            pcp_meta.page_src,
            pcp_meta.boundary_pos,
            pcp_meta.boundary_lead,
            rows,
            kv_cache,
        )
    else:
        torch.ops.sglang_tpu.kv_write_rows_scatter(
            pcp_meta.touched_slots, pcp_meta.kv_row_dst, rows, kv_cache
        )
    kq = kq.contiguous()
    vq = vq.contiguous()

    # 3. One FULL-scope launch: cache = prefix + new, nothing new to the kernel.
    two_x = pcp_meta.kv_lens_full.shape[0]
    zeros_2x = torch.zeros((two_x,), dtype=torch.int32, device=device)
    if sinks is None:
        sinks = getattr(layer, "sinks", None)
    need_lse = sinks is not None
    if _USE_REF_KERNEL and sinks is None:
        ref_op = _OP_CACHE.get(_REF_FULL_OP_NAME) or build_batched_rpa_ref_full_op()
        return ref_op(
            q,
            kq,
            vq,
            kv_cache,
            pcp_meta.kv_lens_ends,
            pcp_meta.full_page_indices,
            pcp_meta.pcp_cu_q_lens,
            base_meta.distribution * 2,
            float(layer.scaling),
            sliding_window,
            float(layer.logit_cap) if getattr(layer, "logit_cap", 0) else None,
            k_scale,
            v_scale,
        )
    # The kernel takes no new tokens here (kv_new_lens = 0) but is compiled
    # for the pool dtype, so hand it the quantized local K/V.
    out, lse = cp_op(
        q,
        kq,
        vq,
        kv_cache,
        pcp_meta.kv_lens_full,
        pcp_meta.full_page_indices,
        pcp_meta.pcp_cu_q_lens,
        base_meta.distribution * 2,
        pcp_meta.q_positions,
        zeros_2x,
        cp_rank_tensor,
        None,
        zeros_2x,
        float(layer.scaling),
        sliding_window,
        float(layer.logit_cap) if getattr(layer, "logit_cap", 0) else None,
        k_scale,
        v_scale,
        0,  # FULL
        1,  # no CP inside the kernel
        None,
        need_lse,
    )
    if sinks is not None:
        out = out * torch.sigmoid(lse - sinks.to(torch.float32))[..., None]
        out = out.to(q.dtype)
    return out


_DCP_FUSED_DECODE = __import__("os").environ.get("SGLANG_DCP_FUSED_DECODE", "1") == "1"
