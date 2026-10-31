# SPDX-License-Identifier: Apache-2.0
"""Decode-context-parallel (DCP) pieces of the TPU Pallas batched RPA backend.

Under DCP the KV of a sequence is split across the ranks of one DCP group by
absolute position: rank ``r`` stores position ``p`` iff ``p % dcp == r``. Each
rank therefore attends only its own shard, and the partial results are merged
across the group by log-sum-exp.

This module holds the parts of that scheme that are independent of the backend
object: the torch custom op for the CP-aware kernel, the fused-pool row layout
and the owner-masked write that replaces the kernel's own KV write, and the
pure-torch merges (cross-rank sink fold, two-pass extend combine).

``attention_pallas_rpa_base`` drives them; the non-DCP path never calls in here.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from typing import NamedTuple

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

__all__ = [
    "DCP_OP_NAME",
    "DcpWriteSpans",
    "broadcast_own_kv_heads",
    "build_dcp_kernel_op",
    "build_owner_write_spans",
    "fused_kv_rows",
    "gather_heads",
    "merge_across_ranks",
    "merge_lse",
    "sink_fold",
    "write_owned_kv",
]

# torch custom-op name of the CP-aware kernel. Distinct from the plain batched op
# so both can be registered in one process and keyed separately by the FX-graph
# compile cache.
DCP_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_dcp"

# torch custom ops are process-global; cache the registration per kernel spec so
# repeated backend construction reuses it (mirrors ``_OP_CACHE`` in the base
# module).
_OP_CACHE: dict = {}


def _import_dcp_kernel():
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def build_dcp_kernel_op(spec):
    """Register the batched ragged_paged_attention kernel (module
    ``sglang_tpu.attention.pallas.batched_rpa.wrapper``) in its CP-aware form
    as a torch custom op for ``spec``'s backend.

    The op mirrors the base module's ``_build_pallas_kernel_op`` and adds the
    DCP arguments of the kernel contract: the global KV lengths and this rank's
    index in the group (both traced, so programs stay rank-uniform), and the
    static ``cache_only`` / ``skip_kv_write`` / ``cp_group_size`` flags. It
    returns ``(output, lse)``; ``lse`` is the natural-log
    ``m + log(l)`` per (token, query head) that the cross-rank merge consumes.

    ``cp_group_size == 0`` means "no context parallelism" and drops the CP
    arguments on the JAX side.
    """
    from sglang_tpu.attention_pallas_rpa_batched import (
        RPA_BATCHED_SPEC,
        kv_layout,
    )

    if spec is not RPA_BATCHED_SPEC:
        raise NotImplementedError(
            f"{spec.op_name} is not the batched RPA kernel; decode context "
            "parallelism on TPU requires --attention-backend pallas_tpu_rpa_batched."
        )

    cached = _OP_CACHE.get(spec.op_name)
    if cached is not None:
        return cached

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    _jax_rpa = _import_dcp_kernel()

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        kv_lens_global: jax.Array,
        cp_rank: jax.Array,
        attention_sink: jax.Array | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        cache_only: bool = False,
        skip_kv_write: bool = False,
        cp_group_size: int = 0,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # The kernel takes ``cp_group_size=None`` for the non-CP form and then
        # rejects the two CP operands, so drop them together.
        cp_size = cp_group_size if cp_group_size else None
        return _jax_rpa(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=True,
            skip_kv_update=False,
            kv_layout=kv_layout(),
            attention_sink=attention_sink,
            cache_only=cache_only,
            skip_kv_write=skip_kv_write,
            return_lse=True,
            cp_group_size=cp_size,
            cp_rank=cp_rank if cp_size else None,
            kv_lens_global=kv_lens_global if cp_size else None,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{DCP_OP_NAME}",
        # Static (non-traced) args by _wrapped position: 11=sm_scale,
        # 12=sliding_window, 13=soft_cap, 14=k_scale, 15=v_scale,
        # 16=cache_only, 17=skip_kv_write, 18=cp_group_size.
        static_argnums=(11, 12, 13, 14, 15, 16, 17, 18),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        DCP_OP_NAME,
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
        kv_lens_global: torch.Tensor,
        cp_rank: torch.Tensor,
        attention_sink: torch.Tensor | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        cache_only: bool = False,
        skip_kv_write: bool = False,
        cp_group_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with tpu_info_for_cpu_export():
            output, _updated_kv, lse = jax_kernel(
                queries,
                keys,
                values,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                kv_lens_global,
                cp_rank,
                attention_sink,
                sm_scale,
                sliding_window,
                soft_cap,
                k_scale,
                v_scale,
                cache_only,
                skip_kv_write,
                cp_group_size,
            )
        return output, lse

    def _fake_impl(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        *args,
        **kwargs,
    ):
        return (
            torch.empty_like(queries),
            queries.new_empty(queries.shape[:2], dtype=torch.float32),
        )

    custom_op.register_fake(_fake_impl)

    _OP_CACHE[spec.op_name] = custom_op
    return custom_op


def fused_kv_rows(
    k: torch.Tensor, v: torch.Tensor, kv_cache: torch.Tensor
) -> torch.Tensor:
    """Rows ``[T, packed, packing, padded_head_dim]`` of ``k``/``v`` (each
    ``[T, heads, head_dim]``) in the fused pool layout of ``kv_cache``.

    K and V are interleaved per head, the head axis is padded to the pool's
    packing and the head dim to 128 -- the layout the kernel's own
    ``prepare_inputs`` produces for its KV write.
    """
    num_tokens, heads, head_dim = k.shape
    packed, packing, padded_head_dim = kv_cache.shape[2:]
    rows = torch.stack([k, v], dim=2).reshape(num_tokens, 2 * heads, head_dim)
    pad_dim = padded_head_dim - head_dim
    pad_heads = packed * packing - 2 * heads
    if pad_dim or pad_heads:
        rows = torch.nn.functional.pad(rows, (0, pad_dim, 0, pad_heads))
    return rows.reshape(num_tokens, packed, packing, padded_head_dim)


def broadcast_own_kv_heads(
    k: torch.Tensor, v: torch.Tensor, dcp_size: int, rank_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``k``/``v`` with every local KV head replaced by one this rank's q heads own.

    K/V are sharded over ``attn_tp // dcp`` groups, so every rank of a DCP group
    holds the same ``Hkv`` heads while its own query heads cover only the
    ``Hkv // dcp`` wide slice that starts at its rank. A kernel call that
    passes this rank's query heads alone still has them grouped over all ``Hkv``
    heads, and the head count has to stay ``Hkv`` because the kernel stages new
    and cached K/V into one buffer shaped by the cache -- so the slice is
    repeated back up to ``Hkv`` heads, ``dcp_size`` copies of each.

    ``rank_index`` is the rank as an ``i64[1]`` device tensor, so every rank
    traces the same program (see ``_all_reduce_gather``).

    Kernel calls that pass the group's gathered query heads take ``k``/``v``
    unchanged.
    """
    num_tokens, heads, head_dim = k.shape
    if heads % dcp_size and dcp_size % heads:
        raise ValueError(
            f"{heads} local KV heads do not divide evenly over a DCP group of "
            f"{dcp_size}: the query heads of one rank would straddle two KV "
            "heads, which the ragged-paged-attention head grouping cannot "
            "express."
        )
    own = max(1, heads // dcp_size)
    if own == heads:
        return k, v
    reps = heads // own
    start = torch.div(rank_index * heads, dcp_size, rounding_mode="floor")
    index = start + torch.arange(own, device=k.device, dtype=torch.int64)

    def _own(x: torch.Tensor) -> torch.Tensor:
        return (
            x.index_select(1, index)[:, :, None, :]
            .expand(num_tokens, own, reps, head_dim)
            .reshape(num_tokens, heads, head_dim)
        )

    return _own(k), _own(v)


class DcpWriteSpans(NamedTuple):
    """Host-built targets of one step's owner-masked KV write.

    ``index`` (i64[num_tokens]) gathers this rank's owned rows to the front in
    token order; the spans (i32[max_spans] each) then name, per run of
    consecutive physical slots, the first slot (``dst``), the first compacted
    row (``src``) and the row count (``lens``, 0 for padding spans).
    """

    index: torch.Tensor
    dst: torch.Tensor
    src: torch.Tensor
    lens: torch.Tensor

    def to_device(self, device) -> "DcpWriteSpans":
        return DcpWriteSpans(*(t.to(device, non_blocking=True) for t in self))


def max_write_spans(num_tokens: int, num_seqs: int, page_size: int) -> int:
    """Static span-table length for a token bucket: every request opens a span
    and every page it touches may open another."""
    return 2 * num_seqs + num_tokens // page_size + 1


def build_owner_write_spans(
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    dcp_size: int,
    rank: int,
    page_size: int,
    num_tokens: int,
    num_seqs: int,
) -> DcpWriteSpans:
    """Compaction index and DMA spans that write this rank's rows of a step.

    A token at position ``p`` with widened logical slot ``loc`` belongs to
    rank ``p % dcp_size`` and lives in physical slot ``loc // dcp_size``.
    Consecutive owned tokens of one request share a page in consecutive
    physical slots, so a span ends only where the physical slot jumps.
    """
    loc = out_cache_loc.to(torch.int64)
    owned = torch.nonzero(positions.to(torch.int64) % dcp_size == rank).flatten()
    phys = loc[owned] // dcp_size
    index = torch.zeros(num_tokens, dtype=torch.int64)
    index[: owned.numel()] = owned
    max_spans = max_write_spans(num_tokens, num_seqs, page_size)
    dst = torch.zeros(max_spans, dtype=torch.int32)
    src = torch.zeros(max_spans, dtype=torch.int32)
    lens = torch.zeros(max_spans, dtype=torch.int32)
    if owned.numel():
        jumps = (phys[1:] != phys[:-1] + 1) | (phys[1:] % page_size == 0)
        breaks = torch.nonzero(jumps).flatten() + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), breaks])
        ends = torch.cat([breaks, torch.tensor([owned.numel()], dtype=torch.int64)])
        if starts.numel() > max_spans:
            raise ValueError(
                f"{starts.numel()} write spans exceed the {max_spans}-span table "
                f"of a {num_tokens}-token, {num_seqs}-sequence bucket."
            )
        n = starts.numel()
        dst[:n] = phys[starts].to(torch.int32)
        src[:n] = starts.to(torch.int32)
        lens[:n] = (ends - starts).to(torch.int32)
    return DcpWriteSpans(index, dst, src, lens)


@compiled_op()
def build_verify_step_on_device(
    seq_lens: torch.Tensor,
    page_indices: torch.Tensor,
    swa_page_indices,
    verify_width: int,
    dcp_size: int,
    rank: int,
    page_size: int,
    num_tokens: int,
):
    bs = seq_lens.shape[0]
    device = seq_lens.device
    i32 = torch.int32
    prefix = seq_lens.to(i32)
    live = prefix > 0
    total = prefix + verify_width
    zero = torch.zeros_like(prefix)

    def shard(lengths):
        return lengths // dcp_size + (lengths % dcp_size > rank).to(i32)

    kv_lens = torch.where(live, shard(total), zero)
    global_lens = torch.where(live, total, zero)
    prefix_lens = torch.where(live, shard(prefix), zero)

    stride = page_size * dcp_size
    pos = (
        prefix[:, None] + torch.arange(verify_width, dtype=i32, device=device)[None, :]
    )
    owned = (pos % dcp_size == rank) & live[:, None]
    count = owned.sum(1, dtype=i32)
    first = (dcp_size - prefix % dcp_size + rank) % dcp_size
    row = torch.arange(bs, dtype=i32, device=device)
    start = ((row[None, :] < row[:, None]).to(i32) * count[None, :]).sum(1, dtype=i32)
    t = torch.arange(num_tokens, dtype=i32, device=device)
    row_of = ((start + count)[None, :] <= t[:, None]).sum(1, dtype=i32)
    pick = (row_of[:, None] == row[None, :]).to(i32)
    first_t = (pick * first[None, :]).sum(1, dtype=i32)
    start_t = (pick * start[None, :]).sum(1, dtype=i32)
    index = torch.where(
        row_of < bs, row_of * verify_width + first_t + (t - start_t) * dcp_size, 0
    ).to(torch.int64)
    max_spans = max_write_spans(num_tokens, bs, page_size)

    def spans(table):
        rows = table.reshape(bs, -1).to(i32)
        cols = torch.arange(rows.shape[1], dtype=i32, device=device)[None, :]
        col0 = (prefix // stride).clamp(max=rows.shape[1] - 1)
        col1 = (col0 + 1).clamp(max=rows.shape[1] - 1)
        page0 = (rows * (cols == col0[:, None]).to(i32)).sum(1, dtype=i32)
        page1 = (rows * (cols == col1[:, None]).to(i32)).sum(1, dtype=i32)
        in_first = (pos // stride).clamp(max=rows.shape[1] - 1) == col0[:, None]
        page = torch.where(in_first, page0[:, None], page1[:, None])
        phys = (page * stride + pos % stride) // dcp_size
        seg_a = owned & in_first
        seg_b = owned & ~in_first
        count_a = seg_a.sum(1, dtype=i32)
        count_b = seg_b.sum(1, dtype=i32)
        far = torch.iinfo(i32).max
        dst_a = torch.where(count_a > 0, torch.where(seg_a, phys, far).amin(1), zero)
        dst_b = torch.where(count_b > 0, torch.where(seg_b, phys, far).amin(1), zero)
        pad = torch.zeros(max_spans - 2 * bs, dtype=i32, device=device)

        def table_of(a, b):
            return torch.cat([torch.stack([a, b], 1).reshape(-1), pad])

        return (
            table_of(dst_a, dst_b),
            table_of(start, start + count_a),
            table_of(count_a, count_b),
        )

    full = spans(page_indices)
    swa = spans(swa_page_indices) if swa_page_indices is not None else None
    return kv_lens, global_lens, prefix_lens, index, full, swa


def write_owned_kv(
    write_op, kv_cache: torch.Tensor, rows: torch.Tensor, spans: DcpWriteSpans
) -> None:
    """Write this rank's rows of ``rows`` into ``kv_cache`` in place.

    ``rows`` is every token's row in the pool layout; ``spans`` compacts the
    owned ones and ``write_op`` (``kv_span_write``) DMAs each span into the
    pool. A torch scatter into the pool inside the compiled forward costs
    about a millisecond per layer.
    """
    rows = rows.index_select(0, spans.index).contiguous()
    write_op(spans.dst, spans.src, spans.lens, rows, kv_cache)


def sink_fold(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Fold an attention sink into a merged ``[T, H, D]`` output.

    The sink is a virtual key with logit ``sinks[h]`` and value 0, so it only
    rescales the softmax: ``out * e^lse / (e^lse + e^sink)``.
    """
    return out * torch.sigmoid(lse - sinks.to(torch.float32))[..., None]


def _stable_lse_pair(lse_a: torch.Tensor, lse_b: torch.Tensor):
    """``(m, ea, eb)`` for two LSEs: the finite-safe max and each exp offset.

    ``-inf`` inputs give a zero weight instead of ``exp(-inf - -inf) = nan``.
    """
    m = torch.maximum(lse_a, lse_b)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m, torch.exp(lse_a - m_safe), torch.exp(lse_b - m_safe)


def _zero_non_finite(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def merge_lse(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two partial softmaxes over disjoint key sets by their LSEs.

    Rows whose key set was empty arrive as ``lse = -inf`` with a ``0/0`` output;
    both their weight and their contribution become 0. Built from elementwise
    ops only (max, where, isfinite, exp, log), which torch_tpu lowers.
    """
    m, ea, eb = _stable_lse_pair(lse_a, lse_b)
    norm = ea + eb
    norm_safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    out = (
        _zero_non_finite(out_a) * ea[..., None]
        + _zero_non_finite(out_b) * eb[..., None]
    ) / norm_safe[..., None]
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    lse = torch.where(torch.isfinite(m), m_safe + torch.log(norm_safe), m)
    return out, lse


def _all_reduce_gather(
    x: torch.Tensor, group, world_size: int, rank_mask: torch.Tensor, dim: int
) -> torch.Tensor:
    """``x`` of every rank concatenated along ``dim`` in rank order.

    Implemented as an in-place all-reduce over a zero-padded copy: every rank
    contributes its own block and zeros elsewhere, so the sum is exact.
    Workaround: inside a compiled graph the torch_tpu all-gather op returns its
    freshly allocated output uninitialised (torch_tpu 0.1.1.dev20260914); the
    all-gather can replace this once it fills its output there.

    ``rank_mask`` is ``arange(world_size) == rank`` as a device tensor: the rank
    enters the graph as data, not as structure, so every rank compiles the same
    program and issues the same collective sequence. A rank baked into the
    graph (which blocks are zero, which slice is taken) gives each rank its own
    program, and rank-divergent programs can return wrong merges on some ranks.
    """
    if world_size == 1:
        return x
    shape = list(x.shape)
    expanded = x.unsqueeze(dim).expand(shape[:dim] + [world_size] + shape[dim:])
    mask = rank_mask.view([1] * dim + [world_size] + [1] * (x.dim() - dim))
    padded = torch.where(mask, expanded, x.new_zeros(()))
    padded = padded.reshape(shape[:dim] + [world_size * shape[dim]] + shape[dim + 1 :])
    return group.all_reduce(padded.contiguous())


def gather_heads(
    x: torch.Tensor, group, world_size: int, rank_mask: torch.Tensor
) -> torch.Tensor:
    """``[T, H, D]`` of every rank as ``[T, H * world_size, D]``, rank-major."""
    return _all_reduce_gather(x, group, world_size, rank_mask, dim=1)


def merge_across_ranks(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    world_size: int,
    rank_mask: torch.Tensor,
    rank_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one partial ``(out [T, H_all, D] fp32, lse [T, H_all])`` per rank.

    Same protocol as ``layers.dcp.comm.cp_lse_ag_out_rs_mha``: gather the
    LSEs, rescale the local output to the group-wide softmax, all-reduce, and
    keep this rank's own head block. The log-sum-exp over ranks is written with
    elementwise ops (pairwise max, exp, log) because torch_tpu implements
    neither ``logsumexp`` nor ``logaddexp``. ``rank_mask`` / ``rank_index`` are
    the rank as device tensors (``bool[world_size]`` one-hot, ``i64[1]``).
    """
    if world_size == 1:
        return out, lse
    lse = lse.contiguous()
    num_tokens, heads_all = lse.shape
    lses = _all_reduce_gather(lse, group, world_size, rank_mask, dim=0).view(
        world_size, num_tokens, heads_all
    )
    m = lses[0]
    for i in range(1, world_size):
        m = torch.maximum(m, lses[i])
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    total = torch.exp(lses[0] - m_safe)
    for i in range(1, world_size):
        total = total + torch.exp(lses[i] - m_safe)
    global_lse = torch.where(torch.isfinite(m), m_safe + torch.log(total), m)
    scale = _zero_non_finite(torch.exp(lse - global_lse))
    out = _zero_non_finite(out) * scale[..., None]
    out = group.all_reduce(out)
    heads = heads_all // world_size
    return (
        out.view(num_tokens, world_size, heads, -1)
        .index_select(1, rank_index)
        .reshape(num_tokens, heads, -1)
        .contiguous(),
        global_lse.view(num_tokens, world_size, heads)
        .index_select(1, rank_index)
        .reshape(num_tokens, heads)
        .contiguous(),
    )
