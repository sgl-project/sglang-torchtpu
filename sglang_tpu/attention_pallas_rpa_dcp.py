# SPDX-License-Identifier: Apache-2.0
"""Decode-context-parallel (DCP) pieces of the TPU Pallas batched RPA backend.

Under DCP the KV of a sequence is split across the ranks of one DCP group by
page: rank ``r`` stores page ``p = pos // page_size`` iff ``p % dcp == r``
(tokens inside a page stay dense and contiguous). Each rank therefore attends
only its own page stripe, and the partial results are merged across the group
by log-sum-exp.

This module holds the parts of that scheme that are independent of the backend
object: the torch custom op for the CP-aware kernel (including the fused
``STRIPED`` decode scope and the stripe-writing ``NEW_TOKENS_ONLY`` chunk
scope), the query-head all-gather and the pure-torch merges (cross-rank
log-sum-exp merge, sink fold, two-pass extend combine). The kernel writes the
step's K/V itself in every mode; nothing stages K/V outside it.

``attention_pallas_rpa_base`` drives them; the non-DCP path never calls in here.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import torch

from sglang_tpu.pallas_export import tpu_info_for_cpu_export

__all__ = [
    "DCP_OP_NAME",
    "build_dcp_kernel_op",
    "gather_heads",
    "merge_across_ranks",
    "merge_lse",
    "own_heads",
    "sink_fold",
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
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention, configs


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

    _jax_rpa, configs = _import_dcp_kernel()

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
        attention_scope: int = 0,
        rpa_cases: int = 0,
        tail_decode: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # The kernel takes ``cp_group_size=None`` for the non-CP form and then
        # rejects the two CP operands, so drop them together.
        cp_size = cp_group_size if cp_group_size else None
        if cache_only and attention_scope == 0:
            attention_scope = 1
        active_cases = None
        if rpa_cases == 1:
            active_cases = (configs.RpaCase.DECODE,)
        elif rpa_cases == 2:
            active_cases = (configs.RpaCase.MIXED,)
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
            attention_scope=attention_scope,
            rpa_cases=active_cases,
            tail_decode=tail_decode,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{DCP_OP_NAME}",
        # Static (non-traced) args by _wrapped position: 11=sm_scale,
        # 12=sliding_window, 13=soft_cap, 14=k_scale, 15=v_scale,
        # 16=cache_only, 17=skip_kv_write, 18=cp_group_size, 19=attention_scope,
        # 20=rpa_cases, 21=tail_decode.
        static_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
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
        attention_scope: int = 0,
        rpa_cases: int = 0,
        tail_decode: bool = False,
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
                attention_scope,
                rpa_cases,
                tail_decode,
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
    x: torch.Tensor,
    group,
    world_size: int,
    rank_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[T, H, D]`` of every rank as ``[T, H * world_size, D]``, rank-major.

    One ``all_gather_into_tensor`` over the group. The zero-padded all-reduce
    form (``_all_reduce_gather``) is kept for groups without an all-gather
    (the CPU test doubles); on torch_tpu 0.2.0 / libtpu 0.0.50 a bf16
    all-reduce of the queries inside the compiled forward deadlocks the
    device on its first execute, while the all-gather runs.
    """
    if world_size == 1:
        return x
    if not hasattr(group, "all_gather_into_tensor"):
        return _all_reduce_gather(x, group, world_size, rank_mask, dim=1)
    num_tokens, heads, head_dim = x.shape
    gathered = x.new_empty((world_size * num_tokens * heads, head_dim))
    group.all_gather_into_tensor(gathered, x.contiguous().view(-1, head_dim))
    return (
        gathered.view(world_size, num_tokens, heads, head_dim)
        .permute(1, 0, 2, 3)
        .reshape(num_tokens, world_size * heads, head_dim)
        .contiguous()
    )


def own_heads(
    x: torch.Tensor, world_size: int, rank_index: torch.Tensor
) -> torch.Tensor:
    """This rank's head block of a rank-major gathered-head tensor.

    ``x`` is ``[T, H * world_size, ...]`` as produced by :func:`gather_heads`
    (or by the kernel run on such queries); the result is ``[T, H, ...]``.
    ``rank_index`` is the rank as an ``i64[1]`` device tensor so the slice is
    data, not program structure (see ``_all_reduce_gather``).
    """
    if world_size == 1:
        return x
    num_tokens, heads_all = x.shape[:2]
    heads = heads_all // world_size
    trailing = x.shape[2:]
    return (
        x.view(num_tokens, world_size, heads, *trailing)
        .index_select(1, rank_index)
        .reshape(num_tokens, heads, *trailing)
        .contiguous()
    )


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
