# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/push_fused_moe/layer.py
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Push routed activations into expert slabs and combine their outputs.

The push build run for every format but K32Z (see the
package's fused_ep_moe_push); layer.py keeps the K32Z program unchanged."""

import functools
import threading
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu._logging import get_logger
from sglang_tpu.fp8_swiglu_mlp import (
    fp8_swiglu_down,
    fp8_swiglu_gate_up,
    fp8_swiglu_mlp,
)
from sglang_tpu.fp8_swiglu_mlp import is_supported as shared_expert_fused
from sglang_tpu.moe.pallas.fused_moe_v2.host import (
    HIDDEN_LANE_BLOCK,
    MAX_ROUTING_BLOCK,
    QB4,
    ROWBLK,
    U32_SUBLANE_TILE,
    GatheredPairs,
    RoutingTables,
    ShardRoutingTables,
    WeightFormat,
    act_scale_slab_rows,
    align_up,
    build_routing_tables,
    fp4_scale_rows,
    local_slab_rows,
    pair_block_hist,
    ragged_stride_bound,
    routing_block,
    shard_count_vector,
    shard_expert_slabs,
    shard_push_tables_in_rows,
    shard_transport_tables_in_blocks,
    weight_form,
    weight_format_of_dtype,
)
from sglang_tpu.moe.pallas.fused_moe_v2.kernel import combine_step_tokens, rowquant_fp8
from sglang_tpu.moe.pallas.fused_moe_v2.layer import (
    SCORING_FNS,
    _apply_scoring_fn,
    _combine_arrivals,
    _exchange_layout,
    _relabel_expert_ids_to_mesh_order,
    _row_routes,
    _to_slots,
    _unpack_routing_exchange,
    _unpermute_load_to_ep_rank_order,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.host import (
    build_routing_tables_sharded,
    local_sort_activations,
    local_sort_tokens,
    push_dispatch_segments,
    push_dispatch_tables,
    routing_pairs_from_tables,
    run_tables,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.kernel import (
    build_combine_decode_kernel,
    build_combine_fs_kernel,
    build_push_moe_kernel,
    decode_combine_applies,
    fs_combine_chunk,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.plan_kernel_k512 import (
    build_exchange_rows_tables,
    build_exchange_rows_tables_split,
    build_local_ranks,
    build_pair_rows,
    build_pair_tables,
    build_push_tables,
    pair_tables_apply,
    wide_pair_tables_apply,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.routing import prepare_routing
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.sc_dispatch import (
    sc_dispatch_chunk_2core,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.tc_exchange_k512 import (
    build_tc_exchange_all_gather,
    build_tc_exchange_all_gather_rows,
)
from sglang_tpu.moe.pallas.fused_moe_v2.router_ops import pallas_select
from sglang_tpu.moe.pallas.fused_moe_v2.token_parallel import TokenReplicaLayout

logger = get_logger(__name__)
_LOGGED_PATHS: set = set()


def _log_path(t_local: int, path: str, used: bool) -> None:
    """Trace-time record, once per (t_local, path), of which push build serves a
    compiled bucket, so a serving log shows engagement per bucket."""
    key = (int(t_local), path, bool(used))
    if key not in _LOGGED_PATHS:
        _LOGGED_PATHS.add(key)
        logger.info(
            "push MoE path: t_local=%d %s %s",
            key[0],
            path,
            "USED" if used else "not used",
        )


# The standard logging.Logger has no warning_once; log each clamp once.
@functools.lru_cache(maxsize=None)
def _warn_sc_prepush_clamped(n_pre, g_local):
    logger.warning(
        "push fused EP MoE: sc_prepush %d exceeds the %d experts per shard; using %d",
        n_pre,
        g_local,
        g_local,
    )


# light_prepush applies from this many local slots up: the heavy experts it
# takes off the pre-push need visits to arrive in (at DP16 the 9-slot kernel
# lost ~20 us a call with it while the 25-slot kernel held its time).
LIGHT_PREPUSH_MIN_SLOTS = 16


def _pack_routing_exchange(pairs, scale_bits, counts, layout):
    """The flat i32 payload: pairs [t*K], row-scale bits [t], counts [E],
    zero-padded to the layout's width."""
    scale_at, count_at, width = layout
    # Padded in place rather than concatenated with zero arrays: XLA keeps
    # those zeros as HBM constants that the concatenation reads back.
    parts = [jnp.pad(pairs, (0, scale_at - pairs.size))]
    if scale_bits is not None:
        parts.append(jnp.pad(scale_bits, (0, count_at - scale_at - scale_bits.size)))
    parts.append(jnp.pad(counts, (0, width - count_at - counts.size)))
    return jnp.concatenate(parts)


def _combine_arrivals_fs(
    arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype, decode=True
):
    """v2's `_combine_arrivals`, with the arrival-scale lookup done inside the
    combine kernel (see kernel._build_combine_fs_kernel) where that kernel
    applies, the one-pass decode combine (kernel._build_combine_decode_kernel)
    at decode sizes where `decode`, and v2's XLA epilogue everywhere else."""
    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    mirror_sublanes, tile_m = arrival_scales.shape
    # A DMA does not clamp an out-of-range index the way a gather does.
    # Neither position is ever negative.
    rows = jnp.minimum(pos, recv_rows - 1)
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    _log_path(
        t_local, "decode_combine", bool(decode and decode_combine_applies(t_local))
    )
    if decode and decode_combine_applies(t_local):
        combine = build_combine_decode_kernel(
            t_local=t_local,
            topk=topk,
            mirror_lanes=tile_m,
            lane_blocks=lane_blocks,
            wire_dtype=arrivals.dtype,
            out_dtype=out_dtype,
        )
        return combine(arrivals, arrival_scales, scl, topk_weights, rows)
    # v2's step rule keeps its COMBINE_MIN_TOKENS floor; the fs kernel tiles
    # by its own chunk.
    if not (combine_step_tokens(t_local) and fs_combine_chunk(t_local, topk)):
        return _combine_arrivals(
            arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
        )
    combine = build_combine_fs_kernel(
        t_local=t_local,
        topk=topk,
        hidden=lane_blocks * lanes,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals.dtype,
        out_dtype=out_dtype,
    )
    return combine(arrivals, arrival_scales, scl, topk_weights, rows)


def _repeat_bits(bits, repeats):
    """jnp.repeat(bits, repeats) for i32 `bits` [n] as one one-hot matmul
    over the four bytes of each word.

    XLA spells the repeat as a [n, repeats] broadcast (one word in each
    128-lane row) and then flattens it; this builds the lane-dense
    [n / 128, 128 * repeats] result directly. Exact for any bit pattern: each
    output is one byte (< 256, exact in bf16) times 1.0, summed with zeros in
    f32.
    """
    (n_in,) = bits.shape
    lanes = HIDDEN_LANE_BLOCK
    n = align_up(n_in, lanes)
    bits = jnp.pad(bits, (0, n - n_in))
    shifts = 8 * jnp.arange(4, dtype=jnp.int32)[:, None, None]
    planes = (
        jnp.right_shift(bits.reshape(1, n // lanes, lanes), shifts) & 255
    ).reshape(4 * n // lanes, lanes)
    spread = (
        jnp.arange(lanes * repeats, dtype=jnp.int32)[None, :] // repeats
        == jnp.arange(lanes, dtype=jnp.int32)[:, None]
    ).astype(jnp.bfloat16)
    out = jnp.dot(
        planes.astype(jnp.bfloat16), spread, preferred_element_type=jnp.float32
    ).astype(jnp.int32)
    out = out.reshape(4, n // lanes, lanes * repeats)
    word = out[0] | (out[1] << 8) | (out[2] << 16) | (out[3] << 24)
    return word.reshape(n * repeats)[: n_in * repeats]


# Up to this many routed pairs (decode-sized steps) the scale slab is built on
# the TensorCore (_scale_slab_onehot); above it the SparseCore scatter stays.
ONEHOT_SCALE_SLAB_MAX_PAIRS = 4096


def _scale_slab_onehot(slab_row, scale_bits, scale_rows):
    """``zeros([scale_rows * 128]).at[slab_row].add(scale_bits, mode="drop")``
    as one one-hot matmul, returned as the f32 [scale_rows, 128] slab.

    Exact for any bit pattern on every row the expert kernel reads, because each
    routed pair lands on a slab row of its own: row r = 128 * hi + lo, an
    [4 * scale_rows, n] operand holds each pair's four bit-pattern bytes on its
    hi row and an [n, 128] one-hot picks its lo lane, so every output is one
    byte (< 256, exact in bf16) times 1.0 summed with zeros in f32. Rows past
    the slab (routing_pairs_from_tables' off-shard sentinel) match no hi row
    and drop. host.local_slab_rows instead sends every off-shard pair below
    this shard to the sink row `shard_stride`, inside the slab's spare
    sublane: there the byte sums carry and differ from the scatter-add's
    (equally meaningless) sum, which is harmless only because nothing reads
    that row (host.act_scale_slab_rows). plan_kernel._build_pair_tables builds the
    same slab in-kernel by the same argument. On the SparseCore the scatter delays the
    pre-push queued behind it; here it is one small matmul in VMEM.
    """
    lanes = HIDDEN_LANE_BLOCK
    (n,) = slab_row.shape
    hi = slab_row // lanes
    lo = slab_row % lanes
    shifts = 8 * jnp.arange(4, dtype=jnp.int32)
    planes = jnp.right_shift(scale_bits[None, :], shifts[:, None]) & 255  # [4, n]
    on_row = hi[None, :] == jnp.arange(scale_rows, dtype=jnp.int32)[:, None]
    a = jnp.where(on_row[None], planes[:, None, :], 0)
    a = a.astype(jnp.bfloat16).reshape(4 * scale_rows, n)
    b = (lo[:, None] == jnp.arange(lanes, dtype=jnp.int32)[None, :]).astype(
        jnp.bfloat16
    )
    out = jnp.dot(a, b, preferred_element_type=jnp.float32).astype(jnp.int32)
    out = out.reshape(4, scale_rows, lanes)
    word = out[0] | (out[1] << 8) | (out[2] << 16) | (out[3] << 24)
    return lax.bitcast_convert_type(word, jnp.float32)


_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def fused_ep_moe_push(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    w1_bias=None,
    w2_bias=None,
    *,
    topk,
    renormalize,
    mesh,
    capacity,
    block=None,
    ragged_stride=None,
    weight_format=WeightFormat.FP8,
    rhs_qb=None,
    act_fn="silu",
    rank=None,
    mesh_ep_ranks=None,
    sharded_plan=False,
    scoring_fn="softmax",
    score_bias=None,
    hash_topk_ids=None,
    routed_scaling_factor=1.0,
    token_replica_groups=None,
    logical_intermediate_size=None,
    preprocess_topology=None,
    sc_prepush=2,
    tc_exchange=False,
    height_rungs_override=0,
    tail_barrier=True,
    onehot_slots=False,
    hbm_staging=False,
    dispatch_ahead=1,
    matmul_scale_repeat=False,
    exchange_row_gather=False,
    bf16_sort_counts=False,
    slim_prepush_barrier=False,
    fused_dispatch_meta=False,
    table_kernel=False,
    exchange_tables=False,
    kernel_plan_tables=False,
    light_prepush=0,
    single_barrier=False,
    early_scale_scatter=0,
    split_exchange=False,
    onehot_scale_slab=False,
    pair_kernel=False,
    wide_pair_kernel=False,
    decode_combine=True,
    fuse_metadata=True,
    expert_slot=None,
    expert_load=None,
    shared_expert=None,
    shared_limit=None,
):
    """Push implementation of fused_ep_moe_v2's fused stage.

    Operands, placement and return values follow fused_ep_moe_v2.
    preprocess_topology is accepted for compatibility and ignored.

    Dispatch and scheduling options:
    - sc_prepush: initial visit positions pushed by SparseCore (>=1,
      clamped to local experts); dispatch_ahead: TensorCore run-ahead.
    - tc_exchange: remote-DMA routing exchange; exchange_row_gather also
      gathers sorted activations and requires tc_exchange.
    - height_rungs_override: tile ladder size (0 selects the default);
      tail_barrier, single_barrier and slim_prepush_barrier control barriers.
    - hbm_staging: stage result tiles in the HBM contribution slab.
    - onehot_slots, matmul_scale_repeat and bf16_sort_counts replace index
      operations with equivalent one-hot/matmul forms.
    - fused_dispatch_meta: build dispatch counts with the segments;
      table_kernel: fuse table construction; exchange_tables runs it inside
      the exchange; kernel_plan_tables also supplies routing-plan tables.
    - light_prepush (1 or 2): visit light experts first.
    - early_scale_scatter: release scale scatter before the shared expert
      (1), or before both the shared expert and SparseCore pre-push (2).
    - split_exchange (with exchange_tables, exchange_row_gather and the
      in-op shared expert on its unfused path): run the shared expert's
      gate_up while the routing exchange's rows are in flight
      (plan_kernel.build_exchange_rows_tables_split).
    - onehot_scale_slab: build the activation scale slab of a step with at
      most ONEHOT_SCALE_SLAB_MAX_PAIRS routed pairs on the TensorCore.
    - pair_kernel (with exchange_tables and kernel_plan_tables): build a
      decode-sized step's per-pair tables and scale slab in one kernel
      (plan_kernel.build_pair_tables); wide_pair_kernel: build a larger
      step's per-pair tables and local sort in two kernels
      (plan_kernel.build_pair_rows, build_local_ranks).
    - decode_combine (default on): combine a step of at most
      kernel.COMBINE_TOKENS tokens a shard in the one-pass decode kernel.
    - fuse_metadata allows routing metadata fusion in its measured envelope.

    The optional replicated FP8 shared_expert runs while pre-push is in
    flight; its output is returned separately, last. The caller owns the
    static expert layout and load accumulator."""
    if early_scale_scatter not in (0, 1, 2):
        raise ValueError(f"early_scale_scatter is 0, 1 or 2; got {early_scale_scatter}")
    dispatch_ahead = int(dispatch_ahead)
    if dispatch_ahead < 1:
        raise ValueError(f"dispatch_ahead must be >= 1; got {dispatch_ahead}")
    if shared_limit is not None and shared_expert is None:
        raise ValueError("shared_limit requires shared_expert")
    del preprocess_topology
    form = weight_form(weight_format)
    rhs_qb = QB4 if rhs_qb is None else int(rhs_qb)
    for name, w in (("w1", w1), ("w2", w2)):
        actual = weight_format_of_dtype(w.dtype)
        if actual != weight_format:
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert weights and "
                f"{name} is {jnp.dtype(w.dtype).name}"
            )
    if form.has_scales != (w1_scale is not None):
        raise ValueError(
            f"weight format {weight_format!r} carries "
            f"{form.scale_layout} weight scales and w1_scale is "
            f"{'present' if w1_scale is not None else 'absent'}"
        )
    if (w1_scale is None) != (w2_scale is None):
        raise ValueError("both weight scales are supplied together or neither is")
    T, hidden = x.shape
    e_total = w1.shape[0]
    has_expert_slot = expert_slot is not None
    has_expert_load = expert_load is not None
    has_shared = shared_expert is not None
    if has_shared and (len(shared_expert) != 4 or shared_limit is None):
        raise ValueError("shared_expert is (w13, s13, w2, s2) with a shared_limit")
    # Global experts the router scores; without a layout, one slot each.
    e_logical = gating.shape[1] if has_expert_slot else e_total
    if has_expert_slot and (
        expert_slot.ndim != 2
        or expert_slot.shape[0] != e_logical
        or expert_slot.dtype != jnp.int32
        or e_total < e_logical
    ):
        raise ValueError(
            f"expert_slot must be i32 [{e_logical}, replicas] over {e_total} "
            f"slots; got {jnp.dtype(expert_slot.dtype).name}"
            f"{tuple(expert_slot.shape)}"
        )
    if has_expert_load and (
        expert_load.shape != (e_total,) or expert_load.dtype != jnp.int32
    ):
        raise ValueError(
            f"expert_load must be i32 [{e_total}], one count per physical slot; "
            f"got {jnp.dtype(expert_load.dtype).name}{tuple(expert_load.shape)}"
        )
    inter = w1.shape[2] // 2
    pad_scale_rows = (
        form.has_scales
        and form.scale_layout == "per_contraction_block"
        and w1_scale.ndim == 2
    )
    if pad_scale_rows:
        for name, s, k, n in (
            ("w1_scale", w1_scale, hidden, 2 * inter),
            ("w2_scale", w2_scale, inter, hidden),
        ):
            want = (e_total * fp4_scale_rows(k, rhs_qb, padded=True), n)
            if tuple(s.shape) != want:
                raise ValueError(
                    f"{name} layout {tuple(s.shape)} is not the row-padded "
                    f"flat per_contraction_block form {want} the kernel takes"
                )
    elif form.has_scales:
        ndim = 3 if form.scale_layout == "per_contraction_block" else 2
        for name, s, n in (
            ("w1_scale", w1_scale, 2 * inter),
            ("w2_scale", w2_scale, hidden),
        ):
            if s.ndim != ndim or s.shape[0] != e_total or s.shape[-1] != n:
                raise ValueError(
                    f"{name} layout {tuple(s.shape)} is not the "
                    f"{form.scale_layout} form the kernel takes"
                )
    (ax,) = mesh.axis_names
    ep = mesh.shape[ax]
    if mesh_ep_ranks is not None:
        mesh_ep_ranks = tuple(int(r) for r in mesh_ep_ranks)
        if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
            raise ValueError(
                f"mesh_ep_ranks must be a permutation of range({ep}); got "
                f"{mesh_ep_ranks}"
            )
    g_local = e_total // ep
    if T % ep or gating.shape != (T, e_logical):
        raise ValueError(
            "input and gating rows must agree and partition evenly over EP"
        )
    token_layout = TokenReplicaLayout.create(
        ep=ep,
        input_rows=T // ep,
        groups=token_replica_groups,
        row_alignment=U32_SUBLANE_TILE,
    )
    t_local = token_layout.local_rows
    T = ep * t_local
    P = jax.sharding.PartitionSpec
    stride_bound = ragged_stride_bound(T, topk, e_total, capacity)
    if ragged_stride is None:
        ragged_stride = stride_bound
    if block is None:
        block = routing_block(t_local, topk)
    wide_pairs = bool(
        wide_pair_kernel
        and exchange_tables
        and kernel_plan_tables
        and wide_pair_tables_apply(t_local, topk, block)
    )
    if ragged_stride % capacity != 0 or ragged_stride < stride_bound:
        raise ValueError(
            f"ragged_stride {ragged_stride} must be a multiple of capacity "
            f"{capacity} and at least the no-drop bound {stride_bound}"
        )
    has_w1_bias = w1_bias is not None
    has_w2_bias = w2_bias is not None
    has_score_bias = score_bias is not None
    has_hash_ids = hash_topk_ids is not None
    if scoring_fn not in SCORING_FNS:
        raise ValueError(f"scoring_fn must be one of {SCORING_FNS}; got {scoring_fn!r}")
    if has_score_bias and score_bias.shape not in (
        (e_logical,),
        (gating.shape[0], e_logical),
    ):
        raise ValueError(
            f"score_bias must be one value per global expert, [{e_logical}], "
            f"or one such row per token; got {tuple(score_bias.shape)}"
        )
    row_bias = score_bias is not None and score_bias.ndim == 2
    if has_hash_ids and (
        hash_topk_ids.shape != (x.shape[0], topk)
        or not jnp.issubdtype(hash_topk_ids.dtype, jnp.integer)
    ):
        raise ValueError(
            f"hash_topk_ids must be integer expert ids [{x.shape[0]}, {topk}]; "
            f"got {hash_topk_ids.dtype}{tuple(hash_topk_ids.shape)}"
        )
    # Visit positions [0, n_pre) go out on the SparseCore before the kernel;
    # at least one, so the kernel always starts on rows that have landed.
    n_pre = int(sc_prepush)
    if n_pre < 1:
        raise ValueError(f"sc_prepush must be >= 1; got {sc_prepush}")
    if n_pre > g_local:
        _warn_sc_prepush_clamped(n_pre, g_local)
        n_pre = g_local
    tc_exchange = bool(tc_exchange)
    exchange_row_gather = bool(exchange_row_gather)
    if kernel_plan_tables and not (table_kernel and sharded_plan):
        raise ValueError("kernel_plan_tables needs table_kernel and sharded_plan")
    if exchange_tables and not (table_kernel and exchange_row_gather):
        raise ValueError("exchange_tables needs table_kernel and exchange_row_gather")
    light_prepush = int(light_prepush)
    if light_prepush not in (0, 1, 2):
        raise ValueError(f"light_prepush is 0, 1 or 2; got {light_prepush}")
    if light_prepush and not table_kernel:
        raise ValueError("light_prepush orders the visits in the table kernel")
    if g_local < LIGHT_PREPUSH_MIN_SLOTS:
        light_prepush = 0
    if exchange_row_gather and not tc_exchange:
        raise ValueError(
            "exchange_row_gather needs the TensorCore exchange (tc_exchange)"
        )
    height_rungs_override = int(height_rungs_override)
    tail_barrier = bool(tail_barrier)
    hbm_staging = bool(hbm_staging)
    # Every window past the last visit position issues the same program.
    dispatch_ahead = min(dispatch_ahead, g_local)
    kfn = build_push_moe_kernel(
        g_local=g_local,
        capacity=capacity,
        hidden=hidden,
        inter=inter,
        ep=ep,
        ragged_rows_alloc=ragged_stride,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        act_fn=act_fn,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        pad_scale_rows=pad_scale_rows,
        logical_intermediate_size=logical_intermediate_size,
        push_tokens=t_local,
        sc_prepush=n_pre,
        height_rungs_override=height_rungs_override,
        tail_barrier=tail_barrier,
        hbm_staging=hbm_staging,
        dispatch_ahead=dispatch_ahead,
        single_barrier=bool(single_barrier),
    )

    layout = _exchange_layout(t_local, topk, e_total, form.quantized_activations)

    def exchange_all_gather(packed, rank_l):
        """The routing exchange all-gather of the flat i32 payload."""
        if tc_exchange:
            return build_tc_exchange_all_gather(
                ep=ep, width=packed.size, single_barrier=bool(single_barrier)
            )(packed, rank_l[0, 0])
        return lax.all_gather(packed, ax, axis=0, tiled=True)

    def unpack_operands(scales_gating_biases):
        """(w1 scale, w2 scale, gate, hash ids, score bias, slot table, load
        accumulator, shared expert, bias iterator)."""
        operands = iter(scales_gating_biases)
        if pad_scale_rows:
            # [G * rows, N] -> [G, rows, N]: whole sublane tiles per expert,
            # so this is a bitcast of the operand rather than a relayout.
            w1s_l = next(operands).reshape(
                g_local, fp4_scale_rows(hidden, rhs_qb, padded=True), 2 * inter
            )
            w2s_l = next(operands).reshape(
                g_local, fp4_scale_rows(inter, rhs_qb, padded=True), hidden
            )
        else:
            w1s_l = next(operands) if form.has_scales else None
            w2s_l = next(operands) if form.has_scales else None
        gate_l = next(operands)
        hash_l = next(operands) if has_hash_ids else None
        score_bias_l = next(operands) if has_score_bias else None
        expert_slot_l = next(operands) if has_expert_slot else None
        expert_load_l = next(operands) if has_expert_load else None
        shared_l = tuple(next(operands) for _ in range(4)) if has_shared else None
        return (
            w1s_l,
            w2s_l,
            gate_l,
            hash_l,
            score_bias_l,
            expert_slot_l,
            expert_load_l,
            shared_l,
            operands,
        )

    def quantize_local(x_l, rank_l, gate_l, hash_l):
        """v2's: split this shard's rows out of a replica group, quantize."""
        me = rank_l[0, 0]
        x_l = token_layout.split(x_l, me)
        gate_l = token_layout.split(gate_l, me, fill_value=-jnp.inf)
        if hash_l is not None:
            hash_l = token_layout.split(hash_l, me)
        if len(token_layout.groups[0]) > 1 and form.quantized_activations:
            # Slicing fused into rowquant changes BF16 rounding on TPU.
            # Keep this boundary only for quantized replicated inputs.
            x_l = lax.optimization_barrier(x_l)
        rows_bf16 = x_l.astype(jnp.bfloat16)
        if form.quantized_activations:
            q_l, row_scale_l = rowquant_fp8(rows_bf16)
        else:
            q_l, row_scale_l = rows_bf16, None
        q_l = q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK)
        return gate_l, hash_l, q_l, row_scale_l

    def select_local(gate_l, hash_l, score_bias_l, row_scale_l, expert_slot_l, rank_l):
        """v2's: top-k, router weights, and this shard's routing payload."""
        me = rank_l[0, 0]
        if row_bias:
            score_bias_l = token_layout.split(score_bias_l, me)
        scores = _apply_scoring_fn(gate_l, scoring_fn)
        select_rows = MAX_ROUTING_BLOCK
        while t_local % select_rows:
            select_rows //= 2
        if hash_l is not None:
            topk_idx = hash_l.astype(jnp.int32)
            topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
            if expert_slot_l is not None:
                topk_idx = _to_slots(
                    topk_idx, expert_slot_l, me * t_local, onehot=onehot_slots
                )
        elif score_bias_l is None:
            topk_weights, topk_idx = pallas_select(
                scores, topk=topk, block_rows=select_rows
            )
        else:
            topk_weights, topk_idx = pallas_select(
                scores + score_bias_l.astype(jnp.float32),
                topk=topk,
                block_rows=select_rows,
                weight_scores=scores,
            )
        if expert_slot_l is not None and hash_l is None:
            topk_idx = _to_slots(
                topk_idx, expert_slot_l, me * t_local, onehot=onehot_slots
            )
        topk_idx = _relabel_expert_ids_to_mesh_order(
            topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
        )
        # A row carrying no real score routes nowhere (see v2).
        row_routes = _row_routes(scores, gate_l)
        if renormalize:
            denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
            topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
        topk_weights = jnp.where(row_routes, topk_weights, 0.0)
        topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(e_total))
        local_blocks = topk_idx.astype(jnp.int32).reshape(
            t_local * topk // block, block
        )
        pairs = local_blocks.reshape(-1)
        if wide_pairs:
            # (counts, seg_start, row map, payload words); the payload carries
            # each pair's rank with its expert, for build_pair_rows.
            local_ranks = build_local_ranks(
                pairs, e_total=e_total, topk=topk, block=block
            )
            block_hist, counts, pairs = None, local_ranks[0], local_ranks[3]
        else:
            local_ranks = None
            block_hist = pair_block_hist(local_blocks, e_total)
            counts = block_hist.sum(axis=0)
        scale_bits_l = (
            None
            if row_scale_l is None
            else (
                lax.bitcast_convert_type(
                    row_scale_l[:, 0].astype(jnp.float32), jnp.int32
                )
            )
        )
        packed = _pack_routing_exchange(pairs, scale_bits_l, counts, layout)
        return topk_weights, packed, local_blocks, block_hist, row_routes, local_ranks

    def compute_local(
        q_sorted,
        seg_start,
        exchange,
        local_blocks,
        block_hist,
        topk_weights,
        row_routes,
        rank_l,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        bias_it,
        out_dtype,
        expert_load_l=None,
        pre_tables=None,
        x_l=None,
        shared_fn=None,
        exchange_slab=None,
    ):
        """Dispatch tables, SparseCore pre-push, plan, kernel and combine.

        `pre_tables`, where the exchange kernel already built them, are
        build_push_tables' outputs; `shared_fn(x_l)`, the shared expert (or
        what is left of it), computed between the pre-push and the kernel."""
        me = rank_l[0, 0]
        pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
            exchange, layout, t_local, topk, e_total
        )
        # The step's rows per physical slot, summed over every source rank:
        # identical on every shard, so the accumulator stays replicated.
        updated_load = (
            None
            if expert_load_l is None
            else expert_load_l
            + _unpermute_load_to_ep_rank_order(
                rows_by_dest.sum(axis=0).astype(jnp.int32),
                g_local=g_local,
                mesh_ep_ranks=mesh_ep_ranks,
            )
        )
        with jax.named_scope("moe_dispatch_tables"):
            runs = run_tables(rows_by_dest)
            shard_tables = None
            if pre_tables is not None:
                seg, dmeta, visit, counts_sc, *shard_tables = pre_tables
            elif table_kernel:
                seg, dmeta, visit, counts_sc, *shard_tables = build_push_tables(
                    rows_by_dest,
                    seg_start,
                    me,
                    ep=ep,
                    n_pre=n_pre,
                    tile_m=capacity,
                    light_prepush=light_prepush,
                )
            elif fused_dispatch_meta:
                seg, dmeta, _, visit, counts_sc = push_dispatch_tables(
                    runs[3],
                    runs[2],
                    rows_by_dest,
                    me,
                    seg_start,
                    e_total=e_total,
                    ep=ep,
                    n_pre=n_pre,
                )
            else:
                seg, dmeta, visit_all = push_dispatch_segments(
                    runs[3],
                    runs[2],
                    rows_by_dest,
                    me,
                    seg_start,
                    e_total=e_total,
                    ep=ep,
                )
                recv_by_e = dmeta[g_local + 1 : 2 * g_local + 1]
                S = ep * g_local
                seg_n = jnp.bitwise_and(seg[:S], 65535).reshape(g_local, ep)
                seg_remote = (jnp.arange(ep)[None, :] != me).astype(jnp.int32)
                visit = lax.dynamic_slice(visit_all, (me, 0), (1, g_local)).reshape(
                    g_local
                )
                # Rows landing here at each pre-pushed visit position.
                vis_s = visit[:n_pre]
                hit_s = (
                    vis_s[:, None] == jnp.arange(g_local, dtype=jnp.int32)[None, :]
                ) & (vis_s[:, None] >= 0)
                rows_at_v = jnp.sum(jnp.where(hit_s, recv_by_e[None, :], 0), axis=1)
                # The kernel waits only the remote rows IT sends: positions >= n_pre.
                n_rem_tc = (
                    jnp.sum(seg_n[n_pre:] * seg_remote).astype(jnp.int32).reshape(1)
                )
                dmeta = jnp.concatenate(
                    [dmeta[: 2 * g_local + 1], n_rem_tc, dmeta[2 * g_local + 2 :]]
                )
                # SparseCore core c pushes the positions v with v % 2 == c.
                counts_sc = jnp.stack(
                    [
                        jnp.sum(rows_at_v[0:n_pre:2]),
                        jnp.sum(rows_at_v[1:n_pre:2]),
                        jnp.sum(seg_n[0:n_pre:2] * seg_remote),
                        jnp.sum(seg_n[1:n_pre:2] * seg_remote),
                    ]
                ).astype(jnp.int32)
        # Compute the pre-push operands before the routing plan, so XLA
        # launches the SparseCore dispatch ahead of it and the two overlap.
        # The plan reads pairs_g, so it follows the barrier either way; the
        # run tables only add a layout copy at it (`slim_prepush_barrier`).
        barriered_runs = () if slim_prepush_barrier else tuple(runs)
        # The shared expert's rows come out of the same barrier as the
        # pre-push operands, so it cannot be scheduled ahead of them.
        shared_x = (x_l,) if shared_fn is not None else ()
        (
            q_sorted,
            seg,
            counts_sc,
            dmeta,
            visit,
            pairs_g,
            scale_bits_g,
            *rest,
        ) = lax.optimization_barrier(
            (q_sorted, seg, counts_sc, dmeta, visit, pairs_g, scale_bits_g)
            + shared_x
            + barriered_runs
        )
        shared_x, barriered_runs = rest[: len(shared_x)], rest[len(shared_x) :]
        if not slim_prepush_barrier:
            runs = barriered_runs
        slab = sc_dispatch_chunk_2core(
            q_sorted,
            seg,
            counts_sc,
            ep=ep,
            g_local=g_local,
            v0=0,
            v1=n_pre,
            slab_rows=ragged_stride,
            name=f"moe_sc_prepush_2c_s{n_pre}",
            rank=me,
        )
        shared_out = ()
        if shared_fn is not None:
            with jax.named_scope("moe_shared_expert"):
                shared_out = (shared_fn(*shared_x),)
        plan_kw = dict(
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=capacity,
            shard_stride=ragged_stride,
        )
        gathered = None
        # The wide path's payload carries ranked words, not expert ids.
        if sharded_plan and not wide_pairs:
            gathered = GatheredPairs(
                expert_blocks=pairs_g.reshape(T * topk // block, block),
                local_blocks=local_blocks,
                block_hist=block_hist,
                rows_by_dest=rows_by_dest,
                n_tokens=T,
                topk=topk,
            )
        pair_tables = None
        routing_inputs = None
        routing = None
        pair_slab = None
        scale_rows = act_scale_slab_rows(ragged_stride)
        _pair = bool(
            not wide_pairs
            and kernel_plan_tables
            and pair_kernel
            and exchange_slab is not None
            and form.quantized_activations
            and pair_tables_apply(t_local, topk, layout[0], scale_rows)
        )
        _log_path(t_local, "wide_pair_kernel", wide_pairs)
        _log_path(t_local, "pair_kernel", _pair)
        if wide_pairs:
            # The per-pair lookups, from the payload's ranked pairs.
            assert shard_tables is not None
            pos_k, mirror_k, slab_row = build_pair_rows(
                exchange_slab,
                *shard_tables[4],
                me,
                ep=ep,
                t_local=t_local,
                topk=topk,
                e_total=e_total,
                shard_stride=ragged_stride,
            )
            pair_tables = (pos_k, mirror_k)
        elif (
            kernel_plan_tables
            and pair_kernel
            and exchange_slab is not None
            and form.quantized_activations
            and pair_tables_apply(t_local, topk, layout[0], scale_rows)
        ):
            # The per-pair lookups and the scale slab, from the payload.
            assert shard_tables is not None
            pos_k, mirror_k, pair_slab = build_pair_tables(
                exchange_slab,
                *shard_tables[4],
                me,
                ep=ep,
                t_local=t_local,
                topk=topk,
                block=block,
                e_total=e_total,
                scale_rows=scale_rows,
            )
            pair_tables = (pos_k, mirror_k)
            slab_row = None
        elif kernel_plan_tables:
            # The table kernel built the plan's small tables; only the
            # per-pair lookups are left.
            assert shard_tables is not None
            pair_tables = routing_pairs_from_tables(
                me,
                gathered,
                *shard_tables[4],
                e_total=e_total,
                ep=ep,
                t_local=t_local,
                block=block,
                shard_stride=ragged_stride,
            )
            slab_row = pair_tables[2]
        elif sharded_plan and shard_tables is None:
            # The sharded plan's consumer inputs, through the fused routing
            # metadata (routing.prepare_routing) where it admits the shape.
            assert gathered is not None
            routing_inputs = prepare_routing(
                me, gathered, tuple(runs), fuse_metadata=fuse_metadata, **plan_kw
            )
            slab_row = routing_inputs.slab_row
        else:
            if sharded_plan:
                routing = build_routing_tables_sharded(me, gathered, runs, **plan_kw)
            else:
                routing = build_routing_tables(pairs_g.reshape(T, topk), **plan_kw)
            # Both slab tables scatter onto this shard's own slab. The rows of
            # the other shards are dropped where they are computed rather than
            # built into a replicated slab and sliced away afterwards.
            slab_row = local_slab_rows(routing, me, shard_stride=ragged_stride)
        # Keep the scale-slab scatter (a SparseCore offload) behind the
        # pre-push on the SparseCore queue. The shared expert joins here, so
        # the expert kernel, which reads the slab, cannot start before it.
        # The pair kernel's finished slab takes the slab rows' place.
        slab_in = slab_row if pair_slab is None else pair_slab
        if early_scale_scatter == 2:
            # Nor behind the pre-push: the scatter may take the SparseCore
            # first; the light pre-push still lands under the shared expert.
            slab_in, scale_bits_g = lax.optimization_barrier((slab_in, scale_bits_g))
        elif early_scale_scatter:
            # The shared expert joins after the scatter instead (below), so
            # the scatter's SparseCore work can run under it.
            slab_in, scale_bits_g, slab = lax.optimization_barrier(
                (slab_in, scale_bits_g, slab)
            )
        else:
            slab_in, scale_bits_g, slab, *shared_out = lax.optimization_barrier(
                (slab_in, scale_bits_g, slab) + shared_out
            )
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on (v2's scale slab; see there).
        if form.quantized_activations and pair_slab is not None:
            scale_slab = slab_in
        elif form.quantized_activations:
            slab_row = slab_in
            scale_bits = (
                _repeat_bits(cast(jax.Array, scale_bits_g), topk)
                if matmul_scale_repeat
                else jnp.repeat(cast(jax.Array, scale_bits_g), topk)
            )
            _onehot = bool(
                onehot_scale_slab and slab_row.size <= ONEHOT_SCALE_SLAB_MAX_PAIRS
            )
            _log_path(t_local, "onehot_scale_slab", _onehot)
            if _onehot:
                scale_slab = _scale_slab_onehot(slab_row, scale_bits, scale_rows)
            else:
                scale_slab = lax.bitcast_convert_type(
                    jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK,), jnp.int32)
                    .at[slab_row]
                    .add(scale_bits, mode="drop"),
                    jnp.float32,
                ).reshape(scale_rows, HIDDEN_LANE_BLOCK)
        else:
            scale_slab = None
        if early_scale_scatter and shared_out and scale_slab is not None:
            scale_slab, slab, *shared_out = lax.optimization_barrier(
                (scale_slab, slab) + tuple(shared_out)
            )
        if shard_tables is not None:
            kernel_tables, expert_rows, slab_base, counts, _ = shard_tables
        elif routing_inputs is not None:
            kernel_tables = routing_inputs.kernel_tables
            expert_rows = routing_inputs.expert_rows
            slab_base = routing_inputs.slab_base
            counts = routing_inputs.counts
        else:
            assert routing is not None
            block_tables = shard_transport_tables_in_blocks(
                routing, me, e_total=e_total, ep=ep
            )
            row_tables = shard_push_tables_in_rows(routing, me, e_total=e_total, ep=ep)
            expert_rows, slab_base = shard_expert_slabs(
                routing, me, e_total=e_total, ep=ep
            )
            counts = shard_count_vector(
                routing, expert_rows, me, e_total=e_total, ep=ep
            )
            kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
        recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
        w1b_k = next(bias_it).reshape(g_local, 2 * inter) if has_w1_bias else None
        w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
        arrivals, arrival_scales = kfn(
            kernel_tables,
            expert_rows,
            slab_base,
            q_sorted,
            scale_slab,
            w1_l,
            w2_l,
            w1s_l,
            w2s_l,
            w1b_k,
            w2b_k,
            recv_rows=recv_rows,
            visit=visit,
            counts=counts,
            rank=rank_l,
            dispatch=(seg, dmeta),
            slab=slab,
        )

        # The destination's own table: one arrival row per selection slot.
        if pair_tables is not None:
            pos, mirror_pos = pair_tables[0], pair_tables[1]
        elif routing_inputs is not None:
            pos, mirror_pos = routing_inputs.pos, routing_inputs.mirror_pos
        elif sharded_plan:
            pos = cast(ShardRoutingTables, routing).pos
            mirror_pos = cast(ShardRoutingTables, routing).mirror_pos
        else:
            pos = lax.dynamic_slice(
                cast(RoutingTables, routing).arrival_row,
                (me * t_local, 0),
                (t_local, topk),
            )
            mirror_pos = lax.dynamic_slice(
                cast(RoutingTables, routing).mirror_row,
                (me * t_local, 0),
                (t_local, topk),
            )
        combined = _combine_arrivals_fs(
            arrivals,
            arrival_scales,
            pos,
            mirror_pos,
            topk_weights,
            out_dtype,
            decode=decode_combine,
        )
        # A row that routed nowhere read arrival rows nobody wrote; its weight is
        # zero, but zero times an unwritten NaN is not.
        combined = jnp.where(row_routes, combined, jnp.zeros_like(combined))
        out = token_layout.restore(combined, ax)
        outs = (out,) + (() if updated_load is None else (updated_load,))
        outs += tuple(shared_out)
        return outs if len(outs) > 1 else out

    def local_fn(x_l, rank_l, w1_l, w2_l, *scales_gating_biases):
        """One shard's half of the layer: route, dispatch, kernel, combine."""
        (
            w1s_l,
            w2s_l,
            gate_l,
            hash_l,
            score_bias_l,
            expert_slot_l,
            expert_load_l,
            shared_l,
            bias_it,
        ) = unpack_operands(scales_gating_biases)
        gate_l, hash_l, q_l, row_scale_l = quantize_local(x_l, rank_l, gate_l, hash_l)
        topk_weights, packed, local_blocks, block_hist, row_routes, local_ranks = (
            select_local(
                gate_l, hash_l, score_bias_l, row_scale_l, expert_slot_l, rank_l
            )
        )
        pre_tables = None
        exchange_slab = None
        shared_in, shared_fn = x_l, None
        # On its unfused path the shared expert splits around the exchange.
        split_shared = (
            split_exchange
            and exchange_row_gather
            and exchange_tables
            and shared_l is not None
            and not shared_expert_fused(x_l, *shared_l)
        )
        if shared_l is not None:
            assert shared_limit is not None
            w13, s13, w2, s2 = shared_l
            if split_shared:
                shared_fn = functools.partial(fp8_swiglu_down, w2=w2, s2=s2)
            else:
                shared_fn = functools.partial(
                    fp8_swiglu_mlp, w13=w13, s13=s13, w2=w2, s2=s2, limit=shared_limit
                )
        if exchange_row_gather:
            # The sorted rows are copied by the exchange kernel itself.
            if wide_pairs:
                # The row map: build_local_ranks' scatter form of `tok`.
                _, seg_start, tok, _ = local_ranks
            else:
                with jax.named_scope("moe_early_qsort"):
                    tok, seg_start = local_sort_tokens(
                        local_blocks.reshape(-1),
                        block_hist.sum(axis=0),
                        e_total=e_total,
                        topk=topk,
                        bf16_counts=bf16_sort_counts,
                    )
            table_kw = dict(
                ep=ep,
                width=layout[-1],
                n_rows=tok.shape[0],
                count_at=layout[1],
                e_total=e_total,
                n_pre=n_pre,
                tile_m=capacity,
                light_prepush=light_prepush,
                single_barrier=bool(single_barrier),
                scatter_rows=wide_pairs,
            )
            if split_shared:
                start, done = build_exchange_rows_tables_split(**table_kw)
                handle = start(packed, rank_l[0, 0], q_l, tok)
                # The shared expert's gate_up runs while the rows are in flight.
                shared_in, handle = lax.optimization_barrier((shared_in, handle))
                with jax.named_scope("moe_shared_expert"):
                    shared_in = fp8_swiglu_gate_up(
                        shared_in, w13, s13, limit=shared_limit
                    )
                shared_in, handle = lax.optimization_barrier((shared_in, handle))
                exchange, q_sorted, *pre_tables, exchange_slab = done(
                    handle, q_l, seg_start
                )
            elif exchange_tables:
                exchange, q_sorted, *pre_tables, exchange_slab = (
                    build_exchange_rows_tables(**table_kw)(
                        packed, rank_l[0, 0], q_l, tok, seg_start
                    )
                )
            else:
                exchange, q_sorted = build_tc_exchange_all_gather_rows(
                    ep=ep,
                    width=layout[-1],
                    n_rows=tok.shape[0],
                    single_barrier=bool(single_barrier),
                )(packed, rank_l[0, 0], q_l, tok)
            exchange = exchange.reshape(ep, layout[-1])
        else:
            # Issue the routing exchange before the local sort, so the sort
            # runs on the TensorCore while the exchange is in flight.
            my_counts = block_hist.sum(axis=0)
            packed, q_l, local_blocks, my_counts = lax.optimization_barrier(
                (packed, q_l, local_blocks, my_counts)
            )
            exchange = exchange_all_gather(packed, rank_l).reshape(ep, layout[-1])
            with jax.named_scope("moe_early_qsort"):
                q_sorted, seg_start = local_sort_activations(
                    local_blocks.reshape(-1),
                    my_counts,
                    q_l,
                    e_total=e_total,
                    topk=topk,
                    bf16_counts=bf16_sort_counts,
                )
        exchange, q_sorted, seg_start = lax.optimization_barrier(
            (exchange, q_sorted, seg_start)
        )
        return compute_local(
            q_sorted,
            seg_start,
            exchange,
            local_blocks,
            block_hist,
            topk_weights,
            row_routes,
            rank_l,
            w1_l,
            w2_l,
            w1s_l,
            w2s_l,
            bias_it,
            x_l.dtype,
            expert_load_l,
            pre_tables,
            shared_in,
            shared_fn,
            exchange_slab,
        )

    bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    scale_args = (w1_scale, w2_scale) if form.has_scales else ()
    # `rank` rides the mesh axis so each shard reads its own index out of it;
    # see v2 for why a torch_tpu caller must pass it as a real operand.
    if rank is None:
        rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
    score_bias_args = (score_bias,) if has_score_bias else ()
    hash_args = (hash_topk_ids,) if has_hash_ids else ()
    # Replicated like the score bias: indexed by global expert / slot id.
    expert_slot_args = (expert_slot,) if has_expert_slot else ()
    expert_load_args = (expert_load,) if has_expert_load else ()
    # Replicated: every shard holds the whole shared expert.
    shared_args = () if shared_expert is None else tuple(shared_expert)
    in_specs = (
        (P(ax),) * (5 + len(scale_args) + len(hash_args))
        # A per-token score bias rides with the gating rows.
        + (P(ax) if row_bias else P(),) * len(score_bias_args)
        + (P(),) * (len(expert_slot_args) + len(expert_load_args) + len(shared_args))
        + (P(ax),) * len(bias_args)
    )
    args = (
        (x, rank, w1, w2)
        + scale_args
        + (gating,)
        + hash_args
        + score_bias_args
        + expert_slot_args
        + expert_load_args
        + shared_args
        + bias_args
    )
    NS = jax.sharding.NamedSharding
    args = tuple(jax.device_put(a, NS(mesh, sp)) for a, sp in zip(args, in_specs))
    key = (
        mesh,
        T,
        hidden,
        e_total,
        inter,
        topk,
        bool(renormalize),
        capacity,
        block,
        ragged_stride,
        weight_format,
        rhs_qb,
        act_fn,
        mesh_ep_ranks,
        token_layout,
        bool(sharded_plan),
        x.dtype,
        w1.dtype,
        w2.dtype,
        form.has_scales and w1_scale.dtype,
        pad_scale_rows,
        gating.dtype,
        has_w1_bias and w1_bias.dtype,
        has_w2_bias and w2_bias.dtype,
        scoring_fn,
        has_score_bias and cast(jax.Array, score_bias).dtype,
        row_bias,
        has_hash_ids and cast(jax.Array, hash_topk_ids).dtype,
        has_expert_slot and tuple(cast(jax.Array, expert_slot).shape),
        has_expert_load,
        float(routed_scaling_factor),
        logical_intermediate_size,
        n_pre,
        tc_exchange,
        height_rungs_override,
        tail_barrier,
        bool(onehot_slots),
        hbm_staging,
        dispatch_ahead,
        bool(matmul_scale_repeat),
        exchange_row_gather,
        bool(bf16_sort_counts),
        bool(slim_prepush_barrier),
        bool(fused_dispatch_meta),
        bool(table_kernel),
        bool(exchange_tables),
        bool(kernel_plan_tables),
        has_shared and tuple((a.shape, a.dtype) for a in shared_args),
        float(shared_limit) if has_shared and shared_limit is not None else False,
        int(light_prepush),
        bool(single_barrier),
        int(early_scale_scatter),
        bool(split_exchange),
        bool(onehot_scale_slab),
        bool(pair_kernel),
        bool(wide_pair_kernel),
        bool(decode_combine),
        bool(fuse_metadata),
    )
    out_specs = (P(ax),) + ((P(),) if has_expert_load else ()) + ((P(ax),) * has_shared)
    out_specs = out_specs if len(out_specs) > 1 else out_specs[0]
    sm = _LAYER_SM_CACHE.get(key)
    if sm is None:
        with _LAYER_SM_CACHE_LOCK:
            sm = _LAYER_SM_CACHE.get(key)
            if sm is None:
                sm = jax.shard_map(
                    local_fn,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_vma=False,
                )
                _LAYER_SM_CACHE[key] = sm
    return sm(*args)
