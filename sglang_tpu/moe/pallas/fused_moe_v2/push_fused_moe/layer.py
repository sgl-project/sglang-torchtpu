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
"""Push routed activations into expert slabs and combine their outputs."""

import functools
import threading
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu._logging import get_logger
from sglang_tpu.fp8_swiglu_mlp import fp8_swiglu_mlp
from sglang_tpu.moe.pallas.fused_moe_v2 import host as _host
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
    k32z_table_geometry,
    local_slab_rows,
    off_shard_slab_row,
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
from sglang_tpu.moe.pallas.fused_moe_v2.kernel import (
    build_combine_kernel,
    combine_step_tokens,
    rowquant_fp8,
)
from sglang_tpu.moe.pallas.fused_moe_v2.layer import (
    SCORING_FNS,
    _apply_scoring_fn,
    _combine_arrivals,
    _exchange_layout,
    _pack_routing_exchange,
    _relabel_expert_ids_to_mesh_order,
    _row_routes,
    _to_slots,
    _unpack_routing_exchange,
    _unpermute_load_to_ep_rank_order,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import dedup as _dedup
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.host import (
    _take_small,
    build_routing_tables_sharded,
    local_sort_activations,
    local_sort_tokens,
    push_dispatch_segments,
    push_dispatch_tables,
    routing_pairs_from_tables,
    run_tables,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.kernel import (
    build_push_moe_kernel,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.plan_kernel import (
    build_exchange_rows_tables,
    build_push_tables,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.routing import prepare_routing
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.sc_dispatch import (
    sc_dispatch_chunk_2core,
)
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.tc_exchange import (
    build_tc_exchange_all_gather,
    build_tc_exchange_all_gather_rows,
)
from sglang_tpu.moe.pallas.fused_moe_v2.router_ops import pallas_select
from sglang_tpu.moe.pallas.fused_moe_v2.token_parallel import TokenReplicaLayout

logger = get_logger(__name__)


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

# The K32Z push schedule per bucket (local rows per rank -> options), read at
# trace time and applied to a K32Z build on the exchange-table path; every
# other build, and every bucket not listed, keeps the plain push program.
# Each option leaves the output bits unchanged.
# - 512: x_tables and fold_select take the window-rank glue off the pre-kernel
#   path (a dedup does not pay here: its copies lift the compute+return
#   bound); hot_visit, run_sems and self_direct are schedule changes.
# - 1024: the dedup (x_dedup) cuts the link-bound dispatch
#   bytes by a quarter; defer_push and early_ship overlap the return with the
#   next tiles; run_sems, self_direct and x_dup_unroll are schedule changes.
K32Z_PUSH_RECIPES = {
    512: {
        "x_tables": True,
        "fold_select": True,
        "hot_visit": 16,
        "run_sems": 8,
        "self_direct": True,
    },
    1024: {
        "x_dedup": True,
        "defer_push": True,
        "fold_select": True,
        "early_ship": 2,
        "run_sems": 8,
        "self_direct": True,
        "x_dup_unroll": 8,
    },
}


def _select_routed(row_routes, combined):
    """A row that routed nowhere read arrival rows nobody wrote; its weight is
    zero, but zero times an unwritten NaN is not."""
    return jnp.where(row_routes, combined, jnp.zeros_like(combined))


def _combine_arrivals_fs(
    arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype, routes=None
):
    """v2's `_combine_arrivals`, with the arrival-scale lookup done inside the
    combine kernel (see kernel._build_combine_kernel) where that kernel
    applies; v2's XLA epilogue everywhere else.

    The fused lookup takes the whole arrival-scale mirror as scalar prefetch.
    Tiles taller than one lane block (K32Z's 256 rows) double that mirror,
    and there the fused combine measured 341-1137 us a call against 24-67 us
    at 128-row tiles (V4.1 EP8 on TPU v7x, 512-2048 rows per rank, mirrors
    of 811-848 KB vs 417-454 KB), so those builds take v2's path,
    whose coefficient gather runs in XLA. It gathers from the mirror viewed
    as 128-lane rows: the same values in the same order, so a mirror
    position indexes both views alike.

    `routes` ([t_local, 1] bool, the layer's row_routes; fold_select) folds
    the layer's zeroing select into the fused combine,
    one scalar predicate per token: the same bits as the select after it.
    Where the fused combine does not apply, the select stays in XLA."""
    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    mirror_sublanes, tile_m = arrival_scales.shape

    def select(out):
        return out if routes is None else _select_routed(routes, out)

    if not combine_step_tokens(t_local):
        return select(
            _combine_arrivals(
                arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
            )
        )
    if tile_m > HIDDEN_LANE_BLOCK:
        return select(
            _combine_arrivals(
                arrivals,
                arrival_scales.reshape(-1, HIDDEN_LANE_BLOCK),
                pos,
                mirror_pos,
                topk_weights,
                out_dtype,
            )
        )
    # A DMA does not clamp an out-of-range index the way a gather does.
    # Neither position is ever negative.
    rows = jnp.minimum(pos, recv_rows - 1)
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    combine = build_combine_kernel(
        fuse_scales=True,
        t_local=t_local,
        topk=topk,
        hidden=lane_blocks * lanes,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals.dtype,
        out_dtype=out_dtype,
        **({} if routes is None else {"fold_select": True}),
    )
    if routes is None:
        return combine(arrivals, arrival_scales, scl, topk_weights, rows)
    return combine(arrivals, arrival_scales, scl, topk_weights, rows, routes=routes)


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
    - fuse_metadata allows routing metadata fusion in its measured envelope.

    K32Z recipes: a K32Z build on the exchange-table path (tc_exchange,
    exchange_row_gather, table_kernel, exchange_tables, kernel_plan_tables and
    sharded_plan, as in fused_ep.tuned_push_options) takes the schedule in
    K32Z_PUSH_RECIPES for its rows per rank. Every other build keeps the
    plain push program. The recipe's parts, all bitwise-neutral:
    - fold_select: the zeroing select of unrouted rows inside the fused
      combine.
    - x_tables: each pair's run position rides in its payload word, and the
      combine and slab-row tables are run base + position lookups.
    - x_dedup (dedup.X_MIN_TOKENS..X_MAX_TOKENS rows per rank only): a
      token's row goes to each remote destination once, to its lowest local
      expert there; the other slots' rows are copied from it in the
      destination's tile loads. Runs are owned rows first; the SparseCore
      positions take whole runs (as do destinations with more than
      dedup.default_src_cap duplicates). Visits go lightest first on the
      SparseCore, then by local expert id, so every owner is visited before
      its duplicates.
    - defer_push, early_ship: see the push kernel builder.
    - hot_visit=k: each destination's heaviest expert moves to visit
      position k, past the SparseCore positions.
    - run_sems, self_direct, x_dup_unroll: see the push kernel builder.
    - slab rows: the plan's per-pair slab rows (the scale-slab scatter's
      index) are computed inside the exchange-table kernel from the exchanged
      pair words instead of XLA's window histograms and lookups; the same
      integers (plan_kernel.slab_rows_from_pairs).

    The optional replicated FP8 shared_expert runs while pre-push is in
    flight; its output is returned separately, last. The caller owns the
    static expert layout and load accumulator.

    K32Z (weight_format=WeightFormat.K32Z) takes fused_ep_moe_v2's K32Z
    operands: float4 codes and the flat uint8 shift tables."""
    del preprocess_topology
    if early_scale_scatter not in (0, 1, 2):
        raise ValueError(f"early_scale_scatter is 0, 1 or 2; got {early_scale_scatter}")
    dispatch_ahead = int(dispatch_ahead)
    if dispatch_ahead < 1:
        raise ValueError(f"dispatch_ahead must be >= 1; got {dispatch_ahead}")
    if shared_limit is not None and shared_expert is None:
        raise ValueError("shared_limit requires shared_expert")
    form = weight_form(weight_format)
    rhs_qb = QB4 if rhs_qb is None else int(rhs_qb)
    for name, w in (("w1", w1), ("w2", w2)):
        # Against the format's own element type: K32Z's weights are float4
        # like FP4's, which weight_format_of_dtype names FP4.
        actual = weight_format_of_dtype(w.dtype)
        if jnp.dtype(w.dtype) != jnp.dtype(form.weight_dtype):
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert weights and "
                f"{name} is {jnp.dtype(w.dtype).name}"
                f"{f', which is the {actual!r} form' if actual else ''}"
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
    # K32Z's shift tables come flat only, [E * rows, cols] uint8, whole u8
    # sublane tiles per expert (so the per-expert view is a bitcast). The
    # gate/up field width is whatever the table's columns say it is.
    k32z_gate_up_bits = None
    # How its FFN body walks the output chunks, read once per trace
    # (host.k32z_roll_plan) and keyed below like every other build choice.
    k32z_ffn_roll = None
    k32z_roll_from = 0
    # And whether its scale mirrors take 128-lane units (host.K32Z_MIRROR_UNIT).
    mirror_unit = 0
    k32z_rungs = None
    plan_tile_m = capacity
    if form.scale_layout == "column_shift":
        cols1 = w1_scale.shape[-1] if w1_scale.ndim == 2 else -1
        k32z_gate_up_bits = (cols1 * 8) // (2 * inter) if inter else 0
        mirror_unit = _host.K32Z_MIRROR_UNIT
        k32z_rungs = (
            None
            if tuple(_host.K32Z_RUNGS) == _host.K32Z_DEFAULT_RUNGS
            else tuple(_host.K32Z_RUNGS)
        )
        # The plan builders use their tile_m for the scale mirror's
        # sublanes alone (and divisibility checks a unit passes), so they
        # take the mirror's lanes (host.mirror_lanes).
        plan_tile_m = _host.mirror_lanes(capacity, mirror_unit)
        geo = k32z_table_geometry(hidden, inter, k32z_gate_up_bits)
        for name, s, want in (
            ("w1_scale", w1_scale, (e_total * geo.rows1, geo.cols1)),
            ("w2_scale", w2_scale, (e_total * geo.rows2, geo.cols2)),
        ):
            if tuple(s.shape) != want or jnp.dtype(s.dtype) != jnp.dtype(jnp.uint8):
                raise ValueError(
                    f"{name} {jnp.dtype(s.dtype).name}{tuple(s.shape)} is not the "
                    f"flat K32Z shift table uint8{want} the kernel takes"
                )
    elif pad_scale_rows:
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
    # The K32Z schedule for this bucket (K32Z_PUSH_RECIPES) applies on the
    # exchange-table path; every other build keeps the plain push program.
    recipe = {}
    if k32z_gate_up_bits is not None and (
        tc_exchange
        and exchange_row_gather
        and table_kernel
        and exchange_tables
        and kernel_plan_tables
        and sharded_plan
    ):
        recipe = K32Z_PUSH_RECIPES.get(t_local, {})
    defer_push = recipe.get("defer_push", False)
    fold_select = recipe.get("fold_select", False)
    x_dedup = recipe.get("x_dedup", False)
    x_tables = recipe.get("x_tables", False) or x_dedup
    early_ship = recipe.get("early_ship", 0)
    if not _dedup.x_applies(t_local):
        # Outside the dedup buckets the build is the default one.
        x_tables = x_dedup = False
    exchange_slab_rows = bool(
        _host.push_slab_rows_mode(
            k32z=k32z_gate_up_bits is not None, t_local=t_local
        )
        and exchange_tables
        and kernel_plan_tables
        and table_kernel
        and exchange_row_gather
        and not x_tables
    )
    x_cap = 0
    x_tag = ""
    visit_order = 0
    if x_dedup:
        x_cap = _dedup.default_src_cap(ep, t_local)
        x_tag = f"_xd{x_cap}"
        # The lowest-id owner needs SparseCore lightest, then id order.
        visit_order = 1
    elif x_tables:
        x_tag = "_xt"
    # Schedule options with preconditions: each drops out where they fail.
    hot_visit = recipe.get("hot_visit", 0)
    if hot_visit and not (
        n_pre <= hot_visit < g_local
        and table_kernel
        and (light_prepush or visit_order == 1)
        and not x_dedup
    ):
        hot_visit = 0
    run_sems = recipe.get("run_sems", 0)
    if run_sems and (ep % run_sems or (ep // run_sems) & (ep // run_sems - 1)):
        run_sems = 0  # ep not split into power-of-two source groups
    self_direct = recipe.get("self_direct", False)
    x_dup_unroll = recipe.get("x_dup_unroll", 0) if x_dedup else 0
    if exchange_row_gather and not tc_exchange:
        raise ValueError(
            "exchange_row_gather needs the TensorCore exchange (tc_exchange)"
        )
    height_rungs_override = int(height_rungs_override)
    tail_barrier = bool(tail_barrier)
    hbm_staging = bool(hbm_staging)
    # Every window past the last visit position issues the same program.
    dispatch_ahead = min(dispatch_ahead, g_local)
    if k32z_gate_up_bits is not None:
        # How the K32Z bodies walk their chunks, from the knobs and this
        # build's slab (host.k32z_roll_plan); keyed below like the rest.
        k32z_ffn_roll, k32z_roll_from = _host.k32z_roll_plan(
            ragged_stride, ep, g_local, hidden, inter
        )
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
        # Recipe options, passed only when on: the default build keeps its
        # cache key as well as its program.
        **{
            k: v
            for k, v in (
                ("defer_push", defer_push),
                ("x_dedup", x_dedup),
                ("early_ship", early_ship),
                ("run_sems", run_sems),
                ("self_direct", self_direct),
                ("x_dup_unroll", x_dup_unroll),
            )
            if v
        },
        **(
            {}
            if k32z_gate_up_bits is None
            else {
                "k32z_gate_up_bits": k32z_gate_up_bits,
                "k32z_ffn_roll": k32z_ffn_roll,
                "k32z_roll_from": k32z_roll_from,
                "mirror_unit": mirror_unit,
                # Read at trace time; None where it is the default ladder.
                "k32z_rungs": k32z_rungs,
            }
        ),
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
        elif k32z_gate_up_bits is not None:
            # The same for the K32Z shift tables: whole u8 sublane tiles.
            geo = k32z_table_geometry(hidden, inter, k32z_gate_up_bits)
            w1s_l = next(operands).reshape(g_local, geo.rows1, geo.cols1)
            w2s_l = next(operands).reshape(g_local, geo.rows2, geo.cols2)
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
        block_hist = pair_block_hist(local_blocks, e_total)
        scale_bits_l = (
            None
            if row_scale_l is None
            else (
                lax.bitcast_convert_type(
                    row_scale_l[:, 0].astype(jnp.float32), jnp.int32
                )
            )
        )
        if not x_tables:
            packed = _pack_routing_exchange(
                local_blocks.reshape(-1), scale_bits_l, block_hist.sum(axis=0), layout
            )
            return topk_weights, packed, local_blocks, block_hist, row_routes, None
        # (dedup.sender_tables): each pair's run position rides in its payload
        # word; under x_dedup the runs are owned rows first and the count
        # words carry the owned counts.
        idx_l, count_words, sort_key, sort_counts = _dedup.sender_tables(
            topk_idx,
            me,
            g_local=g_local,
            e_total=e_total,
            block=block,
            block_hist=block_hist,
            dedup=x_dedup,
        )
        packed = _pack_routing_exchange(
            _dedup.pack_pairs(local_blocks.reshape(-1), idx_l.reshape(-1)),
            scale_bits_l,
            count_words,
            layout,
        )
        x_sender = (idx_l, sort_key, sort_counts)
        return topk_weights, packed, local_blocks, block_hist, row_routes, x_sender

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
        shared_l=None,
        x_sender=None,
        kernel_slab_row=None,
    ):
        """Dispatch tables, SparseCore pre-push, plan, kernel and combine.

        `pre_tables`, where the exchange kernel already built them, are
        build_push_tables' outputs; `shared_l`, the shared expert computed on
        `x_l` between the pre-push and the kernel; `x_sender` (x_tables) the
        sender's (run positions, sort key, sort counts)."""
        me = rank_l[0, 0]
        pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
            exchange, layout, t_local, topk, e_total
        )
        if x_dedup:
            # The count words carry the owned counts above the pair counts.
            rows_by_dest = jnp.bitwise_and(rows_by_dest, _dedup.X_LOW_MASK)
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
                    tile_m=plan_tile_m,
                    light_prepush=light_prepush,
                    **({"visit_order": visit_order} if visit_order else {}),
                    **({"hot_visit": hot_visit} if hot_visit else {}),
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
        # x_dedup's table kernel returns its x tables last.
        x_pair = shard_tables.pop() if x_dedup else None
        # Compute the pre-push operands before the routing plan, so XLA
        # launches the SparseCore dispatch ahead of it and the two overlap.
        # The plan reads pairs_g, so it follows the barrier either way; the
        # run tables only add a layout copy at it (`slim_prepush_barrier`).
        barriered_runs = () if slim_prepush_barrier else tuple(runs)
        # The shared expert's rows come out of the same barrier as the
        # pre-push operands, so it cannot be scheduled ahead of them.
        shared_x = (x_l,) if shared_l is not None else ()
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
        if shared_l is not None:
            assert shared_limit is not None
            with jax.named_scope("moe_shared_expert"):
                shared_out = (fp8_swiglu_mlp(*shared_x, *shared_l, limit=shared_limit),)
        plan_kw = dict(
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=plan_tile_m,
            shard_stride=ragged_stride,
        )
        gathered = None
        if sharded_plan:
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
        x_src = None
        if kernel_plan_tables and x_tables:
            # Every pair's run position came with it, so its arrival row,
            # mirror position and slab row are run base + idx; under x_dedup
            # also SRC, each duplicate row's owner slab row.
            assert shard_tables is not None and x_sender is not None
            recv_base_me, mirror_base, run_base = shard_tables[4]
            slab_row, x_src = _dedup.receiver_tables(
                pairs_g,
                me,
                run_base,
                g_local=g_local,
                t_local=t_local,
                topk=topk,
                shard_stride=ragged_stride,
                ep=ep,
                x_g=None if x_pair is None else x_pair[0],
                cap=x_cap,
            )
            pair_tables = _dedup.pair_positions(
                local_blocks.reshape(t_local, topk),
                x_sender[0],
                recv_base_me,
                mirror_base,
            ) + (slab_row,)
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
                **({} if kernel_slab_row is None else {"slab_row": kernel_slab_row}),
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
        if early_scale_scatter == 2:
            # Nor behind the pre-push: the scatter may take the SparseCore
            # first; the light pre-push still lands under the shared expert.
            slab_row, scale_bits_g = lax.optimization_barrier((slab_row, scale_bits_g))
        elif early_scale_scatter:
            # The shared expert joins after the scatter instead (below), so
            # the scatter's SparseCore work can run under it.
            slab_row, scale_bits_g, slab = lax.optimization_barrier(
                (slab_row, scale_bits_g, slab)
            )
        else:
            slab_row, scale_bits_g, slab, *shared_out = lax.optimization_barrier(
                (slab_row, scale_bits_g, slab) + shared_out
            )
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on (v2's scale slab; see there).
        if form.quantized_activations:
            scale_bits = (
                _repeat_bits(cast(jax.Array, scale_bits_g), topk)
                if matmul_scale_repeat
                else jnp.repeat(cast(jax.Array, scale_bits_g), topk)
            )
            scale_rows = act_scale_slab_rows(ragged_stride)
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
            **({} if x_pair is None else {"x": (x_pair[1], x_src)}),
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
        if fold_select:
            combined = _combine_arrivals_fs(
                arrivals,
                arrival_scales,
                pos,
                mirror_pos,
                topk_weights,
                out_dtype,
                routes=row_routes,
            )
        else:
            combined = _combine_arrivals_fs(
                arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
            )
            combined = _select_routed(row_routes, combined)
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
        selected = select_local(
            gate_l, hash_l, score_bias_l, row_scale_l, expert_slot_l, rank_l
        )
        topk_weights, packed, local_blocks, block_hist, row_routes, x_sender = selected
        pre_tables = None
        kernel_slab_row = None
        my_counts = block_hist.sum(axis=0)
        if exchange_row_gather:
            # The sorted rows are copied by the exchange kernel itself.
            with jax.named_scope("moe_early_qsort"):
                if x_dedup:
                    # Owned rows first in each run: the counting sort over
                    # key 2 * id + dup, read back per expert.
                    tok, seg_start = local_sort_tokens(
                        x_sender[1],
                        x_sender[2],
                        e_total=2 * e_total,
                        topk=topk,
                        bf16_counts=bf16_sort_counts,
                    )
                    seg_start = seg_start[0::2]
                else:
                    tok, seg_start = local_sort_tokens(
                        local_blocks.reshape(-1),
                        my_counts,
                        e_total=e_total,
                        topk=topk,
                        bf16_counts=bf16_sort_counts,
                    )
            if exchange_tables:
                x_kw = {}
                if x_tables:
                    x_kw = dict(
                        x_tag=x_tag, x_dedup=x_dedup, x_npre=n_pre, x_cap=x_cap
                    )
                if exchange_slab_rows:
                    # The plan's slab rows from the kernel too.
                    x_kw = dict(
                        slab_pairs=t_local * topk,
                        slab_off=off_shard_slab_row(ragged_stride, ep),
                        slab_stride=ragged_stride,
                    )
                ex_out = build_exchange_rows_tables(
                    ep=ep,
                    width=layout[-1],
                    n_rows=tok.shape[0],
                    count_at=layout[1],
                    e_total=e_total,
                    n_pre=n_pre,
                    tile_m=plan_tile_m,
                    light_prepush=light_prepush,
                    single_barrier=bool(single_barrier),
                    **({"visit_order": visit_order} if visit_order else {}),
                    **({"hot_visit": hot_visit} if hot_visit else {}),
                    **x_kw,
                )(packed, rank_l[0, 0], q_l, tok, seg_start)
                if exchange_slab_rows:
                    ex_out, kernel_slab_row = ex_out
                exchange, q_sorted, *pre_tables = ex_out
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
            x_l,
            shared_l,
            x_sender,
            kernel_slab_row,
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
        bool(fuse_metadata),
        k32z_gate_up_bits,
        k32z_ffn_roll,
        k32z_roll_from,
        mirror_unit,
        k32z_rungs,
        visit_order,
        defer_push,
        fold_select,
        x_tables,
        x_dedup,
        x_cap,
        early_ship,
    ) + ((True,) if exchange_slab_rows else ())
    schedule = (hot_visit, run_sems, self_direct, x_dup_unroll)
    if schedule != (0, 0, False, 0):
        # Keyed only when one is set (the default key unchanged).
        key = key + (("schedule",) + schedule,)
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
