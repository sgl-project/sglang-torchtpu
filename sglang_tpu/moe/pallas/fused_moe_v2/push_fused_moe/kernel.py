# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/push_fused_moe/kernel.py
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
"""The push-dispatch fused EP MoE kernel.

v2's kernel (fused_moe_v2/kernel.py) with the activation all-gather
replaced by in-kernel pushes of each shard's routed rows into the
destination's expert slab; see `_build_push_moe_kernel`. Everything this
module does not redefine -- the expert FFN bodies, the wire epilogue, the
barrier, the table counts -- is v2's, imported unchanged.
"""

import threading
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu._logging import get_logger
from sglang_tpu.moe.pallas.fused_moe_v2 import host
from sglang_tpu.moe.pallas.fused_moe_v2.host import (
    FP4,
    FP8,
    HIDDEN_LANE_BLOCK,
    OUT_PARITIES,
    PACK4,
    QB4,
    ROWBLK,
)
from sglang_tpu.moe.pallas.fused_moe_v2.kernel import (
    _REF_BITCAST_JAX_VERSION,
    ACT_FNS,
    COLLECTIVE_ID,
    N_PREFETCH,
    N_PUSH_TABLES,
    N_SCALE_TABLES,
    N_SLAB_TABLES,
    N_TRANSPORT_TABLES,
    N_WEIGHT_SLABS,
    WEIGHT_DMA_PRIORITY,
    _all_pairs_barrier,
    _and_nonempty,
    _rows_wait,
    _slot_copy,
    _tile_row_scales,
    expert_ffn_bf16,
    expert_ffn_blockscale,
    expert_ffn_colscale,
    expert_ffn_colscale_rolled,
    expert_ffn_fp8,
    expert_ffn_int8,
    k32z_block_readers,
    k32z_chunk_decoders,
    k32z_fill_colscale,
    wire_row_and_scale,
)

logger = get_logger(__name__)

# Two more scalar-prefetch tables than v2: the dispatch segment table and its
# meta row (host.push_dispatch_segments).
N_DISPATCH_TABLES = 2
# And two more under x_dedup: the x table and SRC.
N_X_TABLES = 2
# x table words: landed rows | first SRC word << X_SHIFT (dedup.X_IDX_SHIFT).
X_SHIFT = 16
# Tile heights the FFN body is emitted at (v2's HEIGHT_RUNGS, chosen for
# this build). Each rung is a full copy of the body in the program text: at
# the DP16 g25 shape four rungs are 47.0k LLO bundles and two 25.7k, and the
# larger program costs more than the rows the finer ladder saves (kernel
# slope at the served g25 / g9 shapes on eight shards: 271 -> 257 and
# 130 -> 116 us a call, bit-exact).
HEIGHT_RUNGS = 2
# early_ship values: 0 off, 2 per-tile pieces (the value is the lag in tiles).
EARLY_SHIP_MODES = (0, 2)


def tile_height_rungs(tile_m, override=0):
    """v2's equally spaced tile-height ladder at this build's HEIGHT_RUNGS.

    v2 emits the ladder only where a routed expert is expected to fit in one
    tile (rows_alloc <= ep * g_local * tile_m); this build always emits it,
    which is what the push dispatch was measured with. `override` (0 = the
    default HEIGHT_RUNGS) is the rung count the caller chose.
    """
    rungs = override or HEIGHT_RUNGS
    step = tile_m // rungs
    if rungs < 2 or step % ROWBLK or step * rungs != tile_m:
        return (tile_m,)
    return tuple(range(step, tile_m + 1, step))


def _entry_barrier_first_phase(ep):
    """_all_pairs_barrier's first phase: every shard signals every shard."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(jnp.int32(i),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, ep)


def _build_push_moe_kernel(
    *,
    g_local,
    capacity,
    hidden,
    inter,
    ep,
    weight_format=host.WeightFormat.FP8,
    rhs_qb=QB4,
    ragged_rows_alloc=None,
    act_fn="silu",
    has_w1_bias=False,
    has_w2_bias=False,
    pad_scale_rows=False,
    logical_intermediate_size=None,
    push_tokens,
    sc_prepush,
    height_rungs_override=0,
    tail_barrier=True,
    hbm_staging=False,
    dispatch_ahead=1,
    single_barrier=False,
    k32z_gate_up_bits=host.K32Z_GATE_UP_BITS[0],
    k32z_ffn_roll=0,
    k32z_roll_from=0,
    mirror_unit=0,
    k32z_rungs=None,
    defer_push=False,
    x_dedup=False,
    early_ship=0,
    run_sems=0,
    self_direct=False,
    x_dup_unroll=0,
):
    """Build the pallas_call and the function that invokes it.

    The v2 kernel with the activation all-gather replaced by push dispatch.
    `push_tokens` is t_local: the lhs operand is only this shard's rows,
    sorted by global expert (host.local_sort_activations), and every shard
    pushes each (destination, expert) run of them as one DMA straight into
    the destination's expert slab (`dispatch_hbm`, aliased to the `slab`
    operand), in the destination's expert visit order, signalling the
    destination's per-local-expert receive semaphore. Visit positions
    [0, sc_prepush) were pushed by the SparseCore before this kernel
    (sc_dispatch.sc_dispatch_chunk_2core), so their tiles start with no wait;
    the TensorCore pushes the rest `dispatch_ahead` visit positions ahead of
    its own compute (the first window right after the head barrier, one more
    position per visit; a value of g_local or more pushes everything there),
    and waits an expert's semaphore for its pair count before loading its
    first tile. Tiles then read their rows as one contiguous slab copy, so no
    token-gather table exists. Only data movement changes: same rows, same
    math as the all-gather build.

    `pad_scale_rows` takes the four-bit scale tables as [G, rows, N] with
    rows = host.fp4_scale_rows(K, rhs_qb, padded=True): K / rhs_qb blocks
    padded to the f32 sublane tile, the [G * rows, N] row-padded flat operand
    viewed per expert. Each slot then stages an expert's whole padded table
    and the FFN reads only its first K / rhs_qb rows, so the result is the
    unpadded build's bit for bit.

    `tail_barrier=False` drops the all-pairs barrier after the transport
    drain. Every write a peer makes into this shard during the call is
    counted by an exact wait before that point (dispatch rows per local
    expert, arrival rows and mirror sublanes), and this shard's own sends are
    drained, so the barrier waits for nothing this call still owes. A peer
    that leaves first next writes here only from a kernel that opens with an
    entry barrier of its own (this kernel's head barrier, the SparseCore
    pre-push's, the TensorCore routing exchange's), which holds the write
    until this shard has entered that kernel too.

    `hbm_staging=True` ships result tiles the two-hop way even where a tile
    could ship straight out of `out_vm`: each tile is committed to the HBM
    contribution slab and every (expert, destination) run is pushed once the
    visit's tiles are committed. Same bytes to the same places; what changes
    is that the remote pushes no longer hold a VMEM staging slot, so when the
    fabric is the busy resource the tile loop runs ahead of it instead of
    waiting at the slot-reuse guard. A staged build commits each tile with
    one copy into a contribution slab shaped like the expert slab, and
    pushes each visit's runs at the end of the NEXT visit, so the commits it
    waits for there have had a whole visit to land.

    `single_barrier=True` opens with the first phase of the all-pairs entry
    barrier only. The second phase guards the barrier semaphore against a
    peer that races into this kernel's next call while this shard still
    waits in this one; the layer's routing exchange stands between any two
    calls, and a peer leaves it only once this shard's row has landed, which
    this shard sends only after leaving this call.

    `weight_format=K32Z` runs v2's K32Z expert compute (the decode, FFN and
    column-scale functions are v2's; `k32z_gate_up_bits` is its gate/up
    shift-field width) on this kernel's own weight copies and waits: rungs
    host.k32z_tile_height_rungs(tile_m), and a visit taller than
    host.K32Z_STREAMED_ROWS lands both slabs at its head. Tiles above 128
    rows stage their scales in 3-D mirrors (host.mirror_shape).

    Return path:
    - `defer_push` (`_dp`): on the two-hop path (commit to the contribution
      slab, push each run from it), each expert's drain and push run in the
      next visit's first tile, after that tile's commits; a staged build would
      otherwise push at the end of the next visit, an unstaged one at the end
      of its own.
    - `early_ship` (`_es2`): on the two-hop path, each run is returned in
      per-tile pieces from inside its expert's tile loop: at tile t the part
      of every run that lies in tile t - 2 ships (one row DMA and one mirror
      DMA per piece; the reuse guard of tile t has landed the commits of
      tiles up to t - 2), and push_expert ships what is left of each run in
      the visit's last two tiles. The same rows and mirror units reach the
      same places as without it, issued earlier.

    Dispatch dedup (push_fused_moe.dedup):
    - `x_dedup` (`_xd`): two more scalar-prefetch tables, x [g_local, ep]
      (rows each (expert, source) run landed | the run's first SRC word
      << 16) and SRC (the owner's slab row of every duplicate row). A run
      lands its owned rows first and its duplicates were not sent, so the
      dispatch waits count landed rows (the dispatch meta's receive counts
      already do) and an expert with duplicates loads each tile as the
      landed stretches between them (a run's pad rows ride with the next
      run's owned rows, so at most ep + 1 copies) plus one row copy per
      duplicate from its owner's row. The owner is a SparseCore position or
      an earlier visit, whose rows its own head wait took. Every copy
      signals the tile's semaphore and they partition the tile's rows, so
      the tile's byte-count wait is unchanged and exact.

    Schedule options (same output bits; the layer enables them per bucket):
    - `run_sems=k` (`_rs<k>`, k in 2, 4, 8 dividing ep): one receive
      semaphore per (local expert, source group of ep / k consecutive
      sources) instead of one per expert. A sender signals its own group's
      semaphore; a TensorCore-position expert waits each group's landed
      rows once, before the first tile whose rows its first run starts in
      (the head wait for tile 0, the tile loop before tile t + 1's fetch),
      so a tile waits only the sources whose runs reach it. Every group is
      waited exactly its landed rows, so the waits balance; the tile reads
      the same slab rows, so the same output bits.
    - `self_direct` (`_sd`): no self dispatch copy at TensorCore positions;
      the tile fetch reads the own run's rows straight from lhs_hbm at the
      self segment's source row (the segment the sender skipped) and the
      rest of the tile from the slab, and the receive waits (per expert, or
      the own group's under run_sems) are the own rows less. The same bytes
      reach lhs_vm, so the same output bits.
    - `x_dup_unroll=k` (`_xu<k>`, k in 2, 4, 8; x_dedup's gather only): the
      per-duplicate row copy loop unrolled k times by hand (lax.fori_loop
      takes `unroll` only with static bounds), then the remainder; the same
      DMAs in the same order.
    """
    x_dedup = bool(x_dedup)
    defer_push = bool(defer_push)
    if early_ship not in EARLY_SHIP_MODES:
        raise ValueError(
            f"early_ship is one of {EARLY_SHIP_MODES} (0: off); got {early_ship!r}"
        )
    early_ship = int(early_ship)
    if dispatch_ahead < 1:
        raise ValueError(
            f"dispatch_ahead {dispatch_ahead} must be at least 1: a visit "
            "position is pushed before the visit that reads it"
        )
    if pad_scale_rows and weight_format != host.WeightFormat.FP4:
        raise ValueError(
            "pad_scale_rows describes the four-bit block-scale tables; "
            f"weight format {weight_format!r} has none"
        )
    if act_fn not in ACT_FNS:
        raise NotImplementedError(
            f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}"
        )
    # The kernel reads its tiles out of the pushed expert slab, and ships
    # results over an all-to-all carrying unweighted rows; it returns
    # (arrival rows, arrival scales f32). The element type of the staging and
    # of the wire is the weight format's, not a literal here.
    form = host.weight_form(weight_format)
    rhs_packed4 = weight_format == host.WeightFormat.FP4
    k32z = weight_format == host.WeightFormat.K32Z
    # Both stream their e2m1 codes as packed u32 words.
    packed4 = rhs_packed4 or k32z
    if k32z:
        # Raises naming what the shape or the field width cannot take.
        host.k32z_table_geometry(hidden, inter, k32z_gate_up_bits)
        if capacity % host.K32Z_RUNG_STEP:
            raise ValueError(
                f"K32Z tile height {capacity} is not a whole number of "
                f"{host.K32Z_RUNG_STEP}-row rungs"
            )
        if height_rungs_override:
            raise ValueError(
                "K32Z emits host.k32z_tile_height_rungs (each K32Z body is "
                "large); height_rungs_override does not apply to it"
            )
        k32z_ffn_roll = host.check_k32z_ffn_roll(k32z_ffn_roll, hidden, inter)
    else:
        k32z_ffn_roll = 0
    if (
        logical_intermediate_size is not None
        and int(logical_intermediate_size) != logical_intermediate_size
    ):
        raise ValueError("logical intermediate size must be an integer")
    effective_inter = (
        inter if logical_intermediate_size is None else int(logical_intermediate_size)
    )
    trim_gate_up_padding = effective_inter != inter
    if trim_gate_up_padding and (
        not rhs_packed4
        or rhs_qb != 512
        or has_w1_bias
        or effective_inter <= 0
        or effective_inter >= inter
        or effective_inter % 256
        or inter - effective_inter >= rhs_qb
    ):
        raise ValueError(
            "logical intermediate optimization requires FP4/G512, no W1 bias, "
            "a positive 256-aligned logical size, and less than one trailing G512 block"
        )
    # This is the last N column read, not the end of the physical allocation.
    w1_last_valid_column = inter + effective_inter

    # The weight prefetch distance and the slot count it needs are chosen
    # below, once the shapes they are sized against have been checked: they
    # are a property of THIS build and not of the process, so everything
    # here reads the two locals and nothing reads a module constant.
    #
    # Everything from here to the ragged-stride check below tests a value
    # that came IN from the caller -- a model shape, a mesh width, a serving
    # config -- so each one refuses rather than asserts. An assert is the
    # right tool for this function's own arithmetic, like the slot relation
    # below, and the wrong tool for the caller's operands: `python
    # -O` and PYTHONOPTIMIZE are ordinary things to find in a container
    # image, and under them an assert lets the illegal configuration build
    # and be cached.
    if capacity % ROWBLK:
        raise ValueError(
            f"tile height {capacity} is not a whole number of {ROWBLK}-row "
            "blocks, which is the unit every transport here moves"
        )
    if rhs_packed4:
        # Tested before the modulo below rather than after it: every other
        # guard in this span was converted from an assert so that a caller's
        # operand is refused by name, and a block size of zero divided by
        # the operand first and came back as a ZeroDivisionError.
        if rhs_qb < 1:
            raise ValueError(
                f"the four-bit block size is {rhs_qb}; it is the number of "
                "contraction rows one weight scale covers and there is no "
                "block smaller than one row"
            )
        if hidden % rhs_qb or inter % rhs_qb:
            raise ValueError(
                f"the four-bit block size {rhs_qb} has to divide both "
                f"hidden={hidden} and inter={inter}; the kernel blocks BOTH "
                "matmuls at one block size"
            )
        # Packed rows per k-block must land on the u32 sublane tile.
        if rhs_qb % (host.U32_SUBLANE_TILE * PACK4):
            raise ValueError(
                f"the four-bit block size {rhs_qb} is not a whole number of "
                f"the packed-weight row tile "
                f"({host.U32_SUBLANE_TILE * PACK4} rows): {PACK4} four-bit "
                f"values pack into a 32-bit word and those words tile to "
                f"{host.U32_SUBLANE_TILE} sublanes"
            )
    # The widening chunk has to tile both contractions exactly.
    if weight_format == host.WeightFormat.INT8 and (
        hidden % host.WIDEN_KCHUNK or inter % host.WIDEN_KCHUNK
    ):
        raise ValueError(
            f"integer weights widen one {host.WIDEN_KCHUNK}-row "
            f"contraction chunk at a time, which has to divide both "
            f"hidden={hidden} and inter={inter}"
        )
    # True-length remote pushes: out_vm, contrib and recv take the per-row
    # [rows, lane blocks, 128] geometry, which row-granular offsets require.
    if (
        hidden < HIDDEN_LANE_BLOCK
        or hidden % HIDDEN_LANE_BLOCK
        or hidden > host.HIDDEN_MAX_BLOCKS * HIDDEN_LANE_BLOCK
    ):
        raise ValueError(
            f"hidden {hidden} is not between one and "
            f"{host.HIDDEN_MAX_BLOCKS} whole {HIDDEN_LANE_BLOCK}-lane "
            "blocks, which the per-row transport geometry requires"
        )
    # The intermediate axis had no floor at all, so an expert whose FFN has
    # no intermediate channels built a program and cached it: the estimate
    # for it clears the budget comfortably and every divisibility check
    # passes, because zero is a whole number of everything. The floor is one
    # lane block, the same rule and the same reason as the hidden axis
    # above -- an intermediate narrower than the vector unit's lane count is
    # not an FFN, and the diagnostic an operator eventually receives names a
    # reduction or a reshape rather than the weight that was empty.
    if inter < HIDDEN_LANE_BLOCK:
        raise ValueError(
            f"inter {inter} is narrower than one {HIDDEN_LANE_BLOCK}-lane "
            "block; an expert whose FFN has no intermediate channels to "
            "speak of is not a layer this kernel can serve"
        )
    lane_blocks = host.row_lane_blocks(hidden)
    has_scales = form.has_scales
    has_act_scale = form.quantized_activations
    # The prefetch schedule this build can afford: the deepest runway whose
    # nbuf copies of one expert still leave the rest of the buffers inside
    # the budget, walked down from the ceiling one slot at a time. Every
    # operand it is sized against has been checked above, which is why the
    # choice is made here and not at the head of the function.
    slots = host.fit_weight_slots(
        g_local,
        capacity,
        hidden,
        inter,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        k32z_gate_up_bits=k32z_gate_up_bits,
        k32z_ffn_roll=k32z_ffn_roll,
        mirror_unit=mirror_unit,
    )
    prefetch_distance = slots.distance
    nbuf = slots.nbuf
    # The relation the whole schedule rests on: a refill writes its slot
    # while the readers of `distance` earlier experts are still live, so
    # distance + 1 consecutive experts have to occupy distinct slots. This
    # function's own arithmetic, hence an assert.
    assert 0 <= prefetch_distance < nbuf, (
        f"weight prefetch distance {prefetch_distance} must satisfy "
        f"0 <= distance < nbuf={nbuf}, so that distance + 1 consecutive "
        "experts occupy distinct weight slots"
    )
    # Distance zero has no runway: one slot, and a visit starts its own
    # weights and then waits on them. Every prefetch site below is switched
    # on this rather than deleted, because the two schedules differ only in
    # WHERE the start happens -- a visit ahead, or at the visit's own head.
    prefetches = prefetch_distance > 0

    # Whether each expert's weights are waited by the block that first reads
    # it rather than waiting at the beginning of each expert's visit.
    # fp4, fp8 expert FFN are running different code paths. This flag can
    # be activated only for fp4 expert FFN now. But the idea should be
    # applicable to fp8 expert FFN as well. K32Z streams them the same way
    # for its short visits and waits them at the visit head for tall ones.
    streams_weights = packed4
    # Whether issue async copy the NEXT expert's weights as soon as we finish
    # reading the current expert's weights, instead of at the beginning of the
    # next expert's visit.
    #
    # The overlap is free in VMEM, because the next start writes the slot its
    # reader has just finished with: for example gate/up is dead from the
    # last gate/up block of an expert's last tile, which is overlapped with
    # a whole down matmul plus an epilogue.
    starts_next_weights = streams_weights and not prefetches
    if packed4:
        # Asked here, beside the other device reads, because the packed
        # four-bit layout is the one place a written-down device fact and a
        # queried one have to agree: the block-size check above uses the
        # constant and the buffers the accounting just sized used the
        # record.
        host.check_u32_sublane_tile()
    # The only thing standing between an over-budget model and a kernel that
    # will not fit. It must not be strippable. What it refuses is now only
    # what the descent above could not rescue: the figures it quotes are the
    # ones at distance zero, where the weight slabs are a single expert and
    # there is no runway left to give back.
    if not slots.fits:
        raise ValueError(
            f"the kernel's VMEM buffers need "
            f"{slots.vmem_bytes / 2**20:.1f}MiB, over the "
            f"{slots.limit / 2**20:.1f}MiB budget, for {g_local} local experts "
            f"of {hidden}x{inter} at capacity={capacity} with the weight "
            f"prefetch already given up (nbuf={nbuf}, one expert of weight "
            f"slabs). Nothing further is bought by shortening the runway; "
            f"the remaining levers are the tile height, the expert shape and "
            f"the number of local experts"
        )
    if prefetch_distance < host.MAX_WEIGHT_PREFETCH_DISTANCE:
        logger.info(
            "fused EP MoE: %d local experts of %dx%d at capacity=%d fit at "
            "prefetch distance %d (%d weight slots, %.1fMiB of a %.1fMiB "
            "budget), below the ceiling of %d",
            g_local,
            hidden,
            inter,
            capacity,
            prefetch_distance,
            nbuf,
            slots.vmem_bytes / 2**20,
            slots.limit / 2**20,
            host.MAX_WEIGHT_PREFETCH_DISTANCE,
        )
    # ragged_rows_alloc must cover every tile READ as well as every commit
    # -- the shard's total rows plus tile_m, aligned to tile_m -- because a
    # tail tile reads a full window.
    if (
        ragged_rows_alloc is None
        or ragged_rows_alloc % ROWBLK
        or ragged_rows_alloc % capacity
    ):
        raise ValueError(
            f"ragged_rows_alloc {ragged_rows_alloc} must be a whole number "
            f"of both {ROWBLK}-row blocks and {capacity}-row tiles, because "
            "a tail tile reads a full window past the shard's last row"
        )
    tile_m = capacity  # the tile height IS the capacity
    tile_blocks = tile_m // ROWBLK
    # The mirror's unit is a tile, the run tables' unit is a block, and the
    # kernel converts between them with a shift.
    tile_block_shift = host.pow2_shift(tile_blocks, "tile_m // ROWBLK")
    tile_shift = host.pow2_shift(tile_m, "tile_m")
    # The scale mirror's lanes per sublane (host.mirror_lanes, as v2): the
    # tile, or a fixed unit below it, and then a tile's scales are unit_k
    # sublanes and every mirror offset counts units instead of tiles.
    mirror_m = host.mirror_lanes(tile_m, mirror_unit)
    unit_k = tile_m // mirror_m
    unit_blocks = mirror_m // ROWBLK
    mirror_block_shift = host.pow2_shift(unit_blocks, "mirror lanes // ROWBLK")
    mirror_shift = host.pow2_shift(mirror_m, "mirror lanes")
    height_rungs = (
        host.k32z_tile_height_rungs(tile_m, k32z_rungs)
        if k32z
        else tile_height_rungs(tile_m, height_rungs_override)
    )
    # K32Z visits whose first tile is taller than this wait both weight slabs
    # at their head (host.K32Z_STREAMED_ROWS for the default ladder, else the
    # ladder's tallest rung up to it: host.k32z_streamed_rows); a tile that
    # tall exists only where tile_m is.
    k32z_streamed = host.k32z_streamed_rows(height_rungs) if k32z else 0
    k32z_head_waits = k32z and tile_m > k32z_streamed
    # Rows one tile's mirror sublane is split over (host.mirror_split): 1 up
    # to 128-row tiles, whose mirrors are [sublanes, tile_m] as before.
    mirror_k = host.mirror_split(tile_m)
    # The contribution slab exists for one reason: to give the push an HBM
    # source. A tile is computed into `out_vm`, copied to `contrib_hbm`, then
    # read straight back out of it by the push that ships it -- two local
    # passes and, at the small buckets, about a quarter of the visit's DMA
    # enqueues, for a buffer nothing else reads. Where a build's experts are
    # expected to fit inside one tile -- the same bound the height ladder
    # above is emitted on -- the tile can be shipped from `out_vm` itself,
    # deleting the commits, the drain that gates them and the slab's traffic.
    # It costs a staging buffer that stays live until the wire has read it,
    # which is why a build whose visits carry several tiles keeps the slab:
    # there the wire is the busy resource and the round trip is what keeps
    # the tile loop off it.
    #
    # This subsumes the deferred commit drain the same bound used to select:
    # deferring a wait one visit is worth nothing once the wait is gone.
    # The bound is in tiles, so a K32Z build (256-row tiles) ships from
    # out_vm up to twice the rows a 128-row build does, as v2's does.
    push_from_vmem = not hbm_staging and ragged_rows_alloc <= ep * g_local * tile_m
    run_sems = int(run_sems)
    self_direct = bool(self_direct)
    x_dup_unroll = int(x_dup_unroll)
    if run_sems and (
        run_sems not in (2, 4, 8)
        or ep % run_sems
        or (ep // run_sems) & (ep // run_sems - 1)
    ):
        raise ValueError(
            f"run_sems is 2, 4 or 8, dividing ep={ep} into power-of-two "
            f"groups; got {run_sems}"
        )
    if x_dup_unroll not in (0, 2, 4, 8):
        raise ValueError(f"x_dup_unroll is 2, 4 or 8; got {x_dup_unroll}")
    if x_dup_unroll and not x_dedup:
        raise ValueError("x_dup_unroll unrolls x_dedup's duplicate gather")
    # run_sems: sources per group, as a shift (ep // run_sems is a power of two).
    rs_shift = (ep // run_sems).bit_length() - 1 if run_sems else 0
    # The deferred drain and push need the two-hop path.
    defer = defer_push and not push_from_vmem
    # early_ship: how many tiles after the tile that completes it a run ships
    # from its expert's tile loop (0: none, every run ships at push_expert).
    # Two-hop path only, as defer.
    es_lag = 2 if early_ship and not push_from_vmem else 0
    es_tag = f"_es{early_ship}" if es_lag else ""
    ls_window_rows = host.act_scale_window_rows(tile_m)
    # A segment word packs dest << 24 | local expert << 16 | rows.
    if push_tokens >= 1 << 16 or ep > 128 or g_local > 256:
        raise ValueError("push dispatch packs token/expert/dest into one i32")
    if not 1 <= sc_prepush <= g_local:
        raise ValueError(f"sc_prepush {sc_prepush} must be in [1, g_local={g_local}]")
    n_segs = ep * g_local
    # Whether the TensorCore pushes any visit position at all: with
    # sc_prepush == g_local the SparseCore delivered every one.
    tc_dispatch = sc_prepush < g_local
    n_prefetch = N_PREFETCH + N_DISPATCH_TABLES + (N_X_TABLES if x_dedup else 0)

    def kernel(*refs):
        it = iter(refs)
        # Block-unit transport tables, from shard_transport_tables_in_blocks.
        # Three of that builder's tables are the same array as another it
        # already ships, so each arrives once and the duplicates are derived
        # here instead of being prefetched again:
        #   push_src == contrib_off  -- a push reads its source at the offset
        #     the commit wrote it to;
        #   push_len == commit_len   -- `region_rows` IS `run_rows_aligned`
        #     reshaped, so slicing either at this shard gives one array;
        #   recv_row_off == push_dst << ROWBLK_SHIFT -- `recv_base` is a cumsum
        #     of ALIGNED run lengths, so the row offset is the block offset in
        #     rows. Asserted over eight shards and several routings by
        #     test/registered/tpu/kernels/moe/test_fused_ep_moe_v2_tables.py; that test
        #     is what keeps this derivation honest if recv_base ever changes to
        #     track true lengths.
        (commit_start_sm, commit_len_sm, contrib_off_sm, push_dst_sm) = (
            next(it) for _ in range(N_TRANSPORT_TABLES)
        )
        # Row-unit push tables, from shard_push_tables_in_rows.
        (true_rows_sm,) = (next(it) for _ in range(N_PUSH_TABLES))
        # (rows, base) i32 [G] row-unit tables from shard_expert_slabs.
        (expert_rows_sm, expert_base_sm) = (next(it) for _ in range(N_SLAB_TABLES))
        # visit_sm[visit_i] is the real expert at compacted step visit_i.
        visit_sm = next(it)
        # The five counts, each still spread over the expert-parallel axis.
        counts_sm = next(it)
        # This shard's expert-parallel index, as data rather than
        # `lax.axis_index` -- see N_RANK_TABLES.
        rank_sm = next(it)
        # Segment table, entry v * ep + d = destination d's v-th visited
        # expert: [dest << 24 | local expert << 16 | rows (S), source row in
        # lhs (S), destination slab row (S)], S = ep * g_local.
        dpairs_sm = next(it)
        # [unused (g_local + 1), receive rows by local expert (g_local),
        #  remote rows this TensorCore pushes (1), rows sent (1)].
        dmeta_sm = next(it)
        # x_dedup: [g_local, ep] landed rows | first SRC word << X_SHIFT per
        # (local expert, source) run, and SRC (owner slab row per duplicate).
        xtab_sm = next(it) if x_dedup else None
        xsrc_sm = next(it) if x_dedup else None

        def count(row):
            """Close one count's reduction on the scalar core.

            The host stopped a step early and left `ep` lanes to add. `ep`
            is a build-time constant, so this is a fixed run of scalar
            loads and adds: no loop, no dynamic bound, and nothing that
            reaches vector memory.
            """
            total = counts_sm[row, 0]
            for d in range(1, ep):
                total = total + counts_sm[row, d]
            return total

        # Consumed at the head and all through the visit loop, so it is
        # closed once here. The transport totals are closed in the drain
        # instead, where they are read, because that is the end of the
        # kernel and their arithmetic has everything before it to hide in.
        n_visit = count(host.COUNT_VISITS)
        # lhs_hbm is this shard's own rows sorted by global expert, the
        # source of every dispatch push. The wire never ships the topk
        # weight, so it is not an operand.
        lhs_hbm = next(it)
        # The activation row scale exists only where the rows were quantized.
        ls_hbm = next(it) if has_act_scale else None
        w1_hbm, w2_hbm = (next(it) for _ in range(2))
        # The weight scale tables exist only where the weights carry scales.
        w1s_hbm = next(it) if has_scales else None
        w2s_hbm = next(it) if has_scales else None
        # Optional per-expert bias tables, one row per local expert, in the
        # same [G, N] layout as the per-channel weight scales beside them.
        w1b_hbm = next(it) if has_w1_bias else None
        w2b_hbm = next(it) if has_w2_bias else None
        if packed4:
            # Packed-u32 weight stream: the four-bit [G, K, N] refs viewed as
            # [G, K//8, N] uint32, so the DMA moves half the bytes. This is
            # the ref-level view, and it is a PRIVATE jax method -- Pallas
            # exports pltpu.bitcast for the value-level widening below but
            # has no public equivalent for a ref. It carries no deprecation
            # entry, so if it moves it fails as a bare AttributeError from
            # inside a kernel body. Say what happened instead.
            if not hasattr(w1_hbm, "bitcast"):
                raise NotImplementedError(
                    f"the four-bit weight stream views a memory ref as "
                    f"packed 32-bit words through the private "
                    f"{type(w1_hbm).__name__}.bitcast, which the installed "
                    f"jax {jax.__version__} no longer carries; this kernel "
                    f"was built against jax {_REF_BITCAST_JAX_VERSION}. "
                    "There is no public Pallas equivalent for a ref-level "
                    "bitcast, so this needs a port rather than a rename."
                )
            w1_hbm = w1_hbm.bitcast(jnp.uint32)
            w2_hbm = w2_hbm.bitcast(jnp.uint32)
        next(it)  # slab_in, aliased to dispatch_hbm (output 4)
        recv_hbm = next(it)
        rscl_hbm = next(it)
        # A `push_from_vmem` build writes neither of these: they stay operands
        # for their row geometry, which the send waits count bytes against.
        contrib_hbm = next(it)
        cscl_hbm = next(it)
        # The expert slab [ragged_rows_alloc, lane blocks, 128] every shard
        # pushes its routed rows into; the SparseCore pre-push filled the
        # first sc_prepush visit positions before the kernel started.
        dispatch_hbm = next(it)
        # In vmem_scratch_arrays order, which is what the scratch list is
        # built from: staging, weights, scale tables where they exist, the
        # bias tables each build asked for, the activation row scale where
        # it exists, then the wire buffer and its scale mirror.
        (lhs_vm, w1_vm, w2_vm) = (next(it) for _ in range(3))
        w1s_vm = next(it) if has_scales else None
        w2s_vm = next(it) if has_scales else None
        # K32Z's decode buffers: the shift words of a chunk and the column
        # scales (host.vmem_scratch_arrays).
        tab_vm = next(it) if k32z else None
        col_vm = next(it) if k32z else None
        acc_vm = next(it) if k32z_ffn_roll else None
        w1b_vm = next(it) if has_w1_bias else None
        w2b_vm = next(it) if has_w2_bias else None
        ls_vm = next(it) if has_act_scale else None
        out_vm = next(it)
        oscl_vm = next(it)  # [parities, mirror_k, tile_m // mirror_k] f32

        def staged_scales(parity):
            """`parity`'s staged scales as one mirror sublane, the source
            every mirror transport takes (a [1, k, lanes] block once the
            mirrors are 3-D)."""
            return oscl_vm.at[parity] if mirror_k == 1 else oscl_vm.at[pl.ds(parity, 1)]

        def staged_unit(parity, j):
            """Unit j of `parity`'s staged scales, one [1, mirror_m] row
            (a mirror unit below the tile only)."""
            return oscl_vm.at[parity, j]

        lhs_sems, w1_sems, w2_sems = (next(it) for _ in range(3))
        w1s_sems = next(it) if has_scales else None
        w2s_sems = next(it) if has_scales else None
        cp_sem = next(it)
        dsems = next(it)  # [g_local] receive, one per local expert
        dsend_sem = next(it)  # remote dispatch sends
        # One send sem per out_vm parity: with a shared sem the other parity's
        # bytes could satisfy a wait, and order is not promised. Where the
        # tile ships itself, the parity's own sem is what says the wire has
        # finished reading the staging slot, so the same split does double
        # duty -- it guards reuse as well as the final drain.
        if push_from_vmem:
            # Own-destination rows ride the remote sems: for either copy the
            # semaphore fires once the SOURCE has been read, which is exactly
            # what the reuse guard needs, and only this parity signals it.
            send_sems, recv_sem = (next(it) for _ in range(2))
            # Scale-mirror DMAs need their own sems: waits are per-buffer.
            send_scl_sems, recv_scl_sem = (next(it) for _ in range(2))
            commit_sems = commit_scl_sems = None
            mehop_sem = mehop_scl_sem = None
        else:
            commit_sems, send_sem, recv_sem = (next(it) for _ in range(3))
            (commit_scl_sems, send_scl_sem, recv_scl_sem) = (next(it) for _ in range(3))
            # Never cp_sem: its start+wait users must not consume these.
            mehop_sem = next(it)
            mehop_scl_sem = next(it)

        me = rank_sm[0, 0]

        def landed_rows(e, d):
            """The rows run (e, d) lands here (x_dedup: its
            owned rows; else every pair's row)."""
            if x_dedup:
                return jnp.bitwise_and(xtab_sm[e, d], 0xFFFF)
            return true_rows_sm[e, d]

        def own_run(e, visit_i):
            """self_direct: (expert-relative first row, rows,
            lhs_hbm row) of this shard's own run of local expert e at visit
            position visit_i, read off the self segment the sender skipped
            (entry visit_i * ep + me: this shard's visit_i-th expert with
            itself as destination). None where no copy was skipped (switch
            off, no TensorCore dispatch, a static SparseCore position); a
            dynamic SparseCore position gets zero rows."""
            if not (self_direct and tc_dispatch):
                return None
            if isinstance(visit_i, int) and visit_i < sc_prepush:
                return None
            p = visit_i * ep + me
            n = jnp.where(visit_i >= sc_prepush, jnp.bitwise_and(dpairs_sm[p], 65535), 0)
            start = dpairs_sm[2 * n_segs + p] - expert_base_sm[e]
            return start, n, dpairs_sm[n_segs + p]

        def rs_sem(el):
            """run_sems: the receive semaphore this shard's rows
            for a destination's local expert `el` signal (its source group)."""
            return dsems.at[el * run_sems + jnp.right_shift(me, rs_shift)]

        def rs_wait(e, u, visit_i):
            """run_sems: wait every source group of expert e whose
            first run starts in the expert's tile u, for the rows the group
            lands (less the own rows under self_direct). A group with rows
            starts in exactly one of the expert's tiles, so each is waited
            once, and a run reaching tile u starts in it or earlier."""
            own = own_run(e, visit_i)
            per = ep // run_sems
            for g in range(run_sems):
                start = commit_start_sm[e, g * per]
                rows_g = landed_rows(e, g * per)
                for d in range(g * per + 1, (g + 1) * per):
                    start = jnp.minimum(start, commit_start_sm[e, d])
                    rows_g = rows_g + landed_rows(e, d)
                if own is not None:
                    rows_g = rows_g - jnp.where(
                        jnp.right_shift(me, rs_shift) == g, own[1], 0
                    )

                @pl.when(
                    jnp.logical_and(
                        jnp.right_shift(start, tile_block_shift) == u, rows_g > 0
                    )
                )
                def _(g=g, rows_g=rows_g):
                    _rows_wait(dsems.at[e * run_sems + g], dispatch_hbm, rows_g)

        def sd_stretch(row_base, lo, hi, a, b, slot, own):
            """self_direct: rows [a, b) of the tile [lo, hi)
            (expert-relative; the tile's first slab row is row_base) into
            lhs_vm[slot], the own run's part from lhs_hbm and the rest from
            the slab; three clipped pieces that partition [a, b), all on
            lhs_sems[slot]."""
            o_lo, o_n, o_src = own
            o_hi = o_lo + o_n

            def piece(x, y, from_lhs):
                x = jnp.maximum(x, lo)
                y = jnp.minimum(y, hi)

                @pl.when(y > x)
                def _():
                    if from_lhs:
                        src = lhs_hbm.at[pl.ds(o_src + (x - o_lo), y - x)]
                    else:
                        src = dispatch_hbm.at[pl.ds(row_base + (x - lo), y - x)]
                    pltpu.make_async_copy(
                        src,
                        lhs_vm.at[slot, pl.ds(x - lo, y - x)],
                        lhs_sems.at[slot],
                    ).start(priority=0)

            piece(a, jnp.minimum(b, o_lo), False)
            piece(jnp.maximum(a, o_lo), jnp.minimum(b, o_hi), True)
            piece(jnp.maximum(a, o_hi), b, False)

        def dup_loop(lo_r, hi_r, body):
            """x_dup_unroll=k: rows [lo_r, hi_r) of `body` (r, c)
            -> c as a loop of k-row chunks unrolled by hand, then the
            remainder: the same calls in the same order."""
            k = x_dup_unroll
            s = k.bit_length() - 1
            n_k = jnp.right_shift(jnp.maximum(hi_r - lo_r, 0), s)

            def chunk(i, c):
                r0 = lo_r + jnp.left_shift(i, s)
                for j in range(k):
                    c = body(r0 + j, c)
                return c

            lax.fori_loop(jnp.int32(0), n_k, chunk, jnp.int32(0))
            lax.fori_loop(lo_r + jnp.left_shift(n_k, s), hi_r, body, jnp.int32(0))

        def sync(src, dst):
            """Copy `src` into `dst` and wait for it before returning."""
            copy = pltpu.make_async_copy(src, dst, cp_sem)
            copy.start()
            copy.wait()

        def w1_copy(expert, slot):
            """The gate/up weight slab of `expert`, into weight slot `slot`."""
            return _slot_copy(w1_hbm.at[expert], w1_vm, w1_sems, slot)

        def w2_copy(expert, slot):
            """The down weight slab of `expert`, into weight slot `slot`."""
            if rhs_packed4 and trim_gate_up_padding:
                # C03's down contractions never read the padded K tail.
                # Start and wait use the same shortened descriptor and bytes.
                packed_active = effective_inter // PACK4
                return pltpu.make_async_copy(
                    w2_hbm.at[expert, pl.ds(0, packed_active), :],
                    w2_vm.at[slot, pl.ds(0, packed_active), :],
                    w2_sems.at[slot],
                )
            return _slot_copy(w2_hbm.at[expert], w2_vm, w2_sems, slot)

        def w1s_copy(expert, slot):
            assert w1s_hbm is not None
            return _slot_copy(w1s_hbm.at[expert], w1s_vm, w1s_sems, slot)

        def w2s_copy(expert, slot):
            assert w2s_hbm is not None
            return _slot_copy(w2s_hbm.at[expert], w2s_vm, w2s_sems, slot)

        def weight_starts(expert, slot):
            """Start every buffer slot `slot` holds for `expert`.

            In the order the body consumes them, which is also the order a
            single DMA queue will retire them -- WEIGHT_DMA_PRIORITY selects
            a queue, it does not reorder within one. The scale tables go
            first: they are half a megabyte and the first contraction block
            needs them, so they are all a streamed visit's head waits for.
            """
            if has_scales:
                w1s_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
                w2s_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
            w1_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
            w2_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)

        def weight_waits(expert, slot):
            """Wait for what the expert's first block needs.

            A streamed slab is not waited for here: its wait sits in the block
            that first reads it, where the wire has had longest to deliver, so
            this head waits for the scale tables alone. Every other format
            takes the slab in one matmul and keeps the whole-slab head wait.
            """
            if not streams_weights:
                w1_copy(expert, slot).wait()
                w2_copy(expert, slot).wait()
            if has_scales:
                w1s_copy(expert, slot).wait()
                w2s_copy(expert, slot).wait()

        def prologue():
            """Land the resident tables and start the first weight refills."""
            if has_w1_bias:
                sync(w1b_hbm, w1b_vm)
            if has_w2_bias:
                sync(w2b_hbm, w2b_vm)
            # Guarded on b < n_visit, matching the head waits. The range is
            # clamped to the visit-list length: at one local expert,
            # visit_sm[1] would be a statically out-of-bounds read before
            # the predicate could save it.
            for b in range(min(prefetch_distance, g_local)):

                @pl.when(jnp.int32(b) < n_visit)
                def _(b=b):
                    weight_starts(visit_sm[b], b)

        prologue()

        def dispatch_issue(lo_v, hi_v):
            """Push the (dest, expert) segments of visit positions [lo_v,
            hi_v); never blocks. Destinations are staggered by rank (d = me +
            1, me + 2, ...) like the SparseCore pre-push, so the shards do not
            all hit the same peer first."""

            def one(p, c):
                a = dpairs_sm[p]
                n = jnp.bitwise_and(a, 65535)

                @pl.when(n > 0)
                def _():
                    dest = jnp.right_shift(a, 24)
                    el = jnp.bitwise_and(jnp.right_shift(a, 16), 255)
                    src = lhs_hbm.at[pl.ds(dpairs_sm[n_segs + p], n)]
                    dst = dispatch_hbm.at[pl.ds(dpairs_sm[2 * n_segs + p], n)]

                    # Local destinations are a local HBM copy on the same
                    # receive semaphore.
                    if self_direct:
                        # self_direct: no self copy; the tile fetch reads the own
                        # rows from lhs_hbm (own_run, sd_stretch).
                        pass
                    else:

                        @pl.when(dest == me)
                        def _():
                            pltpu.make_async_copy(
                                src, dst, dsems.at[el] if not run_sems else rs_sem(el)
                            ).start(priority=0)

                    @pl.when(dest != me)
                    def _():
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=dsend_sem,
                            recv_sem=dsems.at[el] if not run_sems else rs_sem(el),
                            device_id=(dest,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start(priority=0)

                return c

            def one_v(v, c_v):
                base_p = v * ep

                def one_d(d, c_d):
                    return one(base_p + lax.rem(d + me + 1, ep), c_d)

                return lax.fori_loop(0, ep, one_d, c_v)

            lax.fori_loop(lo_v, hi_v, one_v, jnp.int32(0))

        def _start_next_gate_up(visit_i, slot, last_tile):
            """`() -> ()`: start the NEXT expert's gate/up slab, called from
            the reader of the block that retires this expert's.

            The reader is where the start belongs for the same reason the
            wait is: it is the only place that knows which block is being
            read. A later tile of this expert would still be reading the
            slab, so only the last tile starts the next one.
            """

            def start():
                @pl.when(jnp.logical_and(last_tile, visit_i + 1 < n_visit))
                def _():
                    w1_copy(visit_sm[visit_i + 1], slot).start(
                        priority=WEIGHT_DMA_PRIORITY
                    )

            return start

        def _slab_waiter(expert, slot, first_tile):
            """`(copy, block) -> ()`: land the slab that block `block` opens,
            on the expert's first tile.

            A streamed visit owes each slab exactly one wait, and block 0 --
            the first block of its matmul group -- is both where the wire has
            had the longest to deliver it and the only place in the group
            where no accumulation is in flight. Only the first tile owes
            anything: a later tile of the same expert reads a slab that is
            already whole, which is why the wait is predicated rather than
            issued per tile.
            """

            def wait(copy, block):
                if block:
                    return

                @pl.when(first_tile)
                def _():
                    copy(expert, slot).wait()

            return wait

        def _fp4_block_readers(slot, wait_slab=None, slab_retired=None):
            """(w1_block, w2_block) fp4 block readers at weight slot `slot`.

            The one genuinely fp4 place in the kernel: everything around it
            is four-bit generic, and these two lines are where the packed
            words are read as fp4 (e2m1) rather than as some other four-bit
            element type.

            `wait_slab` is where a streamed slab is landed and
            `slab_retired` fires at the last read of the gate/up one: the
            reader is the one place that knows which block is about to be
            contracted, so it is the one place either can sit without
            costing a block of runway.
            """
            # Each widens one k-block: u32 [qb/8, N] -> fp4 [qb, N] -> fp8.
            packed_rows = rhs_qb // PACK4

            def w1_block(b, n_start, n_end):
                # N.B. b is the contraction-block index.
                if wait_slab is not None and n_start == 0:
                    wait_slab(w1_copy, b)
                block = pltpu.bitcast(
                    w1_vm[
                        slot,
                        pl.ds(b * packed_rows, packed_rows),
                        pl.ds(n_start, n_end - n_start),
                    ],
                    FP4,
                ).astype(FP8)
                # slab_retired starts the next expert's gate/up DMA into the
                # same VMEM slot. That is only safe after the last read of
                # the current slab.
                if (
                    slab_retired is not None
                    and b == hidden // rhs_qb - 1
                    and n_end == w1_last_valid_column
                ):
                    slab_retired()
                return block

            def w2_block(b, n_start, n_end, *, block_rows=None):
                active_rows = rhs_qb if block_rows is None else block_rows
                if not 0 < active_rows <= rhs_qb or active_rows % PACK4:
                    raise ValueError("FP4 block rows must be positive and pack-aligned")
                if wait_slab is not None and n_start == 0:
                    wait_slab(w2_copy, b)
                return pltpu.bitcast(
                    w2_vm[
                        slot,
                        pl.ds(b * packed_rows, active_rows // PACK4),
                        pl.ds(n_start, n_end - n_start),
                    ],
                    FP4,
                ).astype(FP8)

            return w1_block, w2_block

        def _int8_chunk_readers(slot):
            """(w1_chunk, w2_chunk) integer chunk readers at slot `slot`.

            Each widens one contraction chunk to bf16, which is what the
            matrix unit takes: TPU v7 has no integer input
            type, so the widening is not an optimization but the only way
            an integer weight reaches a matmul at all.
            """

            def w1_chunk(b):
                return w1_vm[
                    slot, pl.ds(b * host.WIDEN_KCHUNK, host.WIDEN_KCHUNK), :
                ].astype(host.BF16)

            def w2_chunk(b):
                return w2_vm[
                    slot, pl.ds(b * host.WIDEN_KCHUNK, host.WIDEN_KCHUNK), :
                ].astype(host.BF16)

            return w1_chunk, w2_chunk

        # ---- ragged tile machinery ----
        # One-tile lookahead, double-buffered on the global tile parity:
        # a tile's stream rides lhs_sems[parity], waited once at its head.
        def x_issue_tile(row_base, live_blocks, slot, e, t, own=None):
            """x_dedup: tile t of local expert e, whose slab rows start at
            `row_base`. Without duplicates, the one contiguous copy. With
            them: per (expert, source) run, the stretch from the end of the
            previous run's duplicates to the end of this run's landed rows
            (the previous run's pad rows, then this run's owned rows), and
            one row copy per duplicate row from its owner's slab row (SRC);
            then the last run's pad. Clipped to the tile, they partition its
            rows, all on lhs_sems[slot]. (`own` is own_run's
            value under self_direct; every stretch then reads the own run's
            rows from lhs_hbm, sd_stretch.)"""
            assert xtab_sm is not None and xsrc_sm is not None
            n_rows = live_blocks * ROWBLK
            n_dup = jnp.int32(0)
            for d in range(ep):
                n_dup = n_dup + (
                    true_rows_sm[e, d] - jnp.bitwise_and(xtab_sm[e, d], 0xFFFF)
                )

            @pl.when(n_dup == 0)
            def _():
                if own is not None:
                    # self_direct: the tile in up to three pieces.
                    t_lo = t * tile_m
                    sd_stretch(
                        row_base, t_lo, t_lo + n_rows, t_lo, t_lo + n_rows, slot, own
                    )
                    return
                pltpu.make_async_copy(
                    dispatch_hbm.at[pl.ds(row_base, n_rows)],
                    lhs_vm.at[slot, pl.ds(0, n_rows)],
                    lhs_sems.at[slot],
                ).start(priority=0)

            @pl.when(n_dup > 0)
            def _():
                lo = t * tile_m  # the tile's first row, expert-relative
                hi = lo + n_rows

                def stretch(a, b):
                    if own is not None:
                        # self_direct (clips to the tile itself).
                        sd_stretch(row_base, lo, hi, a, b, slot, own)
                        return
                    a = jnp.maximum(a, lo)
                    b = jnp.minimum(b, hi)

                    @pl.when(b > a)
                    def _():
                        pltpu.make_async_copy(
                            dispatch_hbm.at[pl.ds(row_base + (a - lo), b - a)],
                            lhs_vm.at[slot, pl.ds(a - lo, b - a)],
                            lhs_sems.at[slot],
                        ).start(priority=0)

                prev = jnp.int32(0)
                for d in range(ep):
                    run = commit_start_sm[e, d] * ROWBLK
                    word = xtab_sm[e, d]
                    own_end = run + jnp.bitwise_and(word, 0xFFFF)
                    stretch(prev, own_end)
                    # Duplicate row r of the run reads SRC[first + r - own_end].
                    src0 = jnp.right_shift(word, X_SHIFT) - own_end
                    prev = run + true_rows_sm[e, d]

                    def dup_row(r, c, src0=src0):
                        pltpu.make_async_copy(
                            dispatch_hbm.at[pl.ds(xsrc_sm[src0 + r], 1)],
                            lhs_vm.at[slot, pl.ds(r - lo, 1)],
                            lhs_sems.at[slot],
                        ).start(priority=0)
                        return c

                    if x_dup_unroll:
                        # The same row copies, unrolled by hand.
                        dup_loop(
                            jnp.maximum(own_end, lo), jnp.minimum(prev, hi), dup_row
                        )
                    else:
                        lax.fori_loop(
                            jnp.maximum(own_end, lo),
                            jnp.minimum(prev, hi),
                            dup_row,
                            jnp.int32(0),
                        )
                stretch(prev, hi)

        def lhs_issue_tile(row_base, live_blocks, slot, e=None, t=None, visit_i=None):
            """Issue one tile's activations: the rows already sit in slab
            order, so one contiguous copy, plus its row-scale window.
            (x_dedup: tile t of expert e through x_issue_tile.
            self_direct: the own run's rows of visit position visit_i from
            lhs_hbm, sd_stretch.)"""
            own = None if visit_i is None else own_run(e, visit_i)
            if x_dedup:
                x_issue_tile(row_base, live_blocks, slot, e, t, own=own)
            elif own is not None:
                # self_direct: the tile in up to three pieces.
                t_lo = t * tile_m
                t_hi = t_lo + live_blocks * ROWBLK
                sd_stretch(row_base, t_lo, t_hi, t_lo, t_hi, slot, own)
            else:
                pltpu.make_async_copy(
                    dispatch_hbm.at[pl.ds(row_base, live_blocks * ROWBLK)],
                    lhs_vm.at[slot, pl.ds(0, live_blocks * ROWBLK)],
                    lhs_sems.at[slot],
                ).start(priority=0)
            if has_act_scale:
                # ls rides along on the same sem. The slab is the dense
                # lane-block view of one f32 per row, so a tile's window is
                # the sublanes its rows fall in, not a row range.
                assert ls_hbm is not None and ls_vm is not None
                pltpu.make_async_copy(
                    ls_hbm.at[pl.ds(row_base // HIDDEN_LANE_BLOCK, ls_window_rows)],
                    ls_vm.at[slot],
                    lhs_sems.at[slot],
                ).start(priority=0)

        def lhs_ready_tile(live_rows, slot):
            """Wait the token rows and the scale window of one tile."""
            # Sems count bytes, so the split wait is exact under any landing
            # order.
            pltpu.make_async_copy(
                lhs_vm.at[slot, pl.ds(0, live_rows)],
                lhs_vm.at[slot, pl.ds(0, live_rows)],
                lhs_sems.at[slot],
            ).wait()
            if has_act_scale:
                assert ls_vm is not None
                pltpu.make_async_copy(
                    ls_vm.at[slot], ls_vm.at[slot], lhs_sems.at[slot]
                ).wait()

        def _mirror_base(e):
            """Sublane expert e's tiles start at in the contribution mirror.

            The tile its slab rows start in, plus two sublanes of slack per
            expert. `host.contrib_mirror_rows` sizes the mirror for that
            slack, and two is enough because ceil(rows / tile_m) is at most
            (rows >> shift) + 1 while the next expert's base is at least
            that much further on.
            """
            return jnp.right_shift(expert_base_sm[e], mirror_shift) + 2 * e

        def _run_tiles(e, d, region_blocks):
            """A run's mirror sublanes: (source, destination, count).

            The destination follows the same rule on the run's ARRIVAL rows,
            and the run order there is the global expert order, so the same
            two sublanes of slack keep two runs off one sublane. Nonempty
            runs only -- the callers predicate on the length.
            """
            start = commit_start_sm[e, d]
            lo = jnp.right_shift(start, mirror_block_shift)
            hi = jnp.right_shift(start + region_blocks - 1, mirror_block_shift)
            dst = jnp.right_shift(push_dst_sm[e, d], mirror_block_shift) + 2 * (
                me * g_local + e
            )
            return _mirror_base(e) + lo, dst, hi - lo + 1

        def wait_commits(parity, live_blocks):
            """Wait one tile's commits on `parity`: data rows, then scales."""
            assert commit_sems is not None and commit_scl_sems is not None
            _rows_wait(commit_sems.at[parity], contrib_hbm, live_blocks * ROWBLK)
            # One tile committed one mirror sublane (unit_k units), whatever
            # its row count.
            _rows_wait(commit_scl_sems.at[parity], cscl_hbm, unit_k)

        def wait_shipped(parity, rows, subs):
            """Wait until the wire has read `parity`'s staging slot.

            `rows` and `subs` are what the tile that last held this parity
            shipped out of it -- rows of payload and mirror sublanes -- and
            nothing else signals these two semaphores, so the counts are
            exact. `contrib_hbm` and `cscl_hbm` are here only for their row
            geometry: a semaphore counts bytes, and those two carry the same
            bytes per row as the buffers actually shipped.
            """

            @pl.when(rows > 0)
            def _():
                _rows_wait(send_sems.at[parity], contrib_hbm, rows)

            @pl.when(subs > 0)
            def _():
                _rows_wait(send_scl_sems.at[parity], cscl_hbm, subs)

        def prime_expert(e, lhs_slot, visit_i):
            """Wait expert ``e``'s dispatched rows and issue its first tile.

            Visit positions below sc_prepush were delivered by the SparseCore
            before the kernel started, so they owe no wait; a static visit
            index (the first visit) skips the predicate altogether.
            """
            rows = expert_rows_sm[e]
            slab_base = expert_base_sm[e]
            if tc_dispatch and not (isinstance(visit_i, int) and visit_i < sc_prepush):
                # Every row of this expert, from every shard, has landed.
                @pl.when(visit_i >= sc_prepush)
                def _():
                    if run_sems:
                        # run_sems: the source groups tile 0 needs.
                        rs_wait(e, 0, visit_i)
                        return
                    n_in = dmeta_sm[g_local + 1 + e]
                    if self_direct:
                        # self_direct: the own rows were not copied.
                        own = own_run(e, visit_i)
                        if own is not None:
                            n_in = n_in - own[1]

                    @pl.when(n_in > 0)
                    def _():
                        _rows_wait(dsems.at[e], dispatch_hbm, n_in)

            lhs_issue_tile(
                slab_base,
                jnp.minimum(rows, tile_m) // ROWBLK,
                lhs_slot,
                e=e,
                t=0,
                visit_i=visit_i,
            )

        def expert_tiles(e, visit_i, carry):
            """One expert step: a fori_loop over its [tile_m, H] tiles."""
            # carry = (tile_count, <per-parity outstanding counts>): the
            # global tile counter and what each out_vm parity still owes --
            # committed blocks, or shipped rows and mirror sublanes where the
            # tile ships itself.
            # The weight slot indexes a contiguous counter so DISTANCE + 1 of
            # them occupy distinct slots; the DMA base stays the real expert.
            slot = lax.rem(visit_i, jnp.int32(nbuf))
            if not prefetches and not starts_next_weights:
                # No runway: nobody started this visit's weights, so it
                # starts them itself and waits on the two lines below.
                # Refilling the one slot cannot race the previous visit's
                # reads of it, because those reads feed accumulators whose
                # results are stored into out_vm inside the tile loop -- so
                # they have all retired by the loop's back edge, which is
                # before this.
                weight_starts(e, slot)
            if starts_next_weights:
                # Every later visit had its weights started by the one before,
                # which is the whole point; the first has nobody to start them
                # for it.

                @pl.when(visit_i == 0)
                def _():
                    weight_starts(e, slot)

            weight_waits(e, slot)

            # Refill the slot for the expert DISTANCE ahead: the previous
            # expert last read it, and DISTANCE < nbuf keeps it off both live
            # readers, so no wait is needed.
            if prefetches:

                @pl.when(visit_i + prefetch_distance < n_visit)
                def _():
                    refill_slot = lax.rem(visit_i + prefetch_distance, jnp.int32(nbuf))
                    ahead = visit_sm[visit_i + prefetch_distance]
                    # Weight refills take their own DMA priority to stay off
                    # the in-order token-gather queue.
                    weight_starts(ahead, refill_slot)

            rows = expert_rows_sm[e]
            if k32z_head_waits:
                # A K32Z visit whose first tile is taller than
                # K32Z_STREAMED_ROWS runs rung bodies with no reader waits:
                # both slabs land here, after the refill above is queued and
                # before the tile loop. Its tail tiles never wait again.
                @pl.when(rows > k32z_streamed)
                def _():
                    w1_copy(e, slot).wait()
                    w2_copy(e, slot).wait()

            slab_base = expert_base_sm[e]
            n_tiles = -(-rows // tile_m)

            def commit_scales(parity, t):
                """The tile's mirror sublane (its unit_k units) into the
                contribution mirror."""
                if unit_k == 1:
                    pltpu.make_async_copy(
                        staged_scales(parity),
                        cscl_hbm.at[pl.ds(_mirror_base(e) + t, 1)],
                        commit_scl_sems.at[parity],
                    ).start()
                    return
                for j in range(unit_k):
                    pltpu.make_async_copy(
                        staged_unit(parity, j),
                        cscl_hbm.at[pl.ds(_mirror_base(e) + unit_k * t + j, 1)],
                        commit_scl_sems.at[parity],
                    ).start()

            def commit_tile(parity, t, tile_block_base, live_blocks):
                """Commit the tile's intersection with each (expert, dest) run."""
                assert commit_sems is not None and commit_scl_sems is not None
                if hbm_staging:
                    # A staged build lays the contribution slab out like the
                    # expert slab, so the whole tile commits as one copy to
                    # its own slab rows and push_expert reads each run there.
                    pltpu.make_async_copy(
                        out_vm.at[parity, pl.ds(0, live_blocks * ROWBLK)],
                        contrib_hbm.at[
                            pl.ds(slab_base + t * tile_m, live_blocks * ROWBLK)
                        ],
                        commit_sems.at[parity],
                    ).start()
                    commit_scales(parity, t)
                    return
                # The runs tile the expert slab contiguously, so the
                # per-dest lengths sum to exactly live_blocks. Rows go per
                # run; the scales go once, below, because a tile's are one
                # sublane wherever its rows went.
                for d in range(ep):
                    run_start = commit_start_sm[e, d]
                    run_len = commit_len_sm[e, d]
                    lo = jnp.maximum(run_start, tile_block_base)
                    hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
                    overlap = jnp.maximum(hi - lo, 0)
                    # An empty intersection leaves lo unclamped, so the
                    # offsets below can point out of range; the copies are
                    # predicated out rather than issued at zero length. The
                    # commit waits count bytes, and a skipped copy
                    # contributes zero bytes either way.
                    lo = jnp.minimum(lo, hi)
                    src_block = lo - tile_block_base
                    dst_block = contrib_off_sm[e, d] + (lo - run_start)

                    @pl.when(overlap > 0)
                    def _(
                        src_block=src_block,
                        dst_block=dst_block,
                        overlap=overlap,
                        parity=parity,
                    ):
                        pltpu.make_async_copy(
                            out_vm.at[
                                parity, pl.ds(src_block * ROWBLK, overlap * ROWBLK)
                            ],
                            contrib_hbm.at[pl.ds(dst_block * ROWBLK, overlap * ROWBLK)],
                            commit_sems.at[parity],
                        ).start()

                # The tile's scales are one sublane wherever its rows
                # went, so they commit once, outside the per-dest loop.
                commit_scales(parity, t)

            def ship_tile(parity, t, tile_block_base, live_blocks):
                """Ship the tile straight out of `out_vm`, no slab in between.

                Returns what this tile put on the wire -- payload rows and
                mirror sublanes -- because that is what the next tile on this
                parity has to wait for before it may overwrite the slot.

                Every offset here is the one the two-hop path would have
                used: a commit wrote the tile's intersection with run (e, d)
                at `contrib_off + (lo - run_start)` and the push read the
                run's true rows back from `contrib_off`, so the tile's own
                share of that run starts `lo - run_start` blocks into it.
                """
                rows_out = jnp.int32(0)
                subs_out = jnp.int32(0)
                for d in range(ep):
                    run_start = commit_start_sm[e, d]
                    run_len = commit_len_sm[e, d]
                    lo = jnp.maximum(run_start, tile_block_base)
                    hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
                    overlap = jnp.maximum(hi - lo, 0)
                    # An empty intersection leaves lo unclamped, so the
                    # offsets below can point out of range; every copy is
                    # predicated out rather than issued at zero length.
                    lo = jnp.minimum(lo, hi)
                    into_run = (lo - run_start) * ROWBLK
                    # The run's true rows are its leading rows, so this tile
                    # carries whatever is left of them inside its overlap.
                    true_here = jnp.clip(
                        true_rows_sm[e, d] - into_run, 0, overlap * ROWBLK
                    )
                    is_me = jnp.int32(d) == me

                    @pl.when(overlap > 0)
                    def _(
                        lo=lo,
                        into_run=into_run,
                        overlap=overlap,
                        true_here=true_here,
                        is_me=is_me,
                        run_start=run_start,
                        parity=parity,
                        d=d,
                    ):
                        src_row = (lo - tile_block_base) * ROWBLK
                        dst_row = push_dst_sm[e, d] * ROWBLK + into_run
                        # One sublane per tile a run spans, at the destination
                        # base that run's own arrival rows give it.
                        dst_sub = (
                            jnp.right_shift(push_dst_sm[e, d], tile_block_shift)
                            + 2 * (me * g_local + e)
                            + t
                            - jnp.right_shift(run_start, tile_block_shift)
                        )

                        @pl.when(_and_nonempty(jnp.logical_not(is_me), true_here))
                        def _():
                            pltpu.make_async_remote_copy(
                                src_ref=out_vm.at[parity, pl.ds(src_row, true_here)],
                                dst_ref=recv_hbm.at[pl.ds(dst_row, true_here)],
                                send_sem=send_sems.at[parity],
                                recv_sem=recv_sem,
                                device_id=(jnp.int32(d),),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start()

                        # The own-destination run needs no fabric, and it moves
                        # the ALIGNED region the two-hop path moved: its trailing
                        # padding rows are never read, but leaving them out would
                        # change bytes the arrival buffer already holds.
                        @pl.when(is_me)
                        def _():
                            pltpu.make_async_copy(
                                out_vm.at[parity, pl.ds(src_row, overlap * ROWBLK)],
                                recv_hbm.at[pl.ds(dst_row, overlap * ROWBLK)],
                                send_sems.at[parity],
                            ).start()

                        if unit_k == 1:

                            @pl.when(jnp.logical_not(is_me))
                            def _():
                                pltpu.make_async_remote_copy(
                                    src_ref=staged_scales(parity),
                                    dst_ref=rscl_hbm.at[pl.ds(dst_sub, 1)],
                                    send_sem=send_scl_sems.at[parity],
                                    recv_sem=recv_scl_sem,
                                    device_id=(jnp.int32(d),),
                                    device_id_type=pl.DeviceIdType.MESH,
                                ).start()

                            @pl.when(is_me)
                            def _():
                                pltpu.make_async_copy(
                                    staged_scales(parity),
                                    rscl_hbm.at[pl.ds(dst_sub, 1)],
                                    send_scl_sems.at[parity],
                                ).start()

                    if unit_k > 1:
                        # A unit goes where the run's own rows put it: the
                        # units of this tile that hold rows of the run.
                        subs_here = jnp.int32(0)
                        run_unit0 = jnp.right_shift(run_start, mirror_block_shift)
                        for j in range(unit_k):
                            u0 = (unit_k * t + j) * unit_blocks
                            u_lo = jnp.maximum(run_start, u0)
                            u_hi = jnp.minimum(
                                jnp.minimum(run_start + run_len, u0 + unit_blocks),
                                tile_block_base + live_blocks,
                            )
                            u_has = u_hi > u_lo
                            dst_unit = (
                                jnp.right_shift(push_dst_sm[e, d], mirror_block_shift)
                                + 2 * (me * g_local + e)
                                + unit_k * t
                                + j
                                - run_unit0
                            )

                            @pl.when(jnp.logical_and(jnp.logical_not(is_me), u_has))
                            def _(dst_unit=dst_unit, parity=parity, d=d, j=j):
                                pltpu.make_async_remote_copy(
                                    src_ref=staged_unit(parity, j),
                                    dst_ref=rscl_hbm.at[pl.ds(dst_unit, 1)],
                                    send_sem=send_scl_sems.at[parity],
                                    recv_sem=recv_scl_sem,
                                    device_id=(jnp.int32(d),),
                                    device_id_type=pl.DeviceIdType.MESH,
                                ).start()

                            @pl.when(jnp.logical_and(is_me, u_has))
                            def _(dst_unit=dst_unit, parity=parity, j=j):
                                pltpu.make_async_copy(
                                    staged_unit(parity, j),
                                    rscl_hbm.at[pl.ds(dst_unit, 1)],
                                    send_scl_sems.at[parity],
                                ).start()

                            subs_here = subs_here + jnp.where(u_has, 1, 0)
                    sent = jnp.where(is_me, overlap * ROWBLK, true_here)
                    rows_out = rows_out + sent
                    if unit_k > 1:
                        subs_out = subs_out + subs_here
                    else:
                        subs_out = subs_out + jnp.where(overlap > 0, 1, 0)
                return rows_out, subs_out

            def tile_body(t, carried):
                """Compute tile t, stage it for the wire and ship it."""
                tile_count = carried[0]
                row_base = slab_base + t * tile_m
                live_rows = jnp.minimum(rows - t * tile_m, tile_m)
                live_blocks = live_rows // ROWBLK
                tile_block_base = t * tile_blocks  # expert-local block base
                parity = lax.rem(tile_count, jnp.int32(OUT_PARITIES))

                # out_vm[parity] reuse guard. Predicated on the outstanding
                # count rather than on the tile count, because the drain
                # zeroes them.
                if push_from_vmem:
                    wait_shipped(
                        parity,
                        jnp.where(parity == 0, carried[1], carried[2]),
                        jnp.where(parity == 0, carried[3], carried[4]),
                    )
                else:
                    pending = jnp.where(parity == 0, carried[1], carried[2])

                    @pl.when(pending > 0)
                    def _():
                        wait_commits(parity, pending)

                if es_lag:
                    # Early return: the guard above landed this expert's
                    # commits up to tile t - 2, so the part of every run that
                    # lies in tile t - 2 ships now. What is left of each run
                    # (its part in the visit's last two tiles) ships at
                    # push_expert.
                    @pl.when(t >= es_lag)
                    def _():
                        blk = (t - es_lag) * tile_blocks
                        for d in range(ep):
                            _push_piece(e, d, blk, blk + tile_blocks)

                # Wait this tile's stream, then issue tile t+1's into the other
                # slot before compute, so the fetch runs under the MXU window.
                lhs_ready_tile(live_rows, parity)
                other_parity = jnp.int32(1) - parity

                @pl.when(t + 1 < n_tiles)
                def _():
                    next_t = t + 1
                    next_rows = jnp.minimum(rows - next_t * tile_m, tile_m)
                    if run_sems and tc_dispatch:
                        # run_sems: the source groups whose first run starts
                        # in tile t + 1, before its fetch.
                        @pl.when(visit_i >= sc_prepush)
                        def _():
                            rs_wait(e, next_t, visit_i)

                    lhs_issue_tile(
                        slab_base + next_t * tile_m,
                        next_rows // ROWBLK,
                        other_parity,
                        e=e,
                        t=next_t,
                        visit_i=visit_i,
                    )

                # The alternate activation slot has retired its previous tile.
                # Fill the next expert's first tile before this final FFN so
                # its token rows and row scales travel under current compute.
                @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
                def _():
                    nxt = visit_sm[visit_i + 1]
                    prime_expert(nxt, other_parity, visit_i + 1)

                def compute_tile(height):
                    """Run the FFN on the tile's leading `height` rows.

                    Rows past the tile's live rows are never committed, so a
                    height above them is pure waste; `height` is the smallest
                    rung that still covers them.
                    """
                    full = height == tile_m
                    act_rows = (
                        lhs_vm[parity] if full else lhs_vm[parity, pl.ds(0, height)]
                    ).reshape(height, hidden)
                    act_scales = (
                        _tile_row_scales(cast(jax.Ref, ls_vm)[parity], row_base, height)
                        if has_act_scale
                        else None
                    )
                    w1b_row = (
                        cast(jax.Ref, w1b_vm)[pl.ds(e, 1), :] if has_w1_bias else None
                    )
                    if rhs_packed4:
                        # The block scales apply inside
                        # expert_ffn_blockscale, so the epilogues below must
                        # not apply w2s again.
                        assert w1s_vm is not None and w2s_vm is not None
                        if pad_scale_rows:
                            # The slot holds the padded table; the rows
                            # past the last block are padding.
                            w1s_blocks = w1s_vm[slot, pl.ds(0, hidden // rhs_qb), :]
                            w2s_blocks = w2s_vm[slot, pl.ds(0, inter // rhs_qb), :]
                        else:
                            w1s_blocks = w1s_vm[slot]  # [nb1, 2*inter] f32
                            w2s_blocks = w2s_vm[slot]  # [nb2, hidden] f32
                        first_tile, last_tile = t == 0, t == n_tiles - 1
                        w1_block, w2_block = _fp4_block_readers(
                            slot,
                            wait_slab=(
                                _slab_waiter(e, slot, first_tile)
                                if streams_weights
                                else None
                            ),
                            slab_retired=(
                                _start_next_gate_up(visit_i, slot, last_tile)
                                if starts_next_weights
                                else None
                            ),
                        )
                        acc2, mid_scale = expert_ffn_blockscale(
                            act_rows,
                            act_scales,
                            w1_block,
                            w2_block,
                            w1s_blocks,
                            w2s_blocks,
                            qb=rhs_qb,
                            act_fn=act_fn,
                            w1b=w1b_row,
                            logical_intermediate_size=(
                                effective_inter if trim_gate_up_padding else None
                            ),
                        )
                    elif k32z:
                        assert w1s_vm is not None and w2s_vm is not None
                        assert tab_vm is not None and col_vm is not None
                        first_tile, last_tile = t == 0, t == n_tiles - 1
                        # Bodies up to K32Z_STREAMED_ROWS wait each slab in
                        # the block that first reads it, on the expert's first
                        # tile only, so a tail tile never waits; a taller
                        # body's visit waited both slabs at its head.
                        streamed = height <= k32z_streamed
                        waiter = (
                            _slab_waiter(e, slot, first_tile)
                            if streamed
                            else None
                        )
                        # Every tile rebuilds the column scales: a few dozen
                        # row stores, read by this tile alone.
                        k32z_fill_colscale(
                            w2s_vm, slot, col_vm, hidden=hidden, inter=inter
                        )
                        wait_slab = (
                            None
                            if waiter is None
                            else lambda which: waiter(
                                w1_copy if which == 1 else w2_copy, 0
                            )
                        )
                        slab_retired = (
                            _start_next_gate_up(visit_i, slot, last_tile)
                            if starts_next_weights
                            else None
                        )
                        if k32z_ffn_roll and height >= k32z_roll_from:
                            # The looped body (host.k32z_roll_plan): the same
                            # bits from a fraction of the decode text.
                            decode1, decode2 = k32z_chunk_decoders(
                                w1_vm,
                                w2_vm,
                                w1s_vm,
                                w2s_vm,
                                tab_vm,
                                slot,
                                hidden=hidden,
                                inter=inter,
                                gate_up_bits=k32z_gate_up_bits,
                            )
                            acc2, mid_scale = expert_ffn_colscale_rolled(
                                act_rows,
                                act_scales,
                                decode1,
                                decode2,
                                col_vm[pl.ds(0, 1), pl.ds(0, 2 * inter)],
                                acc_vm,
                                hidden=hidden,
                                act_fn=act_fn,
                                w1b=w1b_row,
                                unroll=k32z_ffn_roll,
                                wait_slab=wait_slab,
                                slab_retired=slab_retired,
                            )
                        else:
                            w1_block, w2_block = k32z_block_readers(
                                w1_vm,
                                w2_vm,
                                w1s_vm,
                                w2s_vm,
                                tab_vm,
                                slot,
                                hidden=hidden,
                                inter=inter,
                                gate_up_bits=k32z_gate_up_bits,
                                wait_slab=wait_slab,
                                slab_retired=slab_retired,
                            )
                            acc2, mid_scale = expert_ffn_colscale(
                                act_rows,
                                act_scales,
                                w1_block,
                                w2_block,
                                col_vm[pl.ds(0, 1), pl.ds(0, 2 * inter)],
                                hidden=hidden,
                                act_fn=act_fn,
                                w1b=w1b_row,
                            )
                    elif weight_format == host.WeightFormat.INT8:
                        assert w1s_vm is not None
                        w1_chunk, w2_chunk = _int8_chunk_readers(slot)
                        acc2, mid_scale = expert_ffn_int8(
                            act_rows,
                            w1_chunk,
                            w2_chunk,
                            w1s_vm[pl.ds(slot, 1), :],
                            n_chunks1=hidden // host.WIDEN_KCHUNK,
                            n_chunks2=inter // host.WIDEN_KCHUNK,
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
                    elif weight_format == host.WeightFormat.BF16:
                        acc2, mid_scale = expert_ffn_bf16(
                            act_rows,
                            w1_vm[slot],
                            w2_vm[slot],
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
                    else:
                        assert w1s_vm is not None
                        acc2, mid_scale = expert_ffn_fp8(
                            act_rows,
                            act_scales,
                            w1_vm[slot],
                            w2_vm[slot],
                            w1s_vm[pl.ds(slot, 1), :],
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
                    # The destination applies the router weight, not this.
                    # Four-bit weights carry w2s inside the block sums; the
                    # formats whose second matmul took bf16 rows return no row
                    # scale, so the only thing left to apply is the
                    # per-channel weight scale where the format has one. The
                    # epilogue itself is a module-level function, so the
                    # placement of each of these -- and the down bias's
                    # add-once property in particular -- has a host-side
                    # witness rather than only a device-marked one.
                    if k32z:
                        # The down matmul's column scales, as a per-channel
                        # format's w2s.
                        w2s_row = cast(jax.Ref, col_vm)[pl.ds(1, 1), pl.ds(0, hidden)]
                    elif has_scales and not rhs_packed4:
                        w2s_row = cast(jax.Ref, w2s_vm)[pl.ds(slot, 1), :]
                    else:
                        w2s_row = None
                    wire_rows, wire_scales = wire_row_and_scale(
                        acc2,
                        mid_scale,
                        w2s=w2s_row,
                        w2b=(
                            cast(jax.Ref, w2b_vm)[pl.ds(e, 1), :]
                            if has_w2_bias
                            else None
                        ),
                        wire_dtype=form.wire_dtype,
                        tile_m=height,
                    )
                    wire_rows_staged = wire_rows.reshape(
                        height, lane_blocks, HIDDEN_LANE_BLOCK
                    )
                    wire_scales_staged = wire_scales.reshape(1, height)

                    def store_parity(static_parity):
                        """Stage the tile's rows and scales in one out_vm
                        slot."""
                        if full:
                            out_vm[static_parity] = wire_rows_staged
                        else:
                            out_vm[static_parity, pl.ds(0, height)] = wire_rows_staged
                        if unit_k > 1:
                            # One [1, mirror_m] row per unit the tile fills.
                            for j in range(-(-height // mirror_m)):
                                width = min(mirror_m, height - j * mirror_m)
                                oscl_vm[static_parity, j, :, pl.ds(0, width)] = (
                                    wire_scales_staged[
                                        :, j * mirror_m : j * mirror_m + width
                                    ]
                                )
                        elif mirror_k == 1 and full:
                            oscl_vm[static_parity] = wire_scales_staged
                        elif mirror_k == 1:
                            oscl_vm[static_parity, :, pl.ds(0, height)] = (
                                wire_scales_staged
                            )
                        else:
                            # The sublane split over mirror_k rows of lanes.
                            lanes = tile_m // mirror_k
                            for j in range(-(-height // lanes)):
                                width = min(lanes, height - j * lanes)
                                oscl_vm[static_parity, pl.ds(j, 1), pl.ds(0, width)] = (
                                    wire_scales_staged[:, j * lanes : j * lanes + width]
                                )

                    # Store slots must be static, so these branches stay
                    # static.
                    @pl.when(parity == 0)
                    def _():
                        store_parity(0)

                    @pl.when(parity == 1)
                    def _():
                        store_parity(1)

                # Exactly one rung runs: live_rows is a whole number of
                # ROWBLK in [ROWBLK, tile_m], and the rungs partition that.
                if len(height_rungs) == 1:
                    compute_tile(tile_m)
                else:
                    for rung_i, rung in enumerate(height_rungs):
                        below = height_rungs[rung_i - 1] if rung_i else 0

                        @pl.when(jnp.logical_and(live_rows > below, live_rows <= rung))
                        def _(rung=rung):
                            compute_tile(rung)

                if push_from_vmem:
                    rows_out, subs_out = ship_tile(
                        parity, t, tile_block_base, live_blocks
                    )
                    return (
                        tile_count + 1,
                        jnp.where(parity == 0, rows_out, carried[1]),
                        jnp.where(parity == 1, rows_out, carried[2]),
                        jnp.where(parity == 0, subs_out, carried[3]),
                        jnp.where(parity == 1, subs_out, carried[4]),
                    )

                commit_tile(parity, t, tile_block_base, live_blocks)

                if defer:
                    # Deferred push: the previous expert's drain and push, one
                    # tile late. Its last tile is the one before this, so on
                    # the other parity; its earlier commits were taken by the
                    # reuse guards (this tile's included), so this one wait
                    # completes its runs and the push may read them.
                    other_pending = jnp.where(parity == 0, carried[2], carried[1])
                    deferred = jnp.logical_and(t == 0, visit_i > 0)

                    @pl.when(deferred)
                    def _():
                        @pl.when(other_pending > 0)
                        def _():
                            wait_commits(other_parity, other_pending)

                        push_expert(visit_sm[visit_i - 1])

                    other_pending = jnp.where(deferred, 0, other_pending)
                    return (
                        tile_count + 1,
                        jnp.where(parity == 0, live_blocks, other_pending),
                        jnp.where(parity == 1, live_blocks, other_pending),
                    )

                return (
                    tile_count + 1,
                    jnp.where(parity == 0, live_blocks, carried[1]),
                    jnp.where(parity == 1, live_blocks, carried[2]),
                )

            carry = lax.fori_loop(0, n_tiles, tile_body, carry)

            if starts_next_weights:
                # What the tile loop was still reading to its last block: the
                # down slab, and the scale tables, which are read a block at a
                # time beside the weights they scale.

                @pl.when(visit_i + 1 < n_visit)
                def _():
                    nxt = visit_sm[visit_i + 1]
                    # Scales first: they are all the next visit's head waits
                    # for, and the down slab behind them is a megabyte-scale
                    # queue they would otherwise sit out.
                    if has_scales:
                        w1s_copy(nxt, slot).start(priority=WEIGHT_DMA_PRIORITY)
                        w2s_copy(nxt, slot).start(priority=WEIGHT_DMA_PRIORITY)
                    w2_copy(nxt, slot).start(priority=WEIGHT_DMA_PRIORITY)

            return carry

        def drain_commits(carried):
            """Drain both parities' pending commits, then zero the counts."""
            # The tile reuse guards test pending > 0, so nothing double-waits.
            tile_count, pending0, pending1 = carried

            @pl.when(pending0 > 0)
            def _():
                wait_commits(0, pending0)

            @pl.when(pending1 > 0)
            def _():
                wait_commits(1, pending1)

            return (tile_count, jnp.int32(0), jnp.int32(0))

        def drain_shipped(carried):
            """Wait out whatever the last tile on each parity put on the wire.

            Every earlier tile was already waited by the reuse guard of the
            tile two later on its parity, so these are the only sends left
            and their counts are the whole of the send semaphores' credit.
            """
            for p in range(host.OUT_PARITIES):
                wait_shipped(p, carried[1 + p], carried[3 + p])

        # Visit 0 is a SparseCore-delivered position, so its first tile is
        # issued before the barrier; the TensorCore's own first pushes go out
        # right after it, once every peer is inside the kernel.
        @pl.when(n_visit > 0)
        def _():
            prime_expert(visit_sm[0], jnp.int32(0), 0)

        if single_barrier:
            _entry_barrier_first_phase(ep)
        else:
            _all_pairs_barrier(ep)
        if tc_dispatch:
            dispatch_issue(sc_prepush, min(sc_prepush + dispatch_ahead, g_local))

        def push_expert(e):
            """Push expert e's remote regions and hop its own-dest region:
            every run, or, with the early return, what the tile loop has not
            already shipped (the runs' parts in the visit's last es_lag tiles)."""
            if not es_lag:
                for d in range(ep):
                    _push_run(e, d)
                return
            n_tiles_e = jnp.right_shift(expert_rows_sm[e] + (tile_m - 1), tile_shift)
            # What is left of each run from the visit's last es_lag tiles on
            # (the whole run in a shorter visit).
            blk = (n_tiles_e - es_lag) * tile_blocks
            for d in range(ep):
                _push_piece(e, d, blk, None)

        def _push_piece(e, d, blk_lo, blk_hi):
            """Push the part of run (e, d) in expert-local blocks
            [blk_lo, blk_hi) (blk_hi None: to the run's end), or hop it where
            d is this shard. The pieces of a run partition its blocks at tile
            boundaries, so its true rows and its mirror units (whole units per
            tile) go out once each, to the places _push_run sends them."""
            run_start = commit_start_sm[e, d]
            run_end = run_start + commit_len_sm[e, d]
            lo = jnp.maximum(run_start, blk_lo)
            hi = run_end if blk_hi is None else jnp.minimum(run_end, blk_hi)
            n_blk = hi - lo
            # Rows into the run this piece starts at.
            into = (lo - run_start) * ROWBLK
            src_row = (
                expert_base_sm[e] + commit_start_sm[e, d] * ROWBLK
                if hbm_staging
                else contrib_off_sm[e, d] * ROWBLK
            )
            dst_row = push_dst_sm[e, d] * ROWBLK
            is_me = jnp.int32(d) == me
            # The piece's mirror units, placed as the run's are (_run_tiles).
            u_lo = jnp.right_shift(lo, mirror_block_shift)
            n_sub = jnp.right_shift(hi - 1, mirror_block_shift) - u_lo + 1
            src_sub = _mirror_base(e) + u_lo
            dst_sub = (
                jnp.right_shift(push_dst_sm[e, d], mirror_block_shift)
                + 2 * (me * g_local + e)
                + u_lo
                - jnp.right_shift(run_start, mirror_block_shift)
            )

            @pl.when(jnp.logical_and(jnp.logical_not(is_me), n_blk > 0))
            def _():
                # The run's true rows are its leading rows.
                n_true = jnp.clip(true_rows_sm[e, d] - into, 0, n_blk * ROWBLK)

                @pl.when(n_true > 0)
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=contrib_hbm.at[pl.ds(src_row + into, n_true)],
                        dst_ref=recv_hbm.at[pl.ds(dst_row + into, n_true)],
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=(jnp.int32(d),),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                pltpu.make_async_remote_copy(
                    src_ref=cscl_hbm.at[pl.ds(src_sub, n_sub)],
                    dst_ref=rscl_hbm.at[pl.ds(dst_sub, n_sub)],
                    send_sem=send_scl_sem,
                    recv_sem=recv_scl_sem,
                    device_id=(jnp.int32(d),),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            @pl.when(jnp.logical_and(is_me, n_blk > 0))
            def _():
                pltpu.make_async_copy(
                    contrib_hbm.at[pl.ds(src_row + into, n_blk * ROWBLK)],
                    recv_hbm.at[pl.ds(dst_row + into, n_blk * ROWBLK)],
                    mehop_sem,
                ).start()
                pltpu.make_async_copy(
                    cscl_hbm.at[pl.ds(src_sub, n_sub)],
                    rscl_hbm.at[pl.ds(dst_sub, n_sub)],
                    mehop_scl_sem,
                ).start()

        def _push_run(e, d):
            """Push expert e's run to dest d, or hop it where d is this shard."""
            # Remote pushes go per (expert, dest) at true length; run starts
            # stay aligned, so recv positions are unchanged. Each is predicated
            # on its own length: a zero-length REMOTE DMA must never issue.
            region_blocks = commit_len_sm[e, d]
            # A staged build's contribution slab is shaped like the expert
            # slab (see commit_tile); otherwise it is packed per dest.
            src_row = (
                expert_base_sm[e] + commit_start_sm[e, d] * ROWBLK
                if hbm_staging
                else contrib_off_sm[e, d] * ROWBLK
            )
            dst_block = push_dst_sm[e, d]
            is_me = jnp.int32(d) == me

            @pl.when(_and_nonempty(jnp.logical_not(is_me), region_blocks))
            def _():
                # One expert per push, so the region IS expert e's run.
                true_rows = true_rows_sm[e, d]

                # A nonempty region can hold an empty (e, d) run.
                @pl.when(true_rows > 0)
                def _():
                    """The rows themselves, at their true length."""
                    pltpu.make_async_remote_copy(
                        src_ref=contrib_hbm.at[pl.ds(src_row, true_rows)],
                        dst_ref=recv_hbm.at[
                            pl.ds(push_dst_sm[e, d] * ROWBLK, true_rows)
                        ],
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=(jnp.int32(d),),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                # Scale mirror: the source tiles this run's rows
                # fall in, landing at the base the destination's own
                # plan gave the run.
                src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
                pltpu.make_async_remote_copy(
                    src_ref=cscl_hbm.at[pl.ds(src_sub, n_sub)],
                    dst_ref=rscl_hbm.at[pl.ds(dst_sub, n_sub)],
                    send_sem=send_scl_sem,
                    recv_sem=recv_scl_sem,
                    device_id=(jnp.int32(d),),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            # Safe to defer: recv is read only after the drain.
            @pl.when(_and_nonempty(is_me, region_blocks))
            def _():
                pltpu.make_async_copy(
                    contrib_hbm.at[pl.ds(src_row, region_blocks * ROWBLK)],
                    recv_hbm.at[pl.ds(dst_block * ROWBLK, region_blocks * ROWBLK)],
                    mehop_sem,
                ).start()
                src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
                pltpu.make_async_copy(
                    cscl_hbm.at[pl.ds(src_sub, n_sub)],
                    rscl_hbm.at[pl.ds(dst_sub, n_sub)],
                    mehop_scl_sem,
                ).start()

        def visit_step(visit_i, carry):
            """One step of the visit list: compute, then drain and push."""
            e = visit_sm[visit_i]
            if tc_dispatch:
                # Push the position sc_prepush + dispatch_ahead visits ahead,
                # so a peer's rows for its next experts are in flight while it
                # computes.
                ahead = visit_i + sc_prepush + dispatch_ahead

                @pl.when(ahead < g_local)
                def _():
                    dispatch_issue(ahead, ahead + 1)

            carry = expert_tiles(e, visit_i, carry)
            if push_from_vmem:
                # Each tile shipped itself; there is nothing left to flush.
                return carry
            if defer:
                # The next visit's first tile drains and pushes this expert
                # (tile_body); the last visit's go after the loop.
                return carry
            if not hbm_staging:
                carry = drain_commits(carry)
                push_expert(e)
                return carry
            # A staged build pushes the PREVIOUS visit's runs here, so this
            # visit's last commits land while the next visit computes. Of the
            # previous visit, only its last tile can still owe commits: it
            # sits on the parity this visit did not end on, which this
            # visit's tiles reused (and so waited) unless it ran one tile.
            other = jnp.int32(1) - lax.rem(carry[0] - 1, jnp.int32(OUT_PARITIES))
            for q in range(OUT_PARITIES):
                pend_q = carry[1 + q]

                @pl.when(jnp.logical_and(other == q, pend_q > 0))
                def _(q=q, pend_q=pend_q):
                    wait_commits(q, pend_q)

            carry = (
                carry[0],
                jnp.where(other == 0, 0, carry[1]),
                jnp.where(other == 1, 0, carry[2]),
            )

            @pl.when(visit_i > 0)
            def _():
                push_expert(visit_sm[visit_i - 1])

            return carry

        carry0 = tuple(jnp.int32(0) for _ in range(5 if push_from_vmem else 3))
        carry_end = lax.fori_loop(0, n_visit, visit_step, carry0)
        if hbm_staging or defer:
            # The last visit's runs, deferred past the loop.
            carry_end = drain_commits(carry_end)

            @pl.when(n_visit > 0)
            def _():
                push_expert(visit_sm[n_visit - 1])

        def drain_transport():
            """Consume the pushes and the commits the per-tile waits left."""
            if push_from_vmem:
                # Sends are per staging parity and the reuse guards already
                # took all but the last two; arrivals are still a total,
                # because a peer signals them whatever parity it sent from.
                drain_shipped(carry_end)
                _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
                _rows_wait(recv_scl_sem, rscl_hbm, count(host.COUNT_RECV_MIRROR))
                return
            # The per-tile head waits consume every commit but the last
            # one per parity; the drains below consume those.
            _rows_wait(send_sem, contrib_hbm, count(host.COUNT_SEND_ROWS))
            _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
            # The mirror moves whole sublanes, one per tile a run spans,
            # so its counts are their own rows of the count table rather
            # than a division of the row counts.
            _rows_wait(send_scl_sem, cscl_hbm, count(host.COUNT_SEND_MIRROR))
            _rows_wait(recv_scl_sem, rscl_hbm, count(host.COUNT_RECV_MIRROR))
            # Deferred own-destination drain: the total sums my own-dest
            # region lengths.
            self_blocks = commit_len_sm[0, me]
            for g in range(1, g_local):
                self_blocks = self_blocks + commit_len_sm[g, me]
            _rows_wait(mehop_sem, recv_hbm, self_blocks * ROWBLK)
            _rows_wait(mehop_scl_sem, rscl_hbm, count(host.COUNT_SELF_MIRROR))

        if tc_dispatch:
            # Whatever visit positions the loop did not reach -- this shard
            # may visit fewer experts than its peers -- then the sends.
            dispatch_issue(
                jnp.minimum(n_visit + sc_prepush + dispatch_ahead, g_local), g_local
            )
            n_remote = dmeta_sm[2 * g_local + 1]

            @pl.when(n_remote > 0)
            def _():
                _rows_wait(dsend_sem, dispatch_hbm, n_remote)

        drain_transport()
        if tail_barrier:
            _all_pairs_barrier(ep)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    # The VMEM buffers come from the same list the VMEM accounting sums, in
    # the order the kernel body unpacks them: lhs_vm, w1_vm, w2_vm, the scale
    # tables where the format has them, the bias tables each build asked for,
    # ls_vm where the activations are quantized, out_vm, oscl_vm.
    scratch = (
        [
            pltpu.VMEM(shape, dtype)
            for _, shape, dtype in host.vmem_scratch_arrays(
                g_local,
                capacity,
                hidden,
                inter,
                nbuf=nbuf,
                weight_format=weight_format,
                rhs_qb=rhs_qb,
                has_w1_bias=has_w1_bias,
                has_w2_bias=has_w2_bias,
                pad_scale_rows=pad_scale_rows,
                k32z_gate_up_bits=k32z_gate_up_bits,
                k32z_ffn_roll=k32z_ffn_roll,
                mirror_unit=mirror_unit,
            )
        ]
        + [
            # lhs_sems. Parity-deep, not nbuf-deep: every index into it, and into
            # the lhs_vm and ls_vm buffers it guards, is an out-parity.
            pltpu.SemaphoreType.DMA((OUT_PARITIES,)),
            pltpu.SemaphoreType.DMA((nbuf,)),  # w1_sems
            pltpu.SemaphoreType.DMA((nbuf,)),  # w2_sems
        ]
        + (
            [
                # The scale tables travel with the slabs they scale, so they are
                # slot-deep and need their own sems: a wait is per-buffer.
                pltpu.SemaphoreType.DMA((nbuf,)),  # w1s_sems
                pltpu.SemaphoreType.DMA((nbuf,)),  # w2s_sems
            ]
            if has_scales
            else []
        )
        + [
            pltpu.SemaphoreType.DMA,  # cp_sem
            # dsems (run_sems=k: k per local expert, flat e * k + group)
            pltpu.SemaphoreType.DMA((g_local * run_sems if run_sems else g_local,)),
            pltpu.SemaphoreType.DMA,  # dsend_sem
        ]
        + (
            [
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # send_sems
                pltpu.SemaphoreType.DMA,  # recv_sem
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # send_scl_sems
                pltpu.SemaphoreType.DMA,  # recv_scl_sem
            ]
            if push_from_vmem
            else [
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # commit_sems
                pltpu.SemaphoreType.DMA,  # send_sem
                pltpu.SemaphoreType.DMA,  # recv_sem
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # commit_scl_sems
                pltpu.SemaphoreType.DMA,  # send_scl_sem
                pltpu.SemaphoreType.DMA,  # recv_scl_sem
                pltpu.SemaphoreType.DMA,  # mehop_sem
                pltpu.SemaphoreType.DMA,  # mehop_scl_sem
            ]
        )
    )
    # The sorted rows, the two weight slabs and the dispatch slab are always
    # operands; the activation row scale, the two weight scale tables and the
    # two bias tables are there only where the build has them.
    n_inputs = (
        2
        + N_WEIGHT_SLABS
        + int(has_act_scale)
        + N_SCALE_TABLES * int(has_scales)
        + int(has_w1_bias)
        + int(has_w2_bias)
    )

    def make_call(recv_rows):
        """The pallas_call for this build, at this arrival-buffer height."""
        # A caller value, so it refuses rather than asserts: under `python
        # -O` a non-multiple truncates in the two divisions below and the
        # arrival buffer comes out short by up to ROWBLK - 1 rows, while the
        # combine still gathers at positions reaching recv_rows - 1.
        if recv_rows % ROWBLK:
            raise ValueError(
                f"the arrival buffer height {recv_rows} is not a whole "
                f"number of {ROWBLK}-row blocks, which is the unit every "
                "transport into it moves"
            )
        # contrib and cscl are sized for the no-drop worst case.
        contrib_rows = ragged_rows_alloc
        wire = form.wire_dtype
        out_shape = [
            jax.ShapeDtypeStruct((recv_rows, lane_blocks, HIDDEN_LANE_BLOCK), wire),
            jax.ShapeDtypeStruct(
                host.mirror_shape(
                    host.arrival_mirror_rows(recv_rows, ep * g_local, mirror_m),
                    mirror_m,
                ),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct((contrib_rows, lane_blocks, HIDDEN_LANE_BLOCK), wire),
            jax.ShapeDtypeStruct(
                host.mirror_shape(
                    host.contrib_mirror_rows(contrib_rows, g_local, mirror_m),
                    mirror_m,
                ),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (ragged_rows_alloc, lane_blocks, HIDDEN_LANE_BLOCK), form.act_dtype
            ),
        ]
        # Each suffix appears only when its feature is on, so a kernel that
        # fuses silu on eight-bit float weights with no biases keeps the name
        # it had before any of those operands existed, and every fingerprint
        # taken against it still matches.
        format_tag = {
            host.WeightFormat.FP8: "",
            host.WeightFormat.FP4: "_fp4w",
            host.WeightFormat.INT8: "_int8w",
            host.WeightFormat.BF16: "_bf16w",
            host.WeightFormat.K32Z: (
                "_k32zw"
                if k32z_gate_up_bits == host.K32Z_GATE_UP_BITS[0]
                else f"_k32zw_gu{k32z_gate_up_bits}"
            ),
        }[weight_format]
        bias_tag = ("_w13bias" if has_w1_bias else "") + (
            "_w2bias" if has_w2_bias else ""
        )
        name = (
            f"push_fused_moe"
            f"{format_tag}"
            f"{'' if act_fn == 'silu' else '_' + act_fn}"
            f"{bias_tag}"
            f"_g{g_local}_c{capacity}_nb{nbuf}_s{sc_prepush}"
            f"{'' if tail_barrier else '_notail'}"
            f"{'_hbmst' if hbm_staging else ''}"
            f"{'' if dispatch_ahead == 1 else f'_da{dispatch_ahead}'}"
            f"{'_1b' if single_barrier else ''}"
            f"{f'_r{host.k32z_roll_tag(k32z_ffn_roll)}' if k32z_ffn_roll else ''}"
            f"{f'f{k32z_roll_from}' if k32z_ffn_roll and k32z_roll_from else ''}"
            f"{f'_mu{mirror_m}' if unit_k > 1 else ''}"
            f"{'' if k32z_rungs is None else '_q' + '_'.join(map(str, k32z_rungs))}"
            f"{'_dp' if defer_push else ''}"
            f"{'_xd' if x_dedup else ''}"
            f"{es_tag}"
            f"{f'_rs{run_sems}' if run_sems else ''}"
            f"{'_sd' if self_direct else ''}"
            f"{f'_xu{x_dup_unroll}' if x_dup_unroll else ''}"
        )
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n_prefetch,
                in_specs=[hbm] * n_inputs,
                out_specs=[hbm] * len(out_shape),
                scratch_shapes=tuple(scratch),
                grid=(),
            ),
            out_shape=out_shape,
            # The slab operand (last input) is the dispatch slab (output 4):
            # the SparseCore pre-push's rows stay where it wrote them.
            input_output_aliases={n_prefetch + n_inputs - 1: 4},
            compiler_params=pltpu.CompilerParams(
                collective_id=COLLECTIVE_ID,
                vmem_limit_bytes=host.vmem_limit(),
                # Bounds checks cost time and change no computed value.
                disable_bounds_checks=True,
            ),
            name=name,
        )

    def fn(
        tables,
        expert_rows,
        expert_base,
        act_q_sorted,
        act_scales,
        w1,
        w2,
        w1s,
        w2s,
        w1b=None,
        w2b=None,
        *,
        recv_rows,
        visit,
        counts,
        rank,
        dispatch,
        slab,
        x=None,
    ):
        """The served ragged transport form; returns arrivals and scales.

        `dispatch` = (segment table, meta) from host.push_dispatch_segments,
        `slab` the SparseCore pre-push's output, aliased to the dispatch slab.
        `x` (x_dedup builds only) = (x table [g_local, ep], SRC).
        """
        if (x is not None) != x_dedup:
            raise ValueError(
                f"the kernel was built with x_dedup={x_dedup} and its x tables "
                f"are {'present' if x is not None else 'absent'}"
            )
        # tables = the block-unit transport tuple and the row-unit push
        # tuple, (rows, base) from shard_expert_slabs, act_scales the
        # activation scale slab in the dense lane-block view
        # host.act_scale_slab_rows names, one f32 per slab row.
        # act_scales, w1s and w2s are the operands the weight format decides
        # on: the activation row scale exists where the rows were quantized
        # and the scale tables where the weights carry scales. w1b
        # [G, 2 * inter] and w2b [G, hidden] are the optional expert biases.
        # Whether each is present is a build-time property, so a mismatch is
        # a caller bug rather than a shape error deeper in. These are caller
        # operands, so they refuse rather than assert, for the reason stated
        # at the top of the builder -- and the failure here is not symmetric.
        # Supplying weight scales to a build whose format has none leaves
        # scale_args empty below and the operand is silently DROPPED, so the
        # kernel runs its matmuls with no weight scale applied and returns a
        # plausible, uniformly wrong result; this is the only guard for that
        # direction. (A build that wants scales and gets None fails loudly on
        # None.astype, and a missing bias fails on the pallas_call operand
        # count.) This is also the check that catches "you named bf16 and
        # supplied scales".
        for operand, present, what in (
            (act_scales, has_act_scale, "activation row scale"),
            (w1s, has_scales, "gate/up weight scale table"),
            (w2s, has_scales, "down weight scale table"),
            (w1b, has_w1_bias, "gate/up bias"),
            (w2b, has_w2_bias, "down bias"),
        ):
            if (operand is not None) != present:
                raise ValueError(
                    f"the kernel was built for weight format "
                    f"{weight_format!r} with has_w1_bias={has_w1_bias} "
                    f"has_w2_bias={has_w2_bias}, which "
                    f"{'wants' if present else 'has no operand for'} the "
                    f"{what}, and it is "
                    f"{'present' if operand is not None else 'absent'}"
                )
        act_scale_arg = (
            (
                act_scales.reshape(
                    host.act_scale_slab_rows(ragged_rows_alloc), HIDDEN_LANE_BLOCK
                ),
            )
            if has_act_scale
            else ()
        )
        if k32z:
            # The shift tables are bytes; a widening would change their
            # meaning, so a table that is not uint8 is refused instead.
            for what, table in (("w1s", w1s), ("w2s", w2s)):
                if jnp.dtype(table.dtype) != jnp.dtype(jnp.uint8):
                    raise ValueError(
                        f"the K32Z {what} shift table is uint8; got "
                        f"{jnp.dtype(table.dtype).name}"
                    )
            scale_args = (w1s, w2s)
        else:
            scale_args = (
                (w1s.astype(jnp.float32), w2s.astype(jnp.float32)) if has_scales else ()
            )
        biases = tuple(b.astype(jnp.float32) for b in (w1b, w2b) if b is not None)
        call = make_call(recv_rows)
        res = call(
            *tables,
            expert_rows.astype(jnp.int32),
            expert_base.astype(jnp.int32),
            visit.astype(jnp.int32),
            counts.astype(jnp.int32),
            rank.astype(jnp.int32),
            *(d.astype(jnp.int32) for d in dispatch),
            *(() if x is None else tuple(a.astype(jnp.int32) for a in x)),
            act_q_sorted.reshape(-1, lane_blocks, HIDDEN_LANE_BLOCK),
            *act_scale_arg,
            w1,
            w2,
            *scale_args,
            *biases,
            slab.reshape(-1, lane_blocks, HIDDEN_LANE_BLOCK),
        )
        recv, rscl, _contrib, _contrib_scl, _slab = res
        if rscl.ndim == 3:
            # A split mirror back as [sublanes, tile_m]: the same values in
            # the same order, which is all the combine's lookup reads.
            rscl = rscl.reshape(rscl.shape[0], -1)
        return recv, rscl

    return fn


_BUILD_CACHE = {}
_BUILD_CACHE_LOCK = threading.Lock()


def build_push_moe_kernel(**kwargs):
    """Memoizing front door for _build_push_moe_kernel (see v2's)."""
    key = tuple(sorted(kwargs.items()))
    fn = _BUILD_CACHE.get(key)
    if fn is None:
        with _BUILD_CACHE_LOCK:
            fn = _BUILD_CACHE.get(key)
            if fn is None:
                fn = _build_push_moe_kernel(**kwargs)
                _BUILD_CACHE[key] = fn
                logger.info(
                    "push fused EP MoE: built program %d for %s",
                    len(_BUILD_CACHE),
                    ", ".join(f"{k}={v}" for k, v in key),
                )
    return fn
