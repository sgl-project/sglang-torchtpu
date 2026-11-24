# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/fused_moe/v2/host.py
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
"""Host-side halves of the fused expert-parallel MoE kernel: the layout
constants the two halves share, the routing tables a call builds before it
enters the kernel, and the VMEM accounting that answers whether a build
fits. Nothing here runs inside the Pallas body."""

import enum
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

FP8_MAX = 448.0
FP8 = jnp.float8_e4m3fn
# Storage only: there is no four-bit MXU, so each block is widened to e4m3.
FP4 = jnp.float4_e2m1fn
# Default scale block along the contraction axis for a block-scaled weight.
QB4 = 512
# Four-bit values packed per uint32 word along K. The packing is a property
# of the width, not of the element type: any four-bit form packs eight to a
# word the same way.
PACK4 = 8
BF16 = jnp.bfloat16
# Storage only, the same way four-bit weights are: TPU v7, the generation
# this kernel is validated on, takes no integer input on the matrix unit
# (earlier generations do), so an integer weight is widened to
# bf16 one contraction chunk at a time before it is contracted. The format
# buys weight bytes and weight bandwidth, not matmul work.
INT8 = jnp.int8
# The contraction chunk integer weights widen in. It has to divide both
# matmuls' contractions, and the acceptance check says so.
WIDEN_KCHUNK = 512
# How far ahead of the expert being computed the next weight refill goes
# out: at expert e's head the kernel issues the refill for expert e + 2.
# Two is a schedule choice -- one refill in flight per weight buffer, one
# expert of slack -- not a hardware fact, which is why it is tunable.
#
# It is the CEILING a build starts from, not the distance a build runs at.
# Every slot of runway is a whole expert's weights held in VMEM, so whether
# a shape can afford the distance is a property of THAT SHAPE and not of the
# schedule: `fit_weight_slots` starts here and steps the distance down one
# at a time until the build's buffers fit under the budget.
#
# Zero is the useful other end: no runway at all, one weight slot, an expert
# starting its own weights at its head and waiting on them.
MAX_WEIGHT_PREFETCH_DISTANCE = 2


def nbuf_for(distance):
    """Weight buffer slots a prefetch distance needs.

    A refill writes its slot while the readers of the prefetch distance's
    worth of earlier experts are still live, so the slot count has to exceed
    the distance for all of them to stay distinct. At distance zero there are
    no earlier readers to stay clear of and the one slot is the expert's own.

    This relation IS what a distance costs, and it is written once here
    rather than at each build, because the two numbers travel together
    everywhere: a build that lowered one and kept the other would either
    overwrite a slab a live reader is still contracting against or pay VMEM
    for a slot nothing addresses.
    """
    return distance + 1


MAX_NBUF = nbuf_for(MAX_WEIGHT_PREFETCH_DISTANCE)
# Every transport moves whole blocks of this many rows; dynamic DMA offsets
# are block-aligned. It is also the padding each (expert, dest) run rounds up
# to, so it sets how many rows the gather fetches, the FFN computes and the
# transport moves that no token asked for -- half the arrival buffer at the
# decode bucket is this padding. It has to stay a power of two (ROWBLK_SHIFT)
# and small enough for the routing tables' alignment slot field (the check in
# refuse_bad_plan_args, which is an upper bound only).
ROWBLK = 2
# A token row is a whole number of 128-lane blocks: the lane count of the
# vector unit this kernel is built for.
HIDDEN_LANE_BLOCK = 128
# The token-gather table stays in HBM and is streamed through two scalar-memory
# windows. One window covers this many activation tiles. Grouping amortizes the
# metadata DMA, but the DMA is latency and not bandwidth: a window per tile is
# a smaller round trip at an expert's head, and every window after the first is
# started a tile ahead by the group-boundary path rather than waited for.
TOKEN_GATHER_TILES_PER_WINDOW = 1
# HBM int32 rows are tiled in 128-element units. Expert slabs are only
# ROWBLK-aligned, so a window may start up to 120 rows before its logical base
# and needs one extra HBM tile of overfetch.
TOKEN_GATHER_DMA_ALIGNMENT = HIDDEN_LANE_BLOCK
# The most of those blocks the kernel's row staging holds.
HIDDEN_MAX_BLOCKS = 64
# The share of the chip's VMEM the kernel may claim.
VMEM_FRACTION = 0.98
# The routing tables pack an arrival position and an alignment slot into
# one word as position * this + slot, so the slot field is this wide and
# the alignment slots a mesh needs have to fit under it.
ALIGNMENT_SLOT_FIELD = 64
# The widest block the routing tables bin routed pairs over. A rank pass
# inside a block is quadratic in it and the per-block tables it sizes scale
# the other way, which is why the value sits at the flat bottom of a U
# rather than at either end.
MAX_ROUTING_BLOCK = 256
# The window the SHARDED plan's rank pass self-compares over. A pair's rank
# inside its routing block is its rank inside a `window`-wide sub-block plus
# the same-expert pairs in earlier sub-blocks, and that second term is a
# prefix of the sub-block histogram which every consumer already folds into
# its per-block table. So the quadratic term costs `pairs * window` rather
# than `pairs * block` while the tables keep the block width. It changes no
# table the kernel reads.
ROUTING_WINDOW = 32
# Below this many routed pairs per shard the plan's cost is its operation
# COUNT, not its element count -- at the decode buckets it is ~35 small
# operations with no hot spot -- so the window's extra prefix tables cost
# more than its quadratic term saves and the block-wide pass is kept. The
# crossover is measured: at 2560 pairs the split trades 4.9 model us for
# five more entry operations, and at 10240 it trades 67 for five.
ROUTING_WINDOW_MIN_PAIRS = 4096


def pow2_shift(m, what):
    """The shift a division by `m` is, refusing anything else by name.

    The plan divides and takes remainders by these widths often enough
    that the arithmetic is worth writing as what it is. That rewrite is
    only an identity while the width is a power of two, so the condition
    is checked here, once, at import: a width that stopped being one
    would otherwise turn every rewritten site into silent wrong
    arithmetic rather than an error.
    """
    if m <= 0 or (m & (m - 1)):
        raise ValueError(
            f"{what} is {m}, which is not a power of two. The routing "
            "plan writes its divisions and remainders by this width as a "
            "shift and a mask, which is an identity only for a power of "
            "two, so this width cannot be changed without restoring the "
            "division at every site that reads these constants."
        )
    return m.bit_length() - 1


# A floor division by a power of two IS an arithmetic right shift and a
# remainder by one IS a bitwise and, for every int32 including negatives:
# both round toward negative infinity and both give the non-negative
# residue. So these are spellings of the same integer, not an assumption
# about the sign of anything the plan computes. Written out, they cost a
# shift or a mask; left as a division, they cost that plus the fixup a
# signed division owes, which this compiler emits and cannot fold away.
ROWBLK_SHIFT = pow2_shift(ROWBLK, "ROWBLK")
SLOT_FIELD_SHIFT = pow2_shift(ALIGNMENT_SLOT_FIELD, "ALIGNMENT_SLOT_FIELD")
SLOT_FIELD_MASK = ALIGNMENT_SLOT_FIELD - 1
# out_vm and its scale mirror are double-buffered: one tile computes into
# one parity while the previous tile's commit drains out of the other.
OUT_PARITIES = 2
# The widened view of a weight chunk a format has to widen -- a four-bit
# k-block to fp8, an integer contraction chunk to bf16 -- is double-
# buffered, so the next chunk widens while the current one feeds the MXU.
WIDENED_BLOCK_BUFFERS = 2
# The wire's scale mirror carries one f32 per row and is indexed by the
# TILE the row was computed in: one 128-lane SUBLANE of the mirror holds a
# whole tile's scales, and a run's scales are the sublanes of the tiles it
# spans. Mosaic refuses a DMA slice whose offset is not tile-aligned, so a
# transport can only address a mirror one sublane at a time however the
# array is shaped or typed -- 512 bytes either way. The only question is how
# many scales share those bytes: `tile_m` of them here, ROWBLK of them for a
# [blocks, ROWBLK] mirror.


def align_up(v, m):
    return -(-v // m) * m


def token_gather_window_rows(tile_m):
    """Int32 rows in one aligned HBM-to-SMEM token-gather window."""
    # Keep the DMA window tiled even when the compute tile is below 128 rows.
    # Layer padding and SMEM sizing use this same overfetch extent.
    return align_up(
        TOKEN_GATHER_TILES_PER_WINDOW * tile_m + TOKEN_GATHER_DMA_ALIGNMENT,
        TOKEN_GATHER_DMA_ALIGNMENT,
    )


def token_gather_smem_bytes(tile_m):
    """Fixed scalar-memory footprint of the two token-gather windows."""
    return 2 * token_gather_window_rows(tile_m) * jnp.dtype(jnp.int32).itemsize


def mirror_run_tiles(run_start, run_rows, tile_m):
    """Mirror sublanes a run owns: the source tiles its rows fall in."""
    shift = pow2_shift(tile_m, "tile_m")
    last = run_start + jnp.maximum(run_rows, 1) - 1
    return jnp.where(
        run_rows > 0,
        jnp.right_shift(last, shift) - jnp.right_shift(run_start, shift) + 1,
        0,
    )


def contrib_mirror_rows(rows_alloc, g_local, tile_m):
    """Sublanes the contribution mirror needs.

    An expert's tiles start at the tile its slab rows start in, plus two
    sublanes of slack per expert: that is what makes the base a shift of the
    expert's slab base rather than a table, and two is enough because
    ceil(rows / tile_m) never exceeds (rows >> shift) + 1 while the next
    expert's base moves on by at least that much plus two.
    """
    return rows_alloc // tile_m + 2 * g_local + 2


def arrival_mirror_rows(recv_rows, e_total, tile_m):
    """Sublanes the arrival mirror needs, on the same slack rule."""
    return recv_rows // tile_m + 2 * e_total + 2


def mirror_split(tile_m):
    """Rows of HIDDEN_LANE_BLOCK lanes one mirror sublane is split over.

    A mirror sublane holds one tile's tile_m scales. Up to one lane block it
    is one row of a [sublanes, tile_m] array, which a transport can address a
    row at a time. Mosaic tiles a wider f32 row (8, 128), which no transport
    of one sublane (or of a dynamic count of them) can address, so a taller
    tile's sublane is split over this many 128-lane rows of a 3-D
    [sublanes, rows, 128] array whose sublane axis is untiled.
    """
    return max(1, tile_m // HIDDEN_LANE_BLOCK)


# Lanes one scale-mirror sublane holds where a build asks for a fixed unit
# below its tile height: 0 keeps a sublane per tile (tile_m
# lanes, the 3-D split above HIDDEN_LANE_BLOCK); HIDDEN_LANE_BLOCK makes a
# 256-row K32Z tile write two 128-lane sublanes, so every mirror -- the
# arrival one the combine gathers from included -- is [sublanes, 128] with
# the slack of a 128-row build. Read by the layers at trace time for K32Z
# builds and handed to the builders as mirror_unit. 128 is the default: same
# bits, and a layer call (back to back) AZ -13 us / DZ -30 us at 1024 rows
# per rank; push's fused-scale combine applies again.
K32Z_MIRROR_UNIT = 128


def mirror_lanes(tile_m, mirror_unit=0):
    """Lanes per mirror sublane: tile_m, or `mirror_unit` below it."""
    if mirror_unit and tile_m > mirror_unit:
        if tile_m % mirror_unit or mirror_unit % ROWBLK:
            raise ValueError(
                f"a {mirror_unit}-lane mirror unit does not divide {tile_m}-row "
                f"tiles into whole {ROWBLK}-row blocks"
            )
        return mirror_unit
    return tile_m


def mirror_shape(sublanes, tile_m):
    """The shape of a scale mirror of `sublanes` tile sublanes."""
    k = mirror_split(tile_m)
    return (sublanes, tile_m) if k == 1 else (sublanes, k, tile_m // k)


def row_lane_blocks(hidden):
    """Lane blocks one token row is staged as: a row is (this many, 128)."""
    return hidden // HIDDEN_LANE_BLOCK


def act_scale_slab_rows(rows_alloc):
    """Sublanes the activation row-scale slab is handed to the kernel as.

    The slab is one f32 per slab row, and a column of them is the wrong
    thing to ship: a [rows, 1] array is padded out to a full lane block on
    the way to the kernel, so building it costs a copy of the padded array
    and that copy grows with the row count. The dense [rows / lanes, lanes]
    view holds the same bytes in the same order -- a lane block of f32 is
    exactly the tile the flat array is already laid out in -- so the layer
    hands the kernel that view and pays nothing to build it.

    ONE ROW MORE than the slab needs. The kernel reads a tile's scales as
    the sublanes its first row falls in, and a tile whose first row is not
    on a lane-block boundary reaches into the next sublane; at the last
    tile that sublane is past the slab. Bounds checks are off in this
    kernel, so the row exists rather than being trusted not to be read.
    Nothing selects from it: the rows it carries are past the slab's last.
    """
    return align_up(rows_alloc, HIDDEN_LANE_BLOCK) // HIDDEN_LANE_BLOCK + 1


def act_scale_window_rows(tile_m):
    """Sublanes of the scale slab one tile of `tile_m` rows can touch."""
    return align_up(tile_m, HIDDEN_LANE_BLOCK) // HIDDEN_LANE_BLOCK + 1


class WeightFormat(str, enum.Enum):
    """The weight formats the kernel takes, one member per accepted form.

    A member IS its own spelling: the value is the word the format has
    always been called by, and __str__ and __repr__ render that word, so a
    kernel name, a log line, a cache key and a refusal read exactly what
    they read when the format was a bare string. The point of the type is
    that a format now has one declaration to spell it, and a caller naming
    one that does not exist is refused by the table below rather than
    carried as far as the first comparison that quietly fails.
    """

    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    BF16 = "bf16"
    # Block-32 MXFP4 experts kept lossless: the checkpoint's e2m1 codes plus
    # per-column shift tables (see the K32Z section below). Its weights are
    # float4 like FP4's, so it is only ever selected by name.
    K32Z = "k32z"

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


class WeightForm(NamedTuple):
    """One weight format the kernel takes, and everything that follows it.

    A caller names a format; every other element type in the layer is read
    off this record rather than written down where it is used, so there is
    one place that says what a format implies.

    scale_layout is the layout of the weight scales the caller supplies:
    "per_channel" is one f32 per output channel of each matmul,
    "per_contraction_block" one per (contraction block, output channel),
    "column_shift" the K32Z uint8 shift tables (k32z_table_geometry), and
    "none" is an unquantized weight that carries no scales at all.

    act_dtype is what token rows are staged and contracted in and act_max is
    the peak they are row-quantized against, or None where they are carried
    as they arrive. wire_dtype is what a computed result row crosses the
    transport in; it follows the activation format, because an arrival row
    re-enters its token owner's sum as one more term of the same kind.
    """

    name: WeightFormat
    weight_dtype: jax.typing.DTypeLike
    act_dtype: object
    act_max: object
    scale_layout: str
    wire_dtype: object

    @property
    def has_scales(self):
        """Whether the caller supplies weight scales at all."""
        return self.scale_layout != "none"

    @property
    def quantized_activations(self):
        """Whether token rows are quantized, and so carry a row scale."""
        return self.act_max is not None


# The accepted (weight dtype, weight-scale layout) pairs. A weight element
# type outside this table is refused by name rather than reinterpreted, and
# a refusal lists these keys, so a caller is told what IS accepted.
#
# Two four-bit forms are deliberately absent. Integer-four has no scale
# layout here and no widening; and a four-bit block of 32, which the common
# mixed-format layout uses, is refused by the packed-weight row tile rather
# than by this table: eight values pack into a 32-bit word and those words
# tile to eight sublanes, so a block is a whole number of 64 rows or it is
# not addressable at all.
#
# Only the ELEMENT TYPE is fp4-specific. The packing, the block geometry,
# the staging and the transport are four-bit generic, and the block-scaled
# FFN body contracts whatever the reader hands it, so a block-scaled
# integer-four form would be one more row here plus the bitcast and widening
# target in the block reader -- the body itself would be reused verbatim.
WEIGHT_FORMS = {
    WeightFormat.FP8: WeightForm(
        WeightFormat.FP8, FP8, FP8, FP8_MAX, "per_channel", FP8
    ),
    WeightFormat.FP4: WeightForm(
        WeightFormat.FP4, FP4, FP8, FP8_MAX, "per_contraction_block", FP8
    ),
    WeightFormat.INT8: WeightForm(
        WeightFormat.INT8, INT8, BF16, None, "per_channel", BF16
    ),
    WeightFormat.BF16: WeightForm(WeightFormat.BF16, BF16, BF16, None, "none", BF16),
    WeightFormat.K32Z: WeightForm(
        WeightFormat.K32Z, FP4, FP8, FP8_MAX, "column_shift", FP8
    ),
}

WEIGHT_FORMAT_NAMES = tuple(WEIGHT_FORMS)


def weight_form(weight_format):
    """The record for a format; raises naming the accepted set."""
    try:
        return WEIGHT_FORMS[WeightFormat(weight_format)]
    except ValueError:
        raise NotImplementedError(
            f"the fused EP MoE kernel takes weight formats "
            f"{WEIGHT_FORMAT_NAMES}; got {weight_format!r}"
        ) from None


def weight_format_of_dtype(dtype):
    """The format carrying this weight element type, or None.

    This is the one place a format is DERIVED rather than named, so it is
    also the one place the enum is constructed from something outside it;
    everything downstream carries the member this returns. K32Z shares FP4's
    element type and is never derived: float4 weights are FP4 unless named.
    """
    for weight_format, form in WEIGHT_FORMS.items():
        if weight_format == WeightFormat.K32Z:
            continue
        if jnp.dtype(form.weight_dtype) == jnp.dtype(dtype):
            return WeightFormat(weight_format)
    return None


def ragged_stride_bound(num_tokens, topk, e_total, capacity):
    """Per-shard slab rows the no-drop worst case needs: every routed row
    landing on one shard, the row-block alignment padding and one
    tile-height tail-read window.

    This value is part of the built kernel's cache key, and it is a function
    of the TOKEN BUCKET, so the growth law of that cache is: one Pallas
    program per compiled token bucket, times the weight formats a process
    serves, times the four combinations of the two optional expert biases,
    times the fused activations. A deployment on exponential buckets to
    32768 compiles a handful; one that sets a fixed bucket padding gap
    compiles roughly the span divided by that gap. The cache is unbounded
    and holds every one for the life of the process, so a deployment that
    widens its bucket set pays in host memory and in boot time, and the
    build logs its size on every insert.

    The bound is a MINIMUM, so rounding it up to a coarser bucket spacing
    would collapse that cardinality without changing what the kernel
    computes. That is a design change rather than a bound, and it is not
    made here.
    """
    return align_up(num_tokens * topk + (ROWBLK - 1) * e_total + capacity, capacity)


def gather_clamp_bounds(T):
    """The (low, high) token numbers the routing gather table is clipped to.

    A function rather than an expression inside build_routing_tables so the
    bound has a witness: the interesting value is T = 0, which the table
    builder refuses before it gets here, so nothing that goes through the
    builder can check what this returns there.

    Both bounds matter, and the upper one has to be floored at zero to say
    anything at all. At T = 0 the natural bound is T - 1 = -1, and numpy --
    which jax follows -- makes a lower bound above the upper bound yield the
    UPPER bound, so clipping to (0, -1) returns -1 for every entry: the one
    line that exists to keep the gather in bounds is what puts it out of
    them, and bounds checks are off by construction so nothing downstream
    re-checks. At every T the builder accepts this is the identical pair it
    always was.
    """
    return 0, max(T - 1, 0)


def routing_block(t_local, topk):
    """The widest power-of-two block dividing this shard's routed pairs."""
    b = MAX_ROUTING_BLOCK
    while (t_local * topk) % b:
        b //= 2
    return b


class RoutingTables(NamedTuple):
    """Where every routed (token, expert) pair computes, and where it lands.

    One shard's kernel operands are cut from these by the three shard_
    functions below, which is why the whole set is replicated rather than
    sharded.
    """

    # [T, K]: the arrival row each token's k-th selection comes back on.
    arrival_row: jax.Array
    # [T, K]: the arrival MIRROR element that row's scale comes back in.
    mirror_row: jax.Array
    # [E, ep]: mirror sublanes one (expert, dest) push carries.
    run_tiles: jax.Array
    # [T * K]: the slab row each routed pair computes on.
    slab_row: jax.Array
    # [E, ep]: true rows of expert e whose tokens shard d owns.
    run_rows: jax.Array
    # [E, ep]: the same runs, each padded up to a whole ROWBLK.
    run_rows_aligned: jax.Array
    # [E, ep]: where each padded run starts inside its expert's slab.
    run_start_aligned: jax.Array
    # [source, g, dest]: rows one (expert, dest) push carries.
    region_rows: jax.Array
    # [dest, source, g]: the arrival row that push lands on.
    recv_base: jax.Array
    # [dest]: arrival rows a shard receives, its own rows included.
    recv_rows: jax.Array
    # [E]: padded rows per expert.
    expert_rows_aligned: jax.Array
    # [E]: where each expert's slab starts.
    expert_base: jax.Array
    # Slab rows one shard is allocated.
    rows_alloc: int

    @property
    def pair_grid(self):
        """(tokens, selections) the plan was built over."""
        return self.arrival_row.shape


def _lookup_per_pair(table, expert_blocks, bins):
    """``table[b, expert_blocks[b, i]]`` -- one value per routed pair.

    The one-hot select-and-sum spelling is deliberate. At the served shape,
    lowering this lookup as a gather cuts FLOPs but increases device traffic
    by more than an order of magnitude.
    """
    return jnp.sum(
        jnp.where(
            expert_blocks[:, :, None] == bins[None, None, :], table[:, None, :], 0
        ),
        axis=2,
    )


def _refuse_bad_plan_args(T, K, *, e_total, ep, t_local, block, tile_m):
    """Argument checks shared by the replicated and sharded plan builders."""
    n = T * K
    if T < 1:
        raise ValueError(
            "the routing tables need at least one token; an empty batch "
            "builds a gather table with no in-bounds row to clamp to"
        )
    if e_total % ep:
        raise ValueError(
            f"expert count {e_total} is not divisible by the expert-parallel width {ep}"
        )
    if (ROWBLK - 1) * (ep - 1) >= ALIGNMENT_SLOT_FIELD:
        raise ValueError(
            f"ep={ep} needs up to {(ROWBLK - 1) * (ep - 1)} alignment slots "
            f"and the routing tables pack them into {ALIGNMENT_SLOT_FIELD}; "
            f"widths up to {1 + (ALIGNMENT_SLOT_FIELD - 1) // (ROWBLK - 1)} "
            "are representable"
        )
    max_pos = n + (ROWBLK - 1) * e_total
    if max_pos * ALIGNMENT_SLOT_FIELD >= 2**31:
        raise ValueError(
            f"the routing tables would carry up to {max_pos} arrival rows "
            f"per shard, and packing one alongside a {ALIGNMENT_SLOT_FIELD}"
            f"-wide alignment slot holds {2**31 // ALIGNMENT_SLOT_FIELD}: "
            f"past that the two fields corrupt each other silently"
        )
    if n % block:
        raise ValueError(
            f"the {n} routed pairs are not a whole number of "
            f"{block}-pair routing blocks"
        )
    if (t_local * K) % block:
        raise ValueError(
            f"this shard's {t_local * K} routed pairs are not a "
            f"whole number of {block}-pair routing blocks"
        )
    if tile_m % ROWBLK:
        raise ValueError(
            f"tile height {tile_m} is not a whole number of {ROWBLK}-row blocks"
        )


def build_routing_tables(
    topk_idx, *, e_total, ep, t_local, block, tile_m, shard_stride, mirror_m=None
):
    """Assign every routed (token, expert) pair the slab row it computes on.

    `mirror_m` is the scale mirror's lanes per sublane (mirror_lanes; the
    tile height where None)."""
    # topk_idx [T, K] i32 expert ids, all-gathered. Rows order by expert then
    # by owning shard, each run padded to ROWBLK.
    T, K = topk_idx.shape
    n = T * K
    g_local = e_total // ep
    _refuse_bad_plan_args(
        T, K, e_total=e_total, ep=ep, t_local=t_local, block=block, tile_m=tile_m
    )
    expert_of_pair = topk_idx.reshape(-1).astype(jnp.int32)
    n_blocks = n // block
    expert_blocks = expert_of_pair.reshape(n_blocks, block)
    bins = jnp.arange(e_total, dtype=jnp.int32)

    block_hist = jnp.sum(
        (expert_blocks[:, :, None] == bins[None, None, :]).astype(jnp.int32), axis=1
    )  # [n_blocks, E]
    block_off = jnp.cumsum(block_hist, axis=0) - block_hist  # excl over blocks
    base_per_slot = _lookup_per_pair(
        block_off, expert_blocks, bins
    )  # [n_blocks, block]
    eq = expert_blocks[:, :, None] == expert_blocks[:, None, :]
    tri = jnp.tril(jnp.ones((block, block), dtype=jnp.bool_), k=-1)
    rank = jnp.sum((eq & tri[None]).astype(jnp.int32), axis=2)
    pair_rank = (base_per_slot + rank).reshape(-1)  # [n] rank in its expert

    blocks_per_dest = (t_local * K) // block
    rows_by_dest = block_hist.reshape(ep, blocks_per_dest, e_total).sum(axis=1)
    run_rows = rows_by_dest.T  # [E, ep]
    run_start = jnp.cumsum(run_rows, axis=1) - run_rows  # excl over d

    # Rounding up to a whole block: one add and one mask, where the
    # negate-divide-negate spelling owed a signed division's fixup.
    run_rows_aligned = (run_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    run_start_aligned = jnp.cumsum(run_rows_aligned, axis=1) - run_rows_aligned
    slot_shift = run_start_aligned - run_start  # slot = pair_rank + shift

    # Ragged expert slabs: shard s owns rows [s*stride, (s+1)*stride).
    expert_rows_aligned = run_rows_aligned.sum(axis=1)  # [E] 8-aligned
    if shard_stride % tile_m:
        raise ValueError(
            f"the per-shard slab stride {shard_stride} is not a "
            f"whole number of {tile_m}-row tiles"
        )
    rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
    local_base = jnp.cumsum(rows_by_shard, axis=1) - rows_by_shard  # [s, G]
    expert_base = (
        local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] * shard_stride)
    ).reshape(e_total)
    rows_alloc = ep * shard_stride

    # Every push is per expert, so a receive region is one (source, expert)
    # run: region_rows is run_rows_aligned read as [source, expert, dest].
    region_rows = run_rows_aligned.reshape(ep, g_local, ep)  # [s, g, d]
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(ep, ep * g_local)
    recv_base = (jnp.cumsum(rows_per_dest, axis=1) - rows_per_dest).reshape(
        ep, ep, g_local
    )  # [d, s, g]
    recv_rows = rows_per_dest.sum(axis=1)  # [d] incl. self

    pos_shift = recv_base.transpose(1, 2, 0).reshape(e_total, ep) - run_start
    dest_of_block = (jnp.arange(n_blocks, dtype=jnp.int32) * block) // (t_local * K)

    # One packed pass over both shifts; the ragged expert base needs its
    # own select-sum because its value range is too wide for the word.
    packed = pos_shift * ALIGNMENT_SLOT_FIELD + slot_shift
    packed_blocks = jnp.take(packed.T, dest_of_block, axis=0)  # [n_blocks, E]
    packed_sel = _lookup_per_pair(packed_blocks, expert_blocks, bins).reshape(-1)
    base_of_pair = _lookup_per_pair(
        jnp.broadcast_to(expert_base[None, :], (n_blocks, e_total)), expert_blocks, bins
    ).reshape(-1)
    # Unpacking the two fields of the packed word: the position is the
    # high part and the slot is the low part, which is a shift and a mask.
    pos = pair_rank + jnp.right_shift(packed_sel, SLOT_FIELD_SHIFT)
    slot = pair_rank + (packed_sel & SLOT_FIELD_MASK)

    # The mirror position: the pair's offset within its run, past the lane
    # its source tile's sublane starts on. Its own select-sum -- the packed
    # word above is full.
    mm = tile_m if mirror_m is None else mirror_m
    run_tiles = mirror_run_tiles(run_start_aligned, run_rows_aligned, mm)
    shift = pow2_shift(mm, "the mirror's lane count")
    rb = recv_base.transpose(1, 2, 0).reshape(e_total, ep)
    mirror_shift = (
        (jnp.right_shift(rb, shift) + 2 * jnp.arange(e_total, dtype=jnp.int32)[:, None])
        * mm
        + (run_start_aligned & (mm - 1))
        - run_start
    )
    mirror_blocks = jnp.take(mirror_shift.T, dest_of_block, axis=0)
    mirror_row = pair_rank + _lookup_per_pair(
        mirror_blocks, expert_blocks, bins
    ).reshape(-1)

    on_valid = (expert_of_pair >= 0) & (expert_of_pair < e_total)
    slab_row = jnp.where(
        on_valid,
        base_of_pair + slot,
        jnp.int32(off_shard_slab_row(shard_stride, ep)),
    )

    return RoutingTables(
        arrival_row=pos.reshape(T, K),
        mirror_row=mirror_row.reshape(T, K),
        run_tiles=run_tiles,
        slab_row=slab_row,
        run_rows=run_rows,
        run_rows_aligned=run_rows_aligned,
        run_start_aligned=run_start_aligned,
        region_rows=region_rows,
        recv_base=recv_base,
        recv_rows=recv_rows,
        expert_rows_aligned=expert_rows_aligned,
        expert_base=expert_base,
        rows_alloc=rows_alloc,
    )


class ShardRoutingTables(NamedTuple):
    """One shard's complementary slices of the routing plan.

    ``pos`` contains only this shard's token rows. ``slab_row`` contains a
    live row only for this shard's experts and an out-of-range sentinel for
    every other pair. The remaining small tables are shared with
    ``RoutingTables`` and are reconstructed from an ``[ep, e_total]`` count
    table.
    """

    pos: jax.Array
    mirror_pos: jax.Array
    run_tiles: jax.Array
    slab_row: jax.Array
    run_rows: jax.Array
    run_rows_aligned: jax.Array
    run_start_aligned: jax.Array
    region_rows: jax.Array
    recv_base: jax.Array
    recv_rows: jax.Array
    expert_rows_aligned: jax.Array
    expert_base: jax.Array
    rows_alloc: int
    n_tokens: int
    topk: int

    @property
    def pair_grid(self):
        return self.n_tokens, self.topk


def off_shard_slab_row(shard_stride, ep):
    """Sentinel beyond both slab destinations after any shard rebasing."""
    return ep * shard_stride + act_scale_slab_rows(shard_stride) * HIDDEN_LANE_BLOCK


def _rank_within_block(expert_blocks, block):
    """``#{j < i : e_j == e_i}`` for each pair in a routing block."""
    eq = expert_blocks[:, :, None] == expert_blocks[:, None, :]
    tri = jnp.tril(jnp.ones((block, block), dtype=jnp.bool_), k=-1)
    return jnp.sum((eq & tri[None]).astype(jnp.int32), axis=2)


def _block_hist(expert_blocks, bins):
    """Number of pairs per routing block and expert bin."""
    return jnp.sum(
        (expert_blocks[:, :, None] == bins[None, None, :]).astype(jnp.int32), axis=1
    )


def pair_block_hist(expert_blocks, e_total):
    """Pairs per routing block and expert, over ``e_total`` bins."""
    return _block_hist(expert_blocks, jnp.arange(e_total, dtype=jnp.int32))


def rank_window(block, shard_pairs):
    """The width the rank pass self-compares over, `block` where it pays."""
    if shard_pairs < ROUTING_WINDOW_MIN_PAIRS:
        return block
    w = ROUTING_WINDOW
    while block % w:
        w //= 2
    return w


def _per_sub_block(x, window):
    """``x[b, sub, ...]`` read back as ``x[b, pair, ...]``.

    Index arithmetic, not data movement: the sub-block axis is expanded over
    the pairs that share it and merged away again, both inside whatever
    fusion consumes the result. Materialising a ``[.., sub, window, ..]``
    array instead would pad a 32-wide minor axis out to a 128-lane tile.
    """
    n_blocks, n_sub = x.shape[:2]
    tail = x.shape[2:]
    return jnp.broadcast_to(x[:, :, None], (n_blocks, n_sub, window) + tail).reshape(
        n_blocks, n_sub * window, *tail
    )


def _rank_within_window(expert_blocks, window):
    """``#{j < i : e_j == e_i}`` over i's sub-block rather than its block.

    The block-wide pass compares every pair against every other pair in the
    block, which is ``pairs * block`` element-ops; restricting the comparison
    to a ``window``-wide sub-block makes it ``pairs * window``. What the
    window drops -- same-expert pairs in earlier sub-blocks -- is what
    ``_window_table`` folds into the table the consumer looks up.
    """
    n_blocks, block = expert_blocks.shape
    if window == block:
        return _rank_within_block(expert_blocks, block)
    peers = _per_sub_block(
        expert_blocks.reshape(n_blocks, block // window, window), window
    )
    lower = (
        jnp.arange(window, dtype=jnp.int32)[None, :]
        < jnp.arange(block, dtype=jnp.int32)[:, None] % window
    )
    return jnp.sum(
        ((expert_blocks[:, :, None] == peers) & lower[None]).astype(jnp.int32), axis=2
    )


def _window_hist(expert_blocks, bins, window):
    """Pairs per (routing block, sub-block, expert bin).

    An unsplit block keeps ``_block_hist``'s own pass under a unit sub-block
    axis, so the emitted program is the unsplit one exactly.
    """
    n_blocks, block = expert_blocks.shape
    if window == block:
        return _block_hist(expert_blocks, bins)[:, None, :]
    eb = expert_blocks.reshape(n_blocks, block // window, window)
    return jnp.sum(
        (eb[:, :, :, None] == bins[None, None, None, :]).astype(jnp.int32), axis=2
    )


def _over_sub_blocks(window_hist):
    """The per-block histogram a sub-block histogram refines."""
    return window_hist[:, 0] if window_hist.shape[1] == 1 else window_hist.sum(axis=1)


def _window_table(table, window_hist):
    """A per-block plan table resolved per sub-block, for the lookup.

    The part of the rank a window drops -- same-expert pairs in earlier
    sub-blocks of the pair's own block -- is added here, on the small table,
    rather than to the per-pair result. It is a strictly-lower-triangular
    select-sum and not a cumsum: the sub-block axis is a handful of elements
    long and ``jnp.cumsum`` lowers it to a reduce-window, which becomes a
    module of its own that the entry cost model prices at nothing and the
    schedule pays for anyway.
    """
    n_sub = window_hist.shape[1]
    if n_sub == 1:
        return table[:, None, :]
    tri = jnp.tril(jnp.ones((n_sub, n_sub), dtype=jnp.bool_), k=-1)
    return table[:, None, :] + jnp.sum(
        jnp.where(tri[None, :, :, None], window_hist[:, None, :, :], 0), axis=2
    )


def _lookup_per_sub_pair(table, expert_blocks, bins, window):
    """``table[b, i // window, expert_blocks[b, i]]`` -- one value per pair.

    ``_lookup_per_pair`` with the table resolved per sub-block as well as per
    block. The one-hot summed over is the same size and the result keeps the
    ``[n_blocks, block]`` shape, so the finer table costs nothing.
    """
    if table.shape[1] == 1:
        return _lookup_per_pair(table[:, 0], expert_blocks, bins)
    return jnp.sum(
        jnp.where(
            expert_blocks[:, :, None] == bins[None, None, :],
            _per_sub_block(table, window),
            0,
        ),
        axis=2,
    )


class GatheredPairs(NamedTuple):
    """The routing exchange, already unpacked by the caller.

    ``expert_blocks`` is the gathered ``[n_blocks, block]`` pair grid,
    ``local_blocks`` this shard's own ``[blocks_per_dest, block]`` slice of
    it, ``block_hist`` that slice's per-block expert histogram and
    ``rows_by_dest`` the exchanged ``[ep, e_total]`` count table. A caller
    that carries the count row inside the routing all-gather holds all four
    already, and handing them over removes this builder's own collective and
    the ``[T, topk]`` selection array it would otherwise reshape.
    """

    expert_blocks: jax.Array
    local_blocks: jax.Array
    block_hist: jax.Array
    rows_by_dest: jax.Array
    n_tokens: int
    topk: int


def build_routing_tables_sharded(
    topk_idx,
    me,
    *,
    e_total,
    ep,
    t_local,
    block,
    tile_m,
    shard_stride,
    all_gather_rows=None,
    gathered=None,
    mirror_m=None,
):
    """Build only the routing-plan slices consumed by shard ``me``.

    ``mirror_m`` is the scale mirror's lanes per sublane (mirror_lanes; the
    tile height where None).

    The replicated builder performs four one-hot passes over
    ``T * topk * e_total`` on every rank. The two large outputs are consumed
    in complementary slices: arrival positions need this rank's token block
    over all experts, while slab rows need all tokens over this rank's local
    experts. This builder computes those two 1/ep slices and exchanges only a
    ``[1, e_total]`` i32 count row (2 KiB at Qwen's 512 experts).

    That exchange is this builder's own ``all_gather_rows`` collective.
    ``gathered`` supplies it instead, from a caller that folded the count row
    into the routing all-gather it was already paying for; ``topk_idx`` is
    then unused. Every table below is identical either way.
    """
    if (gathered is None) == (all_gather_rows is None):
        raise ValueError("pass exactly one of all_gather_rows and gathered")
    T, K = topk_idx.shape if gathered is None else (gathered.n_tokens, gathered.topk)
    n = T * K
    g_local = e_total // ep
    _refuse_bad_plan_args(
        T, K, e_total=e_total, ep=ep, t_local=t_local, block=block, tile_m=tile_m
    )
    if shard_stride % tile_m:
        raise ValueError(
            f"the per-shard slab stride {shard_stride} is not a "
            f"whole number of {tile_m}-row tiles"
        )

    n_blocks = n // block
    blocks_per_dest = (t_local * K) // block
    expert_blocks = (
        topk_idx.reshape(-1).astype(jnp.int32).reshape(n_blocks, block)
        if gathered is None
        else gathered.expert_blocks
    )
    bins = jnp.arange(e_total, dtype=jnp.int32)
    window = rank_window(block, blocks_per_dest * block)

    # This rank's token block across every expert. Its histogram row is the
    # only non-local input needed by the small-table reconstruction below.
    if gathered is None:
        my_blocks = lax.dynamic_slice(
            expert_blocks, (me * blocks_per_dest, 0), (blocks_per_dest, block)
        )
        my_hist = _block_hist(my_blocks, bins)
        assert all_gather_rows is not None
        rows_by_dest = all_gather_rows(my_hist.sum(axis=0, keepdims=True))
    else:
        my_blocks, my_hist = gathered.local_blocks, gathered.block_hist
        rows_by_dest = gathered.rows_by_dest
    my_off = jnp.cumsum(my_hist, axis=0) - my_hist

    run_rows = rows_by_dest.T
    run_rows_aligned = (run_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    run_start_aligned = jnp.cumsum(run_rows_aligned, axis=1) - run_rows_aligned
    expert_rows_aligned = run_rows_aligned.sum(axis=1)
    rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
    local_base = jnp.cumsum(rows_by_shard, axis=1) - rows_by_shard
    expert_base = (
        local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] * shard_stride)
    ).reshape(e_total)
    region_rows = run_rows_aligned.reshape(ep, g_local, ep)
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(ep, ep * g_local)
    recv_base = (jnp.cumsum(rows_per_dest, axis=1) - rows_per_dest).reshape(
        ep, ep, g_local
    )
    recv_rows = rows_per_dest.sum(axis=1)

    # For a pair in this rank's token block the source prefix cancels from
    # the arrival row, leaving the destination receive base plus its offset
    # within the source's expert run.
    my_recv_base = lax.dynamic_slice(recv_base, (me, 0, 0), (1, ep, g_local)).reshape(
        e_total
    )
    pos_table = my_off + my_recv_base[None, :]
    # The rank pass this consumer needs is the block-wide one restricted to
    # this rank's own blocks, which is that pass over those blocks alone.
    my_rank = _rank_within_block(my_blocks, block)
    pos = (_lookup_per_pair(pos_table, my_blocks, bins) + my_rank).reshape(t_local, K)

    # The mirror position of the same pair: its offset within the run, past
    # the lane its source tile's sublane starts on. Same shape of lookup as
    # `pos`, over the one-hot the compiler shares between them.
    mm = tile_m if mirror_m is None else mirror_m
    run_tiles = mirror_run_tiles(run_start_aligned, run_rows_aligned, mm)
    shift = pow2_shift(mm, "the mirror's lane count")
    my_run_start = lax.dynamic_slice(run_start_aligned, (0, me), (e_total, 1)).reshape(
        e_total
    )
    mirror_table = (
        my_off
        + (
            (
                jnp.right_shift(my_recv_base, shift)
                + 2 * jnp.arange(e_total, dtype=jnp.int32)
            )
            * mm
            + (my_run_start & (mm - 1))
        )[None, :]
    )
    mirror_pos = (_lookup_per_pair(mirror_table, my_blocks, bins) + my_rank).reshape(
        t_local, K
    )

    # Across all tokens, compute offsets only for this rank's local experts.
    first = me * g_local
    my_experts = first + jnp.arange(g_local, dtype=jnp.int32)
    win_hist = _window_hist(expert_blocks, my_experts, window)
    hist_mine = _over_sub_blocks(win_hist)
    by_source = hist_mine.reshape(ep, blocks_per_dest, g_local)
    off_in_source = jnp.cumsum(by_source, axis=1) - by_source
    run_base = (
        lax.dynamic_slice(expert_base, (first,), (g_local,))[None, :]
        + lax.dynamic_slice(run_start_aligned, (first, 0), (g_local, ep)).T
    )
    row_table = _window_table(
        (off_in_source + run_base[:, None, :]).reshape(n_blocks, g_local), win_hist
    )
    on_me = (expert_blocks >= first) & (expert_blocks < first + g_local)
    my_row = _lookup_per_sub_pair(
        row_table, expert_blocks, my_experts, window
    ) + _rank_within_window(expert_blocks, window)
    slab_row = jnp.where(
        on_me, my_row, jnp.int32(off_shard_slab_row(shard_stride, ep))
    ).reshape(-1)

    return ShardRoutingTables(
        pos=pos,
        mirror_pos=mirror_pos,
        run_tiles=run_tiles,
        slab_row=slab_row,
        run_rows=run_rows,
        run_rows_aligned=run_rows_aligned,
        run_start_aligned=run_start_aligned,
        region_rows=region_rows,
        recv_base=recv_base,
        recv_rows=recv_rows,
        expert_rows_aligned=expert_rows_aligned,
        expert_base=expert_base,
        rows_alloc=ep * shard_stride,
        n_tokens=T,
        topk=K,
    )


def shard_expert_slabs(routing, me, *, e_total, ep):
    """Shard `me`'s local experts: (rows, slab start) i32 [G], in row units."""
    g_local = e_total // ep
    rows = lax.dynamic_slice(routing.expert_rows_aligned, (me * g_local,), (g_local,))
    base_g = lax.dynamic_slice(routing.expert_base, (me * g_local,), (g_local,))
    base = base_g - base_g[0]
    return rows.astype(jnp.int32), base.astype(jnp.int32)


def local_slab_rows(routing, me, *, shard_stride):
    """`routing.slab_row` rebased onto shard `me`'s own slab.

    A pair that computes on another shard is sent past the end of this
    shard's slab, so a scatter through this index carrying mode="drop"
    writes only the rows this shard will read. The off-shard pairs are
    dropped rather than accumulated onto a sink row, because every shard but
    one owns most of the pairs and a sink would take (ep - 1) / ep of them
    as write conflicts on a single row.

    The index is never negative: a row below this shard's slab is mapped to
    shard_stride, not to a negative offset, because jax indexing wraps a
    negative index around the destination instead of dropping it.
    """
    base = me * shard_stride
    return jnp.where(
        routing.slab_row < base, jnp.int32(shard_stride), routing.slab_row - base
    )


def shard_token_gather(routing, me, *, shard_stride, rows=None, scatter_add=None):
    """The token number each of shard `me`'s slab rows computes, [stride].

    `rows` is the destination slab row of each routed pair, which is derived
    here when the caller does not hold it already; passing it in keeps both
    slab scatters on one index.

    The scatter runs against this shard's slab alone rather than against the
    replicated ep-wide slab followed by a slice, which built seven eighths
    of its result to discard it. What the kernel receives is unchanged.

    The kernel fetches an input row by this table value with bounds checks
    off, so the one index that reaches memory is clamped here rather than
    trusted. An expert id outside [0, e_total) is in no expert's count, and
    both plan builders send its pair to `off_shard_slab_row`, so it never
    reaches this scatter's live rows; the layer relies on that for the id
    e_total it gives padding rows. The clamp is the backstop. Clipped
    on BOTH sides, to the bounds gather_clamp_bounds names -- see there for
    why the upper one is floored at zero. build_routing_tables refuses T = 0
    before any of this; the clip is written so that it would hold on its own.
    """
    T, K = routing.pair_grid
    # The token number of each routed pair is a counted repeat, not a
    # division: pair i belongs to token i // K, which is the token index
    # broadcast K times. K is the selection width and need not be a power
    # of two, so this is the spelling that removes the division rather
    # than the one that shifts it.
    token_of_pair = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(-1)
    row = (
        local_slab_rows(routing, me, shard_stride=shard_stride)
        if rows is None
        else rows
    )
    gather_lo, gather_hi = gather_clamp_bounds(T)
    if scatter_add is not None:
        # The same scatter-add, placed by the caller (SparseCore core 1,
        # host.sc_side_core).
        scattered = scatter_add(shard_stride, row, token_of_pair)
    else:
        scattered = (
            jnp.zeros((shard_stride,), jnp.int32)
            .at[row]
            .add(token_of_pair, mode="drop")
        )
    return jnp.clip(scattered, jnp.int32(gather_lo), jnp.int32(gather_hi))


def expert_visit_list(rows, g_local):
    """The local experts to visit, and how many, from rows [g_local] i32."""
    # visit[:n_visit] = the local expert indices with rows > 0, most rows
    # first; the tail is never visited. Ties break by ascending index. The
    # visit order does not change the output, only the weight refill order.
    mask = rows > 0
    n_visit = mask.sum().astype(jnp.int32).reshape(1)
    order = jnp.arange(g_local, dtype=jnp.int32)
    rows_i = rows.astype(jnp.int32)
    # One key, ranked once. The negated row count is the high factor of the
    # word and the index the low one, so the packed integer compares exactly
    # as the (rows descending, index ascending) pair does: every index is
    # below g_local, so the low factor is exact and no two experts can tie.
    # An empty expert's negated count is zero, the largest value the high
    # factor takes, so the empty experts land last without an activity key
    # of their own. The word is int32, and an expert's rows never exceed the
    # shard's slab row allocation, so the pack is exact while that
    # allocation stays under 2**31 // g_local.
    #
    # No two keys are equal, so the sort is an inversion of the key's rank
    # and both halves are one g_local-square compare. `jnp.argsort` of the
    # same key is exact too and costs a sorting network of its own -- a
    # module in the compiled program, one of its largest, for 64 elements.
    key = order - rows_i * g_local
    rank = jnp.sum((key[None, :] < key[:, None]).astype(jnp.int32), axis=1)
    perm = jnp.sum(
        jnp.where(rank[None, :] == order[:, None], order[None, :], 0), axis=1
    )
    visit = jnp.minimum(perm, jnp.int32(g_local - 1)).astype(jnp.int32)
    return visit, n_visit


def shard_push_tables_in_rows(routing, me, *, e_total, ep):
    """Shard `me`'s push tables in ROW units, for true-length pushes."""
    # true_rows [G, ep] rows per (e, d), recv_row_off [G, ep] the arrival row
    # that run starts at, totals [2] send and remote recv rows. Only the
    # pushed lengths shrink: the recv and contrib layouts stay aligned, so the
    # arrival tables do not move.
    g_local = e_total // ep
    all_run_rows = routing.run_rows  # [E, ep]
    true_rows = lax.dynamic_slice(all_run_rows, (me * g_local, 0), (g_local, ep))
    recv_base = routing.recv_base  # [d, s, g]
    my_recv_base = lax.dynamic_slice(recv_base, (0, me, 0), (ep, 1, g_local))[:, 0]
    recv_row_off = my_recv_base.T  # [G, d]
    not_me = (jnp.arange(ep) != me).astype(true_rows.dtype)
    send_true = (true_rows * not_me[None, :]).sum()
    self_true = (true_rows * (1 - not_me)[None, :]).sum()
    recv_true = (
        lax.dynamic_slice(all_run_rows.sum(axis=0).astype(jnp.int32), (me,), (1,))[0]
        - self_true
    )
    return (
        true_rows.astype(jnp.int32),
        recv_row_off.astype(jnp.int32),
        jnp.stack([send_true, recv_true]).astype(jnp.int32),
    )


def shard_transport_tables_in_blocks(routing, me, *, e_total, ep):
    """Shard `me`'s transport tables in ROWBLK-row block units."""
    # All i32. contrib: regions per dest d, packed in d order, each region =
    # groups asc, experts asc. recv: regions per (src asc, group asc).
    g_local = e_total // ep
    # Every table value below is in 8-row BLOCK units, which the
    # conversions at the bottom reach by a shift.
    aligned_rows = lax.dynamic_slice(
        routing.run_rows_aligned, (me * g_local, 0), (g_local, ep)
    )
    run_start = lax.dynamic_slice(
        routing.run_start_aligned, (me * g_local, 0), (g_local, ep)
    )
    my_region_rows = lax.dynamic_slice(
        routing.region_rows, (me, 0, 0), (1, g_local, ep)
    )[0]
    recv_base = routing.recv_base  # [d, s, g]

    not_me = jnp.arange(ep) != me
    # contrib includes the own-dest region, which hops to recvbuf later.
    out_total = my_region_rows.sum(axis=0)  # [d] incl. me
    contrib_base = jnp.cumsum(out_total) - out_total  # [d]
    grp_off = jnp.cumsum(my_region_rows, axis=0) - my_region_rows  # [g, d]

    # One expert per push, so an expert's contrib offset is its region's.
    contrib_off = contrib_base[None, :] + grp_off  # [G, d]

    push_src = contrib_base[None, :] + grp_off  # [g, d]
    push_len = my_region_rows
    # recv_base[d (receiver), me (src), g] for every d: [d, g] -> [g, d].
    my_recv_base = lax.dynamic_slice(recv_base, (0, me, 0), (ep, 1, g_local))[:, 0]
    push_dst = my_recv_base.T
    not_me_i = not_me.astype(my_region_rows.dtype)
    send_rows = (my_region_rows * not_me_i[None, :]).sum()
    self_rows = jnp.sum(aligned_rows * (1 - not_me_i)[None, :])
    recv_remote = lax.dynamic_slice(routing.recv_rows, (me,), (1,))[0] - self_rows
    totals = jnp.stack(
        [
            jnp.right_shift(send_rows, ROWBLK_SHIFT),
            jnp.right_shift(recv_remote, ROWBLK_SHIFT),
        ]
    ).astype(jnp.int32)

    def i32(a):
        """One table in block units: the row count's high bits."""
        return jnp.right_shift(a, ROWBLK_SHIFT).astype(jnp.int32)

    return (
        i32(run_start),
        i32(aligned_rows),
        i32(contrib_off),
        i32(push_src),
        i32(push_len),
        i32(push_dst),
        totals,
    )


# Rows of the count table, in the order the kernel reads them. The two
# transport rows are in ROWS; the kernel divides them into blocks, which
# is where that division is cheapest (see shard_count_vector).
COUNT_SEND_ROWS = 0
COUNT_RECV_ROWS = 1
COUNT_SEND_ALIGNED_ROWS = 2
COUNT_RECV_ALIGNED_ROWS = 3
COUNT_VISITS = 4
COUNT_SEND_MIRROR = 5
COUNT_RECV_MIRROR = 6
COUNT_SELF_MIRROR = 7
N_COUNTS = 8
# A build with the early-ship selector appends one row to the count table,
# whose lane 0 is the per-call early-ship switch (early_ship_flag_row).
COUNT_ES_FLAG = N_COUNTS


def shard_count_vector(routing, expert_rows, me, *, e_total, ep):
    """Shard `me`'s five kernel counts, one per row, [N_COUNTS, ep] i32.

    The kernel needs five integers: the rows it sends and receives, the
    same two in block units, and how many local experts it visits. Built
    one at a time they are five separate reductions to a scalar, and a
    reduction that lands on a scalar is the expensive shape in this stage:
    the neighbouring reduction that keeps a vector reads a far larger
    array for a fraction of the time.

    So all five are built in one pass that stops one step early, keeping
    the expert-parallel axis, and the kernel closes them. Summing the last
    axis is free where it is finished: those tables live in scalar memory,
    `ep` is a build-time constant, and the sum is a fixed run of loads and
    adds with no loop.

    Two of the five used a dynamic index to pick this shard's own entry
    out of a per-destination vector. Here that is a mask instead, so the
    pick joins the same pass rather than standing as its own operation.
    `first` is what keeps such a per-destination total from being added
    once per local expert: it survives on one row and is zero on the rest.

    Two spellings of the same algebra are NOT the same cost, and the
    cheap one is not the obvious one.

    The rows are reduced first and joined afterwards. Stacking the five
    terms and reducing the stack materializes a [N_COUNTS, G, ep] array to
    read it once; reducing each term and joining the results concatenates
    five [1, ep] rows instead, which is a few dozen values. The compiler's
    own cost note prefers the second by more than the whole operation this
    is trying to remove.

    The block counts stay in ROW units here and the kernel divides. Every
    aligned run is a whole number of ROWBLK rows, so the quotient is the
    same taken before or after the sum, but taken here it is a floor
    division on a signed array, which carries its sign fixup on every
    element; taken in the kernel it is one scalar division by a build-time
    constant, on values that are already reduced.
    """
    g_local = e_total // ep
    all_run_rows = routing.run_rows  # [E, ep]
    true_rows = lax.dynamic_slice(
        all_run_rows, (me * g_local, 0), (g_local, ep)
    ).astype(jnp.int32)
    aligned = lax.dynamic_slice(
        routing.run_rows_aligned, (me * g_local, 0), (g_local, ep)
    ).astype(jnp.int32)
    mine = (jnp.arange(ep, dtype=jnp.int32) == me).astype(jnp.int32)  # [ep]
    other = 1 - mine
    first = (jnp.arange(g_local, dtype=jnp.int32) == 0).astype(jnp.int32)[
        :, None
    ]  # [G, 1]
    recv_rows_all = all_run_rows.sum(axis=0).astype(jnp.int32)  # [ep]
    recv_total = routing.recv_rows.astype(jnp.int32)  # [ep]
    all_tiles = routing.run_tiles.astype(jnp.int32)  # [E, ep]
    tiles = lax.dynamic_slice(all_tiles, (me * g_local, 0), (g_local, ep))
    recv_tiles_all = all_tiles.sum(axis=0)  # [ep]
    active = (expert_rows > 0).astype(jnp.int32)[:, None]  # [G, 1]

    def over_experts(term):
        """One count's reduction, keeping the expert-parallel axis."""
        return term.sum(axis=0, keepdims=True)  # [1, ep]

    return jnp.concatenate(
        [
            over_experts(true_rows * other[None, :]),
            over_experts(
                (recv_rows_all * mine)[None, :] * first - true_rows * mine[None, :]
            ),
            over_experts(aligned * other[None, :]),
            over_experts(
                (recv_total * mine)[None, :] * first - aligned * mine[None, :]
            ),
            over_experts(active * mine[None, :]),
            over_experts(tiles * other[None, :]),
            over_experts(
                (recv_tiles_all * mine)[None, :] * first - tiles * mine[None, :]
            ),
            over_experts(tiles * mine[None, :]),
        ],
        axis=0,
    ).astype(jnp.int32)  # [N_COUNTS, ep]


def vmem_limit(info=None):
    """VMEM budget for the kernel, read from this generation's capacity.

    Takes the device record the rest of the accounting already holds, so a
    caller sizing several candidate builds against one budget reads the
    device once rather than once per candidate.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    return int(info.vmem_capacity_bytes * VMEM_FRACTION)


# The oldest chip generation this kernel has been built and measured on.
# Everything below reads its geometry off the device record, so an earlier
# generation produces a kernel that compiles and runs and has never been
# correctness-checked or timed anywhere. The number is written down here
# rather than at the serving adapter, because it is a statement about what
# this kernel has been run on.
MIN_GENERATION = 7


def chip_generation(info=None):
    """The generation of the chip this build is for.

    Read off the same device record the VMEM accounting reads, so a host
    with no chip to name raises here the way it raises there, and one
    guard in the caller answers for both.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    return info.generation


def array_vmem_bytes(shape, dtype, info):
    """Bytes one VMEM array of this shape and dtype occupies.

    The minor dimension pads to the chip's lane count and the second-minor
    to this dtype's sublane tiling; both numbers come off the device record
    rather than being written down. A one-dimensional array is laid out as
    a single row of lanes. Padding is most of the difference between an
    array's element count and the memory it costs: a scale mirror four
    lanes wide still pays for a full row of lanes.
    """
    itemsize = jnp.dtype(dtype).itemsize
    if len(shape) == 1:
        return align_up(shape[0], info.num_lanes) * itemsize
    lanes = align_up(shape[-1], info.num_lanes)
    sublanes = align_up(shape[-2], info.get_sublane_tiling(dtype))
    return math.prod(shape[:-2]) * sublanes * lanes * itemsize


# A 32-bit array's second-minor dimension tiles to this many sublanes, so a
# packed four-bit k-block has to be a whole number of them deep. Written
# down rather than read off the device record, unlike everything else in
# this file's layout arithmetic, because the acceptance check that uses it
# answers before any device read and the serving adapter builds the same
# refusal string at import on a host with no chip attached.
# check_u32_sublane_tile is what keeps the literal honest.
U32_SUBLANE_TILE = 8


def fp4_scale_rows(k, rhs_qb, *, padded=False):
    """Rows one expert's four-bit block-scale table takes: K / rhs_qb blocks,
    rounded up to the f32 sublane tile when `padded`.

    The padded form is the "row-padded flat" scale operand: the table for E
    experts comes in as one 2-D [E * rows, N] f32 array rather than as
    [E, K / rhs_qb, N]. A device buffer of the 3-D shape is free to take a
    layout of its own -- torch_tpu stores f32 [48, 10, 5120] as {2,0,1}, the
    block axis major, so that 10 rows need not pad to 16 -- and the kernel's
    operand is {2,1,0}, so XLA re-lays the table out on every call. A 2-D
    buffer has only the one layout, and with whole sublane tiles per expert
    the [E, rows, N] view the kernel takes of it is a bitcast. The padded
    rows are zero and never read.
    """
    blocks = k // rhs_qb
    return align_up(blocks, U32_SUBLANE_TILE) if padded else blocks


def check_u32_sublane_tile(info=None):
    """Refuse a chip whose 32-bit sublane tiling is not the one the packed
    four-bit weight layout is written for.

    array_vmem_bytes sizes the packed w1_vm and w2_vm buffers off the device
    record and the four-bit acceptance check decides whether the block size
    that indexes them is addressable from the constant above. The two are
    load-bearing for one layout, so on a generation where they disagree the
    accounting adapts and the acceptance check does not, and the
    disagreement surfaces as a Mosaic slice error inside the kernel body
    with nothing naming the constant. This names it.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    queried = info.get_sublane_tiling(jnp.uint32)
    if queried != U32_SUBLANE_TILE:
        raise ValueError(
            f"the four-bit weight stream is packed as 32-bit words whose "
            f"second-minor dimension tiles to {U32_SUBLANE_TILE} sublanes, "
            f"and this chip tiles them to {queried}; the packed-weight row "
            f"tile the block size is checked against is written for "
            f"{U32_SUBLANE_TILE} and the buffers beside it are sized from "
            "the device record, so the two no longer describe one layout"
        )


# --------------------------------------------------------------------------- #
# K32Z: block-32 MXFP4 experts kept lossless (DeepSeek-V4.1). The operands are
# built by sglang_tpu/dsv4_mxfp4_k32z.py; this mirrors its layout.
#
# The e2m1 codes stream as packed u32 words exactly as FP4's do. Per output
# column n there is a base exponent b, the minimum over the column's K groups
# of 32 rows, and per (group g, column n) a shift d = e - b in a `bits`-wide
# uint8 field: byte (g, (T // F) * 128 + l) at bit bits * (T % F), for column
# tile T = n // 128, lane l = n % 128 and F = 8 // bits fields per byte. The
# base bytes ride in the down table's padding rows: row G2 + T % 2 holds the
# down bases of column tile T, row G2 + 2 + T % 2 the gate/up bases, both at
# byte columns (T // 2) * 128 + l. The kernel converts a block of codes to
# e5m2 (exact), adds 4 * d to every byte (d to the exponent field), contracts
# it and applies the column scale 2^(b - 127) once after the K sum.
# --------------------------------------------------------------------------- #
K32Z_GROUP = 32  # K rows per checkpoint exponent, so per shift field
K32Z_LANE_TILE = 128  # columns per shift-field tile
K32Z_U8_SUBLANE_TILE = 32  # a uint8 table pads its rows to this
K32Z_BASE_ROWS = 4  # down bases (even/odd tiles), then gate/up bases
K32Z_GATE_UP_BITS = (2, 4)  # the gate/up field widths a build takes
K32Z_DOWN_BITS = 4
# A decoded block is [K32Z_KBLOCK, K32Z_CHUNK] e5m2: one output chunk of 256
# columns over 512 contraction rows (the last block of a matmul may be 256).
K32Z_CHUNK = 256
K32Z_KBLOCK = 512
# K32Z tile heights are whole multiples of this (and so is a K32Z tile_m).
K32Z_RUNG_STEP = 32
# The tile heights a K32Z build emits below tile_m, which is always the top
# one. Each is another copy of the FFN body in the program, and a K32Z body
# (its decode unrolled over every block) is large: with all eight multiples
# of 32 up to 256 the program outgrew the TensorCore's instruction memory
# [hyp] and a visit took about 8x longer (one V4.1 layer: 1564
# vs 216 us per call at 8 rows per rank), while four bodies measured fine.
# 128 is the default ladder's streamed height (K32Z_STREAMED_ROWS): where a
# visit's slab waits move from the readers to its head. A ladder without it
# streams up to its tallest rung below 128 instead (k32z_streamed_rows).
K32Z_RUNGS = (32, 64, 128)
# The ladder production builds take; a build whose layer is traced with
# another K32Z_RUNGS gets it as k32z_rungs, keyed and named.
K32Z_DEFAULT_RUNGS = K32Z_RUNGS
# A rung body of up to this many rows waits each weight slab in the block
# that first reads it, as FP4 does. A visit whose first tile is taller waits
# both slabs at its head, before the tile loop, so its bodies hold no waits.
K32Z_STREAMED_ROWS = 128
# How a K32Z FFN body walks its K32Z_CHUNK-column output chunks. Unrolled
# (0), every chunk carries one decode per block, 280 at V4.1's shape (18
# gate/up chunks x 10 blocks, 20 down chunks x 5), and at decode sizes the
# program carried instruction-overlay waits. Looped (u > 0), the chunks run in
# a loop of u chunks a step through an f32 accumulator scratch (acc_vm): the
# same block products summed in the same order, so the same bits, from u
# chunks' decode text per matmul. A looped body exposes ~0.12 us per loop
# step, so only rungs a bucket rarely runs can afford it.
#
# A decode build -- every expert expected inside the first rung, ragged rows
# <= ep * g_local * K32Z_RUNGS[0] -- loops the rungs from K32Z_DECODE_ROLL_FROM
# up (the 128- and 256-row bodies, which a decode bucket runs only for a hot
# expert), walking K32Z_DECODE_STEP (gate/up, down) chunks a step (each must
# divide its matmul's chunk count, 18 / 20 at V4.1's shape; a shape the pair
# does not divide takes K32Z_DECODE_ROLL's count for both). Every other build
# keeps the unrolled bodies.
K32Z_DECODE_ROLL = 2
K32Z_DECODE_ROLL_FROM = 128
K32Z_DECODE_STEP = (3, 4)
# Larger builds drop the rung bodies their buckets rarely run (each rung is
# another copy of the FFN body in the program), and defer the return's drain
# and push into the next visit's first tile. Ragged rows > ep * g_local * rows
# per expert takes the tallest tier exceeded (entries replace the defaults):
# - > 128 rows per expert (1024 rows per rank): no 32 rung (64/128/256: three
#   bodies; the instruction-overlay wait per call is gone), the push-from-VMEM
#   bound counted in 128-row tiles (the two-hop return) and the deferred
#   drain/push.
# - > 256 rows per expert (2048): no 64 rung (32/128/256) and the deferred
#   drain/push.
K32Z_LARGE_PLAN = (
    (128, {"rungs": (64, 128), "push_from_vmem_rows": 128, "defer_push": True}),
    (256, {"rungs": (32, 128), "defer_push": True}),
)


def k32z_large_tier(rows_alloc, ep, g_local):
    """The K32Z_LARGE_PLAN entries a build of `rows_alloc` ragged rows takes:
    {} below every tier."""
    taken = {}
    for rows, entries in sorted(K32Z_LARGE_PLAN, key=lambda tier: tier[0]):
        if rows_alloc > ep * g_local * rows:
            taken = dict(entries)
    return taken


# The all-gather build serves K32Z at 2048 local rows per rank with these
# schedule choices (each leaves the output bits unchanged; every other build
# keeps the plain program). V4.1 Flash EP8 on TPU v7x, layer 3:
# - combine fused scales: the combine kernel runs the arrival-scale lookup
#   itself (mirror, mirror positions and router weights in scalar prefetch)
#   instead of an XLA gather of the row scales in front of it, where the
#   combine kernel applies and the scale mirror is at most one lane block
#   wide (K32Z's 128-lane mirror unit): -41 us.
# - side core: the two slab scatters run on SparseCore core 1, off core 0's
#   queue that holds XLA's token all-gather: -19 us. (Also moving the metadata
#   all-gather there lost 33 us: it overlaps the token all-gather on the
#   shared links.)
# - fold select: the combine zeroes the rows of tokens that routed nowhere
#   inside its kernel (one route flag per token in scalar prefetch) instead of
#   in a select after it, on every K32Z build the combine kernel applies to.
# - early ship selector and gather unroll, below.
ALL_GATHER_RECIPE_TOKENS = (2048,)
SC_SIDE_CORE_MODE = 2


def combine_fuse_scales(*, k32z, t_local):
    """Whether an all-gather build fuses the arrival-scale lookup into its
    combine kernel."""
    return bool(k32z) and t_local in ALL_GATHER_RECIPE_TOKENS


def sc_side_core(*, k32z, t_local):
    """The all-gather build's side-work mode on SparseCore core 1: 2 (the two
    slab scatters) for K32Z at 2048 rows per rank, 0 elsewhere."""
    return SC_SIDE_CORE_MODE if (k32z and t_local in ALL_GATHER_RECIPE_TOKENS) else 0


# Early shipping on a K32Z all-gather two-hop build at 2048 local rows: every
# own expert ships each (expert, dest) run two tiles after the tile that
# completes it (its commits are then landed) instead of all its runs at its
# visit's end. The same DMAs and totals. It is chosen per call inside one
# kernel (the kernel's early_ship_flag; a lax.cond between two kernels cost
# +15 to +20 us per call) from a plan model of the busiest rank's return tail:
# every rank visits its experts back to back, most rows first, a visit of R
# aligned rows taking max(W, A * R) us; visit k's runs leave after visit
# k + 1's first tile (defer_push); each directed ICI link of the 2 x 2 chip
# grid serves B us per true row in release order. X = the latest link finish
# minus the busiest rank's compute end; early shipping iff X > THETA. A / B /
# THETA were fitted on six layers' measurements (EP8, 2048 rows/rank, real
# text); W is the per-expert weight-stream floor.
K32Z_ES_A_US = 0.0402
K32Z_ES_B_US = 0.05553
K32Z_ES_W_US = 5.44
K32Z_ES_THETA_US = 171.5
# The link model follows the mesh's device placement: shard d sits on the chip
# of mesh device d, and a push routes x first over that chip's (x, y) grid
# position. Serving's mesh is not in chip order (shard 2 is on physical device
# 4), so the positions are read from the devices, not assumed from d // 2.
# Only an EP8 build on a 2 x 2 grid of four two-core chips selects.
K32Z_ES_GRID = 2


def early_ship_shard_xy(mesh):
    """Each shard's chip (x, y) on the 2 x 2 chip grid, from the mesh's devices
    (shard d = mesh index d), or None when the mesh is not eight shards on four
    chips, two per chip, in a 2 x 2 grid."""
    devices = list(mesh.devices.reshape(-1))
    if len(devices) != 2 * K32Z_ES_GRID**2:
        return None
    try:
        xy = [tuple(int(c) for c in d.coords)[:2] for d in devices]
    except (AttributeError, TypeError):
        return None
    if any(len(c) != 2 for c in xy):
        return None
    x0 = min(c[0] for c in xy)
    y0 = min(c[1] for c in xy)
    xy = tuple((x - x0, y - y0) for x, y in xy)
    grid = {(x, y) for x in range(K32Z_ES_GRID) for y in range(K32Z_ES_GRID)}
    if set(xy) != grid or any(xy.count(c) != 2 for c in grid):
        return None
    return xy


def early_ship_select(*, k32z, t_local, ep, shard_xy):
    """Whether an all-gather build picks early shipping per call: K32Z at 2048
    rows per rank on an EP8 mesh of four two-core chips (shard_xy is not None,
    from early_ship_shard_xy)."""
    return (
        bool(k32z)
        and t_local in ALL_GATHER_RECIPE_TOKENS
        and shard_xy is not None
        and ep == len(shard_xy)
    )


# The all-gather kernel's token-row gather loop is unrolled by hand, k blocks
# per step then a remainder loop (the kernel's gather_unroll; the same DMAs in
# the same order; the traced bounds rule out fori_loop's unroll=). K32Z at 2048
# local rows per rank unrolls 8, every other build 0 (off).
K32Z_GATHER_UNROLL = 8


def gather_unroll_mode(*, k32z, t_local):
    """The kernel's gather_unroll (0: off)."""
    return K32Z_GATHER_UNROLL if k32z and t_local in ALL_GATHER_RECIPE_TOKENS else 0


def early_ship_link_incidence(shard_xy):
    """[links][src][dst] 0 / 1: whether the push src -> dst crosses each
    directed ICI link of the 2 x 2 chip grid (x first; one chip's two shards
    and an own-rank hop cross none). shard_xy[d] is shard d's chip (x, y).
    Chips are numbered (0,0), (1,0), (0,1), (1,1) and links listed by chip
    pair, so the identity placement gives the table the model was fitted on."""
    ep = len(shard_xy)
    xy = tuple((x, y) for y in range(K32Z_ES_GRID) for x in range(K32Z_ES_GRID))
    chip_of = {c: i for i, c in enumerate(xy)}
    links = [
        (a, b)
        for a in range(len(xy))
        for b in range(len(xy))
        if abs(xy[a][0] - xy[b][0]) + abs(xy[a][1] - xy[b][1]) == 1
    ]
    inc = [[[0] * ep for _ in range(ep)] for _ in links]
    for s in range(ep):
        for d in range(ep):
            cur, dst = list(shard_xy[s]), tuple(shard_xy[d])
            for axis in (0, 1):
                while cur[axis] != dst[axis]:
                    nxt = list(cur)
                    nxt[axis] += 1 if dst[axis] > cur[axis] else -1
                    inc[links.index((chip_of[tuple(cur)], chip_of[tuple(nxt)]))][s][d] = 1
                    cur = nxt
    return inc


def early_ship_tail_us(
    run_rows, expert_rows_aligned, *, ep, g_local, tile_m, shard_xy
):
    """The selector's X in us (f32 scalar) from the replicated plan.

    run_rows [E, ep] true rows per (expert, dest shard); expert_rows_aligned
    [E]. No sort: the visit order is a g_local-square compare per shard and
    each link's finish a pairwise compare of the release times.
    """
    f32 = jnp.float32
    rows = expert_rows_aligned.reshape(ep, g_local).astype(f32)
    live = rows > 0
    dur = jnp.where(
        live, jnp.maximum(f32(K32Z_ES_W_US), f32(K32Z_ES_A_US) * rows), f32(0)
    )
    idx = jnp.arange(g_local, dtype=jnp.int32)
    r_g, r_h = rows[:, :, None], rows[:, None, :]
    # before[s, g, h]: shard s visits h before g (rows descending, then index).
    before = jnp.logical_and(
        jnp.logical_or(
            r_h > r_g,
            jnp.logical_and(r_h == r_g, idx[None, None, :] < idx[None, :, None]),
        ),
        live[:, None, :],
    )
    start = jnp.sum(jnp.where(before, dur[:, None, :], f32(0)), axis=2)
    pos = jnp.sum(before.astype(jnp.int32), axis=2)
    first = dur * jnp.minimum(rows, f32(tile_m)) / jnp.maximum(rows, f32(1))
    nxt = jnp.logical_and(pos[:, None, :] == pos[:, :, None] + 1, live[:, None, :])
    # defer_push: g's runs leave after the next visit's first tile; the last
    # visit's at the shard's compute end.
    rel = (start + dur + jnp.sum(jnp.where(nxt, first[:, None, :], f32(0)), axis=2))
    comp_end = jnp.max(jnp.sum(dur, axis=1))
    inc = jnp.asarray(early_ship_link_incidence(shard_xy), jnp.int32)  # [l, s, d]
    rr = run_rows.reshape(ep, g_local, ep).astype(jnp.int32)  # [s, g, d]
    w = jnp.sum(inc[:, :, None, :] * rr[None], axis=3).reshape(inc.shape[0], -1)
    rel = rel.reshape(-1)
    later = rel[None, :] >= rel[:, None]  # [i, j]: j released at or after i
    suffix = jnp.sum(jnp.where(later[None], w[:, None, :], 0), axis=2)  # [l, i]
    fin = jnp.where(
        w > 0, rel[None, :] + f32(K32Z_ES_B_US) * suffix.astype(f32), -jnp.inf
    )
    return jnp.max(fin) - comp_end


def early_ship_pick(
    run_rows, expert_rows_aligned, *, ep, g_local, tile_m, shard_xy
):
    """Mode 2's per-call choice: early shipping iff X > K32Z_ES_THETA_US."""
    return (
        early_ship_tail_us(
            run_rows,
            expert_rows_aligned,
            ep=ep,
            g_local=g_local,
            tile_m=tile_m,
            shard_xy=shard_xy,
        )
        > K32Z_ES_THETA_US
    )


# A K32Z push build at these local rows per rank takes the plan's per-pair slab
# rows from the exchange-table kernel (the push layer's slab-rows option). It
# applies on the exchange-table plan path without x_tables only (8-256 here;
# 512 / 1024 take x_tables). The same integers either way.
K32Z_PUSH_SLAB_ROWS_TOKENS = (8, 16, 32, 64, 128, 256)


def push_slab_rows_mode(*, k32z, t_local):
    """Whether a push build takes the plan's slab rows from the table kernel."""
    return bool(k32z) and t_local in K32Z_PUSH_SLAB_ROWS_TOKENS


def early_ship_flag_row(pick, ep):
    """The count table's COUNT_ES_FLAG row, [1, ep] i32: the pick in lane 0."""
    lane0 = (jnp.arange(ep, dtype=jnp.int32) == 0).astype(jnp.int32)
    return (pick.astype(jnp.int32) * lane0)[None, :]


def k32z_streamed_rows(height_rungs):
    """The tallest rung a K32Z build streams its weight slabs into: the
    tallest of `height_rungs` up to K32Z_STREAMED_ROWS (0 where none is).
    A visit whose rows exceed it waits both slabs at its head; bodies up to
    it wait each slab in the block that first reads it. The default ladder
    gives K32Z_STREAMED_ROWS itself."""
    return max((r for r in height_rungs if r <= K32Z_STREAMED_ROWS), default=0)


def k32z_step_applies(step, hidden, inter):
    """False only where a (gate/up, down) loop step of positive counts does
    not divide its matmul's output chunk count at this shape; anything else
    goes on to check_k32z_ffn_roll, which refuses a bad pair by name."""
    up, down = (int(u) for u in step)
    if up <= 0 or down <= 0:
        return True
    return (2 * inter // K32Z_CHUNK) % up == 0 and (hidden // K32Z_CHUNK) % down == 0


def k32z_roll_plan(rows_alloc, ep, g_local, hidden=None, inter=None):
    """(k32z_ffn_roll, k32z_roll_from) a K32Z build takes at trace time: decode
    builds loop their bodies from K32Z_DECODE_ROLL_FROM up, walking
    K32Z_DECODE_STEP chunks a step where the shape (`hidden`, `inter`)
    divides its chunk counts, K32Z_DECODE_ROLL otherwise; every other build
    keeps the unrolled bodies."""
    if rows_alloc <= ep * g_local * K32Z_DEFAULT_RUNGS[0]:
        step = tuple(int(u) for u in K32Z_DECODE_STEP)
        if hidden is None or k32z_step_applies(step, hidden, inter):
            return (step if step[0] != step[1] else step[0]), K32Z_DECODE_ROLL_FROM
        return K32Z_DECODE_ROLL, K32Z_DECODE_ROLL_FROM
    return 0, 0


def k32z_roll_steps(roll):
    """(gate/up, down) output chunks per loop step of a looped K32Z body:
    `roll` is one count for both or a (gate/up, down) pair."""
    if isinstance(roll, tuple):
        return int(roll[0]), int(roll[1])
    return int(roll), int(roll)


def k32z_roll_tag(roll):
    """`roll` as it appears in a program name: "2", or "3x4" for a pair."""
    if isinstance(roll, tuple):
        return f"{int(roll[0])}x{int(roll[1])}"
    return str(int(roll))


def check_k32z_ffn_roll(roll, hidden, inter):
    """`roll` as an int (or a (gate/up, down) pair of ints) a K32Z
    build at this shape can take; raises naming why not. Each matmul's chunk
    count must be a whole multiple of its count."""
    pair = isinstance(roll, (tuple, list))
    parts = tuple(roll) if pair else (roll,)
    if (
        len(parts) not in ((2,) if pair else (1,))
        or any(isinstance(u, bool) or int(u) != u or u < 0 for u in parts)
        or (pair and not all(int(u) > 0 for u in parts))
    ):
        raise ValueError(
            f"k32z_ffn_roll is a chunk count >= 0 or a (gate/up, down) pair of "
            f"counts > 0; got {roll!r}"
        )
    steps = tuple(int(u) for u in parts) * (1 if pair else 2)
    for (what, n_out), u in zip((("gate/up", 2 * inter), ("down", hidden)), steps):
        if u and (n_out // K32Z_CHUNK) % u:
            raise ValueError(
                f"k32z_ffn_roll {roll} does not divide the {what} matmul's "
                f"{n_out // K32Z_CHUNK} output chunks"
            )
    if pair:
        return steps if steps[0] != steps[1] else steps[0]
    return steps[0]


# The K32Z build choices k32z_build_plan returns, with the builders' defaults.
K32Z_PLAN_DEFAULTS = {
    "k32z_rungs": None,
    "k32z_ffn_roll": 0,
    "k32z_roll_from": 0,
    "k32z_roll_rungs": None,
    "push_from_vmem_rows": None,
    "defer_push": False,
}


def k32z_build_plan(rows_alloc, ep, g_local, hidden=None, inter=None):
    """The K32Z build choices a layer takes at trace time, as the builders'
    keyword arguments (keys of K32Z_PLAN_DEFAULTS): the decode roll
    (k32z_roll_plan), then, for a large build, its K32Z_LARGE_PLAN tier's
    entries. k32z_rungs is None wherever the ladder is the default one, so a
    default build keeps its name."""
    roll, roll_from = k32z_roll_plan(rows_alloc, ep, g_local, hidden, inter)
    plan = dict(K32Z_PLAN_DEFAULTS)
    plan["k32z_ffn_roll"], plan["k32z_roll_from"] = roll, roll_from
    for key, value in k32z_large_tier(rows_alloc, ep, g_local).items():
        if key == "rungs":
            rungs = tuple(int(r) for r in value)
            plan["k32z_rungs"] = None if rungs == K32Z_DEFAULT_RUNGS else rungs
        else:
            plan[key] = value
    return plan


class K32ZTableGeometry(NamedTuple):
    """One expert's K32Z shift tables: (rows, byte columns) of each matmul's
    uint8 table, the K groups each covers and the gate/up field width."""

    g1: int
    g2: int
    rows1: int
    cols1: int
    rows2: int
    cols2: int
    gate_up_bits: int

    @property
    def tab_rows(self):
        """Rows of the decode's shift-word buffer: the taller table's groups."""
        return align_up(max(self.g1, self.g2), U32_SUBLANE_TILE)


def k32z_table_geometry(hidden, inter, gate_up_bits=K32Z_GATE_UP_BITS[0]):
    """The K32Z tables for an expert shape; raises naming what it cannot take.

    Blocks and chunks are whole 256s on both matmuls, which also makes every
    shift-field tile of a 256-column chunk sit in one packed byte column. The
    gate/up bases take inter byte columns of the down table's padding rows.
    """
    problems = []
    if gate_up_bits not in K32Z_GATE_UP_BITS:
        problems.append(
            f"the gate/up shift field is {K32Z_GATE_UP_BITS} bits wide, "
            f"not {gate_up_bits}"
        )
    if hidden % K32Z_CHUNK or inter % K32Z_CHUNK:
        problems.append(
            f"hidden={hidden} and inter={inter} must be multiples of "
            f"{K32Z_CHUNK}: both matmuls decode whole {K32Z_CHUNK}-column "
            f"chunks of {K32Z_KBLOCK}-row blocks with a {K32Z_CHUNK}-row tail"
        )
    if inter > hidden * K32Z_DOWN_BITS // 8:
        problems.append(
            f"the gate/up bases need {inter} byte columns of the down table, "
            f"which has {hidden * K32Z_DOWN_BITS // 8}"
        )
    if problems:
        raise ValueError("K32Z cannot lay out these experts: " + "; ".join(problems))
    g1, g2 = hidden // K32Z_GROUP, inter // K32Z_GROUP
    return K32ZTableGeometry(
        g1=g1,
        g2=g2,
        rows1=align_up(g1, K32Z_U8_SUBLANE_TILE),
        cols1=2 * inter * gate_up_bits // 8,
        rows2=align_up(g2 + K32Z_BASE_ROWS, K32Z_U8_SUBLANE_TILE),
        cols2=hidden * K32Z_DOWN_BITS // 8,
        gate_up_bits=gate_up_bits,
    )


def k32z_tile_height_rungs(tile_m, rungs=None):
    """The ascending tile heights a K32Z build emits: `rungs` (K32Z_RUNGS
    where None) below tile_m, then tile_m. Always all of them, whatever the
    expected expert size: an expert taller than a tile re-decodes per tile
    and its tail tile takes the rung that covers it. A ladder without
    K32Z_STREAMED_ROWS streams up to its tallest rung below it instead
    (k32z_streamed_rows); the kernels read that, not the constant."""
    rungs = K32Z_RUNGS if rungs is None else tuple(rungs)
    if rungs != tuple(sorted(set(rungs))) or any(
        r <= 0 or r % K32Z_RUNG_STEP for r in rungs
    ):
        raise ValueError(
            f"K32Z rungs {rungs} must ascend in positive multiples of {K32Z_RUNG_STEP}"
        )
    return tuple(r for r in rungs if r < tile_m) + (tile_m,)


def vmem_scratch_arrays(
    g_local,
    capacity,
    hidden,
    inter,
    *,
    nbuf,
    weight_format=WeightFormat.FP8,
    rhs_qb=QB4,
    has_w1_bias=False,
    has_w2_bias=False,
    pad_scale_rows=False,
    k32z_gate_up_bits=K32Z_GATE_UP_BITS[0],
    k32z_ffn_roll=0,
    mirror_unit=0,
):
    """The kernel's VMEM scratch buffers, as (name, shape, dtype).

    _build_fused_ep_moe_kernel declares its scratch from this list, in this
    order, and the VMEM accounting below sums this same list, so a buffer
    cannot be resized on one side only.

    `pad_scale_rows` sizes the four-bit scale slots for the row-padded flat
    scale operand (fp4_scale_rows). The accounting leaves it at its default:
    an f32 slot's rows tile to U32_SUBLANE_TILE sublanes either way, so the
    two forms cost the same VMEM. `k32z_gate_up_bits` sizes K32Z's gate/up
    shift table and is ignored by every other format, as is `k32z_ffn_roll`
    (the looped body), which adds its accumulator.

    Every element type here comes off the weight format's record: the
    weight slabs, the row staging the activations arrive in, the wire
    buffer results leave in, and whether the scale tables and the
    activation row scale exist at all.

    The staging buffers are three-dimensional on purpose. An eight-bit
    array's second-minor dimension tiles to 32 sublanes, so a flat
    [rows, hidden] buffer could only be sliced at row offsets that are
    multiples of 32; splitting a token row across the two minor dimensions
    leaves the row axis untiled, which is what makes an 8-row transport
    offset legal. It does that for every element type, which is why a
    sixteen-bit wire takes the same geometry at twice the bytes.
    """
    form = weight_form(weight_format)
    lane_blocks = row_lane_blocks(hidden)
    # The tile height IS the capacity, and the out pair is in tile units.
    tile_m = capacity
    # A scale table rides the weight slots, not the expert list: it is read
    # only alongside the slab it scales, so it is staged with that slab and
    # is nbuf deep.
    if weight_format == WeightFormat.FP4:
        # Four-bit weights stream as PACKED u32 words ([K/8, N]), so the
        # transfer moves half the bytes an eight-bit slab would.
        weights = [
            ("w1_vm", (nbuf, hidden // PACK4, 2 * inter), jnp.uint32),
            ("w2_vm", (nbuf, inter // PACK4, hidden), jnp.uint32),
            (
                "w1s_vm",
                (
                    nbuf,
                    fp4_scale_rows(hidden, rhs_qb, padded=pad_scale_rows),
                    2 * inter,
                ),
                jnp.float32,
            ),
            (
                "w2s_vm",
                (nbuf, fp4_scale_rows(inter, rhs_qb, padded=pad_scale_rows), hidden),
                jnp.float32,
            ),
        ]
    elif weight_format == WeightFormat.K32Z:
        geo = k32z_table_geometry(hidden, inter, k32z_gate_up_bits)
        weights = [
            # The codes stream packed exactly as FP4's do.
            ("w1_vm", (nbuf, hidden // PACK4, 2 * inter), jnp.uint32),
            ("w2_vm", (nbuf, inter // PACK4, hidden), jnp.uint32),
            ("w1s_vm", (nbuf, geo.rows1, geo.cols1), jnp.uint8),
            ("w2s_vm", (nbuf, geo.rows2, geo.cols2), jnp.uint8),
            # The shift words of one 256-column chunk ([column tile in the
            # chunk, group, lane], d * 0x04040404), double-buffered on the
            # chunk's parity so the next chunk's build is not ordered behind
            # the current chunk's last read.
            (
                "tab_vm",
                (2, K32Z_CHUNK // K32Z_LANE_TILE, geo.tab_rows, K32Z_LANE_TILE),
                jnp.int32,
            ),
            # The column scales 2^(b - 127): row 0 gate|up, row 1 down. Filled
            # and read by the same tile, so one deep.
            ("col_vm", (U32_SUBLANE_TILE, max(2 * inter, hidden)), jnp.float32),
        ]
        if k32z_ffn_roll:
            # The looped body's chunk results: the gate/up accumulator, then
            # the down one in the same rows (the looped body).
            weights.append(("acc_vm", (capacity, max(2 * inter, hidden)), jnp.float32))
    else:
        weights = [
            ("w1_vm", (nbuf, hidden, 2 * inter), form.weight_dtype),
            ("w2_vm", (nbuf, inter, hidden), form.weight_dtype),
        ]
        if form.has_scales:
            weights += [
                ("w1s_vm", (nbuf, 2 * inter), jnp.float32),
                ("w2s_vm", (nbuf, hidden), jnp.float32),
            ]
    # The optional expert biases are resident for every local expert, one
    # row per expert on each matmul's output channels. They carry no block
    # dimension, so they take the same shape on every weight format -- the
    # shape and the cost of the per-channel scale tables beside them.
    biases = []
    if has_w1_bias:
        biases.append(("w1b_vm", (g_local, 2 * inter), jnp.float32))
    if has_w2_bias:
        biases.append(("w2b_vm", (g_local, hidden), jnp.float32))
    # The activation row scale exists only where the rows were quantized.
    # It rides per tile, so it is sized by the OUT-PARITY count: a tile's
    # stream is issued into, waited on and read at the parity its results
    # will be staged in, and never at any other index. nbuf is the depth the
    # WEIGHT slabs need so a refill and the live readers of the prefetch
    # distance's worth of earlier experts occupy distinct slots, which is a
    # different question and a larger answer.
    act_scale = (
        [
            (
                "ls_vm",
                (OUT_PARITIES, act_scale_window_rows(capacity), HIDDEN_LANE_BLOCK),
                jnp.float32,
            )
        ]
        if form.quantized_activations
        else []
    )
    unit = mirror_lanes(capacity, mirror_unit)
    oscl_shape = (
        (OUT_PARITIES, capacity // unit, 1, unit)
        if unit < capacity
        else (OUT_PARITIES, mirror_split(capacity), capacity // mirror_split(capacity))
    )
    return [
        # The indirect form keeps each token row as one lane-block row.
        # Parity-deep for the same reason ls_vm is.
        (
            "lhs_vm",
            (OUT_PARITIES, capacity, lane_blocks, HIDDEN_LANE_BLOCK),
            form.act_dtype,
        ),
        *weights,
        *biases,
        *act_scale,
        (
            "out_vm",
            (OUT_PARITIES, tile_m, lane_blocks, HIDDEN_LANE_BLOCK),
            form.wire_dtype,
        ),
        # One tile's scales: one mirror sublane, split as the mirrors are;
        # with a mirror unit below the tile, one [1, unit] row per unit, so
        # each is a DMA source of its own.
        ("oscl_vm", oscl_shape, jnp.float32),
    ]


def vmem_tile_body_arrays(
    capacity,
    hidden,
    inter,
    *,
    weight_format=WeightFormat.FP8,
    rhs_qb=QB4,
    k32z_ffn_roll=0,
):
    """What one tile body keeps live, as (name, shape, dtype).

    These are not declared scratch: the compiler places them, in the same
    memory as the buffers above. In order, the first matmul's accumulator,
    the bf16 intermediate the activation chunk loop builds up before it is
    concatenated, that concatenation -- as an fp8 requantization where the
    format has one and as a second bf16 array where it does not -- the
    second matmul's accumulator, its bf16 result, and the wire row that
    result is quantized into where the wire is eight-bit.

    A format whose weights reach the matrix unit through a widening adds one
    widened contraction chunk, double-buffered; the two matmuls widen in
    turn and never overlap, so the peak is the wider of the two. Four-bit
    weights widen a scale block to fp8, integer weights a fixed contraction
    chunk to bf16. K32Z decodes one [K32Z_KBLOCK, K32Z_CHUNK] block at a
    time (its output chunk loop is outermost), so it counts that block and
    the shift words added to it, double-buffered, rather than a whole row of
    blocks.

    The row staging a tile reads is a view of lhs_vm, not a second array,
    so it is not counted again here; nor is a looped K32Z body's first
    accumulator (`k32z_ffn_roll`), which is the declared acc_vm.
    """
    form = weight_form(weight_format)
    tile_m = capacity
    arrays: list[tuple[str, tuple[int, ...], Any]] = [
        ("acc1", (tile_m, 2 * inter), jnp.float32),
        ("mid_chunks", (tile_m, inter), jnp.bfloat16),
    ]
    if weight_format == WeightFormat.K32Z and k32z_ffn_roll:
        arrays = arrays[1:]
    # The second matmul takes the intermediate quantized only where its
    # weights are. The other formats contract the bf16 rows directly -- but
    # they still materialize a second copy: the non-requantizing path
    # concatenates the chunk list into one [tile_m, inter] bf16 array while
    # the chunks that feed it are still live. The requantizing path does not,
    # because it quantizes straight out of the chunk list into mid_q.
    if form.quantized_activations:
        arrays.append(("mid_q", (tile_m, inter), FP8))
    else:
        arrays.append(("mid_concat", (tile_m, inter), jnp.bfloat16))
    arrays += [
        ("acc2", (tile_m, hidden), jnp.float32),
        ("down_bf16", (tile_m, hidden), jnp.bfloat16),
    ]
    # An eight-bit wire quantizes the down projection into a fresh array
    # while down_bf16, the bf16 source it reduces over, is still live. A
    # sixteen-bit wire ships down_bf16 itself and materializes nothing.
    if form.wire_dtype is FP8:
        arrays.append(("wire_rows", (tile_m, hidden), FP8))
    if weight_format == WeightFormat.FP4:
        arrays.append(
            (
                "widened_weight_block",
                (WIDENED_BLOCK_BUFFERS, rhs_qb, max(2 * inter, hidden)),
                FP8,
            )
        )
    elif weight_format == WeightFormat.INT8:
        arrays.append(
            (
                "widened_weight_block",
                (WIDENED_BLOCK_BUFFERS, WIDEN_KCHUNK, max(2 * inter, hidden)),
                BF16,
            )
        )
    elif weight_format == WeightFormat.K32Z:
        arrays += [
            (
                "decoded_weight_block",
                (WIDENED_BLOCK_BUFFERS, K32Z_KBLOCK, K32Z_CHUNK),
                jnp.float8_e5m2,
            ),
            (
                "shift_words",
                (WIDENED_BLOCK_BUFFERS, K32Z_KBLOCK // 4, K32Z_CHUNK),
                jnp.int32,
            ),
        ]
    return arrays


def vmem_estimate_bytes(
    g_local,
    capacity,
    hidden,
    inter,
    nbuf,
    weight_format=WeightFormat.FP8,
    rhs_qb=QB4,
    has_w1_bias=False,
    has_w2_bias=False,
    info=None,
    k32z_gate_up_bits=K32Z_GATE_UP_BITS[0],
    k32z_ffn_roll=0,
    mirror_unit=0,
):
    """VMEM one built kernel occupies, from the arrays it declares.

    An upper bound, and deliberately so: the declared scratch is live for
    the whole call, while the tile body's values are counted as though all
    of them were live at once even though several die before the next is
    born. Nothing is left out, which is what a caller asking "will this
    fit" needs; a caller wanting the exact high-water mark should read the
    figure the compiler reports for a built kernel.

    Reads the chip's lane count and per-dtype sublane tiling off the device
    record, so a host with no chip to name raises rather than answering.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    arrays = vmem_scratch_arrays(
        g_local,
        capacity,
        hidden,
        inter,
        nbuf=nbuf,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        k32z_gate_up_bits=k32z_gate_up_bits,
        k32z_ffn_roll=k32z_ffn_roll,
        mirror_unit=mirror_unit,
    )
    arrays += vmem_tile_body_arrays(
        capacity,
        hidden,
        inter,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        k32z_ffn_roll=k32z_ffn_roll,
    )
    return sum(array_vmem_bytes(shape, dtype, info) for _, shape, dtype in arrays)


class WeightSlots(NamedTuple):
    """The weight-slot schedule one build runs at, and what it costs.

    `distance` is how far ahead of the computing expert a refill goes out,
    `nbuf` the slots that distance needs, `vmem_bytes` what the whole build
    comes to at that depth and `limit` the budget it was measured against.
    """

    distance: int
    nbuf: int
    vmem_bytes: int
    limit: int

    @property
    def fits(self):
        """Whether the build at this depth is inside the budget."""
        return self.vmem_bytes <= self.limit


def fit_weight_slots(
    g_local,
    capacity,
    hidden,
    inter,
    *,
    max_distance=MAX_WEIGHT_PREFETCH_DISTANCE,
    weight_format=WeightFormat.FP8,
    rhs_qb=QB4,
    has_w1_bias=False,
    has_w2_bias=False,
    info=None,
    limit=None,
    k32z_gate_up_bits=K32Z_GATE_UP_BITS[0],
    k32z_ffn_roll=0,
    mirror_unit=0,
):
    """The deepest prefetch schedule this shape's VMEM leaves room for.

    The weight slabs are the only buffer sized by the schedule rather than
    by the model: they are `nbuf` copies of one expert, and `nbuf` is ours
    to pick. So a shape that misses at the ceiling often fits one slot
    shallower. Walk down a slot at a time rather than dropping to zero --
    even distance one keeps a refill in flight, which is most of the win.

    Distance zero has nothing left to give back. A shape still over budget
    there is returned NOT FITTING rather than raised on, carrying the
    distance-zero figures for the builder to quote in its refusal.

    Queries the device record once and reuses it for every candidate.
    """
    if max_distance < 0:
        raise ValueError(
            f"the maximum prefetch distance is {max_distance}; a negative "
            "distance is not a schedule, and zero already means no "
            "prefetching"
        )
    if info is None:
        info = pltpu.get_tpu_info()
    if limit is None:
        limit = vmem_limit(info)
    slots = None
    for distance in range(max_distance, -1, -1):
        nbuf = nbuf_for(distance)
        slots = WeightSlots(
            distance,
            nbuf,
            vmem_estimate_bytes(
                g_local,
                capacity,
                hidden,
                inter,
                nbuf=nbuf,
                weight_format=weight_format,
                rhs_qb=rhs_qb,
                has_w1_bias=has_w1_bias,
                has_w2_bias=has_w2_bias,
                info=info,
                k32z_gate_up_bits=k32z_gate_up_bits,
                k32z_ffn_roll=k32z_ffn_roll,
                mirror_unit=mirror_unit,
            ),
            limit,
        )
        if slots.fits:
            break
    # The first depth that fits, or distance zero over budget for the
    # caller to refuse by name.
    assert slots is not None
    return slots
