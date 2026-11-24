# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/fused_moe/v2/kernel.py
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
"""Fused expert-parallel MoE kernel: one Pallas TPU kernel computes the
expert half of an MoE layer for one expert-parallel shard and pushes each
result row to the shard that owns its token, so the layer needs no dense
combine reduce-scatter."""

import functools
import threading
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu._logging import get_logger as init_logger

# The layout constants this kernel is built against, and the host module
# whose accounting decides what it may declare.
from sglang_tpu.moe.pallas.fused_moe_v2 import host

# The clamped activations are copied from vLLM-TorchTPU's grouped-matmul
# kernel (kernels/megablox/gmm_v2.py) into activations.py.
from sglang_tpu.moe.pallas.fused_moe_v2.activations import (
    silu_and_mul_with_clamp,
    swigluoai,
)
from sglang_tpu.moe.pallas.fused_moe_v2.host import (
    FP4,
    FP8,
    FP8_MAX,
    HIDDEN_LANE_BLOCK,
    OUT_PARITIES,
    PACK4,
    QB4,
    ROWBLK,
)

logger = init_logger(__name__)

DN = (((1,), (0,)), ((), ()))
# The single-axis mesh name the layer's shard_map uses.
AXIS = "d"
# Columns of the intermediate the requantization does at a time. Chunking
# it keeps the live bf16 intermediate down; the width is a tuned choice.
QCHUNK = 512
# Output columns the four-bit FFN contracts at a time.
FP4_OUTPUT_CHUNK = 256
# How many tile heights the FFN body is emitted at, where it is emitted at
# more than one. A tile runs the smallest rung that covers its live rows, so
# an expert's short tail tile stops computing a full tile_m of rows nothing
# will commit. Every rung is another copy of the body in the program text --
# which is why this is a handful of heights and not one per ROWBLK -- and the
# top rung is always tile_m, so a full tile is unchanged.
HEIGHT_RUNGS = 4
# DMA priority the weight refills take, so that they stay off the in-order
# queue the token gather issues on.
WEIGHT_DMA_PRIORITY = 1
# The collective this kernel's barrier and remote copies belong to. Shared
# with any other collective kernel built with the same id.
COLLECTIVE_ID = 0
# The jax whose private ref-level bitcast the four-bit weight stream binds
# to. Recorded so the failure names the version this was built against
# rather than only the version that broke it.
_REF_BITCAST_JAX_VERSION = "0.11.2"
# The scalar-prefetch operands, in the order the kernel body unpacks them.
# The grid spec is told the total, so a table cannot be added on one side
# without the other.
# These are what the KERNEL takes, which is no longer everything the
# builders return: the two totals and the visit count are rows of the
# count table instead, and the push source is the same table as the
# commit offset, so it rides once. The builders' own signatures are
# unchanged and the values the kernel stopped taking leave the program
# with the rest of the dead code.
N_TRANSPORT_TABLES = 4  # of the seven shard_transport_tables_in_blocks has
N_PUSH_TABLES = 1  # of the three shard_push_tables_in_rows has
N_SLAB_TABLES = 2  # what shard_expert_slabs returns: (rows, base)
N_VISIT_TABLES = 1  # the visit list; its count is a row of the table below
N_COUNT_TABLES = 1  # shard_count_vector: [N_COUNTS, ep] i32
# This shard's index along the expert-parallel axis, as an i32[1] operand.
# Upstream reads it with `lax.axis_index`, which lowers to `partition-id`;
# SGLang reaches this kernel through `sglang_tpu.pallas_jax_op`'s `jax.export`
# bridge, which torch_tpu recompiles and whose SPMD partitioner rejects that
# instruction ("partitionId instruction is not supported for SPMD
# partitioning"). vLLM-TorchTPU passes the index as data for the same reason.
N_RANK_TABLES = 1
N_PREFETCH = (
    N_TRANSPORT_TABLES
    + N_PUSH_TABLES
    + N_SLAB_TABLES
    + N_VISIT_TABLES
    + N_COUNT_TABLES
    + N_RANK_TABLES
)
# The HBM operands, in the order the kernel body unpacks them. Only these
# two are on every build: the rest -- the activation row scale, the two
# weight scale tables, the two expert bias tables -- are there where the
# weight format and the model carry them, and each absent one has to drop
# out of the MIDDLE of the tuple, leaving the operands after it in place.
N_STAGING_INPUTS = 2  # token-gather table and ungathered token buffer
N_WEIGHT_SLABS = 2  # the gate/up and down weight slabs
N_SCALE_TABLES = 2  # a weight scale table per matmul, where the format has
#                     them

# The FFN activations this kernel can fuse. "silu" is the default and the
# only one the kernel carried before the selector existed; "swigluoai" is
# the clamped GPT-OSS form and "silu_and_mul_with_clamp" from DeepSeek-V4
ACT_FNS = ("silu", "swigluoai", "silu_and_mul_with_clamp")
def _apply_act(gate, up, act_fn):
    """The FFN activation, on the POST-SCALE gate and up halves.

    Both weight forms reach here with the accumulator already multiplied
    by its scales, which the clamped forms require: their clip is defined
    on the true activation value, not on a raw accumulator.
    """
    if act_fn == "silu":
        return jax.nn.silu(gate) * up
    if act_fn == "swigluoai":
        return swigluoai(gate, up)
    if act_fn == "silu_and_mul_with_clamp":
        return silu_and_mul_with_clamp(gate, up)
    raise NotImplementedError(
        f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}"
    )


def tile_height_rungs(tile_m, rows_alloc, g_local, ep):
    """The ascending static tile heights a build emits, ending at tile_m.

    Equally spaced so that a tail tile's rounding-up waste is bounded by one
    step. A tile height must be a whole number of ROWBLK -- it is the unit
    every commit offset is in -- so a tile_m that does not divide into whole
    ROWBLK steps keeps the single full height it had.

    Only a TAIL tile is short, so the rungs pay in proportion to the share of
    tiles that are tails, while their cost -- a copy of the FFN body in the
    program text per rung -- is paid by every tile. A build gets them only
    where a routed expert is expected to fit inside one tile, which is where
    every tile is the tail: `rows_alloc` bounds the shard slab for the case
    of every routed row landing on one shard, so `ep` shards' worth of it is
    what a shard's `g_local` experts actually expect to hold.

    """
    rungs = HEIGHT_RUNGS
    step = tile_m // rungs
    if (
        rungs < 2
        or step % ROWBLK
        or step * rungs != tile_m
        or rows_alloc > ep * g_local * tile_m
    ):
        return (tile_m,)
    return tuple(range(step, tile_m + 1, step))


def _row_scale(amax):
    """The row scale and its inverse, with the rows that have no usable one.

    A row whose maximum is zero has nothing to scale by, and a row whose
    maximum is not finite has nothing USEFUL to scale by: the reciprocal of
    an infinite maximum is zero, so every finite value in that row would
    multiply to zero and only the infinity would survive, as a NaN. That
    leaves a plausible-looking mostly-zero row where the row's real content
    was, and sends an infinite scale over the wire for the combine at the
    destination shard to multiply back in. Both cases are answered the same
    way, by the caller zeroing the row whole against `usable`: an obviously
    empty row rather than a quietly wrong one.
    """
    scale = amax.astype(jnp.float32) / FP8_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    return scale, sinv, usable


def rowquant_fp8(x):
    """Per-row dynamic fp8 quant, reducing and applying in bf16."""
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale, sinv, usable = _row_scale(amax)
    return _quant_apply(x, sinv.astype(x.dtype), usable), scale


def _absmax_rows(x):
    return jnp.max(jnp.abs(x), axis=-1, keepdims=True)


def _quant_apply(x, sinv_native, usable):
    """Quantize a row, or zero it whole where its scale is unusable."""
    return jnp.where(usable, x * sinv_native, jnp.zeros_like(x)).astype(FP8)


def wire_row_and_scale(
    acc2, mid_scale, *, w2s=None, w2b=None, wire_dtype=FP8, tile_m=None
):
    """One tile's rows as they go on the wire, and the scale that rides with
    them.

    The tile epilogue: apply the intermediate row scale, apply the
    per-channel weight scale where the format has one, add the down bias
    where the model carries one, and quantize for the wire where the wire is
    eight-bit. Every line is plain jnp on values, so it lives here rather
    than inside the Pallas body -- the down bias is the only operand
    placement in the layer with no host-side witness otherwise, and its
    add-once property is the one correctness claim the biases commit makes.

    mid_scale is None for the formats whose second matmul took bf16 rows and
    whose accumulator is therefore already the true value. w2s is the
    per-channel weight scale the four-bit path has already folded into its
    block sums and passes as None. w2b is the down bias, which lands on the
    POST-SCALE row and before the wire quantization: expert parallelism
    shards the second matmul on the expert axis rather than on its
    contraction, so a routed row's whole down projection is computed in one
    place and this add happens exactly once for it. The destination's
    per-slot router weight then weights the bias per selected expert, which
    is what the sum over selected experts wants.

    An unquantized wire carries the row as it is, against a scale mirror of
    exactly one, so the destination combine reads one transport for every
    format and multiplying by it changes no value. The mirror is 16 bytes
    per row against a row of 2 * hidden, so keeping it costs well under one
    percent of the wire.
    """
    down = acc2 if mid_scale is None else acc2 * mid_scale
    if w2s is not None:
        down = down * w2s
    if w2b is not None:
        down = down + w2b
    down_bf16 = down.astype(jnp.bfloat16)
    if wire_dtype is FP8:
        return rowquant_fp8(down_bf16)
    rows = tile_m if tile_m is not None else down_bf16.shape[0]
    return down_bf16, jnp.ones((rows, 1), jnp.float32)


def _intermediate_rows(
    acc1,
    act_scale,
    inter,
    *,
    w1s,
    act_fn,
    w1b,
    requantize=True,
    logical_intermediate_size=None,
):
    """acc1's gate and up halves through the activation, as the second
    matmul's left-hand side.

    Returns that left-hand side and the row scale the caller still owes what
    the second matmul produces. Every weight format shares this; what
    differs is what the halves owe before the activation and whether the
    result goes back through a quantization on the way out.

    act_scale is the activation row scale a quantized activation owes its
    accumulator, and is None where the rows were never quantized. w1s is the
    per-channel weight scale the eight-bit and integer paths still owe their
    halves, which the four-bit path has already folded into acc1 and passes
    as None. requantize is False for the formats whose second matmul takes
    bf16 rows directly; those return a row scale of None, because the
    accumulator they produce is already the true value.
    """

    def half(lo, hi):
        """One post-scale half of the accumulator, columns [lo, hi)."""
        scaled = acc1[:, lo:hi]
        if act_scale is not None:
            scaled = scaled * act_scale
        if w1s is not None:
            scaled = scaled * w1s[:, lo:hi]
        return scaled

    active_inter = (
        inter if logical_intermediate_size is None else logical_intermediate_size
    )
    mid_chunks, amax = [], None
    for c0 in range(0, active_inter, QCHUNK):
        c1 = min(c0 + QCHUNK, active_inter)
        gate = half(c0, c1)
        up = half(inter + c0, inter + c1)
        # The gate and up biases land on the POST-SCALE halves, before the
        # activation: the clamped form's clip is defined on the true
        # activation value, so a bias added to a raw accumulator would clip
        # against the wrong scale. w1b is the [1, 2 * inter] gate|up row.
        if w1b is not None:
            gate = gate + w1b[:, c0:c1]
            up = up + w1b[:, inter + c0 : inter + c1]
        chunk = _apply_act(gate, up, act_fn).astype(jnp.bfloat16)
        mid_chunks.append(chunk)
        if requantize:
            chunk_amax = _absmax_rows(chunk)
            amax = chunk_amax if amax is None else jnp.maximum(amax, chunk_amax)
    if not requantize:
        return jnp.concatenate(mid_chunks, axis=-1), None
    mid_scale, mid_sinv, usable = _row_scale(amax)
    sinv_bf16 = mid_sinv.astype(jnp.bfloat16)
    mid_q = jnp.concatenate(
        [_quant_apply(chunk, sinv_bf16, usable) for chunk in mid_chunks], axis=-1
    )
    return mid_q, mid_scale


def _chunked_dot(lhs, rhs_chunk, n_chunks, kb):
    """Contract lhs against a weight handed over one contraction chunk at a
    time, summing the chunks' partial products.

    Four-bit and integer weights both reach the matrix unit through a
    widening, and a whole widened slab does not fit beside the buffered
    ones, so both contract in chunks. rhs_chunk(b) returns contraction chunk
    b already in an element type the matrix unit takes.
    """
    acc = None
    for b in range(n_chunks):
        part = lax.dot_general(
            lhs[:, b * kb : (b + 1) * kb],
            rhs_chunk(b),
            DN,
            preferred_element_type=jnp.float32,
        )
        acc = part if acc is None else acc + part
    return acc


def expert_ffn_fp8(act_q, act_scale, w1, w2, w1s, act_fn="silu", w1b=None):
    """One expert's FFN on eight-bit weights: the gate/up matmul, the
    activation, the intermediate requantization and the down matmul.

    act_q fp8 [m, k] and act_scale f32 [m, 1] are the row-quantized
    activation rows. Returns the down matmul's accumulator and the
    intermediate row scale the caller still has to apply to it.
    """
    inter = w1.shape[-1] // 2
    acc1 = lax.dot_general(act_q, w1, DN, preferred_element_type=jnp.float32)
    mid_q, mid_scale = _intermediate_rows(
        acc1, act_scale, inter, w1s=w1s, act_fn=act_fn, w1b=w1b
    )
    acc2 = lax.dot_general(mid_q, w2, DN, preferred_element_type=jnp.float32)
    return acc2, mid_scale


def expert_ffn_blockscale(
    act_q,
    act_scale,
    w1_block,
    w2_block,
    w1s_blocks,
    w2s_blocks,
    *,
    qb=QB4,
    act_fn="silu",
    w1b=None,
    logical_intermediate_size=None,
):
    """One expert's FFN on block-scaled weights: the same body as
    expert_ffn_fp8 with each matmul summed over contraction blocks carrying
    their own scales, w1s_blocks [nb1, 2*inter] and w2s_blocks [nb2, hidden].

    Each contraction block carries its own scale, on blocks already widened
    to fp8: w1_block(b, n_start, n_end) and w2_block(b, n_start, n_end) return columns
    [n_start, n_end) of weight k-block b in fp8, and
    the widening is the CALLER's. It has to be, because it is a Mosaic
    bitcast on a VMEM reference, which has no host form and so cannot be
    shared with the eight-bit body the way the rest of this function is.
    Nothing below depends on what the block was stored as; every other line
    here runs on either, and a block-scaled integer-four form would reuse
    this body verbatim with a different bitcast in the reader.
    """
    # Padding elision adapted from vllm-project/vllm-torchtpu PR #2160.
    # Keep decode height rungs and cross-expert prefetch unchanged.
    n_blocks1, n_blocks2 = w1s_blocks.shape[0], w2s_blocks.shape[0]
    inter = w1s_blocks.shape[-1] // 2
    hidden = w2s_blocks.shape[-1]

    def contract(lhs, rhs_block, scales, n_blocks, n_out, *, down=False):
        chunks = []
        for n_start in range(0, n_out, FP4_OUTPUT_CHUNK):
            n_end = min(n_start + FP4_OUTPUT_CHUNK, n_out)
            if (
                not down
                and logical_intermediate_size is not None
                and (
                    logical_intermediate_size <= n_start < inter
                    or inter + logical_intermediate_size <= n_start
                )
            ):
                chunks.append(jnp.zeros((lhs.shape[0], n_end - n_start), jnp.float32))
                continue
            acc = None
            for b in range(n_blocks):
                active_k = qb
                if down and logical_intermediate_size is not None:
                    active_k = min(qb, logical_intermediate_size - b * qb)
                    rhs = rhs_block(b, n_start, n_end, block_rows=active_k)
                else:
                    rhs = rhs_block(b, n_start, n_end)
                part = lax.dot_general(
                    lhs[:, b * qb : b * qb + active_k],
                    rhs,
                    DN,
                    preferred_element_type=jnp.float32,
                )
                part = part * scales[b : (b + 1), n_start:n_end]
                acc = part if acc is None else acc + part
            assert acc is not None
            chunks.append(acc)
        return jnp.concatenate(chunks, axis=-1)

    acc1 = contract(act_q, w1_block, w1s_blocks, n_blocks1, 2 * inter)
    mid_q, mid_scale = _intermediate_rows(
        acc1,
        act_scale,
        inter,
        w1s=None,
        act_fn=act_fn,
        w1b=w1b,
        logical_intermediate_size=logical_intermediate_size,
    )
    acc2 = contract(mid_q, w2_block, w2s_blocks, n_blocks2, hidden, down=True)
    return acc2, mid_scale


# --------------------------------------------------------------------------- #
# K32Z expert compute (host.WeightFormat.K32Z; the layout is in host.py).
# --------------------------------------------------------------------------- #
# What a K32Z block decodes to. Every e2m1 value converts to e5m2 exactly, and
# an e5m2 byte's exponent field starts at its bit 2, so adding 4 * d to a
# nonzero byte multiplies its value by 2^d. A zero code becomes 2^(d - 15)
# instead of 0 where d >= 1: that, and only that, is what K32Z changes.
K32Z_WEIGHT_DTYPE = jnp.float8_e5m2
# One e5m2 exponent step in every byte of a 32-bit word.
K32Z_SHIFT_WORD = 0x04040404
# 32-bit words a K group of one column takes: four e5m2 bytes to a word.
K32Z_GROUP_WORDS = host.K32Z_GROUP // 4


def k32z_blocks(k):
    """(first row, rows) of each decoded contraction block of a K-row matmul:
    K32Z_KBLOCK rows each, the last one possibly a 256-row tail."""
    return tuple(
        (k0, min(host.K32Z_KBLOCK, k - k0)) for k0 in range(0, k, host.K32Z_KBLOCK)
    )


def k32z_shift_words(raw, field, bits):
    """One column tile's shift words out of its packed table bytes.

    `raw` is the tile's byte column of the table widened to int32 [groups,
    128], and the tile's shift sits in field `field` (T % F) of each byte.
    Returns int32 [groups, 128] holding d * 0x04040404, i.e. d added to the
    exponent field of each of the four e5m2 bytes of a 32-bit word. Plain jnp
    on values, so the host can hold it to the loader's fields.
    """
    mask = (1 << bits) - 1
    return jnp.bitwise_and(jnp.right_shift(raw, bits * field), mask) * K32Z_SHIFT_WORD


def k32z_build_table(shift_vm, slot, n_start, groups, tab_vm, parity, bits):
    """The shift words of output chunk [n_start, n_start + K32Z_CHUNK) into
    tab_vm[parity] ([column tile in the chunk, group, lane]).

    A chunk's column tiles are consecutive fields of one packed byte column,
    which host.k32z_table_geometry guarantees by keeping chunks whole.
    """
    lanes = host.K32Z_LANE_TILE
    byte_tile, field = divmod(n_start // lanes, 8 // bits)
    raw = shift_vm[slot, :, pl.ds(byte_tile * lanes, lanes)].astype(jnp.int32)
    raw = raw[:groups]
    for t in range(host.K32Z_CHUNK // lanes):
        tab_vm[parity, t, pl.ds(0, groups), :] = k32z_shift_words(raw, field + t, bits)


def k32z_fill_colscale(w2s_vm, slot, col_vm, *, hidden, inter):
    """col_vm row 0 <- the gate|up column scales, row 1 <- the down ones.

    Each is 2^(b - 127), built as the f32 bit pattern b << 23 from the base
    bytes in the down table's padding rows; the loader refuses a base outside
    [1, 254], so the pattern is always a normal power of two.
    """
    lanes = host.K32Z_LANE_TILE
    g2 = inter // host.K32Z_GROUP
    # The uint8 sublane tile(s) the four base rows sit in.
    first = g2 - g2 % host.K32Z_U8_SUBLANE_TILE
    rows = host.align_up(g2 + host.K32Z_BASE_ROWS, host.K32Z_U8_SUBLANE_TILE) - first
    r0 = g2 - first

    def scale_row(block, r):
        return pltpu.bitcast(jnp.left_shift(block[r : r + 1], 23), jnp.float32)

    for byte_tile in range(hidden * host.K32Z_DOWN_BITS // 8 // lanes):
        block = w2s_vm[
            slot, pl.ds(first, rows), pl.ds(byte_tile * lanes, lanes)
        ].astype(jnp.int32)
        for half in range(2):
            tile = 2 * byte_tile + half
            cols = pl.ds(tile * lanes, lanes)
            col_vm[pl.ds(1, 1), cols] = scale_row(block, r0 + half)
            if tile * lanes < 2 * inter:
                col_vm[pl.ds(0, 1), cols] = scale_row(block, r0 + 2 + half)


def k32z_block_readers(
    w1_vm,
    w2_vm,
    w1s_vm,
    w2s_vm,
    tab_vm,
    slot,
    *,
    hidden,
    inter,
    gate_up_bits,
    wait_slab=None,
    slab_retired=None,
    interpret=False,
):
    """(w1_block, w2_block) K32Z block readers at weight slot `slot`.

    `w_block(b, n_start, n_end)` returns contraction block b of output chunk
    [n_start, n_end) as e5m2 [rows, K32Z_CHUNK]: the codes converted in
    hardware, then d added to every byte's exponent, a 32-bit add of the
    chunk's shift words loaded with a stride-0 broadcast per K group.

    A chunk's first block also builds that chunk's shift words, ahead of its
    reads, into the half of tab_vm its parity selects. `wait_slab(which)`
    (1 gate/up, 2 down) lands a slab before the first block of its matmul,
    and `slab_retired()` fires after the last read of the gate/up slab, as
    the four-bit readers do. `interpret` loads a table row and broadcasts it
    instead, for the CPU interpreter, which has no stride-0 load (tests).
    """
    lanes = host.K32Z_LANE_TILE
    tiles = host.K32Z_CHUNK // lanes
    blocks = (k32z_blocks(hidden), k32z_blocks(inter))
    groups = (hidden // host.K32Z_GROUP, inter // host.K32Z_GROUP)
    bits = (gate_up_bits, host.K32Z_DOWN_BITS)
    slabs = ((w1_vm, w1s_vm), (w2_vm, w2s_vm))
    last_column = (2 * inter, hidden)

    def shift_words(parity, g0, n_groups):
        """int32 [n_groups * K32Z_GROUP_WORDS, K32Z_CHUNK]: each group's words
        on the sublanes its 32 rows of codes take."""

        def load(t, g):
            if interpret:
                row = tab_vm[parity, t, pl.ds(g, 1), :]
                return jnp.broadcast_to(row, (K32Z_GROUP_WORDS, lanes))
            return tab_vm[parity, t, pl.ds(g, K32Z_GROUP_WORDS, stride=0), :]

        return jnp.concatenate(
            [
                jnp.concatenate([load(t, g0 + i) for t in range(tiles)], axis=1)
                for i in range(n_groups)
            ],
            axis=0,
        )

    def reader(m):
        w_vm, shift_vm = slabs[m]

        def block(b, n_start, n_end):
            if n_end - n_start != host.K32Z_CHUNK or n_start % host.K32Z_CHUNK:
                raise ValueError(
                    f"K32Z decodes whole {host.K32Z_CHUNK}-column chunks; got "
                    f"columns [{n_start}, {n_end})"
                )
            k0, rows = blocks[m][b]
            parity = (n_start // host.K32Z_CHUNK) % 2
            if b == 0:
                if wait_slab is not None and n_start == 0:
                    wait_slab(m + 1)
                k32z_build_table(
                    shift_vm, slot, n_start, groups[m], tab_vm, parity, bits[m]
                )
            codes = w_vm[
                slot, pl.ds(k0 // PACK4, rows // PACK4), pl.ds(n_start, n_end - n_start)
            ]
            e5 = pltpu.bitcast(codes, FP4).astype(K32Z_WEIGHT_DTYPE)
            words = pltpu.bitcast(e5, jnp.int32) + shift_words(
                parity, k0 // host.K32Z_GROUP, rows // host.K32Z_GROUP
            )
            if (
                slab_retired is not None
                and m == 0
                and b == len(blocks[0]) - 1
                and n_end == last_column[0]
            ):
                slab_retired()
            return pltpu.bitcast(words, K32Z_WEIGHT_DTYPE)

        return block

    return reader(0), reader(1)


def expert_ffn_colscale(
    act_q, act_scale, w1_block, w2_block, w1s_col, *, hidden, act_fn="silu", w1b=None
):
    """One expert's FFN on K32Z weights.

    Each matmul is, per output chunk, a plain sum over K of exact e4m3 x e5m2
    block products: K32Z folds every group's exponent into the decoded block,
    so no block carries a post-scale. What is left is one power of two per
    column, the f32 w1s_col [1, 2 * inter], which applies to the gate/up
    halves before the activation exactly where the per-channel formats apply
    theirs. The down matmul's column scale is the caller's, through the wire
    epilogue's w2s. Returns (acc2, mid_scale) as expert_ffn_blockscale does.
    """
    inter = w1s_col.shape[-1] // 2

    def contract(lhs, rhs_block, k, n_out):
        chunks = []
        for n_start in range(0, n_out, host.K32Z_CHUNK):
            n_end = n_start + host.K32Z_CHUNK
            acc = None
            for b, (k0, rows) in enumerate(k32z_blocks(k)):
                part = lax.dot_general(
                    lhs[:, k0 : k0 + rows],
                    rhs_block(b, n_start, n_end),
                    DN,
                    preferred_element_type=jnp.float32,
                )
                acc = part if acc is None else acc + part
            chunks.append(acc)
        return jnp.concatenate(chunks, axis=-1)

    acc1 = contract(act_q, w1_block, hidden, 2 * inter)
    mid_q, mid_scale = _intermediate_rows(
        acc1, act_scale, inter, w1s=w1s_col, act_fn=act_fn, w1b=w1b
    )
    acc2 = contract(mid_q, w2_block, inter, hidden)
    return acc2, mid_scale


def _pow2_split(x, d):
    """(x // d, x % d) of a traced int32 x >= 0 and a static power of two d."""
    if d == 1:
        return x, 0
    return jnp.right_shift(x, d.bit_length() - 1), jnp.bitwise_and(x, d - 1)


def k32z_chunk_decoders(
    w1_vm,
    w2_vm,
    w1s_vm,
    w2s_vm,
    tab_vm,
    slot,
    *,
    hidden,
    inter,
    gate_up_bits,
    interpret=False,
):
    """(decode1, decode2): k32z_block_readers' decode for a body that walks
    its output chunks in a loop (host.k32z_roll_plan).

    `decode(b, n, parity)` returns contraction block b of output chunk `n` --
    a traced chunk index, so columns [n * K32Z_CHUNK, (n + 1) * K32Z_CHUNK)
    -- as e5m2 [rows, K32Z_CHUNK], the bytes k32z_block_readers' block(b,
    n * K32Z_CHUNK, ...) returns. Block 0 first builds the chunk's shift
    words into tab_vm[parity], a static half. The slab waits and the gate/up
    retire are the caller's, since a loop step cannot tell statically whether
    it holds the first or the last chunk.
    """
    lanes = host.K32Z_LANE_TILE
    tiles = host.K32Z_CHUNK // lanes
    blocks = (k32z_blocks(hidden), k32z_blocks(inter))
    groups = (hidden // host.K32Z_GROUP, inter // host.K32Z_GROUP)
    bits = (gate_up_bits, host.K32Z_DOWN_BITS)
    slabs = ((w1_vm, w1s_vm), (w2_vm, w2s_vm))

    def shift_words(parity, g0, n_groups):
        def load(t, g):
            if interpret:
                row = tab_vm[parity, t, pl.ds(g, 1), :]
                return jnp.broadcast_to(row, (K32Z_GROUP_WORDS, lanes))
            return tab_vm[parity, t, pl.ds(g, K32Z_GROUP_WORDS, stride=0), :]

        return jnp.concatenate(
            [
                jnp.concatenate([load(t, g0 + i) for t in range(tiles)], axis=1)
                for i in range(n_groups)
            ],
            axis=0,
        )

    def build_table(shift_vm, n, n_groups, parity, nbits):
        # Chunk n's column tiles are fields [field, field + tiles) of byte
        # column byte_tile (k32z_build_table); a chunk of whole bytes always
        # starts at field 0.
        fields = 8 // nbits
        byte_tile, field = _pow2_split(n * tiles, fields)
        if tiles % fields == 0:
            field = 0
        start = pl.multiple_of(byte_tile * lanes, lanes)
        raw = shift_vm[slot, :, pl.ds(start, lanes)].astype(jnp.int32)
        raw = raw[:n_groups]
        for t in range(tiles):
            tab_vm[parity, t, pl.ds(0, n_groups), :] = k32z_shift_words(
                raw, field + t, nbits
            )

    def decoder(m):
        w_vm, shift_vm = slabs[m]

        def block(b, n, parity):
            k0, rows = blocks[m][b]
            if b == 0:
                build_table(shift_vm, n, groups[m], parity, bits[m])
            n_start = pl.multiple_of(n * host.K32Z_CHUNK, host.K32Z_CHUNK)
            codes = w_vm[
                slot,
                pl.ds(k0 // PACK4, rows // PACK4),
                pl.ds(n_start, host.K32Z_CHUNK),
            ]
            e5 = pltpu.bitcast(codes, FP4).astype(K32Z_WEIGHT_DTYPE)
            words = pltpu.bitcast(e5, jnp.int32) + shift_words(
                parity, k0 // host.K32Z_GROUP, rows // host.K32Z_GROUP
            )
            return pltpu.bitcast(words, K32Z_WEIGHT_DTYPE)

        return block

    return decoder(0), decoder(1)


def expert_ffn_colscale_rolled(
    act_q,
    act_scale,
    decode1,
    decode2,
    w1s_col,
    acc_vm,
    *,
    hidden,
    act_fn="silu",
    w1b=None,
    unroll=1,
    wait_slab=None,
    slab_retired=None,
):
    """expert_ffn_colscale with each matmul's output chunks walked by a loop
    of `unroll` chunks a step (host.k32z_roll_plan).

    Every chunk is the same sum of the same block products in the same
    order, so the result is expert_ffn_colscale's bit for bit; what changes
    is that the program holds `unroll` chunks' decode instead of all of
    them. A step stores its chunks' accumulators into acc_vm (f32
    [>= rows, >= max(2 * inter, hidden)]), the gate/up one and then, once
    the activation has read it, the down one. `decode1` / `decode2` are
    k32z_chunk_decoders'. `wait_slab(which)` (1 gate/up, 2 down) lands a
    slab before its matmul and `slab_retired()` fires after the last read of
    the gate/up slab, as the readers of expert_ffn_colscale do.
    """
    rows = act_q.shape[0]
    inter = w1s_col.shape[-1] // 2
    chunk = host.K32Z_CHUNK
    # `unroll` is one chunk count for both matmuls or a (gate/up,
    # down) pair (host.K32Z_DECODE_STEP). A step's chunks alternate the two
    # tab_vm halves; with an odd count the next step's first chunk reuses
    # the half its last chunk read, after it in program order.
    unroll_gu, unroll_dn = host.k32z_roll_steps(unroll)

    def contract(lhs, decode, k, n_out, unroll):
        steps, rem = divmod(n_out // chunk, unroll)
        if rem or n_out % chunk:
            raise ValueError(
                f"{n_out} columns are not whole steps of {unroll} {chunk}-column chunks"
            )
        blocks = k32z_blocks(k)

        def step(j, carry):
            for u in range(unroll):
                n = j * unroll + u
                acc = None
                for b, (k0, k_rows) in enumerate(blocks):
                    part = lax.dot_general(
                        lhs[:, k0 : k0 + k_rows],
                        decode(b, n, u % 2),
                        DN,
                        preferred_element_type=jnp.float32,
                    )
                    acc = part if acc is None else acc + part
                n_start = pl.multiple_of(n * chunk, chunk)
                acc_vm[pl.ds(0, rows), pl.ds(n_start, chunk)] = acc
            return carry

        lax.fori_loop(0, steps, step, jnp.int32(0))
        return acc_vm[pl.ds(0, rows), pl.ds(0, n_out)]

    if wait_slab is not None:
        wait_slab(1)
    acc1 = contract(act_q, decode1, hidden, 2 * inter, unroll_gu)
    if slab_retired is not None:
        slab_retired()
    mid_q, mid_scale = _intermediate_rows(
        acc1, act_scale, inter, w1s=w1s_col, act_fn=act_fn, w1b=w1b
    )
    if wait_slab is not None:
        wait_slab(2)
    acc2 = contract(mid_q, decode2, inter, hidden, unroll_dn)
    return acc2, mid_scale


def expert_ffn_bf16(act, w1, w2, act_fn="silu", w1b=None):
    """One expert's FFN on unquantized sixteen-bit weights.

    Nothing here is quantized and nothing is dequantized. The rows arrive as
    the caller produced them, the weights carry no scales, and the
    intermediate goes into the second matmul as the activation left it, so
    both accumulators are already the true values and there is no row scale
    to return: a model that did not ask for quantization gets none.

    Returns the down matmul's accumulator and None, where the quantized
    bodies return the intermediate row scale their caller still owes it.
    """
    inter = w1.shape[-1] // 2
    acc1 = lax.dot_general(act, w1, DN, preferred_element_type=jnp.float32)
    mid, _ = _intermediate_rows(
        acc1, None, inter, w1s=None, act_fn=act_fn, w1b=w1b, requantize=False
    )
    acc2 = lax.dot_general(mid, w2, DN, preferred_element_type=jnp.float32)
    return acc2, None


def expert_ffn_int8(
    act,
    w1_chunk,
    w2_chunk,
    w1s,
    *,
    n_chunks1,
    n_chunks2,
    kb=host.WIDEN_KCHUNK,
    act_fn="silu",
    w1b=None,
):
    """One expert's FFN on eight-bit INTEGER weights with per-channel scales.

    TPU v7's matrix unit has no integer input type (a generation with
    integer input could add a native int8 body as another weight form), so an
    integer weight cannot be contracted as it is stored: w1_chunk(b) and
    w2_chunk(b) return contraction chunk b already widened to bf16, the way
    the four-bit body's callers widen a scale block to fp8. Widening a whole
    slab would need a second slab beside the buffered ones, which is why
    both matmuls contract in chunks.

    Because the widening is to bf16, the rows are contracted unquantized and
    the intermediate stays bf16 too. The scales are per output channel, so
    they apply after the contraction, exactly where the eight-bit float path
    applies its own: w1s [1, 2 * inter] here, and the caller applies the
    second matmul's own per-channel scale to what this returns.

    Returns the down matmul's accumulator and None, for the same reason
    expert_ffn_bf16 does.
    """
    inter = w1s.shape[-1] // 2
    acc1 = _chunked_dot(act, w1_chunk, n_chunks1, kb)
    mid, _ = _intermediate_rows(
        acc1, None, inter, w1s=w1s, act_fn=act_fn, w1b=w1b, requantize=False
    )
    acc2 = _chunked_dot(mid, w2_chunk, n_chunks2, kb)
    return acc2, None


def _all_pairs_barrier(ep):
    """Barrier over every shard: transport peers are not only neighbours."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(jnp.int32(i),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, ep)

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _(second):
        for i in range(ep):
            pl.semaphore_signal(
                second,
                inc=1,
                device_id=(jnp.int32(i),),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(second, ep)


def _slot_copy(src, vm, sems, slot):
    """A copy of `src` into buffer slot `slot`, on that slot's semaphore."""
    return pltpu.make_async_copy(src, vm.at[slot], sems.at[slot])


def _rows_wait(sem, ref, rows):
    """Block until `rows` rows' worth of DMAs on `sem` have landed."""
    pltpu.make_async_copy(ref.at[pl.ds(0, rows)], ref.at[pl.ds(0, rows)], sem).wait()


def _and_nonempty(pred, rows):
    """`pred` and a nonzero length: a zero-length DMA must never issue."""
    return jnp.logical_and(pred, rows > 0)


def _tile_row_scales(window, row_base, tile_m):
    """One tile's activation row scales as the [tile_m, 1] column the FFN
    takes, out of the [sublanes, lanes] window its rows fall in.

    The slab holds one f32 per row in the dense lane-block layout the flat
    array already has, so row r is at sublane r / lanes, lane r % lanes.
    A tile's first row is row-block aligned but not lane-block aligned, so
    the tile straddles two sublanes: rotating the window's lanes by the
    first row's lane and selecting between each sublane and the one after
    it puts the tile's rows in order on one sublane, and the column the
    FFN wants is that sublane read the other way round.

    Every step here is a vector-unit operation on about a lane block of
    values, held beside two expert matmuls. The alternative is for the
    caller to build the column, which is a whole-slab retile in HBM.
    """
    lanes = HIDDEN_LANE_BLOCK
    resid = lax.rem(row_base, jnp.int32(lanes))
    rot = pltpu.roll(window, lanes - resid, 1)
    lane = lax.broadcasted_iota(jnp.int32, rot[:-1].shape, 1)
    rows = jnp.where(lane < lanes - resid, rot[:-1], rot[1:])
    return rows.reshape(-1)[:tile_m][:, None]


def _build_fused_ep_moe_kernel(
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
    k32z_gate_up_bits=host.K32Z_GATE_UP_BITS[0],
    k32z_ffn_roll=0,
    k32z_roll_from=0,
    mirror_unit=0,
    k32z_rungs=None,
    k32z_roll_rungs=None,
    push_from_vmem_rows=None,
    defer_push=False,
    early_ship=False,
    early_ship_flag=False,
    gather_unroll=0,
):
    """Build the pallas_call and the function that invokes it.

    `gather_unroll=k` (2, 4 or 8; 0: off) issues a tile's token-row
    gather k blocks per loop step, then a remainder loop: the same DMAs in the
    same order (name token `_gu<k>`).

    Build choices, K32Z only (host.k32z_build_plan; every
    other format keeps the defaults): `k32z_roll_rungs` the rung heights
    whose bodies loop when `k32z_ffn_roll` is on (None: `k32z_roll_from` and
    up), `push_from_vmem_rows` the tile height the push-from-VMEM bound
    counts in (None: tile_m), `defer_push` a two-hop build's commit drain and
    push one tile into the next visit. (A non-zero DMA priority for the
    return's remote copies was tried and is refused by Mosaic: "non-zero
    priority is not supported for remote DMA", jax 0.11.2.)

    `pad_scale_rows` takes the four-bit scale tables as [G, rows, N] with
    rows = host.fp4_scale_rows(K, rhs_qb, padded=True): K / rhs_qb blocks
    padded to the f32 sublane tile, the [G * rows, N] row-padded flat operand
    viewed per expert. Each slot then stages an expert's whole padded table
    and the FFN reads only its first K / rhs_qb rows, so the result is the
    unpadded build's bit for bit.

    `k32z_gate_up_bits` is the K32Z gate/up shift-field width (2 or 4; the
    other formats ignore it), and `k32z_ffn_roll` how its FFN body walks the
    output chunks (host.k32z_roll_plan; 0 unrolls them). `mirror_unit` below
    the tile height makes every scale mirror [sublanes, mirror_unit], a
    tile's scales that many-lane rows (host.K32Z_MIRROR_UNIT; 0 keeps a
    sublane per tile).
    """
    if pad_scale_rows and weight_format != host.WeightFormat.FP4:
        raise ValueError(
            "pad_scale_rows describes the four-bit block-scale tables; "
            f"weight format {weight_format!r} has none"
        )
    if act_fn not in ACT_FNS:
        raise NotImplementedError(
            f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}"
        )
    # The kernel fetches lhs rows by index from the ungathered token buffer
    # [tokens, lane blocks, 128], using the token-gather table, and ships
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
        k32z_ffn_roll = host.check_k32z_ffn_roll(k32z_ffn_roll, hidden, inter)
    else:
        k32z_ffn_roll = 0
    # These choices are K32Z-only; each is normalized so that a
    # choice which changes nothing in this build keeps the default name.
    if not k32z and (
        k32z_roll_rungs is not None or push_from_vmem_rows is not None or defer_push
    ):
        raise ValueError(
            "k32z_roll_rungs, push_from_vmem_rows and defer_push are K32Z build choices"
        )
    if k32z_roll_rungs is not None:
        k32z_roll_rungs = (
            tuple(sorted({int(h) for h in k32z_roll_rungs})) if k32z_ffn_roll else None
        )
    if push_from_vmem_rows is not None:
        if (
            isinstance(push_from_vmem_rows, bool)
            or int(push_from_vmem_rows) != push_from_vmem_rows
            or push_from_vmem_rows <= 0
            or push_from_vmem_rows % ROWBLK
        ):
            raise ValueError(
                f"push_from_vmem_rows {push_from_vmem_rows!r} is not a positive "
                f"whole number of {ROWBLK}-row blocks"
            )
        push_from_vmem_rows = (
            int(push_from_vmem_rows) if push_from_vmem_rows < capacity else None
        )
    defer_push = bool(defer_push)
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
    # The scale mirror's lanes per sublane (host.mirror_lanes): the tile, or
    # a fixed unit below it, and then a tile's scales are unit_k sublanes and
    # every mirror offset counts units instead of tiles.
    mirror_m = host.mirror_lanes(tile_m, mirror_unit)
    unit_k = tile_m // mirror_m
    unit_blocks = mirror_m // ROWBLK
    mirror_block_shift = host.pow2_shift(unit_blocks, "mirror lanes // ROWBLK")
    mirror_shift = host.pow2_shift(mirror_m, "mirror lanes")
    height_rungs = (
        host.k32z_tile_height_rungs(tile_m, k32z_rungs)
        if k32z
        else tile_height_rungs(tile_m, ragged_rows_alloc, g_local, ep)
    )
    # K32Z visits whose first tile is taller than this wait both weight slabs
    # at their head (host.K32Z_STREAMED_ROWS for the default ladder, else the
    # ladder's tallest rung up to it: host.k32z_streamed_rows); a tile that
    # tall exists only where tile_m is.
    k32z_streamed = host.k32z_streamed_rows(height_rungs) if k32z else 0
    k32z_head_waits = k32z and tile_m > k32z_streamed
    if k32z_roll_rungs is not None and not set(k32z_roll_rungs) <= set(height_rungs):
        raise ValueError(
            f"k32z_roll_rungs {k32z_roll_rungs} are not all rungs of this "
            f"build's ladder {height_rungs}"
        )
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
    #
    # `push_from_vmem_rows` counts the bound in a lower tile
    # height than tile_m: 128 gives a 256-row K32Z build the bound of a
    # 128-row K512 build. Where that does not change the path it is dropped,
    # so the build keeps its default name.
    push_from_vmem = ragged_rows_alloc <= ep * g_local * tile_m
    if push_from_vmem_rows is not None:
        capped = ragged_rows_alloc <= ep * g_local * push_from_vmem_rows
        if capped == push_from_vmem:
            push_from_vmem_rows = None
        push_from_vmem = capped
    # Deferring the drain and push applies to the two-hop path only (and
    # needs a return); elsewhere it changes nothing and keeps the name.
    defer_push = defer_push and not push_from_vmem
    # Early shipping (a kernel-only return-path change, no work moves): on the
    # two-hop path every own expert ships each (expert, dest) run two tiles
    # after the tile that completes it, instead of all of them at its visit's
    # end. Same DMAs and totals, earlier.
    early_ship = bool(early_ship) and k32z and not push_from_vmem
    # The early shipping above is switched per call by lane 0 of the counts
    # table's extra row COUNT_ES_FLAG (the layer's plan-model pick); off, the
    # kernel pushes every run at its visit's end as without early shipping.
    # The same DMAs either way.
    early_ship_flag = bool(early_ship_flag) and early_ship
    gather_unroll = int(gather_unroll)
    if gather_unroll not in (0, 2, 4, 8):
        raise ValueError(f"gather_unroll is 0 (off), 2, 4 or 8; got {gather_unroll}")
    ls_window_rows = host.act_scale_window_rows(tile_m)
    gather_tiles = host.TOKEN_GATHER_TILES_PER_WINDOW
    gather_payload_rows = gather_tiles * tile_m
    gather_window_rows = host.token_gather_window_rows(tile_m)

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
        # The per-call early-ship switch (early_ship_flag).
        es_on = counts_sm[host.COUNT_ES_FLAG, 0] != 0 if early_ship_flag else None
        # The full token-gather table stays in HBM. Two fixed SMEM windows
        # below stream only the rows the current expert tiles consume.
        token_gather_hbm = next(it)
        # lhs_hbm is the ungathered token buffer [tokens, lane blocks, 128].
        # The wire never ships the topk weight, so it is not an operand.
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
        recv_hbm = next(it)
        rscl_hbm = next(it)
        # A `push_from_vmem` build writes neither of these: they stay operands
        # for their row geometry, which the send waits count bytes against.
        contrib_hbm = next(it)
        cscl_hbm = next(it)
        # Two independent refs (rather than a leading slot dimension) let the
        # scalar core branch once per activation tile and then use static SMEM
        # indexing for all of that tile's token rows.
        token_gather_sm0, token_gather_sm1 = (next(it) for _ in range(2))
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

        token_gather_sem0, token_gather_sem1 = (next(it) for _ in range(2))
        lhs_sems, w1_sems, w2_sems = (next(it) for _ in range(3))
        w1s_sems = next(it) if has_scales else None
        w2s_sems = next(it) if has_scales else None
        cp_sem = next(it)
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
                go = jnp.int32(b) < n_visit

                @pl.when(go)
                def _(b=b):
                    weight_starts(visit_sm[b], b)

        prologue()
        _all_pairs_barrier(ep)

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
        def gather_dma_coords(logical_base):
            """Aligned HBM base and in-window offset for a slab row."""
            alignment = jnp.int32(host.TOKEN_GATHER_DMA_ALIGNMENT)
            shift = lax.rem(logical_base, alignment)
            dma_base = pl.multiple_of(
                logical_base - shift, host.TOKEN_GATHER_DMA_ALIGNMENT
            )
            return dma_base, shift

        def issue_gather_window(logical_base, slot):
            """Start one fixed token-index window in SMEM slot ``slot``."""
            dma_base, _ = gather_dma_coords(logical_base)

            @pl.when(slot == 0)
            def _():
                pltpu.make_async_copy(
                    token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
                    token_gather_sm0,
                    token_gather_sem0,
                ).start(priority=0)

            @pl.when(slot == 1)
            def _():
                pltpu.make_async_copy(
                    token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
                    token_gather_sm1,
                    token_gather_sem1,
                ).start(priority=0)

        def wait_gather_window(logical_base, slot):
            """Wait for the fixed token-index window in ``slot``."""
            dma_base, _ = gather_dma_coords(logical_base)

            @pl.when(slot == 0)
            def _():
                pltpu.make_async_copy(
                    token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
                    token_gather_sm0,
                    token_gather_sem0,
                ).wait()

            @pl.when(slot == 1)
            def _():
                pltpu.make_async_copy(
                    token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
                    token_gather_sm1,
                    token_gather_sem1,
                ).wait()

        def lhs_issue_tile(row_base, live_blocks, slot, gather_slot, gather_offset):
            """Issue one tile's activations using a resident index window."""

            def issue_from(gather_smem):
                def issue_block(i, _):
                    """Fetch eight token rows of block i by their numbers."""
                    for r in range(ROWBLK):
                        token = gather_smem[gather_offset + i * ROWBLK + r]
                        pltpu.make_async_copy(
                            lhs_hbm.at[token],
                            lhs_vm.at[slot, i * ROWBLK + r],
                            lhs_sems.at[slot],
                        ).start()
                    return _

                if not gather_unroll:
                    lax.fori_loop(0, live_blocks, issue_block, jnp.int32(0))
                else:
                    # gather_unroll (_gu<k>) blocks per step, then the
                    # remainder; the same DMAs in the same order. (The bounds
                    # are traced, so fori_loop's unroll= is refused.)
                    def issue_chunk(c, carry):
                        for j in range(gather_unroll):
                            issue_block(c * gather_unroll + j, carry)
                        return carry

                    n_chunks = jnp.right_shift(
                        live_blocks, gather_unroll.bit_length() - 1
                    )
                    lax.fori_loop(0, n_chunks, issue_chunk, jnp.int32(0))
                    lax.fori_loop(
                        n_chunks * gather_unroll,
                        live_blocks,
                        issue_block,
                        jnp.int32(0),
                    )

            @pl.when(gather_slot == 0)
            def _():
                issue_from(token_gather_sm0)

            @pl.when(gather_slot == 1)
            def _():
                issue_from(token_gather_sm1)

            if has_act_scale:
                # ls rides along on the same sem. The slab is the dense
                # lane-block view of one f32 per row, so a tile's window is
                # the sublanes its rows fall in, not a row range.
                assert ls_hbm is not None and ls_vm is not None
                pltpu.make_async_copy(
                    ls_hbm.at[pl.ds(row_base // HIDDEN_LANE_BLOCK, ls_window_rows)],
                    ls_vm.at[slot],
                    lhs_sems.at[slot],
                ).start()

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

        def prime_expert(e, lhs_slot, wslot, window_started, visit_i):
            """Prime expert ``e``'s metadata and first activation tile.

            ``wslot`` is the scalar-memory slot this expert's group-zero index
            window lives in -- a global alternation rather than a per-expert
            one, so the previous expert can start it a tile early; that early
            start is what ``window_started`` reports, and then only the wait
            is owed. The window is a bare HBM round trip, and issued here it
            is the one arrival at an expert's head with nothing behind it.
            """
            rows = expert_rows_sm[e]
            slab_base = expert_base_sm[e]

            @pl.when(jnp.logical_not(window_started))
            def _():
                issue_gather_window(slab_base, wslot)

            wait_gather_window(slab_base, wslot)
            _, shift = gather_dma_coords(slab_base)
            lhs_issue_tile(
                slab_base, jnp.minimum(rows, tile_m) // ROWBLK, lhs_slot, wslot, shift
            )

            # Keep group one in flight while tile zero computes. Later group
            # boundaries refill the just-retired alternate slot in the same
            # way, so at most two fixed windows are ever resident.
            @pl.when(rows > gather_payload_rows)
            def _():
                issue_gather_window(
                    slab_base + gather_payload_rows, jnp.int32(1) - wslot
                )

            # A one-tile expert has no following index group. Its alternate
            # SMEM slot can hold the next expert's first window instead.
            @pl.when(jnp.logical_and(rows <= tile_m, visit_i + 1 < n_visit))
            def _():
                issue_gather_window(
                    expert_base_sm[visit_sm[visit_i + 1]], jnp.int32(1) - wslot
                )

        def expert_tiles(e, visit_i, carry):
            """One expert step: a fori_loop over its [tile_m, H] tiles."""
            # carry = (tile_count, <per-parity outstanding counts>, wslot):
            # the global tile counter, what each out_vm parity still owes --
            # committed blocks, or shipped rows and mirror sublanes where the
            # tile ships itself -- and, ALWAYS LAST, the scalar-memory slot
            # this expert's first index window is in.
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
            wslot = carry[-1]
            # Index windows alternate slots across the WHOLE visit list, so
            # the slot after this expert's last group is the one its
            # second-to-last group retired -- free from this expert's last
            # tile on, whatever its group count.
            n_groups = -(-n_tiles // gather_tiles)
            next_wslot = lax.rem(wslot + n_groups, jnp.int32(2))

            def commit_tile(parity, t, tile_block_base, live_blocks):
                """Commit the tile's intersection with each (expert, dest) run."""
                assert commit_sems is not None and commit_scl_sems is not None
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
                if unit_k == 1:
                    pltpu.make_async_copy(
                        staged_scales(parity),
                        cscl_hbm.at[pl.ds(_mirror_base(e) + t, 1)],
                        commit_scl_sems.at[parity],
                    ).start()
                else:
                    for j in range(unit_k):
                        pltpu.make_async_copy(
                            staged_unit(parity, j),
                            cscl_hbm.at[pl.ds(_mirror_base(e) + unit_k * t + j, 1)],
                            commit_scl_sems.at[parity],
                        ).start()

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
                    src_row = (lo - tile_block_base) * ROWBLK
                    into_run = (lo - run_start) * ROWBLK
                    dst_row = push_dst_sm[e, d] * ROWBLK + into_run
                    # The run's true rows are its leading rows, so this tile
                    # carries whatever is left of them inside its overlap.
                    true_here = jnp.clip(
                        true_rows_sm[e, d] - into_run, 0, overlap * ROWBLK
                    )
                    is_me = jnp.int32(d) == me
                    # One sublane per tile a run spans, at the destination
                    # base that run's own arrival rows give it.
                    dst_sub = (
                        jnp.right_shift(push_dst_sm[e, d], tile_block_shift)
                        + 2 * (me * g_local + e)
                        + t
                        - jnp.right_shift(run_start, tile_block_shift)
                    )

                    @pl.when(_and_nonempty(jnp.logical_not(is_me), true_here))
                    def _(
                        src_row=src_row,
                        dst_row=dst_row,
                        true_here=true_here,
                        parity=parity,
                        d=d,
                    ):
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
                    @pl.when(_and_nonempty(is_me, overlap))
                    def _(
                        src_row=src_row, dst_row=dst_row, overlap=overlap, parity=parity
                    ):
                        pltpu.make_async_copy(
                            out_vm.at[parity, pl.ds(src_row, overlap * ROWBLK)],
                            recv_hbm.at[pl.ds(dst_row, overlap * ROWBLK)],
                            send_sems.at[parity],
                        ).start()

                    if unit_k == 1:

                        @pl.when(_and_nonempty(jnp.logical_not(is_me), overlap))
                        def _(dst_sub=dst_sub, parity=parity, d=d):
                            pltpu.make_async_remote_copy(
                                src_ref=staged_scales(parity),
                                dst_ref=rscl_hbm.at[pl.ds(dst_sub, 1)],
                                send_sem=send_scl_sems.at[parity],
                                recv_sem=recv_scl_sem,
                                device_id=(jnp.int32(d),),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start()

                        @pl.when(_and_nonempty(is_me, overlap))
                        def _(dst_sub=dst_sub, parity=parity):
                            pltpu.make_async_copy(
                                staged_scales(parity),
                                rscl_hbm.at[pl.ds(dst_sub, 1)],
                                send_scl_sems.at[parity],
                            ).start()

                    else:
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

                    rows_out = rows_out + jnp.where(is_me, overlap * ROWBLK, true_here)
                    subs_out = subs_out + (
                        jnp.where(overlap > 0, 1, 0) if unit_k == 1 else subs_here
                    )
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

                    if early_ship:
                        # A nonempty run whose last block is in tile t - 2 ships
                        # now (early_ship); the rest ship at the
                        # visit's end (_push_own_expert).
                        own = t >= 2
                        if early_ship_flag:
                            own = jnp.logical_and(own, es_on)

                        @pl.when(own)
                        def _():
                            for d in range(ep):
                                pl.when(
                                    jnp.logical_and(
                                        commit_len_sm[e, d] > 0,
                                        _own_run_last_tile(e, d) == t - 2,
                                    )
                                )(functools.partial(_push_own_run, e, d))

                # Wait this tile's stream, then issue tile t+1's into the other
                # slot before compute, so the fetch runs under the MXU window.
                lhs_ready_tile(live_rows, parity)
                other_parity = jnp.int32(1) - parity

                @pl.when(t + 1 < n_tiles)
                def _():
                    next_t = t + 1
                    next_group = next_t // gather_tiles
                    next_within = lax.rem(next_t, jnp.int32(gather_tiles))
                    next_group_slot = lax.rem(wslot + next_group, jnp.int32(2))
                    next_group_base = slab_base + next_group * gather_payload_rows

                    # The next group's window was issued when the previous
                    # group became current. Wait only at the boundary, then
                    # reuse the retired slot for the following group before
                    # issuing this activation tile.
                    @pl.when(next_within == 0)
                    def _():
                        wait_gather_window(next_group_base, next_group_slot)
                        following = next_group + 1

                        @pl.when(following * gather_tiles < n_tiles)
                        def _():
                            issue_gather_window(
                                slab_base + following * gather_payload_rows,
                                jnp.int32(1) - next_group_slot,
                            )

                    _, next_shift = gather_dma_coords(next_group_base)
                    next_rows = jnp.minimum(rows - next_t * tile_m, tile_m)
                    lhs_issue_tile(
                        slab_base + next_t * tile_m,
                        next_rows // ROWBLK,
                        other_parity,
                        next_group_slot,
                        next_shift + next_within * tile_m,
                    )

                # The penultimate tile has issued the last tile's rows, so
                # neither needs the retired index window again.
                @pl.when(jnp.logical_and(t + 2 == n_tiles, visit_i + 1 < n_visit))
                def _():
                    issue_gather_window(
                        expert_base_sm[visit_sm[visit_i + 1]], next_wslot
                    )

                # The alternate activation slot has retired its previous tile.
                # Fill the next expert's first tile before this final FFN so
                # its token rows and row scales travel under current compute.
                @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
                def _():
                    nxt = visit_sm[visit_i + 1]
                    prime_expert(
                        nxt, other_parity, next_wslot, jnp.bool_(True), visit_i + 1
                    )

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
                        # Bodies up to the streamed height (K32Z_STREAMED_ROWS
                        # on the default ladder) wait each slab in the block
                        # that first reads it, on the expert's first tile
                        # only, so a tail tile never waits; a taller body's
                        # visit waited both slabs at its head.
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
                        if k32z_ffn_roll and (
                            height in k32z_roll_rungs
                            if k32z_roll_rungs is not None
                            else height >= k32z_roll_from
                        ):
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
                        wslot,
                    )

                commit_tile(parity, t, tile_block_base, live_blocks)

                if defer_push:
                    # Deferred push: the previous expert's drain and push, one
                    # tile late. Its last tile is the one before this, so
                    # on the other parity; its earlier commits were taken
                    # by the reuse guards (this tile's included), so this
                    # one wait completes its slab and the push may read it.
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
                        wslot,
                    )

                return (
                    tile_count + 1,
                    jnp.where(parity == 0, live_blocks, carried[1]),
                    jnp.where(parity == 1, live_blocks, carried[2]),
                    wslot,
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

            return carry[:-1] + (next_wslot,)

        def drain_commits(carried):
            """Drain both parities' pending commits, then zero the counts."""
            # The tile reuse guards test pending > 0, so nothing double-waits.
            tile_count, pending0, pending1, wslot = carried

            @pl.when(pending0 > 0)
            def _():
                wait_commits(0, pending0)

            @pl.when(pending1 > 0)
            def _():
                wait_commits(1, pending1)

            return (tile_count, jnp.int32(0), jnp.int32(0), wslot)

        def drain_shipped(carried):
            """Wait out whatever the last tile on each parity put on the wire.

            Every earlier tile was already waited by the reuse guard of the
            tile two later on its parity, so these are the only sends left
            and their counts are the whole of the send semaphores' credit.
            """
            for p in range(host.OUT_PARITIES):
                wait_shipped(p, carried[1 + p], carried[3 + p])

        @pl.when(n_visit > 0)
        def _():
            prime_expert(
                visit_sm[0], jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.int32(0)
            )

        def push_expert(e):
            """Push expert e's remote regions and hop its own-dest region."""
            _push_own_expert(e)

        def _own_run_last_tile(e, d):
            """The expert-local tile run (e, d)'s last block falls in (nonempty runs)."""
            return jnp.right_shift(
                commit_start_sm[e, d] + commit_len_sm[e, d] - 1, tile_block_shift
            )

        def _push_own_expert(e):
            """Push own expert e's remote regions and hop its own-dest region:
            every run, or (early_ship) the runs whose last tile is one of the
            visit's last two, the others having shipped from its tile loop."""
            if not early_ship:
                for d in range(ep):
                    _push_own_run(e, d)
                return
            n_tiles_e = jnp.right_shift(expert_rows_sm[e] + (tile_m - 1), tile_shift)
            for d in range(ep):
                late = _own_run_last_tile(e, d) >= n_tiles_e - 2
                if early_ship_flag:
                    # Switched off: every run ships here, in dest order.
                    late = jnp.logical_or(late, jnp.logical_not(es_on))
                pl.when(late)(functools.partial(_push_own_run, e, d))

        def _push_own_run(e, d):
            """Push own expert e's run to dest d, or hop it where d is this shard."""
            # Remote pushes go per (expert, dest) at true length; run starts
            # stay aligned, so recv positions are unchanged. Each is predicated
            # on its own length: a zero-length REMOTE DMA must never issue.
            region_blocks = commit_len_sm[e, d]
            src_block = contrib_off_sm[e, d]
            dst_block = push_dst_sm[e, d]
            is_me = jnp.int32(d) == me

            @pl.when(_and_nonempty(jnp.logical_not(is_me), region_blocks))
            def _():
                # One expert per push, so the region IS expert e's run.
                true_rows = true_rows_sm[e, d]

                def push_run():
                    """The rows themselves, at their true length."""
                    pltpu.make_async_remote_copy(
                        src_ref=contrib_hbm.at[
                            pl.ds(contrib_off_sm[e, d] * ROWBLK, true_rows)
                        ],
                        dst_ref=recv_hbm.at[
                            pl.ds(push_dst_sm[e, d] * ROWBLK, true_rows)
                        ],
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=(jnp.int32(d),),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                # A nonempty region can hold an empty (e, d) run.
                pl.when(true_rows > 0)(push_run)
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
                    contrib_hbm.at[
                        pl.ds(src_block * ROWBLK, region_blocks * ROWBLK)
                    ],
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
            carry = expert_tiles(e, visit_i, carry)
            if push_from_vmem:
                # Each tile shipped itself; there is nothing left to flush.
                return carry
            if defer_push:
                # The next visit's first tile drains and pushes this expert
                # (tile_body); the last visit's go after the loop.
                return carry
            carry = drain_commits(carry)
            push_expert(e)
            return carry

        carry0 = tuple(jnp.int32(0) for _ in range(6 if push_from_vmem else 4))
        carry_end = lax.fori_loop(0, n_visit, visit_step, carry0)
        if defer_push:
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
            # The per-tile head waits consume every commit but the last one
            # per parity; the drains below consume those.
            _rows_wait(send_sem, contrib_hbm, count(host.COUNT_SEND_ROWS))
            _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
            # The mirror moves whole sublanes, one per tile a run spans,
            # so its counts are their own rows of the count table rather
            # than a division of the row counts.
            _rows_wait(send_scl_sem, cscl_hbm, count(host.COUNT_SEND_MIRROR))
            _rows_wait(recv_scl_sem, rscl_hbm, count(host.COUNT_RECV_MIRROR))
            # Deferred own-destination drain: the total sums my own-dest
            # region lengths, and a skipped pair owes exactly zero rows.
            self_blocks = commit_len_sm[0, me]
            for g in range(1, g_local):
                self_blocks = self_blocks + commit_len_sm[g, me]
            _rows_wait(mehop_sem, recv_hbm, self_blocks * ROWBLK)
            _rows_wait(mehop_scl_sem, rscl_hbm, count(host.COUNT_SELF_MIRROR))

        drain_transport()
        _all_pairs_barrier(ep)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    # The VMEM buffers come from the same list the VMEM accounting sums, in
    # the order the kernel body unpacks them: lhs_vm, w1_vm, w2_vm, the scale
    # tables where the format has them, the bias tables each build asked for,
    # ls_vm where the activations are quantized, out_vm, oscl_vm.
    scratch = (
        [
            # Two independent scalar-memory windows keep dynamic slot selection
            # out of the per-token loop. Their total is fixed at 5 KiB for the
            # 128-row, four-tile schedule, regardless of request size.
            pltpu.SMEM((gather_window_rows,), jnp.int32),
            pltpu.SMEM((gather_window_rows,), jnp.int32),
        ]
        + [
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
            pltpu.SemaphoreType.DMA,  # token_gather_sem0
            pltpu.SemaphoreType.DMA,  # token_gather_sem1
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
    # The staging buffer and the two weight slabs are always operands; the
    # activation row scale, the two weight scale tables and the two bias
    # tables are there only where the build has them.
    n_inputs = (
        N_STAGING_INPUTS
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
        # Which rung bodies loop: a set ("s"), or from a height up.
        if k32z_roll_rungs is not None:
            roll_tag = "s" + "_".join(map(str, k32z_roll_rungs))
        else:
            roll_tag = f"f{k32z_roll_from}" if k32z_ffn_roll and k32z_roll_from else ""
        # Return-path choices, each only where it changes the build.
        r2_tag = (
            f"_pv{push_from_vmem_rows}" if push_from_vmem_rows is not None else ""
        ) + ("_dp" if defer_push else "")
        name = (
            f"fused_ep_moe_v2"
            f"{format_tag}"
            f"{'' if act_fn == 'silu' else '_' + act_fn}"
            f"{bias_tag}"
            f"_g{g_local}_c{capacity}_nb{nbuf}"
            f"{f'_r{host.k32z_roll_tag(k32z_ffn_roll)}' if k32z_ffn_roll else ''}"
            f"{roll_tag}"
            f"{f'_mu{mirror_m}' if unit_k > 1 else ''}"
            f"{'' if k32z_rungs is None else '_q' + '_'.join(map(str, k32z_rungs))}"
            f"{r2_tag}"
            f"{'_es' if early_ship else ''}"
            f"{'x' if early_ship_flag else ''}"
            f"{f'_gu{gather_unroll}' if gather_unroll else ''}"
        )
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=N_PREFETCH,
                in_specs=[hbm] * n_inputs,
                out_specs=[hbm] * len(out_shape),
                scratch_shapes=tuple(scratch),
                grid=(),
            ),
            out_shape=out_shape,
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
        token_gather,
        expert_rows,
        expert_base,
        act_q_gathered,
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
    ):
        """The served ragged transport form; returns arrivals and scales."""
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
            token_gather.astype(jnp.int32),
            act_q_gathered.reshape(-1, lane_blocks, HIDDEN_LANE_BLOCK),
            *act_scale_arg,
            w1,
            w2,
            *scale_args,
            *biases,
        )
        recv, rscl, _contrib, _contrib_scl = res[:4]
        if rscl.ndim == 3:
            # A split mirror back as [sublanes, tile_m]: the same values in
            # the same order, which is all the combine's lookup reads.
            rscl = rscl.reshape(rscl.shape[0], -1)
        return recv, rscl

    return fn


# Pallas keys its kernel-to-jaxpr cache on the `kernel` object itself, so
# memoizing the build is what makes pallas hit across builds.
_BUILD_CACHE = {}
# Two threads missing on the same key would each build a kernel, destroying
# the object identity this cache holds, so the miss path is serialized.
_BUILD_CACHE_LOCK = threading.Lock()


def build_fused_ep_moe_kernel(**kwargs):
    """Memoizing front door for _build_fused_ep_moe_kernel.

    The cache is unbounded. host.ragged_stride_bound's docstring states the
    growth law that decides how many entries a deployment ends up holding;
    the size is logged on every insert so a boot that is accumulating
    programs says so, rather than being read off host memory afterwards.
    """
    key = tuple(sorted(kwargs.items()))
    fn = _BUILD_CACHE.get(key)
    if fn is None:
        with _BUILD_CACHE_LOCK:
            fn = _BUILD_CACHE.get(key)
            if fn is None:
                fn = _build_fused_ep_moe_kernel(**kwargs)
                _BUILD_CACHE[key] = fn
                logger.info(
                    "fused EP MoE: built program %d for %s",
                    len(_BUILD_CACHE),
                    ", ".join(f"{k}={v}" for k, v in key),
                )
    return fn


# --------------------------------------------------------------------------- #
# The arrival combine.
#
# `arrivals[pos]` in XLA is a SparseCore gather, and an offloaded gather writes
# its output to HBM at every size -- in this program the 80 KiB scale gather is
# untagged HBM exactly like the 80 MiB row gather. At t_local=2048 that is an
# 80 MiB temp written and read straight back: 160 MiB of round trip on top of
# the 80 MiB the weighted sum has to read anyway, and then a 16 MiB relayout
# copy because [t, lane blocks, 128] and [t, hidden] do not share a tiling.
# Here each arrival row lands in VMEM as one DMA and is reduced where it lands,
# so the combine moves the rows in and the tokens out and nothing else.
#
# The row DMAs are emitted BESIDE the reduction rather than in a loop of their
# own: they are scalar-unit work against a body that runs the vector unit, so
# in one basic block the scheduler packs them into slots the reduction leaves
# empty, and in a block of their own they are 20480 serial descriptor issues.
#
# What the schedule costs is the DESCRIPTOR, not the arithmetic: the body holds
# at ~7.8 bundles per 4 KiB row however much vector work is taken out of it, so
# the two things worth doing to it are shortening the scalar sequence a row
# needs and keeping both DMA queues fed.
# --------------------------------------------------------------------------- #
# Tokens one combine step stages: the staging buffer is this many times topk
# arrival rows, double-buffered against the step in flight.
COMBINE_TOKENS = 64
# Tokens one output relayout covers -- the sublane count of one bf16 output
# tile, since the sum runs in the arrival row's layout and is transposed into
# the output's a group at a time.
COMBINE_GROUP = 16
# Tokens one reduction chunk holds. The chunk is a real loop body, so it is the
# scheduler's whole region and therefore the register budget: a token's
# accumulator is hidden/1024 f32 registers wide and everything in the chunk is
# live at once. Unrolled straight-line over a whole step, the list scheduler
# hoists every arrival load and spills.
COMBINE_CHUNK = 16
# Below this many local tokens XLA does not offload the gather at all (b64 and
# b32 emit no OFFLOAD_GATHER), so there is no round trip to remove and a second
# Mosaic call's fixed cost would be the whole effect.
COMBINE_MIN_TOKENS = 256
COMBINE_PARITIES = 4
# Steps of row lead. A step's rows are waited at its head, so at one step of
# lead the DMA queue drains to empty at every boundary and refills from cold;
# more lead keeps descriptors in flight across it. Must divide the rotation:
# 0 < COMBINE_LEAD < COMBINE_PARITIES.
COMBINE_LEAD = 2


def combine_step_tokens(t_local):
    """Tokens a combine step stages, or 0 where this path does not apply.

    The step loop is unrolled by parity so that every VMEM store has a static
    buffer index, which needs an even step count; the group width has to divide
    the step; and the fixed cost is only worth paying above COMBINE_MIN_TOKENS.
    """
    if t_local < COMBINE_MIN_TOKENS:
        return 0
    tb = COMBINE_TOKENS
    while tb >= COMBINE_GROUP:
        steps = t_local // tb
        if (
            not t_local % tb
            and not tb % COMBINE_GROUP
            and steps > COMBINE_LEAD
            and steps >= COMBINE_PARITIES
            and not steps % COMBINE_PARITIES
        ):
            return tb
        tb //= 2
    return 0


def _build_combine_kernel(
    *,
    t_local,
    topk,
    hidden,
    lane_blocks,
    wire_dtype,
    out_dtype,
    fuse_scales=False,
    fold_select=False,
):
    """out[i] = sum_k coef[i, k] * arrivals[pos[i * topk + k]].

    `pos` is a scalar-prefetch table of arrival rows, already clamped into the
    buffer; `coef` is the arrival row's scale folded with the router weight and
    rides SMEM beside it, so a slot's weight reaches the vector unit as a
    scalar splat. The result is [t_local, hidden] in the layer's own output
    layout.

    An arrival row is [lane blocks, 128] -- one register -- while the output
    is [tokens, hidden], which is that layout transposed: a token's hidden
    axis lies along sublanes in the row and along lanes in the output. The sum
    therefore runs in the ROW's layout and only its bf16 result is relaid out,
    so the sublane shuffle sees one bf16 token per group instead of topk
    packed-f8 rows, and the live accumulator is one token wide.

    With ``fuse_scales``, scalar prefetch carries the mirror indices, router
    weights and arrival-scale mirror instead of precomputed coefficients.
    The same f32 scale * weight product runs in SMEM, avoiding an XLA gather
    behind push's other SparseCore work. All row DMAs and reduction ordering
    are shared by the two builds; the choice is static at trace time.

    With ``fold_select`` (on the coefficient build too), scalar prefetch also carries one route
    flag per token, and a token whose flag is zero stores zeros instead of
    its sum: the layer's where(row_routes, out, 0) pass after the combine,
    folded in, the same bits.
    """
    tb = combine_step_tokens(t_local)
    steps = t_local // tb
    gt = COMBINE_GROUP
    groups = tb // gt
    lanes = HIDDEN_LANE_BLOCK

    def kernel(pos_sm, *refs):
        if fuse_scales:
            scl_sm, tw_sm, scales_sm, *refs = refs
        else:
            coef_sm, *refs = refs
        route_sm = None
        if fold_select:
            route_sm, *refs = refs
        arr_hbm, out_hbm, rows_vm, tile_vm, out_vm, row_sems, out_sems = refs

        def start_row(base, parity, i, k):
            # Alternate queues by slot: this kernel is one 4 KiB descriptor per
            # arrival row and an in-order queue retires descriptors at a fixed
            # rate, which for a row this small is slower than the row's own
            # bytes. Both queues signal the one semaphore the step waits on,
            # which counts bytes and not descriptors.
            pltpu.make_async_copy(
                arr_hbm.at[pos_sm[(base + i) * topk + k]],
                rows_vm.at[parity, k * tb + i],
                row_sems.at[parity],
            ).start(priority=k % 2)

        def reduce_token(base, parity, i):
            """One token's weighted sum, in the arrival rows' own layout."""
            acc = None
            for k in range(topk):
                if fuse_scales:
                    idx = (base + i) * topk + k
                    coefficient = scales_sm[scl_sm[idx]] * tw_sm[idx]
                    term = rows_vm[parity, k * tb + i].astype(jnp.float32) * coefficient
                else:
                    term = (
                        rows_vm[parity, k * tb + i].astype(jnp.float32)
                        * coef_sm[(base + i) * topk + k]
                    )
                acc = term if acc is None else acc + term
            if route_sm is not None:
                # An unrouted token read rows nobody wrote: its sum may be
                # NaN, so it is selected away, not multiplied by zero.
                keep = jnp.full(acc.shape, route_sm[base + i], jnp.int32) != 0
                acc = jnp.where(keep, acc, 0.0)
            tile_vm[i] = cast(jax.Array, acc).astype(out_dtype)

        def one_step(step, parity):
            # The step's rows were started a step ago; the next step's are
            # started beside this one's reduction, so their descriptor issue
            # rides in slots the vector work leaves empty. The last step
            # re-starts its own rows rather than branching -- the copy is never
            # read and is drained below.
            nxt = jnp.minimum(step + COMBINE_LEAD, steps - 1) * tb
            base = step * tb
            npar = (parity + COMBINE_LEAD) % COMBINE_PARITIES
            pltpu.make_async_copy(
                rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
            ).wait()

            @pl.when(step >= COMBINE_PARITIES)
            def _():
                pltpu.make_async_copy(
                    out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
                ).wait()

            def one_chunk(chunk, carry):
                for u in range(COMBINE_CHUNK):
                    i = chunk * COMBINE_CHUNK + u
                    for k in range(topk):
                        start_row(nxt, npar, i, k)
                    reduce_token(base, parity, i)
                return carry

            lax.fori_loop(0, tb // COMBINE_CHUNK, one_chunk, jnp.int32(0))
            for g in range(groups):
                out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
                    gt, hidden
                )
            pltpu.make_async_copy(
                out_vm.at[parity], out_hbm.at[pl.ds(step * tb, tb)], out_sems.at[parity]
            ).start()

        def step_pair(pair, carry):
            for parity in range(COMBINE_PARITIES):
                one_step(pair * COMBINE_PARITIES + parity, parity)
            return carry

        for p in range(COMBINE_LEAD):
            for i in range(tb):
                for k in range(topk):
                    start_row(p * tb, p, i, k)
        lax.fori_loop(0, steps // COMBINE_PARITIES, step_pair, jnp.int32(0))
        # The last COMBINE_LEAD steps re-start the final step's rows into the
        # buffers the loop no longer reads; drain them, and the stores.
        for p in range(COMBINE_LEAD):
            pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
        for p in range(COMBINE_PARITIES):
            pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=(4 if fuse_scales else 2) + int(fold_select),
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM(
                    (COMBINE_PARITIES, topk * tb, lane_blocks, lanes), wire_dtype
                ),
                pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
                pltpu.VMEM((COMBINE_PARITIES, tb, hidden), out_dtype),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=host.vmem_limit(), disable_bounds_checks=True
        ),
        name=(
            f"moe_v2_combine_t{t_local}_k{topk}_b{tb}"
            + (f"_p{COMBINE_PARITIES}_fs" if fuse_scales else "")
            + ("_sel" if fold_select else "")
        ),
    )

    if fuse_scales:

        def fn_with_scales(
            arrivals, arrival_scales, scl, topk_weights, pos, routes=None
        ):
            if (routes is not None) != fold_select:
                raise ValueError(
                    f"the combine was built with fold_select={fold_select}, "
                    f"so it {'takes' if fold_select else 'has no operand for'} "
                    "the route flags"
                )
            route_arg = ()
            if fold_select:
                route_arg = (routes.astype(jnp.int32).reshape(-1),)
            return call(
                pos.astype(jnp.int32).reshape(-1),
                scl.astype(jnp.int32).reshape(-1),
                topk_weights.astype(jnp.float32).reshape(-1),
                arrival_scales.astype(jnp.float32).reshape(-1),
                *route_arg,
                arrivals,
            )

        return fn_with_scales

    def fn(arrivals, coef, pos, routes=None):
        # Both tables flatten TOKEN-major, so a token's topk entries are one
        # contiguous SMEM run and its 2 * topk scalar reads share a single
        # address computation. That is also the order the plan builds them in,
        # so neither flatten is a transpose.
        if pos.shape != (t_local, topk) or coef.shape != (t_local, topk):
            raise ValueError(
                f"the combine takes token-major "
                f"[{t_local}, {topk}] tables, got pos "
                f"{pos.shape} and coef {coef.shape}"
            )
        if (routes is not None) != fold_select:
            raise ValueError(
                f"the combine was built with fold_select={fold_select}, "
                f"so it {'takes' if fold_select else 'has no operand for'} "
                "the route flags"
            )
        route_arg = ()
        if fold_select:
            route_arg = (routes.astype(jnp.int32).reshape(-1),)
        return call(
            pos.astype(jnp.int32).reshape(-1), coef.reshape(-1), *route_arg, arrivals
        )

    return fn


_COMBINE_CACHE = {}
_COMBINE_CACHE_LOCK = threading.Lock()


def build_combine_kernel(**kwargs):
    """Memoizing front door; see build_fused_ep_moe_kernel for why."""
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    fn = _COMBINE_CACHE.get(key)
    if fn is None:
        with _COMBINE_CACHE_LOCK:
            fn = _COMBINE_CACHE.get(key)
            if fn is None:
                fn = _build_combine_kernel(**kwargs)
                _COMBINE_CACHE[key] = fn
    return fn
