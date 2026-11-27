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

"""MiniMax H3's DiT block stack, every layer in one Pallas kernel.

``stack`` runs ``minimax_h3_reference``'s blocks in one ``pallas_call``:
the weights stay in HBM and stream through VMEM tile by tile, and the
activations, norms, RoPE, softmax, residuals and the sequence-parallel
exchange stay in the kernel between layers. Each layer runs three phases:

- A: per row tile, norm and modulate, then streamed QKV tiles of two heads
  each with q/k norm and RoPE in their epilogue; each head's q, k, v goes
  by DMA to the device that owns the head (Ulysses) while later tiles
  compute.
- B: flash attention over every row for this device's heads; each output
  tile goes by DMA to the device that owns its rows. A layer whose q and k
  norm weights bound its scores runs the softmax at a fixed max, the rest
  with a running one.
- C: per row tile, the output projection and gated residual, then norm,
  modulate, and the MLP as streamed (gate|up, down) tile pairs.

Under a ``shard_map`` over ``axis`` each device holds ``T / P`` rows and
every weight. The exchanges land in HBM buffers double-buffered by layer
parity: a device writes a peer's parity-``p`` buffer two layers later only
after that peer sent it data that needed the buffer finished.

With more than one head a device (the bulk exchange) phase A sends each
owner only the head its phase B starts with and parks the rest in local
HBM; phase B streams those on while it attends, each head's rows ahead of
the owner's first item of it.
A device is in phase B only once every peer sent it phase A's head, after
that peer finished the layer two back: the parity rule above still holds.
"""

from __future__ import annotations

import contextlib
import functools
import math
import numbers
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

_F32 = jnp.float32
_BF16 = jnp.bfloat16
TABLE_LANES = ref.TABLE_LANES
# Barrier namespace; internal_ring_attention uses 7.
_COLLECTIVE_ID = 8
_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
# A layer whose q and k norms bound every score, in log2 units, by at most
# ``_FIXED_BOUND`` takes ``exp2(s - _FIXED_M)`` without a running max: the
# largest weight of a row stays above 2^-126 and a sum of 2^15 rows times
# values under 2^20 below 2^127.
_FIXED_BOUND, _FIXED_M = 109.0, 17.0
# Rounding slack on the bound: q and k each round to bf16 (unit roundoff
# 2^-8) three times, the normed product, RoPE's bf16 tables and the output,
# which grows |q| |k| by at most (1 + 2^-8)^6 < 1.024.
_BOUND_SLACK = 1.025
# sum_bounds: a fixed-max softmax is the softmax up to rounding when each real
# row's sum of weights exp2(s - _FIXED_M) lies in [2^_SUM_LO, 2^_SUM_HI]. No
# weight then exceeds 2^_SUM_HI (each is at most its row's sum), so with
# values under 2^20 (as the norm-proven layers assume) neither the sum nor any
# partial P V sum passes 2^124; and the weights under 2^-126 that may flush to
# zero, at most 2^15.2 of them, move a row's sum and output by under
# 2^-110.8 / 2^_SUM_LO = 2^-50.8 of the row's sum, far below f32 rounding.
_SUM_LO, _SUM_HI = -60.0, 104.0
# The six AdaLN tables of a block, in the checkpoint's chunk order.
_SHIFT_MSA, _SCALE_MSA, _GATE_MSA, _SHIFT_MLP, _SCALE_MLP, _GATE_MLP = range(6)
_CONTRACT_LAST = (((1,), (1,)), ((), ()))  # a @ b.T
_CONTRACT_FIRST = (((0,), (0,)), ((), ()))  # a.T @ b
# ``ref.pack_weights``'s arrays, in the kernel's operand order.
_WEIGHT_ORDER = "norm1 norm2 q_norm k_norm wqkv wo w13 w2".split()


# ------------------------------------------------------------------- kernel


def _matmul(lhs: Any, rhs: Any, acc: Any, rows: int, sub: int, cols: int):
    """``acc[:rows] = lhs[:rows] @ rhs``, a ``[sub, K] x [K, cols]`` call a piece.

    The MXU accumulates every K step of a piece itself. A bf16 ``acc`` gets
    the f32 dot rounded once (``round_bf16``'s rounding).
    """
    width = rhs.shape[1]
    cols = tl.divisor(width, cols, 128)
    columns = width // cols

    def body(index: jax.Array, carry: None) -> None:
        rs = pl.ds(pl.multiple_of(index // columns * sub, sub), sub)
        cs = pl.ds(pl.multiple_of(index % columns * cols, cols), cols)
        acc[rs, cs] = jnp.dot(
            lhs[rs, :], rhs[:, cs], preferred_element_type=_F32
        ).astype(acc.dtype)
        return carry

    if rows // sub * columns == 1:
        body(0, None)
        return
    lax.fori_loop(0, rows // sub * columns, body, None)


def _for_chunks(rows: int, chunk: int, fn: Any) -> None:
    """Calls ``fn(row_slice)`` for every ``chunk`` rows of ``rows``."""

    def body(index: jax.Array, carry: None) -> None:
        fn(pl.ds(pl.multiple_of(index * chunk, chunk), chunk))
        return carry

    lax.fori_loop(0, rows // chunk, body, None)


def _stream_pipelined(
    tiles: int,
    fetch: Any,
    compute: Any,
    finish: Any,
    prologue: Any = None,
    finish_first: bool = False,
    lead: Any = contextlib.nullcontext,
) -> None:
    """A stream of weight tiles through a two-slot ring, each tile's VPU
    epilogue under the next tile's MXU work.

    ``fetch(i, slot)`` lists the copies that bring tile ``i`` into ``slot``;
    ``prologue`` runs while the first tile streams. ``compute(i, slot)`` runs
    tile ``i``'s matmul into result buffer ``i % 2``; ``finish(i)``
    (unrolled, no loops or branches) consumes it, emitted in the same basic
    block as ``compute(i + 1)`` so the two overlap. ``finish_first``: each step emits ``finish(i - 1)`` before ``compute(i)``,
    so one result buffer serves both (``finish`` reads it before the matmul's
    result pops overwrite it) and the matmul's pushes run under the epilogue.
    ``lead``: a context (a trace region) around the first tile's load and
    matmul.
    """
    with lead():
        for copy in fetch(0, 0):
            copy.start()
        if prologue is not None:
            prologue()
        for copy in fetch(0, 0):
            copy.wait()
        if tiles > 1:
            for copy in fetch(1, 1):
                copy.start()
        compute(0, 0)

    def step(index: jax.Array, carry: None) -> None:
        slot = index % 2
        for copy in fetch(index, slot):
            copy.wait()

        @pl.when(index + 1 < tiles)
        def _next():
            for copy in fetch(index + 1, 1 - slot):
                copy.start()

        if finish_first:
            finish(index - 1)
            compute(index, slot)
        else:
            compute(index, slot)
            finish(index - 1)
        return carry

    lax.fori_loop(1, tiles, step, None)
    finish(tiles - 1)


def _stream_ahead(
    tiles: int,
    fetch: Any,
    compute: Any,
    gates: Any,
    finish: Any,
    prologue: Any = None,
    gate_slots: int = 2,
    lead: Any = contextlib.nullcontext,
    first: tuple[Any, Any] | None = None,
) -> None:
    """``_stream_pipelined`` for a gated epilogue, its gates a step ahead.

    Step ``i`` runs one of two copies of its block, one per ``p = i % 2``,
    so the buffers each copy touches are static: ``compute(i, slot, p)``
    runs tile ``i``'s matmul (weights in ring slot ``slot``) into result
    buffer ``p``; ``gates(i, g)`` looks tile ``i``'s gates up (one-hot
    matmuls, after the main one as ``finish``'s lookups follow it in
    ``_stream_pipelined``'s block, so the MXUs split the main one alike) into
    gate buffer ``g``; ``finish(i - 1, 1 - p, g)`` is then VPU work on
    buffers the block's matmul does not write, so none of it waits for that
    matmul's results. Tile 0's gates run under its weight tile's load.
    ``gate_slots``: 2 runs a block as ``compute(i)``, ``gates(i)`` into
    buffer ``p``, ``finish(i - 1)`` with buffer ``1 - p``, the ring's wait
    and next start ahead of the two copies; 1 runs ``compute(i)``,
    ``finish(i - 1)``, then ``gates(i)`` into the buffer ``finish`` just
    read, each copy with its own wait and start (static ring slots).
    ``lead``: as ``_stream_pipelined``'s. ``first``: ``(copies, compute0)``,
    tile 0's weights loaded elsewhere ahead of the stream (``copies``, waited
    here in place of ``fetch(0, 0)``) and its matmul from there.
    """
    with lead():
        if first is None:
            for copy in fetch(0, 0):
                copy.start()
        gates(0, 0)
        if prologue is not None:
            prologue()
        for copy in fetch(0, 0) if first is None else first[0]:
            copy.wait()
        if tiles > 1:
            for copy in fetch(1, 1):
                copy.start()
        if first is None:
            compute(0, 0, 0)
        else:
            first[1]()

    def block(index: jax.Array, p: int) -> None:
        """``gate_slots`` 1: step ``index`` of parity ``p``."""
        for copy in fetch(index, p):
            copy.wait()

        @pl.when(index + 1 < tiles)
        def _next():
            for copy in fetch(index + 1, 1 - p):
                copy.start()

        compute(index, p, p)
        finish(index - 1, 1 - p, 0)
        gates(index, 0)

    def pair(index: jax.Array, slot: jax.Array, p: int) -> None:
        """``gate_slots`` 2: step ``index``'s block of parity ``p``."""
        compute(index, slot, p)
        gates(index, p)
        finish(index - 1, 1 - p, 1 - p)

    def step(index: jax.Array, carry: None) -> None:
        if gate_slots == 1:
            odd = index % 2 == 1
            pl.when(odd)(lambda: block(index, 1))
            pl.when(jnp.logical_not(odd))(lambda: block(index, 0))
            return carry
        slot = index % 2
        for copy in fetch(index, slot):
            copy.wait()

        @pl.when(index + 1 < tiles)
        def _next():
            for copy in fetch(index + 1, 1 - slot):
                copy.start()

        for p in (0, 1):
            pl.when(slot == p)(functools.partial(pair, index, slot, p))
        return carry

    lax.fori_loop(1, tiles, step, None)
    last = (tiles - 1) % 2
    finish(tiles - 1, last, last if gate_slots == 2 else 0)


def _unrolled_chunks(rows: int, chunk: int, fn: Any) -> None:
    """``_for_chunks`` unrolled: every chunk in the caller's basic block."""
    for index in range(rows // chunk):
        fn(pl.ds(index * chunk, chunk))


# pylint: disable-next=too-many-statements
def _kernel(
    meta_ref: Any,  # SMEM int32: rows before the padding; per layer bit 0 =
    # fixed-max softmax, bit 5 = the norm weights bound it (no peaks); per
    # layer two words of fixed-max head bits (bit d heads/P + j: device d's
    # local head j); per phase A row tile its row_tile_modes; uniform_adaln:
    # then per phase C row tile 1 if its rows read one AdaLN row
    # (uniform_row_tiles)
    x_in: Any,  # aliased to x_ref
    onehot_ref: Any,  # [T_l, 128] bf16
    cos_ref: Any,  # [T_l, D] bf16
    sin_ref: Any,  # [T_l, D] bf16
    tables_ref: Any,  # [L, 6, R, H] bf16
    norm1_ref: Any,  # [L, 1, H] bf16
    norm2_ref: Any,  # [L, 1, H] bf16
    qn_ref: Any,  # [L, 1, D] bf16
    kn_ref: Any,  # [L, 1, D] bf16
    wqkv_ref: Any,  # [L, heads/g, H, g 3D] bf16: g heads' q | k | v a tile
    wo_ref: Any,  # [L, H/out_cols, W, out_cols] bf16
    w13_ref: Any,  # [L, F/ffn_cols, H, 2 ffn_cols] bf16: gate | up
    w2_ref: Any,  # [L, H/down_cols, F, down_cols] bf16
    x_ref: Any,  # [T_l, H] bf16, updated in place layer by layer
    kv_ref: Any,  # [2, heads/P, 3, T, D] bf16: q, k, v of my heads
    attn_ref: Any,  # [2, T_l, W] bf16: every head's output, my rows
    peaks_ref: Any,  # [L, 2, 8, 128] f32: each head's (lane's) largest
    # squared q (row 0 of [0]) and k ([1]) norm over my rows, as phase B
    # reads them; bounds the scores of a fixed-max softmax (sum_bounds: the
    # softmax sums, ``sum_safe_heads``)
    norms: Any,  # VMEM [2, 2, 1, H] bf16: norm1, norm2 by layer parity
    qk_norms: Any,  # VMEM [2, 2, 1, D] bf16: q_norm, k_norm by layer parity
    norm_sems: Any,  # DMA [4]
    kv_send: Any,  # DMA [stage_slots g] per staging slot and head
    kv_recv: Any,  # DMA [2] per layer parity (bulk_exchange: [2, heads/P]
    # per layer parity and local head)
    attn_send: Any,  # DMA [2] per staging slot
    attn_recv: Any,  # DMA [2] per layer parity
    peaks: Any,  # VMEM [2, 8, 128] f32: this layer's peaks so far
    peaks_sem: Any,  # DMA [1]
    head_order: Any,  # SMEM [heads/P + 1] int32: this layer's local heads,
    # those at a fixed max first
    *,
    config: ref.Config,
    tiling: tl.Tiling,
    axis: str | None,
    devices: int,
    tail_rows: int = 0,
    trace: int = 0,
    cut_subs: tuple[int, int] = (0, 0),
    cut_full: tuple[int, int] = (0, 0),
    cut_keys: int = 0,
    split_items: bool = False,
    epilogue_chunk: int = 0,
    epilogue_chunk_peaks: int = 0,
    modulate_pieces: int = 0,
    uniform_adaln: bool = False,
    zero_ahead: bool = False,
    mask_scores: bool = False,
    sum_bounds: bool = False,
    cast_first: bool = False,
    step_tables: bool = False,
    slab_modulate: bool = False,
    kv_group: tuple[int, int] = (1, 1),
    static_used: int = 0,
    prefetch_down: bool = False,
    fixed_max: float = _FIXED_M,
    bulk_exchange: bool = False,
    kv_mine_ref: Any = None,  # bulk_exchange: [P, heads/P, 3, T_l, D] bf16,
    # my q, k, v rows of each owner's heads that wait for phase B
    bulk_send: Any = None,  # bulk_exchange: DMA [1], phase B's sends
    orders: Any = None,  # bulk_exchange: SMEM [P, heads/P + 1] int32, every
    # device's head_order this layer
) -> None:
    """All layers for this device's rows; see the module docstring."""
    del x_in
    hidden, ffn = config.hidden, config.ffn
    dim, heads = config.head_dim, config.heads
    # Tile widths the weights were packed with (``ref.pack_weights``).
    group = wqkv_ref.shape[-1] // (3 * dim)
    out_cols, down_cols = wo_ref.shape[-1], w2_ref.shape[-1]
    ffn_cols = w13_ref.shape[-1] // 2
    local_heads = heads // devices
    rows = x_ref.shape[0]
    table_rows = tables_ref.shape[2]
    chunk = tiling.chunk_rows
    # A traced ``used`` is not checked: past the rows it reads as all rows.
    used = jnp.clip(meta_ref[0], 0, rows * devices)
    layers = norm1_ref.shape[0]
    me = lax.axis_index(axis) if devices > 1 else 0
    scale = math.log2(math.e) / math.sqrt(dim)
    copy = pltpu.make_async_copy

    def computed_head(layer, local):
        """The head my local head ``local`` computes in ``layer``: its columns
        of the attention rows."""
        del layer
        return me * local_heads + local

    def scope(name, level=1):
        """An xprof trace region (``trace >= level``), else nothing."""
        return jax.named_scope(name) if trace >= level else contextlib.nullcontext()

    matmul = functools.partial(_matmul, cols=tiling.n_chunk)

    def device(index: jax.Array) -> dict[str, jax.Array]:
        # The other mesh axes keep this device's coordinates.
        return {axis: index}

    def transfer(src, dst, local_sem, send_sem, recv_sem, owner):
        """``(start, wait)`` of ``src`` -> ``dst`` on ``owner``, local if me.

        ``wait`` returns once ``src`` may be reused.
        """
        local = copy(src, dst, local_sem)
        if devices == 1:
            return local.start, local.wait
        remote = pltpu.make_async_remote_copy(
            src, dst, send_sem, recv_sem, device_id=device(owner)
        )

        def start():
            pl.when(owner == me)(local.start)
            pl.when(owner != me)(remote.start)

        def wait():
            pl.when(owner == me)(local.wait)
            pl.when(owner != me)(remote.wait_send)

        return start, wait

    def owner_local(spot):
        """``(spot // local_heads, spot % local_heads)``: the device that computes
        head spot ``spot`` and its local head there (step_tables: a table load)."""
        if step_tables and "heads" in tables_at:
            word = tables_at["heads"][spot]
            return word & 0xFF, (word >> 8) & 0xFF
        return spot // local_heads, spot % local_heads

    def bulk_transfer(src, parity, spot, row, count, local_sem, send_sem):
        """bulk_exchange's ``transfer`` of my rows ``[row, row + count)`` of a
        head's q, k, v (``src``): ``(start, wait)``. ``spot``: its owner and
        local head there (``head_spot``).

        Into its owner's ``kv_ref`` if the owner is me (a local copy) or the
        head comes first in the owner's order this layer (a remote one, onto
        the owner's receive semaphore of that head); else into ``kv_mine``
        (local), which phase B sends on.
        """
        owner, local = owner_local(spot)
        rs = pl.ds(me * rows + row, count)
        mine = copy(src, kv_ref.at[parity, local, :, rs], local_sem)
        later = copy(src, kv_mine_ref.at[owner, local, :, pl.ds(row, count)], local_sem)
        remote = pltpu.make_async_remote_copy(
            src,
            kv_ref.at[parity, local, :, rs],
            send_sem,
            kv_recv.at[parity, local],
            device_id=device(owner),
        )
        first = orders[owner, 0] == local
        sent, kept = (owner != me) & first, (owner != me) & ~first

        def begin():
            pl.when(owner == me)(mine.start)
            pl.when(sent)(remote.start)
            pl.when(kept)(later.start)

        def end():
            pl.when(owner == me)(mine.wait)
            pl.when(sent)(remote.wait_send)
            pl.when(kept)(later.wait)

        return begin, end

    def await_all(view, send_sem, recv_sem, units, count):
        """Waits for ``count`` incoming copies shaped like ``units`` in one wait
        on ``view``, as many bytes (DMA semaphores count bytes)."""
        del units, count
        if devices == 1:
            return
        pltpu.make_async_remote_copy(
            view, view, send_sem, recv_sem, device_id=device(me)
        ).wait_recv()

    def load_tables(layer, picks, buffer, sems):
        """Starts copying tables ``picks`` of ``layer`` into ``buffer``.

        The one-hot's lanes past the real rows are zero, but zero times stale
        VMEM can be NaN, so the rows past the tables are zeroed.
        """
        copies = [
            copy(
                tables_ref.at[layer, pick],
                buffer.at[slot, pl.ds(0, table_rows)],
                sems.at[slot],
            )
            for slot, pick in enumerate(picks)
        ]
        for one in copies:
            one.start()
        if table_rows < TABLE_LANES:
            for slot in range(len(picks)):
                buffer[slot, pl.ds(table_rows, TABLE_LANES - table_rows), :] = (
                    jnp.zeros((TABLE_LANES - table_rows, hidden), _BF16)
                )
        return copies

    def lookup(onehot, buffer, slot, rs, cols=slice(None)):
        """``[chunk, cols]`` f32 AdaLN parameters of rows ``rs``."""
        table = buffer.at[slot][:, cols]
        return jnp.dot(onehot[rs, :], table, preferred_element_type=_F32)

    def shared_rows(onehot, buffer, rows):
        """``rows [2, 1, H]`` f32 = the shift and scale (``buffer`` slots 0, 1)
        of a row tile whose one-hots are all its first row's: one lookup of a
        bf16 sublane tile of its rows. A row of a product depends on that row
        alone, so this is bit for bit every chunk's lookup."""
        for k in (0, 1):
            rows[k] = lookup(onehot, buffer, k, pl.ds(0, 16))[0:1]

    def fill(firsts, onehot, buffer, slots):
        """uniform_adaln, a uniform tile: ``firsts[i]`` (``[1, H]`` f32 VMEM) =
        table ``slots[i]``'s row every row of the tile reads (``shared_rows``'
        one lookup)."""
        for i, slot in enumerate(slots):
            firsts[i] = lookup(onehot, buffer, slot, pl.ds(0, 16))[0:1]

    def modulate(src, dst, onehot, buffer, weight, count, rows=None):
        """``dst = bf16(rms(src) * weight * (1 + scale) + shift)``, in chunks.

        ``rows``: the ``[1, H]`` f32 refs of a shift and scale every row shares
        (``shared_rows``, ``fill``), in place of each chunk's lookups.
        """

        def one(rs):
            normed = ref.rms(src[rs, :].astype(_F32), config.eps)
            normed *= weight[...].astype(_F32)
            if rows is None:
                shift, scale_ = (lookup(onehot, buffer, k, rs) for k in (0, 1))
            else:
                shift, scale_ = (row[...] for row in rows)
            dst[rs, :] = (normed * (1.0 + scale_) + shift).astype(dst.dtype)

        _for_chunks(count, chunk, one)

    def modulate_slabs(src, dst, weight, rows, factors, count, base=0):
        """``modulate`` of rows ``[base, base + count)`` that share one shift
        and scale (``rows``), every chunk unrolled into the caller's basic
        block, the same values in an order that block can overlap with a
        matmul: per chunk, ``ref.rms``'s factor of each row into ``factors``
        (``[T, 1]`` f32 VMEM; the same sum of squares, lowered alike), then the
        elementwise pass in lane slabs, each slab's weight, shift and scale
        loaded beside it. ``modulate`` holds a chunk's whole rows and broadcast
        parameters instead, whose spills keep the scheduler from interleaving
        it with the matmul. ``base`` may be traced (a multiple of the chunk).
        """
        width = tl.divisor(hidden, 256, 128)
        shift, scale_ = rows

        def one(rs):
            x = src[rs, :].astype(_F32)
            # ref.rms(x) = x * this factor.
            factors[rs, :] = lax.rsqrt(
                jnp.mean(x * x, axis=-1, keepdims=True) + config.eps
            )
            for lane in range(0, hidden, width):
                ls = pl.ds(lane, width)
                normed = src[rs, ls].astype(_F32) * factors[rs, :]
                normed *= weight[:, ls].astype(_F32)
                dst[rs, ls] = (normed * (1.0 + scale_[:, ls]) + shift[:, ls]).astype(
                    dst.dtype
                )

        for index in range(count // chunk):
            start = base + index * chunk
            if not isinstance(start, int):
                start = pl.multiple_of(start, chunk)
            one(pl.ds(start, chunk))

    def sub_rows(tile):
        return tl.divisor(tile, tiling.sub_rows, 16)

    def rope(value, cos, sin):
        lane = lax.broadcasted_iota(jnp.int32, value.shape, 1)
        half = config.rope_dims // 2
        # Rolling by -half brings x2 under x1's lanes; by +half, x1 under x2's.
        ahead = pltpu.roll(value, dim - half, 1)
        behind = pltpu.roll(value, half, 1)
        rotated = jnp.where(
            lane < half, -ahead, jnp.where(lane < config.rope_dims, behind, 0.0)
        )
        return value * cos + rotated * sin

    def layer_norms(parity):
        """This layer's ``(norms, qk_norms)`` VMEM: ``[2, 1, H]`` norm1, norm2
        and ``[2, 1, D]`` q_norm, k_norm (the slot of the layer's parity)."""
        return norms.at[parity], qk_norms.at[parity]

    # step_tables: SMEM tables of phase A's step scalars, filled once a call
    # (``with_step_tables``): per step its row tile, its (rotated and raw) head
    # group and its staging slot; per head its owner and local index there.
    tables_at: dict[str, Any] = {}

    def with_step_tables(body):
        """``body()`` with phase A's step tables in SMEM (step_tables), else
        ``body()``."""
        if not step_tables:
            body()
            return
        tile, slots = tiling.qkv_rows, tiling.stage_slots
        groups = heads // group
        steps = rows // tile * groups
        offset = me * groups // devices

        def scoped(step_ref, head_ref):
            def fill(step, carry):
                raw = step % groups
                step_ref[step] = (
                    step // groups
                    | ((raw + offset) % groups) << 8
                    | raw << 16
                    | (step % slots) << 24
                )
                return carry

            lax.fori_loop(0, steps, fill, None)
            for head in range(heads):
                head_ref[head] = head // local_heads | (head % local_heads) << 8
            tables_at["steps"], tables_at["heads"], tables_at["n"] = (
                step_ref,
                head_ref,
                steps,
            )
            body()

        pl.run_scoped(
            scoped, pltpu.SMEM((steps,), jnp.int32), pltpu.SMEM((heads,), jnp.int32)
        )

    def step_word(step):
        """step_tables: the word of ``step`` (clamped into the table: a guarded
        step outside it reads its neighbour, never used)."""
        return tables_at["steps"][jnp.clip(step, 0, tables_at["n"] - 1)]

    # ------------------------------------------------------------ phase A
    def phase_a(layer: jax.Array, parity: jax.Array, with_peaks: bool) -> None:
        tile, slots = tiling.qkv_rows, tiling.stage_slots
        groups = heads // group
        tiles, steps = rows // tile, rows // tile * groups
        # Each device walks the head groups from its own offset, so at any step
        # the devices send to different owners instead of all into one.
        offset = me * groups // devices

        # Each step's epilogue runs under the next step's matmul (double
        # buffered acc, cos, sin); each row tile's cos, sin and one-hot load a
        # row tile ahead, cos and sin into the other of two buffers.
        norm_ref, qk_norm_ref = layer_norms(parity)
        # meta's row_tile_modes, after the head words.
        modes_at = 1 + 3 * layers
        # Row tile i + 1 modulates in ``parts`` pieces of whole chunks, from
        # step ``ahead_at`` of row tile i's head groups on (``with_ahead``):
        # halfway, by when its x and one-hot (started at row tile i's first
        # step) have landed, and before row tile i ends (modulate_pieces: that
        # many pieces).
        parts = modulate_pieces or (2 if groups > 1 and tile // chunk % 2 == 0 else 1)
        ahead_at = min(groups // 2, groups - parts)

        def scoped(
            h, w, acc, stage, cos, sin, onehot, tables, sems, w_sems, shared, factors
        ):
            # shared: the shift and scale rows a row tile shares ([2, 1, H] f32,
            # shared_rows); factors: a row tile's rms factors ([tile, 1] f32,
            # modulate_slabs).

            def start_next(index, copy_of):
                """Starts ``copy_of()``, a copy of row tile ``index + 1``, if that
                tile exists: a static ``index`` decides here, so a device of one row
                tile builds no copy past its rows."""
                if isinstance(index, numbers.Integral):
                    if index + 1 < tiles:
                        copy_of().start()
                    return
                pl.when(index + 1 < tiles)(copy_of().start)

            def place(step):
                """Step's row tile and head group (a static step: static)."""
                if step_tables and not isinstance(step, numbers.Integral):
                    word = step_word(step)
                    return word & 0xFF, (word >> 8) & 0xFF
                return step // groups, (step % groups + offset) % groups

            def raw_group(step):
                """``step % groups`` (its unrotated head group)."""
                if step_tables and not isinstance(step, numbers.Integral):
                    return (step_word(step) >> 16) & 0xFF
                return step % groups

            def stage_slot(step):
                """``step % slots``."""
                if step_tables and not isinstance(step, numbers.Integral):
                    return (step_word(step) >> 24) & 0xFF
                return step % slots

            def x_copy(index):
                rs = pl.ds(index * tile, tile)
                return copy(x_ref.at[rs], h.at[index % 2], sems.at[index % 2])

            def kv_transfer(step, j):
                """Head j of step's group: q, k, v -> its owner (``head_spot``), at
                my rows."""
                index, first_head = place(step)
                head, slot = first_head * group + j, stage_slot(step)
                spot = head  # its owner and local head there: owner_local
                if bulk_exchange:
                    return bulk_transfer(
                        stage.at[slot, j],
                        parity,
                        spot,
                        index * tile,
                        tile,
                        sems.at[5 + slot * group + j],
                        kv_send.at[slot * group + j],
                    )
                rs = pl.ds(me * rows + index * tile, tile)
                owner, local = owner_local(spot)
                return transfer(
                    stage.at[slot, j],
                    kv_ref.at[parity, local, :, rs],
                    sems.at[5 + slot * group + j],
                    kv_send.at[slot * group + j],
                    kv_recv.at[parity],
                    owner,
                )

            def side_copies(index, side):
                """Row tile ``index``'s cos, sin and one-hot, into ``side``."""
                rs = pl.ds(index * tile, tile)
                return [
                    copy(cos_ref.at[rs], cos.at[side], sems.at[2]),
                    copy(sin_ref.at[rs], sin.at[side], sems.at[3]),
                    copy(onehot_ref.at[rs], onehot, sems.at[4]),
                ]

            def prologue():
                picks = (_SHIFT_MSA, _SCALE_MSA)
                table_sems = sems.at[pl.ds(5 + slots * group, 2)]
                table_copies = load_tables(layer, picks, tables, table_sems)
                # Row tile 0 (x, cos, sin, one-hot) streams in with the tables;
                # next_rows waits for it.
                x_copy(0).start()
                for one in side_copies(0, 0):
                    one.start()
                for one in table_copies:
                    one.wait()

            def prefetch_rows(step):
                """At a row tile's first step, after the epilogue of the step
                before it (the last to read the buffers), the next row
                tile's cos and sin start in (its one-hot in ``next_rows``)."""
                index, _ = place(step)

                @pl.when((raw_group(step) == 0) & (index + 1 < tiles))
                def _prefetch():
                    for one in side_copies(index + 1, (index + 1) % 2)[:2]:
                        one.start()

            def epilogue(acc_v, cos_v, sin_v, slot, first_head, rs):
                """Rounds q, k, v; norms and rotates q and k, as the reference (the
                RoPE tables and norm weights cast once a call instead of once a
                head, v cast to bf16 once: the same values).
                """
                qkv = acc_v[rs, :]  # q and k round below, v on its way to bf16
                # q carries the softmax scale, so scores come out in log2 units.
                norm_weights = (
                    qk_norm_ref[0].astype(_F32) * scale,
                    qk_norm_ref[1].astype(_F32),
                )
                rope_tables = (cos_v[rs, :].astype(_F32), sin_v[rs, :].astype(_F32))
                lane = lax.broadcasted_iota(jnp.int32, (1, TABLE_LANES), 1)
                for j in range(group):
                    base = j * 3 * dim
                    for part in (0, 1):
                        value = qkv[:, base + part * dim : base + (part + 1) * dim]
                        value = ref.rms(ref.round_bf16(value), config.qk_eps)
                        value = rope(
                            ref.round_bf16(value * norm_weights[part]), *rope_tables
                        )
                        value = value.astype(_BF16)
                        stage[slot, j, part, rs, :] = value
                        if not with_peaks:  # the norm weights already bound this layer
                            continue
                        # The largest squared norm over these rows, at head's lane.
                        value = value.astype(_F32)
                        peak = jnp.sum(value * value, axis=1, keepdims=True).max(
                            axis=0, keepdims=True
                        )
                        hit = lane == first_head * group + j
                        peaks[part, 0:1, :] = jnp.maximum(
                            peaks[part, 0:1, :], jnp.where(hit, peak, 0.0)
                        )
                    v_part = qkv[:, base + 2 * dim : base + 3 * dim]
                    stage[slot, j, 2, rs, :] = v_part.astype(_BF16)

            def next_rows(step, modulated=False):
                """Step's row tile: its x, cos, sin and one-hot in; modulate.

                The cos, sin and one-hot copies started a row tile back
                (``prefetch_rows``, the one-hot here once modulated: its one
                buffer is then free); a tile whose rows share one AdaLN row
                modulates with one lookup of it (``row_tile_modes``).
                ``modulated``: the row tiles after the first, whose x
                and one-hot ``with_ahead`` waited for and modulated a row tile back:
                here only their cos and sin land and the next row tile's x and
                one-hot start (the one-hot buffer is free since that modulate).
                """
                index, _ = place(step)
                side = index % 2
                if not modulated:
                    x_copy(index).wait()
                start_next(index, lambda: x_copy(index + 1))
                # All started ahead (the prologue, prefetch_rows, below).
                loads = side_copies(index, side)

                def next_onehot():
                    return side_copies(index + 1, (index + 1) % 2)[2]

                for one in loads[:2] if modulated else loads:
                    one.wait()
                if modulated:
                    start_next(index, next_onehot)
                    return
                view = h.at[index % 2]

                def modulate_tile(rows_=None):
                    modulate(view, view, onehot, tables, norm_ref.at[0], tile, rows_)

                mode = meta_ref[modes_at + index]
                pl.when(mode == 1)(lambda: shared_rows(onehot, tables, shared))
                pl.when(mode == 0)(modulate_tile)
                pl.when(mode != 0)(lambda: modulate_tile((shared.at[0], shared.at[1])))
                start_next(index, next_onehot)

            def rows_when(step, modulated=False):
                @pl.when(raw_group(step) == 0)
                def _next_rows():
                    with scope("A.rows", 2):
                        next_rows(step, modulated)

            def with_ahead(step, main):
                """``main()`` (the step's matmul block); at steps ``ahead_at`` to
                ``ahead_at + parts - 1`` of row tile i, also a piece of row tile
                i + 1's modulate each.

                At the first, its x and one-hot have landed (started at row tile
                i's first step, the one-hot once row tile i's modulate freed its
                buffer) and a tile in mode 1 looks its shared row up. A tile whose
                rows share one row (modes 1, 2) modulates in ``main``'s basic block,
                unrolled after the matmul (``modulate_slabs``), so part of the VPU
                work runs under the MXU's; the second copy of the block without it
                serves every other step. A tile of per-row lookups (mode 0)
                modulates whole after the first block, its one-hot matmuls after
                the QKV matmul as before (the MXUs split that one alike).
                """
                index = place(step)[0]
                nxt = index + 1
                part = raw_group(step) - ahead_at
                now = (part >= 0) & (part < parts) & (nxt < tiles)
                if isinstance(now, bool) and not now:  # a static step elsewhere
                    main()
                    return
                first = (part == 0) & (nxt < tiles)
                mode = meta_ref[modes_at + jnp.minimum(nxt, tiles - 1)]
                view = h.at[nxt % 2]

                @pl.when(first)
                def _land():
                    x_copy(nxt).wait()
                    side_copies(nxt, nxt % 2)[2].wait()
                    pl.when(mode == 1)(lambda: shared_rows(onehot, tables, shared))

                def main_modulated():
                    main()
                    modulate_slabs(
                        view,
                        view,
                        norm_ref.at[0],
                        (shared.at[0], shared.at[1]),
                        factors,
                        tile // parts,
                        part * (tile // parts),
                    )

                shared_now = now & (mode != 0)
                pl.when(shared_now)(main_modulated)
                pl.when(jnp.logical_not(shared_now))(main)
                pl.when(first & (mode == 0))(
                    lambda: modulate(view, view, onehot, tables, norm_ref.at[0], tile)
                )

            def free_stage(step):
                """The staging slot of ``step`` is free once the sends ``slots``
                steps back have left; the ICI drains them while later steps
                compute."""

                @pl.when(step >= slots)
                def _free_stage():
                    for j in range(group):
                        kv_transfer(step - slots, j)[1]()

            def epilogue_step(step):
                """Step's epilogue, unrolled into the caller's block (it overlaps
                the next step's matmul). epilogue_chunk: in chunks of that
                many rows and a shorter last one."""
                index, first_head = place(step)
                epi = (
                    (epilogue_chunk_peaks if with_peaks else 0)
                    or epilogue_chunk
                    or tiling.epilogue_rows
                )
                for at in range(0, tile, epi):
                    epilogue(
                        acc.at[step % 2],
                        cos.at[index % 2],
                        sin.at[index % 2],
                        stage_slot(step),
                        first_head,
                        pl.ds(at, min(epi, tile - at)),
                    )

            def send_step(step):
                for j in range(group):
                    kv_transfer(step, j)[0]()

            def finish_step(step):
                epilogue_step(step)
                send_step(step)

            def fetch(step, slot):
                return [
                    copy(
                        wqkv_ref.at[layer, place(step)[1]], w.at[slot], w_sems.at[slot]
                    )
                ]

            def pipelined_stream():
                """The weight-tile ring, each step's matmul with the previous step's
                epilogue in its basic block."""
                for one in fetch(0, 0):
                    one.start()
                prologue()
                for one in fetch(0, 0):
                    one.wait()
                if steps > 1:
                    for one in fetch(1, 1):
                        one.start()
                rows_when(0)
                with_ahead(
                    0, lambda: matmul(h.at[0], w.at[0], acc.at[0], tile, sub_rows(tile))
                )
                prefetch_rows(0)

                def step_body(step, carry):
                    slot = step % 2
                    for one in fetch(step, slot):
                        one.wait()

                    @pl.when(step + 1 < steps)
                    def _next():
                        for one in fetch(step + 1, 1 - slot):
                            one.start()

                    free_stage(step - 1)
                    rows_when(step, modulated=True)
                    index, _ = place(step)

                    def main():
                        # The epilogue first: the scheduler fills the matmul around it.
                        epilogue_step(step - 1)
                        matmul(
                            h.at[index % 2],
                            w.at[slot],
                            acc.at[step % 2],
                            tile,
                            sub_rows(tile),
                        )

                    with_ahead(step, main)
                    send_step(step - 1)
                    prefetch_rows(step)
                    return carry

                lax.fori_loop(1, steps, step_body, None)
                free_stage(steps - 1)
                finish_step(steps - 1)

            with scope("A.stream", 2):
                pipelined_stream()
            with scope("A.drain"):
                for step in range(max(steps - slots, 0), steps):
                    for j in range(group):
                        kv_transfer(step, j)[1]()
            # Every other device sends each of my heads its row tiles
            # (bulk_exchange: only the first head of my order; the rest come in
            # phase B).
            with scope("A.await"):
                peers = rows * (devices - 1)  # the rows a head gets from peers
                if bulk_exchange:
                    await_all(
                        kv_ref.at[parity, 0, :, pl.ds(0, peers)],
                        kv_send.at[0],
                        kv_recv.at[parity, orders[me, 0]],
                        kv_ref.at[parity, 0, :, pl.ds(0, tile)],
                        tiles * (devices - 1),
                    )
                else:
                    await_all(
                        kv_ref.at[parity, :, :, pl.ds(0, peers)],
                        kv_send.at[0],
                        kv_recv.at[parity],
                        kv_ref.at[parity, 0, :, pl.ds(0, tile)],
                        tiles * local_heads * (devices - 1),
                    )

        pl.run_scoped(
            scoped,
            pltpu.VMEM((2, tile, hidden), _BF16),
            pltpu.VMEM((2, hidden, group * 3 * dim), wqkv_ref.dtype),
            pltpu.VMEM((2, tile, group * 3 * dim), _F32),
            pltpu.VMEM((slots, group, 3, tile, dim), _BF16),
            pltpu.VMEM((2, tile, dim), _BF16),
            pltpu.VMEM((2, tile, dim), _BF16),
            pltpu.VMEM((tile, TABLE_LANES), _BF16),
            pltpu.VMEM((2, TABLE_LANES, hidden), _BF16),
            # x slots, cos, sin, one-hot, staging (slot, head), two tables.
            pltpu.SemaphoreType.DMA((7 + slots * group,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((2, 1, hidden), _F32),
            pltpu.VMEM((tile, 1), _F32),
        )

    # ------------------------------------------------------------ phase B
    # pylint: disable-next=too-many-statements
    # pylint: disable-next=too-many-arguments
    def phase_b(
        layer: jax.Array,
        parity: jax.Array,
        fixed: bool,
        item_lo: jax.Array,
        item_hi: jax.Array,
        final: bool,
    ) -> None:
        """Items ``[item_lo, item_hi)`` (in ``head_order``) at one softmax; the
        ``final`` call also waits for every peer's output tiles."""
        # sum_bounds: the norm weights bound this layer (no sums recorded).
        proven = (meta_ref[1 + layer] & 32) != 0
        # kv_group (the dense walk only; fixed-max, online): each DMA and key
        # step moves that many of the tiling's key tiles, run as the same
        # sub-blocks.
        block_q = tiling.block_q
        block_kv = tiling.block_kv * kv_group[0 if fixed else 1]
        sub_kv = tl.divisor(
            tiling.block_kv,
            tiling.block_kv_compute if fixed else tiling.block_kv_compute_online,
            16,
        )
        q_tiles = rows * devices // block_q
        # A tile that does not divide a device's rows crosses one boundary, at
        # its middle (tl.Tiling.fit): its output goes out in two halves.
        pieces = 1 if rows % block_q == 0 else 2
        unit = block_q // pieces
        kv_tiles = -(-rows * devices // block_kv)
        real_tiles = (used + block_kv - 1) // block_kv
        # Each head walks the query tiles from just past my own rows, so its
        # last outputs (and the layer's, waited on last) stay on this device.
        if pieces == 1:
            shift = (me + 1) * (rows // block_q) % q_tiles
        else:  # the tile after the one that holds my last row
            shift = (((me + 1) * rows - 1) // block_q + 1) % q_tiles

        def q_tile_of(item):
            return (item % q_tiles + shift) % q_tiles

        if bulk_exchange:
            # While item k q_tiles + t runs, chunk t of my rows of the next head
            # (k + 1) in each peer's order goes to that peer: the chunks span the
            # first half of head k's items, so a peer half a head behind still
            # lands them before head k + 1 starts.
            bulk_rows = min(
                rows_
                for rows_ in range(16, rows + 1, 16)
                if rows % rows_ == 0 and rows // rows_ <= max(1, q_tiles // 2)
            )
            chunks = rows // bulk_rows

        # split_items: the dense walk's items in two loops of one query width
        # each, every head's full-width items in head_order, then each head's
        # tail item. Walk position ``at`` runs item ``item_of(at)``: the query,
        # stage and output slots, the waits and the primes go by position, the
        # head and the query tile by item.
        split = bool(split_items and static_used and tail_rows and q_tiles > 1)
        if split:
            full = q_tiles - 1  # full-width items a head
            middle = item_lo + (item_hi - item_lo) // q_tiles * full
            # The tail query tile's place in a head's walk (``shift``).
            tail_at = ((static_used - 1) // block_q + q_tiles - shift) % q_tiles

        def split_place(at):
            """split_items: walk position ``at``'s item, its head's place in
            head_order, the item's rank among that head's full-width items, and
            whether ``at`` is in the full-width loop."""
            j = at - item_lo
            in_full = at < middle
            rank = j % full
            head = jnp.where(
                in_full,
                item_lo // q_tiles + j // full,
                item_lo // q_tiles + (at - middle),
            )
            tile_at = jnp.where(
                in_full, rank + (rank >= tail_at).astype(jnp.int32), tail_at
            )
            return head * q_tiles + tile_at, head, rank, in_full

        def item_of(at):
            """The item walk position ``at`` runs: ``at`` itself, or under
            split_items its place in the two loops."""
            return split_place(at)[0] if split else at

        # pylint: disable-next=too-many-statements
        def scoped(q, k, m_run, l_run, o_run, stage, sems, *extra):
            # sum_bounds: a copy of an item's final sum and max rows.
            if sum_bounds:
                (sums_copy,) = extra

            def kv_copies(head, index, slot):
                """The K and V rows of key tile ``index`` in one strided copy."""
                rs = pl.ds(index * block_kv, block_kv)
                return (
                    copy(
                        kv_ref.at[parity, head, pl.ds(1, 2), rs],
                        k.at[slot],
                        sems.at[slot],
                    ),
                )

            def keys_of(slot, keys):
                return k[slot, 0, keys, :]

            def values_of(slot, keys):
                return k[slot, 1, keys, :]

            def q_copy(item, at=None):
                """Item's query tile into the query slot of walk position ``at``
                (None: ``item``)."""
                at = item if at is None else at
                head, q_start = head_order[item // q_tiles], q_tile_of(item) * block_q
                window = kv_ref.at[parity, head, 0, pl.ds(q_start, block_q)]
                return copy(window, q.at[at % 2], sems.at[6 + at % 2])

            def span(item):
                """An item's head, query tile, whether it is real, its first key
                tile and its count of key tiles.

                A query tile with any real row sees the real keys, ``[0, used)``,
                so its padding rows (never read) get finite garbage; a tile of
                padding rows sees the padding keys.
                """
                q_tile = q_tile_of(item)
                head, q_start = head_order[item // q_tiles], q_tile * block_q
                real = q_start < used
                lo = jnp.where(real, 0, used // block_kv)
                hi = jnp.where(real, real_tiles, kv_tiles)
                return head, q_tile, real, lo, hi - lo

            def tile(lo, index):
                """The key tile an item whose walk starts at ``lo`` visits
                ``index``-th."""
                return lo + index

            def out_transfer(item, at=None):
                """Item's output tile -> the device(s) that own its rows, at its
                head's columns (``computed_head``): one piece, or two halves when
                the tile crosses a device boundary there. From the stage slot of
                walk position ``at`` (None: ``item``)."""
                head, start = head_order[item // q_tiles], q_tile_of(item) * block_q
                slot = (item if at is None else at) % 2
                parts = []
                for piece in range(pieces):
                    first = start + piece * unit
                    owner = first // rows
                    window = attn_ref.at[
                        parity,
                        pl.ds(first - owner * rows, unit),
                        pl.ds(computed_head(layer, head) * dim, dim),
                    ]
                    parts.append(
                        transfer(
                            stage.at[slot, pl.ds(piece * unit, unit)],
                            window,
                            sems.at[4 + slot],
                            attn_send.at[slot],
                            attn_recv.at[parity],
                            owner,
                        )
                    )

                def start_all():
                    for begin, _ in parts:
                        begin()

                def wait_all():
                    for _, end in parts:
                        end()

                return start_all, wait_all

            def prime(item, slot, at=None):
                """Starts an item's query tile (into walk position ``at``'s slot) and
                its first key tile into ``slot``."""
                head, _, _, lo, _ = span(item)
                q_copy(item, at).start()
                for one in kv_copies(head, tile(lo, 0), slot):
                    one.start()

            def bulk_copy(owner, head, chunk):
                """bulk_exchange: chunk ``chunk`` of my rows of ``owner``'s local
                ``head``, kv_mine -> the owner's kv_ref."""
                rs = pl.ds(chunk * bulk_rows, bulk_rows)
                return pltpu.make_async_remote_copy(
                    kv_mine_ref.at[owner, head, :, rs],
                    kv_ref.at[
                        parity, head, :, pl.ds(me * rows + chunk * bulk_rows, bulk_rows)
                    ],
                    bulk_send.at[0],
                    kv_recv.at[parity, head],
                    device_id=device(owner),
                )

            def bulk_sends(item, also=None):
                """bulk_exchange: ``(start, wait)`` of item ``k q_tiles + t``'s
                sends, chunk ``t`` of head ``k + 1`` of every peer's order (only
                where ``also`` holds, if given)."""
                k, t = item // q_tiles, item % q_tiles
                live = (k + 1 < local_heads) & (t < chunks)
                if also is not None:
                    live = live & also

                def start():
                    for offset in range(1, devices):
                        owner = (me + offset) % devices
                        bulk_copy(owner, orders[owner, k + 1], t).start()

                def wait():
                    # Every send is one chunk: any descriptor of one waits for it.
                    for _ in range(1, devices):
                        bulk_copy(me, 0, 0).wait_send()

                return (lambda: pl.when(live)(start)), (lambda: pl.when(live)(wait))

            def await_head(item, first=None):
                """bulk_exchange, before ``item``'s prime: the first item of a later
                head in my order waits for every peer's rows of that head.
                ``first``: whether ``item`` is its head's first (None: its first
                query tile)."""
                first = item % q_tiles == 0 if first is None else first

                @pl.when(first & (item >= q_tiles))
                def _await():
                    head = head_order[item // q_tiles]
                    with scope("B.head"):
                        await_all(
                            kv_ref.at[parity, head, :, pl.ds(0, rows * (devices - 1))],
                            bulk_send.at[0],
                            kv_recv.at[parity, head],
                            kv_ref.at[parity, head, :, pl.ds(0, bulk_rows)],
                            chunks * (devices - 1),
                        )

            def sends(at):
                """bulk_exchange: the sends of walk position ``at``: item ``at``'s,
                or under split_items chunk ``r`` of head ``k + 1`` at head k's r-th
                full-width item (chunks <= full: every chunk before head k + 1's
                first item), none at a tail item."""
                if not split:
                    return bulk_sends(at)
                _, head, rank, in_full = split_place(at)
                return bulk_sends(head * q_tiles + rank, in_full)

            def await_at(at):
                """bulk_exchange: ``await_head`` before priming walk position
                ``at``; under split_items its head's first item is its first
                full-width one."""
                if not split:
                    await_head(at)
                    return
                item, _, rank, in_full = split_place(at)
                await_head(item, in_full & (rank == 0))

            # The cut key tile's sub-blocks that hold real keys (static used);
            # the rest of that tile is padding no real row sees.
            subs_cut = cut_subs[0 if fixed else 1] or block_kv // sub_kv
            # Of those, the leading ones that hold real keys only: a real item
            # runs them unmasked.
            full_cut = cut_full[0 if fixed else 1]

            # pylint: disable-next=too-many-arguments
            def update(
                at,
                slot,
                k_start,
                real,
                masked,
                width=block_q,
                subs=None,
                first=False,
                clear=0,
                last_keys=0,
            ):
                """One key tile, as ``block_kv / sub_kv`` sub-blocks (the first
                ``subs``), for the first ``width`` query rows of the query slot of
                walk position ``at``; ``masked`` masks every sub-block but the first
                ``clear``. ``last_keys``
                (cut_keys): the last sub-block runs only its first that many keys.

                The item's ``first`` tile, its state unset, writes the running
                state from its first sub-block instead of adding to a zero
                fill, and online the running max and sum stay in registers through
                the sub-blocks, loaded and stored once a tile. The same bits: ``0 +
                x`` is ``x`` (no sum of probabilities is -0), and online the first
                sub-block's max starts from the mask value as on the zero fill, its
                alpha scaling only zeros.
                """
                cols = slice(0, width)
                shape = (l_run.shape[0], width)
                carry = not fixed
                m = l = None
                if carry and not first:
                    m, l = m_run[:, cols], l_run[:, cols]
                n_subs = subs or block_kv // sub_kv

                def size(start):
                    """Keys of the sub-block at ``start`` (a static start)."""
                    if last_keys and start == (n_subs - 1) * sub_kv:
                        return last_keys
                    return sub_kv

                def qk(start):
                    # Scores transposed, [sub_kv, block_q], in log2 units: the softmax
                    # reductions run down sublanes; the running max and sum are lane
                    # rows.
                    return lax.dot_general(
                        keys_of(slot, pl.ds(start, size(start))),
                        q[at % 2, cols, :],
                        _CONTRACT_LAST,
                        preferred_element_type=_F32,
                    )

                def one(start, scores, masked, fresh, m, l):
                    """Sub-block ``start``'s update from its ``scores``: the new
                    carried (m, l) of the online softmax, else (None, None)."""
                    keys = pl.ds(start, size(start))
                    if masked:
                        # Real tiles see keys below ``used``, padding tiles the rest.
                        key = lax.broadcasted_iota(jnp.int32, (size(start), 1), 0)
                        visible = (key + k_start + start < used) == real
                    if fixed:
                        bias = fixed_max
                        if masked and mask_scores:
                            # A masked key subtracts +inf instead (a per-key column):
                            # exp2(-inf) is the select's +0, every visible score the same
                            # difference, so the same probabilities with no select.
                            probs = jnp.exp2(scores - jnp.where(visible, bias, jnp.inf))
                        else:
                            probs = jnp.exp2(scores - bias)
                            if masked:
                                probs = jnp.where(visible, probs, 0.0)
                        weights_bf16 = probs.astype(_BF16)
                        if fresh:
                            # The first sub-block writes, the rest add in VMEM as below:
                            # here a sum carried in registers schedules worse.
                            l_run[:, cols] = jnp.broadcast_to(
                                probs.sum(axis=0, keepdims=True), shape
                            )
                            o_run[:, cols] = lax.dot_general(
                                values_of(slot, keys),
                                weights_bf16,
                                _CONTRACT_FIRST,
                                preferred_element_type=_F32,
                            )
                            return m, l
                        l_run[:, cols] += probs.sum(axis=0, keepdims=True)
                        # Every sub-block start is a static int (the sub-blocks are
                        # unrolled), so this rounding guard sees every key tile.
                        if start and start % tiling.block_kv == 0:
                            # kv_group: a later key tile of a group adds its P V to the
                            # output as its own key step would, not summed with the tile
                            # before it in the MXU first (the compiler folds o + a + b into
                            # o + (a + b)): an identity select it cannot see through.
                            product = lax.dot_general(
                                values_of(slot, keys),
                                weights_bf16,
                                _CONTRACT_FIRST,
                                preferred_element_type=_F32,
                            )
                            o_run[:, cols] = (
                                jnp.where(used > 0, o_run[:, cols], 0.0) + product
                            )
                            return m, l
                        o_run[:, cols] += lax.dot_general(
                            values_of(slot, keys),
                            weights_bf16,
                            _CONTRACT_FIRST,
                            preferred_element_type=_F32,
                        )
                        return m, l
                    # Masked keys never enter the max: a fused multiply-add would turn
                    # ``huge - huge`` into its residual.
                    if masked and mask_scores:
                        # s - 0 is s, s - inf is -inf (scores are finite): the select's
                        # values from a per-key column; exp2(top - m) is then the
                        # probabilities' select too (m is finite: at least the mask
                        # value), so a masked sub-block has no select left.
                        top = scores - jnp.where(visible, 0.0, jnp.inf)
                    else:
                        top = jnp.where(visible, scores, -jnp.inf) if masked else scores
                    live = top if masked and mask_scores else scores
                    if carry:
                        m_prev = jnp.full(shape, _MASK_VALUE, _F32) if fresh else m
                        m = jnp.maximum(m_prev, top.max(axis=0, keepdims=True))
                        probs = jnp.exp2(live - m[0:1])
                        if masked and not mask_scores:
                            probs = jnp.where(visible, probs, 0.0)
                        total = probs.sum(axis=0, keepdims=True)
                        product = lax.dot_general(
                            values_of(slot, keys),
                            probs.astype(_BF16),
                            _CONTRACT_FIRST,
                            preferred_element_type=_F32,
                        )
                        if fresh:
                            l = jnp.broadcast_to(total, shape)
                            o_run[:, cols] = product
                        else:
                            alpha = jnp.exp2(m_prev - m)
                            l = alpha * l + total
                            o_run[:, cols] = alpha[0:1] * o_run[:, cols] + product
                        return m, l
                    m_prev = m_run[:, cols]
                    m_next = jnp.maximum(m_prev, top.max(axis=0, keepdims=True))
                    probs = jnp.exp2(live - m_next[0:1])
                    if masked and not mask_scores:
                        probs = jnp.where(visible, probs, 0.0)
                    alpha = jnp.exp2(m_prev - m_next)
                    l_run[:, cols] = alpha * l_run[:, cols] + probs.sum(
                        axis=0, keepdims=True
                    )
                    # V^T P: [D, block_q], the output transposed like the scores.
                    o_run[:, cols] = alpha[0:1] * o_run[:, cols] + lax.dot_general(
                        values_of(slot, keys),
                        probs.astype(_BF16),
                        _CONTRACT_FIRST,
                        preferred_element_type=_F32,
                    )
                    m_run[:, cols] = m_next
                    return m, l

                # Online: the next sub-block's scores are issued before this one's
                # max, so the MXU runs under the reductions.
                ahead = not fixed
                starts = range(0, n_subs * sub_kv, sub_kv)
                pending = qk(starts[0]) if ahead and len(starts) else None
                for index, start in enumerate(starts):
                    if ahead:
                        scores = pending
                        if index + 1 < len(starts):
                            pending = qk(starts[index + 1])
                    else:
                        scores = qk(start)
                    m, l = one(
                        start,
                        scores,
                        masked and start // sub_kv >= clear,
                        first and start == 0,
                        m,
                        l,
                    )
                if carry:
                    m_run[:, cols], l_run[:, cols] = m, l

            def visit(item, slot, k_start, real, masked, fresh=False):
                """``update`` with the traced ``masked``, on the last real query
                tile's ``tail_rows`` rows only when it has no more real ones.

                The item's first tile where ``fresh`` (True, or traced: both
                kinds of ``update``, each where it holds).
                """
                tail = (q_tile_of(item) == (used - 1) // block_q) if tail_rows else None
                narrow = subs_cut < block_kv // sub_kv
                split = narrow or full_cut > 0
                firsts = (fresh,) if isinstance(fresh, bool) else (True, False)
                for cut in (True, False):
                    for width in (block_q, tail_rows) if tail_rows else (block_q,):
                        for short in (True, False) if cut and split else (False,):
                            when = masked if cut else ~masked
                            if tail is not None:
                                when = when & (tail if width == tail_rows else ~tail)
                            if cut and split:
                                # A real tile's cut tile: only its real sub-blocks, the
                                # leading all-real ones unmasked.
                                when = when & (real if short else ~real)
                            args = (
                                item,
                                slot,
                                k_start,
                                real,
                                cut,
                                width,
                                subs_cut if short and narrow else None,
                            )
                            clear = full_cut if short else 0
                            for first in firsts:
                                gate = when
                                if len(firsts) > 1:
                                    gate = gate & (fresh if first else ~fresh)
                                pl.when(gate)(
                                    functools.partial(
                                        update, *args, first=first, clear=clear
                                    )
                                )

            def zero_runs(item):
                """The running state an item starts from: the max at the mask value,
                the sum and the output zero."""
                if not fixed:
                    m_run[...] = jnp.full(m_run.shape, _MASK_VALUE, _F32)
                l_run[...] = jnp.zeros(l_run.shape, _F32)
                o_run[...] = jnp.zeros(o_run.shape, _F32)
                if tail_rows:
                    # The tail tile's rows past tail_rows (padding) never update: a
                    # finite 0 / 1 for them.

                    def pad_sums():
                        l_run[:, tail_rows:] = jnp.ones(
                            (l_run.shape[0], block_q - tail_rows), _F32
                        )

                    pl.when(q_tile_of(item) == (used - 1) // block_q)(pad_sums)

            def zero_state(item):
                """zero_ahead: ``zero_runs(item)`` with no branch (the tail tile's
                padding sums by a select), so it shares the previous item's
                epilogue block: the same state."""
                if not fixed:
                    m_run[...] = jnp.full(m_run.shape, _MASK_VALUE, _F32)
                l_run[...] = jnp.zeros(l_run.shape, _F32)
                o_run[...] = jnp.zeros(o_run.shape, _F32)
                if tail_rows:
                    tail = q_tile_of(item) == (used - 1) // block_q
                    l_run[:, tail_rows:] = jnp.where(
                        tail,
                        jnp.ones((l_run.shape[0], block_q - tail_rows), _F32),
                        jnp.zeros((l_run.shape[0], block_q - tail_rows), _F32),
                    )

            def pad_tail():
                """``zero_runs``'s state of the tail tile's padding rows, which no
                update writes."""
                pad = (l_run.shape[0], block_q - tail_rows)
                if not fixed:
                    m_run[:, tail_rows:] = jnp.full(pad, _MASK_VALUE, _F32)
                l_run[:, tail_rows:] = jnp.ones(pad, _F32)
                o_run[:, tail_rows:] = jnp.zeros((dim, block_q - tail_rows), _F32)

            def item_body(item: jax.Array, base: jax.Array) -> jax.Array:
                # ``base``: the ring slot the previous item primed this item's first
                # key tile into, with its query tile, during its last key tile.
                if bulk_exchange:
                    bulk_sends(item)[0]()
                head, _, real, lo, count = span(item)
                # The first key tile writes the running state, so only the tail
                # tile's padding rows start from the zero fill.
                if tail_rows:
                    tail = q_tile_of(item) == (used - 1) // block_q
                    pl.when(tail)(pad_tail)
                q_copy(item).wait()
                # The keys a tile sees are cut at ``used`` only in one key tile: a
                # real tile's last real one, a padding tile's first.
                cut = (used % block_kv != 0) & (~real)
                cut_real = (used % block_kv != 0) & real

                def prime_after(slot):
                    """Primes the next item into ``1 - slot``, the slot no key tile of
                    this item refills."""
                    if bulk_exchange:

                        def prime_next():
                            await_head(item + 1)
                            prime(item + 1, 1 - slot)

                        pl.when(item + 1 < item_hi)(prime_next)
                    else:
                        pl.when(item + 1 < item_hi)(lambda: prime(item + 1, 1 - slot))

                def kv_body(index, slot, fresh=False):
                    """One key tile: the last primes the next item, and the first
                    (peeled) writes the running state where ``fresh``."""
                    now = tile(lo, index)
                    for one in kv_copies(head, now, slot):
                        one.wait()

                    @pl.when(index + 1 < count)
                    def _next():
                        for one in kv_copies(head, tile(lo, index + 1), 1 - slot):
                            one.start()

                    pl.when(index + 1 == count)(lambda: prime_after(slot))
                    masked = (cut & (index == 0)) | (cut_real & (now == real_tiles - 1))
                    visit(item, slot, now * block_kv, real, masked, fresh)
                    return 1 - slot

                after = kv_body(0, base, True)
                slot = 1 - lax.fori_loop(1, count, kv_body, after)
                finish_item(item)
                return 1 - slot

            def placed_transfer(at):
                """``out_transfer`` of walk position ``at``'s item."""
                return out_transfer(item_of(at), at)

            def finish_item(at, then=None):
                """The output of walk position ``at``'s item: two positions back's
                transfers waited, o / l into its stage slot and out to the query
                tile's owner. ``then``: run after o / l are read, before the
                transfer starts (zero_ahead: the next item's state, in this block)."""
                pl.when(at >= item_lo + 2)(placed_transfer(at - 2)[1])
                if bulk_exchange:
                    pl.when(at >= item_lo + 2)(sends(at - 2)[1])
                if cast_first:  # the same bf16 values, transposed packed
                    stage[at % 2] = (o_run[...] / l_run[0:1]).astype(_BF16).T
                else:
                    stage[at % 2] = (o_run[...] / l_run[0:1]).T.astype(_BF16)
                if sum_bounds:
                    # A copy of the final sum (and max) rows in this block, before
                    # ``then`` (zero_ahead) clears them; the recording branches
                    # off after the transfer starts, so the block stays whole.
                    sums_copy[0:1, :] = l_run[0:1, :]
                    if not fixed:
                        sums_copy[1:2, :] = m_run[0:1, :]
                if then is not None:
                    then()
                placed_transfer(at)[0]()
                if sum_bounds:
                    pl.when(~proven)(functools.partial(record_sums, item_of(at)))

            def record_sums(item):
                """sum_bounds: the largest and the largest reciprocal of item's
                real rows' fixed-max sums (``l``; online ``l * 2^(m - fixed_max)``,
                the sum a fixed max would take) into ``peaks`` [0] and [1], row 0,
                at its head's lane: maxed over the head's items (zeroed each
                layer). A sum that is not finite reads +inf (its reciprocal 0),
                a zero sum's reciprocal +inf."""
                head = head_order[item // q_tiles]
                lanes = lax.broadcasted_iota(jnp.int32, (1, block_q), 1)
                real = lanes + q_tile_of(item) * block_q < used
                sums = sums_copy[0:1, :]
                if not fixed:
                    sums = sums * jnp.exp2(sums_copy[1:2, :] - fixed_max)
                sums = jnp.where(jnp.abs(sums) < jnp.inf, sums, jnp.inf)
                top = jnp.max(jnp.where(real, sums, 0.0), axis=1, keepdims=True)
                inv = jnp.max(jnp.where(real, 1.0 / sums, 0.0), axis=1, keepdims=True)
                lane = lax.broadcasted_iota(jnp.int32, (1, TABLE_LANES), 1)
                hit = lane == computed_head(layer, head)
                for k, value in ((0, top), (1, inv)):
                    peaks[k, 0:1, :] = jnp.maximum(
                        peaks[k, 0:1, :], jnp.where(hit, value, 0.0)
                    )

            def static_item(
                at: jax.Array, base: jax.Array, width: int = 0
            ) -> jax.Array:
                """static_used: every item is real and walks key tiles 0 .. last
                (``used`` static), the last cut iff ``used`` is not a multiple of
                block_kv. One loop per query width, its body one softmax variant
                (a small loop body), then the last tile peeled. ``at``: the walk
                position (of item ``item_of(at)``); ``width``: split_items' query
                width of every item of the loop (0: the item's own)."""
                item = item_of(at)
                if bulk_exchange:
                    sends(at)[0]()
                head = head_order[item // q_tiles]
                if not zero_ahead:  # else the item before zeroed it (zero_state)
                    zero_runs(item)
                q_copy(item, at).wait()
                last = (static_used + block_kv - 1) // block_kv - 1
                cut = static_used % block_kv != 0

                def prime_after(slot):
                    if bulk_exchange:

                        def prime_next():
                            await_at(at + 1)
                            nxt = at + 1
                            prime(item_of(nxt), 1 - slot, nxt)

                        pl.when(at + 1 < item_hi)(prime_next)
                    else:

                        def prime_next():
                            nxt = at + 1
                            prime(item_of(nxt), 1 - slot, nxt)

                        pl.when(at + 1 < item_hi)(prime_next)

                def walk(width):

                    def body(index, slot):
                        for one in kv_copies(head, index, slot):
                            one.wait()
                        for one in kv_copies(head, index + 1, 1 - slot):
                            one.start()
                        update(at, slot, index * block_kv, True, False, width)
                        return 1 - slot

                    slot = lax.fori_loop(0, last, body, base)
                    for one in kv_copies(head, last, slot):
                        one.wait()
                    prime_after(slot)
                    narrow = cut and subs_cut < block_kv // sub_kv
                    update(
                        at,
                        slot,
                        last * block_kv,
                        True,
                        cut,
                        width,
                        subs_cut if narrow else None,
                        clear=full_cut if cut else 0,
                        last_keys=cut_keys if cut and fixed else 0,
                    )

                if width:
                    walk(width)
                elif tail_rows:
                    tail = q_tile_of(item) == (static_used - 1) // block_q
                    pl.when(tail)(lambda: walk(tail_rows))
                    pl.when(~tail)(lambda: walk(block_q))
                else:
                    walk(block_q)
                # zero_ahead: the next item's state, under this one's epilogue.
                finish_item(
                    at,
                    functools.partial(zero_state, item_of(at + 1))
                    if zero_ahead
                    else None,
                )
                return 1 - (base + last) % 2

            with scope("B.items", 2):
                if bulk_exchange:

                    def prime_first():
                        await_at(item_lo)
                        prime(item_of(item_lo), 0, item_lo)

                    pl.when(item_lo < item_hi)(prime_first)
                else:
                    pl.when(item_lo < item_hi)(
                        lambda: prime(item_of(item_lo), 0, item_lo)
                    )
                if zero_ahead and static_used:
                    zero_state(item_of(item_lo))
                if split:
                    # The last full-width item primes the first tail item.
                    base = lax.fori_loop(
                        item_lo,
                        middle,
                        functools.partial(static_item, width=block_q),
                        jnp.int32(0),
                    )
                    lax.fori_loop(
                        middle,
                        item_hi,
                        functools.partial(static_item, width=tail_rows),
                        base,
                    )
                else:
                    lax.fori_loop(
                        item_lo,
                        item_hi,
                        static_item if static_used else item_body,
                        jnp.int32(0),
                    )
            with scope("B.drain"):
                for back in (2, 1):
                    last = item_hi - back
                    pl.when(last >= item_lo)(
                        lambda last=last: placed_transfer(last)[1]()
                    )
                    if bulk_exchange:
                        pl.when(last >= item_lo)(sends(last)[1])
            if not final:
                return
            # Every other device sends each of its heads for my query tiles.
            with scope("B.await"):
                await_all(
                    attn_ref.at[parity, :, pl.ds(0, local_heads * (devices - 1) * dim)],
                    attn_send.at[0],
                    attn_recv.at[parity],
                    attn_ref.at[parity, pl.ds(0, unit), pl.ds(0, dim)],
                    rows // unit * local_heads * (devices - 1),
                )

        pl.run_scoped(
            scoped,
            pltpu.VMEM((2, block_q, dim), _BF16),
            # K and V: [slot, k | v, rows, D].
            pltpu.VMEM((2, 2, block_kv, dim), _BF16),
            # Running max and sum, one row per sublane as custom splash keeps
            # them; row 0 is read.
            pltpu.VMEM((8, block_q), _F32),
            pltpu.VMEM((8, block_q), _F32),
            pltpu.VMEM((dim, block_q), _F32),
            pltpu.VMEM((2, block_q, dim), _BF16),
            # K slots, V slots, staging slots, query slots.
            pltpu.SemaphoreType.DMA((8,)),
            *((pltpu.VMEM((8, block_q), _F32),) if sum_bounds else ()),
        )

    # ------------------------------------------------------------ phase C
    def phase_c(layer: jax.Array, parity: jax.Array) -> None:
        tile = tiling.mlp_rows
        sub = sub_rows(tile)
        # Each streamed tile's epilogue (gated residual, SiLU * up) runs under
        # the next tile's matmul; the output and down projections look their
        # gates up a step ahead (``_stream_ahead``). prefetch_down: the down
        # projection's first weight tile loads during the up projection, into
        # a buffer of the MLP's scope.
        down_lead = prefetch_down
        norm_ref = layer_norms(parity)[0]
        # uniform_adaln: meta's per row tile flags, after the head words and
        # the row_tile_modes.
        uniform_at = 1 + 3 * layers + rows // tiling.qkv_rows

        def outer(index, x, onehot, tables, uniform):
            """Row tile ``index`` (its x and one-hot already in): the output
            projection and gated residual, then the MLP. ``tables``: this layer's
            gate_msa, shift_mlp, scale_mlp, gate_mlp. ``uniform``: uniform_adaln's
            traced flag, every row of the tile reads one AdaLN row; else None."""
            rs_tile = pl.ds(index * tile, tile)
            loads = []
            ptables, mtables = tables.at[pl.ds(0, 1)], tables.at[pl.ds(1, 3)]

            def first_rows(count):
                """uniform_adaln: run_scoped's scratch for ``count`` tables' ``fill``
                (the MLP's shift and scale)."""
                if uniform is None:
                    return []
                return [pltpu.VMEM((count, 1, hidden), _F32)]

            def prepare(firsts, tables, slots):
                """uniform_adaln: ``fill`` ``firsts`` where the tile is uniform."""
                pl.when(uniform)(functools.partial(fill, firsts, onehot, tables, slots))

            def branch(fn, first, name="first"):
                """``fn()``, or under uniform_adaln ``fn(name=first)`` where the tile
                is uniform and ``fn()`` where not."""
                if uniform is None:
                    fn()
                    return
                pl.when(uniform)(functools.partial(fn, **{name: first}))
                pl.when(~uniform)(fn)

            def ahead_parts(lhs, w, acc, ahead, tables, slot, width, x_first):
                """``_stream_ahead``'s compute, gates and finish of a stream into
                ``width`` columns of x a tile, gated by table ``slot``: ``ahead``
                (``[slots, tile, width]`` f32) holds the lookups of the gates, the
                one-hot rows' table entries (exact: a one-hot row picks one entry),
                and ``finish`` is ``x += gate * bf16(acc)``. ``x_first``: ``finish`` loads x
                before the gate and the product (the down projection schedules
                shorter that way, the output projection in the other order)."""

                def cols(column):
                    return pl.ds(pl.multiple_of(column * width, width), width)

                def compute(column, ring, buf):
                    matmul(lhs, w.at[ring], acc.at[buf], tile, sub)

                def look(column, g):
                    cs = cols(column)

                    def one(rs):
                        ahead.at[g][rs, :] = lookup(onehot, tables, slot, rs, cs)

                    _unrolled_chunks(tile, chunk, one)

                def finish(column, buf, g):
                    cs, result, gate = cols(column), acc.at[buf], ahead.at[g]

                    def one(rs):
                        if x_first:
                            value = x[rs, cs].astype(_F32) + gate[
                                rs, :
                            ] * ref.round_bf16(result[rs, :])
                        else:
                            factor, product = gate[rs, :], result[rs, :]
                            value = x[rs, cs].astype(_F32) + factor * ref.round_bf16(
                                product
                            )
                        x[rs, cs] = value.astype(_BF16)

                    _unrolled_chunks(tile, chunk, one)

                return compute, look, finish

            def projection(attn, w, acc, p_sems, gates):
                tables = ptables
                loads.append(copy(attn_ref.at[parity, rs_tile], attn, p_sems.at[0]))
                loads[-1].start()

                def prologue():
                    for one in loads:
                        one.wait()

                def fetch(column, slot):
                    return [
                        copy(wo_ref.at[layer, column], w.at[slot], p_sems.at[1 + slot])
                    ]

                _stream_ahead(
                    hidden // out_cols,
                    fetch,
                    *ahead_parts(
                        attn, w, acc, gates, tables, 0, out_cols, x_first=False
                    ),
                    prologue,
                    gate_slots=1,
                    lead=lambda: scope("C.proj.lead", 3),
                )

            with scope("C.proj", 2):
                pl.run_scoped(
                    projection,
                    pltpu.VMEM((tile, config.width), _BF16),
                    pltpu.VMEM((2, config.width, out_cols), wo_ref.dtype),
                    pltpu.VMEM((2, tile, out_cols), _F32),
                    pltpu.SemaphoreType.DMA((3,)),
                    # The gates, a step ahead.
                    pltpu.VMEM((1, tile, out_cols), _F32),
                )

            def mlp(act, *lead_scratch):
                tables = mtables
                lead_copy = (
                    copy(w2_ref.at[layer, 0], lead_scratch[0], lead_scratch[1].at[0])
                    if down_lead
                    else None
                )

                # Up: every FFN channel's activation for the row tile, into VMEM.
                def up(h, w13, gate_up, u_sems, *scratch):
                    # uniform_adaln: fill's shift and scale; slab_modulate (uniform
                    # tiles): then the rms factors' scratch.
                    slabs = slab_modulate and bool(first_rows(2))
                    factors = scratch[-1] if slabs else None
                    if slabs:
                        scratch = scratch[:-1]
                    firsts = None if not first_rows(2) else scratch[-1]

                    def prologue():
                        if lead_copy is not None:  # behind this stream's first tile
                            lead_copy.start()
                        if firsts is not None:  # under the first weight tile's load
                            prepare(firsts, tables, (0, 1))
                        rows = None if firsts is None else (firsts.at[0], firsts.at[1])
                        if slabs:
                            # A uniform tile: modulate_slabs' values (each row's rms
                            # factor, then 256-lane slabs), no whole rows held.
                            pl.when(uniform)(
                                functools.partial(
                                    modulate_slabs,
                                    x,
                                    h,
                                    norm_ref.at[1],
                                    rows,
                                    factors,
                                    tile,
                                )
                            )
                            pl.when(~uniform)(
                                functools.partial(
                                    modulate, x, h, onehot, tables, norm_ref.at[1], tile
                                )
                            )
                        else:
                            branch(
                                functools.partial(
                                    modulate, x, h, onehot, tables, norm_ref.at[1], tile
                                ),
                                rows,
                                "rows",
                            )

                    def activation(column, result, slot, chunks):
                        cols = pl.ds(
                            pl.multiple_of(column * ffn_cols, ffn_cols), ffn_cols
                        )

                        def activate(rs):
                            product = result[rs, :]
                            gate = ref.round_bf16(
                                jax.nn.silu(ref.round_bf16(product[:, :ffn_cols]))
                            )
                            value = gate * ref.round_bf16(product[:, ffn_cols:])
                            act[rs, cols] = value.astype(_BF16)

                        chunks(tile, chunk, activate)

                    # One result buffer, read by finish(i - 1) before compute(i)
                    # overwrites it, holding the bf16 rounding the activation
                    # applies first to both halves.

                    def compute(column, slot):
                        matmul(h, w13.at[slot], gate_up.at[0], tile, sub)

                    def finish(column):
                        activation(column, gate_up.at[0], None, _unrolled_chunks)

                    def fetch(column, slot):
                        return [
                            copy(
                                w13_ref.at[layer, column], w13.at[slot], u_sems.at[slot]
                            )
                        ]

                    _stream_pipelined(
                        ffn // ffn_cols,
                        fetch,
                        compute,
                        finish,
                        prologue,
                        finish_first=True,
                        lead=lambda: scope("C.up.lead", 3),
                    )

                with scope("C.up", 2):
                    pl.run_scoped(
                        up,
                        pltpu.VMEM((tile, hidden), _BF16),
                        pltpu.VMEM((2, hidden, 2 * ffn_cols), w13_ref.dtype),
                        pltpu.VMEM((1, tile, 2 * ffn_cols), _BF16),
                        pltpu.SemaphoreType.DMA((2,)),
                        *first_rows(2),
                        *(
                            [pltpu.VMEM((tile, 1), _F32)]
                            if slab_modulate and first_rows(2)
                            else []
                        ),
                    )

                # Down: each column tile contracts every FFN channel in one MXU
                # accumulation, with the second gated residual in its epilogue.
                def down(w2, acc, d_sems, gates):

                    def fetch(column, slot):
                        return [
                            copy(w2_ref.at[layer, column], w2.at[slot], d_sems.at[slot])
                        ]

                    _stream_ahead(
                        hidden // down_cols,
                        fetch,
                        *ahead_parts(
                            act, w2, acc, gates, tables, 2, down_cols, x_first=True
                        ),
                        None,
                        gate_slots=2,
                        lead=lambda: scope("C.down.lead", 3),
                        first=None
                        if lead_copy is None
                        else (
                            [lead_copy],
                            lambda: matmul(act, lead_scratch[0], acc.at[0], tile, sub),
                        ),
                    )

                with scope("C.down", 2):
                    pl.run_scoped(
                        down,
                        pltpu.VMEM((2, ffn, down_cols), _BF16),
                        pltpu.VMEM((2, tile, down_cols), _F32),
                        pltpu.SemaphoreType.DMA((2,)),
                        # The gates, a step ahead.
                        pltpu.VMEM((2, tile, down_cols), _F32),
                    )

            with scope("C.mlp", 2):
                pl.run_scoped(
                    mlp,
                    pltpu.VMEM((tile, ffn), _BF16),
                    *(
                        [
                            pltpu.VMEM((ffn, down_cols), _BF16),
                            pltpu.SemaphoreType.DMA((1,)),
                        ]
                        if down_lead
                        else []
                    ),
                )

        tiles = rows // tile

        def phase(tables, xs, onehots, sems):
            """The layer's four phase C tables once; each row tile's x and one-hot
            double buffered (the next tile's load and this tile's store overlap
            compute)."""
            picks = (_GATE_MSA, _SHIFT_MLP, _SCALE_MLP, _GATE_MLP)
            for one in load_tables(layer, picks, tables, sems.at[pl.ds(6, 4)]):
                one.wait()

            def loads(index):
                rs, slot = pl.ds(index * tile, tile), index % 2
                return (
                    copy(x_ref.at[rs], xs.at[slot], sems.at[slot]),
                    copy(onehot_ref.at[rs], onehots.at[slot], sems.at[2 + slot]),
                )

            def store(index):
                rs, slot = pl.ds(index * tile, tile), index % 2
                return copy(xs.at[slot], x_ref.at[rs], sems.at[4 + slot])

            for one in loads(0):
                one.start()

            def row_tile(index: jax.Array, carry: None) -> None:
                slot = index % 2
                for one in loads(index):
                    one.wait()

                @pl.when(index >= 1)
                def _stored():  # the buffer the next load refills
                    store(index - 1).wait()

                @pl.when(index + 1 < tiles)
                def _prefetch():
                    for one in loads(index + 1):
                        one.start()

                uniform = meta_ref[uniform_at + index] != 0 if uniform_adaln else None
                outer(index, xs.at[slot], onehots.at[slot], tables, uniform)
                store(index).start()
                return carry

            lax.fori_loop(0, tiles, row_tile, None)
            store(tiles - 1).wait()

        pl.run_scoped(
            phase,
            pltpu.VMEM((4, TABLE_LANES, hidden), _BF16),
            pltpu.VMEM((2, tile, hidden), _BF16),
            pltpu.VMEM((2, tile, TABLE_LANES), _BF16),
            # x loads, one-hot loads, stores, four tables.
            pltpu.SemaphoreType.DMA((10,)),
        )

    # ------------------------------------------------------------- layers
    def norm_copies(layer, slot):
        """``layer``'s norm1, norm2, q_norm and k_norm into ``slot`` of their
        double buffers."""
        sources = (norm1_ref, norm2_ref, qn_ref, kn_ref)
        targets = (
            norms.at[slot, 0],
            norms.at[slot, 1],
            qk_norms.at[slot, 0],
            qk_norms.at[slot, 1],
        )
        return [
            copy(src.at[layer], dst, norm_sems.at[i])
            for i, (src, dst) in enumerate(zip(sources, targets))
        ]

    def peaks_write(layer):
        """``layer``'s peaks, VMEM -> ``peaks_ref``."""
        return copy(peaks, peaks_ref.at[layer], peaks_sem.at[0])

    for one in norm_copies(0, 0):  # layer 0's norms start before the barrier
        one.start()
    if devices > 1:
        # Every peer must be inside the kernel before anyone writes into it.
        barrier = pltpu.get_barrier_semaphore()
        for offset in range(1, devices):
            pl.semaphore_signal(barrier, 1, device_id=device((me + offset) % devices))
        pl.semaphore_wait(barrier, devices - 1)

    def layer_body(layer: jax.Array, carry: None) -> None:
        parity = layer % 2
        with scope("L.norms"):
            # This layer's norms loaded during the last layer (layer 0's at the
            # start); the next layer's load during this one, into the slot the
            # last layer read.
            for one in norm_copies(layer, parity):
                one.wait()

            @pl.when(layer + 1 < layers)
            def _prefetch():
                for one in norm_copies(layer + 1, 1 - parity):
                    one.start()

        with scope("L.peaks"):
            # The last layer's peaks went out during its phases B and C; the
            # buffer is zeroed only once they are out.
            pl.when(layer > 0)(peaks_write(layer - 1).wait)
        peaks[...] = jnp.zeros(peaks.shape, _F32)
        proven_layer = (meta_ref[1 + layer] & 32) != 0
        if bulk_exchange:
            # Every device's head order this layer (as head_order below builds
            # mine): phase A sends each owner only its first head, phase B the
            # rest in this order.
            at = 1 + layers + 2 * layer
            for owner in range(devices):
                count = jnp.int32(0)
                for fixed_pass in (True, False):
                    for head in range(local_heads):
                        index = owner * local_heads + head
                        bit = (meta_ref[at + index // 32] >> (index % 32)) & 1
                        orders[owner, count] = head
                        count = count + (bit if fixed_pass else 1 - bit)
        with scope("A"):
            if sum_bounds:  # the softmax sums certify, not the q/k peaks
                phase_a(layer, parity, False)
            else:
                pl.when(proven_layer)(functools.partial(phase_a, layer, parity, False))
                pl.when(~proven_layer)(functools.partial(phase_a, layer, parity, True))

        def write_peaks():
            with scope("L.peaks"):
                # Waited before the buffer's next use.
                peaks_write(layer).start()

        if not sum_bounds:
            write_peaks()
        # This layer's local heads, those that run at a fixed max (meta's head
        # bits) first: two item loops, one per softmax.
        q_tiles = rows * devices // tiling.block_q
        heads_at = 1 + layers + 2 * layer
        count = jnp.int32(0)
        for head in range(local_heads):
            index = me * local_heads + head
            bit = (meta_ref[heads_at + index // 32] >> (index % 32)) & 1
            head_order[count] = head
            count = count + bit
        split = count
        for head in range(local_heads):
            index = me * local_heads + head
            bit = (meta_ref[heads_at + index // 32] >> (index % 32)) & 1
            head_order[count] = head
            count = count + (1 - bit)
        items = local_heads * q_tiles
        attend = functools.partial(phase_b, layer, parity)
        with scope("B"):
            attend(True, jnp.int32(0), split * q_tiles, False)
            attend(False, split * q_tiles, jnp.int32(items), True)
        if sum_bounds:  # phase B wrote them
            write_peaks()
        with scope("C"):
            phase_c(layer, parity)
        return carry

    with_step_tables(lambda: lax.fori_loop(0, layers, layer_body, None))
    with scope("L.peaks"):  # the last layer's peaks are out before the end
        peaks_write(layers - 1).wait()


def _kernel_bulk(*refs: Any, inner: Any, outputs: int, **kwargs: Any) -> None:
    """``inner`` with bulk_exchange's refs passed by name: kv_mine after the
    four outputs (which start at ref ``outputs``), and the two scratch refs
    after the others."""
    kv_mine = refs[outputs + 4]
    inner(
        *refs[: outputs + 4],
        *refs[outputs + 5 : -2],
        bulk_exchange=True,
        kv_mine_ref=kv_mine,
        bulk_send=refs[-2],
        orders=refs[-1],
        **kwargs,
    )


def fixed_max_layers(weights: Mapping[str, jax.Array], config: ref.Config) -> jax.Array:
    """``[L]`` bool: the layers whose softmax runs at a fixed max.

    RMSNorm leaves q and k at norm ``sqrt(D)`` before their norm weights, and
    RoPE rotates dim ``d`` with ``d + R/2``. Cauchy-Schwarz over each rotated
    pair, then over the pairs and the unrotated dims, gives ``|q . k| <= D
    max c``: ``c`` is a pair's largest q weight times its largest k weight, or
    an unrotated dim's q weight times its k weight. So the norm weights bound
    every score, padding rows' too.
    """
    half = config.rope_dims // 2
    q, k = (jnp.abs(weights[name].astype(_F32))[:, 0] for name in ("q_norm", "k_norm"))

    def pairs(w):
        return jnp.maximum(w[:, :half], w[:, half : 2 * half])

    rotated = (pairs(q) * pairs(k)).max(axis=1)
    plain = (q[:, 2 * half :] * k[:, 2 * half :]).max(axis=1, initial=0.0)
    peak = jnp.maximum(rotated, plain)
    bound = peak * math.sqrt(config.head_dim) * math.log2(math.e)
    return bound * _BOUND_SLACK <= _FIXED_BOUND


def row_tile_modes(onehot: jax.Array, tile: int) -> jax.Array:
    """``[T_l / tile]`` int32: how phase A modulates each row tile.

    0: with a one-hot lookup per chunk of rows, as without it. 1: every
    row's one-hot is the tile's first row's, bit for bit, so one lookup of
    its first rows gives the shift and scale of all (a row of a product
    depends on that row alone). 2: as 1, and the row is also the one of the
    last tile before it in mode 1 or 2, so it reuses that tile's lookup.

    Args:
      onehot: ``[T_l, 128]`` bf16, this device's rows of
        ``ref.table_onehot``.
      tile: Rows a phase A row tile holds (``tl.Tiling.fit``'s
        ``qkv_rows``).
    """
    bits = lax.bitcast_convert_type(jnp.asarray(onehot, _BF16), jnp.uint16)
    bits = bits.reshape(-1, tile, bits.shape[-1])
    first = bits[:, 0]
    shared = (bits == first[:, None]).all(axis=(1, 2))
    index = jnp.arange(first.shape[0], dtype=jnp.int32)
    # The last tile before each that shares one row, or -1.
    last = lax.cummax(jnp.where(shared, index, -1))
    last = jnp.concatenate((jnp.full((1,), -1, jnp.int32), last[:-1]))
    again = (last >= 0) & (first[jnp.maximum(last, 0)] == first).all(axis=1)
    return jnp.where(shared, jnp.where(again, 2, 1), 0).astype(jnp.int32)


def uniform_row_tiles(onehot: jax.Array, tile: int) -> jax.Array:
    """``[T_l / tile]`` int32: 1 for each ``tile``-row tile of this device's
    ``onehot`` whose rows are all the same, bit for bit (they read one AdaLN
    row), else 0: ``stack``'s ``uniform_adaln`` tiles."""
    width = onehot.dtype.itemsize * 8
    bits = lax.bitcast_convert_type(onehot, jnp.dtype(f"uint{width}"))
    bits = bits.reshape(-1, tile, onehot.shape[-1])
    return (bits == bits[:, :1]).all(axis=(1, 2)).astype(jnp.int32)


def stack(
    x: jax.Array,
    weights: Mapping[str, jax.Array],
    tables: jax.Array,
    onehot: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    used: int | jax.Array,
    *,
    config: ref.Config,
    tiling: tl.Tiling,
    axis: str | None = None,
    interpret: bool | Any = False,
    fixed: jax.Array | None = None,
    return_peaks: bool = False,
    narrow_tail: bool = False,
    trace: int = 0,
    fixed_heads: jax.Array | None = None,
    static_items: bool = True,
    prefetch_down: bool = True,
    fixed_max: float = _FIXED_M,
    static_structure: int | None = None,
    cut_kv_compute: int = 0,
    split_items: bool = False,
    epilogue_chunk: int = 0,
    epilogue_chunk_peaks: int = 0,
    modulate_pieces: int = 0,
    uniform_adaln: bool = False,
    zero_ahead: bool = False,
    mask_scores: bool = False,
    sum_bounds: bool = False,
    cast_first: bool = False,
    step_tables: bool = False,
    slab_modulate: bool = False,
    kv_group: int | tuple[int, int] = 1,
) -> Any:
    """Runs every block of ``weights`` on this device's rows.

    Args:
      x: ``[T_l, H]`` bf16, this device's contiguous slice of the packed
        rows, a multiple of 128 (see ``padded_rows``).
      weights: ``ref.pack_weights``'s output, on every device.
      tables: ``[L, 6, R, H]`` from ``ref.adaln_tables``.
      onehot: ``[T_l, 128]``, ``cos`` and ``sin`` ``[T_l, D]``: this
        device's rows of ``ref.table_onehot`` and ``ref.rope_tables``.
      cos: See ``onehot``.
      sin: See ``onehot``.
      used: Rows of the whole sequence before its padding.
      config: The stack's hyper-parameters.
      tiling: Bounds on how the kernel cuts the work.
      axis: The sequence-parallel axis of the enclosing ``shard_map``, or
        None on one device; other mesh axes hold independent replicas.
      interpret: Passed to ``pallas_call`` (CPU tests).
      fixed: ``[L]`` bool, layers to run with the fixed-max softmax besides
        those the norm weights prove (``fixed_max_layers``). The caller
        vouches for their scores: ``layer_bounds`` of these peaks must stay
        at or under the fixed-max bound, or the result is not the reference's.
      return_peaks: Also return ``[L, 2, 8, 128]`` f32 peaks: per layer, row 0
        of ``[0]`` / ``[1]`` holds each head's (lane's) largest squared q / k
        norm over this device's rows (q in log2 score units).
      fixed_heads: ``[L, heads]`` bool: heads to run at a fixed max besides
        whole layers (``fixed`` and the proven ones); the caller vouches for
        them as for ``fixed`` (``safe_heads``).
      trace: xprof trace regions in the kernel: 1 per layer phase and
        their tails, 2 also per row tile and stream (profiling only).
      narrow_tail: With a static ``used``, run the last real query tile's
        attention on its real rows only (rounded up to 128): the same math,
        rounded differently (one ulp here and there).
      static_items: With an integer ``used`` that leaves no query tile of
        padding rows, every item walks
        the same real key tiles: phase B runs one loop per query width whose
        body is a single softmax variant (small loop bodies stay resident in
        the instruction memory), the last key tile peeled with the one cut
        variant. Each item starts from the zero fill. The same real rows, bit
        for bit.
      prefetch_down: Phase C's down projection gets its first weight tile from
        a ``[ffn, down_cols]`` bf16 buffer of the MLP's scope, loaded during
        the up projection, so its first matmul waits for no load.
      fixed_max: The fixed-max softmax's offset, ``exp2(s - fixed_max)`` (a
        constant of the compiled kernel; 17 by default). The norm bound vouches
        layers and heads for 17 only (scores within +-109 log2 units keep a
        row's top weight in [2^-126, 2^92]); another offset is the caller's to
        vouch, e.g. measured row maxima and an all-finite check of the output.
        A larger one lets heads with larger row maxima run fixed-max at no
        cost per score.
      static_structure: With a traced ``used``, a static row count whose
        ``loop_structure`` the traced one shares (the caller checks that it
        does): ``static_items`` and ``narrow_tail`` shape the loops from it as from a static ``used``, while the masks read the
        traced one. The result is the static ``used``'s, bit for bit; one
        compiled program serves every ``used`` of the structure. None: a
        traced ``used`` takes the generic walk, without those three.
      cut_kv_compute: With ``static_items``' dense walk and ``narrow_tail``,
        the fixed-max softmax halves the last sub-block of its cut key tile
        that holds real keys while those keys fit in the first half and the
        half keeps at least this many keys (a multiple of 16; 0: off): at
        2,048-key sub-blocks and at most 1,024 real keys there, one 1,024-key
        sub-block instead of 2,048 keys. The dropped keys are padding, their
        probabilities exactly 0: the row sum and the P V product lose only
        trailing zero terms, the same bits where those reductions keep the
        first half's order (interpret mode checks it). ``loop_structure``
        takes the same value.
      split_items: With ``static_items``' dense walk and a narrowed tail
        (``narrow_tail``), phase B runs every head's full-width items in one
        loop and the tail items after them in a second, each loop body one
        query width (two smaller hot loops). Each item keeps its own running
        state, key walk and output rows, so only their order changes: the
        same bits. bulk_exchange's sends for head k + 1 go out during head
        k's full-width items.
      epilogue_chunk: Phase A's pipelined QKV epilogue (q/k norms, RoPE,
        staging) in chunks of this many rows and a shorter last one instead
        of ``tiling.epilogue_rows`` (a divisor of the row tile: 32 at 608-row
        tiles). 0: off.
      epilogue_chunk_peaks: The same for the layers whose epilogue also takes
        the q/k peaks (``with_peaks``); 0: as the others. The same bits.
      modulate_pieces: The next row tile's modulate
        in this many pieces of whole chunks (``tiling.chunk_rows``), one per
        head group step from the middle of the tile on, instead of 2 (1 at an
        odd number of chunks): at one chunk a piece, each piece fits under
        its step's QKV matmul instead of running after it with the MXUs idle.
        Each chunk computes the same values (``modulate_slabs``), so the same
        bits. 0: off.
      uniform_adaln: In phase C, a row tile whose one-hot rows are all the
        same (``uniform_row_tiles``) modulates the MLP's input with norm2's
        shift and scale looked up once (``fill``'s ``[1, H]`` f32 rows)
        instead of a one-hot matmul per chunk; other tiles keep the lookups.
        ``fill`` is the same lookup on the tile's first rows and every row of
        the tile has the same one-hot row: the same bits. (The residual gates
        of the output and down projections keep their lookups: their streams
        schedule no shorter without them.)
      zero_ahead: With ``static_items``' dense walk, each item's running
        state (sum and output zero, the online max at the mask value, the tail
        tile's padding sums one) is written at the end of the item before it,
        in its epilogue's block, instead of at the start of the item: the
        stores run under the epilogue's transposes rather than ahead of the
        item's first key tile. The same state, so the same bits.
      mask_scores: A masked sub-block (keys past ``used``) subtracts a per-key
        column from its scores, the fixed max (the online softmax: 0) at a
        visible key and +inf at a masked one, in place of selecting its
        probabilities (the online softmax's masked max too): a visible score
        gets the same difference, a masked one -inf and ``exp2(-inf)`` the
        select's +0. The same probabilities and max, so the same bits, with no
        per-element select between the EUP's results and the sums.
      sum_bounds: The peaks certify by the softmax's own row sums instead of
        the q/k norms: phase A records no peaks, and each item of a layer the
        norm weights leave unproven records, at its head's lane, the largest
        fixed-max row sum of its real rows (``l``; online ``l * 2^(m -
        fixed_max)``, the sum a fixed max would take) in row 0 of peaks[0] and
        the largest reciprocal of one in peaks[1], maxed over the head's items
        (``sum_safe_heads``). A fixed-max head whose sums stay in
        [2^_SUM_LO, 2^_SUM_HI] ran the softmax up to rounding (see
        ``_SUM_LO``), whatever its q/k norms. Only the peaks change, not the
        rows.
      cast_first: Phase B's item epilogue rounds o / l to bf16 before the
        transpose into the output tile instead of after it (half the bytes
        through the transposes): the same values, the same bits.
      step_tables: Phase A's per-step scalars (row tile, head group, staging
        slot) and each head's owner and local index come from SMEM tables
        filled once a call instead of integer divisions every step, whose
        chains lengthen the scalar blocks around each step's matmul (the
        MXUs idle there). The same steps, the same bits.
      slab_modulate: With uniform_adaln, phase C's uniform tiles modulate the
        MLP input as phase A's modulate does (``modulate_slabs``: each
        row's rms factor, then 256-lane slabs) instead of whole rows a chunk,
        whose spills lengthen the pass the MXUs idle through between the
        output and the up projection. The same values, the same bits.
      kv_group: With ``static_items``' dense walk, phase B moves and walks the
        keys this many of ``tiling``'s key tiles at a time: one DMA and one
        step of the key loop per group, its sub-blocks those of one key tile
        (``block_kv_compute``, ``block_kv_compute_online``), so each row's
        running sum and output add the same sub-blocks in the same order and
        every cut parameter (``cut_kv_compute``, ``narrow_tail``) sees the
        same last sub-block: the same bits, with
        1 / kv_group of the key steps and their DMAs. (A fixed-max key tile
        after a group's first adds its P V to the running output through an
        identity select on a traced flag: without it the compiler sums two
        tiles' P V in the MXU before adding them to the output, another
        rounding.) The q, k, v buffer grows to a whole number of groups (its
        extra rows are only ever copied, never read); a group's K and V take
        kv_group times the VMEM. The loop structure is the plain one's
        (``loop_structure``): a group's cut follows from the key tiles'. 1:
        off, the plain program; other walks ignore it. A pair: the fixed-max
        and the online softmax's groups.

    Returns:
      ``[T_l, H]`` bf16, the rows after the last block. Rows from ``used``
      on are padding: finite, but not the reference's. Then the peaks if
      ``return_peaks``.

    Raises:
      ValueError: If the shapes do not tile (see ``tl.Tiling.fit``) or an
        option is out of range.
    """
    if any(
        value < 0 or value % 16
        for value in (cut_kv_compute, epilogue_chunk, epilogue_chunk_peaks)
    ):
        raise ValueError(
            "cut_kv_compute, epilogue_chunk and epilogue_chunk_peaks take"
            f" multiples of 16 rows; got {cut_kv_compute}, {epilogue_chunk} and"
            f" {epilogue_chunk_peaks}"
        )
    groups = (kv_group,) * 2 if isinstance(kv_group, numbers.Integral) else kv_group
    if len(groups) != 2 or any(
        not isinstance(group, numbers.Integral) or group < 1 for group in groups
    ):
        raise ValueError(
            f"kv_group takes a whole number >= 1 or a pair of them; got {kv_group}"
        )
    devices = lax.axis_size(axis) if axis is not None else 1
    rows = x.shape[0]
    layers = weights["wqkv"].shape[0]
    if tables.shape[0] != layers or (
        isinstance(used, numbers.Integral) and not 0 < used <= rows * devices
    ):
        raise ValueError(
            f"MiniMax H3 megakernel takes tables for its {layers} layers and"
            f" 0 < used <= {rows * devices} rows; got {tables.shape[0]} and"
            f" {used}"
        )
    tiling = tiling.fit(config, rows, devices)
    group = weights["wqkv"].shape[-1] // (3 * config.head_dim)
    chunks = tiling.qkv_rows // tiling.chunk_rows
    if modulate_pieces < 0 or (
        modulate_pieces
        and (modulate_pieces > config.heads // group or chunks % modulate_pieces)
    ):
        raise ValueError(
            f"modulate_pieces takes a divisor of the row tile's {chunks} chunks"
            f" up to its {config.heads // group} head groups; got"
            f" {modulate_pieces}"
        )
    if tables.shape[2] > TABLE_LANES:
        raise ValueError(f"{tables.shape[2]} AdaLN rows exceed {TABLE_LANES}")
    dim = config.head_dim
    local_heads = config.heads // devices
    # The bulk exchange (see the module docstring); one device exchanges
    # nothing, and one head a device sends it all first.
    bulk = devices > 1 and local_heads > 1
    attn_shape = (2, rows, config.width)
    norm_proven = fixed_max_layers(weights, config)
    proven = norm_proven
    if fixed is not None:
        proven = proven | fixed.astype(bool)
    # Bit 5: the norm weights bound the layer (no peaks).
    flags = proven.astype(jnp.int32) | 32 * norm_proven.astype(jnp.int32)
    if config.heads > 64:
        raise ValueError(f"{config.heads} heads exceed two 32-bit flag words")
    head_fixed = jnp.broadcast_to(proven[:, None], (layers, config.heads))
    if fixed_heads is not None:
        head_fixed = head_fixed | fixed_heads.astype(bool)
    head_bits = jnp.pad(head_fixed.astype(jnp.uint32), ((0, 0), (0, 64 - config.heads)))
    head_bits = head_bits.reshape(layers, 2, 32) << jnp.arange(32, dtype=jnp.uint32)
    head_words = lax.bitcast_convert_type(
        head_bits.sum(axis=2, dtype=jnp.uint32), jnp.int32
    )
    meta = jnp.concatenate(
        (
            jnp.asarray(used, jnp.int32).reshape(1),
            flags,
            head_words.reshape(-1),
            row_tile_modes(onehot, tiling.qkv_rows),
            *((uniform_row_tiles(onehot, tiling.mlp_rows),) if uniform_adaln else ()),
        )
    )
    # The loops' shape comes from a static ``used`` (or ``static_structure``
    # standing in for a traced one); the masks read the meta word.
    shape_used = used
    if static_structure is not None and not isinstance(used, numbers.Integral):
        if not 0 < static_structure <= rows * devices:
            raise ValueError(
                f"static_structure takes 0 < rows <= {rows * devices}; got"
                f" {static_structure}"
            )
        shape_used = int(static_structure)
    # A static ``used`` whose last real query tile holds few real rows: that
    # tile's attention runs on its first ``tail_rows`` (a multiple of 128).
    tail_rows, cut_subs, cut_full = 0, (0, 0), (0, 0)
    # static_items: a static ``used`` that leaves no query tile of padding
    # rows, the dense walk: every item walks the same real key tiles.
    static_used = 0
    if (
        static_items
        and isinstance(shape_used, numbers.Integral)
        and rows * devices - shape_used < tiling.block_q
    ):
        static_used = int(shape_used)
    if (rows * devices) % tiling.block_kv and not static_used:
        raise ValueError(
            f"a key tile of {tiling.block_kv} rows that does not divide"
            f" {rows * devices} rows takes the dense walk (static_items, a"
            " static live length that leaves no query tile of padding)"
        )
    # The dense walk's key tiles (fixed-max, online): kv_group of the tiling's.
    kv_group = tuple(int(group) for group in groups) if static_used else (1, 1)
    walks = tuple(tiling.block_kv * group for group in kv_group)
    # The q, k, v buffer holds whole key groups: the last group's DMA may
    # reach past the rows, into rows nothing reads.
    kv_shape = (
        2,
        local_heads,
        3,
        max(-(-rows * devices // walk) * walk for walk in walks),
        dim,
    )
    # The exchange buffers start uninitialized: the kernel writes every row
    # before it reads it.
    kv = jax.ShapeDtypeStruct(kv_shape, _BF16)
    attn = jax.ShapeDtypeStruct(attn_shape, _BF16)
    if narrow_tail and isinstance(shape_used, numbers.Integral):
        in_last = shape_used - (shape_used - 1) // tiling.block_q * tiling.block_q
        if -(-in_last // 128) * 128 < tiling.block_q:
            tail_rows = -(-in_last // 128) * 128
        # The last real key tile's real rows, in sub-blocks of each softmax.
        in_cuts = tuple(shape_used - (shape_used - 1) // walk * walk for walk in walks)
        computes = (tiling.block_kv_compute, tiling.block_kv_compute_online)
        cut_subs = tuple(
            -(-in_cut // tl.divisor(tiling.block_kv, compute, 16))
            for in_cut, compute in zip(in_cuts, computes)
        )
        # Of those, the ones that hold real keys only.
        cut_full = tuple(
            in_cut // tl.divisor(tiling.block_kv, compute, 16)
            for in_cut, compute in zip(in_cuts, computes)
        )
        cut_keys = _cut_keys(
            in_cuts[0],
            tl.divisor(tiling.block_kv, tiling.block_kv_compute, 16),
            cut_kv_compute,
        )
    else:
        cut_keys = 0
    operands = (x, onehot, cos, sin, tables, *(weights[name] for name in _WEIGHT_ORDER))
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    kernel = functools.partial(
        _kernel,
        config=config,
        tiling=tiling,
        axis=axis,
        devices=devices,
        tail_rows=tail_rows,
        trace=trace,
        cut_subs=cut_subs,
        cut_full=cut_full,
        cut_keys=cut_keys,
        split_items=split_items,
        epilogue_chunk=epilogue_chunk,
        epilogue_chunk_peaks=epilogue_chunk_peaks,
        modulate_pieces=modulate_pieces,
        uniform_adaln=uniform_adaln,
        zero_ahead=zero_ahead,
        sum_bounds=sum_bounds,
        cast_first=cast_first,
        step_tables=step_tables,
        slab_modulate=slab_modulate,
        mask_scores=mask_scores,
        kv_group=kv_group,
        static_used=static_used,
        prefetch_down=prefetch_down,
        fixed_max=float(fixed_max),
    )
    # bulk_exchange: kv_mine, an output after the four; kv receive per (parity,
    # local head); phase B's send semaphore and every device's head order
    # after the other scratch.
    bulk_out, bulk_scratch = (), ()
    # The norm weights, double buffered by layer parity.
    norm_slots = (2,)
    kv_recv = pltpu.SemaphoreType.DMA((2,))
    if bulk:
        kernel = functools.partial(
            _kernel_bulk, inner=kernel, outputs=1 + len(operands)
        )
        bulk_out = (jax.ShapeDtypeStruct((devices, local_heads, 3, rows, dim), _BF16),)
        kv_recv = pltpu.SemaphoreType.DMA((2, local_heads))
        bulk_scratch = (
            pltpu.SemaphoreType.DMA((1,)),
            # One spare slot a row, as head_order.
            pltpu.SMEM((devices, local_heads + 1), jnp.int32),
        )
    out = pl.pallas_call(
        kernel,
        out_shape=(
            *(jax.ShapeDtypeStruct(a.shape, _BF16) for a in (x, kv, attn)),
            jax.ShapeDtypeStruct((layers, 2, 8, TABLE_LANES), _F32),
            *bulk_out,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * len(operands),
            out_specs=[hbm] * (4 + len(bulk_out)),
            scratch_shapes=[
                pltpu.VMEM((*norm_slots, 2, 1, config.hidden), _BF16),
                pltpu.VMEM((*norm_slots, 2, 1, dim), _BF16),
                pltpu.SemaphoreType.DMA((4,)),
                # kv send per (slot, head), kv receive, attn send, attn
                # receive.
                pltpu.SemaphoreType.DMA((tiling.stage_slots * group,)),
                kv_recv,
                *[pltpu.SemaphoreType.DMA((2,))] * 2,
                pltpu.VMEM((2, 8, TABLE_LANES), _F32),
                pltpu.SemaphoreType.DMA((1,)),
                # One spare slot: the compaction writes before it counts.
                pltpu.SMEM((local_heads + 1,), jnp.int32),
                *bulk_scratch,
            ],
        ),
        # Operand 0 is the scalar prefetch; x (1) is updated in place.
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(
            collective_id=_COLLECTIVE_ID if devices > 1 else None,
            vmem_limit_bytes=tiling.vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="minimax_h3_stack",
    )(meta, *operands)
    result = (out[0],) + ((out[3],) if return_peaks else ())
    return result if len(result) > 1 else result[0]


def _cut_keys(in_cut: int, sub: int, minimum: int) -> int:
    """``stack``'s cut_kv_compute: the keys the last sub-block that holds real
    keys of a cut key tile with ``in_cut`` real keys runs, at ``sub``-key
    sub-blocks: the sub-block halved while its real keys fit in the first
    half and the half keeps ``minimum`` keys; 0 if it is not halved (or
    ``minimum`` is 0)."""
    if not minimum:
        return 0
    real = in_cut - (-(-in_cut // sub) - 1) * sub
    keys = sub
    while keys % 32 == 0 and keys // 2 >= max(real, minimum):
        keys //= 2
    return keys if keys < sub else 0


def loop_structure(
    used: int,
    rows: int,
    devices: int,
    config: ref.Config,
    tiling: tl.Tiling,
    cut_kv_compute: int = 0,
) -> tuple[int, ...]:
    """What ``stack``'s loops take from a static ``used`` (``static_items``
    and ``narrow_tail`` on): the last real query tile and its
    rows rounded up to 128, the last real key tile, whether ``used`` cuts it
    and its real and all-real sub-blocks of each softmax, and whether a query
    tile holds only padding; with ``stack``'s ``cut_kv_compute`` also the
    keys its fixed-max cut tile's last real sub-block runs. Two row counts
    with the same structure run the same loops; ``static_structure`` lets a
    traced ``used`` take them.
    """
    fitted = tiling.fit(config, rows, devices)
    block_q, block_kv = fitted.block_q, fitted.block_kv
    in_last = used - (used - 1) // block_q * block_q
    in_cut = used - (used - 1) // block_kv * block_kv
    subs = [
        tl.divisor(block_kv, compute, 16)
        for compute in (fitted.block_kv_compute, fitted.block_kv_compute_online)
    ]
    return (
        (used - 1) // block_q,
        -(-in_last // 128) * 128,
        (used - 1) // block_kv,
        int(used % block_kv != 0),
        *(-(-in_cut // sub) for sub in subs),
        *(in_cut // sub for sub in subs),
        int(rows * devices - used < block_q),
        *((_cut_keys(in_cut, subs[0], cut_kv_compute),) if cut_kv_compute else ()),
    )


def safe_heads(peaks: jax.Array, margin: float = 1.0) -> jax.Array:
    """``[L, heads]`` bool: ``safe_layers`` per head (lane)."""
    return layer_bounds(peaks) * margin * _BOUND_SLACK <= _FIXED_BOUND


def sum_safe_heads(peaks: jax.Array, margin: float = 0.0) -> jax.Array:
    """``[L, heads]`` bool from ``stack``'s ``sum_bounds`` peaks (maxed over
    devices first): every real row's fixed-max sum of the head within
    [2^(_SUM_LO + margin), 2^(_SUM_HI - margin)] (``margin`` in log2 units).
    A head with no rows recorded (norm-proven layers) reads True."""
    top, inv = peaks[..., 0, 0, :], peaks[..., 1, 0, :]
    return (top <= 2.0 ** (_SUM_HI - margin)) & (inv <= 2.0 ** (-_SUM_LO - margin))


def layer_bounds(peaks: jax.Array) -> jax.Array:
    """``[L, heads]`` score bounds (log2 units) from ``stack``'s peaks,
    maxed over devices first: by Cauchy-Schwarz no score exceeds its head's
    largest q norm times its largest k norm."""
    return jnp.sqrt(peaks[..., 0, 0, :] * peaks[..., 1, 0, :])


def safe_layers(peaks: jax.Array, margin: float = 1.0) -> jax.Array:
    """``[L]`` bool: layers whose every head's bound, times ``margin``, keeps
    the fixed-max softmax exact (the f32 dot of bf16 rows is within
    128 x 2^-24 of the bound, far inside the slack)."""
    return (layer_bounds(peaks) * (margin * _BOUND_SLACK) <= _FIXED_BOUND).all(axis=-1)
