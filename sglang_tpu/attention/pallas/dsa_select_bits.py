# SPDX-License-Identifier: Apache-2.0
"""Per-query DSA selections as bitmaps, built on the SparseCore.

The query-blocked sparse MLA prefill (``mla_sparse_qblock_tokens``) reads a
block's pages once and restores each query's own selection with a mask. For
plain DSA (one token per selection, Kimi-K3.1) the selection is the indexer's
top-k token list, ``i32[num_tokens, topk]`` of positions with ``-1`` padding.
This module turns each list into a bitmap of the request's positions.

Bit layout ("strided"): with ``num_words`` a power of two, position ``p`` is
bit ``p // num_words`` of word ``p % num_words``. A page of ``page_size``
consecutive positions (``page_size`` a power of two dividing ``num_words``) is
then ``page_size`` consecutive words at one bit index, which the attention
kernel reads as one lane-aligned ``[queries, page_size]`` tile and a shift.

Building the bitmap is a scatter of ``topk`` bits per row. XLA's scatter takes
~10 ms for a 1024 x 2048 batch on TPU7x; here each SparseCore subcore owns
whole rows and scatters into its tile memory (~50 us for that batch).
``addupdate_scatter`` adds atomically, also between lanes of one vector that
hit the same word, and a row lists each position at most once, so adding the
bits is OR-ing them.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

LANES = 16
# Rows per subcore pipeline step: amortizes the step's DMA issue and wait.
_ROWS_PER_STEP = 4


def _cdiv(a, b):
    return (a + b - 1) // b


def select_bits_num_words(num_positions: int, page_size: int) -> int:
    """Words per row for ``num_positions`` positions: a power of two, at least
    ``page_size`` (so a page never wraps) and holding 32 positions per word."""
    nw = page_size
    while nw * 32 < num_positions:
        nw *= 2
    return nw


def _bits_kernel(
    idx_hbm,  # i32[padded_rows * topk]
    out_hbm,  # i32[padded_rows * num_words]
    *,
    rows: int,
    topk: int,
    num_words: int,
    rows_per_step: int,
    num_waves: int,
    num_workers: int,
):
    sc = pltpu.get_tpu_info().sparse_core
    flat = lax.axis_index("core") * sc.num_subcores + lax.axis_index("subcore")
    shift = num_words.bit_length() - 1
    zeros = jnp.zeros((LANES,), jnp.int32)
    last_step = _cdiv(rows, rows_per_step) - 1

    def step(idx_vmem, out_vmem):
        for r in range(rows_per_step):
            out_row = out_vmem.at[pl.ds(r * num_words, num_words)]

            def zero(i, out_row=out_row):
                out_row[pl.ds(i * LANES, LANES)] = zeros

            plsc.parallel_loop(0, num_words // LANES, unroll=8)(zero)

            def scatter(i, r=r, out_row=out_row):
                p = idx_vmem[pl.ds(r * topk + i * LANES, LANES)]
                ok = p >= 0
                p = jnp.maximum(p, 0)
                word = jnp.bitwise_and(p, num_words - 1)
                bit = jnp.left_shift(jnp.int32(1), jnp.right_shift(p, shift))
                plsc.addupdate_scatter(out_row, (word,), bit, mask=ok)

            plsc.parallel_loop(0, topk // LANES, unroll=8)(scatter)

    buf = 1 if num_waves == 1 else 2
    pltpu.emit_pipeline(
        step,
        grid=(num_waves,),
        in_specs=(
            # Subcores past the last rows re-read the last step's rows; their
            # output rows are padding and are sliced off.
            pl.BlockSpec(
                (rows_per_step * topk,),
                lambda w: (jnp.minimum(w * num_workers + flat, last_step),),
                pipeline_mode=pl.Buffered(buf),
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                (rows_per_step * num_words,),
                lambda w: (w * num_workers + flat,),
                pipeline_mode=pl.Buffered(buf),
            ),
        ),
    )(idx_hbm, out_hbm)


@functools.partial(jax.jit, static_argnames=("num_words",))
def select_bits(topk_indices: jax.Array, *, num_words: int) -> jax.Array:
    """``i32[rows, topk]`` position lists (``-1`` padded, no repeats) to
    ``i32[rows, num_words]`` bitmaps in the strided layout (module docstring).

    Positions must be below ``32 * num_words``; ``num_words`` is a power of two
    and a multiple of 16.
    """
    rows, topk = topk_indices.shape
    if num_words & (num_words - 1) or num_words % LANES:
        raise ValueError(f"num_words must be a power of two >= 16, got {num_words}")
    if topk % LANES:
        raise ValueError(f"topk must be a multiple of {LANES}, got {topk}")
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    num_workers = sc.num_cores * sc.num_subcores
    rps = _ROWS_PER_STEP if rows % _ROWS_PER_STEP == 0 else 1
    num_waves = _cdiv(rows // rps, num_workers)
    padded = num_waves * num_workers * rps
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
    )
    out = pl.kernel(
        functools.partial(
            _bits_kernel,
            rows=rows,
            topk=topk,
            num_words=num_words,
            rows_per_step=rps,
            num_waves=num_waves,
            num_workers=num_workers,
        ),
        out_type=jax.ShapeDtypeStruct((padded * num_words,), jnp.int32),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        name=f"dsa_select_bits_r{rows}_k{topk}_w{num_words}",
    )(topk_indices.astype(jnp.int32).reshape(-1))
    return out.reshape(padded, num_words)[:rows]


def select_bits_reference(topk_indices: jax.Array, *, num_words: int) -> jax.Array:
    """XLA version of ``select_bits`` (slow: a scatter of every entry)."""
    rows, _ = topk_indices.shape
    shift = num_words.bit_length() - 1
    p = jnp.maximum(topk_indices, 0)
    word = jnp.where(topk_indices >= 0, p & (num_words - 1), num_words)
    bit = jnp.left_shift(jnp.uint32(1), (p >> shift).astype(jnp.uint32))
    r = jnp.broadcast_to(jnp.arange(rows)[:, None], p.shape)
    out = jnp.zeros((rows, num_words), jnp.uint32).at[r, word].add(bit, mode="drop")
    return lax.bitcast_convert_type(out, jnp.int32)
