# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two small TP kernels at the decoder's token ends, for ops inside a ``tp``
``shard_map`` whose ranks hold the vocabulary in mesh order (rank ``r``: rows
``[r * V / n, (r + 1) * V / n)`` of the embedding, columns of the LM head).

- :func:`owner_embedding` (``KernelConfig.decode_embedding_broadcast``): the
  rows' input embeddings. The rank holding a token's row reads it and sends
  it to the other ranks, replacing the local gather and the FP32 ``psum``.
- :func:`greedy_decisions` (``KernelConfig.verify_greedy_kernel``): a greedy
  verify's accepted counts and bonus tokens from the vocabulary-sharded
  logits, with one exchange of per-shard maxima, replacing ``jnp.argmax``,
  an ``all_gather`` and the small XLA ops around them.

Both return the XLA code's bits; each function says why.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

AXIS = "tp"
# Barrier semaphores (CompilerParams.collective_id) of their own.
GREEDY_COLLECTIVE_ID = 27
EMBEDDING_COLLECTIVE_ID = 28
# Largest batches the kernels take (rows along the 128 lanes for the greedy
# exchange; one VMEM row block per row for the embedding).
GREEDY_MAX_ROWS = 128
EMBEDDING_MAX_ROWS = 32
INT32_MIN = -(2**31)


def _ranks() -> int:
    ranks = jax.lax.axis_size(AXIS)
    if ranks & (ranks - 1):
        raise ValueError(f"the token kernels need a power-of-two {AXIS} axis: {ranks}")
    return ranks


def _signal_barrier(rank, ranks):
    """Signal every peer; ``_wait_barrier`` then waits for all of them."""
    semaphore = tpu.get_barrier_semaphore()
    for offset in range(1, ranks):
        pl.semaphore_signal(
            semaphore,
            1,
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    return semaphore


def _wait_barrier(semaphore, ranks):
    pl.semaphore_wait(semaphore, ranks - 1)


def _remote(source, target, send, receive, peer):
    return tpu.make_async_remote_copy(
        source,
        target,
        send,
        receive,
        device_id=(peer,),
        device_id_type=pl.DeviceIdType.MESH,
    )


# --------------------------------------------------------------------------
# Embedding rows from their owners


def _pack_row(row):
    """A BF16-valued FP32 ``[1, width]`` row -> FP32 ``[width / 256, 128]``
    words of BF16 pairs (``dspark.pack``)."""
    width = row.shape[1]
    lanes = row.reshape(width // 128, 128).astype(jnp.bfloat16)
    return jax.lax.bitcast_convert_type(tpu.bitcast(lanes, jnp.uint32), jnp.float32)


def _unpack_row(words, width):
    """Inverse of :func:`_pack_row`: FP32 ``[1, width]`` (``dspark.unpack``)."""
    values = tpu.bitcast(jax.lax.bitcast_convert_type(words, jnp.uint32), jnp.bfloat16)
    return values.astype(jnp.float32).reshape(1, width)


def owner_embedding(tokens: jax.Array, table: jax.Array) -> jax.Array:
    """``[rows, hidden]`` BF16: row ``b`` the embedding of ``tokens[b]``, the
    same on every rank, from this rank's ``table`` ``[V / n, hidden]``.

    The rank holding a token's row reads its 16-row block from HBM and
    selects the row with a one-hot FP32 sum, then sends it to the other ranks
    as packed FP32 words; a token outside the vocabulary gives a zero row.

    This matches ``kimi.tp_sum`` of the local rows (out of range: 0) cast to
    BF16 bit for bit. That psum adds the owner's BF16 value, as FP32, to
    zeros from the other ranks, which is exact except that -0.0 becomes +0.0.
    The one-hot sum adds zeros to the row in the same way, so -0.0 also
    becomes +0.0.
    """
    (rows,) = tokens.shape
    vocabulary, hidden = table.shape
    if rows > EMBEDDING_MAX_ROWS or vocabulary % 16 or hidden % 256:
        raise ValueError(f"owner_embedding: {rows} rows of a {table.shape} table")
    ranks = _ranks()
    words = hidden // 256

    def body(tokens_ref, table_hbm, out_ref, blocks, packed, copies, sends, receives):
        rank = jax.lax.axis_index(AXIS)
        owners, local = [], []
        for b in range(rows):
            token = tokens_ref[b]
            inside = (token >= 0) & (token < ranks * vocabulary)
            owners.append(jnp.where(inside, token // vocabulary, -1))
            local.append(jnp.clip(token - rank * vocabulary, 0, vocabulary - 1))

        def block_copy(b):
            start = pl.multiple_of((local[b] // 16) * 16, 16)
            return tpu.make_async_copy(
                table_hbm.at[pl.ds(start, 16)], blocks.at[b], copies.at[b]
            )

        def row_copy(b, peer):
            return _remote(
                packed.at[b], packed.at[b], sends.at[b], receives.at[b], peer
            )

        # This rank's rows leave HBM before the barrier (local copies only).
        for b in range(rows):
            pl.when(owners[b] == rank)(block_copy(b).start)
        barrier = _signal_barrier(rank, ranks)
        for b in range(rows):

            @pl.when(owners[b] == rank)
            def select(b=b):
                block_copy(b).wait()
                index = jax.lax.broadcasted_iota(jnp.int32, (16, hidden), 0)
                row = jnp.sum(
                    jnp.where(
                        index == local[b] % 16, blocks[b].astype(jnp.float32), 0.0
                    ),
                    axis=0,
                    keepdims=True,
                )
                packed[b] = _pack_row(row)

            @pl.when(owners[b] < 0)
            def zero(b=b):
                packed[b] = jnp.zeros((words, 128), jnp.float32)

        _wait_barrier(barrier, ranks)
        for b in range(rows):

            @pl.when(owners[b] == rank)
            def send(b=b):
                @pl.loop(1, ranks)
                def start(offset):
                    row_copy(b, rank ^ offset).start()

        for b in range(rows):

            @pl.when((owners[b] >= 0) & (owners[b] != rank))
            def receive(b=b):
                row_copy(b, rank).wait_recv()

        for b in range(rows):

            @pl.when(owners[b] == rank)
            def sent(b=b):
                @pl.loop(1, ranks)
                def wait(offset):
                    row_copy(b, rank ^ offset).wait_send()

        out_ref[...] = jnp.concatenate(
            [_unpack_row(packed[b], hidden) for b in range(rows)], axis=0
        ).astype(out_ref.dtype)

    return pl.pallas_call(
        body,
        name="kimi_owner_embedding",
        out_shape=jax.ShapeDtypeStruct((rows, hidden), table.dtype),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
            scratch_shapes=[
                tpu.VMEM((rows, 16, hidden), table.dtype),
                tpu.VMEM((rows, words, 128), jnp.float32),
                tpu.SemaphoreType.DMA((rows,)),
                tpu.SemaphoreType.DMA((rows,)),
                tpu.SemaphoreType.DMA((rows,)),
            ],
        ),
        compiler_params=tpu.CompilerParams(collective_id=EMBEDDING_COLLECTIVE_ID),
    )(tokens.astype(jnp.int32), table)


# --------------------------------------------------------------------------
# Greedy verify decisions


def _first_maxima(logits, rank, width):
    """``[n, 128]`` int32 packet rows of ``logits`` ``[n, width]`` FP32: lane 0
    the row's maximum (FP32 bits), lane 1 the global id of its first index,
    lane 2 1 if the row holds a NaN (then lane 1 is the first NaN's id)."""
    n = logits.shape[0]
    index = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
    nan = logits != logits
    first_nan = jnp.min(jnp.where(nan, index, width), axis=1, keepdims=True)
    clean = jnp.where(nan, -jnp.inf, logits)
    best = jnp.max(clean, axis=1, keepdims=True)
    first_best = jnp.min(jnp.where(clean == best, index, width), axis=1, keepdims=True)
    has_nan = first_nan < width
    local_id = jnp.where(has_nan, first_nan, first_best)
    lane = jax.lax.broadcasted_iota(jnp.int32, (n, 128), 1)
    return jnp.where(
        lane == 0,
        jax.lax.bitcast_convert_type(best, jnp.int32),
        jnp.where(
            lane == 1,
            local_id + rank * width,
            jnp.where(lane == 2, has_nan.astype(jnp.int32), 0),
        ),
    )


def greedy_decisions(
    local_logits: jax.Array, tokens: jax.Array, forced: jax.Array
) -> jax.Array:
    """``[requests, 2]`` int32 (accepted count, bonus token), replicated:
    ``sampling.verify_acceptance``'s greedy decisions for ``tokens``
    ``[requests, rows]`` (row 0 the anchor, row k the draft verified at row
    k - 1) from this rank's ``local_logits`` ``[requests * rows, V / n]``
    FP32, ``forced`` ``[requests]`` (``>= 0``: the count).

    Each rank finds every row's first-index maximum in its shard and the
    ranks exchange them (rows along lanes, one 31-peer round at 32 ranks);
    the lowest rank holding the overall maximum gives the row's target. That
    is ``sampling.greedy``: ``jnp.argmax`` takes the first index of the
    maximum in a shard and, over the gathered maxima, the lowest rank, so the
    targets are each row's first maximum in vocabulary order. NaN counts as
    the maximum and the first NaN wins, as in ``jnp.argmax``. Then, as
    ``verify_acceptance``: the count is the leading drafts equal to their
    rows' targets (or ``forced``) and the bonus the target at row ``count``
    (``INT32_MIN`` past the last row, ``take_along_axis``'s fill)."""
    requests, rows = tokens.shape
    total, width = local_logits.shape
    if (
        total != requests * rows
        or total > GREEDY_MAX_ROWS
        or rows < 2
        or local_logits.dtype != jnp.float32
    ):
        raise ValueError(
            f"greedy_decisions: {local_logits.shape} {local_logits.dtype} logits "
            f"for {requests}x{rows} rows"
        )
    ranks = _ranks()

    def body(
        tokens_ref,
        forced_ref,
        logits_hbm,
        out_ref,
        logits,
        gathered,
        copy,
        sends,
        receives,
    ):
        rank = jax.lax.axis_index(AXIS)
        load = tpu.make_async_copy(logits_hbm, logits, copy.at[0])
        load.start()
        barrier = _signal_barrier(rank, ranks)
        load.wait()
        packets = [
            _first_maxima(logits[start : min(start + 8, total), :], rank, width)
            for start in range(0, total, 8)
        ]
        if total < 128:
            packets.append(jnp.zeros((128 - total, 128), jnp.int32))
        square = jnp.concatenate(packets, axis=0)
        # Rows along lanes: sublane 0 maxima, 1 ids, 2 NaN flags.
        gathered[rank] = jnp.transpose(square)[:8]
        _wait_barrier(barrier, ranks)
        exchange = [
            _remote(
                gathered.at[rank],
                gathered.at[rank],
                sends.at[offset - 1],
                receives.at[offset - 1],
                rank ^ offset,
            )
            for offset in range(1, ranks)
        ]
        for item in exchange:
            item.start()
        for item in exchange:
            item.wait()
        packets = gathered[...]  # [ranks, 8, 128]
        maxima = jax.lax.bitcast_convert_type(packets[:, 0:1, :], jnp.float32)
        ids, nan = packets[:, 1:2, :], packets[:, 2:3, :] != 0
        source = jax.lax.broadcasted_iota(jnp.int32, (ranks, 1, 128), 0)
        first_nan = jnp.min(jnp.where(nan, source, ranks), axis=0)
        clean = jnp.where(nan, -jnp.inf, maxima)
        best = jnp.max(clean, axis=0)
        first_best = jnp.min(jnp.where(clean == best[None], source, ranks), axis=0)
        winner = jnp.where(first_nan < ranks, first_nan, first_best)
        targets = jnp.sum(jnp.where(source == winner[None], ids, 0), axis=0)
        # Lane b: row b's target; ``drafts`` lane b: the draft row b verified.
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        drafts = jnp.zeros((1, 128), jnp.int32)
        for b in range(total):
            if b % rows < rows - 1:
                drafts = jnp.where(lane == b, tokens_ref[b + 1], drafts)
        miss = drafts != targets
        row = jax.lax.broadcasted_iota(jnp.int32, (requests, 128), 0)
        column = jax.lax.broadcasted_iota(jnp.int32, (requests, 128), 1)
        decisions = jnp.zeros((requests, 128), jnp.int32)
        for r in range(requests):
            offset = lane - r * rows
            first = jnp.min(
                jnp.where((offset >= 0) & (offset < rows - 1) & miss, offset, rows - 1),
                axis=1,
                keepdims=True,
            )
            forced_r = jnp.zeros((1, 1), jnp.int32) + forced_ref[r]
            count = jnp.where(forced_r >= 0, forced_r, first)
            bonus = jnp.sum(
                jnp.where(offset == count, targets, 0), axis=1, keepdims=True
            )
            bonus = jnp.where(count < rows, bonus, INT32_MIN)
            decisions = jnp.where(
                (row == r) & (column == 0),
                count,
                jnp.where((row == r) & (column == 1), bonus, decisions),
            )
        out_ref[...] = decisions[:, :2]

    return pl.pallas_call(
        body,
        name="kimi_greedy_decisions",
        out_shape=jax.ShapeDtypeStruct((requests, 2), jnp.int32),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
            scratch_shapes=[
                tpu.VMEM((total, width), jnp.float32),
                tpu.VMEM((ranks, 8, 128), jnp.int32),
                tpu.SemaphoreType.DMA((1,)),
                tpu.SemaphoreType.DMA((ranks - 1,)),
                tpu.SemaphoreType.DMA((ranks - 1,)),
            ],
        ),
        compiler_params=tpu.CompilerParams(collective_id=GREEDY_COLLECTIVE_ID),
    )(
        tokens.reshape(-1).astype(jnp.int32),
        forced.astype(jnp.int32),
        local_logits,
    )
