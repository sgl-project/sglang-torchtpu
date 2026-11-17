# SPDX-License-Identifier: Apache-2.0
"""TPU causal depthwise convolution on native per-layer Mamba state.

Prefill streams token tiles and their K-1 halo through double DMA buffers.
Only the first tile reads old state; the final state is committed once, after
all tiles finish. Decode batches eight requests into the vector sublanes.
The full state pool is never gathered, padded, transposed or copied.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_TOKEN_TILE = 64
_DECODE_TILE = 8


def _channel_tile(channels):
    # Every native state row and every channel partition must be DMA aligned.
    # Full-width copies alone do not make small channel counts DMA safe.
    if channels <= 0 or channels % 128:
        raise ValueError(
            "TPU ShortConv requires channels to be a positive multiple of 128"
        )
    # A partition of the rank-one bias also needs a power-of-two block size.
    return next(c for c in (512, 256, 128) if channels % c == 0)


def _prefill_kernel(
    slots,
    starts,
    has_prefix,
    x,
    state_in,
    weight,
    bias,
    state,
    out,
    history,
    tail,
    inputs,
    outputs,
    read_sem,
    write_sem,
    state_sem,
    *,
    tile,
    channels,
):
    cb, request = pl.program_id(0), pl.program_id(1)
    col = cb * channels
    slot = slots[request]
    start, end = starts[request], starts[request + 1]
    length = end - start
    h = history.shape[0]

    def state_dma():
        return pltpu.make_async_copy(
            state_in.at[slot, :, :, pl.ds(col, channels)], history, state_sem
        )

    def input_dma(i):
        offset = i * tile
        n = jnp.minimum(tile, length - offset)
        halo = jnp.where(i == 0, 0, h)
        return pltpu.make_async_copy(
            x.at[pl.ds(start + offset - halo, n + halo), :, pl.ds(col, channels)],
            inputs.at[i % 2, pl.ds(h - halo, n + halo)],
            read_sem.at[i % 2],
        )

    def output_dma(i):
        n = jnp.minimum(tile, length - i * tile)
        return pltpu.make_async_copy(
            outputs.at[i % 2, pl.ds(0, n)],
            out.at[pl.ds(start + i * tile, n), :, pl.ds(col, channels)],
            write_sem.at[i % 2],
        )

    def zeros(begin, stop):
        outputs[0] = jnp.zeros(outputs.shape[1:], outputs.dtype)

        @pl.loop(0, pl.cdiv(stop - begin, tile))
        def zero_tile(i):
            n = jnp.minimum(tile, stop - begin - i * tile)
            dma = pltpu.make_async_copy(
                outputs.at[0, pl.ds(0, n)],
                out.at[pl.ds(begin + i * tile, n), :, pl.ds(col, channels)],
                write_sem.at[0],
            )
            dma.start()
            dma.wait()

    @pl.when((slot > 0) & (length > 0))
    def active():
        history[...] = jnp.zeros(history.shape, jnp.float32)

        @pl.when(has_prefix[request] != 0)
        def load_history():
            state_dma().start()
            state_dma().wait()

        old = history[...]
        w = weight[...].astype(jnp.float32)
        b = None if bias is None else bias[...].astype(jnp.float32)[None, :]
        num_tiles = pl.cdiv(length, tile)
        input_dma(0).start()

        @pl.loop(0, num_tiles)
        def token_tile(i):
            input_dma(i).wait()

            @pl.when(i + 1 < num_tiles)
            def prefetch():
                input_dma(i + 1).start()

            # The first tile's halo comes from cache, subsequent halos from x.
            # Rows beyond the final partial DMA are never stored or committed.
            window = inputs[i % 2].astype(jnp.float32)
            window = jnp.where(
                i == 0, jnp.concatenate((old, window[h:]), axis=0), window
            )
            value = jnp.zeros((tile, 1, channels), jnp.float32)
            for k in range(h + 1):
                value = value + window[k : k + tile] * w[k : k + 1]
            if b is not None:
                value = value + b

            @pl.when(i >= 2)
            def free_output():
                output_dma(i - 2).wait()

            outputs[i % 2] = value.astype(outputs.dtype)
            output_dma(i).start()

        @pl.loop(jnp.maximum(num_tiles - 2, 0), num_tiles)
        def drain(i):
            output_dma(i).wait()

        # Read only the final K-1 inputs. For a short chunk preserve the
        # relevant old history (zeroed above on a fresh request).
        n = jnp.minimum(length, h)
        dma = pltpu.make_async_copy(
            x.at[pl.ds(end - n, n), :, pl.ds(col, channels)],
            tail.at[pl.ds(0, n)],
            read_sem.at[0],
        )
        dma.start()
        dma.wait()
        tail_values = tail[...].astype(jnp.float32)
        for size in range(1, h + 1):

            @pl.when(n == size)
            def update(size=size):
                values = [history[i + size, 0] for i in range(h - size)]
                values += [tail_values[i, 0] for i in range(size)]
                for i in range(h):
                    history[i, 0] = values[i]

        dma = pltpu.make_async_copy(
            history, state.at[slot, :, :, pl.ds(col, channels)], state_sem
        )
        dma.start()
        dma.wait()

    @pl.when((slot == 0) & (length > 0))
    def inactive():
        zeros(start, end)

    @pl.when(request == slots.shape[0] - 1)
    def padding():
        zeros(starts[slots.shape[0]], x.shape[0])


def _decode_kernel(
    slots,
    x,
    state_in,
    weight,
    bias,
    state,
    out,
    history,
    read_sem,
    write_sem,
    *,
    batch_tile,
    channels,
):
    cb, rb = pl.program_id(0), pl.program_id(1)
    col = cb * channels
    h = history.shape[1]
    history[...] = jnp.zeros(history.shape, jnp.float32)

    def load(row):
        return pltpu.make_async_copy(
            state_in.at[slots[rb * batch_tile + row], :, :, pl.ds(col, channels)],
            history.at[row],
            read_sem.at[row],
        )

    def store(row):
        return pltpu.make_async_copy(
            history.at[row],
            state.at[slots[rb * batch_tile + row], :, :, pl.ds(col, channels)],
            write_sem.at[row],
        )

    for row in range(batch_tile):
        pl.when(slots[rb * batch_tile + row] > 0)(lambda row=row: load(row).start())
    for row in range(batch_tile):
        pl.when(slots[rb * batch_tile + row] > 0)(lambda row=row: load(row).wait())

    value = jnp.zeros((batch_tile, channels), jnp.float32)
    w = weight[...].astype(jnp.float32)
    current = x[...].astype(jnp.float32)
    for k in range(h):
        prev = jnp.concatenate(
            [history[row, k, 0][None, :] for row in range(batch_tile)], axis=0
        )
        value = value + prev * w[k : k + 1]
    value = value + current * w[h : h + 1]
    if bias is not None:
        value = value + bias[...].astype(jnp.float32)[None, :]
    # Concatenate i32 rows before comparing: concatenating predicate vectors
    # asks Mosaic to bitcast i1 registers to i32 on multi-request tiles.
    slot_rows = jnp.concatenate(
        [
            jnp.broadcast_to(slots[rb * batch_tile + row], (1, channels))
            for row in range(batch_tile)
        ],
        axis=0,
    )
    # Store the whole BF16 tile: single-row stores would require packed
    # sublane read/modify/write on TPU.
    out[...] = jnp.where(slot_rows > 0, value, 0).astype(out.dtype)
    for row in range(batch_tile):
        active = slots[rb * batch_tile + row] > 0

        @pl.when(active)
        def commit(row=row):
            for k in range(h - 1):
                history[row, k, 0] = history[row, k + 1, 0]
            history[row, h - 1, 0] = current[row]
            store(row).start()

    for row in range(batch_tile):
        pl.when(slots[rb * batch_tile + row] > 0)(lambda row=row: store(row).wait())


def short_conv_pallas(
    x,
    weight,
    bias,
    state,
    slots,
    starts,
    has_prefix,
    *,
    decode,
    interpret=False,
):
    """Return (aliased FP32 state, x.dtype output); real slots must be unique.

    Shapes: x[T,C], weight[C,K], state[S,K-1,1,C]. C must be a positive
    multiple of 128 for native-state DMA. Slot 0 is never written.
    starts[B+1] describes packed prefill; prefix[B] controls old state reads.
    Decode consumes one token per slot. Metadata values do not specialize JIT.
    ``interpret=pltpu.InterpretParams(...)`` uses the TPU interpreter on CPU;
    the generic ``interpret=True`` does not support ragged dynamic-size DMA.
    """
    tokens, channels = x.shape
    h = weight.shape[1] - 1
    assert h > 0 and state.shape[1:] == (h, 1, channels)
    assert state.dtype == jnp.float32
    bc = _channel_tile(channels)
    # Transpose the small weights; pass the state tensor verbatim.
    weight = weight.T.astype(jnp.float32)
    bias = None if bias is None else bias.astype(jnp.float32)
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    wspec = pl.BlockSpec((h + 1, bc), lambda c, r, *_: (0, c))
    bspec = None if bias is None else pl.BlockSpec((bc,), lambda c, r, *_: (c,))
    state_shape = jax.ShapeDtypeStruct(state.shape, state.dtype)

    if decode:
        bt = min(_DECODE_TILE, tokens)
        padding = -tokens % bt
        x = jnp.pad(x, ((0, padding), (0, 0)))
        slots = jnp.pad(slots.astype(jnp.int32), ((0, padding),))
        xspec = pl.BlockSpec((bt, bc), lambda c, r, *_: (r, c))
        state, output = pl.pallas_call(
            functools.partial(_decode_kernel, batch_tile=bt, channels=bc),
            out_shape=(state_shape, jax.ShapeDtypeStruct(x.shape, x.dtype)),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(channels // bc, x.shape[0] // bt),
                in_specs=(xspec, hbm, wspec, bspec),
                out_specs=(hbm, xspec),
                scratch_shapes=(
                    pltpu.VMEM((bt, h, 1, bc), jnp.float32),
                    pltpu.SemaphoreType.DMA((bt,)),
                    pltpu.SemaphoreType.DMA((bt,)),
                ),
            ),
            input_output_aliases={2: 0},
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel")
            ),
            name="lfm2_short_conv_decode",
        )(slots, x, state, weight, bias)
        return state, output[:tokens]

    # Keep token offsets outside the two tiled minor dimensions. In [T,C],
    # ragged starts and the K-1 halo can cut a VMEM tile (including packed
    # BF16 rows). [T,1,C] permits whole-row DMAs at arbitrary token offsets;
    # use the same shape for input, output and tail staging. State stays native.
    x = x[:, None, :]
    state, output = pl.pallas_call(
        functools.partial(_prefill_kernel, tile=_TOKEN_TILE, channels=bc),
        out_shape=(state_shape, jax.ShapeDtypeStruct(x.shape, x.dtype)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(channels // bc, slots.shape[0]),
            in_specs=(hbm, hbm, wspec, bspec),
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((h, 1, bc), jnp.float32),
                pltpu.VMEM((h, 1, bc), x.dtype),
                pltpu.VMEM((2, _TOKEN_TILE + h, 1, bc), x.dtype),
                pltpu.VMEM((2, _TOKEN_TILE, 1, bc), x.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={4: 0},
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="lfm2_short_conv_prefill",
    )(
        slots.astype(jnp.int32),
        starts.astype(jnp.int32),
        has_prefix.astype(jnp.int32),
        x,
        state,
        weight,
        bias,
    )
    return state, output[:, 0, :]
