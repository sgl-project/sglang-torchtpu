# SPDX-License-Identifier: Apache-2.0
"""All-to-all by remote DMA for the Muse SP + EP programs, decode (decode/spep megakernel,
in-kernel) and prefill (prefill/spep, XLA-level pallas_calls).

Shared pieces: segments of rows start on SEG-row boundaries, move as SEG-row DMAs (local when
the peer is this rank, remote otherwise) and are waited for chunk by chunk (seg_copy,
wait_chunks).

Prefill (plan / pack_rows / unpack_rows / exchange): sender m sorts its (row, slot) pairs by
global expert g = d * El + e (d the expert's rank); receiver d lays out what it gets by local
expert, then by source, so the expert FFN reads the buffer as is. Every (source, expert)
segment starts on a SEG-row boundary on both sides; the combine sends each segment's outputs
back into the source's padded copy of its sorted order. Local experts form n_chunks chunks:
send kernels only start DMAs and return their semaphores, wait kernels block on one chunk, so
chunk j's expert FFN runs while chunk j + 1 arrives and chunk j - 1 returns. Dispatch rows
are u32 words (pack_rows): byte b of word c holds e4m3 element b * He / 4 + c, so unpack_rows
decodes lane-aligned slices; word He / 4 holds the f32 scale.

Decode (mk_*): inside the EP megakernel, every rank's slot counts first (one row each), then
per (source, destination) one contiguous segment of slot rows; the y rows go back one by one
from the expert loop (decode/spep/layer.py).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.common.config import MKConfig
from sglang_tpu.muse_megakernel.common.consts import (
    LANE,
    SP_COMBINE_COLLECTIVE_ID,
    SP_DISPATCH_COLLECTIVE_ID,
)

SEG = 8  # rows per DMA: an HBM (8, 128) tile of u32 / bf16 rows (offsets must be tile-aligned)


def seg_copy(src_buf, dst_buf, s0, d0, i, peer, me, recv, sent) -> None:
    """Start chunk i (SEG rows) of the segment at src_buf row s0 into peer's dst_buf row d0.
    The receiver's recv semaphore counts it; a remote copy also counts on this rank's sent."""
    src = src_buf.at[pl.ds(pl.multiple_of(s0 + i * SEG, SEG), SEG)]
    dst = dst_buf.at[pl.ds(pl.multiple_of(d0 + i * SEG, SEG), SEG)]

    @pl.when(peer == me)
    def _():
        pltpu.make_async_copy(src, dst, recv).start()

    @pl.when(peer != me)
    def _():
        pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=sent,
            recv_sem=recv,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()


def wait_chunks(buf, sem, n, rows: int = SEG) -> None:
    """Wait for n DMAs of `rows` rows of buf on sem (DMA semaphores count bytes)."""
    tmpl = pltpu.make_async_copy(buf.at[pl.ds(0, rows)], buf.at[pl.ds(0, rows)], sem)

    def one(_, carry):
        tmpl.wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


# ===== prefill (prefill/spep) =====

# each expert's group starts on this row in the receive buffer: gmm_v2 faults (TPU
# UserFatalError) on bf16 groups starting off a 16-row tile; the gap rows are never sent
GROUP_ALIGN = 16
UNPACK_ROWS = 256  # unpack_rows block

_HBM = pl.BlockSpec(memory_space=pltpu.HBM)
_SEM = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)


def _pad(n):
    return (n + SEG - 1) // SEG * SEG


def _exclusive_cumsum(x, axis: int = 0):
    return jnp.cumsum(x, axis=axis) - x


class A2APlan(NamedTuple):
    """One rank's segment tables, int32 [chunk, segments, 4]: from row, to row, SEG chunks,
    peer; counts int32 [chunk, 2]: SEG chunks received, SEG chunks sent to other ranks."""

    dispatch: jax.Array  # this rank's global experts, into their ranks' receive buffers
    dispatch_counts: jax.Array
    combine: jax.Array  # (source, local expert) segments, back to the sources
    combine_counts: jax.Array
    group_sizes: jax.Array  # [El] padded rows per local expert in the receive buffer
    chunk_rows: jax.Array  # [n_chunks + 1] receive-buffer row where each chunk starts
    # [g] padded-layout row of a global expert's first pair minus its sorted index; this
    # rank's send buffer and its combine destination share that layout
    send_shift: jax.Array


def plan(cnt_all, me, tp: int, el: int, n_chunks: int) -> A2APlan:
    """cnt_all [tp, tp * el] int32: source s's pairs per global expert (every rank's view)."""
    g, ec = tp * el, el // n_chunks
    pc = _pad(cnt_all)  # [s, g]
    pc3 = pc.reshape(tp, tp, el)  # [s, d, e]
    # [d, e] receiver rows per expert, GROUP_ALIGN-padded
    tot = (pc3.sum(0) + GROUP_ALIGN - 1) // GROUP_ALIGN * GROUP_ALIGN
    # receiver d: expert-major, then source
    dst_off = _exclusive_cumsum(tot, axis=1)[None] + _exclusive_cumsum(pc3, axis=0)
    pad_src = _exclusive_cumsum(pc, axis=1).reshape(
        tp, tp, el
    )  # [s, d, e] sender layout
    ranks = jnp.broadcast_to(jnp.arange(tp, dtype=jnp.int32)[:, None], (tp, el))
    others = ranks != me  # [peer, e]

    def by_chunk(x):  # [peer, e] -> [chunk, peer * ec]
        return x.reshape(tp, n_chunks, ec).transpose(1, 0, 2).reshape(n_chunks, -1)

    dispatch = jnp.stack(
        [by_chunk(t) for t in (pad_src[me], dst_off[me], pc3[me] // SEG, ranks)],
        axis=-1,
    )
    mine_in = pc3[:, me, :]  # [s, e] what I receive (and later send back)
    combine = jnp.stack(
        [
            by_chunk(t)
            for t in (dst_off[:, me, :], pad_src[:, me, :], mine_in // SEG, ranks)
        ],
        axis=-1,
    )
    dispatch_counts = jnp.stack(
        [
            by_chunk(mine_in).sum(1),
            by_chunk(jnp.where(others, pc3[me], 0)).sum(1),
        ],
        axis=1,
    )
    combine_counts = jnp.stack(
        [
            by_chunk(pc3[me]).sum(1),
            by_chunk(jnp.where(others, mine_in, 0)).sum(1),
        ],
        axis=1,
    )
    chunk_tot = tot[me].reshape(n_chunks, ec).sum(1)
    chunk_rows = jnp.concatenate(
        [jnp.zeros((1,), chunk_tot.dtype), jnp.cumsum(chunk_tot)]
    )
    return A2APlan(
        dispatch.astype(jnp.int32),
        (dispatch_counts // SEG).astype(jnp.int32),
        combine.astype(jnp.int32),
        (combine_counts // SEG).astype(jnp.int32),
        tot[me].astype(jnp.int32),
        chunk_rows.astype(jnp.int32),
        (pad_src[me].reshape(g) - _exclusive_cumsum(cnt_all[me])).astype(jnp.int32),
    )


def padded_rows(pl_: A2APlan, sorted_ids) -> jax.Array:
    """[P] the padded-layout row of each sorted pair (sorted_ids: its global expert)."""
    return jnp.arange(sorted_ids.shape[0], dtype=jnp.int32) + pl_.send_shift[sorted_ids]


def chunk_group_sizes(pl_: A2APlan, j: int, n_chunks: int):
    """gmm_v2 group sizes (use group_offset 1) covering only chunk j's experts: a leading
    skipped group spans the rows before the chunk."""
    el = pl_.group_sizes.shape[0]
    ec = el // n_chunks
    mask = (jnp.arange(el) // ec) == j
    return jnp.concatenate(
        [pl_.chunk_rows[j : j + 1], jnp.where(mask, pl_.group_sizes, 0)]
    ).astype(jnp.int32)


def row_words(he: int) -> int:
    """u32 words per dispatch row: He e4m3 bytes, the scale, lane padding."""
    return he // 4 + LANE


def pack_rows(x8, s):
    """e4m3 rows [n, He] and f32 scales [n, 1] -> u32 dispatch rows [n, row_words(He)]."""
    n, he = x8.shape
    b = jax.lax.bitcast_convert_type(x8, jnp.uint8).astype(jnp.uint32)
    b = b.reshape(n, 4, he // 4)
    words = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
    return jnp.concatenate(
        [
            words,
            jax.lax.bitcast_convert_type(s, jnp.uint32),
            jnp.zeros((n, LANE - 1), jnp.uint32),
        ],
        axis=1,
    )


def unpack_rows(words, lo, hi, he: int):
    """u32 dispatch rows [cap, row_words(he)] -> bf16 [cap, he]; only the UNPACK_ROWS blocks
    covering rows [lo, hi) (traced) are decoded, the rest of the output is left unwritten."""
    cap = words.shape[0]
    q = he // 4
    if cap % UNPACK_ROWS:
        raise ValueError(f"receive rows {cap} not a multiple of {UNPACK_ROWS}")

    def kernel(blk_ref, w_ref, o_ref):
        @pl.when(blk_ref[0] + pl.program_id(0) < blk_ref[1])
        def _():
            s = pltpu.bitcast(w_ref[:, q : q + LANE], jnp.float32)[:, :1]
            w = w_ref[:, :q]
            for b in range(4):
                byte = (w >> (8 * b)) & 0xFF
                # e4m3 sign | exponent, mantissa moved into f32's fields: the value times
                # 2^-120 (e4m3 subnormals land on f32 subnormals, flushed to zero)
                bits = ((byte & 0x80) << 24) | ((byte & 0x7F) << 20)
                x = pltpu.bitcast(bits, jnp.float32) * (2.0**120)
                o_ref[:, b * q : (b + 1) * q] = (x * s).astype(o_ref.dtype)

    def block(i, blk):
        # past the last block, revisit it: no fetch, no write-back
        return (jnp.minimum(blk[0] + i, jnp.maximum(blk[1] - 1, blk[0])), 0)

    blk = jnp.stack([lo // UNPACK_ROWS, (hi + UNPACK_ROWS - 1) // UNPACK_ROWS]).astype(
        jnp.int32
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((cap, he), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(cap // UNPACK_ROWS,),
            in_specs=[pl.BlockSpec((UNPACK_ROWS, words.shape[1]), block)],
            out_specs=pl.BlockSpec((UNPACK_ROWS, he), block),
        ),
        name="sp_moe_unpack",
    )(blk, words)


def silu_mul_chunks(h, lo, hi, chunk: int):
    """bf16 [cap, 2I] w13 outputs whose columns are per-chunk [gate | up] (decode/spep
    chunk_cols) -> bf16 [cap, I] silu(gate) * up, chunk by chunk in lane-aligned slices;
    only the UNPACK_ROWS blocks covering rows [lo, hi) are computed (no relayout of the
    whole receive buffer)."""
    cap, two_i = h.shape
    n = two_i // (2 * chunk)
    if cap % UNPACK_ROWS:
        raise ValueError(f"receive rows {cap} not a multiple of {UNPACK_ROWS}")

    def kernel(blk_ref, h_ref, o_ref):
        @pl.when(blk_ref[0] + pl.program_id(0) < blk_ref[1])
        def _():
            for j in range(n):
                g = h_ref[:, 2 * j * chunk : (2 * j + 1) * chunk].astype(jnp.float32)
                u = h_ref[:, (2 * j + 1) * chunk : (2 * j + 2) * chunk].astype(
                    jnp.float32
                )
                o_ref[:, j * chunk : (j + 1) * chunk] = (jax.nn.silu(g) * u).astype(
                    o_ref.dtype
                )

    def block(i, blk):
        return (jnp.minimum(blk[0] + i, jnp.maximum(blk[1] - 1, blk[0])), 0)

    blk = jnp.stack([lo // UNPACK_ROWS, (hi + UNPACK_ROWS - 1) // UNPACK_ROWS]).astype(
        jnp.int32
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((cap, two_i // 2), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(cap // UNPACK_ROWS,),
            in_specs=[pl.BlockSpec((UNPACK_ROWS, two_i), block)],
            out_specs=pl.BlockSpec((UNPACK_ROWS, two_i // 2), block),
        ),
        name="sp_moe_silu_mul",
    )(blk, h)


def _send(
    tp: int, rows_out: int, width: int, dtype, cid: int, name: str, chained: bool
):
    """Start every table chunk's SEG-row DMAs of src into the peers' out buffers; returns
    (out, recv_0, sent_0, recv_1, ...) scalar DMA semaphores per chunk without waiting
    (Mosaic rejects semaphore-array operands). chained: out aliases the trailing operand
    (a buffer earlier sends of this exchange wrote)."""

    def kernel(tab_ref, me_ref, src_hbm, *refs):
        n_chunks = tab_ref.shape[0]
        out_hbm, sems = refs[-1 - 2 * n_chunks], refs[-2 * n_chunks :]
        me = me_ref[0]
        bar = pltpu.get_barrier_semaphore()
        for p in range(tp):
            pl.semaphore_signal(
                bar, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(bar, tp)
        n_seg = tab_ref.shape[1]

        for c in range(n_chunks):
            recv, sent = sems[2 * c], sems[2 * c + 1]

            def segment(i, carry, c=c, recv=recv, sent=sent):
                fr, to, n, peer = (tab_ref[c, i, j] for j in range(4))

                def chunk(k, cc):
                    seg_copy(src_hbm, out_hbm, fr, to, k, peer, me, recv, sent)
                    return cc

                return jax.lax.fori_loop(0, n, chunk, carry)

            jax.lax.fori_loop(0, n_seg, segment, 0)

    def run(tab, me, src, prev=None):
        n_chunks = tab.shape[0]
        operands = (tab, me, src) + ((prev,) if chained else ())
        return pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((rows_out, width), dtype),)
            + (pltpu.SemaphoreType.DMA(()),) * (2 * n_chunks),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[_HBM] * (2 if chained else 1),
                out_specs=(_HBM,) + (_SEM,) * (2 * n_chunks),
            ),
            input_output_aliases={3: 0} if chained else {},
            compiler_params=pltpu.CompilerParams(collective_id=cid),
            name=name,
        )(*operands)

    return run


def _wait(name: str):
    """(out, counts [m] SEG chunks, m scalar DMA semaphores, *keep) -> out, once each
    semaphore saw its count of SEG-row chunks. keep: buffers outstanding DMAs still read."""

    def run(out, counts, sems, *keep):
        m = len(sems)

        def kernel(cnt_ref, out_hbm, *refs):
            for i in range(m):
                wait_chunks(out_hbm, refs[i], cnt_ref[i])

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[_HBM] + [_SEM] * m + [_HBM] * len(keep),
                out_specs=_HBM,
            ),
            input_output_aliases={1: 0},
            name=name,
        )(counts, out, *sems, *keep)

    return run


class Exchange(NamedTuple):
    """The streamed dispatch / combine kernels of one SP MoE geometry."""

    dispatch_send: object
    combine_first: object  # chunk 0's combine send: allocates the destination
    combine_next: object  # later chunks: chained onto the same destination
    wait: object


def exchange(tp: int, cap: int, padded_send_rows: int, he: int) -> Exchange:
    def combine(chained):
        return _send(
            tp,
            padded_send_rows,
            he,
            jnp.bfloat16,
            SP_COMBINE_COLLECTIVE_ID,
            "sp_moe_combine",
            chained=chained,
        )

    return Exchange(
        _send(
            tp,
            cap,
            row_words(he),
            jnp.uint32,
            SP_DISPATCH_COLLECTIVE_ID,
            "sp_moe_dispatch",
            chained=False,
        ),
        combine(False),
        combine(True),
        _wait("sp_moe_wait"),
    )


# ===== decode megakernel (decode/spep) =====


def mk_exchange_counts(c: MKConfig, S: dict, me) -> None:
    """ep_cnt_v[s] = rank s's slots per expert, on every rank."""
    src = S["ep_csend"].at[pl.ds(0, 1)]
    dst = S["ep_cnt_v"].at[pl.ds(me, 1)]
    recv, sent = S["ep_cnt_sems"].at[0], S["ep_cnt_sems"].at[1]
    pltpu.make_async_copy(src, dst, recv).start()
    for d in range(1, c.tp):
        peer = (me + d) % c.tp
        pltpu.make_async_remote_copy(
            src_ref=src, dst_ref=dst, send_sem=sent, recv_sem=recv, device_id=(peer,)
        ).start()
    tmpl = pltpu.make_async_copy(src, dst, recv)
    for _ in range(c.tp):
        tmpl.wait()
    sent_tmpl = pltpu.make_async_copy(src, dst, sent)
    for _ in range(c.tp - 1):
        sent_tmpl.wait()


def mk_dispatch(c: MKConfig, S: dict, me) -> None:
    """This rank's send segment for each destination d (ep_xs) into d's receive buffer
    (ep_xr) after the sources before this rank."""
    from sglang_tpu.muse_megakernel.decode.spep.route import T_DST, T_SEND, seg_dmas

    tab = S["ep_tab_s"]
    recv, sent = S["ep_a2a_sems"].at[0], S["ep_a2a_sems"].at[1]
    for d in range(c.tp):
        s0, d0 = tab[T_SEND, d], tab[T_DST, d]

        def one(i, carry, s0=s0, d0=d0, d=d):
            seg_copy(S["ep_xs"], S["ep_xr"], s0, d0, i, d, me, recv, sent)
            return carry

        jax.lax.fori_loop(0, seg_dmas(S, me, d), one, 0)


def mk_wait_dispatch(c: MKConfig, S: dict, me) -> None:
    """Waits for every source's segment into ep_xr and this rank's sends out of ep_xs."""
    from sglang_tpu.muse_megakernel.decode.spep.route import T_DMAS, seg_dmas

    tab = S["ep_tab_s"]
    wait_chunks(S["ep_xr"], S["ep_a2a_sems"].at[0], tab[T_DMAS, 0])
    wait_chunks(
        S["ep_xs"], S["ep_a2a_sems"].at[1], tab[T_DMAS, 1] - seg_dmas(S, me, me)
    )


def mk_wait_give_back(c: MKConfig, S: dict, me) -> None:
    """Waits for the y rows of this rank's slots on other ranks into ep_yb (the expert loop
    returns each item's rows to their sources, own ones locally)."""
    from sglang_tpu.muse_megakernel.decode.spep.route import T_N

    tab = S["ep_tab_s"]
    n = sum(tab[T_N + me, d] for d in range(c.tp)) - tab[T_N + me, me]
    wait_chunks(S["ep_yb"], S["ep_a2a_sems"].at[2], n, rows=1)
