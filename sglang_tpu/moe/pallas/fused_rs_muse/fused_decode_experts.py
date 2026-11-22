# SPDX-License-Identifier: Apache-2.0
"""Whole FP4 experts, biased top-k and owner fan-out from source router logits."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
    _fp4_k256_dual_gmm,
    _fp4_k256_gmm,
)


def route_weights(logits, bias):
    """Use the serving router's tie rule and sum its eight weights in selection order."""
    scores = 1.0 / (1.0 + jnp.exp(-logits))
    choice = scores + bias
    ids = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    slots = lax.broadcasted_iota(jnp.int32, (scores.shape[0], 8), 1)
    in_order = jnp.zeros(slots.shape, jnp.float32)
    dense = jnp.zeros_like(scores)
    for slot in range(8):
        peak = jnp.max(choice, axis=1, keepdims=True)
        best = jnp.min(
            jnp.where(choice == peak, ids, scores.shape[1]), axis=1, keepdims=True
        )
        pick = ids == best
        selected = jnp.where(pick, scores, 0.0)
        in_order = jnp.where(
            slots == slot, selected.sum(axis=1, keepdims=True), in_order
        )
        dense = jnp.where(pick, scores, dense)
        choice = jnp.where(pick, -jnp.inf, choice)
    return dense / (in_order.sum(axis=1, keepdims=True) + 1e-15)


def _kernel(
    rank,
    logits,
    bias,
    x,
    w1,
    s1,
    w2,
    s2,
    norm,
    output,
    w1_buf,
    s1_buf,
    w2_buf,
    s2_buf,
    act,
    y,
    accum,
    selected,
    active_vec,
    active_smem,
    dma,
    send,
    recv,
    send_sem,
    recv_sem,
    owner,
    fan_recv,
    fan_send_sem,
    fan_recv_sem,
    *prefetch_scratch,
):
    ep, nc = 256 // w1.shape[0], 1024
    block = x.shape[1] // s1.shape[1]
    barrier = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier,
            device_id={"model": jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )
    b, h = x.shape
    w1 = w1.bitcast(jnp.uint32)
    w2 = w2.bitcast(jnp.uint32)
    local, _, i2 = w1.shape
    i = i2 // 2
    weights = route_weights(logits[...], bias[...])
    ids = lax.broadcasted_iota(jnp.int32, weights.shape, 1)
    selected[...] = weights.astype(jnp.bfloat16).astype(jnp.float32)
    active = jnp.max((weights > 0).astype(jnp.int32), axis=0, keepdims=True)
    active_vec[...] = jnp.broadcast_to(active, active_vec.shape)
    cp = pltpu.make_async_copy(active_vec, active_smem, dma.at[0])
    cp.start()
    cp.wait()
    accum[...] = jnp.zeros(accum.shape, jnp.float32)
    first = rank[0] * local

    def start_gate_up(e):
        pltpu.make_async_copy(w1.at[e], w1_buf, dma.at[0]).start()
        pltpu.make_async_copy(s1.at[e], s1_buf, dma.at[1]).start()

    if prefetch_scratch:
        (next_expert,) = prefetch_scratch
        next_expert[local] = -1

        @pl.loop(0, local)
        def find_next(index):
            e = local - 1 - index
            next_expert[e] = jnp.where(
                active_smem[0, first + e] != 0, e, next_expert[e + 1]
            )

        @pl.when(next_expert[0] >= 0)
        def first_load():
            start_gate_up(next_expert[0])

    @pl.loop(0, local)
    def expert(e):
        @pl.when(active_smem[0, first + e] != 0)
        def compute():
            ops = [
                pltpu.make_async_copy(src.at[e], dst, dma.at[n])
                for n, (src, dst) in enumerate(
                    ((w1, w1_buf), (s1, s1_buf), (w2, w2_buf), (s2, s2_buf))
                )
            ]
            for op in ops[2:] if prefetch_scratch else ops:
                op.start()
            for op in ops[:2]:
                op.wait()

            @pl.loop(0, i // nc)
            def up(j):
                col = pl.multiple_of(j * nc, nc)
                g, u = _fp4_k256_dual_gmm(
                    x, w1_buf, s1_buf, h, col, i + col, nc, block == 256, block
                )
                act[:, pl.ds(col, nc)] = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)

            if prefetch_scratch:

                @pl.when(next_expert[e + 1] >= 0)
                def load_next():
                    # The previous gate/up dot no longer owns these two VMEM
                    # buffers. Transfer the next active expert during W2.
                    start_gate_up(next_expert[e + 1])

            for op in ops[2:]:
                op.wait()

            @pl.loop(0, h // nc)
            def down(j):
                col = pl.multiple_of(j * nc, nc)
                d = _fp4_k256_gmm(
                    act, w2_buf, s2_buf, i, col, nc, block == 256, block
                ).astype(jnp.bfloat16)
                y[:, pl.ds(col, nc)] = d * (
                    norm[:, pl.ds(col, nc)].astype(jnp.float32) + 1.0
                ).astype(jnp.bfloat16)

            yf = y[...].astype(jnp.float32)
            rms = lax.rsqrt(
                jnp.sum(yf * yf, axis=1, keepdims=True) * (1.0 / h) + 1e-8
            ).astype(jnp.bfloat16)
            wy = (y[...] * rms).astype(jnp.float32)
            weight = jnp.sum(
                jnp.where(ids == first + e, selected[...], 0.0), axis=1, keepdims=True
            )
            accum[...] += wy * weight

    rows = b // ep
    me = rank[0]
    for dest in range(ep):
        part = accum[pl.ds(dest * rows, rows), :]
        send[dest] = part.reshape(send.shape[1:])
    pl.semaphore_wait(barrier, ep)
    copies = [
        pltpu.make_async_remote_copy(
            send.at[d],
            recv.at[me],
            send_sem.at[d],
            recv_sem,
            device_id={"model": jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        for d in range(ep)
    ]
    for op in copies:
        op.start()
    pltpu.make_async_copy(recv, recv, recv_sem).wait()
    total = recv[0]
    for src in range(1, ep):
        total += recv[src]
    # Each owner contributes its rounded token rows to a private slice of
    # each peer's buffer. The entry barrier also protects this second
    # receive area. Only the completed phase-one local sum can be sent.
    owner[...] = total.astype(jnp.bfloat16)
    fan_copies = [
        pltpu.make_async_remote_copy(
            owner,
            fan_recv.at[me],
            fan_send_sem.at[dest],
            fan_recv_sem,
            device_id={"model": jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        for dest in range(ep)
    ]
    for copy in fan_copies:
        copy.start()
    pltpu.make_async_copy(fan_recv, fan_recv, fan_recv_sem).wait()
    output[...] = fan_recv[...].reshape(output.shape)
    for copy in fan_copies:
        copy.wait_send()
    for op in copies:
        op.wait_send()


def fused_decode_from_logits(rank, logits, bias, x, w1, s1, w2, s2, norm, *, ep=16):
    """Return replicated BF16 [B,4096]. All FP4 expert tensors are the serving buffers."""
    b, h = x.shape
    if b not in (16, 32) or h != 4096 or logits.shape != (b, 256):
        raise ValueError(
            "Muse fused decode requires B16/B32, 4096 expert channels and 256 logits"
        )
    if ep not in (8, 16):
        raise ValueError("Muse fused decode supports EP8 and EP16")
    i, n = 4096, 8192
    local, block = 256 // ep, 16 if ep == 8 else 256
    if (
        w1.shape != (local, h, n)
        or w2.shape != (local, i, h)
        or w1.dtype != jnp.float4_e2m1fn
        or w2.dtype != jnp.float4_e2m1fn
        or s1.shape != (local, h // block, n)
        or s2.shape != (local, i // block, h)
    ):
        raise ValueError(
            f"EP{ep} fused decode requires native FP4 whole experts with K{block} scales"
        )
    packet = (8, b // ep * h // 8)
    wire = (ep, *packet)
    vmem = pltpu.VMEM
    v = pl.BlockSpec(memory_space=vmem)
    hb = pl.BlockSpec(memory_space=pltpu.HBM)
    sem = pltpu.SemaphoreType.DMA
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((b, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(v, v, v, hb, hb, hb, hb, v),
            out_specs=v,
            scratch_shapes=(
                vmem((h // 8, n), jnp.uint32),
                vmem(s1.shape[1:], s1.dtype),
                vmem((i // 8, h), jnp.uint32),
                vmem(s2.shape[1:], s2.dtype),
                vmem((b, i), jnp.bfloat16),
                vmem((b, h), jnp.bfloat16),
                vmem((b, h), jnp.float32),
                vmem((b, 256), jnp.float32),
                vmem((8, 256), jnp.int32),
                pltpu.SMEM((8, 256), jnp.int32),
                sem((4,)),
                vmem(wire, jnp.float32),
                vmem(wire, jnp.float32),
                sem((ep,)),
                sem,
                vmem(packet, jnp.bfloat16),
                vmem(wire, jnp.bfloat16),
                sem((ep,)),
                sem,
            )
            + ((pltpu.SMEM((local + 1,), jnp.int32),) if ep == 8 else ()),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2, collective_id=0
        ),
        name=(
            f"muse_from_logits_ep8_b{b}_k16_rs_nc1024_pre_fan"
            if ep == 8
            else f"muse_from_logits_ep16_b{b}_k256_rs_nc1024_fan"
        ),
    )(rank, logits, bias, x, w1, s1, w2, s2, norm)
