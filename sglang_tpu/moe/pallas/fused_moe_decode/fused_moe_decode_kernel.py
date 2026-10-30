# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode kernel: routed experts, shared expert and all-reduce in one call.

Ported from vllm-torchtpu (`vllm_torchtpu/kernels/experimental/fused_moe_decode/fused_moe_decode_kernel.py`)
with extensions for `sglang-torchtpu`:
  - Optional `swiglu_limit` clamping on both routed and shared SwiGLU activations
    (used by GLM-5.3-Flash / DeepSeek-V2.5 / Glm5Next where `swiglu_limit = 10.0`).
  - Safe handling of negative `topk_ids` (`-1` padding) prior to subtracting
    `expert_start`.
  - Support for both shared+routed (`has_shared=True`) and routed-only
    (`has_shared=False`) MoE layers, and optional in-kernel all-reduce
    (`reduce_output=True/False`).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Output columns per second-matmul step.
_N_CHUNK = 512
# Upper bound on the weight ring depth.
_NUM_BUFFERS = 6
# Columns per FP8 activation scale.
_Q_BLOCK = 512


def _kernel(
    active_ref,
    rank_ref,
    x_ref,
    coeff_ref,
    shared_gate_ref,
    w1_hbm,
    w2_hbm,
    w1_scale_hbm,
    w2_scale_hbm,
    shared_w1_hbm,
    shared_w1_scale_hbm,
    shared_w2_hbm,
    shared_w2_scale_hbm,
    out_ref,
    w1_ring,
    w2_ring,
    w1_scale_ring,
    w2_scale_ring,
    shared_w1_vmem,
    shared_w1_scale_vmem,
    shared_w2_vmem,
    shared_w2_scale_vmem,
    xq_ref,
    xs_ref,
    acc_ref,
    acc_bf16,
    recv,
    red,
    ids_ref,
    sem_ref,
    shared_sem,
    a2a_send,
    a2a_recv,
    ag_send,
    ag_recv,
    *,
    m,
    h,
    i,
    experts,
    nb,
    n_ranks,
    weight_dtype,
    rows=None,
    ib=None,
    swiglu_limit=None,
    has_shared=True,
    reduce_output=True,
):
    """One rank's MoE block: shared + routed experts + all-reduce in one Pallas call."""
    nblk = 1 if ib is None else i // ib
    me = rank_ref[0]
    qmax = float(jnp.finfo(weight_dtype).max)
    chunk = h // n_ranks
    if reduce_output and n_ranks > 1:
        barrier = pltpu.get_barrier_semaphore()
        for d in range(n_ranks):
            pl.semaphore_signal(
                barrier,
                inc=1,
                device_id=(jnp.int32(d),),
                device_id_type=pl.DeviceIdType.MESH,
            )

    if has_shared:
        shared_copies = [
            pltpu.make_async_copy(shared_w1_hbm, shared_w1_vmem, shared_sem.at[0]),
            pltpu.make_async_copy(shared_w2_hbm, shared_w2_vmem, shared_sem.at[1]),
            pltpu.make_async_copy(
                shared_w1_scale_hbm, shared_w1_scale_vmem, shared_sem.at[2]
            ),
            pltpu.make_async_copy(
                shared_w2_scale_hbm, shared_w2_scale_vmem, shared_sem.at[3]
            ),
        ]
        for copy in shared_copies:
            copy.start()

    acc_ref[...] = jnp.zeros((m, h), jnp.float32)
    for k in range(0, h, _Q_BLOCK):
        x = x_ref[:, k : k + _Q_BLOCK].astype(jnp.float32)
        xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / qmax
        xq_ref[:, k : k + _Q_BLOCK] = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(
            weight_dtype
        )
        xs_ref[k // _Q_BLOCK] = jnp.broadcast_to(xsc, (m, 128))
    cols = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))
    units = count if ib is None else count * nblk

    def transfers(step):
        if ib is not None:
            return block_transfers(step)
        e, slot = ids_ref[step], step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ring.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ring.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(
                w1_scale_hbm.at[e, 0], w1_scale_ring.at[slot], sem_ref.at[slot, 2]
            ),
            pltpu.make_async_copy(
                w2_scale_hbm.at[e, 0], w2_scale_ring.at[slot], sem_ref.at[slot, 3]
            ),
        ]

    def block_transfers(step):
        assert ib is not None
        e, slot = ids_ref[step // nblk], step % nb
        gate = pl.multiple_of((step % nblk) * ib, ib)
        up = pl.multiple_of(i + (step % nblk) * ib, ib)
        copies = []
        for dst, src in ((0, gate), (ib, up)):
            copies.append(
                pltpu.make_async_copy(
                    w1_hbm.at[e, :, pl.ds(src, ib)],
                    w1_ring.at[slot, :, pl.ds(dst, ib)],
                    sem_ref.at[slot, len(copies)],
                )
            )
            copies.append(
                pltpu.make_async_copy(
                    w1_scale_hbm.at[e, 0, :, pl.ds(src, ib)],
                    w1_scale_ring.at[slot, :, pl.ds(dst, ib)],
                    sem_ref.at[slot, len(copies)],
                )
            )
        copies.append(
            pltpu.make_async_copy(
                w2_hbm.at[e, pl.ds(gate, ib)], w2_ring.at[slot], sem_ref.at[slot, 4]
            )
        )
        copies.append(
            pltpu.make_async_copy(
                w2_scale_hbm.at[e, 0], w2_scale_ring.at[slot], sem_ref.at[slot, 5]
            )
        )
        return copies

    def first_matmul(w1):
        acc = None
        for k in range(0, h, _Q_BLOCK):
            part = jnp.matmul(
                xq_ref[:, k : k + _Q_BLOCK],
                w1[k : k + _Q_BLOCK, :],
                preferred_element_type=jnp.float32,
            )
            xs = xs_ref[k // _Q_BLOCK]
            part = part * jnp.concatenate([xs] * (part.shape[1] // 128), axis=1)
            acc = part if acc is None else acc + part
        return acc

    def accumulate(acc1, coeff, w1_scale, w2, w2_scale):
        width = acc1.shape[1] // 2
        acc1 = acc1 * w1_scale
        gate = acc1[:, :width]
        up = acc1[:, width:]
        if swiglu_limit is not None:
            gate = jnp.minimum(gate, swiglu_limit)
            up = jnp.clip(up, -swiglu_limit, swiglu_limit)
        mid = jnp.where(coeff != 0, jax.nn.silu(gate) * up * coeff, 0.0)
        qa = min(_Q_BLOCK, width)
        blocks = []
        for k in range(0, width, qa):
            blk = mid[:, k : k + qa]
            ms = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / qmax
            blocks.append(
                ((blk * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(weight_dtype), ms)
            )
        for n in range(0, h, _N_CHUNK):
            acc2 = None
            for b, (mq, ms) in enumerate(blocks):
                part = jnp.matmul(
                    mq,
                    w2[b * qa : (b + 1) * qa, n : n + _N_CHUNK],
                    preferred_element_type=jnp.float32,
                )
                part = part * ms
                acc2 = part if acc2 is None else acc2 + part
            acc_ref[:, n : n + _N_CHUNK] += acc2 * w2_scale[:, n : n + _N_CHUNK]

    @pl.when(count > 0)
    def prime_routed():
        for initial in range(nb):

            @pl.when(units > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

    if has_shared:
        for copy in shared_copies:
            copy.wait()
        shared_coeff = jnp.max(shared_gate_ref[...], axis=1, keepdims=True)
        accumulate(
            first_matmul(shared_w1_vmem),
            shared_coeff,
            shared_w1_scale_vmem[...],
            shared_w2_vmem,
            shared_w2_scale_vmem[...],
        )

    @pl.when(count > 0)
    def run_routed():
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < units)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            acc1_next = first_matmul(w1_ring.at[nxt % nb])
            slot = step % nb
            e = ids_ref[step] if ib is None else ids_ref[step // nblk]
            coeff = jnp.sum(
                jnp.where(cols == e, coeff_ref[...], 0.0), axis=1, keepdims=True
            )
            accumulate(
                acc1, coeff, w1_scale_ring[slot], w2_ring.at[slot], w2_scale_ring[slot]
            )

            @pl.when(step + nb < units)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(
            0,
            units,
            expert_body,
            first_matmul(w1_ring.at[0]),
        )

    if not reduce_output or n_ranks <= 1:
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)
        return

    def all_to_all_sum(send_slice, dst):
        pl.semaphore_wait(barrier, n_ranks)
        copies = [
            pltpu.make_async_remote_copy(
                send_slice(d),
                recv.at[me],
                a2a_send.at[d],
                a2a_recv,
                device_id=(jnp.int32(d),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for d in range(n_ranks)
        ]
        for copy in copies:
            copy.start()
        pltpu.make_async_copy(recv, recv, a2a_recv).wait()
        acc = recv[0].astype(jnp.float32)
        for s in range(1, n_ranks):
            acc = acc + recv[s].astype(jnp.float32)
        dst[...] = acc.astype(dst.dtype)
        for copy in copies:
            copy.wait_send()

    if rows is not None:
        for d in range(n_ranks):
            acc_bf16[d, :rows, :] = acc_ref[d * rows : (d + 1) * rows, :].astype(
                acc_bf16.dtype
            )
        all_to_all_sum(lambda d: acc_bf16.at[d], out_ref)
        return

    acc_bf16[...] = acc_ref[...].astype(acc_bf16.dtype)
    all_to_all_sum(lambda d: acc_bf16.at[:, pl.ds(d * chunk, chunk)], red)
    col = pl.multiple_of(me * chunk, chunk)
    gather = [
        pltpu.make_async_remote_copy(
            red,
            out_ref.at[:, pl.ds(col, chunk)],
            ag_send.at[d],
            ag_recv,
            device_id=(jnp.int32(d),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for d in range(n_ranks)
    ]
    for copy in gather:
        copy.start()
    pltpu.make_async_copy(out_ref, out_ref, ag_recv).wait()
    for copy in gather:
        copy.wait_send()


def _weight_ring_depth(h, i, weight_dtype):
    slot = 3 * h * i * jnp.dtype(weight_dtype).itemsize
    fit = pltpu.get_tpu_info().vmem_capacity_bytes // 2 // slot
    return max(3, min(_NUM_BUFFERS, fit))


_I_BLOCKS = (256, 128)
_VMEM_RESERVE = 8 << 20


def _unit_bytes(h, width, wt):
    return 3 * h * width * wt + (2 * width + h) * 4


def _token_side_bytes(m, h, i_s, experts, n_ranks, wt):
    f32, bf16 = 4, 2
    shared = _unit_bytes(h, i_s, wt) if i_s > 0 else 0
    rows = m * (
        h * bf16
        + (experts + (-experts) % 128) * f32
        + 128 * f32
        + h * bf16
        + h * wt
        + (h // _Q_BLOCK) * 128 * f32
        + h * f32
        + h * bf16
        + n_ranks * (h // n_ranks) * bf16
        + (h // n_ranks) * bf16
    )
    return shared + rows


def plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype, i_block=None):
    wt = jnp.dtype(weight_dtype).itemsize
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    if i_block is None:
        widths = [w for w in _I_BLOCKS if w < i and i % w == 0]
        if 3 * _unit_bytes(h, i, wt) <= capacity or not widths:
            return None, _weight_ring_depth(h, i, weight_dtype)
    elif i % i_block or i_block % 128:
        raise ValueError(f"i_block {i_block} must divide i={i} in lanes of 128")
    else:
        widths = [i_block]

    def fit(w):
        live = 2 * m * 2 * w * 4
        free = (
            capacity
            - _VMEM_RESERVE
            - live
            - _token_side_bytes(m, h, i_s, experts, n_ranks, wt)
        )
        return free // _unit_bytes(h, w, wt)

    ib = next((w for w in widths if fit(w) >= 3), widths[-1])
    return ib, max(3, min(_NUM_BUFFERS, fit(ib)))


def vmem_bytes(tokens, h, i, i_s, experts, n_ranks, weight_dtype):
    m = tokens + (-tokens) % 16
    wt = jnp.dtype(weight_dtype).itemsize
    ib, nb = plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype)
    return nb * _unit_bytes(h, i if ib is None else ib, wt) + _token_side_bytes(
        m, h, i_s, experts, n_ranks, wt
    )


def fused_moe_decode(
    rank,
    hidden,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    topk_weights,
    topk_ids,
    *,
    n_ranks,
    expert_start=None,
    scatter_rows=False,
    routed_scale=1.0,
    swiglu_limit=None,
    has_shared=True,
    reduce_output=True,
    i_block=None,
):
    """Per-rank body; call inside shard_map over the TP/EP mesh."""
    tokens, h = hidden.shape
    if h % _Q_BLOCK:
        raise ValueError(f"hidden size {h} must be a multiple of {_Q_BLOCK}")
    if scatter_rows and tokens % n_ranks:
        raise ValueError(f"scatter_rows needs tokens % n_ranks == 0; got {tokens}")
    rows = tokens // n_ranks if scatter_rows else None
    experts, i, _ = w2.shape
    pad = (-tokens) % 16
    valid_k = topk_ids >= 0
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)), constant_values=-1)
        valid_k = jnp.pad(valid_k, ((0, pad), (0, 0)), constant_values=False)
    m = tokens + pad
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    if routed_scale != 1.0:
        weights = weights * jnp.float32(routed_scale)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(topk_ids.shape[1]):
        selected = (topk_ids[:, route : route + 1] == expert_ids) & valid_k[
            :, route : route + 1
        ]
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    if not has_shared or shared_gate_weight is None:
        shared_gate = jnp.ones((m, 128), jnp.float32)
    else:
        shared_gate = jax.nn.sigmoid(
            jnp.dot(
                hidden,
                shared_gate_weight.astype(hidden.dtype),
                preferred_element_type=jnp.float32,
            )
        )
        shared_gate = jnp.broadcast_to(shared_gate, (m, 128))
    weight_dtype = w1.dtype
    if has_shared and shared_w1 is not None and shared_w2 is not None:
        i_s = shared_w2.shape[0]
        sw1_in = shared_w1
        sw1s_in = shared_w1_scale.reshape(1, 2 * i_s).astype(jnp.float32)
        sw2_in = shared_w2
        sw2s_in = shared_w2_scale.reshape(1, h).astype(jnp.float32)
    else:
        has_shared = False
        i_s = 128
        sw1_in = jnp.zeros((h, 2 * i_s), dtype=weight_dtype)
        sw1s_in = jnp.zeros((1, 2 * i_s), dtype=jnp.float32)
        sw2_in = jnp.zeros((i_s, h), dtype=weight_dtype)
        sw2s_in = jnp.zeros((1, h), dtype=jnp.float32)

    ib, nb = plan_weight_ring(
        m, h, i, i_s if has_shared else 0, experts, n_ranks, weight_dtype, i_block
    )
    w = i if ib is None else ib
    chunk = h // max(1, n_ranks)
    if rows is None:
        out_rows, wire = m, hidden.dtype
        acc_bf16_shape, recv_shape, red_shape = (
            (m, h),
            (max(1, n_ranks), m, chunk),
            (m, chunk),
        )
    else:
        wire = hidden.dtype if rows % 16 == 0 else jnp.float32
        out_rows = rows if rows % 16 == 0 else -(-rows // 8) * 8
        acc_bf16_shape = recv_shape = (max(1, n_ranks), out_rows, h)
        red_shape = (8, 128)
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            nb=nb,
            n_ranks=n_ranks,
            weight_dtype=weight_dtype,
            rows=rows,
            ib=ib,
            swiglu_limit=swiglu_limit,
            has_shared=has_shared,
            reduce_output=reduce_output,
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows, h), wire),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * w), weight_dtype),
                pltpu.VMEM((nb, w, h), weight_dtype),
                pltpu.VMEM((nb, 1, 2 * w), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.VMEM((h, 2 * i_s), weight_dtype),
                pltpu.VMEM((1, 2 * i_s), jnp.float32),
                pltpu.VMEM((i_s, h), weight_dtype),
                pltpu.VMEM((1, h), jnp.float32),
                pltpu.VMEM((m, h), weight_dtype),
                pltpu.VMEM((h // _Q_BLOCK, m, 128), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.VMEM(acc_bf16_shape, wire),
                pltpu.VMEM(recv_shape, wire),
                pltpu.VMEM(red_shape, wire),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((nb, 4 if ib is None else 6)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((max(1, n_ranks),)),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA((max(1, n_ranks),)),
                pltpu.SemaphoreType.DMA(()),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0 if (reduce_output and n_ranks > 1) else None,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"fused_moe_decode{'_rs' if rows else ''}-e_{experts}-m_{m}-h_{h}"
        f"-i_{i}-is_{i_s}-n_{n_ranks}-buffers_{nb}{f'-ib_{ib}' if ib else ''}",
    )(
        active,
        rank.reshape(1).astype(jnp.int32),
        hidden,
        coeff,
        shared_gate,
        w1,
        w2,
        w1_scale,
        w2_scale,
        sw1_in,
        sw1s_in,
        sw2_in,
        sw2s_in,
    )
    if rows is not None:
        return out[:rows].astype(hidden.dtype)
    return out[:tokens] if pad else out
