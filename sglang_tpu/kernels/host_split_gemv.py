# SPDX-License-Identifier: Apache-2.0
"""Host-split single-token decode GEMV (Pallas) for Kimi-K3 on TPU.

Under TP32 the MoE latent down-proj, router gate and latent up-proj are
replicated: every rank streams the whole weight each decode step. Here each
rank of a host computes only its slice of the output columns and the ranks
of the host exchange their slices, so each rank reads 1/7 or 1/8 of the
weight. Every output column is still computed by one rank from the full
input with the replicated weight's fp8 values and scales.

One kernel per projection streams this rank's fp8 [K, N] slice in K blocks
like kernels/static_fp8_gemv.py (and, for the MoE front, its bf16
router-gate slice), signals the host barrier on the first block so the handshake
overlaps the weight stream, and after the last block sends the gathered
columns to the other ranks of the host in one DMA each (dense (rows, 128)
tiles in one VMEM buffer) and concatenates what it receives.

Outputs: the gathered columns ``[1, used * cols]`` (the first ``cols`` GEMV
columns of the first ``used`` ranks of the host), the gathered router logits
``[1, used_e * 128]`` when a router slice is given, and this rank's remaining
GEMV columns ``[1, n - cols]`` (the shared experts' gate_up) when there are
any. The Kimi-K3 static fp8 binder (sglang_tpu/kimi_k3/static_fp8.py) builds
the ops and binds the column slices.
"""

import functools

# One fp8 sublane tile, as in static_fp8_gemv.py; the row is zero-padded.
_ROWS = 32
_FP8_MAX = 448.0
_ALIGN_ROWS = 16
_VMEM_LIMIT = 40 * 1024 * 1024
# Measured best for the Kimi-K3 TP32 GEMV shapes (static_fp8_gemv.py).
_MAX_BLOCK_K = 1792
# Barrier namespace: 0 fused GMM reduce-scatter, 1 two-level all-reduce.
_COLLECTIVE_ID = 2


def block_k(k: int) -> int:
    for bk in range(min(_MAX_BLOCK_K, k) // 128 * 128, 0, -128):
        if k % bk == 0:
            return bk
    return 0


def _vmem_limit(bk, n, cols, used, used_e, has_router, total, local, w_bytes, extra=0):
    """Scoped VMEM from the block and scratch shapes (vmem_limit.py): the
    double-buffered x / weight / router blocks, the outputs, the
    accumulators, the exchange buffers and the f32 activation / dot
    temporaries."""
    from sglang_tpu.kernels.vmem_limit import padded_bytes, scoped_limit

    need = (
        2 * padded_bytes((1, bk), 2)
        + 2 * padded_bytes((bk, n), w_bytes)
        + 2 * padded_bytes((n,), 4)
        + 2 * padded_bytes((1, used * cols), 2)
        + 2 * padded_bytes((1, n), 2)
        + padded_bytes((_ROWS, n), 4)
        + padded_bytes((_ROWS, 128), 4)
        + padded_bytes((total, 128), 2)
        + padded_bytes((local, total, 128), 2)
        + 2 * padded_bytes((_ROWS, bk), 4)
        + padded_bytes((_ROWS, n), 4)
    )
    if has_router:
        need += 2 * padded_bytes((bk, 128), 2) + 2 * padded_bytes((1, used_e * 128), 2)
    return scoped_limit(_VMEM_LIMIT, need + extra)


def _kernel(
    *refs,
    nk,
    local,
    cols,
    used,
    has_router,
    used_e,
    has_rest,
    n_prefetch=0,
    tail=None,
):
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    refs = list(refs)
    inv_ref, x_ref, w_ref, s_ref = refs[:4]
    del refs[:4]
    r_ref = refs.pop(0) if has_router else None
    if tail is not None:
        # The tail inputs stay where XLA put them (pl.ANY); the kernel copies
        # them into VMEM scratch behind the GEMV instead of before step 0.
        n_t = 4 + tail["has_prefix"]
        t_src = refs[:n_t]
        del refs[:n_t]
    pf_in = [refs.pop(0) for _ in range(n_prefetch)]
    gat_out = refs.pop(0)
    logit_out = refs.pop(0) if has_router else None
    rest_out = refs.pop(0) if has_rest else None
    normed_out = refs.pop(0) if tail is not None else None
    pf_out = [refs.pop(0) for _ in range(n_prefetch)]
    if tail is not None:
        t_sem = refs.pop()
        t_buf = refs[-n_t:]
        del refs[-n_t:]
        t_copies = [
            pltpu.make_async_copy(src, dst, t_sem.at[i])
            for i, (src, dst) in enumerate(zip(t_src, t_buf))
        ]
        sh_ref = t_buf[0]
        pre_ref = t_buf[1] if tail["has_prefix"] else None
        bank_ref, cw_ref, nw_ref = t_buf[-3:]
    pf_sem = refs.pop() if n_prefetch else None
    acc, racc, packed, gather, send_sem, recv_sem = refs

    k = pl.program_id(0)
    me = lax.axis_index("model")
    grp, loc = me // local, lax.rem(me, local)
    peers = [grp * local + lax.rem(loc + d, local) for d in range(1, local)]
    mesh_id = pl.DeviceIdType.MESH
    barrier = pltpu.get_barrier_semaphore()

    @pl.when(k == 0)
    def _():
        acc[...] = jnp.zeros_like(acc)
        racc[...] = jnp.zeros_like(racc)
        for p in peers:
            pl.semaphore_signal(barrier, 1, device_id=(p,), device_id_type=mesh_id)

    if tail is not None:

        @pl.when(k == 0)
        def _():
            for c in t_copies:
                c.start()

    if n_prefetch:
        # Copy the next GEMV's weight HBM -> VMEM while this kernel computes
        # and waits for the host exchange (its HBM traffic is idle then).
        pf_copies = [
            pltpu.make_async_copy(src, dst, pf_sem.at[i])
            for i, (src, dst) in enumerate(zip(pf_in, pf_out))
        ]

        @pl.when(k == 0)
        def _():
            for c in pf_copies:
                c.start()

    x = x_ref[...]
    x = jnp.concatenate([x, jnp.zeros((_ROWS - x.shape[0], x.shape[1]), x.dtype)], 0)
    xq = jnp.clip(x.astype(jnp.float32) * inv_ref[0], -_FP8_MAX, _FP8_MAX)
    acc[...] += jnp.dot(
        xq.astype(w_ref.dtype), w_ref[...], preferred_element_type=jnp.float32
    )
    if has_router:
        racc[...] += jnp.dot(x, r_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == nk - 1)
    def _():
        y = (acc[0:1, :] * s_ref[...].reshape(1, -1)).astype(gat_out.dtype)
        rows = cols // 128
        # The alignment gap rows are sent but never read back, so they are
        # left unwritten.
        packed[0:rows, :] = y[:, :cols].reshape(rows, 128)
        if has_router:
            off = -(-rows // _ALIGN_ROWS) * _ALIGN_ROWS
            packed[off : off + 1, :] = racc[0:1, :].astype(packed.dtype)
        if has_rest:
            rest_out[...] = y[:, cols:]
        pl.semaphore_wait(barrier, len(peers))
        copies = []
        for p in peers:
            c = pltpu.make_async_remote_copy(
                packed,
                gather.at[loc],
                send_sem,
                recv_sem,
                device_id=(p,),
                device_id_type=mesh_id,
            )
            c.start()
            copies.append(c)
        gather[loc] = packed[...]
        if tail is not None:
            # The bank half of the next layer's aggregation does not need
            # this GEMV: it runs while the exchange DMAs are in flight.
            from sglang_tpu.kimi_attn_residual import bank_scores

            for c in t_copies:
                c.wait()
            cw = cw_ref[...]
            bank_vals, bank_sc = bank_scores(
                bank_ref, cw, nvb=tail["nvb"], eps=tail["eps"]
            )
        for c in copies:
            c.wait_send()
        for c in copies:
            c.wait_recv()
        y_all = jnp.concatenate(
            [gather[j, 0:rows, :] for j in range(used)], axis=0
        ).reshape(gat_out.shape)
        if tail is None:
            gat_out[...] = y_all
        else:
            # Kimi-K3 MoE tail and the next layer's attention-side
            # aggregation: out = (up + shared) [+ prefix] with the two bf16
            # roundings of the torch adds, then the attention-residual decode
            # math on out (kimi_attn_residual: same ops, same order).
            from sglang_tpu.kimi_attn_residual import add_prefix, decode_finish

            o = add_prefix(y_all, sh_ref[...])
            if pre_ref is not None:
                o = add_prefix(o, pre_ref[...])
            gat_out[...] = o
            normed_out[...] = decode_finish(
                o,
                bank_vals,
                bank_sc,
                cw,
                nw_ref[...],
                eps=tail["eps"],
                out_eps=tail["out_eps"],
                out_dtype=normed_out.dtype,
            )
        if has_router:
            logit_out[...] = jnp.concatenate(
                [gather[j, off : off + 1, :] for j in range(used_e)], axis=0
            ).reshape(logit_out.shape)
        for c in pf_copies if n_prefetch else ():
            c.wait()


def host_split_gemv_local(
    x,
    w,
    out_scale,
    inv_x,
    router=None,
    *,
    local,
    cols,
    used,
    used_e,
    prefetch=None,
    tail=None,
):
    """Per-rank body inside ``shard_map`` over the host-major "model" axis.
    x [1, K] bf16; w fp8 [K, N]; out_scale f32 [N]; inv_x f32 [1];
    router bf16 [K, 128] or None. ``prefetch``: an array (kept in HBM) the
    kernel also copies whole into a VMEM output, returned last, so a later
    kernel reads it from VMEM without waiting for a copy (the copy overlaps
    this kernel's exchange). ``tail``: (shared,
    prefix or None, bank, cw, norm_w, nvb, eps, out_eps) -- the gathered
    output becomes (up + shared) [+ prefix] (the Kimi-K3 MoE tail add) and
    the kernel also returns the next layer's attention-residual decode
    aggregation of it (after the GEMV outputs, before the prefetch output)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    kdim, n = w.shape
    bk = block_k(kdim)
    nk = kdim // bk
    has_router = router is not None
    has_rest = n > cols
    rows = cols // 128
    total = -(-rows // _ALIGN_ROWS) * _ALIGN_ROWS + (_ALIGN_ROWS if has_router else 0)
    const = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec((1, bk), lambda i: (0, i)),
        pl.BlockSpec((bk, n), lambda i: (i, 0)),
        # 1-D out_scale: a [1, N] reshape outside the kernel is a relayout per call.
        const((n,)),
    ]
    args = [
        inv_x.astype(jnp.float32),
        x,
        w,
        out_scale.reshape(n).astype(jnp.float32),
    ]
    out_shape = [jax.ShapeDtypeStruct((1, used * cols), x.dtype)]
    if has_router:
        in_specs.append(pl.BlockSpec((bk, 128), lambda i: (i, 0)))
        args.append(router)
        out_shape.append(jax.ShapeDtypeStruct((1, used_e * 128), x.dtype))
    if has_rest:
        out_shape.append(jax.ShapeDtypeStruct((1, n - cols), x.dtype))
    tail_cfg = None
    tail_bytes = 0
    if tail is not None:
        from sglang_tpu.kernels.vmem_limit import padded_bytes

        shared, prefix, bank, cw, norm_w, nvb, eps, out_eps = tail
        hidden = used * cols
        tail_cfg = dict(
            has_prefix=prefix is not None,
            nvb=int(nvb),
            eps=float(eps),
            out_eps=float(out_eps),
        )
        t_in = [shared.reshape(1, hidden)]
        if prefix is not None:
            t_in.append(prefix.reshape(1, hidden))
        t_in += [
            bank,
            cw.reshape(1, hidden).astype(jnp.float32),
            norm_w.reshape(1, hidden).astype(jnp.float32),
        ]
        in_specs += [pl.BlockSpec(memory_space=pl.ANY) for _ in t_in]
        args += t_in
        out_shape.append(jax.ShapeDtypeStruct((1, hidden), x.dtype))
        # Double-buffered tail inputs, the extra output and the f32
        # temporaries of the aggregation (nvb + 1 rows and a few more).
        tail_bytes = 2 * sum(padded_bytes(t.shape, t.dtype.itemsize) for t in t_in)
        tail_bytes += 2 * padded_bytes((1, hidden), 2)
        tail_bytes += (int(nvb) + 6) * padded_bytes((1, hidden), 4)
    out_specs = [const(o.shape) for o in out_shape]
    pfs = () if prefetch is None else (prefetch,)
    for pf in pfs:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
        args.append(pltpu.with_memory_space_constraint(pf, pltpu.HBM))
        out_shape.append(pltpu.VMEM(pf.shape, pf.dtype))
        out_specs.append(pl.BlockSpec(memory_space=pltpu.VMEM))
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            nk=nk,
            local=local,
            cols=cols,
            used=used,
            has_router=has_router,
            used_e=used_e,
            has_rest=has_rest,
            n_prefetch=len(pfs),
            tail=tail_cfg,
        ),
        out_shape=tuple(out_shape),
        grid=(nk,),
        in_specs=in_specs,
        out_specs=tuple(out_specs),
        scratch_shapes=[
            pltpu.VMEM((_ROWS, n), jnp.float32),
            pltpu.VMEM((_ROWS, 128), jnp.float32),
            pltpu.VMEM((total, 128), x.dtype),
            pltpu.VMEM((local, total, 128), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ]
        + ([pltpu.SemaphoreType.DMA((len(pfs),))] if pfs else [])
        + (
            [pltpu.VMEM(t.shape, t.dtype) for t in t_in]
            + [pltpu.SemaphoreType.DMA((len(t_in),))]
            if tail is not None
            else []
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_vmem_limit(
                bk,
                n,
                cols,
                used,
                used_e,
                has_router,
                total,
                local,
                w.dtype.itemsize,
                extra=tail_bytes,
            ),
            collective_id=_COLLECTIVE_ID,
        ),
        name=f"host_split_gemv_k{kdim}_n{n}_c{cols}_u{used}"
        + (f"_r{used_e}" if has_router else "")
        + ("_pf" if pfs else "")
        + (f"_tail{int(tail[5])}" if tail is not None else ""),
    )(*args)
    return tuple(outs)
