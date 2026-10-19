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
any. The Kimi-K3 static fp8 binder (sglang_tpu/models/kimi_k3/static_fp8.py) builds
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


def _kernel(*refs, nk, local, cols, used, has_router, used_e, has_rest):
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    refs = list(refs)
    inv_ref, x_ref, w_ref, s_ref = refs[:4]
    del refs[:4]
    r_ref = refs.pop(0) if has_router else None
    gat_out = refs.pop(0)
    logit_out = refs.pop(0) if has_router else None
    rest_out = refs.pop(0) if has_rest else None
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
        y = (acc[0:1, :] * s_ref[...]).astype(gat_out.dtype)
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
        for c in copies:
            c.wait_send()
        for c in copies:
            c.wait_recv()
        gat_out[...] = jnp.concatenate(
            [gather[j, 0:rows, :] for j in range(used)], axis=0
        ).reshape(gat_out.shape)
        if has_router:
            logit_out[...] = jnp.concatenate(
                [gather[j, off : off + 1, :] for j in range(used_e)], axis=0
            ).reshape(logit_out.shape)


def host_split_gemv_local(
    x, w, out_scale, inv_x, router=None, *, local, cols, used, used_e
):
    """Per-rank body inside ``shard_map`` over the host-major "model" axis.
    x [1, K] bf16; w fp8 [K, N]; out_scale f32 [N]; inv_x f32 [1];
    router bf16 [K, 128] or None."""
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
        const((1, n)),
    ]
    args = [
        inv_x.astype(jnp.float32),
        x,
        w,
        out_scale.reshape(1, n).astype(jnp.float32),
    ]
    out_shape = [jax.ShapeDtypeStruct((1, used * cols), x.dtype)]
    if has_router:
        in_specs.append(pl.BlockSpec((bk, 128), lambda i: (i, 0)))
        args.append(router)
        out_shape.append(jax.ShapeDtypeStruct((1, used_e * 128), x.dtype))
    if has_rest:
        out_shape.append(jax.ShapeDtypeStruct((1, n - cols), x.dtype))
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
        ),
        out_shape=tuple(out_shape),
        grid=(nk,),
        in_specs=in_specs,
        out_specs=tuple(const(s.shape) for s in out_shape),
        scratch_shapes=[
            pltpu.VMEM((_ROWS, n), jnp.float32),
            pltpu.VMEM((_ROWS, 128), jnp.float32),
            pltpu.VMEM((total, 128), x.dtype),
            pltpu.VMEM((local, total, 128), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT,
            collective_id=_COLLECTIVE_ID,
        ),
        name=f"host_split_gemv_k{kdim}_n{n}_c{cols}_u{used}"
        + (f"_r{used_e}" if has_router else ""),
    )(*args)
    return tuple(outs)
