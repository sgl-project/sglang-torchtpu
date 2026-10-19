# SPDX-License-Identifier: Apache-2.0
"""Pallas decode GEMV for static-input-scale fp8 weights (``StaticFp8Weight``).

``apply_tpu_static_fp8_linear`` (sglang_tpu/fp8_linear.py, which also
registers this kernel as an op) lowers to an activation-quantize fusion, an XLA
fp8 dot over 16 padded rows and a scale fusion. At decode the dot reads
the fp8 weight well below HBM bandwidth (Kimi-K3 TP32: the KDA [q, k, v, g]
projection took 12.3 us, the MoE front 12.6 us). This kernel streams the
[K, N] weight in K blocks with the default double buffering, quantizes the
activation in VMEM, accumulates in f32 and applies the per-channel output
scale before the store, so one custom call replaces the three ops (KDA qkvg
12.3 -> 3.6 us, MoE front 12.6 -> 10.7 us in the full model).

The math matches the XLA path: the same clamp-then-cast activation quantize,
an fp8 x fp8 MXU dot with f32 accumulation and the f32 output scale.
"""

import functools

# One fp8 sublane tile; the activation is zero-padded to it.
_ROWS = 32
# Decode-sized batches only, up to the padded tile; larger M keeps the XLA dot.
MAX_TOKENS = _ROWS
_FP8_MAX = 448.0
# Scoped VMEM the kernel may use: the double-buffered weight block dominates.
_VMEM_LIMIT = 40 * 1024 * 1024
_VMEM_BUDGET = 32 * 1024 * 1024
# Largest K block tried; 1792 measured best on the Kimi-K3 TP32 shapes.
_MAX_BLOCK_K = 1792


def block_k(k: int, n: int) -> int:
    """Largest 128-multiple divisor of ``k`` that is <= _MAX_BLOCK_K and whose
    working set fits the VMEM budget; 0 when none fits."""
    want = min(_MAX_BLOCK_K, k)
    # f32 accumulator + bf16 output tile, plus the double-buffered x block.
    fixed = _ROWS * n * (4 + 2 * 2)
    for bk in range(want - want % 128, 0, -128):
        if k % bk:
            continue
        if fixed + 2 * bk * n + 2 * _ROWS * bk * 2 <= _VMEM_BUDGET:
            return bk
    return 0


def _kernel(inv_ref, x_ref, w_ref, s_ref, o_ref, acc_ref, *, nk: int):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    k = pl.program_id(0)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    x = x_ref[...].astype(jnp.float32)
    if x.shape[0] != _ROWS:
        # Pad the activation to the fp8 sublane tile in VMEM; an XLA pad
        # before the call was a separate ~0.5 us op per GEMV.
        x = jnp.concatenate(
            [x, jnp.zeros((_ROWS - x.shape[0], x.shape[1]), x.dtype)], axis=0
        )
    xq = jnp.clip(x * inv_ref[0], -_FP8_MAX, _FP8_MAX).astype(w_ref.dtype)
    acc_ref[...] += jnp.dot(xq, w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == nk - 1)
    def _():
        o_ref[...] = (acc_ref[...] * s_ref[...]).astype(o_ref.dtype)


def static_fp8_gemv_jax(x, w, out_scale, inv_x, bk: int):
    """x [M, K] (M <= MAX_TOKENS), w fp8 [K, N], out_scale f32 [N],
    inv_x f32 [1] (the static activation scale's inverse) -> x.dtype [M, N]."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m, kdim = x.shape
    n = w.shape[1]
    nk = kdim // bk
    out = pl.pallas_call(
        functools.partial(_kernel, nk=nk),
        out_shape=jax.ShapeDtypeStruct((_ROWS, n), x.dtype),
        grid=(nk,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec((m, bk), lambda i: (0, i)),
            pl.BlockSpec((bk, n), lambda i: (i, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((_ROWS, n), lambda i: (0, 0)),
        scratch_shapes=[pltpu.VMEM((_ROWS, n), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT,
        ),
        name=f"static_fp8_gemv_k{kdim}_n{n}_bk{bk}",
    )(inv_x.astype(jnp.float32), x, w, out_scale.reshape(1, n).astype(jnp.float32))
    return out[:m]
