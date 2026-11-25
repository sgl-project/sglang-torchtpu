# SPDX-License-Identifier: Apache-2.0
"""One-kernel DeepSeek-V4.1 attention input chain for decode-sized batches.

Replaces, per layer, the chain from the normed attention input x to the
attention's q_lora and kv:

    qkv_a = W8A8(x, wqkv_a)                    # apply_tpu_fp8_linear
    q_lora = q_norm(qkv_a[:, :q_rank])         # RMSNorm
    kv = rope(kv_norm(qkv_a[:, q_rank:]))      # RMSNorm, RoPE on the last 64

with one Pallas call. The wqkv_a weight streams HBM -> VMEM in column chunks,
each with its own DMA semaphore, all issued at kernel entry: the activation
quantization runs under the first chunk's DMA and each chunk's matmul under the
next chunks' DMAs, instead of one XLA op after another, each with its own
pipeline fill.

Numerics follow the XLA path op by op: the per-row amax / fp8_max scale floored
at 1e-12, an unsaturated fp8 cast, an f32-accumulated fp8 dot over the whole K
(column chunks keep each output's accumulation a single dot), then
``acc * x_scale * w_scale`` rounded to bf16; both RMSNorms in f32 over the bf16
values, ``(x * weight)`` rounded to bf16; RoPE in f32 over the bf16 normed kv,
the interleaved (GPT-J) pairs of the last ``rotary_dim`` channels.

The mHC boundary that produces x runs in front of it in the same call
(qa_chain_op.try_boundary_qa).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.dsv4.rope import LANE, _cos_sin_lanes, _rotate_gptj

# Rows the single-block kernel takes; larger batches keep the XLA chain.
MAX_TOKENS = 128
# Column chunk of the streamed weight (lanes); N must be a multiple.
CHUNK_N = 256
# Weight chunks in flight. TPU XProf per call at c4: all 7 at entry 7.4 us,
# 2 in flight 9.0 us (vs 4.9 us for the XLA wqkv_a linear alone).
DMA_DEPTH = 8
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def supports(tokens: int, k: int, n: int, q_rank: int, rotary_dim: int) -> bool:
    """Whether the kernel takes x [tokens, k] @ w [k, n] with q = [:, :q_rank]."""
    kv_dim = n - q_rank
    return (
        0 < tokens <= MAX_TOKENS
        and k % 128 == 0
        and n % CHUNK_N == 0
        and q_rank % LANE == 0
        and kv_dim % LANE == 0
        and 0 < rotary_dim <= LANE
        and rotary_dim % 2 == 0
        # fp8 weight resident in VMEM, plus the bf16 product.
        and k * n + 2 * tokens * n <= _VMEM_LIMIT_BYTES // 2
    )


def _rms_norm(y, weight, eps):
    """Core RMSNorm over f32 values of bf16 inputs: ``(y * rsqrt) * w`` -> bf16."""
    variance = jnp.mean(y * y, axis=-1, keepdims=True)
    return (y * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)).astype(
        jnp.bfloat16
    )


def _weight_copies(w_hbm, w_vmem, sems):
    """One DMA per CHUNK_N-column chunk of the [K, N] weight."""
    return [
        pltpu.make_async_copy(
            w_hbm.at[:, pl.ds(c * CHUNK_N, CHUNK_N)],
            w_vmem.at[:, pl.ds(c * CHUNK_N, CHUNK_N)],
            sems.at[c],
        )
        for c in range(w_vmem.shape[1] // CHUNK_N)
    ]


def _quantize(x_ref, fp8_dtype):
    """Per-row fp8 activation: (x_q, x_scale), as apply_tpu_fp8_linear."""
    x = x_ref[...].astype(jnp.float32)
    fp8_max = float(jnp.finfo(fp8_dtype).max)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True) / fp8_max, 1e-12)
    return (x / x_scale).astype(fp8_dtype), x_scale


def _project(x_q, x_scale, w_tile, ws_tile):
    """One column tile of ``acc * x_scale * w_scale`` rounded to bf16."""
    acc = jax.lax.dot_general(
        x_q, w_tile, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    return (acc * x_scale * ws_tile).astype(jnp.bfloat16)


def _finish(y_vmem, qn_ref, kvn_ref, cs_ref, q_ref, kv_ref, *, q_rank, eps):
    """Both RMSNorms over the bf16 product, then RoPE on kv's last lanes."""
    n = y_vmem.shape[1]
    q_ref[...] = _rms_norm(
        y_vmem[:, :q_rank].astype(jnp.float32), qn_ref[...], eps
    ).astype(q_ref.dtype)
    kv = _rms_norm(y_vmem[:, q_rank:].astype(jnp.float32), kvn_ref[...], eps)
    if cs_ref is None:
        kv_ref[...] = kv.astype(kv_ref.dtype)
        return
    kv_dim = n - q_rank
    kv_ref[:, : kv_dim - LANE] = kv[:, : kv_dim - LANE].astype(kv_ref.dtype)
    tail = kv[:, kv_dim - LANE :].astype(jnp.float32)
    cos, sin = _cos_sin_lanes(cs_ref[...], rotary_dim=cs_ref.shape[-1], inverse=False)
    kv_ref[:, kv_dim - LANE :] = (tail * cos + _rotate_gptj(tail) * sin).astype(
        kv_ref.dtype
    )


def _chain(
    x_ref,
    copies,
    w_vmem,
    y_vmem,
    ws_ref,
    qn_ref,
    kvn_ref,
    cs_ref,
    q_ref,
    kv_ref,
    *,
    q_rank,
    eps,
):
    """The chain after the first DMA_DEPTH weight chunks were issued: waits on
    each chunk in turn and issues the one DMA_DEPTH ahead."""
    x_q, x_scale = _quantize(x_ref, w_vmem.dtype)
    for c, copy in enumerate(copies):
        copy.wait()
        if c + DMA_DEPTH < len(copies):
            copies[c + DMA_DEPTH].start()
        cols = pl.ds(c * CHUNK_N, CHUNK_N)
        y_vmem[:, cols] = _project(x_q, x_scale, w_vmem[:, cols], ws_ref[:, cols])
    _finish(y_vmem, qn_ref, kvn_ref, cs_ref, q_ref, kv_ref, q_rank=q_rank, eps=eps)


def _grid_kernel(
    x_ref, w_ref, ws_ref, qn_ref, kvn_ref, cs_ref, q_ref, kv_ref, y_vmem, *, q_rank, eps
):
    """One CHUNK_N column tile per grid step; Pallas double-buffers the weight
    tiles (fp8_linear_small_m's pipeline: 3.3 us at c4 on this shape). The
    norms and RoPE run once the last tile has landed."""
    j = pl.program_id(0)
    x_q, x_scale = _quantize(x_ref, w_ref.dtype)
    y_vmem[:, pl.ds(pl.multiple_of(j * CHUNK_N, CHUNK_N), CHUNK_N)] = _project(
        x_q, x_scale, w_ref[...], ws_ref[...]
    )

    @pl.when(j == pl.num_programs(0) - 1)
    def _():
        _finish(y_vmem, qn_ref, kvn_ref, cs_ref, q_ref, kv_ref, q_rank=q_rank, eps=eps)


def qa_chain(
    x: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    q_norm_weight: jax.Array,
    kv_norm_weight: jax.Array,
    cos_sin: jax.Array,
    *,
    q_rank: int,
    eps: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """x [T, K] bf16, weight [K, N] fp8, weight_scale [N] (or [N, 1]) f32,
    norm weights [q_rank] and [N - q_rank], cos_sin [T, rotary_dim] f32 rows
    (``rope.cos_sin_rows``). Returns (normed q_lora [T, q_rank], normed RoPE'd
    kv [T, N - q_rank]), both in x's dtype."""
    tokens, k = x.shape
    n = weight.shape[1]
    rotary_dim = cos_sin.shape[-1]
    kv_dim = n - q_rank
    ws = weight_scale.astype(jnp.float32).reshape(1, n)

    def whole(shape):
        return pl.BlockSpec(shape, lambda j: (0,) * len(shape))

    return pl.pallas_call(
        functools.partial(_grid_kernel, q_rank=q_rank, eps=eps),
        grid=(n // CHUNK_N,),
        in_specs=[
            whole((tokens, k)),
            pl.BlockSpec((k, CHUNK_N), lambda j: (0, j)),
            pl.BlockSpec((1, CHUNK_N), lambda j: (0, j)),
            whole((1, q_rank)),
            whole((1, kv_dim)),
            whole((tokens, rotary_dim)),
        ],
        out_specs=[whole((tokens, q_rank)), whole((tokens, kv_dim))],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, q_rank), x.dtype),
            jax.ShapeDtypeStruct((tokens, kv_dim), x.dtype),
        ],
        scratch_shapes=[pltpu.VMEM((tokens, n), jnp.bfloat16)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=_VMEM_LIMIT_BYTES
        ),
        interpret=interpret,
        name=f"dsv41_qa_chain_{k}x{n}",
    )(
        x,
        weight,
        ws,
        q_norm_weight.reshape(1, q_rank),
        kv_norm_weight.reshape(1, kv_dim),
        cos_sin,
    )


def qa_chain_epilogue(
    tokens: int,
    weight: jax.Array,
    weight_scale: jax.Array,
    q_norm_weight: jax.Array,
    kv_norm_weight: jax.Array,
    cos_sin: jax.Array | None,
    *,
    q_rank: int,
    eps: float,
):
    """The chain as a ``PostPreEpilogue`` of the mHC boundary kernel: its weight
    DMAs start at the boundary's entry and the chain reads the boundary's VMEM
    value. The boundary then also returns (q_lora, kv); kv is RoPE'd only when
    ``cos_sin`` rows are given."""
    from sglang_tpu.models.deepseek_v41.kernels.mhc.shifted_post_pre_rms import (
        PostPreEpilogue,
    )

    k, n = weight.shape
    kv_dim = n - q_rank
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)

    def start(in_refs, out_refs, scratch):
        w_hbm = in_refs[0]
        w_vmem, _, sems = scratch
        for copy in _weight_copies(w_hbm, w_vmem, sems)[:DMA_DEPTH]:
            copy.start()

    def run(value_ref, in_refs, out_refs, scratch):
        w_hbm, ws_ref, qn_ref, kvn_ref, *rows = in_refs
        cs_ref = rows[0] if rows else None
        w_vmem, y_vmem, sems = scratch
        _chain(
            value_ref,
            _weight_copies(w_hbm, w_vmem, sems),
            w_vmem,
            y_vmem,
            ws_ref,
            qn_ref,
            kvn_ref,
            cs_ref,
            *out_refs,
            q_rank=q_rank,
            eps=eps,
        )

    return PostPreEpilogue(
        inputs=(
            pltpu.with_memory_space_constraint(weight, pltpu.HBM),
            weight_scale.astype(jnp.float32).reshape(1, n),
            q_norm_weight.reshape(1, q_rank),
            kv_norm_weight.reshape(1, kv_dim),
        )
        + (() if cos_sin is None else (cos_sin,)),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM), vmem, vmem, vmem)
        + (() if cos_sin is None else (vmem,)),
        out_shapes=(
            jax.ShapeDtypeStruct((tokens, q_rank), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, kv_dim), jnp.bfloat16),
        ),
        out_specs=(vmem, vmem),
        scratch_shapes=(
            pltpu.VMEM((k, n), weight.dtype),
            pltpu.VMEM((tokens, n), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((n // CHUNK_N,)),
        ),
        start=start,
        run=run,
    )


def qa_chain_reference(
    x, weight, weight_scale, q_norm_weight, kv_norm_weight, cos_sin, *, q_rank, eps
):
    """The XLA chain the kernel replaces, op for op (jnp)."""
    n = weight.shape[1]
    fp8_max = float(jnp.finfo(weight.dtype).max)
    xf = x.astype(jnp.float32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=1, keepdims=True) / fp8_max, 1e-12)
    x_q = (xf / x_scale).astype(weight.dtype)
    acc = jnp.dot(x_q, weight, preferred_element_type=jnp.float32)
    y = (acc * x_scale * weight_scale.astype(jnp.float32).reshape(1, n)).astype(x.dtype)
    q = _rms_norm(y[:, :q_rank].astype(jnp.float32), q_norm_weight, eps).astype(x.dtype)
    kv = _rms_norm(y[:, q_rank:].astype(jnp.float32), kv_norm_weight, eps)
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    cos, sin = cos_sin[:, :half], cos_sin[:, half:]
    tail = kv[:, -rotary_dim:].astype(jnp.float32).reshape(kv.shape[0], half, 2)
    even, odd = tail[..., 0], tail[..., 1]
    rot = jnp.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
    kv = jnp.concatenate(
        [kv[:, :-rotary_dim], rot.reshape(kv.shape[0], rotary_dim).astype(kv.dtype)], -1
    ).astype(x.dtype)
    return q, kv
