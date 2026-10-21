# SPDX-License-Identifier: Apache-2.0
"""Small-batch fused MoE that streams the selected experts' weights.

At batch size 1 with all experts local (tensor-parallel experts, EP = 1), the
routed rows are k copies of one token. A grid over the k slots with the expert
ids scalar-prefetched makes each step's weight blocks w1[ids[s]] / w2[ids[s]],
so the pipeline fetches the next expert's weights while the current one
computes, and the weighted sum accumulates on-chip. This replaces route-meta,
the row broadcast, both grouped GEMMs, the activation and the combine.

Per expert it reproduces gmm_v2's quantized path (per-row FP8 quantization of
the lhs in 512-wide K blocks, block partials rescaled by lhs and per-channel rhs
scales, bf16 accumulation) and fused_moe's activation (silu(min(gate, l)) * clip(up, l),
rounded once). Each expert output is rounded to the activation dtype before
the f32 weighted sum, as the fused-MoE combine does.

For 2..8 tokens (a speculative verify window) a grid over the distinct selected
experts streams each expert's weights once for all rows and accumulates each
token's routing weight on chip.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.gmm_v2 import _FP32_MOE_SCALES, _scale_quantized_partial

_LHS_QUANT_BLOCK = 512
# One expert's FP8 w1 + w2 blocks, double-buffered; larger experts keep gmm_v2.
_EXPERT_BLOCK_BUDGET = 24 * 1024 * 1024


def supports(tokens: int, hidden: int, two_n: int) -> bool:
    expert_bytes = hidden * two_n + (two_n // 2) * hidden
    return (
        tokens == 1
        and hidden % 128 == 0
        and two_n % 256 == 0
        and 2 * expert_bytes <= _EXPERT_BLOCK_BUDGET
    )


def _block_quant_dot(x, w, w_scale, *, fp8_dtype):
    """gmm_v2's quantized matmul for one row: x [1, K] in the activation dtype,
    w [K, N]. The lhs scale, its reciprocal and the accumulator are in the
    activation dtype, as gmm_v2 runs a quantized lhs (bf16 accumulation); each
    block partial is rescaled with gmm_v2's own helper and scale dtype."""
    fp8_max = float(jnp.finfo(fp8_dtype).max)
    dt = x.dtype
    k = x.shape[1]
    block = min(_LHS_QUANT_BLOCK, k)
    acc = None
    for start in range(0, k, block):
        xb = x[:, start : start + block]
        scale = jnp.max(jnp.abs(xb), axis=1, keepdims=True) / fp8_max
        scale_inv = jnp.where(scale == 0, 0, 1 / scale)
        partial = jax.lax.dot_general(
            (xb * scale_inv).astype(fp8_dtype),
            w[start : start + block, :],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(dt)
        partial = _scale_quantized_partial(
            partial,
            scale,
            w_scale,
            scales_dtype=jnp.float32 if _FP32_MOE_SCALES else dt,
        )
        acc = partial if acc is None else acc + partial
    return acc


def _kernel(
    ids_ref,
    wts_ref,
    x_ref,
    w1_ref,
    s1_ref,
    w2_ref,
    s2_ref,
    o_ref,
    acc_ref,
    *,
    n,
    limit,
    fp8_dtype,
):
    s = pl.program_id(0)

    @pl.when(s == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    dt = o_ref.dtype
    dot = functools.partial(_block_quant_dot, fp8_dtype=fp8_dtype)
    gate_up = dot(x_ref[...], w1_ref[0], s1_ref[0])
    gate, up = gate_up[:, :n], gate_up[:, n:]
    if limit is not None:
        gate = jnp.minimum(gate, limit)
        up = jnp.clip(up, -limit, limit)
    h = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(dt)
    y = dot(h, w2_ref[0], s2_ref[0])
    acc_ref[...] += y.astype(jnp.float32) * wts_ref[s]

    @pl.when(s == pl.num_programs(0) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(dt)


def supports_few_tokens(tokens: int, hidden: int, two_n: int) -> bool:
    # Every grid step computes all rows, so wider batches waste MXU work.
    return 1 < tokens <= 8 and supports(1, hidden, two_n)


def _few_token_kernel(
    uniq_ref,
    count_ref,
    wv_ref,
    x_ref,
    w1_ref,
    s1_ref,
    w2_ref,
    s2_ref,
    o_ref,
    acc_ref,
    *,
    n,
    limit,
    fp8_dtype,
):
    s = pl.program_id(0)

    @pl.when(s == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Steps past the distinct-expert count repeat the last expert's blocks, so
    # the pipeline fetches nothing new; skip their compute.
    @pl.when(s < count_ref[0])
    def _():
        dt = o_ref.dtype
        dot = functools.partial(_block_quant_dot, fp8_dtype=fp8_dtype)
        gate_up = dot(x_ref[...], w1_ref[0], s1_ref[0])
        gate, up = gate_up[:, :n], gate_up[:, n:]
        if limit is not None:
            gate = jnp.minimum(gate, limit)
            up = jnp.clip(up, -limit, limit)
        h = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(dt)
        y = dot(h, w2_ref[0], s2_ref[0])
        # [tokens, 1] routing weight of this expert per token, 0 if not selected.
        acc_ref[...] += y.astype(jnp.float32) * wv_ref[0]

    @pl.when(s == pl.num_programs(0) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _distinct_experts(topk_ids, topk_weights, num_experts):
    """Distinct selected experts padded with the last one, their count, and the
    [slots, tokens, 1] f32 weight of each slot's expert per token. Ids outside
    [0, num_experts) (padding rows) select nothing."""
    tokens, k = topk_ids.shape
    slots = tokens * k
    ids = topk_ids.astype(jnp.int32)
    valid = (ids >= 0) & (ids < num_experts)
    srt = jnp.sort(jnp.where(valid, ids, num_experts).reshape(slots))
    first = jnp.concatenate([jnp.ones((1,), bool), srt[1:] != srt[:-1]])
    first = first & (srt < num_experts)
    count = jnp.sum(first.astype(jnp.int32))
    pos = jnp.where(first, jnp.cumsum(first.astype(jnp.int32)) - 1, slots)
    uniq = jnp.zeros((slots,), jnp.int32).at[pos].set(srt, mode="drop")
    last = uniq[jnp.maximum(count - 1, 0)]
    live = jnp.arange(slots) < count
    uniq = jnp.where(live, uniq, last)
    hit = (uniq[:, None, None] == ids[None]) & valid[None] & live[:, None, None]
    weights = jnp.sum(
        jnp.where(hit, topk_weights.astype(jnp.float32)[None], 0.0), axis=-1
    )
    return uniq, count.reshape(1), weights.reshape(slots, tokens, 1)


def moe_stream_few_tokens(
    x: jax.Array,
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array,
    w2: jax.Array,
    w2_scale: jax.Array,
    clamp_limit: float | None,
) -> jax.Array:
    """x [T, H] with T <= 8; ids / weights [T, k]; weights as moe_stream_bs1.

    Streams each distinct selected expert's weights once for all T rows. Sums
    experts in id order rather than each token's top-k order, so the f32
    combine is not bit-identical to the routed path. Returns [T, H]."""
    tokens, hidden = x.shape
    num_experts, _, two_n = w1.shape
    n = two_n // 2
    slots = tokens * topk_ids.shape[1]
    uniq, count, weights = _distinct_experts(topk_ids, topk_weights, num_experts)
    s1 = w1_scale.reshape(num_experts, 1, two_n).astype(jnp.float32)
    s2 = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(slots,),
        in_specs=[
            pl.BlockSpec((1, tokens, 1), lambda s, u, c: (s, 0, 0)),
            pl.BlockSpec((tokens, hidden), lambda s, u, c: (0, 0)),
            pl.BlockSpec((1, hidden, two_n), lambda s, u, c: (u[s], 0, 0)),
            pl.BlockSpec((1, 1, two_n), lambda s, u, c: (u[s], 0, 0)),
            pl.BlockSpec((1, n, hidden), lambda s, u, c: (u[s], 0, 0)),
            pl.BlockSpec((1, 1, hidden), lambda s, u, c: (u[s], 0, 0)),
        ],
        out_specs=pl.BlockSpec((tokens, hidden), lambda s, u, c: (0, 0)),
        scratch_shapes=[pltpu.VMEM((tokens, hidden), jnp.float32)],
    )
    return pl.pallas_call(
        functools.partial(
            _few_token_kernel, n=n, limit=clamp_limit, fp8_dtype=w1.dtype
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"moe_stream_few_{tokens}x{num_experts}x{hidden}x{two_n}",
    )(uniq, count, weights, x, w1, s1, w2, s2)


def moe_stream_bs1(
    x: jax.Array,
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array,
    w2: jax.Array,
    w2_scale: jax.Array,
    clamp_limit: float | None,
) -> jax.Array:
    """x [1, H]; ids / weights [1, k]; w1 [E, H, 2n], w2 [E, n, H] FP8 with
    per-channel scales [E, 1, 1, 2n] / [E, 1, 1, H]. Returns [1, H]. Ids
    outside [0, E) (TopK's padded tokens) contribute nothing."""
    _, hidden = x.shape
    k = topk_ids.shape[1]
    num_experts, _, two_n = w1.shape
    n = two_n // 2
    ids = topk_ids.reshape(k).astype(jnp.int32)
    valid = (ids >= 0) & (ids < num_experts)
    ids = jnp.where(valid, ids, 0)
    wts = jnp.where(valid, topk_weights.reshape(k).astype(jnp.float32), 0.0)
    s1 = w1_scale.reshape(num_experts, 1, two_n).astype(jnp.float32)
    s2 = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(k,),
        in_specs=[
            pl.BlockSpec((1, hidden), lambda s, ids, wts: (0, 0)),
            pl.BlockSpec((1, hidden, two_n), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, 1, two_n), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, n, hidden), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, 1, hidden), lambda s, ids, wts: (ids[s], 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, hidden), lambda s, ids, wts: (0, 0)),
        scratch_shapes=[pltpu.VMEM((1, hidden), jnp.float32)],
    )
    return pl.pallas_call(
        functools.partial(_kernel, n=n, limit=clamp_limit, fp8_dtype=w1.dtype),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((1, hidden), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"moe_stream_bs1_{num_experts}x{hidden}x{two_n}",
    )(ids, wts, x, w1, s1, w2, s2)


def maybe_moe_stream(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    *,
    activation_name,
    no_combine,
    use_group_offset,
    w_quant_bits,
):
    """The streaming kernel's output for a fused-MoE call it covers, else None.

    Covers 1..8 bf16 tokens with every expert local (EP = 1), unbiased FP8
    experts with per-channel scales and the clamped SiLU, and experts small
    enough for VMEM. Plain SiLU fuses the activation into gmm_v2 and keeps that
    path.
    """
    if not activation_name.startswith("silu_and_mul_clamp:"):
        return None
    clamp_limit = float(activation_name.split(":", 1)[1])
    tokens, hidden = hidden_states.shape
    num_experts, w1_k, two_n = w1.shape
    if (
        no_combine
        or use_group_offset
        or w_quant_bits
        or w1_bias is not None
        or w2_bias is not None
        or w1_scale is None
        or w2_scale is None
        or hidden_states.dtype != jnp.bfloat16
        or w1.dtype != jnp.float8_e4m3fn
        or w2.dtype != jnp.float8_e4m3fn
        or w1_k != hidden
        or w2.shape != (num_experts, two_n // 2, hidden)
        or w1_scale.shape != (num_experts, 1, 1, two_n)
        or w2_scale.shape != (num_experts, 1, 1, hidden)
    ):
        return None
    if supports(tokens, hidden, two_n):
        kernel = moe_stream_bs1
    elif supports_few_tokens(tokens, hidden, two_n):
        kernel = moe_stream_few_tokens
    else:
        return None
    return kernel(
        hidden_states,
        topk_ids,
        topk_weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
        clamp_limit,
    )
