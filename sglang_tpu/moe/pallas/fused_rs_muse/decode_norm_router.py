# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas VMEM decode glue kernels for Muse Spark (D5 router + D6 norms/residuals).

All kernels operate on small decode batches (tokens <= 256) with 1024-column VMEM strip
tiling (<= 32 VREGs live -> zero register spills on Ghostfish) and canonical 2D column-minor
HBM layouts ``{1,0:T(8,128)}``, eliminating cross-lane ``input_layernorm/mean`` reductions
and 4D ``[16,8,8,1024]`` post-expert TP all-gather transpose copies.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from threading import Lock
from typing import Callable

import torch

_LOCK = Lock()
_ROUTER_OP: list[Callable] = []
_ROUTER_EPLB_OPS: dict[tuple[int, int], Callable] = {}
_RMSNORM_OP: list[Callable] = []
_RES_NORM_OP: list[Callable] = []
_RES_ONLY_OP: list[Callable] = []
_TP_AG_RES_ONLY_OP: list[Callable] = []
_TP_AG_RES_NORM_OP: list[Callable] = []

_COL_STRIP = 1024
_LANE = 128


def _vpu_sum_128(x):
    """Fold [bt, K] (K % 128 == 0) into [bt, 128] on VPU without cross-lane XLU reductions."""
    n = x.shape[1] // _LANE
    tiles = [x[:, i * _LANE : (i + 1) * _LANE] for i in range(n)]
    while len(tiles) > 1:
        tiles = [
            (tiles[2 * i] + tiles[2 * i + 1]) if (2 * i + 1 < len(tiles)) else tiles[2 * i]
            for i in range((len(tiles) + 1) // 2)
        ]
    return tiles[0]


def _row_block(tokens: int) -> int:
    """Pick a 16-row-aligned block size so bf16 T(16,128)(2,1) never slices mid-tile."""
    for bt in (32, 16):
        if tokens >= bt and tokens % bt == 0:
            return bt
    return tokens


def _use_norm_fast() -> bool:
    from sglang_tpu.envs import envs

    f = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_FAST", None)
    return bool(f is not None and f.get())


def _rmsnorm_pallas_kernel(
    x_ref,
    gain_ref,
    out_ref,
    *,
    eps: float,
    has_gain: bool,
    scale_input: bool,
    lane_fast: bool = False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    bt, hidden = x_ref.shape
    if hidden % _COL_STRIP == 0 and bt <= 64:
        n_strips = hidden // _COL_STRIP
        if lane_fast:
            sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                xc = x_ref[:, col].astype(jnp.float32)
                if has_gain and scale_input:
                    xc = xc * gain_ref[0:1, col].astype(jnp.float32)
                sq_128 = sq_128 + _vpu_sum_128(xc * xc)
            sq_sum = jnp.sum(sq_128, axis=-1, keepdims=True)
        else:
            sq_sum = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                xc = x_ref[:, col].astype(jnp.float32)
                if has_gain and scale_input:
                    xc = xc * gain_ref[0:1, col].astype(jnp.float32)
                sq_sum = sq_sum + jnp.sum(xc * xc, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(sq_sum * (1.0 / hidden) + eps)
        for s in range(n_strips):
            col = pl.ds(s * _COL_STRIP, _COL_STRIP)
            xc = x_ref[:, col].astype(jnp.float32)
            if has_gain:
                gc = gain_ref[0:1, col].astype(jnp.float32)
                yc = (xc * gc) * inv if scale_input else (xc * inv) * gc
            else:
                yc = xc * inv
            out_ref[:, col] = yc.astype(out_ref.dtype)
        return

    x = x_ref[...].astype(jnp.float32)
    if has_gain and scale_input:
        g = gain_ref[...].reshape(1, -1).astype(jnp.float32)
        xs = x * g
        var = jnp.mean(xs * xs, axis=-1, keepdims=True)
        y = xs * jax.lax.rsqrt(var + eps)
    else:
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        y = x * jax.lax.rsqrt(var + eps)
        if has_gain:
            g = gain_ref[...].reshape(1, -1).astype(jnp.float32)
            y = y * g
    out_ref[...] = y.astype(out_ref.dtype)


def rmsnorm_pallas_jax(
    x,
    gain,
    *,
    eps: float,
    has_gain: bool,
    scale_input: bool,
    out_bf16: bool,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = x.shape
    bt = _row_block(tokens)
    lane_fast = _use_norm_fast()
    out_dtype = jnp.bfloat16 if out_bf16 else jnp.float32
    row_spec = pl.BlockSpec((bt, hidden), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _rmsnorm_pallas_kernel,
            eps=eps,
            has_gain=has_gain,
            scale_input=scale_input,
            lane_fast=lane_fast,
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), out_dtype),
        grid=(tokens // bt,),
        in_specs=[row_spec, vec_spec],
        out_specs=row_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=12 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=f"muse_decode_rmsnorm_{tokens}x{hidden}_b{bt}_{'bf16' if out_bf16 else 'f32'}{'_lf' if lane_fast else ''}",
    )(x, gain.reshape(1, hidden).astype(jnp.float32))


def _residual_only_pallas_kernel(
    stream_ref,
    branch_ref,
    branch_gain_ref,
    alpha_ref,
    beta_ref,
    stream_out_ref,
    *,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
    lane_fast: bool = False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    bt, hidden = stream_ref.shape
    if hidden % _COL_STRIP == 0:
        n_strips = hidden // _COL_STRIP
        if lane_fast:
            sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                bc = branch_ref[:, col].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_128 = sq_128 + _vpu_sum_128(bc * bc)
            sq_sum = jnp.sum(sq_128, axis=-1, keepdims=True)
        else:
            sq_sum = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                bc = branch_ref[:, col].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_sum = sq_sum + jnp.sum(bc * bc, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(sq_sum * (1.0 / hidden) + branch_eps)
        for s in range(n_strips):
            col = pl.ds(s * _COL_STRIP, _COL_STRIP)
            sc = stream_ref[:, col].astype(jnp.float32)
            bc = branch_ref[:, col].astype(jnp.float32)
            if has_branch_gain:
                bg = branch_gain_ref[0:1, col].astype(jnp.float32)
                b_norm = (bc * bg) * inv if branch_scale_input else (bc * inv) * bg
            else:
                b_norm = bc * inv
            alp = alpha_ref[0:1, col].astype(jnp.float32)
            bet = beta_ref[0:1, col].astype(jnp.float32)
            stream_out_ref[:, col] = alp * sc + bet * b_norm
        return

    s = stream_ref[...].astype(jnp.float32)
    b = branch_ref[...].astype(jnp.float32)
    if has_branch_gain and branch_scale_input:
        bg = branch_gain_ref[...].reshape(1, -1).astype(jnp.float32)
        bs = b * bg
        b_norm = bs * jax.lax.rsqrt(jnp.mean(bs * bs, axis=-1, keepdims=True) + branch_eps)
    else:
        b_norm = b * jax.lax.rsqrt(jnp.mean(b * b, axis=-1, keepdims=True) + branch_eps)
        if has_branch_gain:
            bg = branch_gain_ref[...].reshape(1, -1).astype(jnp.float32)
            b_norm = b_norm * bg
    alpha = alpha_ref[...].reshape(1, -1).astype(jnp.float32)
    beta = beta_ref[...].reshape(1, -1).astype(jnp.float32)
    stream_out_ref[...] = alpha * s + beta * b_norm


def _residual_and_next_norm_pallas_kernel(
    stream_ref,
    branch_ref,
    branch_gain_ref,
    alpha_ref,
    beta_ref,
    next_gain_ref,
    stream_out_ref,
    next_norm_out_ref,
    *,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
    next_eps: float,
    lane_fast: bool = False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    bt, hidden = stream_ref.shape
    if hidden % _COL_STRIP == 0:
        n_strips = hidden // _COL_STRIP
        if lane_fast:
            sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                bc = branch_ref[:, col].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_128 = sq_128 + _vpu_sum_128(bc * bc)
            sq_sum = jnp.sum(sq_128, axis=-1, keepdims=True)
        else:
            sq_sum = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                bc = branch_ref[:, col].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_sum = sq_sum + jnp.sum(bc * bc, axis=-1, keepdims=True)
        inv_b = jax.lax.rsqrt(sq_sum * (1.0 / hidden) + branch_eps)
        if lane_fast:
            next_sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                sc = stream_ref[:, col].astype(jnp.float32)
                bc = branch_ref[:, col].astype(jnp.float32)
                if has_branch_gain:
                    bg = branch_gain_ref[0:1, col].astype(jnp.float32)
                    b_norm = (bc * bg) * inv_b if branch_scale_input else (bc * inv_b) * bg
                else:
                    b_norm = bc * inv_b
                alp = alpha_ref[0:1, col].astype(jnp.float32)
                bet = beta_ref[0:1, col].astype(jnp.float32)
                new_sc = alp * sc + bet * b_norm
                stream_out_ref[:, col] = new_sc
                next_sq_128 = next_sq_128 + _vpu_sum_128(new_sc * new_sc)
            next_sq = jnp.sum(next_sq_128, axis=-1, keepdims=True)
        else:
            next_sq = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(n_strips):
                col = pl.ds(s * _COL_STRIP, _COL_STRIP)
                sc = stream_ref[:, col].astype(jnp.float32)
                bc = branch_ref[:, col].astype(jnp.float32)
                if has_branch_gain:
                    bg = branch_gain_ref[0:1, col].astype(jnp.float32)
                    b_norm = (bc * bg) * inv_b if branch_scale_input else (bc * inv_b) * bg
                else:
                    b_norm = bc * inv_b
                alp = alpha_ref[0:1, col].astype(jnp.float32)
                bet = beta_ref[0:1, col].astype(jnp.float32)
                new_sc = alp * sc + bet * b_norm
                stream_out_ref[:, col] = new_sc
                next_sq = next_sq + jnp.sum(new_sc * new_sc, axis=-1, keepdims=True)
        inv_n = jax.lax.rsqrt(next_sq * (1.0 / hidden) + next_eps)
        for s in range(n_strips):
            col = pl.ds(s * _COL_STRIP, _COL_STRIP)
            new_sc = stream_out_ref[:, col]
            ng = next_gain_ref[0:1, col].astype(jnp.float32)
            next_norm_out_ref[:, col] = (new_sc * inv_n * ng).astype(
                next_norm_out_ref.dtype
            )
        return

    s = stream_ref[...].astype(jnp.float32)
    b = branch_ref[...].astype(jnp.float32)
    if has_branch_gain and branch_scale_input:
        bg = branch_gain_ref[...].reshape(1, -1).astype(jnp.float32)
        bs = b * bg
        b_norm = bs * jax.lax.rsqrt(jnp.mean(bs * bs, axis=-1, keepdims=True) + branch_eps)
    else:
        b_norm = b * jax.lax.rsqrt(jnp.mean(b * b, axis=-1, keepdims=True) + branch_eps)
        if has_branch_gain:
            bg = branch_gain_ref[...].reshape(1, -1).astype(jnp.float32)
            b_norm = b_norm * bg
    alpha = alpha_ref[...].reshape(1, -1).astype(jnp.float32)
    beta = beta_ref[...].reshape(1, -1).astype(jnp.float32)
    new_s = alpha * s + beta * b_norm
    stream_out_ref[...] = new_s
    ng = next_gain_ref[...].reshape(1, -1).astype(jnp.float32)
    n_norm = new_s * jax.lax.rsqrt(jnp.mean(new_s * new_s, axis=-1, keepdims=True) + next_eps) * ng
    next_norm_out_ref[...] = n_norm.astype(next_norm_out_ref.dtype)


def residual_only_pallas_jax(
    stream,
    branch,
    branch_gain,
    alpha,
    beta,
    *,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = stream.shape
    bt = _row_block(tokens)
    lane_fast = _use_norm_fast()
    row_spec = pl.BlockSpec((bt, hidden), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _residual_only_pallas_kernel,
            branch_eps=branch_eps,
            has_branch_gain=has_branch_gain,
            branch_scale_input=branch_scale_input,
            lane_fast=lane_fast,
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
        grid=(tokens // bt,),
        in_specs=[row_spec, row_spec, vec_spec, vec_spec, vec_spec],
        out_specs=row_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=16 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=f"muse_decode_res_only_{tokens}x{hidden}_b{bt}{'_lf' if lane_fast else ''}",
    )(
        stream.astype(jnp.float32),
        branch,
        branch_gain.reshape(1, hidden).astype(jnp.float32),
        alpha.reshape(1, hidden).astype(jnp.float32),
        beta.reshape(1, hidden).astype(jnp.float32),
    )


def residual_and_next_norm_pallas_jax(
    stream,
    branch,
    branch_gain,
    alpha,
    beta,
    next_gain,
    *,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
    next_eps: float,
    next_out_bf16: bool,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = stream.shape
    bt = _row_block(tokens)
    lane_fast = _use_norm_fast()
    next_dtype = jnp.bfloat16 if next_out_bf16 else jnp.float32
    row_spec = pl.BlockSpec((bt, hidden), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _residual_and_next_norm_pallas_kernel,
            branch_eps=branch_eps,
            has_branch_gain=has_branch_gain,
            branch_scale_input=branch_scale_input,
            next_eps=next_eps,
            lane_fast=lane_fast,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
            jax.ShapeDtypeStruct((tokens, hidden), next_dtype),
        ),
        grid=(tokens // bt,),
        in_specs=[row_spec, row_spec, vec_spec, vec_spec, vec_spec, vec_spec],
        out_specs=(row_spec, row_spec),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=16 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=f"muse_decode_res_norm_{tokens}x{hidden}_b{bt}_{'bf16' if next_out_bf16 else 'f32'}{'_lf' if lane_fast else ''}",
    )(
        stream.astype(jnp.float32),
        branch,
        branch_gain.reshape(1, hidden).astype(jnp.float32),
        alpha.reshape(1, hidden).astype(jnp.float32),
        beta.reshape(1, hidden).astype(jnp.float32),
        next_gain.reshape(1, hidden).astype(jnp.float32),
    )


def _tp_ag_res_only_pallas_kernel(
    stream_ref,
    gathered_ref,
    branch_gain_ref,
    alpha_ref,
    beta_ref,
    stream_out_ref,
    *,
    tokens: int,
    tp_size: int,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
    lane_fast: bool = False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    _, hidden = stream_ref.shape
    shard_cols = hidden // tp_size
    bt = 32 if (tokens >= 32 and tokens % 32 == 0) else 16
    for r0 in range(0, tokens, bt):
        row_s = pl.ds(r0, bt)
        if lane_fast:
            sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(tp_size):
                col = pl.ds(s * shard_cols, shard_cols)
                bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_128 = sq_128 + _vpu_sum_128(bc * bc)
            sq_sum = jnp.sum(sq_128, axis=-1, keepdims=True)
        else:
            sq_sum = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(tp_size):
                col = pl.ds(s * shard_cols, shard_cols)
                bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_sum = sq_sum + jnp.sum(bc * bc, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(sq_sum * (1.0 / hidden) + branch_eps)
        for s in range(tp_size):
            col = pl.ds(s * shard_cols, shard_cols)
            sc = stream_ref[row_s, col].astype(jnp.float32)
            bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
            if has_branch_gain:
                bg = branch_gain_ref[0:1, col].astype(jnp.float32)
                b_norm = (bc * bg) * inv if branch_scale_input else (bc * inv) * bg
            else:
                b_norm = bc * inv
            alp = alpha_ref[0:1, col].astype(jnp.float32)
            bet = beta_ref[0:1, col].astype(jnp.float32)
            stream_out_ref[row_s, col] = alp * sc + bet * b_norm


def _tp_ag_res_norm_pallas_kernel(
    stream_ref,
    gathered_ref,
    branch_gain_ref,
    alpha_ref,
    beta_ref,
    next_gain_ref,
    stream_out_ref,
    next_norm_out_ref,
    *,
    tokens: int,
    tp_size: int,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
    next_eps: float,
    lane_fast: bool = False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    _, hidden = stream_ref.shape
    shard_cols = hidden // tp_size
    bt = 32 if (tokens >= 32 and tokens % 32 == 0) else 16
    for r0 in range(0, tokens, bt):
        row_s = pl.ds(r0, bt)
        if lane_fast:
            sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(tp_size):
                col = pl.ds(s * shard_cols, shard_cols)
                bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_128 = sq_128 + _vpu_sum_128(bc * bc)
            sq_sum = jnp.sum(sq_128, axis=-1, keepdims=True)
        else:
            sq_sum = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(tp_size):
                col = pl.ds(s * shard_cols, shard_cols)
                bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
                if has_branch_gain and branch_scale_input:
                    bc = bc * branch_gain_ref[0:1, col].astype(jnp.float32)
                sq_sum = sq_sum + jnp.sum(bc * bc, axis=-1, keepdims=True)
        inv_b = jax.lax.rsqrt(sq_sum * (1.0 / hidden) + branch_eps)
        if lane_fast:
            next_sq_128 = jnp.zeros((bt, _LANE), dtype=jnp.float32)
            for s in range(tp_size):
                col = pl.ds(s * shard_cols, shard_cols)
                sc = stream_ref[row_s, col].astype(jnp.float32)
                bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
                if has_branch_gain:
                    bg = branch_gain_ref[0:1, col].astype(jnp.float32)
                    b_norm = (bc * bg) * inv_b if branch_scale_input else (bc * inv_b) * bg
                else:
                    b_norm = bc * inv_b
                alp = alpha_ref[0:1, col].astype(jnp.float32)
                bet = beta_ref[0:1, col].astype(jnp.float32)
                new_sc = alp * sc + bet * b_norm
                stream_out_ref[row_s, col] = new_sc
                next_sq_128 = next_sq_128 + _vpu_sum_128(new_sc * new_sc)
            next_sq = jnp.sum(next_sq_128, axis=-1, keepdims=True)
        else:
            next_sq = jnp.zeros((bt, 1), dtype=jnp.float32)
            for s in range(tp_size):
                col = pl.ds(s * shard_cols, shard_cols)
                sc = stream_ref[row_s, col].astype(jnp.float32)
                bc = gathered_ref[pl.ds(s * tokens + r0, bt), :].astype(jnp.float32)
                if has_branch_gain:
                    bg = branch_gain_ref[0:1, col].astype(jnp.float32)
                    b_norm = (bc * bg) * inv_b if branch_scale_input else (bc * inv_b) * bg
                else:
                    b_norm = bc * inv_b
                alp = alpha_ref[0:1, col].astype(jnp.float32)
                bet = beta_ref[0:1, col].astype(jnp.float32)
                new_sc = alp * sc + bet * b_norm
                stream_out_ref[row_s, col] = new_sc
                next_sq = next_sq + jnp.sum(new_sc * new_sc, axis=-1, keepdims=True)
        inv_n = jax.lax.rsqrt(next_sq * (1.0 / hidden) + next_eps)
        for s in range(tp_size):
            col = pl.ds(s * shard_cols, shard_cols)
            new_sc = stream_out_ref[row_s, col]
            ng = next_gain_ref[0:1, col].astype(jnp.float32)
            next_norm_out_ref[row_s, col] = (new_sc * inv_n * ng).astype(
                next_norm_out_ref.dtype
            )


def tp_ag_res_only_pallas_jax(
    stream,
    gathered_2d,
    branch_gain,
    alpha,
    beta,
    *,
    tp_size: int,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
):
    """Consume raw 2D ``[tp_size * B, hidden // tp_size]`` gather output in VMEM without 4D transpose copy."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = stream.shape
    shard_cols = hidden // tp_size
    lane_fast = _use_norm_fast()
    assert tokens % 16 == 0 and shard_cols % 128 == 0
    full_s_spec = pl.BlockSpec((tokens, hidden), lambda i: (0, 0))
    ag_spec = pl.BlockSpec((tp_size * tokens, shard_cols), lambda i: (0, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _tp_ag_res_only_pallas_kernel,
            tokens=tokens,
            tp_size=tp_size,
            branch_eps=branch_eps,
            has_branch_gain=has_branch_gain,
            branch_scale_input=branch_scale_input,
            lane_fast=lane_fast,
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
        grid=(1,),
        in_specs=[full_s_spec, ag_spec, vec_spec, vec_spec, vec_spec],
        out_specs=full_s_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(56 if tokens > 256 else 32) * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=f"muse_decode_tpag_res_only_{tokens}x{hidden}_tp{tp_size}{'_lf' if lane_fast else ''}",
    )(
        stream.astype(jnp.float32),
        gathered_2d,
        branch_gain.reshape(1, hidden).astype(jnp.float32),
        alpha.reshape(1, hidden).astype(jnp.float32),
        beta.reshape(1, hidden).astype(jnp.float32),
    )


def tp_ag_res_norm_pallas_jax(
    stream,
    gathered_2d,
    branch_gain,
    alpha,
    beta,
    next_gain,
    *,
    tp_size: int,
    branch_eps: float,
    has_branch_gain: bool,
    branch_scale_input: bool,
    next_eps: float,
    next_out_bf16: bool,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = stream.shape
    shard_cols = hidden // tp_size
    lane_fast = _use_norm_fast()
    assert tokens % 16 == 0 and shard_cols % 128 == 0
    next_dtype = jnp.bfloat16 if next_out_bf16 else jnp.float32
    full_s_spec = pl.BlockSpec((tokens, hidden), lambda i: (0, 0))
    ag_spec = pl.BlockSpec((tp_size * tokens, shard_cols), lambda i: (0, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _tp_ag_res_norm_pallas_kernel,
            tokens=tokens,
            tp_size=tp_size,
            branch_eps=branch_eps,
            has_branch_gain=has_branch_gain,
            branch_scale_input=branch_scale_input,
            next_eps=next_eps,
            lane_fast=lane_fast,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), jnp.float32),
            jax.ShapeDtypeStruct((tokens, hidden), next_dtype),
        ),
        grid=(1,),
        in_specs=[full_s_spec, ag_spec, vec_spec, vec_spec, vec_spec, vec_spec],
        out_specs=(full_s_spec, full_s_spec),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(56 if tokens > 256 else 32) * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=f"muse_decode_tpag_res_norm_{tokens}x{hidden}_tp{tp_size}_{'bf16' if next_out_bf16 else 'f32'}{'_lf' if lane_fast else ''}",
    )(
        stream.astype(jnp.float32),
        gathered_2d,
        branch_gain.reshape(1, hidden).astype(jnp.float32),
        alpha.reshape(1, hidden).astype(jnp.float32),
        beta.reshape(1, hidden).astype(jnp.float32),
        next_gain.reshape(1, hidden).astype(jnp.float32),
    )


def get_muse_decode_router_topk_op(
    eplb_replicas: int = 0, ep_size: int = 8
) -> Callable:
    if eplb_replicas <= 0:
        if _ROUTER_OP:
            return _ROUTER_OP[0]
        with _LOCK:
            if not _ROUTER_OP:
                import jax  # noqa: F401
                from sglang_tpu.kernels.router_topk import router_topk_jax
                from sglang_tpu.pallas_jax_op import cpu_export_jax_op

                def _wrapped(
                    logits: jax.Array,
                    bias: jax.Array,
                    k: int,
                    eps: float,
                    scale: float,
                ) -> tuple[jax.Array, jax.Array]:
                    return router_topk_jax(logits, bias, k, True, eps, scale)

                def _fake(logits, bias, k, eps, scale):
                    t = logits.shape[0]
                    return (
                        logits.new_empty((t, k), dtype=torch.float32),
                        logits.new_empty((t, k), dtype=torch.int32),
                    )

                op = cpu_export_jax_op(
                    "sglang_tpu_pallas::muse_decode_router_topk", _wrapped
                )
                op.register_fake(_fake)
                _ROUTER_OP.append(op)
        return _ROUTER_OP[0]

    key = (int(eplb_replicas), int(ep_size))
    cached = _ROUTER_EPLB_OPS.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _ROUTER_EPLB_OPS.get(key)
        if cached is None:
            import jax  # noqa: F401
            from sglang_tpu.kernels.router_topk import router_topk_jax
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            r_cnt, ep_cnt = key
            base_local = 256 // ep_cnt

            def _wrapped_eplb(
                logits: jax.Array,
                bias: jax.Array,
                k: int,
                eps: float,
                scale: float,
            ) -> tuple[jax.Array, jax.Array]:
                return router_topk_jax(
                    logits,
                    bias,
                    k,
                    True,
                    eps,
                    scale,
                    eplb_replicas=r_cnt,
                    ep_size=ep_cnt,
                    num_base_local=base_local,
                )

            def _fake_eplb(logits, bias, k, eps, scale):
                t = logits.shape[0]
                return (
                    logits.new_empty((t, k), dtype=torch.float32),
                    logits.new_empty((t, k), dtype=torch.int32),
                )

            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_decode_router_topk_eplb_r{r_cnt}_ep{ep_cnt}",
                _wrapped_eplb,
            )
            op.register_fake(_fake_eplb)
            _ROUTER_EPLB_OPS[key] = op
            cached = op
    return cached


def get_muse_decode_rmsnorm_op() -> Callable:
    if _RMSNORM_OP:
        return _RMSNORM_OP[0]
    with _LOCK:
        if not _RMSNORM_OP:
            import jax  # noqa: F401
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                x: jax.Array,
                gain: jax.Array,
                eps: float,
                has_gain: bool,
                scale_input: bool,
                out_bf16: bool,
            ) -> jax.Array:
                return rmsnorm_pallas_jax(
                    x,
                    gain,
                    eps=eps,
                    has_gain=has_gain,
                    scale_input=scale_input,
                    out_bf16=out_bf16,
                )

            def _fake(x, gain, eps, has_gain, scale_input, out_bf16):
                dt = torch.bfloat16 if out_bf16 else torch.float32
                return x.new_empty(x.shape, dtype=dt)

            op = cpu_export_jax_op("sglang_tpu_pallas::muse_decode_rmsnorm", _wrapped)
            op.register_fake(_fake)
            _RMSNORM_OP.append(op)
    return _RMSNORM_OP[0]


def get_muse_decode_res_only_op() -> Callable:
    if _RES_ONLY_OP:
        return _RES_ONLY_OP[0]
    with _LOCK:
        if not _RES_ONLY_OP:
            import jax  # noqa: F401
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                stream: jax.Array,
                branch: jax.Array,
                branch_gain: jax.Array,
                alpha: jax.Array,
                beta: jax.Array,
                branch_eps: float,
                has_branch_gain: bool,
                branch_scale_input: bool,
            ) -> jax.Array:
                return residual_only_pallas_jax(
                    stream,
                    branch,
                    branch_gain,
                    alpha,
                    beta,
                    branch_eps=branch_eps,
                    has_branch_gain=has_branch_gain,
                    branch_scale_input=branch_scale_input,
                )

            def _fake(
                stream,
                branch,
                branch_gain,
                alpha,
                beta,
                branch_eps,
                has_branch_gain,
                branch_scale_input,
            ):
                return stream.new_empty(stream.shape, dtype=torch.float32)

            op = cpu_export_jax_op("sglang_tpu_pallas::muse_decode_res_only", _wrapped)
            op.register_fake(_fake)
            _RES_ONLY_OP.append(op)
    return _RES_ONLY_OP[0]


def get_muse_decode_res_norm_op() -> Callable:
    if _RES_NORM_OP:
        return _RES_NORM_OP[0]
    with _LOCK:
        if not _RES_NORM_OP:
            import jax  # noqa: F401
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                stream: jax.Array,
                branch: jax.Array,
                branch_gain: jax.Array,
                alpha: jax.Array,
                beta: jax.Array,
                next_gain: jax.Array,
                branch_eps: float,
                has_branch_gain: bool,
                branch_scale_input: bool,
                next_eps: float,
                next_out_bf16: bool,
            ) -> tuple[jax.Array, jax.Array]:
                return residual_and_next_norm_pallas_jax(
                    stream,
                    branch,
                    branch_gain,
                    alpha,
                    beta,
                    next_gain,
                    branch_eps=branch_eps,
                    has_branch_gain=has_branch_gain,
                    branch_scale_input=branch_scale_input,
                    next_eps=next_eps,
                    next_out_bf16=next_out_bf16,
                )

            def _fake(
                stream,
                branch,
                branch_gain,
                alpha,
                beta,
                next_gain,
                branch_eps,
                has_branch_gain,
                branch_scale_input,
                next_eps,
                next_out_bf16,
            ):
                dt = torch.bfloat16 if next_out_bf16 else torch.float32
                return (
                    stream.new_empty(stream.shape, dtype=torch.float32),
                    stream.new_empty(stream.shape, dtype=dt),
                )

            op = cpu_export_jax_op("sglang_tpu_pallas::muse_decode_res_norm", _wrapped)
            op.register_fake(_fake)
            _RES_NORM_OP.append(op)
    return _RES_NORM_OP[0]


def get_muse_decode_tp_ag_res_only_op() -> Callable:
    if _TP_AG_RES_ONLY_OP:
        return _TP_AG_RES_ONLY_OP[0]
    with _LOCK:
        if not _TP_AG_RES_ONLY_OP:
            import jax  # noqa: F401
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                stream: jax.Array,
                gathered_2d: jax.Array,
                branch_gain: jax.Array,
                alpha: jax.Array,
                beta: jax.Array,
                tp_size: int,
                branch_eps: float,
                has_branch_gain: bool,
                branch_scale_input: bool,
            ) -> jax.Array:
                return tp_ag_res_only_pallas_jax(
                    stream,
                    gathered_2d,
                    branch_gain,
                    alpha,
                    beta,
                    tp_size=tp_size,
                    branch_eps=branch_eps,
                    has_branch_gain=has_branch_gain,
                    branch_scale_input=branch_scale_input,
                )

            def _fake(
                stream,
                gathered_2d,
                branch_gain,
                alpha,
                beta,
                tp_size,
                branch_eps,
                has_branch_gain,
                branch_scale_input,
            ):
                return stream.new_empty(stream.shape, dtype=torch.float32)

            op = cpu_export_jax_op(
                "sglang_tpu_pallas::muse_decode_tp_ag_res_only", _wrapped
            )
            op.register_fake(_fake)
            _TP_AG_RES_ONLY_OP.append(op)
    return _TP_AG_RES_ONLY_OP[0]


def get_muse_decode_tp_ag_res_norm_op() -> Callable:
    if _TP_AG_RES_NORM_OP:
        return _TP_AG_RES_NORM_OP[0]
    with _LOCK:
        if not _TP_AG_RES_NORM_OP:
            import jax  # noqa: F401
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def _wrapped(
                stream: jax.Array,
                gathered_2d: jax.Array,
                branch_gain: jax.Array,
                alpha: jax.Array,
                beta: jax.Array,
                next_gain: jax.Array,
                tp_size: int,
                branch_eps: float,
                has_branch_gain: bool,
                branch_scale_input: bool,
                next_eps: float,
                next_out_bf16: bool,
            ) -> tuple[jax.Array, jax.Array]:
                return tp_ag_res_norm_pallas_jax(
                    stream,
                    gathered_2d,
                    branch_gain,
                    alpha,
                    beta,
                    next_gain,
                    tp_size=tp_size,
                    branch_eps=branch_eps,
                    has_branch_gain=has_branch_gain,
                    branch_scale_input=branch_scale_input,
                    next_eps=next_eps,
                    next_out_bf16=next_out_bf16,
                )

            def _fake(
                stream,
                gathered_2d,
                branch_gain,
                alpha,
                beta,
                next_gain,
                tp_size,
                branch_eps,
                has_branch_gain,
                branch_scale_input,
                next_eps,
                next_out_bf16,
            ):
                dt = torch.bfloat16 if next_out_bf16 else torch.float32
                return (
                    stream.new_empty(stream.shape, dtype=torch.float32),
                    stream.new_empty(stream.shape, dtype=dt),
                )

            op = cpu_export_jax_op(
                "sglang_tpu_pallas::muse_decode_tp_ag_res_norm", _wrapped
            )
            op.register_fake(_fake)
            _TP_AG_RES_NORM_OP.append(op)
    return _TP_AG_RES_NORM_OP[0]
