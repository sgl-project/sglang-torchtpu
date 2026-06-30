"""TPU quantized matmul kernels."""

from threading import Lock
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import torch
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas

MXU_SIZE = 256


class _TunedValue(NamedTuple):
    batch_block_size: int
    out_block_size: int
    in_block_size: int
    n_lane_multiplier: int = 1


def _next_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., None],
) -> None:
    """Turn scalar JAX bools into Python bools to reduce runtime branching."""
    if conditions:
        arg = conditions[0]
        if isinstance(arg, bool):
            _unfold_args(conditions[1:], fn_conditions + (arg,), fn)
        else:
            assert arg.dtype == jnp.bool and arg.size == 1
            jax.lax.cond(
                arg,
                lambda: _unfold_args(conditions[1:], fn_conditions + (True,), fn),
                lambda: _unfold_args(conditions[1:], fn_conditions + (False,), fn),
            )
    else:
        fn(*fn_conditions)


def _get_max_min(target_dtype):
    if jnp.issubdtype(target_dtype, jnp.floating):
        return (
            jnp.finfo(target_dtype).max.astype(jnp.float32),
            jnp.finfo(target_dtype).min.astype(jnp.float32),
        )
    return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min


def _quantize_block(data: jax.Array, axis: int, target_dtype: jnp.dtype):
    abs_max = jnp.max(jnp.abs(data), axis=axis, keepdims=True)
    dtype_max, dtype_min = _get_max_min(target_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)
    if jnp.issubdtype(target_dtype, jnp.floating):
        data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    else:
        data_q = jnp.round(data / scale).astype(target_dtype)
    return data_q, scale


def _get_vmem_limit(
    *,
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
) -> int:
    x_size = batch_block_size * in_block_size * dtypes.itemsize_bits(x_dtype)
    x_abs_max_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)
    w_q_size = out_block_size * in_block_size * dtypes.itemsize_bits(w_q_dtype)
    w_scale_size = out_block_size * dtypes.itemsize_bits(scale_dtype)
    out_size = batch_block_size * out_block_size * dtypes.itemsize_bits(out_dtype)

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2
    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0
    vmem_in_out += x_abs_max_size if n_batch > 1 else 0
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0
    vmem_in_out += w_scale_size if n_out > 1 else 0
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0

    acc_size = batch_block_size * out_block_size * dtypes.itemsize_bits(acc_dtype)
    x_q_size = batch_block_size * in_block_size * dtypes.itemsize_bits(x_q_dtype)
    x_scale_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2

    # TPU v7 has a smaller usable VMEM budget than v6. 48 MiB is the
    # torchtpu-vllm v7 limit and is conservative on v6 as well.
    return min((vmem_in_out + vmem_scratch) // 8, 48 * 1024 * 1024)


def _get_tuned_value(
    *,
    n_batch: int,
    n_out: int,
    n_in: int,
    block_size: int,
) -> _TunedValue:
    # Keep the custom-op boundary compact first. More exhaustive shape tuning can
    # be ported from torchtpu-vllm after correctness/perf are established.
    batch_block_size = 16 if n_batch <= 16 else 128
    out_block_size = 1024 if n_out % 1024 == 0 else 512 if n_out % 512 == 0 else 256
    in_block_size = block_size
    return _TunedValue(batch_block_size, out_block_size, in_block_size)


@jax.jit(static_argnames=["block_size", "x_q_dtype"])
def _blockwise_quantized_matmul_kernel(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    *,
    block_size: int,
    x_q_dtype: jnp.dtype,
) -> jax.Array:
    """Blockwise W8A8 matmul.

    Args:
        x: [M, K] bf16 activations.
        w_q: [N, K] FP8 weights.
        w_scale: [K / block_size, 1, N] weight scales.
    """

    orig_m, orig_k = x.shape
    orig_n, weight_k = w_q.shape
    if orig_k != weight_k:
        raise ValueError(f"Input K {orig_k} must match weight K {weight_k}.")
    if w_scale.ndim != 3 or w_scale.shape[1] != 1 or w_scale.shape[2] != orig_n:
        raise ValueError(
            "w_scale must have shape [K / block_size, 1, N]; "
            f"got {w_scale.shape=} for {orig_n=}."
        )
    if orig_k % block_size != 0:
        raise ValueError(f"{orig_k=} must be divisible by {block_size=}.")
    if w_scale.shape[0] != orig_k // block_size:
        raise ValueError(
            f"w_scale block dim {w_scale.shape[0]} must match "
            f"{orig_k // block_size=}."
        )

    tuned = _get_tuned_value(
        n_batch=orig_m,
        n_out=orig_n,
        n_in=orig_k,
        block_size=block_size,
    )
    batch_block_size = tuned.batch_block_size
    out_block_size = tuned.out_block_size
    in_block_size = tuned.in_block_size
    n_lane_multiplier = tuned.n_lane_multiplier

    if in_block_size % block_size != 0:
        raise ValueError(f"{in_block_size=} must be divisible by {block_size=}.")

    padded_m = _next_multiple(orig_m, batch_block_size)
    if orig_m < padded_m:
        x = jnp.pad(x, ((0, padded_m - orig_m), (0, 0)))

    padded_n = _next_multiple(orig_n, out_block_size)
    if orig_n < padded_n:
        w_q = jnp.pad(w_q, ((0, padded_n - orig_n), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, 0), (0, 0), (0, padded_n - orig_n)))

    if orig_k % in_block_size != 0:
        raise ValueError(f"{orig_k=} must be divisible by {in_block_size=}.")

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = padded_m // batch_block_size
    n_out = padded_n // out_block_size
    n_in = orig_k // in_block_size

    save_acc = n_in > 1
    save_x_q = n_in == 1 and n_out > 1
    acc_dtype = jnp.bfloat16
    steps_k = in_block_size // block_size
    compute_tile_n = MXU_SIZE * n_lane_multiplier
    if out_block_size % compute_tile_n != 0:
        raise ValueError(f"{out_block_size=} must be divisible by {compute_tile_n=}.")
    steps_n = out_block_size // compute_tile_n

    vmem_limit_bytes = _get_vmem_limit(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        x_dtype=x.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q.dtype,
        scale_dtype=jnp.float32,
        out_dtype=x.dtype,
        acc_dtype=acc_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
    )

    def kernel(lhs_ref, rhs_ref, scale_ref, out_ref, acc_scratch):
        pid_k = pl.program_id(2)
        is_first_step = pid_k == 0
        is_last_step = pid_k == (orig_k // in_block_size - 1)

        def accum(is_first_step, is_last_step):
            accumulators = [None] * steps_n
            for i in range(steps_k):
                k_start = i * block_size
                k_end = k_start + block_size
                lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
                lhs_q, lhs_scale = _quantize_block(lhs_sub, 1, x_q_dtype)
                lhs_scale = lhs_scale.astype(acc_dtype)

                rhs_q_full = rhs_ref[:, k_start:k_end]
                rhs_scale_full = scale_ref[i, :, :].astype(acc_dtype)

                for j in range(steps_n):
                    n_start = j * compute_tile_n
                    n_end = n_start + compute_tile_n
                    rhs_q_slice = rhs_q_full[n_start:n_end, :]
                    rhs_scale_slice = rhs_scale_full[:, n_start:n_end]
                    dot_res = jax.lax.dot_general(
                        lhs_q,
                        rhs_q_slice,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    res = dot_res.astype(acc_dtype)
                    res = res * lhs_scale
                    res = res * rhs_scale_slice
                    if i == 0:
                        accumulators[j] = res
                    else:
                        accumulators[j] += res

            acc_block = jnp.concatenate(accumulators, axis=1)
            if not is_first_step:
                acc_block += acc_scratch[...]

            if is_last_step:
                out_ref[...] = acc_block.astype(out_ref.dtype)
            else:
                acc_scratch[...] = acc_block

        _unfold_args((is_first_step, is_last_step), (), accum)

    pallas_kernel = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (batch_block_size, in_block_size),
                    lambda b, o, i: (b, i),
                    memory_space=pltpu.VMEM,
                ),
                pl.BlockSpec(
                    (out_block_size, in_block_size),
                    lambda b, o, i: (o, i),
                    memory_space=pltpu.VMEM,
                ),
                pl.BlockSpec(
                    (steps_k, 1, out_block_size),
                    lambda b, o, i: (i, 0, o),
                    memory_space=pltpu.VMEM,
                ),
            ],
            out_specs=pl.BlockSpec(
                (batch_block_size, out_block_size), lambda b, o, i: (b, o)
            ),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), jnp.bfloat16)
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_m, padded_n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    with jax.named_scope("sglang_tpu_block_fp8_matmul"):
        out = pallas_kernel(x, w_q, w_scale)
    return out[:orig_m, :orig_n]


def _get_x_q_dtype(w_q_dtype: jnp.dtype) -> jnp.dtype:
    if jnp.issubdtype(w_q_dtype, jnp.integer):
        return jnp.int8
    if jnp.issubdtype(w_q_dtype, jnp.floating):
        return jnp.float8_e4m3fn
    raise ValueError(f"Unsupported quantized dtype: {w_q_dtype}")


def _blockwise_quantized_matmul_jax(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
) -> jax.Array:
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(
            f"Input hidden dim {x.shape[1]} must match weight hidden dim {w_q.shape[1]}."
        )
    if w_scale.ndim != 3 or w_scale.shape[1] != 1 or w_scale.shape[2] != w_q.shape[0]:
        raise ValueError(
            "Blockwise weight scale must have shape [K / block_size, 1, N]; "
            f"got {w_scale.shape=} for weight shape {w_q.shape=}."
        )
    if x.shape[1] % w_scale.shape[0] != 0:
        raise ValueError(
            f"Input hidden dim {x.shape[1]} must be divisible by scale blocks "
            f"{w_scale.shape[0]}."
        )
    return _blockwise_quantized_matmul_kernel(
        x,
        w_q,
        w_scale,
        block_size=x.shape[1] // w_scale.shape[0],
        x_q_dtype=_get_x_q_dtype(w_q.dtype),
    )


_BLOCK_FP8_OP: Callable | None = None
_BLOCK_FP8_OP_LOCK = Lock()


def _get_block_fp8_matmul_op() -> Callable:
    global _BLOCK_FP8_OP
    if _BLOCK_FP8_OP is not None:
        return _BLOCK_FP8_OP

    with _BLOCK_FP8_OP_LOCK:
        if _BLOCK_FP8_OP is not None:
            return _BLOCK_FP8_OP

        op = pallas.jax_op(
            "sglang_tpu_pallas::block_fp8_quantized_matmul",
            _blockwise_quantized_matmul_jax,
        )

        def _fake_block_fp8_matmul(
            x: torch.Tensor,
            w_q: torch.Tensor,
            w_scale: torch.Tensor,
        ):
            del w_scale
            return torch.empty(
                x.shape[0],
                w_q.shape[0],
                dtype=x.dtype,
                device=x.device,
            )

        op.register_fake(_fake_block_fp8_matmul)
        _BLOCK_FP8_OP = op
        return op


def blockwise_quantized_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run true block-FP8 W8A8 matmul through one TPU Pallas custom op.

    ``weight`` is ``[N, K]`` and ``weight_scale`` is ``[K_blocks, 1, N]``.
    """

    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input hidden dim {x.shape[-1]} must match weight hidden dim "
            f"{weight.shape[-1]}."
        )
    out_shape = (*x.shape[:-1], weight.shape[0])
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _get_block_fp8_matmul_op()(x_2d, weight, weight_scale)
    return out_2d.reshape(out_shape)


def register_tpu_block_fp8_matmul_op() -> None:
    """Register the block-FP8 Pallas op before torch.compile traces a model."""

    _get_block_fp8_matmul_op()
