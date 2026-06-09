"""TPU FP8 dense linear helpers."""

# NOTE: do *not* add ``from __future__ import annotations`` here.
# ``torch_tpu._internal.pallas.jax_op`` validates annotations at registration
# time and requires real ``jax.Array`` objects, not stringified annotations.

from threading import Lock
from typing import Callable

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


def _quantize_tensor(x: jax.Array, dtype: jnp.dtype, dim: int = -1):
    amax = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
    dtype_max = float(jnp.finfo(dtype).max)
    scale = jnp.maximum(amax / dtype_max, 1e-12)
    scale_inv = jnp.nan_to_num(1.0 / scale, dtype_max)
    return (x * scale_inv).astype(dtype), scale.astype(jnp.float32)


def _fp8_linear_jax(
    x: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
) -> jax.Array:
    if x.shape[1] != weight.shape[0]:
        raise ValueError(
            f"Input hidden dim {x.shape[1]} must match weight hidden dim "
            f"{weight.shape[0]}."
        )
    if weight_scale.ndim > 1:
        raise ValueError(
            "TPU dense FP8 linear only supports per-tensor or per-output-channel "
            f"weight scales; got {weight_scale.shape=}."
        )

    x_q, x_scale = _quantize_tensor(x, weight.dtype, dim=1)
    out = jax.lax.dot_general(
        x_q,
        weight,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    out *= x_scale
    out *= jnp.reshape(weight_scale.astype(jnp.float32), (1, -1))
    return out.astype(x.dtype)


_fp8_linear_op: Callable | None = None
_fp8_linear_op_lock = Lock()


def _get_fp8_linear_op() -> Callable:
    global _fp8_linear_op
    if _fp8_linear_op is not None:
        return _fp8_linear_op

    with _fp8_linear_op_lock:
        if _fp8_linear_op is not None:
            return _fp8_linear_op

        op = pallas.jax_op("pallas::sglang_tpu_fp8_linear", _fp8_linear_jax)

        def _fake_fp8_linear(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ):
            del weight_scale
            return torch.empty(
                x.shape[0],
                weight.shape[1],
                dtype=x.dtype,
                device=x.device,
            )

        op.register_fake(_fake_fp8_linear)
        _fp8_linear_op = op
        return op


def prebuild_tpu_fp8_linear_op() -> None:
    """Register the TPU FP8 linear custom op before torch.compile tracing."""

    _get_fp8_linear_op()


def apply_tpu_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply dense FP8 linear on TPU.

    ``weight`` follows SGLang's post-load dense-linear layout ``[K, N]``.
    ``weight_scale`` may be scalar or per-output-channel ``[N]``.
    """

    if weight_scale.ndim > 1:
        raise NotImplementedError(
            "TPU dense FP8 linear only supports per-tensor or per-output-channel "
            f"weight scales; got {tuple(weight_scale.shape)}."
        )
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"Input hidden dim {x.shape[-1]} must match weight hidden dim "
            f"{weight.shape[0]}."
        )

    out_shape = (*x.shape[:-1], weight.shape[1])
    x_2d = x.reshape(-1, x.shape[-1])
    op = _fp8_linear_op
    if op is None:
        op = _get_fp8_linear_op()
    out = op(x_2d, weight, weight_scale.reshape(-1))
    out = out.reshape(out_shape)
    if bias is not None:
        out = out + bias
    return out
