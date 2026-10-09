# SPDX-License-Identifier: Apache-2.0
"""Utility functions for ragged paged attention."""

import jax
import jax.numpy as jnp
from jax._src import dtypes

from sglang.srt.hardware_backend.tpu.pallas_export import get_torch_tpu_device_kind

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_F32 = jnp.finfo(jnp.float32)
# UE8M0 stores a power-of-two scale in one byte: the exponent plus 127.
UE8M0_EXPONENT_BIAS = 127


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = dtypes.itemsize_bits(dtype)
    return 32 // bits


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    try:
        kind = get_torch_tpu_device_kind()
    except (ImportError, RuntimeError):
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]

    # Extract version number after "TPU"
    # Supports formats: "TPU v5", "TPU v6", "TPU7x", "TPU6e", etc.
    if kind.startswith("TPU v"):
        # Format: "TPU v7x" or "TPU v5"
        version_str = kind[len("TPU v") :]
    elif kind.startswith("TPU"):
        # Format: "TPU7x" or "TPU5"
        version_str = kind[len("TPU") :]
    else:
        raise ValueError(f"Unexpected TPU device kind format: {kind}")

    # Extract the numeric part (first consecutive digits)
    version_digits = ""
    for char in version_str:
        if char.isdigit():
            version_digits += char
        else:
            break

    if not version_digits:
        raise ValueError(f"Could not extract TPU version from: {kind}")

    return int(version_digits)


def get_dtype_bitwidth(dtype):
    return (
        dtypes.bit_width(dtype)
        if hasattr(dtypes, "bit_width")
        else dtypes.itemsize_bits(dtype)
    )


def next_power_of_2(x):
    if x <= 0:
        return 1
    x = x - 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


def quantize_fp8_ue8m0(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize each vector along the last axis to FP8 with a power-of-two scale.

    Returns three arrays:

    * ``values``: ``float8_e4m3fn``, same shape as ``x``; ``values * scale``
      approximates ``x``.
    * ``scale``: ``float32[..., 1]``, always a power of two.
    * ``scale_byte``: ``uint8[..., 1]``, the same scale in UE8M0 format, so
      ``scale == 2 ** (scale_byte - 127)``.
    """
    x = x.astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    # We want the smallest power of two that is >= abs_max / FP8_MAX, so the
    # largest value still fits in FP8. frexp writes a number as
    # mantissa * 2 ** exponent with the mantissa in [0.5, 1). It works on the
    # float's bits, so it is exact; log2 and exp2 are not exact on TPU.
    mantissa, exponent = jnp.frexp(abs_max / _FP8_MAX)
    # A mantissa of exactly 0.5 means the number is already a power of two.
    exponent = jnp.where(mantissa == 0.5, exponent - 1, exponent)
    # Keep the scale a normal float32. An all-zero vector gets scale 1.
    exponent = jnp.clip(exponent, _F32.minexp, _F32.maxexp - 1)
    scale = jnp.ldexp(jnp.float32(1.0), exponent)
    values = jnp.clip(x / scale, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn)
    scale_byte = (exponent + UE8M0_EXPONENT_BIAS).astype(jnp.uint8)
    return values, scale, scale_byte
