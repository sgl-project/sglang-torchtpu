# SPDX-License-Identifier: Apache-2.0
"""Small helpers the DeepSeek-V4.1 kernels share, so the pack imports no
framework module from a kernel."""

import jax.numpy as jnp
import torch

FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def dtype_max(dtype: jnp.dtype) -> float:
    info = jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)
    return float(info.max)


def largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of `x` that is <= `cap`."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1
