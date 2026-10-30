# Adapted from vllm-project/tpu-inference, the megablox gmm_v2 kernel's
# activation/alignment helpers (``swigluoai`` and ``align_to``):
#   https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/megablox/gmm_v2.py
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Activation and shape-alignment helpers used by the Pallas fused-MoE
kernel (``gmm_v2.py``).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""

    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Apply a fused activation function to the accumulator.

    The input ``acc`` is expected to contain concatenated results for
    both the 'gate' and 'up' projections (last dim = ``2 * tile_n``).

    Returns ``acc`` unchanged when ``fuse_act`` is None. Raises
    ``NotImplementedError`` for unsupported activations. Supported:
    ``silu``, ``gelu``, ``swigluoai``.
    """

    if fuse_act is None:
        return acc

    acc_gate, acc_up = jnp.split(acc, 2, -1)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "swigluoai":
            return swigluoai(acc_gate, acc_up)
        case _:
            raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    """Round ``x`` up to the next multiple of ``a``."""
    return pl.cdiv(x, a) * a
