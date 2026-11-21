# SPDX-License-Identifier: Apache-2.0
"""Shared by the true on-policy kernels: op registration, row block sizing, flag refusal."""

from torch_tpu._internal import pallas as _torch_pallas


def register(name, fn):
    return _torch_pallas.jax_op(f"sglang_tpu_true_on_policy::{name}", fn)


def row_block(m, cap, floor):
    """Rows per block: the power of two covering m rows, within [floor, cap]."""
    return min(cap, max(floor, 1 << (m - 1).bit_length()))


def save_inputs(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def refuse_flags(flags, what):
    from sglang_tpu.envs import envs

    if on := [name for name in flags if getattr(envs, name).get()]:
        raise ValueError(
            f"True on-policy {what} needs {', '.join(on)} unset: they change the rounding"
        )
