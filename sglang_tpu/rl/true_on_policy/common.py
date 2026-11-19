# SPDX-License-Identifier: Apache-2.0
"""What the true on-policy kernels share: op registration, row block sizing and the plain backward set-up."""

from torch_tpu._internal import pallas as _torch_pallas


def register(name, fn):
    # jax_op derives the schema and the fake from the function's export; registered at import, outside any trace
    return _torch_pallas.jax_op(f"sglang_tpu_true_on_policy::{name}", fn)


# Row and column blocks are sized for speed only (measured on v7x): large ones keep HBM traffic and grid steps down
# for a packed trainer batch, and row_block shrinks the row block for a small decode batch.
def row_block(m, cap, floor):
    """Rows per block: the power of two covering m rows, within [floor, cap]."""
    return min(cap, max(floor, 1 << (m - 1).bit_length()))


def save_inputs(ctx, inputs, output):
    ctx.save_for_backward(*inputs)
