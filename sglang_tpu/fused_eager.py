# SPDX-License-Identifier: Apache-2.0
"""Run a short sequence of TorchTPU eager ops as one program.

Under the serving eager mode (``SGLANG_TPU_EAGER_MODE=DEFER_NEVER``) every op
is its own executable. PJRT keeps at most 32 executions in flight, so a decode
step that issues more small ops than that between two compiled graphs blocks
TorchTPU's dispatch thread until the running graph ends; the next graph's
launch then waits behind it on the host, and the TensorCore idles between
steps. ``fused_eager_ops`` defers the ops of a region (``DEFER_AND_FUSE``) and
``submit`` launches them as one program, asynchronously.

A region must hold only device compute on its own tensors: no collective (an
in-place collective does not see deferred inputs, see hooks/spec_tp_sync.py),
no host-to-device upload (it would enter the program), no host read. To keep
the values of the per-op programs, hold only elementwise ops, data movement,
random draws and order-free reductions (max, argmax): XLA may reduce a sum in
another order inside a fused program (a softmax over 100 rows of a ~200K
vocabulary differed in its last bits; softmax and log_softmax stay outside).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch


def _defers_now() -> bool:
    """Whether a region may switch the eager mode: not while a graph is traced
    (direct_compile and Dynamo trace the same code), and only from the
    per-op mode (a server that already defers keeps its mode)."""
    if torch.compiler.is_compiling():
        return False
    try:
        from torch_tpu._internal import execution_mode
    except ImportError:  # a host without TorchTPU runs every op on its device
        return False
    return execution_mode.eager_mode == execution_mode.EagerMode.DEFER_NEVER


@contextlib.contextmanager
def fused_eager_ops(enabled: bool = True) -> Iterator[bool]:
    """Defer the TorchTPU eager ops of the block. Yields whether it defers;
    the caller then passes the block's results to ``submit``."""
    if not (enabled and _defers_now()):
        yield False
        return
    from torch_tpu._internal import execution_mode

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
        yield True


def submit(tensors: Iterable[object]) -> None:
    """Launch the deferred programs that produce ``tensors`` (non-tensors and
    host tensors are skipped), without waiting for them."""
    pending = [
        tensor
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "tpu"
    ]
    if pending:
        from torch_tpu._internal.sync.sync import synchronize

        synchronize(pending, wait=False)
