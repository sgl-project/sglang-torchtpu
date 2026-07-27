"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

import functools
from typing import Any, List

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_tpu


def compiled_op(fn):
    """Lazily ``torch.compile`` a pure tensor helper into one fused dispatch.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe.
    """
    compiled_holder: List[Any] = []

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
            return fn(*args, **kwargs)
        if not compiled_holder:
            compiled_holder.append(
                torch.compile(fn, backend="tpu", fullgraph=False, dynamic=False)
            )
        return compiled_holder[0](*args, **kwargs)

    return wrapper
