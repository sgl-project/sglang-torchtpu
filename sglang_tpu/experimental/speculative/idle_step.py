# SPDX-License-Identifier: Apache-2.0
"""Marks an idle DP rank's all-padding spec step (TpuSpecWorker._forward_idle_step),
whose forward inputs repeat per batch bucket, so an attention backend may reuse
the metadata it built for the same shape (SGLANG_TPU_SPEC_IDLE_INPUT_CACHE)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

_active = False


@contextlib.contextmanager
def idle_step() -> Iterator[None]:
    global _active
    _active = True
    try:
        yield
    finally:
        _active = False


def in_idle_step() -> bool:
    return _active
