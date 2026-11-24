# SPDX-License-Identifier: Apache-2.0
"""Hand each temporal chunk of a VAE decode to a consumer while the decode runs.

A VAE that decodes video in temporal chunks calls :func:`emit_decoded_chunk`
with every chunk's frames; within :func:`stream_decoded_chunks` a consumer (the
output rank's streaming video save) receives them, so encoding chunk k can
overlap decoding chunk k + 1. Without a consumer the call does nothing.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Callable, Optional

import torch

_sink: contextvars.ContextVar[Optional[Callable[[torch.Tensor], None]]] = (
    contextvars.ContextVar("decoded_chunk_sink", default=None)
)


@contextmanager
def stream_decoded_chunks(sink: Optional[Callable[[torch.Tensor], None]]):
    """Route chunks to ``sink`` within the block; None routes them nowhere."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit_decoded_chunk(
    chunk: torch.Tensor,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Pass one decoded chunk ([B, C, T, H, W], before the decode's final
    clamp) to the active consumer; ``transform`` (e.g. an unpatchify the decode
    applies to the whole video) runs only when there is one."""
    sink = _sink.get()
    if sink is not None:
        sink(chunk if transform is None else transform(chunk))
