"""Host pools in a prefill stage's block layout, for a decode that is not a
pipeline (stage_layout_store.py).

Why: raiden's host reads move whole blocks between two managers with the same
     block layout, and a manager takes its layout from the device tensors it
     is built over. The pinned tpu-sync ships no host-only manager.
How: the decode builds an ordinary manager over placeholder device tensors,
     one page each, shaped like the stage's tensors, and uses only its host
     blocks. The placeholders are never read or written; they cost one page of
     HBM per pool.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, List

import torch

from sglang_tpu.disaggregation.decode_dram_offload.stage_layout_store import (
    StagePoolLayout,
)


def stage_pool_layout(
    host_pool: Any, *, stage: int, stages: int, window: bool = False
) -> StagePoolLayout:
    """The block layout of a prefill stage's HiCache host pool, as its pushes
    describe it to the decode."""
    tensors = [t for layer in host_pool.transfer._device_tensors for t in layer]
    return StagePoolLayout(
        stage=stage,
        stages=stages,
        tensors=tuple(
            (str(t.dtype).removeprefix("torch."), tuple(int(x) for x in t.shape[1:]))
            for t in tensors
        ),
        page_bytes=sum(int(t.nbytes) // int(t.shape[0]) for t in tensors),
        window=window,
    )


class _PlaceholderPool:
    """The device-pool surface RaidenKVTransfer reads."""

    def __init__(self, buffers: List[torch.Tensor]) -> None:
        self._buffers = buffers

    def page_buffers(self) -> List[torch.Tensor]:
        return self._buffers


# The transfer manager sets this for the engines it builds later in the
# process (raiden/conn.py); a manager built with it refuses host reads
# ("host transfers require num_slots == 0 and no dynamic host staging").
_DYNAMIC_HOST_STAGING = "TPU_RAIDEN_DYNAMIC_HOST_STAGING"


@contextlib.contextmanager
def _static_host_pool() -> Iterator[None]:
    """Build a manager whose host blocks are a plain pool, as HiCache's own
    host pool is when it is built before the transfer manager."""
    previous = os.environ.get(_DYNAMIC_HOST_STAGING)
    os.environ[_DYNAMIC_HOST_STAGING] = "0"
    try:
        yield
    finally:
        if previous is None:
            del os.environ[_DYNAMIC_HOST_STAGING]
        else:
            os.environ[_DYNAMIC_HOST_STAGING] = previous


def build_stage_pool(layout: StagePoolLayout, pages: int, *, device: Any) -> Any:
    """A host pool of ``pages`` pages in ``layout``, with the host read methods
    of RaidenKVTransfer. Raises if the layout is not the one it reports."""
    from sglang_tpu.raiden_transfer import RaidenKVTransfer

    buffers = [
        torch.zeros((1, *shape), dtype=getattr(torch, dtype), device=device)
        for dtype, shape in layout.tensors
    ]
    page_bytes = sum(int(t.nbytes) for t in buffers)
    if page_bytes != layout.page_bytes:
        raise ValueError(
            f"stage {layout.stage} pages are {layout.page_bytes} bytes, its "
            f"tensors describe {page_bytes}"
        )
    with _static_host_pool():
        return RaidenKVTransfer(_PlaceholderPool(buffers), num_host_pages=pages)
