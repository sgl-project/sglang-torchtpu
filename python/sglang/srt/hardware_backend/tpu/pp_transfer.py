"""Host-channel tensor-dict hand-off for TPU, under `parallel_state`.

torch_tpu has no device-to-device send/recv, so a tensor dict crossing ranks
travels through host memory: the sender copies each device tensor to the host,
the copy goes out on the group's gloo cpu_group, and the receiver lands it in a
host buffer and uploads it to the device again.

This is not the pipeline activation path. Stage activations cross over ICI as a
`jax.lax.ppermute` ring in `pp_wave.py`; what still comes through here is the
last stage's output dict and control traffic, at host-TCP speed.

The device-to-host copy is also the only execution fence torch_tpu offers: it
returns once the program that produced the tensor has finished.
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

_pinned_alloc_failed = False


def to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous host copy of a device tensor (host tensors pass
    through unchanged). Blocks until the tensor's producing program finished."""
    if tensor.is_cpu:
        return tensor
    return tensor.contiguous().cpu()


def alloc_recv_buffer(size, dtype: torch.dtype) -> torch.Tensor:
    """Allocate the host buffer a stage receives a device tensor into.

    Pinned host memory lets the following upload take the DMA path; when the
    pinned allocator is unavailable the buffer is plain host memory and the
    upload is a synchronous copy.
    """
    global _pinned_alloc_failed
    size = tuple(size)
    if not _pinned_alloc_failed and _numel(size) > 0:
        try:
            return torch.empty(size, dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError as error:
            _pinned_alloc_failed = True
            logger.warning(
                "Pinned host allocation for the pipeline hand-off failed (%s); "
                "receiving into unpinned host memory from now on.",
                error,
            )
    return torch.empty(size, dtype=dtype, device="cpu")


def to_device(tensor: torch.Tensor, device_type: str) -> torch.Tensor:
    """Upload a received host tensor to the device type the sender recorded."""
    return tensor.to(device_type, non_blocking=True)


def _numel(size) -> int:
    count = 1
    for dim in size:
        count *= int(dim)
    return count
