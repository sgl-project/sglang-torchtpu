# SPDX-License-Identifier: Apache-2.0
"""One upload per dtype for a forward's host-built attention metadata.

Each non-blocking host-to-device copy costs the host about 0.1 ms to issue, and
the DSV4.1 metadata made ~16 of them per forward, on the host path between two
programs. ``pack_upload`` concatenates the host tensors of each dtype, uploads
one buffer per dtype and returns the fields as views of those buffers.
"""

from typing import Dict, List, Sequence

import torch


def pack_upload(
    tensors: Sequence[torch.Tensor], *, device: torch.device
) -> List[torch.Tensor]:
    """``tensors`` on ``device``, in order, as views of one upload per dtype."""
    groups: Dict[torch.dtype, List[int]] = {}
    for i, t in enumerate(tensors):
        groups.setdefault(t.dtype, []).append(i)
    out: List[torch.Tensor] = [None] * len(tensors)  # type: ignore[list-item]
    for indices in groups.values():
        host = torch.cat([tensors[i].reshape(-1) for i in indices])
        buffer = host.to(device=device, non_blocking=True)
        offset = 0
        for i in indices:
            numel = tensors[i].numel()
            out[i] = buffer[offset : offset + numel].view(tensors[i].shape)
            offset += numel
    return out
