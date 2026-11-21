# SPDX-License-Identifier: Apache-2.0
"""Host copies of the forward inputs the compile policy uploads at the bucket shape.

The policy pads host-staged inputs (positions, request slots, cache locations) on
the CPU and uploads them right before the attention backend builds its metadata,
which needs them on the host again. Reading the uploads back blocks on the device
queue. The policy records each host tensor next to the device tensor it created;
``host_copy`` returns the host tensor while the batch still holds that exact device
tensor, so a later rewrite of the field falls back to reading the device value.

One record, for the most recent batch: the backend consumes it in the same
forward, right after the upload, and releases it (``release``) so the tensors
are not kept alive past that forward.

Identity, not content, ties a record to the batch: nothing may write the host
tensor or the uploaded device tensor in place between the upload and the
metadata build (today both are fresh per forward; a reused staging buffer would
need to drop the record).
"""

import weakref
from typing import Any, Dict, Optional, Tuple

import torch

# Fields the DeepSeek-V4.1 attention metadata reads on the host.
MIRRORED_FIELDS = ("positions", "req_pool_indices", "out_cache_loc")

_record: Optional[
    Tuple["weakref.ref", Dict[str, Tuple[torch.Tensor, torch.Tensor]]]
] = None


def record_upload(
    forward_batch: Any, name: str, host: torch.Tensor, device: torch.Tensor
) -> None:
    """Remember ``host`` as the value of ``forward_batch.<name>`` = ``device``."""
    global _record
    if name not in MIRRORED_FIELDS or host.device.type != "cpu":
        return
    if _record is None or _record[0]() is not forward_batch:
        try:
            ref = weakref.ref(forward_batch)
        except TypeError:  # not weak-referenceable: the backend reads the device
            _record = None
            return
        _record = (ref, {})
    _record[1][name] = (host, device)


def host_copy(forward_batch: Any, name: str) -> Optional[torch.Tensor]:
    """The host tensor uploaded as ``forward_batch.<name>``, if the field is unchanged."""
    if _record is None or _record[0]() is not forward_batch:
        return None
    entry = _record[1].get(name)
    if entry is None or getattr(forward_batch, name, None) is not entry[1]:
        return None
    return entry[0]


def release(forward_batch: Any) -> None:
    """Drop the record once ``forward_batch``'s metadata has read it."""
    global _record
    if _record is not None and _record[0]() is forward_batch:
        _record = None


def clear() -> None:
    global _record
    _record = None
