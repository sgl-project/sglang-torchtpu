"""Host-to-TPU uploads that keep the host ahead of the TPU.

On torch_tpu a blocking H2D copy (``cpu.to(tpu)``, ``torch.tensor(x, device=tpu)``,
``dst.copy_(cpu)``) returns only after every program already queued on the TPU
has finished. In a decode round that stalls the host for the rest of the queue
(a whole target verify, for an upload right after its launch), and the TPU then
idles while the host catches up. Hot-path code uploads through these helpers.

A non-blocking upload still runs in queue order, so later programs read the
uploaded values; torch_tpu copies the host bytes during the call, so the source
may be changed or freed afterwards. test/registered/tpu/unit/test_host_upload.py
pins both properties.
"""

import torch

from sglang_tpu.host_metadata_cache import HostMetadataCache


def upload(t: torch.Tensor, device) -> torch.Tensor:
    """``t.to(device)`` without waiting for queued TPU work.

    A host tensor is queued for copy and returned at once; a tensor already on
    ``device`` comes back unchanged, as with ``.to``.
    """
    return t.to(device, non_blocking=True)


_scalars = HostMetadataCache()


def device_scalar(value: int, dtype: torch.dtype, device) -> torch.Tensor:
    """A 0-d tensor on ``device`` holding ``value``, uploaded once per
    (value, dtype, device).

    Forwards still in flight share the returned tensor: callers must not
    write to it.
    """
    value = int(value)
    return _scalars.get_by_layout(
        ("scalar", value, dtype),
        lambda: torch.tensor(value, dtype=dtype),
        device,
    )
