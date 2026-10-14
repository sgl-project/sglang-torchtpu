from __future__ import annotations

from enum import Enum


class TensorSourceOfTruth(Enum):
    """Where a batch keeps its host-born tensors (req_pool_indices, sampling params).

    HOST_PADDED makes the CPU copy authoritative and pads per-request rows to the
    batch bucket, recording the real size, so filter/merge and consumers only see
    bucket shapes on device.
    """

    DEVICE = "device"
    HOST_PADDED = "host_padded"
