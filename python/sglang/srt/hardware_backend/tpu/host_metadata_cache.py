"""Host-to-device cache for read-only attention metadata."""

from typing import Callable, Hashable

import torch


class HostMetadataCache:
    """Reuse device copies of CPU attention metadata across forwards.

    Returned tensors can be shared by forwards still in flight; callers must
    treat them as read-only.
    """

    def __init__(self):
        self._by_layout = {}
        self._by_value = {}

    def get_by_layout(
        self, key: Hashable, build: Callable[[], torch.Tensor], device
    ) -> torch.Tensor:
        """Build once per (key, device); reuse on later calls.

        The key must fully determine the tensor contents, as with batch offsets.
        """
        key = (key, str(device))
        cached = self._by_layout.get(key)
        if cached is not None:
            return cached[1]
        value = build()
        assert value.device.type == "cpu"
        # Keep the CPU source alive for the non-blocking copy.
        resident = value.to(device, non_blocking=True)
        self._by_layout[key] = value, resident
        return resident

    def get_by_value(self, name: str, value: torch.Tensor, device) -> torch.Tensor:
        """Keep the latest upload per name, as with request page tables.

        Reuse it only when the device, dtype, shape, and CPU values all match.
        """
        assert value.device.type == "cpu"
        cached = self._by_value.get(name)
        key = str(device)
        if cached is not None:
            previous, device_key, resident = cached
            if (
                device_key == key
                and previous.dtype == value.dtype
                and previous.shape == value.shape
                and torch.equal(previous, value)
            ):
                return resident
        # Keep our own CPU snapshot: callers may subsequently mutate their input.
        snapshot = value.clone()
        resident = snapshot.to(device, non_blocking=True)
        self._by_value[name] = snapshot, key, resident
        return resident
