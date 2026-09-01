import os
import threading
from contextlib import contextmanager

_TPU_INFO_PATCH_LOCK = threading.RLock()
_PALLAS_DEVICE_KIND_BY_TORCHTPU = {
    "TPU v6e": "TPU v6 lite",
    "TPU v7": "TPU7x",
}


class _ExportMockClient:
    """Mock PJRT client for abstract AOT compilation devices."""

    platform: str = "tpu"
    platform_version: str = "TPU v7"
    runtime_type: str = "tpu"

    def process_index(self) -> int:
        return 0

    def device_count(self) -> int:
        return int(os.environ.get("WORLD_SIZE", "1"))

    def devices(self) -> list:
        return []

    def local_devices(self) -> list:
        return []


_MOCK_CLIENT = _ExportMockClient()


class ExportAbstractDevice:
    """Compile-only abstract device compatible with JAX mesh/sharding property inspection."""

    def __init__(
        self,
        device_kind: str,
        num_cores: int = 1,
        platform: str = "tpu",
        id: int = 0,
        process_index: int = 0,
    ):
        self.device_kind = device_kind
        self.num_cores = num_cores
        self.platform = platform
        self.id = id
        self.process_index = process_index
        self.client = _MOCK_CLIENT

    def __repr__(self) -> str:
        return f"AbstractDevice(id={self.id}, device_kind='{self.device_kind}', platform='{self.platform}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExportAbstractDevice):
            return False
        return (
            self.id == other.id
            and self.device_kind == other.device_kind
            and self.platform == other.platform
        )

    def __hash__(self) -> int:
        return hash((self.id, self.device_kind, self.platform))


def get_torchtpu_export_devices(world_size: int | None = None):
    """Return the ordered abstract TPU devices for global partitions without calling jax.devices()."""
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_kind = get_pallas_tpu_device_kind()
    abstract_devices = [
        ExportAbstractDevice(
            device_kind=device_kind,
            num_cores=1,
            platform="tpu",
            id=i,
            process_index=0,
        )
        for i in range(world_size)
    ]
    return tuple(abstract_devices)


def get_torch_tpu_device_kind() -> str:
    """Return physical TPU hardware metadata."""
    from torch_tpu._internal.utils.hardware import get_tpu_device_name

    return get_tpu_device_name()


def get_pallas_tpu_device_kind() -> str:
    """Return the physical device spelling expected by JAX Pallas."""
    device_kind = get_torch_tpu_device_kind()
    return _PALLAS_DEVICE_KIND_BY_TORCHTPU.get(device_kind, device_kind)


@contextmanager
def tpu_info_for_cpu_export():
    """Ensure Pallas kernels receive physical TPU7x hardware metadata during export."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    with _TPU_INFO_PATCH_LOCK:
        original_public = pltpu.get_tpu_info
        original_private = tpu_info.get_tpu_info
        original_device_kind = getattr(tpu_info, "get_device_kind", None)
        original_is_tpu_device = getattr(tpu_info, "is_tpu_device", None)
        device_name = get_pallas_tpu_device_kind()
        chip_version = tpu_info.chip_version_from_device_kind(device_name)
        if chip_version is not None:
            info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
            pltpu.get_tpu_info = lambda: info
            tpu_info.get_tpu_info = lambda: info
            tpu_info.get_device_kind = lambda: device_name
            tpu_info.is_tpu_device = lambda: True
        try:
            yield
        finally:
            pltpu.get_tpu_info = original_public
            tpu_info.get_tpu_info = original_private
            if original_device_kind is not None:
                tpu_info.get_device_kind = original_device_kind
            if original_is_tpu_device is not None:
                tpu_info.is_tpu_device = original_is_tpu_device
