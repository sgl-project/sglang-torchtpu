import threading
from contextlib import contextmanager

_TPU_INFO_PATCH_LOCK = threading.RLock()
_PALLAS_DEVICE_KIND_BY_TORCHTPU = {
    "TPU v6e": "TPU v6 lite",
    "TPU v7": "TPU7x",
}


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
