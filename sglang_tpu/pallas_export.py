import inspect
import threading
from collections.abc import Callable
from contextlib import contextmanager

_TPU_INFO_PATCH_LOCK = threading.RLock()
_PALLAS_DEVICE_KIND_BY_TORCHTPU = {
    "TPU v6e": "TPU v6 lite",
    "TPU v7": "TPU7x",
}


def argnums(function: Callable, *names: str) -> tuple[int, ...]:
    """Return the positions of the named parameters of ``function``.

    ``custom_jax_kernel`` only accepts argument positions. Looking them up by
    name keeps the call site readable, and the numbers cannot go stale when
    the signature changes.
    """
    parameters = list(inspect.signature(function).parameters)
    return tuple(parameters.index(name) for name in names)


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

    # JAX 0.11 moved infer_tiling into the common TPU-info module. The
    # Mosaic module still exports compatibility aliases, but patching those
    # aliases alone leaves DMA pipeline lowering reading the CPU backend.
    try:
        from jax._src import tpu_info as common_tpu_info
    except ImportError:
        common_tpu_info = tpu_info
    info_modules = tuple(dict.fromkeys((tpu_info, common_tpu_info)))

    with _TPU_INFO_PATCH_LOCK:
        original_public = pltpu.get_tpu_info
        originals = [
            (module, name, getattr(module, name))
            for module in info_modules
            for name in ("get_tpu_info", "get_device_kind", "is_tpu_device")
            if hasattr(module, name)
        ]
        device_name = get_pallas_tpu_device_kind()
        chip_version = tpu_info.chip_version_from_device_kind(device_name)
        if chip_version is not None:
            info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
            pltpu.get_tpu_info = lambda: info
            replacements = {
                "get_tpu_info": lambda: info,
                "get_device_kind": lambda: device_name,
                "is_tpu_device": lambda: True,
            }
            for module, name, _ in originals:
                setattr(module, name, replacements[name])
        try:
            yield
        finally:
            pltpu.get_tpu_info = original_public
            for module, name, value in originals:
                setattr(module, name, value)
