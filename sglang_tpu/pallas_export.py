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


def get_tpu_info_or_none():
    """Target metadata on TPU or during CPU export; None on an untargeted CPU.

    JAX's default backend describes the tracing host, not the lowering target.
    Use the same Pallas metadata as tiling and lowering, including an abstract
    TPU mesh or ``tpu_info_for_cpu_export``. Do not cache across those contexts.
    """
    from jax.experimental.pallas import tpu as pltpu

    try:
        return pltpu.get_tpu_info()
    except ValueError:
        # Pallas rejects a CPU/GPU device without a TPU target context.
        return None


def get_tpu_generation() -> int:
    """The lowering target's generation, or -1 without a TPU target."""
    info = get_tpu_info_or_none()
    return -1 if info is None else info.generation


@contextmanager
def tpu_info_for_cpu_export():
    """Ensure Pallas kernels receive physical TPU7x hardware metadata during export."""
    from jax._src import tpu_info as core_tpu_info
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    # Since JAX 0.11, jax._src.pallas.mosaic.tpu_info only re-exports
    # jax._src.tpu_info. Mosaic lowering reads the re-export, while tiling
    # inference and lax lowerings read the core module, so patch both.
    patched = ("get_tpu_info", "get_device_kind", "is_tpu_device")
    modules = (tpu_info, core_tpu_info)
    with _TPU_INFO_PATCH_LOCK:
        original_public = pltpu.get_tpu_info
        originals = [
            {name: getattr(module, name) for name in patched} for module in modules
        ]
        device_name = get_pallas_tpu_device_kind()
        chip_version = tpu_info.chip_version_from_device_kind(device_name)
        if chip_version is not None:
            info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
            pltpu.get_tpu_info = lambda: info
            for module in modules:
                module.get_tpu_info = lambda: info
                module.get_device_kind = lambda: device_name
                module.is_tpu_device = lambda: True
        try:
            yield
        finally:
            pltpu.get_tpu_info = original_public
            for module, original in zip(modules, originals, strict=True):
                for name, value in original.items():
                    setattr(module, name, value)
