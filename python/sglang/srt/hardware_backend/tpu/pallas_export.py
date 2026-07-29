# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import threading
from contextlib import contextmanager

_CPU_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_TPU_INFO_PATCH_LOCK = threading.RLock()
_PALLAS_DEVICE_KIND_BY_TORCHTPU = {
    "TPU v6e": "TPU v6 lite",
    "TPU v7": "TPU7x",
}


def configure_jax_export_platform() -> None:
    """Select CPU-only JAX before importing JAX in a TorchTPU process."""
    configured = os.environ.get("JAX_PLATFORMS")
    if configured is not None and configured.strip().lower() != "cpu":
        raise RuntimeError(
            "TorchTPU Pallas export requires JAX_PLATFORMS=cpu so JAX does not "
            f"initialize a competing TPU client; got {configured!r}."
        )
    if configured is None and "jax" in sys.modules:
        raise RuntimeError(
            "JAX was imported before the TorchTPU Pallas export platform was "
            "configured. Set JAX_PLATFORMS=cpu before importing JAX."
        )
    os.environ["JAX_PLATFORMS"] = "cpu"


def _jax_backend_initialized() -> bool:
    bridge = sys.modules.get("jax._src.xla_bridge")
    return bool(getattr(bridge, "_backends", None))


def configure_jax_export_env(world_size: int) -> None:
    """Configure distinct logical CPU devices before JAX backend startup."""
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")

    configure_jax_export_platform()
    xla_flags = os.environ.get("XLA_FLAGS", "")
    pattern = re.compile(
        rf"(?:^|\s){re.escape(_CPU_DEVICE_COUNT_FLAG)}(?:=|\s+)(\d+)(?=\s|$)"
    )
    configured_counts = [int(value) for value in pattern.findall(xla_flags)]
    if configured_counts:
        if len(configured_counts) != 1 or configured_counts[0] != world_size:
            raise RuntimeError(
                f"{_CPU_DEVICE_COUNT_FLAG} must equal WORLD_SIZE={world_size}; "
                f"XLA_FLAGS={xla_flags!r}."
            )
        return

    if _jax_backend_initialized():
        raise RuntimeError(
            "The JAX backend was initialized before TorchTPU configured its "
            f"{world_size} logical CPU export devices."
        )
    os.environ["XLA_FLAGS"] = " ".join(
        part
        for part in (xla_flags.strip(), f"{_CPU_DEVICE_COUNT_FLAG}={world_size}")
        if part
    )


def get_jax_export_devices(world_size: int):
    """Return the preconfigured CPU devices used for global partitions."""

    import jax

    devices = tuple(jax.devices("cpu"))
    if len(devices) != world_size:
        raise RuntimeError(
            f"Expected {world_size} logical CPU export devices, got {len(devices)}. "
            "JAX may have initialized before the TorchTPU export environment."
        )
    if len(set(devices)) != world_size:
        raise RuntimeError("JAX returned duplicate logical CPU export devices.")
    return devices


def get_torch_tpu_device_kind() -> str:
    """Return physical TPU hardware metadata without querying JAX devices."""
    from torch_tpu._internal.utils.hardware import get_tpu_device_name

    return get_tpu_device_name()


def get_pallas_tpu_device_kind() -> str:
    """Return the physical device spelling expected by JAX Pallas."""
    device_kind = get_torch_tpu_device_kind()
    return _PALLAS_DEVICE_KIND_BY_TORCHTPU.get(device_kind, device_kind)


@contextmanager
def tpu_info_for_cpu_export():
    """Use physical TPU metadata while JAX exports with CPU placeholders."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    with _TPU_INFO_PATCH_LOCK:
        original_public = pltpu.get_tpu_info
        original_private = tpu_info.get_tpu_info
        original_device_kind = tpu_info.get_device_kind
        original_is_tpu_device = tpu_info.is_tpu_device
        device_name = get_pallas_tpu_device_kind()
        chip_version = tpu_info.chip_version_from_device_kind(device_name)
        if chip_version is None:
            raise RuntimeError(
                f"Unsupported TorchTPU device for JAX export: {device_name}"
            )
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
            tpu_info.get_device_kind = original_device_kind
            tpu_info.is_tpu_device = original_is_tpu_device
