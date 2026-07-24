from contextlib import contextmanager


@contextmanager
def tpu_info_for_cpu_export():
    """Use physical TPU metadata while JAX exports with CPU placeholders."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal.utils.hardware import get_tpu_device_name

    original_public = pltpu.get_tpu_info
    original_private = tpu_info.get_tpu_info
    device_name = get_tpu_device_name()
    if device_name == "TPU v7":
        device_name = "TPU7x"
    chip_version = tpu_info.chip_version_from_device_kind(device_name)
    if chip_version is None:
        raise RuntimeError(f"Unsupported TorchTPU device for JAX export: {device_name}")
    info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
    pltpu.get_tpu_info = lambda: info
    tpu_info.get_tpu_info = lambda: info
    try:
        yield
    finally:
        pltpu.get_tpu_info = original_public
        tpu_info.get_tpu_info = original_private
