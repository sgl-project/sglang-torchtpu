# SPDX-License-Identifier: Apache-2.0
"""TPU hooks of a multimodal_gen worker.

The diffusion TPU platform (``sglang.multimodal_gen.runtime.platforms.tpu``)
reaches the plugin through this module only.
"""


def prepare_env(world_size: int) -> None:
    """Set the TPU env of a worker; call before ``torch_tpu`` is imported."""
    from sglang_tpu.patches import _prepare_tpu_env

    _prepare_tpu_env(world_size=world_size, resolve_single_host_topology=True)


def install_runtime_patches() -> None:
    """Patch ``torch_tpu`` for a worker; call after it is imported."""
    from sglang_tpu.patches import (
        _patch_set_device_tolerate_mismatch,
        _register_aten_fallbacks,
    )

    # PJRT owns the chiplet binding; local_rank is only a label.
    _patch_set_device_tolerate_mismatch()
    # sglang_tpu's kernels for ATen ops torch_tpu lacks (e.g. logaddexp).
    _register_aten_fallbacks()


def configure_compile_cache() -> None:
    """Scope persisted torch.compile artifacts to what FX graphs cannot see.

    torch_tpu persists XLA executables in AOTAutogradCache, whose key sees
    neither the device assignment behind collectives nor the opaque Pallas
    kernel sources, compiler flags, stack versions and chip. Call once per
    worker after distributed initialization and before the first compile.
    """
    from sglang_tpu.compile_cache import (
        add_tpu_kernel_inputs_to_compile_cache_key,
        add_tpu_rank_assignment_to_compile_cache_key,
    )

    # srt adds the rank-assignment tag in TpuGraphRunner; diffusion compiles
    # outside it, so it adds the tag itself (a no-op for one worker).
    add_tpu_rank_assignment_to_compile_cache_key()
    add_tpu_kernel_inputs_to_compile_cache_key()
