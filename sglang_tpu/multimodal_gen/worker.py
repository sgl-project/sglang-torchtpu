# SPDX-License-Identifier: Apache-2.0
"""Worker-side steps of the diffusion TPU platform (``TpuOOTPlatform``)."""


def install_runtime_patches() -> None:
    """Patch a worker; call after ``torch_tpu`` is imported."""
    import torch

    from sglang_tpu.multimodal_gen import rope, sdpa_mask
    from sglang_tpu.patches import _register_aten_fallbacks

    # sglang_tpu's kernels for ATen ops torch_tpu lacks (e.g. logaddexp).
    _register_aten_fallbacks()
    # torch_tpu runs float32 matmuls and convolutions as a single bf16 pass
    # ("medium", 2.3e-3 relative error); CUDA runs them in float32. Keep the
    # float32 paths (VAE decode, embeddings) float32.
    torch.set_float32_matmul_precision("highest")
    rope.install()
    sdpa_mask.install()


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
