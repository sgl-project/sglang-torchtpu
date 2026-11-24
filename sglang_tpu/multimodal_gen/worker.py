# SPDX-License-Identifier: Apache-2.0
"""Worker-side steps of the diffusion TPU platform (``TpuOOTPlatform``)."""

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def install_runtime_patches() -> None:
    """Patch a worker; call after ``torch_tpu`` is imported."""
    import torch

    from sglang_tpu.multimodal_gen import (
        minimax_h3,
        minimax_h3_qk_norm_rope,
        minimax_h3_ulysses_overlap,
        rope,
    )
    from sglang_tpu.patches import _register_aten_fallbacks

    # sglang_tpu's kernels for ATen ops torch_tpu lacks (e.g. logaddexp).
    _register_aten_fallbacks()
    # torch_tpu runs float32 matmuls and convolutions as a single bf16 pass
    # ("medium", 2.3e-3 relative error); CUDA runs them in float32. Keep the
    # float32 paths (VAE decode, embeddings) float32.
    torch.set_float32_matmul_precision("highest")
    rope.install()
    minimax_h3.install()
    minimax_h3_qk_norm_rope.install()
    minimax_h3_ulysses_overlap.install()


def _wants_ulysses_overlap_flags() -> bool:
    """Whether this worker serves the MiniMax-H3 head-group pipeline: an H3
    pipeline (MiniMaxH3PipelineConfig or a subclass) with Ulysses > 1 and no
    ring, as the server arguments the scheduler set before the worker
    initializes its device say."""
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    try:
        args = get_global_server_args()
    except ValueError:  # not set: no server worker
        return False
    # By class name: the config's module is already imported (it came with the
    # server arguments); importing it here could pull more in before libtpu.
    config_classes = type(args.pipeline_config).__mro__
    return (
        any(cls.__name__ == "MiniMaxH3PipelineConfig" for cls in config_classes)
        and (args.ulysses_degree or 1) > 1
        and (args.ring_degree or 1) == 1
    )


def apply_libtpu_flags() -> None:
    """Append the diffusion worker's libtpu flags; call before the worker's
    device is initialized.

    libtpu reads LIBTPU_INIT_ARGS once, when torch_tpu creates its PJRT client
    on the worker's first device initialization; ``import torch_tpu``, which
    every worker has already run by then, defers that. Flags the launcher
    already set keep their values. The async all-to-all flags apply only to a
    worker that serves the MiniMax-H3 Ulysses pipeline (they apply to every
    XLA program of that worker: the DiT, and the text encoder and VAEs when
    they run on TPU).
    """
    from sglang_tpu.env_override import _override_libtpu_flags
    from sglang_tpu.multimodal_gen.minimax_h3_ulysses_overlap import LIBTPU_FLAGS

    if not _wants_ulysses_overlap_flags():
        return
    appended = _override_libtpu_flags(LIBTPU_FLAGS)
    if appended:
        logger.info("diffusion worker appended libtpu flags: %s", " ".join(appended))


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
