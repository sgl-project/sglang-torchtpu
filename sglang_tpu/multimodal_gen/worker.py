# SPDX-License-Identifier: Apache-2.0
"""Worker-side steps of the diffusion TPU platform (``TpuOOTPlatform``)."""


def install_runtime_patches() -> None:
    """Patch a worker; call after ``torch_tpu`` is imported."""
    import torch

    from sglang_tpu.multimodal_gen import minimax_h3, rope

    # torch_tpu runs float32 matmuls and convolutions as a single bf16 pass
    # ("medium", 2.3e-3 relative error); CUDA runs them in float32. Keep the
    # float32 paths (VAE decode, embeddings) float32.
    torch.set_float32_matmul_precision("highest")
    rope.install()
    minimax_h3.install()
