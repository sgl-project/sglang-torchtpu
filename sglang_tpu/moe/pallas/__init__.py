"""Pallas fused-MoE kernels for the TPU torch_tpu platform.

The ``gmm_v2`` kernel is vendored from sglang-jax (which adapts the upstream
`tpu-inference` megablox v2 kernel) and exposed to PyTorch via
``torch_tpu._internal.pallas.jax_op``. It is imported lazily by the
``fused_moe_op`` bridge so that environments without JAX installed can still
load the rest of SGLang.
"""

from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

__all__ = [
    "fused_moe_call",
]
