"""Pallas kernels for the SGLang TPU attention backend.

These kernels are vendored from sglang-jax (which adapts the upstream
`tpu-inference` v3 kernel) and exposed to PyTorch via
`torch_tpu._internal.pallas.jax_op`. They are imported lazily by the
`pallas` attention backend so that environments without JAX installed
can still load the rest of SGLang.
"""

from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
    get_kv_cache_shape,
    ragged_paged_attention,
)

__all__ = [
    "get_kv_cache_shape",
    "ragged_paged_attention",
]
