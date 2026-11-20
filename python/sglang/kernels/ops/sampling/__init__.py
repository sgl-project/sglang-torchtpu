"""Sampling kernels (top-k / top-p probability renormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch


class OutputTopK(BaseFusedOp):
    """Descending output-score TopK values and int64 vocabulary indices."""

    op = "sampling.output_topk"
    format_signature = FormatSignature(
        description="[rows, vocabulary] scores and positive k; returns values, int64 indices"
    )

    def forward_native(
        self, values: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return values.topk(k, dim=-1)

    def _torch_compile_forward(self, num_tokens: int):
        # Backend implementations use traceable operations, so preserve dispatch.
        return None


output_topk = register_fused_op(OutputTopK(), __name__, "output_topk")


register_kernel(
    KernelSpec(
        op="sampling.softmax",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.sampling:softmax",
        capabilities=frozenset(
            {
                CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3)),
            }
        ),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="FP32 [batch, vocab] softmax probabilities",
        ),
        description="Cake Blackwell softmax distributed by FlashInfer.",
    )
)


def cake_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Explicit Cake softmax entry point for Blackwell qualification."""
    return get_kernel("sampling.softmax", KernelBackend.FLASHINFER)(logits)


def softmax(
    logits: torch.Tensor, temperatures: torch.Tensor | None = None
) -> torch.Tensor:
    """Softmax for sampler logits, optionally with per-row temperatures."""
    import torch

    if temperatures is not None and logits.is_cuda and torch.version.hip is not None:
        from sglang.kernels.ops.sampling.temperature_softmax import (
            temperature_softmax,
        )

        return temperature_softmax(logits, temperatures)

    if temperatures is not None:
        logits = logits / temperatures
    from sglang.kernels.cake_kernels.sampling import supports_softmax

    if supports_softmax(logits):
        return cake_softmax(logits)
    return torch.softmax(logits, dim=-1)


register_kernel(
    KernelSpec(
        op="sampling.top_k_renorm_probs",
        backend=KernelBackend.AOT,
        target="sgl_kernel.sampling:top_k_renorm_probs",
        format_signature=FormatSignature(
            description="renormalize probs by top-k thresholding; returns tensor"
        ),
        description="Top-k probability renormalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_p_renorm_probs",
        backend=KernelBackend.AOT,
        target="sgl_kernel.sampling:top_p_renorm_probs",
        format_signature=FormatSignature(
            description="renormalize probs by top-p thresholding; returns tensor"
        ),
        description="Top-p probability renormalization (sgl_kernel wheel).",
    )
)


def top_k_renorm_probs(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-k thresholding."""
    return get_kernel("sampling.top_k_renorm_probs", KernelBackend.AOT)(probs, top_k)


def top_p_renorm_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-p thresholding."""
    return get_kernel("sampling.top_p_renorm_probs", KernelBackend.AOT)(probs, top_p)


__all__ = [
    "OutputTopK",
    "output_topk",
    "cake_softmax",
    "softmax",
    "top_k_renorm_probs",
    "top_p_renorm_probs",
]


# Migrated from srt/layers/utils/hash.py (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.murmur_hash:murmur_hash32",
    )
)
