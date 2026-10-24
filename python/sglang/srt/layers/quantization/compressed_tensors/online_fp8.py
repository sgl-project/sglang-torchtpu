# SPDX-License-Identifier: Apache-2.0
"""Load a BF16 checkpoint as a compressed-tensors ``FP8_DYNAMIC`` checkpoint.

``FP8_DYNAMIC`` (llm-compressor) is float8_e4m3 weights with one static scale
per output channel (``weight_scale`` ``[N, 1]``) and dynamic per-token FP8
activations. A model whose quant config is that scheme creates FP8 parameters
from the start, so the full-precision weights never reach the device: this
module rewrites the checkpoint stream on the host, one tensor at a time,
turning each quantized ``<module>.weight`` into ``<module>.weight`` (fp8) plus
``<module>.weight_scale`` (the checkpoint dtype, as llm-compressor writes it).

Every tensor is quantized whole, before tensor-parallel sharding, so a
row-parallel shard carries the same per-channel scale as in an offline FP8
checkpoint. The rounding replicates llm-compressor's, so the result is bit
identical to running llm-compressor offline on the same weights.

Use it from ``load_weights`` when the quant config came from a JSON override
on a BF16 checkpoint; FP8 tensors (an offline-quantized checkpoint) pass
through untouched, so the same model code serves both.
"""

from typing import Callable, Iterable, Iterator, Optional, Tuple

import torch

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max
_FULL_PRECISION = (torch.bfloat16, torch.float16, torch.float32)


def quantize_fp8_per_channel(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N, K]`` weight -> (fp8 ``[N, K]``, scale ``[N, 1]`` in ``weight.dtype``).

    Matches compressed-tensors' symmetric channel-strategy float quantization:
    ``scale = amax(|w|, dim=1) / 448`` (an all-zero channel gets the dtype's
    eps) and ``q = fp8(clamp(w / scale))``, both computed in the weight dtype.
    The division rounds to that dtype before the FP8 cast; dividing in fp32
    instead changes about 3% of BF16 elements.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D weight, got {tuple(weight.shape)}")
    if weight.dtype not in _FULL_PRECISION:
        raise ValueError(f"Expected a full-precision weight, got {weight.dtype}")
    amax = weight.abs().amax(dim=1, keepdim=True)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.finfo(scale.dtype).eps, scale)
    scaled = weight / scale
    # compressed-tensors adds the (zero) zero point, turning -0.0 into +0.0.
    scaled += 0
    qweight = torch.clamp(scaled, -_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
    return qweight, scale


def online_fp8_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    is_quantized_module: Callable[[str], bool],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Quantize each full-precision ``<module>.weight`` that ``is_quantized_module``.

    ``is_quantized_module`` gets the checkpoint module name (``name`` without
    ``.weight``). Other tensors, including already-FP8 weights and their
    scales, are yielded unchanged.
    """
    for name, tensor in weights:
        module = _quantizable_module(name, tensor)
        if module is None or not is_quantized_module(module):
            yield name, tensor
            continue
        qweight, scale = quantize_fp8_per_channel(tensor)
        yield name, qweight
        yield f"{module}.weight_scale", scale


def _quantizable_module(name: str, tensor: torch.Tensor) -> Optional[str]:
    if (
        not name.endswith(".weight")
        or tensor.dim() != 2
        or tensor.dtype not in _FULL_PRECISION
    ):
        return None
    return name[: -len(".weight")]
