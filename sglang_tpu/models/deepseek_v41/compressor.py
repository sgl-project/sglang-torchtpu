# SPDX-License-Identifier: Apache-2.0
"""Model-owned ratio-1 adapters consumed by the TPU attention backend."""

import logging
import math
from dataclasses import dataclass

import torch

from sglang_tpu.models.deepseek_v41.preprocess import _is_mqa

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PreparedCompressor:
    """Canonical transpose plus exact original gamma reference.

    The BF16 transpose allocates 5,242,880 bytes (5 MiB) PER eligible layer in
    addition to the source weight. Rebind after loading/replacement/in-place
    weight updates, outside forwards. No tensor is cached per forward and no
    duplicate parameters or buffers are registered in state_dict. Adapter
    tensors are detached for inference; core parameters keep their original
    requires_grad flags (eval alone does not freeze them). Gamma shares the
    original storage without rounding or allocation.
    """

    weight: torch.Tensor
    gamma: torch.Tensor
    eps: float


def compressor_eligibility_reason(module: torch.nn.Module) -> str | None:
    if not _is_mqa(module) or not getattr(module, "is_dsv41", False):
        return "not a core V4.1 MQALayer"
    compressor = getattr(module, "compressor", None)
    if not any(
        cls.__name__ == "DeepseekV41Compressor"
        and cls.__module__ == "sglang.srt.layers.attention.dsv4.dsv41_sparse"
        for cls in type(compressor).__mro__
    ):
        return "not a core DeepseekV41Compressor"
    if getattr(compressor, "compress_ratio", None) != 1:
        return "only stateless ratio 1 is supported"
    linear = getattr(compressor, "wkv", None)
    if type(linear) is not torch.nn.Linear or linear.bias is not None:
        return "requires bias-free core nn.Linear"
    weight = linear.weight
    if weight.shape != (512, 5120) or weight.dtype != torch.bfloat16:
        return "requires BF16 [512,5120] weight"
    if getattr(module, "qk_rope_head_dim", None) != 64:
        return "requires rotary_dim=64"
    norm = compressor.norm
    gamma = getattr(norm, "weight", None)
    if (
        not isinstance(gamma, torch.Tensor)
        or gamma.shape != (512,)
        or gamma.dtype not in (torch.bfloat16, torch.float32)
    ):
        return "requires original BF16 or FP32 gamma [512]"
    if gamma.device != weight.device:
        return "gamma must be on weight device"
    if not any(
        cls.__name__ == "RMSNorm"
        and cls.__module__ == "sglang.srt.layers.attention.dsv4.dsv41_sparse"
        for cls in type(norm).__mro__
    ):
        return "requires core compressor RMSNorm"
    eps = getattr(norm, "eps", None)
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        return "requires positive finite norm epsilon"
    if (
        getattr(norm, "cast_x_before_out_mul", False)
        or getattr(norm, "variance_size_override", None) is not None
        or getattr(norm, "override_orig_dtype", None) not in (None, torch.bfloat16)
    ):
        return "requires default FP32 RMS normalization and gamma multiply"
    return None


def prepare_compressor_adapter(module: torch.nn.Module) -> PreparedCompressor | None:
    if compressor_eligibility_reason(module) is not None:
        return None
    core = module.compressor
    return PreparedCompressor(
        core.wkv.weight.detach().t().contiguous(),
        core.norm.weight.detach(),
        float(core.norm.eps),
    )


def bind_model_compressor(model: torch.nn.Module) -> int:
    """Refresh _dsv41_compressor after loading; leave core/state/dispatch alone."""
    count = 0
    for name, module in model.named_modules():
        if not _is_mqa(module):
            continue
        reason = compressor_eligibility_reason(module)
        prepared = prepare_compressor_adapter(module) if reason is None else None
        module._dsv41_compressor = prepared
        count += prepared is not None
        outcome = (
            reason or "bound ratio-1 BF16 compressor + RoPE + encode (5 MiB transpose)"
        )
        if getattr(module, "_dsv41_compressor_bind_outcome", None) != outcome:
            logger.info("DeepSeek-V4.1 compressor %s: %s", name or "<root>", outcome)
            module._dsv41_compressor_bind_outcome = outcome
    logger.info("DeepSeek-V4.1 compressor bound %d ratio-1 adapters", count)
    return count
