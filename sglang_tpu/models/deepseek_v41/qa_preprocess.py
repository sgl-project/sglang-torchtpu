# SPDX-License-Identifier: Apache-2.0
"""Prepare merged Q-A/KV adapters after canonical channelwise postquant.

Independent of the backend and early-QKV seam: binding alone does not connect
execution. No core import, JAX import, quantization, offline concatenation, or
runtime-context read occurs here. Rebind after weight replacement, outside any
forward. Blockwise and separate wq_a/wkv linears retain the generic path.
"""

import logging
import math
from dataclasses import dataclass

import torch

from sglang_tpu.models.deepseek_v41.preprocess import _is_mqa

logger = logging.getLogger(__name__)

_QR_RANK = 1280
_HEAD_DIM = 512
_HIDDEN_SIZE = 5120
_EPS = 1e-6


@dataclass(frozen=True, eq=False)
class PreparedQA:
    """Model-local immutable references, never registered as duplicate state.

    The merged FP8 parameter is aliased, not copied. FP32 gamma tensors are
    exact, one-time upcasts of frozen BF16 core norm weights: this selects the
    kernel's FP32 normalized-value multiply boundary. They are not a cast-back
    approximation to originally FP32 weights. Consumers must not mutate them.
    """

    weight: torch.Tensor
    scale: torch.Tensor
    gamma_qr: torch.Tensor
    gamma_kv: torch.Tensor
    eps: float = _EPS


def qa_eligibility_reason(module: torch.nn.Module) -> str | None:
    """Return a structural fallback reason; no tensor reads or allocations."""
    if not _is_mqa(module):
        return "not a core MQALayer"
    if not getattr(module, "is_dsv41", False):
        return "not a DeepSeek-V4.1 attention layer"
    if (
        getattr(module, "hidden_size", None) != _HIDDEN_SIZE
        or getattr(module, "q_lora_rank", None) != _QR_RANK
        or getattr(module, "head_dim", None) != _HEAD_DIM
        or getattr(module, "qk_rope_head_dim", None) != 64
    ):
        return "Q-A requires hidden=5120, QR=1280, KV=512, rotary_dim=64"
    linear = getattr(module, "wqkv_a", None)
    if linear is None or not getattr(module, "fuse_wqa_wkv", False):
        return "missing active merged wqkv_a; separate wq_a/wkv are unsupported"
    if getattr(linear, "bias", None) is not None:
        return "Q-A bias is unsupported"
    if (
        getattr(linear, "weight_block_size", None) is not None
        or getattr(linear, "weight_scale_inv", None) is not None
    ):
        return "blockwise weight_scale_inv requires the generic Q-A path"
    if getattr(linear, "input_scale", None) is not None:
        return "fixed input_scale requires the generic Q-A path"
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float8_e4m3fn:
        return "Q-A weight is not postprocessed FP8 E4M3FN"
    if (
        weight.shape != (_HIDDEN_SIZE, _QR_RANK + _HEAD_DIM)
        or not weight.is_contiguous()
    ):
        return "Q-A weight is not contiguous canonical [5120,1792]"
    scale = getattr(linear, "weight_scale", None)
    if (
        not isinstance(scale, torch.Tensor)
        or scale.dtype != torch.float32
        or scale.shape not in ((_QR_RANK + _HEAD_DIM,), (_QR_RANK + _HEAD_DIM, 1))
    ):
        return "Q-A needs FP32 per-column weight_scale [1792] or [1792,1]"
    if scale.device != weight.device:
        return "Q-A weight and scale devices differ"
    if weight.requires_grad or scale.requires_grad:
        return "Q-A weight and scale must be frozen for inference"
    eps = getattr(getattr(module, "q_norm", None), "variance_epsilon", None)
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        return "both norm eps must share a positive finite kernel epsilon"
    for name, size in (("q_norm", _QR_RANK), ("kv_norm", _HEAD_DIM)):
        norm = getattr(module, name, None)
        gamma = getattr(norm, "weight", None)
        if (
            not isinstance(gamma, torch.Tensor)
            or gamma.dtype != torch.bfloat16
            or gamma.shape != (size,)
        ):
            return f"{name} needs original BF16 gamma [{size}] for exact FP32 upcast"
        # Core RMSNorm keeps Parameter.requires_grad=True even in eval models.
        # The inference adapter snapshots a detached FP32 upcast below; rejecting
        # that metadata would silently disable every normally loaded QA layer.
        if gamma.device != weight.device:
            return f"{name} gamma must be on the Q-A weight device"
        if getattr(norm, "variance_epsilon", None) != eps:
            return "both norm eps must equal the shared kernel epsilon"
        if (
            getattr(norm, "cast_x_before_out_mul", False)
            or getattr(norm, "variance_size_override", None) is not None
            or getattr(norm, "override_orig_dtype", None) not in (None, torch.bfloat16)
        ):
            return f"{name} requires default FP32 normalization and multiply semantics"
    return None


def prepare_qa_adapter(module: torch.nn.Module) -> PreparedQA | None:
    """Prepare references once after postquant, or preserve generic fallback."""
    if qa_eligibility_reason(module) is not None:
        return None
    return PreparedQA(
        weight=module.wqkv_a.weight,
        scale=module.wqkv_a.weight_scale.detach().reshape(-1).contiguous(),
        gamma_qr=module.q_norm.weight.detach().to(torch.float32).contiguous(),
        gamma_kv=module.kv_norm.weight.detach().to(torch.float32).contiguous(),
        eps=float(module.q_norm.variance_epsilon),
    )


def bind_model_qa(model: torch.nn.Module) -> int:
    """Bind ``MQALayer._dsv41_qa``; return eligible layer count, not activation.

    Invoke only AFTER all canonical channelwise quantization postprocessing.
    Unsupported states clear stale adapters. Rebinding refreshes all references
    after reload. The caller still owns op registration and early-QKV dispatch;
    this module does not install a seam, modify forwards, or change quant policy.
    """
    bound = 0
    for name, module in model.named_modules():
        if not _is_mqa(module):
            continue
        reason = qa_eligibility_reason(module)
        module._dsv41_qa = prepare_qa_adapter(module) if reason is None else None
        bound += reason is None
        outcome = reason or "bound canonical FP8 merged Q-A/KV + norms + KV RoPE"
        if getattr(module, "_dsv41_qa_bind_outcome", None) != outcome:
            logger.info("DeepSeek-V4.1 Q-A %s: %s", name or "<root>", outcome)
            module._dsv41_qa_bind_outcome = outcome
    return bound
