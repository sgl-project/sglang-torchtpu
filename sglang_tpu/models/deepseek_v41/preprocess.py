# SPDX-License-Identifier: Apache-2.0
"""Bind Q-B adapters after dense-linear quantization postprocessing.

This module deliberately does not import the core model (or the JAX bridge).
Adapters are immutable model-owned metadata, not modules or registered tensor
copies. The original linear remains the sole owner of its FP8 parameter.
Rebind after weight replacement; never prepare adapters in a model forward.
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PreparedQB:
    """Canonical postquant tensors consumed by ``ops.preprocess.qb_rope``.

    ``weight`` aliases the linear's parameter; ``scale`` is a detached,
    contiguous FP32 view prepared once at bind time, with no requantization.
    Neither is registered again in the model's parameters/buffers/state_dict.
    """

    weight: torch.Tensor
    scale: torch.Tensor
    rank: int
    heads: int


def _is_mqa(module: torch.nn.Module) -> bool:
    # Inspect the already constructed instance's MRO instead of importing the
    # giant core module. Subclasses retain the exact core MQALayer base.
    return any(
        cls.__name__ == "MQALayer" and cls.__module__ == "sglang.srt.models.deepseek_v4"
        for cls in type(module).__mro__
    )


def _is_v41_config(config) -> bool:
    if config is None:
        return False
    if getattr(config, "model_type", None) in ("deepseek_v41", "deepseek_v41_text"):
        return True
    architectures = getattr(config, "architectures", None) or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    if "DeepseekV41ForCausalLM" in architectures:
        return True
    text = getattr(config, "text_config", None)
    return text is not None and text is not config and _is_v41_config(text)


def qb_eligibility_reason(module: torch.nn.Module, config=None) -> str | None:
    """Return an explicit fallback reason, or None for a supported MQALayer.

    Only structural metadata is read: no host/device tensor reads, allocation,
    quantization, runtime-context lookup, or model import is performed.
    ``config`` is the owning model's HF config, not a ServerArgs record.
    """
    if not _is_mqa(module):
        return "not a core MQALayer"
    if not (getattr(module, "is_dsv41", False) or _is_v41_config(config)):
        return "not a DeepSeek-V4.1 attention layer"
    if getattr(module, "q_head_norm", True):
        return "per-head Q normalization is not fused by Q-B + RoPE"
    if (
        getattr(module, "head_dim", None) != 512
        or getattr(module, "qk_rope_head_dim", None) != 64
    ):
        return "Q-B + RoPE requires head_dim=512 and rotary_dim=64"
    heads = getattr(module, "n_local_heads", 0)
    rank = getattr(module, "q_lora_rank", 0)
    if not isinstance(heads, int) or heads < 8 or heads % 4:
        return "local heads must be a multiple of 4 and at least 8"
    if not isinstance(rank, int) or rank <= 0 or rank % 128:
        return "Q rank must be a positive multiple of 128"
    norm = getattr(module, "q_norm", None)
    if getattr(getattr(norm, "weight", None), "dtype", None) != torch.bfloat16:
        return "q_norm must produce BF16 normalized rows"
    linear = getattr(module, "wq_b", None)
    if linear is None:
        return "missing wq_b linear"
    if getattr(linear, "bias", None) is not None:
        return "Q-B bias is unsupported"
    if (
        getattr(linear, "weight_block_size", None) is not None
        or getattr(linear, "weight_scale_inv", None) is not None
    ):
        return "blockwise weight_scale_inv requires the generic Q-B path"
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float8_e4m3fn:
        return "Q-B weight is not postprocessed FP8 E4M3FN"
    if weight.shape != (rank, heads * 512) or not weight.is_contiguous():
        return "Q-B weight is not contiguous canonical [rank, local_heads*512]"
    scale = getattr(linear, "weight_scale", None)
    if (
        not isinstance(scale, torch.Tensor)
        or scale.dtype != torch.float32
        or scale.shape not in ((heads * 512,), (heads * 512, 1))
    ):
        return "Q-B needs FP32 per-column weight_scale [N] or [N,1]"
    if scale.device != weight.device:
        return "Q-B weight and scale devices differ"
    if weight.requires_grad or scale.requires_grad:
        return "Q-B weight and scale must be frozen for inference"
    return None


def prepare_qb_adapter(module: torch.nn.Module, config=None) -> PreparedQB | None:
    """Prepare canonical read-only references once, or leave generic fallback."""
    if qb_eligibility_reason(module, config) is not None:
        return None
    linear = module.wq_b
    return PreparedQB(
        weight=linear.weight,
        scale=linear.weight_scale.detach().reshape(-1).contiguous(),
        rank=module.q_lora_rank,
        heads=module.n_local_heads,
    )


def bind_model_preprocessing(model: torch.nn.Module) -> int:
    """Bind ``MQALayer._dsv41_qb`` after all quant methods finish postprocessing.

    Unsupported layers receive None and keep the backend's generic path. Logs
    are emitted once per model-local layer name and outcome, not per forward.
    Calling again refreshes references after a weight reload. Returns bound
    layer count; registering the Torch/JAX op remains the backend's responsibility.
    """
    config = getattr(model, "config", None)
    names = {id(module): name or "<root>" for name, module in model.named_modules()}
    bound = 0
    for module in model.modules():
        if not _is_mqa(module):
            continue
        reason = qb_eligibility_reason(module, config)
        module._dsv41_qb = (
            prepare_qb_adapter(module, config) if reason is None else None
        )
        bound += reason is None
        outcome = reason or "bound canonical FP8 Q-B + RoPE"
        if getattr(module, "_dsv41_qb_bind_outcome", None) != outcome:
            logger.info("DeepSeek-V4.1 Q-B %s: %s", names[id(module)], outcome)
            module._dsv41_qb_bind_outcome = outcome
    return bound
