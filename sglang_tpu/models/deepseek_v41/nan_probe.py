# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (not for merge): per-layer NaN flags inside the compiled forward.

Enabled by SGLANG_TPU_DSV41_NAN_PROBE=1. Each attention, Engram and MoE output
writes isnan(out).any() into a device buffer; the shim reads it after each
forward and logs the slots that went NaN.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_TPU_DSV41_NAN_PROBE") == "1"
_KINDS = ("attn", "engram", "moe")
_MAX_LAYERS = 128
_BUF = None


def _ensure(device) -> torch.Tensor:
    global _BUF
    if _BUF is None:
        _BUF = torch.zeros(len(_KINDS) * _MAX_LAYERS, dtype=torch.int32, device=device)
    return _BUF


def _mark(kind: int, layer: int, out):
    if _BUF is None or not isinstance(out, torch.Tensor):
        return
    slot = kind * _MAX_LAYERS + layer
    flag = torch.isnan(out).any().to(torch.int32)
    _BUF[slot] = torch.maximum(_BUF[slot], flag)


def attn_forward(original, layer, *args, **kwargs):
    out = original(layer, *args, **kwargs)
    _mark(0, layer.layer_id, out)
    return out


def engram_forward(original, engram, *args, **kwargs):
    out = original(engram, *args, **kwargs)
    _mark(1, engram.layer_hash_index, out)
    return out


def moe_forward(original, moe, *args, **kwargs):
    out = original(moe, *args, **kwargs)
    _mark(2, moe.layer_id, out)
    return out


def before_forward(device) -> None:
    _ensure(device).zero_()


def after_forward(forward_batch, out) -> None:
    from sglang_tpu.compilation.pending_call import PendingCall

    if _BUF is None or isinstance(out, PendingCall):
        return
    hits = torch.nonzero(_BUF.cpu()).flatten().tolist()
    logits = getattr(out, "next_token_logits", None)
    logits_nan = (
        bool(torch.isnan(logits).any().item()) if isinstance(logits, torch.Tensor) else None
    )
    named = [f"{_KINDS[h // _MAX_LAYERS]}{h % _MAX_LAYERS}" for h in hits]
    logger.info(
        "[nan probe] mode=%s tokens=%d logits_nan=%s first=%s all=%s",
        forward_batch.forward_mode.name,
        int(forward_batch.input_ids.shape[0]),
        logits_nan,
        named[:1],
        named[:40],
    )


def hook_specs():
    if not ENABLED:
        return []
    from sglang.srt.plugins.hook_registry import HookType

    return [
        ("sglang.srt.models.deepseek_v4.MQALayer.forward", attn_forward, HookType.AROUND),
        ("sglang.srt.layers.engram.Engram.forward", engram_forward, HookType.AROUND),
        ("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward", moe_forward, HookType.AROUND),
    ]
