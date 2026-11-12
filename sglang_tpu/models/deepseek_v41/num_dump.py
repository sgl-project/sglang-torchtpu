# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (not for merge): dump layer 0-2 activations for vLLM comparison.

Enabled by SGLANG_TPU_DSV41_DUMP=1. Inside the compiled forward each hook point
copies its tensor into a preallocated fp32 device buffer (no graph break).
After a forward, if /tmp/dsv41_dump_on exists, the buffers plus the batch's
input_ids/positions are saved to /tmp/dsv41_dump/<n>_<mode>_r<rank>.pt.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("SGLANG_TPU_DSV41_DUMP") == "1"
LAYERS = (0, 1, 2)
POINTS = (
    "embed_in",
    "embed_calls",
    "embed_out",
    "embed_ref",
    "embed_clone",
    "hash_ids",
    "attn_in",
    "attn_out",
    "engram_in",
    "engram_out",
    "moe_in",
    "moe_out",
)
_ROWS, _COLS = 128, 20480
_BUF: dict = {}
_COUNT = [0]
_HASHER: list = []
_SHAPES: dict = {}
_OUT = "/tmp/dsv41_dump"


def _ensure(device):
    if not _BUF:
        for layer in LAYERS:
            for p in POINTS:
                _BUF[(p, layer)] = torch.zeros(_ROWS, _COLS, device=device)
    return _BUF


def _put(point, layer, x):
    key = (point, layer)
    if key not in _BUF or not isinstance(x, torch.Tensor) or x.dim() == 0:
        return
    t = x.shape[0]
    _SHAPES[key] = tuple(x.shape)
    flat = x.reshape(t, -1)
    rows, cols = min(t, _ROWS), min(flat.shape[1], _COLS)
    _BUF[key][:rows, :cols].copy_(flat[:rows, :cols].float())


def attn_forward(original, layer, x, *args, **kwargs):
    _put("attn_in", layer.layer_id, x)
    out = original(layer, x, *args, **kwargs)
    _put("attn_out", layer.layer_id, out)
    return out


def engram_forward(original, engram, x, *args, **kwargs):
    # Keyed by Engram index (0 = first Engram layer), not decoder layer.
    _put("engram_in", engram.layer_hash_index, x)
    out = original(engram, x, *args, **kwargs)
    _put("engram_out", engram.layer_hash_index, out)
    return out


def hasher_forward(original, hasher, input_ids, forward_batch, *args, **kwargs):
    if not _HASHER:
        _HASHER.append(hasher)
    out = original(hasher, input_ids, forward_batch, *args, **kwargs)
    # Same slot for every layer key 0; hash ids are [T, n_layers, cols].
    _put("hash_ids", 0, out)
    return out


def moe_forward(original, moe, hidden_states, *args, **kwargs):
    _put("moe_in", moe.layer_id, hidden_states)
    out = original(moe, hidden_states, *args, **kwargs)
    _put("moe_out", moe.layer_id, out)
    return out


def embed_forward(original, embed, input_):
    out = original(embed, input_)
    # The ids this call really embedded (DP/SP layouts may reorder rows), and a
    # per-forward call counter: a second call would overwrite embed_out.
    _put("embed_in", 0, input_.reshape(-1, 1))
    calls = _BUF.get(("embed_calls", 0))
    if calls is not None:
        calls[0, 0].add_(1.0)
    _put("embed_out", 0, out)
    # Same ids gathered again in-graph straight from the weight, and a clone of
    # the output: tells a miscompiled gather from a bad dump copy.
    _put("embed_ref", 0, embed.weight.index_select(0, input_.reshape(-1).long()))
    _put("embed_clone", 0, out.clone())
    return out


def before_forward(device) -> None:
    # Zero every slot so a point not reached this forward reads 0, not the
    # previous forward's rows.
    for buf in _ensure(device).values():
        buf.zero_()
    _SHAPES.clear()


def after_forward(forward_batch) -> None:
    if not _BUF or not os.path.exists("/tmp/dsv41_dump_on"):
        return
    from sglang.srt.runtime_context import get_parallel

    os.makedirs(_OUT, exist_ok=True)
    rank = get_parallel().tp_rank
    rec = {
        "mode": forward_batch.forward_mode.name,
        "input_ids": forward_batch.input_ids.cpu(),
        "positions": forward_batch.positions.cpu(),
        "extend_seq_lens_cpu": getattr(forward_batch, "extend_seq_lens_cpu", None),
        "seq_lens": forward_batch.seq_lens.cpu(),
    }
    if _HASHER and _HASHER[0].history is not None:
        # Eager read after the compiled forward: did its history commit land?
        req = forward_batch.req_pool_indices.to(torch.int64)
        rec["history_after"] = _HASHER[0].history[req].cpu()
        rec["req_pool_indices"] = forward_batch.req_pool_indices.cpu()
    rec["shapes"] = {f"{p}{layer}": s for (p, layer), s in _SHAPES.items()}
    t = min(int(forward_batch.input_ids.shape[0]), _ROWS)
    for layer in LAYERS:
        for p in POINTS:
            rec[f"{p}{layer}"] = _BUF[(p, layer)][:t].cpu()
    n = _COUNT[0]
    _COUNT[0] += 1
    torch.save(rec, f"{_OUT}/{n:04d}_{rec['mode']}_r{rank}.pt")


def hook_specs():
    if not ENABLED:
        return []
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward",
            embed_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.MQALayer.forward",
            attn_forward,
            HookType.AROUND,
        ),
        ("sglang.srt.layers.engram.Engram.forward", engram_forward, HookType.AROUND),
        (
            "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward",
            moe_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.forward",
            hasher_forward,
            HookType.AROUND,
        ),
    ]
