# SPDX-License-Identifier: Apache-2.0
"""Static expert placement of the EP megakernel (SGLANG_TPU_MUSE_EXPERT_PLACEMENT).

A placement is a per-layer permutation ``slot -> checkpoint expert``: slot s sits on EP rank
``s // (n_experts / tp)`` as every EP kernel assumes, and the permutation picks which expert
that is. Loading permutes the converted expert stacks (gathered from every rank) and the
router's columns and score-correction bias the same way, so the model computes the same
function with experts relabelled; spreading hot experts over ranks evens out the per-step
expert work. JSON, inline or a file path: ``{"placement": [[e_0, ..., e_{E-1}], ...]}``, one
row per decoder layer (a shorter list leaves the remaining layers in checkpoint order).
"""

import json
import os

import numpy as np
import torch

from sglang.srt.layers.utils import copy_or_rebind_param
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.expert_loading import _wait_for
from sglang_tpu.fp4_moe import _materialize_packed_fp4
from sglang_tpu.muse_megakernel_v2.common.ep_weights import _is_ep_layout


def load_placement(*, n_layers: int, n_experts: int) -> np.ndarray | None:
    """[n_layers, n_experts] int64 slot -> checkpoint expert, or None without a placement."""
    spec = tpu_envs.SGLANG_TPU_MUSE_EXPERT_PLACEMENT.get()
    if not spec:
        return None
    if spec.lstrip().startswith("{"):
        text = spec
    else:
        with open(os.path.expanduser(spec)) as f:
            text = f.read()
    rows = json.loads(text)["placement"]
    if len(rows) > n_layers:
        raise ValueError(f"expert placement: {len(rows)} layers, model {n_layers}")
    out = np.tile(np.arange(n_experts, dtype=np.int64), (n_layers, 1))
    for layer, row in enumerate(rows):
        perm = np.asarray(row, dtype=np.int64)
        if perm.shape != (n_experts,) or not np.array_equal(
            np.sort(perm), np.arange(n_experts)
        ):
            raise ValueError(
                f"expert placement layer {layer}: not a permutation of {n_experts}"
            )
        out[layer] = perm
    return out


def permute_ep_experts(
    experts: torch.nn.Module, kind: str, slot_experts: np.ndarray, *, rank: int
) -> None:
    """Rebind converted EP experts so local slot j holds checkpoint expert slot_experts[j],
    gathered from every rank's EP weights. Every rank must call it for the layer."""
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    if not _is_ep_layout(experts):
        raise NotImplementedError("experts not converted to the EP layout")
    el = experts.w13_weight.data.shape[0]
    if len(slot_experts) != el:
        raise ValueError(
            f"expert placement: {len(slot_experts)} slots for {el} experts"
        )
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        mine = getattr(experts, name).data
        # fp4 code pairs travel as u8 (the copy back into an fp4 buffer is a fresh leaf)
        raw = mine.view(torch.uint8) if mine.dtype == torch.float4_e2m1fn_x2 else mine
        every = tensor_model_parallel_all_gather(raw.contiguous(), dim=0)
        if not torch.equal(every[rank * el : (rank + 1) * el], raw):
            raise RuntimeError(
                f"Muse EP placement: rank {rank}'s gathered {name} block"
            )
        src = torch.as_tensor(slot_experts, dtype=torch.int32, device=raw.device)
        new = every.index_select(0, src)
        del every
        if name in ("w13_weight", "w2_weight") and kind == "fp4":
            new = _materialize_packed_fp4(new)
        _wait_for(new)
        copy_or_rebind_param(experts, name, new)
