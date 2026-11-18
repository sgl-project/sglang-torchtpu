# SPDX-License-Identifier: Apache-2.0
"""Static expert placement for Muse Spark (SGLANG_TPU_MUSE_EXPERT_PLACEMENT).

A placement is a per-layer permutation ``slot -> checkpoint expert``. Loading permutes the
expert stacks and the router rows (gate weight, score-correction bias) the same way, so the
model computes the same function with experts relabelled: every kernel keeps placing slot s on
EP rank ``s // (n_experts / ep_size)``, and the permutation chooses which expert sits there.
JSON: ``{"placement": [[e_0, ..., e_{E-1}], ...]}``, one row per decoder layer; a layer
missing from a shorter list keeps the identity.
"""

import json
from functools import lru_cache

import numpy as np
import torch

from sglang_tpu.envs import envs as tpu_envs


def parse_placement(text: str, *, n_layers: int, n_experts: int) -> np.ndarray:
    """[n_layers, n_experts] int64 slot -> expert; raises on a row that is not a permutation."""
    rows = json.loads(text)["placement"]
    if len(rows) > n_layers:
        raise ValueError(f"expert placement has {len(rows)} layers, model {n_layers}")
    out = np.tile(np.arange(n_experts, dtype=np.int64), (n_layers, 1))
    for layer, row in enumerate(rows):
        perm = np.asarray(row, dtype=np.int64)
        if perm.shape != (n_experts,) or not np.array_equal(
            np.sort(perm), np.arange(n_experts)
        ):
            raise ValueError(f"expert placement layer {layer}: not a permutation of {n_experts}")
        out[layer] = perm
    return out


@lru_cache(maxsize=1)
def _placement_from_env(n_layers: int, n_experts: int) -> np.ndarray | None:
    path = tpu_envs.SGLANG_TPU_MUSE_EXPERT_PLACEMENT.get()
    if not path:
        return None
    with open(path) as f:
        return parse_placement(f.read(), n_layers=n_layers, n_experts=n_experts)


def slot_experts(
    *, layer: int, n_layers: int, n_experts: int, start: int, count: int
) -> torch.Tensor | None:
    """Checkpoint experts of slots [start, start + count) of ``layer``; None without a
    placement (the slots hold experts start.. in order)."""
    placement = _placement_from_env(n_layers, n_experts)
    if placement is None:
        return None
    return torch.from_numpy(placement[layer, start : start + count].copy())
