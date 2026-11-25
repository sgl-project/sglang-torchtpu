# SPDX-License-Identifier: Apache-2.0
"""Static expert placement for DeepSeek-V4.1 fused EP: a per-layer permutation
of the routed experts over the EP ranks (one slot per expert, no replicas).

SGLANG_TPU_DSV41_FUSED_EP_LAYOUT names a JSON file (expert_placements
format, ``slots == experts / shards``)::

    {"experts": 384, "shards": 8, "slots": 48,
     "layers": {"<layer id>": [[48 logical expert ids of rank 0], ..., [rank 7]]}}

Physical slot ``r * slots + j`` of a listed layer holds logical expert
``layers[l][r][j]``; unlisted layers keep the contiguous order. The target
model's loader reads each rank's listed experts (``keep_expert``) and its
FusedMoE layers store them in that slot order (``place_expert_ids``, AROUND
FusedMoE._weight_loader_expert_ids); fused EP then passes the kernel the
logical -> physical table (``expert_slot_table``). Gate logits, the score
bias and the routing stay in logical order, so the outputs are unchanged.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)
# Set by the V4.1 loader on the target model's FusedMoE layers: this EP rank.
LAYOUT_RANK_ATTR = "_dsv41_layout_rank"
# Local slots each tagged layer has received a routed-expert tensor for.
FILLED_ATTR = "_dsv41_layout_filled"

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^(.*)\.experts\.(\d+)\.")

Layout = dict[int, list[list[int]]]


# Layouts shipped with the package; the env may name one by file name.
PACKAGED_DIR = Path(__file__).parent / "expert_layouts"


def layout_path() -> Optional[str]:
    """The env's layout file: a path, or the name of a packaged layout."""
    from sglang_tpu.envs import envs as tpu_envs

    path = tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_LAYOUT.get()
    if not path:
        return None
    if not Path(path).exists() and (PACKAGED_DIR / path).exists():
        return str(PACKAGED_DIR / path)
    return path


@functools.cache
def _read(path: str) -> Layout:
    data = json.loads(Path(path).read_text())
    return {int(k): [[int(e) for e in s] for s in v] for k, v in data["layers"].items()}


def active_layout() -> Optional[Layout]:
    """The layout to load and route with. Only fused EP places experts by it
    (place_expert_ids is a fused EP hook), so without SGLANG_TPU_DSV41_FUSED_EP
    the loader must keep the contiguous experts and the layout is ignored."""
    from sglang_tpu.envs import envs as tpu_envs

    path = layout_path()
    if path is None:
        return None
    if not tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.get():
        _warn_layout_ignored(path)
        return None
    return _read(path)


@functools.cache
def _warn_layout_ignored(path: str) -> None:
    logger.warning(
        "SGLANG_TPU_DSV41_FUSED_EP_LAYOUT=%s ignored: expert layouts need "
        "SGLANG_TPU_DSV41_FUSED_EP=1",
        path,
    )


def validate(layout: Layout, n_experts: int, ep: int) -> None:
    """Each listed layer: ``ep`` shards of ``n_experts / ep`` slots holding
    every logical expert exactly once."""
    if n_experts % ep:
        raise ValueError(f"{n_experts} experts do not divide EP {ep}")
    per = n_experts // ep
    for layer, shards in layout.items():
        if len(shards) != ep or any(len(s) != per for s in shards):
            raise ValueError(
                f"expert layout layer {layer}: expected {ep} shards of {per} "
                f"slots, got {[len(s) for s in shards]}"
            )
        if sorted(e for s in shards for e in s) != list(range(n_experts)):
            raise ValueError(
                f"expert layout layer {layer} is not a permutation of "
                f"{n_experts} experts (replicas are not supported)"
            )


def rank_experts(layout: Layout, layer: int, rank: int) -> Optional[list[int]]:
    shards = layout.get(int(layer))
    return None if shards is None else shards[rank]


def expert_slot_table(layout: Layout, layer: int, n_experts: int) -> torch.Tensor:
    """int32 [E, 1]: the physical slot of each logical expert (kernel
    ``expert_slot`` with one replica)."""
    shards = layout[int(layer)]
    per = len(shards[0])
    table = torch.empty(n_experts, 1, dtype=torch.int32)
    for r, shard in enumerate(shards):
        for j, e in enumerate(shard):
            table[e, 0] = r * per + j
    return table


def routing_mode() -> str:
    from sglang_tpu.envs import envs as tpu_envs

    mode = tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_LAYOUT_ROUTING.get() or "slot"
    if mode not in ("slot", "gate"):
        raise ValueError(
            f"SGLANG_TPU_DSV41_FUSED_EP_LAYOUT_ROUTING must be slot or gate, got {mode!r}"
        )
    return mode


def physical_order(layout: Layout, layer: int) -> list[int]:
    """Logical expert of each physical slot (the flattened shards)."""
    return [e for shard in layout[int(layer)] for e in shard]


def permute_router_to_slots(gate: torch.nn.Module, order: list[int]) -> None:
    """ "gate" routing: router weight rows and score bias in physical-slot order,
    so the logits' column p is logical expert ``order[p]``. Each logit column
    is the same dot product as before, so the values are unchanged."""
    index = torch.tensor(order, dtype=torch.long, device=gate.weight.device)
    gate.weight.data = gate.weight.data.index_select(0, index).contiguous()
    for name in ("e_score_correction_bias", "e_score_correction_bias_vl"):
        bias = getattr(gate, name, None)
        if bias is not None:
            bias.data = bias.data.index_select(0, index.to(bias.device)).contiguous()


def keep_expert(
    name: str,
    layout: Layout,
    rank: int,
    expert_range: Optional[tuple[int, int]],
) -> Optional[bool]:
    """Whether this rank reads checkpoint tensor ``name``: None for a tensor
    that is not a routed expert; listed layers keep this rank's layout experts,
    others the contiguous ``expert_range`` (None keeps every expert)."""
    m = _EXPERT_RE.match(name)
    if m is None:
        return None
    eid = int(m.group(2))
    layer = _LAYER_RE.search(m.group(1) + ".")
    experts = (
        None
        if layer is None or m.group(1).startswith("mtp")
        else rank_experts(layout, int(layer.group(1)), rank)
    )
    if experts is not None:
        return eid in experts
    return expert_range is None or expert_range[0] <= eid < expert_range[1]


def tag_target_layers(model: torch.nn.Module, layout: Layout) -> int:
    """Mark the target model's FusedMoE layers listed in ``layout`` with their
    EP rank (``place_expert_ids`` then stores experts in slot order)."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    tagged = 0
    for m in model.modules():
        if not isinstance(m, FusedMoE) or int(m.layer_id) not in layout:
            continue
        ep = int(m.moe_ep_size)
        validate({int(m.layer_id): layout[int(m.layer_id)]}, int(m.num_experts), ep)
        if getattr(m, "num_fused_shared_experts", 0):
            raise ValueError("expert layout: fused shared experts are not supported")
        setattr(m, LAYOUT_RANK_ATTR, int(m.moe_ep_rank))
        setattr(m, FILLED_ATTR, set())
        tagged += 1
    return tagged


def place_expert_ids(original, moe, param, expert_id):
    """AROUND FusedMoE._weight_loader_expert_ids: a tagged layer stores
    logical expert ``expert_id`` in its layout slot on this rank, or skips it."""
    rank = getattr(moe, LAYOUT_RANK_ATTR, None)
    if (
        rank is None
        or expert_id is None
        or getattr(param, "_sglang_require_global_experts", False)
    ):
        return original(moe, param, expert_id)
    experts = rank_experts(active_layout() or {}, moe.layer_id, rank)
    if experts is None:
        return original(moe, param, expert_id)
    return _slot_of(moe, experts, int(expert_id))


def _slot_of(moe, experts, expert_id):
    if expert_id in experts:
        slot = experts.index(expert_id)
        getattr(moe, FILLED_ATTR).add(slot)
        yield slot


def solve_lpt(counts: list[int], ep: int) -> list[list[int]]:
    """Greedy longest-processing-time permutation: heaviest expert first onto
    the least loaded rank with a free slot. Each shard is sorted by expert id."""
    per = len(counts) // ep
    load, shards = [0] * ep, [[] for _ in range(ep)]
    for e in sorted(range(len(counts)), key=lambda i: (-counts[i], i)):
        r = min((b for b in range(ep) if len(shards[b]) < per), key=lambda b: load[b])
        shards[r].append(e)
        load[r] += counts[e]
    return [sorted(s) for s in shards]
