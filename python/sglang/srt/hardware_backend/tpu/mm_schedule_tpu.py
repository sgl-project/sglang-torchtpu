"""TPU-specific batching of the multimodal encoder call made by mm_schedule."""

from __future__ import annotations

import logging
from typing import List

import torch

from sglang.srt.managers.mm_schedule import DataEmbeddingFunc
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.multimodal.embedding import Embedding, concat_embeddings
from sglang.srt.multimodal.evs import EVSEmbeddingResult

logger = logging.getLogger(__name__)


def _patch_budget() -> int:
    """Patches one vision-encoder call may carry."""
    from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
        precompiled_max_patch_count,
    )

    return precompiled_max_patch_count()


def _item_patch_grid(item: MultimodalDataItem) -> object | None:
    if item.is_video():
        return item.model_specific_data.get("video_grid_thw")
    if item.is_image():
        return item.model_specific_data.get("image_grid_thw")
    return None


def _item_patch_count(item: MultimodalDataItem) -> int:
    grid = _item_patch_grid(item)
    if grid is None:
        raise ValueError("multimodal item does not expose a vision patch grid")
    return int(torch.as_tensor(grid).prod().item())


def _split_items_by_patch_budget(
    items: List[MultimodalDataItem], budget: int
) -> List[List[MultimodalDataItem]]:
    """Greedily pack items into groups whose total patch count fits the budget."""
    groups: List[List[MultimodalDataItem]] = []
    current: List[MultimodalDataItem] = []
    current_patches = 0
    for item in items:
        patches = _item_patch_count(item)
        if current and current_patches + patches > budget:
            groups.append(current)
            current, current_patches = [], 0
        current.append(item)
        current_patches += patches
    if current:
        groups.append(current)
    return groups


def encode_vision_items(
    data_embedding_func: DataEmbeddingFunc,
    items: List[MultimodalDataItem],
) -> Embedding | List[Embedding] | EVSEmbeddingResult:
    """Encode items as several calls that each fit the precompiled patch budget.

    A batch is concatenated into one encoder call, and TPU precompiles that call
    for a bounded set of patch counts, so a batch above the largest one falls off
    the set and recompiles.
    """
    # Some external processors derive patch geometry from feature tensors and
    # own their batching. Apply this generic split only when every item exposes
    # the grid metadata it requires.
    if any(_item_patch_grid(item) is None for item in items):
        return data_embedding_func(items)

    budget = _patch_budget()
    groups = _split_items_by_patch_budget(items, budget)
    if len(groups) == 1:
        return data_embedding_func(items)

    logger.info(
        "Splitting the vision batch: %d items / %d patches exceed the encoder "
        "patch budget (%d), so it runs as %d calls of %s patches.",
        len(items),
        sum(_item_patch_count(item) for item in items),
        budget,
        len(groups),
        [sum(_item_patch_count(item) for item in group) for group in groups],
    )
    outputs = [data_embedding_func(group) for group in groups]
    if isinstance(outputs[0], list):
        return [embedding for output in outputs for embedding in output]
    return concat_embeddings(outputs)
