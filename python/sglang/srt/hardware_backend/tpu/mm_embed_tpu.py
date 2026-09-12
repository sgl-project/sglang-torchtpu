"""Place multimodal rows using CPU metadata and bucket-sized device operations."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.embedding import (
    Embedding,
    PaddedEmbedding,
    embedding_num_tokens,
    flatten_embedding,
)


def apply_device_mm_embeds(
    input_embeds: torch.Tensor,
    source: torch.Tensor,
    row_indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gather a padded source into the prefill bucket, then select media rows.

    Only the source and prefill buckets specialize; row offsets, valid lengths,
    and item counts are tensor data. No embedding rows pass through the host.
    """
    device = input_embeds.device
    selected = torch.index_select(
        source.to(device=device, dtype=input_embeds.dtype),
        0,
        row_indices.to(device),
    )
    return torch.where(mask.to(device), selected, input_embeds)


def warmup_device_mm_embeds(
    sources: List[PaddedEmbedding],
    token_buckets: List[int],
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Enqueue every bucket pair before waiting for eager async compilation."""
    from torch_tpu._internal.sync.sync import synchronize

    start = time.perf_counter()
    count = 0
    prompts = {}
    for embedding in sources:
        columns = (
            embedding.split_columns(hidden_size)
            if embedding.hidden_size != hidden_size
            else (embedding,)
        )
        for column in columns:
            for part in column.parts:
                for rows in token_buckets:
                    key = (rows, column.hidden_size)
                    if key not in prompts:
                        prompts[key] = torch.zeros(key, dtype=dtype, device=device)
                    result = apply_device_mm_embeds(
                        prompts[key],
                        part.data,
                        torch.zeros(rows, dtype=torch.int32),
                        torch.zeros((rows, 1), dtype=torch.bool),
                    )
                    synchronize(result, wait=False)
                    # Keep one output per prefill shape, rather than retaining
                    # a full prompt matrix for every source/prefill pair.
                    prompts[key] = result
                    count += 1
    synchronize(list(prompts.values()), wait=True)
    logging.getLogger(__name__).info(
        "Warmed %d MM placement bucket pairs in %.2fs",
        count,
        time.perf_counter() - start,
    )


def _merge_mm_embedding(
    input_embeds: torch.Tensor, embedding: Embedding, mask: torch.Tensor
) -> torch.Tensor:
    positions = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if positions.numel() != embedding_num_tokens(embedding):
        raise ValueError(
            "Multimodal placement expects one embedding row per masked position"
        )
    embedding = flatten_embedding(embedding)
    if not isinstance(embedding, PaddedEmbedding):
        embedding = PaddedEmbedding.from_tensor(embedding)

    # Items split from one encoder call share a bucket. Merge their row maps
    # so the device reads that bucket only once, even on repeated cache hits.
    sources = {}
    offset = 0
    rows = input_embeds.shape[0]
    for part in embedding.parts:
        count = part.indices.numel()
        if count:
            key = id(part.data)
            if key not in sources:
                sources[key] = (
                    part.data,
                    torch.zeros(rows, dtype=torch.int32),
                    torch.zeros((rows, 1), dtype=torch.bool),
                )
            _, row_indices, selected = sources[key]
            dest = positions[offset : offset + count]
            row_indices[dest] = part.indices
            selected[dest] = True
        offset += count
    for source, row_indices, selected in sources.values():
        input_embeds = apply_device_mm_embeds(
            input_embeds, source, row_indices, selected
        )
    return input_embeds


def embed_mm_on_tpu(
    *,
    input_embeds: torch.Tensor,
    deepstack_shape: Optional[torch.Size],
    modalities: List[Modality],
    embeddings: List[Optional[Embedding]],
    masks: List[Optional[torch.Tensor]],
    deepstack_embeddings: List[Optional[Embedding]],
    use_deepstack: Dict[Modality, bool],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if all(
        e is None or isinstance(e, torch.Tensor) and e.device.type == "cpu"
        for e in embeddings + deepstack_embeddings
    ):
        return embed_mm_on_host(
            input_embeds=input_embeds,
            deepstack_shape=deepstack_shape,
            modalities=modalities,
            embeddings=embeddings,
            masks=masks,
            deepstack_embeddings=deepstack_embeddings,
            use_deepstack=use_deepstack,
        )

    deepstack = (
        torch.zeros(
            deepstack_shape, device=input_embeds.device, dtype=input_embeds.dtype
        )
        if deepstack_shape is not None
        else None
    )
    for modality, embedding, mask, extra in zip(
        modalities, embeddings, masks, deepstack_embeddings
    ):
        if embedding is not None and mask is not None:
            input_embeds = _merge_mm_embedding(input_embeds, embedding, mask)
        if (
            deepstack is not None
            and use_deepstack.get(modality)
            and extra is not None
            and mask is not None
        ):
            deepstack = _merge_mm_embedding(deepstack, extra, mask)
    return input_embeds, deepstack


def _place_mm_rows(
    *,
    masks: List[Optional[torch.Tensor]],
    embeddings: List[Optional[torch.Tensor]],
    rows: int,
    hidden: int,
    dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Scatter host embeddings into a zero-filled [rows, hidden] host matrix.

    Args:
        masks: Per-modality host bool masks of shape [rows, 1]; None to skip.
        embeddings: Per-modality host embeddings, row-aligned in order to their
            mask's set positions; None to skip.
        rows: Padded token count of the batch.
        hidden: Width of the destination.
        dtype: Destination dtype.

    Returns:
        The filled matrix and the union of the masks that contributed, or None
        when no modality contributed rows.

    Raises:
        ValueError: An embedding's row count does not match its mask.
    """
    out: Optional[torch.Tensor] = None
    merged_mask: Optional[torch.Tensor] = None
    for mask, embedding in zip(masks, embeddings):
        if embedding is None or mask is None:
            continue
        indices = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if indices.numel() != embedding.shape[0]:
            raise ValueError(
                "Multimodal placement expects one embedding row per masked "
                f"position, got {embedding.shape[0]} rows for "
                f"{indices.numel()} positions."
            )
        if out is None:
            out = torch.zeros((rows, hidden), dtype=dtype)
            merged_mask = torch.zeros((rows, 1), dtype=torch.bool)
        out[indices] = embedding.to(dtype)
        merged_mask |= mask
    if out is None:
        return None
    return out, merged_mask


def apply_host_mm_embeds(
    *,
    input_embeds: torch.Tensor,
    host_src: torch.Tensor,
    host_mask: torch.Tensor,
) -> torch.Tensor:
    """Overwrite the masked rows of input_embeds with the host-placed rows."""
    device = input_embeds.device
    return torch.where(
        host_mask.to(device),
        host_src.to(device, input_embeds.dtype),
        input_embeds,
    )


def embed_mm_on_host(
    *,
    input_embeds: torch.Tensor,
    deepstack_shape: Optional[torch.Size],
    modalities: List[Modality],
    embeddings: List[Optional[torch.Tensor]],
    masks: List[Optional[torch.Tensor]],
    deepstack_embeddings: List[Optional[torch.Tensor]],
    use_deepstack: Dict[Modality, bool],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Place multimodal rows on the host, then move them to the device.

    The deepstack grid is zero outside the multimodal rows, so its host matrix
    is already the whole answer and needs no device-side merge.

    Returns:
        The merged input embeds, and the deepstack grid or None when unused.
    """
    rows = input_embeds.shape[0]
    dtype = input_embeds.dtype
    placed = _place_mm_rows(
        masks=masks,
        embeddings=embeddings,
        rows=rows,
        hidden=input_embeds.shape[-1],
        dtype=dtype,
    )
    if placed is not None:
        host_src, merged_mask = placed
        input_embeds = apply_host_mm_embeds(
            input_embeds=input_embeds, host_src=host_src, host_mask=merged_mask
        )

    if deepstack_shape is None:
        return input_embeds, None

    placed_deepstack = _place_mm_rows(
        masks=[
            mask if use_deepstack.get(modality, None) else None
            for modality, mask in zip(modalities, masks)
        ],
        embeddings=deepstack_embeddings,
        rows=rows,
        hidden=deepstack_shape[-1],
        dtype=dtype,
    )
    if placed_deepstack is None:
        return input_embeds, torch.zeros(
            deepstack_shape, device=input_embeds.device, dtype=dtype
        )
    return input_embeds, placed_deepstack[0].to(input_embeds.device, dtype)
