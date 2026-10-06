"""TPU multimodal row placement using host-built indices and masks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.mm_embedding import (
    MultimodalEmbedding,
    PaddedEmbedding,
)


def scatter_mm_on_device(
    dest: torch.Tensor, mask: torch.Tensor, src: MultimodalEmbedding
) -> None:
    """Merge device embeddings using the scheduler's host row mask."""
    if mask.device.type != "cpu":
        raise ValueError("TPU multimodal placement requires a host mask")
    num_rows = int(mask.sum())
    if num_rows != src.shape[0]:
        raise RuntimeError(
            "Multimodal placement expects one embedding row per masked "
            f"position, got {src.shape[0]} rows for {num_rows} positions."
        )
    if num_rows == 0:
        return
    if isinstance(src, PaddedEmbedding):
        _scatter_padded_embedding(dest, mask, src)
        return

    # Keep indices at the padded token count; only their values depend on
    # image lengths. The elementwise fallback scans tokens * hidden instead.
    indices = (torch.cumsum(mask.reshape(-1), dim=0, dtype=torch.int32) - 1).clamp_(
        min=0
    )
    _scatter_mm_rows(dest, src, indices, mask)


def _scatter_mm_rows(
    dest: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    src = src.to(device=dest.device, dtype=dest.dtype)
    if src.shape[0] < dest.shape[0]:
        # Gather compilation must depend on the text bucket, not image length.
        src = F.pad(src, (0, 0, 0, dest.shape[0] - src.shape[0]))
    rows = src.index_select(0, indices.to(dest.device))
    dest.copy_(torch.where(mask.to(dest.device), rows, dest))


def _scatter_padded_embedding(
    dest: torch.Tensor, mask: torch.Tensor, src: PaddedEmbedding
) -> None:
    positions = mask.reshape(-1).nonzero(as_tuple=True)[0]
    # One merge per distinct physical buffer, even when several requests reuse
    # different spans of the same cached image.
    placements: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    offset = 0
    for span in src.spans:
        if span.length == 0:
            continue
        key = id(span.buffer)
        if key not in placements:
            placements[key] = (
                span.buffer,
                torch.zeros(dest.shape[0], dtype=torch.int32),
                torch.zeros(dest.shape[0], 1, dtype=torch.bool),
            )
        _buffer, indices, active = placements[key]
        target_rows = positions[offset : offset + span.length]
        indices[target_rows] = torch.arange(
            span.start, span.start + span.length, dtype=torch.int32
        )
        active[target_rows] = True
        offset += span.length

    for buffer, indices, active in placements.values():
        _scatter_mm_rows(dest, buffer, indices, active)


def _place_mm_rows(
    *,
    masks: List[Optional[torch.Tensor]],
    embeddings: List[Optional[MultimodalEmbedding]],
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
        if isinstance(embedding, PaddedEmbedding):
            embedding = embedding.to_host()
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
    embeddings: List[Optional[MultimodalEmbedding]],
    masks: List[Optional[torch.Tensor]],
    deepstack_embeddings: List[Optional[MultimodalEmbedding]],
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
