"""TPU-specific placement of multimodal embeddings into the token embedding grid.

The vision tower already returns host tensors, so the rows are placed at their
final positions on the host. The device then only selects between two aligned
matrices, instead of running masked_scatter_'s cumsum over num_tokens x hidden
to rediscover that mapping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Modality


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
