"""In-place merge of staged K/V slots into real decode pages."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

Merge = Tuple[int, int, int]  # (staging_slot, decode_slot, num_slots)

_INT_OF_WIDTH = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def bucket_pages(
    real_pages: Sequence[int], staging_pages: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Pad real and staging page lists to the next power of two using dummy page 0."""
    if len(real_pages) != len(staging_pages):
        raise ValueError("real and staging page lists must pair one-to-one")
    n = len(real_pages)
    size = 1 << max(0, (n - 1).bit_length())
    pad = [0] * (size - n)
    return list(real_pages) + pad, list(staging_pages) + pad


def _slot_pairs(merges: Sequence[Merge], kv_packing: int):
    """((stage_pack, stage_lane), (dst_pack, dst_lane)) for every merged slot."""
    pairs = []
    for stage_slot, decode_slot, num_slots in merges:
        for i in range(num_slots):
            pairs.append(
                (
                    divmod(stage_slot + i, kv_packing),
                    divmod(decode_slot + i, kv_packing),
                )
            )
    return tuple(pairs)


def merge_layer(
    buf: torch.Tensor,
    real_idx: torch.Tensor,
    stage_idx: torch.Tensor,
    merges: Sequence[Merge],
    write_pages: Optional[Callable[..., None]] = None,
) -> None:
    """Merge staged slots into real pages for one layer buffer via write_pages."""
    if write_pages is None:
        from sglang_tpu.kv_page_write import write_pages_torch as write_pages
    target = buf
    if buf.device.type == "cpu":
        target = buf.view(_INT_OF_WIDTH[buf.element_size()])
    pages = target[real_idx]
    staged = target[stage_idx]
    for (sp, sl), (dp, dl) in _slot_pairs(merges, int(buf.shape[3])):
        pages[:, :, dp, dl] = staged[:, :, sp, sl]
    write_pages(real_idx, pages, target)


def merge_staged_pages(
    bufs: Sequence[torch.Tensor],
    real_pages: Sequence[int],
    staging_pages: Sequence[int],
    merges: Sequence[Merge],
    write_pages: Optional[Callable[..., None]] = None,
) -> None:
    """Merge staged slots across all layer buffers in a pool."""
    if not bufs or not real_pages or not merges:
        return
    for buf in bufs:
        if len(buf.shape) != 5:
            raise ValueError(
                f"Raiden reshard merge needs fused 5-D KV buffers, got {tuple(buf.shape)}"
            )
    real, stage = bucket_pages(real_pages, staging_pages)
    device = bufs[0].device
    real_idx = torch.tensor(real, dtype=torch.long, device=device)
    stage_idx = torch.tensor(stage, dtype=torch.long, device=device)
    for buf in bufs:
        merge_layer(buf, real_idx, stage_idx, merges, write_pages)
