"""TPU multimodal row placement using host-built indices and masks."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_disagg, get_server_args
from sglang.srt.utils import flatten_nested_list
from sglang_tpu.mm_embedding import (
    MultimodalEmbedding,
    PaddedEmbedding,
)
from sglang_tpu.mm_schedule_tpu import (
    DataEmbeddingFunc,
    get_embedding_and_mask,
)


def merge_mm_rows(
    dest: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    src = src.to(dtype=dest.dtype)
    if src.shape[0] < dest.shape[0]:
        src = F.pad(src, (0, 0, 0, dest.shape[0] - src.shape[0]))
    return torch.where(mask, src.index_select(0, indices), dest)


def scatter_mm_on_device(
    dest: torch.Tensor,
    mask: torch.Tensor,
    src: MultimodalEmbedding,
    *,
    merge_rows: Callable = merge_mm_rows,
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
        _scatter_padded_embedding(dest, mask, src, merge_rows=merge_rows)
        return

    # Keep indices at the padded token count; only their values depend on
    # image lengths. The elementwise fallback scans tokens * hidden instead.
    flat = mask.reshape(-1)
    indices = torch.cumsum(flat, dim=0, dtype=torch.int32) - 1
    # Rows outside the mask read their own row; repeated indices slow the gather.
    indices = torch.where(flat, indices, torch.arange(flat.numel(), dtype=torch.int32))
    _scatter_mm_rows(dest, src, indices, mask, merge_rows=merge_rows)


def _scatter_mm_rows(
    dest: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    *,
    merge_rows: Callable = merge_mm_rows,
) -> None:
    dest.copy_(
        merge_rows(
            dest,
            src.to(device=dest.device, dtype=dest.dtype),
            indices.to(dest.device),
            mask.to(dest.device),
        )
    )


def _scatter_padded_embedding(
    dest: torch.Tensor,
    mask: torch.Tensor,
    src: PaddedEmbedding,
    *,
    merge_rows: Callable = merge_mm_rows,
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
                # Inactive rows read their own row; repeated indices slow the gather.
                torch.arange(dest.shape[0], dtype=torch.int32),
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
        _scatter_mm_rows(dest, buffer, indices, active, merge_rows=merge_rows)


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


def _embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: torch.nn.Embedding,
    multimodal_model: torch.nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Optional[Dict[Modality, bool]] = None,
    input_ids_cpu: Optional[torch.Tensor] = None,
    merge_rows: Callable = merge_mm_rows,
) -> Tuple[torch.Tensor, dict]:
    use_deepstack = use_deepstack or {}
    other_info = {}

    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    # deepstack_embeddings: per-modality
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            pad_values = [item.pad_value for item in items]
            placeholder_tensor = torch.as_tensor(pad_values, device="cpu")
            # calculate per request items length offset
            items_size = [0]
            items_offsets = []
            for mm_inputs in mm_inputs_list:
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size.append(items_size[-1] + len(mm_items))
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )

            original_input_ids = input_ids
            embedding, mask, input_ids = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
                input_ids_cpu=input_ids_cpu,
            )
            if input_ids is not original_input_ids:
                # EVS rewrites token positions for all subsequent modalities too.
                input_ids_cpu = None

            if use_deepstack.get(modality, None) and embedding is not None:
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            else:
                deepstack_embeddings += [None]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Build masks before replacing image hash placeholders with vocabulary IDs.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # deepstack embedding
    if use_deepstack:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)

        deepstack_embedding_shape = input_embeds.shape[:-1] + (
            input_embeds.shape[-1] * num_deepstack_embeddings,
        )

    # 4. scatter embeddings into input embedding
    if not envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
        input_embeds, input_deepstack_embeds = embed_mm_on_host(
            input_embeds=input_embeds,
            deepstack_shape=deepstack_embedding_shape if use_deepstack else None,
            modalities=modalities,
            embeddings=embeddings,
            masks=masks,
            deepstack_embeddings=deepstack_embeddings,
            use_deepstack=use_deepstack,
        )
        if use_deepstack:
            other_info["input_deepstack_embeds"] = input_deepstack_embeds
        return input_embeds, other_info

    if use_deepstack:
        input_deepstack_embeds = torch.zeros(
            deepstack_embedding_shape,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        scatter_mm_on_device(
            dest=input_embeds, mask=mask, src=embedding, merge_rows=merge_rows
        )
        if use_deepstack.get(modality, None):
            scatter_mm_on_device(
                dest=input_deepstack_embeds,
                mask=mask,
                src=deepstack_embeddings[i],
                merge_rows=merge_rows,
            )

    return input_embeds, other_info


def _embed_mm_inputs_with_split(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embedding: torch.nn.Embedding,
    multimodal_model: torch.nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Optional[Dict[Modality, bool]] = None,
    input_ids_cpu: Optional[torch.Tensor] = None,
    merge_rows: Callable = merge_mm_rows,
) -> Tuple[torch.Tensor, dict]:
    use_deepstack = use_deepstack or {}
    precomputed_req_indices = []
    non_precomputed_req_indices = []
    for idx, mm_input in enumerate(mm_inputs_list):
        items = [item for item in mm_input.mm_items if item is not None]
        if items and all(item.precomputed_embeddings is not None for item in items):
            precomputed_req_indices.append(idx)
        else:
            non_precomputed_req_indices.append(idx)

    embed_kwargs = dict(
        multimodal_model=multimodal_model,
        input_embedding=input_embedding,
        data_embedding_func_mapping=data_embedding_func_mapping,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack,
        merge_rows=merge_rows,
    )

    if not precomputed_req_indices or not non_precomputed_req_indices:
        return _embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            input_ids_cpu=input_ids_cpu,
            **embed_kwargs,
        )

    all_seq_lens = forward_batch.extend_seq_lens_cpu
    mm_batch_indices = [
        i for i, mm in enumerate(forward_batch.mm_inputs) if mm is not None
    ]
    token_starts = []
    cumulative = 0
    for sl in all_seq_lens:
        token_starts.append(cumulative)
        cumulative += sl

    vocab_size = input_embedding.num_embeddings
    input_embeds = input_embedding(input_ids.clamp(min=0, max=vocab_size - 1))
    other_info = {}

    input_deepstack_embeds = None
    if use_deepstack and multimodal_model is not None:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)
        input_deepstack_embeds = torch.zeros(
            input_ids.shape[0],
            input_embedding.embedding_dim * num_deepstack_embeddings,
            device=input_ids.device,
            dtype=input_embedding.weight.dtype,
        )
        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    for group_req_indices in [precomputed_req_indices, non_precomputed_req_indices]:
        sub_mm_inputs = [mm_inputs_list[i] for i in group_req_indices]
        sub_prefix_lens = [extend_prefix_lens[i] for i in group_req_indices]
        sub_seq_lens = [extend_seq_lens[i] for i in group_req_indices]
        group_batch_indices = [mm_batch_indices[i] for i in group_req_indices]
        sub_slices = [
            input_ids[token_starts[bi] : token_starts[bi] + all_seq_lens[bi]]
            for bi in group_batch_indices
        ]
        sub_input_ids = torch.cat(sub_slices)
        sub_input_ids_cpu = (
            torch.cat(
                [
                    input_ids_cpu[
                        token_starts[bi] : token_starts[bi] + all_seq_lens[bi]
                    ]
                    for bi in group_batch_indices
                ]
            )
            if input_ids_cpu is not None
            else None
        )

        sub_embeds, sub_info = _embed_mm_inputs(
            mm_inputs_list=sub_mm_inputs,
            extend_prefix_lens=sub_prefix_lens,
            extend_seq_lens=sub_seq_lens,
            input_ids=sub_input_ids,
            input_ids_cpu=sub_input_ids_cpu,
            **embed_kwargs,
        )

        offset = 0
        for bi in group_batch_indices:
            req_len = all_seq_lens[bi]
            start = token_starts[bi]
            input_embeds[start : start + req_len] = sub_embeds[
                offset : offset + req_len
            ]
            if (
                input_deepstack_embeds is not None
                and "input_deepstack_embeds" in sub_info
            ):
                input_deepstack_embeds[start : start + req_len] = sub_info[
                    "input_deepstack_embeds"
                ][offset : offset + req_len]
            offset += req_len

    return input_embeds, other_info


def embed_mm_inputs_tpu(
    *,
    forward_batch: ForwardBatch,
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: torch.nn.Embedding,
    multimodal_model: Optional[torch.nn.Module] = None,
    data_embedding_func_mapping: Optional[Dict[Modality, DataEmbeddingFunc]] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    use_deepstack: Optional[Dict[Modality, bool]] = None,
    merge_rows: Callable = merge_mm_rows,
) -> Tuple[torch.Tensor, dict]:
    """Prepare TPU multimodal rows; padded embeddings never escape this boundary."""
    kwargs = dict(
        mm_inputs_list=mm_inputs_list,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        input_ids=input_ids,
        input_ids_cpu=forward_batch.mm_input_ids_cpu,
        input_embedding=input_embedding,
        multimodal_model=multimodal_model,
        data_embedding_func_mapping=data_embedding_func_mapping,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack or {},
        merge_rows=merge_rows,
    )
    if get_server_args() and get_disagg().enable_adaptive_dispatch_to_encoder:
        return _embed_mm_inputs_with_split(forward_batch=forward_batch, **kwargs)
    return _embed_mm_inputs(**kwargs)
