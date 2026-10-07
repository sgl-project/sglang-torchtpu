"""CPU coverage for padded TPU embedding scheduling and cache ownership."""

from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import mm_schedule_tpu
from sglang.srt.hardware_backend.tpu.mm_embed_tpu import scatter_mm_on_device
from sglang.srt.hardware_backend.tpu.mm_embedding import (
    PaddedEmbedding,
    TpuEmbeddingResult,
    concat_embeddings,
)
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize("cpu_ids", [[], [1, 99, 99, 2, 100], [99] * 8])
def test_mm_mask_uses_cpu_prefix_without_device_readback(cpu_ids):
    # Meta tensors have no data to read back, just like the device-only boundary
    # this path must avoid. Placeholder zero must not select padding/decode rows.
    device_ids = torch.empty(8, dtype=torch.long, device="meta")
    original_ids = torch.tensor(cpu_ids, dtype=torch.long)
    actual = mm_schedule_tpu._get_multimodal_mask(
        device_ids, torch.tensor([0, 99, 100]), original_ids
    )
    expected = torch.tensor(
        [token in (0, 99, 100) for token in cpu_ids] + [False] * (8 - len(cpu_ids)),
        dtype=torch.bool,
    ).unsqueeze(-1)
    assert torch.equal(actual, expected)
    assert original_ids.tolist() == cpu_ids


def test_mm_mask_rejects_cpu_prefix_longer_than_device_input():
    with pytest.raises(ValueError, match="prefix exceeds"):
        mm_schedule_tpu._get_multimodal_mask(
            torch.empty(2, device="meta"), torch.tensor([1]), torch.ones(3)
        )


@pytest.mark.parametrize(
    "layout",
    ["tensor", "padded", "padded_combined"],
)
@pytest.mark.parametrize(
    "offsets,prefixes,lengths",
    [
        ([[(2, 5)]], [4], [3]),
        ([[(2, 5), (8, 11)]], [4], [6]),
        ([[(2, 5)], [(1, 3)]], [4, 0], [3, 4]),
    ],
)
def test_chunk_assembly_and_merge_preserve_cached_images(
    layout, offsets, prefixes, lengths
):
    padded = layout != "tensor"
    items, embeddings, chunks, masks = [], {}, [], []
    items_size = [0]
    for req_offsets, prefix, length in zip(offsets, prefixes, lengths):
        mask = torch.zeros(length, 1, dtype=torch.bool)
        for start, end in req_offsets:
            item_hash = len(items) + 1
            rows = end - start + 1
            embedding = (
                torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)[:, ::2]
                + item_hash * 100
            )
            embeddings[item_hash] = embedding
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    hash=item_hash,
                    offsets=[(start, end)],
                    feature=torch.zeros(1),
                )
            )
            overlap_start = max(start, prefix)
            overlap_end = min(end + 1, prefix + length)
            chunks.append(embedding[overlap_start - start : overlap_end - start])
            mask[overlap_start - prefix : overlap_end - prefix] = True
        items_size.append(len(items))
        masks.append(mask)

    expected_chunk = torch.cat(chunks)
    encoded = {}
    for item_hash, embedding in embeddings.items():
        if padded:
            buffer = torch.full((8, embedding.shape[1]), torch.nan)
            buffer[: embedding.shape[0]] = embedding
            encoded[item_hash] = PaddedEmbedding.from_buffer(buffer, embedding.shape[0])
        else:
            encoded[item_hash] = embedding
    mask = torch.cat(masks)
    input_ids = torch.zeros(sum(lengths), dtype=torch.long)
    dest = torch.arange(sum(lengths) * 4, dtype=torch.float32).reshape(-1, 4)
    expected = dest.clone().masked_scatter_(mask.expand_as(dest), expected_chunk)
    cache = MultiModalStaticCache(max_size=4096)

    def encode_items(_embedder, misses):
        results = [encoded[item.hash] for item in misses]
        return concat_embeddings(results) if layout == "padded_combined" else results

    with (
        patch.object(mm_schedule, "embedding_cache", cache),
        patch.object(
            mm_schedule_tpu,
            "_encode_vision_items",
            side_effect=encode_items,
        ) as encode,
    ):
        for _ in range(2):
            with (
                patch.object(torch, "cat", wraps=torch.cat) as cat,
                patch.object(torch, "concat", wraps=torch.concat) as concat,
            ):
                chunk, result_ids = mm_schedule_tpu._get_chunked_prefill_embedding(
                    data_embedding_func=Mock(),
                    embedding_items=items,
                    items_size=items_size,
                    prefix_length=prefixes,
                    extend_length=lengths,
                    items_offset_list=offsets,
                    input_ids=input_ids,
                )
                if padded or len(items) == 1:
                    cat.assert_not_called()
                    concat.assert_not_called()
            assert result_ids is input_ids
            assert torch.equal(
                chunk.to_host() if isinstance(chunk, PaddedEmbedding) else chunk,
                expected_chunk,
            )
            actual = dest.clone()
            scatter_mm_on_device(actual, mask, chunk)
            assert torch.equal(actual, expected)
            for item_hash, embedding in embeddings.items():
                cached = cache.get_single(item_hash).embedding
                assert torch.equal(
                    cached.to_host() if isinstance(cached, PaddedEmbedding) else cached,
                    embedding,
                )
                if padded:
                    assert cached.spans[0].buffer is encoded[item_hash].spans[0].buffer
            if padded:
                assert cache.current_size == sum(
                    embedding.storage_nbytes() for embedding in encoded.values()
                )
        encode.assert_called_once()


def test_padded_full_request_and_per_image_chunks_share_batch_order():
    offsets = [[(2, 3), (6, 7), (10, 12)], [(1, 2)]]
    prefixes, lengths = [3, 0], [9, 4]
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=i + 1,
            offsets=item_offsets,
            feature=torch.zeros(1),
        )
        for i, item_offsets in enumerate([offsets[0][:2], offsets[0][2:], offsets[1]])
    ]
    buffers = [torch.arange(32).reshape(8, 4).float() + i * 100 for i in range(3)]
    embeddings = [
        PaddedEmbedding.from_buffer(buffer, rows)
        for buffer, rows in zip(buffers, [4, 3, 2])
    ]
    encode = Mock(
        side_effect=lambda batch: [embeddings[item.hash - 1] for item in batch]
    )
    expected_rows = torch.cat([buffers[0][1:4], buffers[1][:2], buffers[2][:2]])
    mask = torch.zeros(13, 1, dtype=torch.bool)
    mask[[0, 3, 4, 7, 8, 10, 11]] = True
    cache = MultiModalStaticCache(4096)
    with (
        patch.object(mm_schedule, "embedding_cache", cache),
        patch.object(
            mm_schedule_tpu,
            "_encode_vision_items",
            side_effect=lambda fn, batch: fn(batch),
        ),
        patch.object(mm_schedule_tpu, "_acknowledge_deferred_cuda_ipc_cache_hits"),
    ):
        for _ in range(2):
            with patch.object(
                torch, "cat", side_effect=AssertionError("must retain padded buffers")
            ):
                chunk, _ = mm_schedule_tpu._get_chunked_prefill_embedding(
                    encode,
                    items,
                    [0, 2, 3],
                    prefixes,
                    lengths,
                    offsets,
                    torch.zeros(13, dtype=torch.long),
                )
            assert isinstance(chunk, PaddedEmbedding)
            assert torch.equal(chunk.to_host(), expected_rows)
            dest = torch.full((13, 4), -1.0)
            expected = dest.clone().masked_scatter_(mask.expand_as(dest), expected_rows)
            scatter_mm_on_device(dest, mask, chunk)
            assert torch.equal(dest, expected)
    assert encode.call_count == 2
    assert cache.current_size == sum(
        buffer.untyped_storage().nbytes() for buffer in buffers
    )


def test_padded_cache_charges_full_buffer_and_evicts_by_physical_size():
    buffer = torch.arange(32).reshape(8, 4).float()
    rows = PaddedEmbedding.from_buffer(buffer, 2)
    cache = MultiModalStaticCache(buffer.untyped_storage().nbytes())
    assert cache.set(1, TpuEmbeddingResult(embedding=rows))
    assert cache.current_size == 128
    assert cache.get_single(1).embedding is rows
    assert cache.set(
        2, TpuEmbeddingResult(embedding=PaddedEmbedding.from_buffer(buffer.clone(), 1))
    )
    assert cache.get_single(1) is None
    assert cache.current_size == 128
    cache.free(2, None)
    assert cache.current_size == 0


@pytest.mark.parametrize("entry_type", [EmbeddingResult, TpuEmbeddingResult])
def test_tensor_cache_compacts_views_and_preserves_admission_and_eviction(entry_type):
    buffer = torch.arange(64).reshape(8, 8).float()
    rows = buffer[1:3, ::2]
    row_bytes = rows.numel() * rows.element_size()
    cache = MultiModalStaticCache(row_bytes)
    assert cache.set(1, entry_type(embedding=rows))
    cached = cache.get_single(1).embedding
    assert torch.equal(cached, rows)
    assert cached.untyped_storage().nbytes() == row_bytes
    assert cached.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()
    assert cache.current_size == row_bytes

    assert cache.set(2, entry_type(embedding=rows + 1))
    assert cache.get_single(1) is None
    assert torch.equal(cache.get_single(2).embedding, rows + 1)
    assert cache.current_size == row_bytes
    cache.clear()
    assert cache.current_size == 0
    assert cache.get_single(2) is None

    assert not cache.set(3, entry_type(embedding=torch.zeros(3, 4)))
    assert cache.current_size == 0
    assert cache.get_single(3) is None


def test_tpu_scheduling_observes_shared_cache_clear_and_reinitialization():
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=123,
        offsets=[(1, 2)],
        feature=torch.zeros(1),
    )
    rows = PaddedEmbedding.from_buffer(torch.arange(32).reshape(8, 4).float(), 2)
    encoder = Mock(return_value=[rows])
    ids = torch.tensor([0, 123, 123, 0])
    with patch.object(mm_schedule, "embedding_cache", None):
        mm_schedule.init_mm_embedding_cache(4096)

        def get_chunk():
            result, _ = mm_schedule_tpu._get_chunked_prefill_embedding(
                encoder, [item], [0, 1], [0], [4], [[(1, 2)]], ids
            )
            assert result.spans[0].buffer is rows.spans[0].buffer

        get_chunk()
        get_chunk()
        encoder.assert_called_once()
        cache = mm_schedule.embedding_cache
        assert cache.current_size == rows.storage_nbytes()
        cache.clear()
        assert cache.current_size == 0
        get_chunk()
        assert encoder.call_count == 2
        mm_schedule.init_mm_embedding_cache(2048)
        assert mm_schedule.embedding_cache is not cache
        get_chunk()
        assert encoder.call_count == 3
        assert mm_schedule.embedding_cache.current_size == rows.storage_nbytes()


def test_get_embedding_and_mask_uses_offset_count_without_readback():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    cpu_ids = input_ids.clone()
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = Mock()
    mask.sum.side_effect = AssertionError("mask count must stay on device")

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False),
        patch.object(
            mm_schedule_tpu, "_get_precomputed_embedding", return_value=embedding
        ),
        patch.object(
            mm_schedule_tpu, "_get_multimodal_mask", return_value=mask
        ) as get_mask,
    ):
        result, result_mask, result_input_ids = mm_schedule_tpu.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            input_ids_cpu=cpu_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    assert get_mask.call_args.args[2] is cpu_ids
    mask.sum.assert_not_called()
    assert result is embedding
    assert result_mask is mask
    assert result_input_ids is input_ids


def test_get_embedding_and_mask_falls_back_after_input_ids_rewrite():
    input_ids = torch.zeros(8, dtype=torch.long)
    rewritten_input_ids = input_ids.clone()
    embedding = torch.zeros(2, 4)
    mask_sum = Mock()
    mask_sum.item.return_value = 2
    mask = Mock()
    mask.sum.return_value = mask_sum

    with (
        patch.object(mm_schedule_tpu, "_get_precomputed_embedding", return_value=None),
        patch.object(
            mm_schedule_tpu,
            "_get_chunked_prefill_embedding",
            return_value=(embedding, rewritten_input_ids),
        ),
        patch.object(
            mm_schedule_tpu, "_get_multimodal_mask", return_value=mask
        ) as get_mask,
    ):
        result, result_mask, result_input_ids = mm_schedule_tpu.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            input_ids_cpu=input_ids.clone(),
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    assert get_mask.call_args.args[2] is None
    mask.sum.assert_called_once_with()
    mask_sum.item.assert_called_once_with()
    assert result is embedding
    assert result_mask is mask
    assert result_input_ids is rewritten_input_ids
