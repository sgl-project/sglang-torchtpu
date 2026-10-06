from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.mm_embed_tpu import scatter_mm_on_device
from sglang.srt.managers import mm_schedule as mm_utils
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.mm_embedding import (
    PaddedEmbedding,
    concat_embeddings,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize("cpu_ids", [[], [1, 99, 99, 2, 100], [99] * 8])
def test_mm_mask_uses_cpu_prefix_without_device_readback(cpu_ids):
    # Meta tensors have no data to read back, just like the device-only boundary
    # this path must avoid. Placeholder zero must not select padding/decode rows.
    device_ids = torch.empty(8, dtype=torch.long, device="meta")
    original_ids = torch.tensor(cpu_ids, dtype=torch.long)
    with patch.object(mm_utils, "_is_tpu", True):
        actual = mm_utils._get_multimodal_mask(
            device_ids, torch.tensor([0, 99, 100]), original_ids
        )
    expected = torch.tensor(
        [token in (0, 99, 100) for token in cpu_ids] + [False] * (8 - len(cpu_ids)),
        dtype=torch.bool,
    ).unsqueeze(-1)
    assert torch.equal(actual, expected)
    assert original_ids.tolist() == cpu_ids


def test_mm_mask_rejects_cpu_prefix_longer_than_device_input():
    with patch.object(mm_utils, "_is_tpu", True):
        with pytest.raises(ValueError, match="prefix exceeds"):
            mm_utils._get_multimodal_mask(
                torch.empty(2, device="meta"), torch.tensor([1]), torch.ones(3)
            )


@pytest.mark.parametrize(
    "is_tpu,layout",
    [(False, "tensor"), (True, "tensor"), (True, "padded"), (True, "padded_combined")],
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
    is_tpu, layout, offsets, prefixes, lengths
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
        patch.object(mm_utils, "_is_tpu", is_tpu),
        patch.object(mm_utils, "_is_hip", False),
        patch.object(mm_utils, "_is_npu", False),
        patch.object(mm_utils, "_is_xpu", False),
        patch.object(mm_utils, "embedding_cache", cache),
        patch.object(
            mm_utils,
            "_encode_vision_items",
            side_effect=encode_items,
        ) as encode,
    ):
        for _ in range(2):
            with (
                patch.object(torch, "cat", wraps=torch.cat) as cat,
                patch.object(torch, "concat", wraps=torch.concat) as concat,
            ):
                chunk, result_ids = mm_utils._get_chunked_prefill_embedding(
                    data_embedding_func=Mock(),
                    embedding_items=items,
                    items_size=items_size,
                    prefix_length=prefixes,
                    extend_length=lengths,
                    items_offset_list=offsets,
                    input_ids=input_ids,
                )
                if padded or (is_tpu and len(items) == 1):
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
        patch.object(mm_utils, "_is_tpu", True),
        patch.object(mm_utils, "_is_hip", False),
        patch.object(mm_utils, "_is_npu", False),
        patch.object(mm_utils, "_is_xpu", False),
        patch.object(mm_utils, "embedding_cache", cache),
        patch.object(
            mm_utils, "_encode_vision_items", side_effect=lambda fn, batch: fn(batch)
        ),
        patch.object(mm_utils, "_acknowledge_deferred_cuda_ipc_cache_hits"),
    ):
        for _ in range(2):
            with patch.object(
                torch, "cat", side_effect=AssertionError("must retain padded buffers")
            ):
                chunk, _ = mm_utils._get_chunked_prefill_embedding(
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
    assert cache.set(1, EmbeddingResult(embedding=rows))
    assert cache.current_size == 128
    assert cache.get_single(1).embedding is rows
    assert cache.set(
        2, EmbeddingResult(embedding=PaddedEmbedding.from_buffer(buffer.clone(), 1))
    )
    assert cache.get_single(1) is None
    assert cache.current_size == 128
    cache.free(2, None)
    assert cache.current_size == 0


@pytest.mark.parametrize(
    (
        "prefix_length",
        "extend_length",
        "items_offset_list",
        "expected",
    ),
    [
        ([8], [16], [[(2, 5), (9, 14), (20, 24)]], 10),
        ([30], [0], [[(2, 5), (9, 14), (20, 24)]], 0),
        (
            [4, 0, 10],
            [4, 10, 10],
            [[(2, 5)], [], [(5, 12), (18, 25)]],
            7,
        ),
    ],
)
def test_count_mm_tokens_in_extend(
    prefix_length, extend_length, items_offset_list, expected
):
    input_ids = []
    for prefix, extend, item_offsets in zip(
        prefix_length, extend_length, items_offset_list
    ):
        seq_len = max(
            prefix + extend,
            max((item_end + 1 for _, item_end in item_offsets), default=0),
        )
        req_input_ids = torch.zeros(seq_len, dtype=torch.long)
        for item_start, item_end in item_offsets:
            req_input_ids[item_start : item_end + 1] = 1
        input_ids.append(req_input_ids[prefix : prefix + extend])

    actual = torch.isin(torch.cat(input_ids), torch.tensor([1])).sum().item()
    derived = mm_utils._count_mm_tokens_in_extend(
        prefix_length=prefix_length,
        extend_length=extend_length,
        items_offset_list=items_offset_list,
    )
    assert actual == derived == expected


def test_get_embedding_and_mask_uses_offset_count_without_readback():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    cpu_ids = input_ids.clone()
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = Mock()
    mask.sum.side_effect = AssertionError("mask count must stay on device")

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False),
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=embedding),
        patch.object(mm_utils, "_get_multimodal_mask", return_value=mask) as get_mask,
    ):
        result, result_mask, result_input_ids = mm_utils.get_embedding_and_mask(
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


def test_get_embedding_and_mask_async_asserts_offset_count():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True),
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=embedding),
        patch.object(mm_utils.torch, "_assert_async") as assert_async,
    ):
        mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    assert_async.assert_called_once()
    condition, message = assert_async.call_args.args
    assert condition.item()
    assert "derived from offsets" in message


def test_adjust_embedding_length_crops_overlong_embedding():
    embedding = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    server_args = Mock(chunked_prefill_size=-1)

    with patch.object(mm_utils, "get_schedule", return_value=server_args):
        result = mm_utils._adjust_embedding_length(embedding, 3, Mock())

    torch.testing.assert_close(result, embedding[-3:], rtol=0, atol=0)


def test_adjust_embedding_length_rejects_short_embedding():
    embedding = torch.zeros(2, 4)

    with pytest.raises(RuntimeError, match="Insufficient multimodal embedding length"):
        mm_utils._adjust_embedding_length(embedding, 3, Mock())


def test_get_embedding_and_mask_falls_back_after_input_ids_rewrite():
    input_ids = torch.zeros(8, dtype=torch.long)
    rewritten_input_ids = input_ids.clone()
    embedding = torch.zeros(2, 4)
    mask_sum = Mock()
    mask_sum.item.return_value = 2
    mask = Mock()
    mask.sum.return_value = mask_sum

    with (
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=None),
        patch.object(
            mm_utils,
            "_get_chunked_prefill_embedding",
            return_value=(embedding, rewritten_input_ids),
        ),
        patch.object(mm_utils, "_get_multimodal_mask", return_value=mask) as get_mask,
    ):
        result, result_mask, result_input_ids = mm_utils.get_embedding_and_mask(
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
