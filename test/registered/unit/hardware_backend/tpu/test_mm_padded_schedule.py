"""CPU coverage for padded TPU embedding scheduling and cache ownership."""

import sys

import pytest
import torch

from sglang.srt.hardware_backend.tpu import mm_schedule_tpu
from sglang.srt.hardware_backend.tpu.mm_embedding import (
    PaddedEmbedding,
    TpuEmbeddingResult,
)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
