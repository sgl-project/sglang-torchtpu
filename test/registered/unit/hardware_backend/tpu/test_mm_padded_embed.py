"""CPU coverage for TPU merge indexing and embedding preparation."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.managers.mm_embed_tpu import scatter_mm_on_device
from sglang_tpu.managers.mm_embedding import (
    EmbeddingSpan,
    PaddedEmbedding,
    concat_embeddings,
)

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

NUM_TOKENS = 64


def _make_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    if pattern == "interleaved":
        mask[::3] = True
    elif pattern == "blocks":
        mask[5:20] = True
        mask[40:41] = True
    elif pattern == "all_true":
        mask[:] = True
    return mask.unsqueeze(-1)


@pytest.mark.parametrize("width", [8, 24])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mask_pattern", ["interleaved", "blocks", "all_true", "all_false"]
)
def test_scatter_matches_masked_scatter_bitwise(width, src_dtype, mask_pattern):
    """The row-index mm embedding merge must stay bitwise identical to
    masked_scatter_ semantics, whose internal transients it avoids."""
    torch.default_generator.manual_seed(0)
    mask = _make_mask(mask_pattern)
    dest = torch.randn(NUM_TOKENS, width).to(torch.bfloat16)
    src = torch.randn(int(mask.sum()), width, dtype=src_dtype)

    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    actual = dest.clone()
    scatter_mm_on_device(dest=actual, mask=mask, src=src)
    assert torch.equal(actual, expected)


def test_scatter_row_count_mismatch_fails_loud():
    """A mask/src row-count mismatch must raise, not silently corrupt rows."""
    dest = torch.zeros(8, 4)
    src_short_mask = _make_mask("all_false")[:8]
    src_short_mask[1] = True
    with pytest.raises((RuntimeError, IndexError)):
        scatter_mm_on_device(dest=dest, mask=src_short_mask, src=torch.ones(3, 4))
    mask_heavy = src_short_mask.clone()
    mask_heavy[2:6] = True
    with pytest.raises((RuntimeError, IndexError)):
        scatter_mm_on_device(dest=dest, mask=mask_heavy, src=torch.ones(1, 4))


def test_device_scatter_empty_token_batch():
    dest = torch.empty(0, 8)
    scatter_mm_on_device(dest, torch.empty(0, 1, dtype=torch.bool), torch.empty(0, 8))
    assert dest.shape == (0, 8)


def test_device_scatter_padded_cached_view_preserves_backing_storage():
    cache = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16)
    original_cache = cache.clone()
    src = cache[7:25:2, ::2]
    dest = torch.full((32, 8), -1.0, dtype=torch.bfloat16)
    mask = torch.zeros(32, 1, dtype=torch.bool)
    mask[3:30:3] = True
    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    scatter_mm_on_device(dest, mask, src)

    assert torch.equal(dest, expected)
    assert torch.equal(cache, original_cache)


def test_device_scatter_full_bucket_source_alias():
    dest = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    expected = dest.clone()

    scatter_mm_on_device(dest, torch.ones(32, 1, dtype=torch.bool), dest)

    assert torch.equal(dest, expected)


@pytest.mark.parametrize(
    "physical_rows,token_bucket,valid_lengths",
    [(256, 512, [256, 240]), (512, 1024, [512, 480]), (512, 256, [200, 180])],
)
def test_padded_merge_preserves_physical_bucket_across_valid_lengths(
    physical_rows, token_bucket, valid_lengths
):
    for valid_rows in valid_lengths:
        buffer = torch.arange(physical_rows * 8).reshape(physical_rows, 8).float()
        buffer[valid_rows:] = torch.nan
        src = PaddedEmbedding.from_buffer(buffer, valid_rows)
        mask = torch.zeros(token_bucket, 1, dtype=torch.bool)
        mask[3 : 3 + valid_rows] = True
        dest = torch.full((token_bucket, 8), -7.0, dtype=torch.bfloat16)
        expected = dest.clone().masked_scatter_(
            mask.expand_as(dest), buffer[:valid_rows].to(dest.dtype)
        )
        scatter_mm_on_device(dest, mask, src)
        assert torch.equal(dest, expected)
        assert src.spans[0].buffer is buffer


def test_padded_merge_preserves_multiple_cached_spans():
    buffer = torch.arange(64).reshape(16, 4).float()
    src = PaddedEmbedding((EmbeddingSpan(buffer, 5, 3), EmbeddingSpan(buffer, 1, 2)))
    mask = torch.zeros(16, 1, dtype=torch.bool)
    mask[[1, 4, 5, 9, 13]] = True
    dest = torch.full((16, 4), -1.0)
    expected = dest.clone().masked_scatter_(
        mask.expand_as(dest), torch.cat([buffer[5:8], buffer[1:3]])
    )
    scatter_mm_on_device(dest, mask, src)
    assert torch.equal(dest, expected)


def test_padded_and_unpadded_chunks_merge_in_order():
    first = PaddedEmbedding.from_buffer(torch.arange(32).reshape(8, 4).float(), 3)
    second = torch.full((2, 4), 50.0)
    src = concat_embeddings([first.slice(1, 3), second])
    mask = torch.tensor([False, True, False, True, True, True]).unsqueeze(-1)
    dest = torch.full((6, 4), -1.0)
    expected = dest.clone().masked_scatter_(mask.expand_as(dest), src.to_host())
    scatter_mm_on_device(dest, mask, src)
    assert torch.equal(dest, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
