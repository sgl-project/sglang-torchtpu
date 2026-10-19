"""Equivalence of the vectorised TPU alloc_extend against the reference loop."""

import random
import sys

import pytest
import torch

from sglang.srt.mem_cache.allocator import alloc_extend_naive
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.mem_cache.allocator.allocator_tpu import alloc_extend_vectorized_cpu

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference(prefix_lens, seq_lens, last_loc, free_pages, page_size):
    total = int((seq_lens - prefix_lens).sum())
    out = torch.full((total,), -7, dtype=torch.int32)
    alloc_extend_naive(
        prefix_lens, seq_lens, last_loc, free_pages, out, page_size, "cpu"
    )
    return out


def _vectorized(prefix_lens, seq_lens, last_loc, free_pages, page_size):
    total = int((seq_lens - prefix_lens).sum())
    out = torch.full((total,), -7, dtype=torch.int32)
    alloc_extend_vectorized_cpu(
        prefix_lens, seq_lens, last_loc, free_pages, out, page_size
    )
    return out


def _last_loc_for(prefix_lens, page_size, rng):
    # alloc_extend's debug invariant: (last_loc + 1) % page == prefix % page.
    # Pick an arbitrary page for each request's current partial page.
    pages = torch.tensor(
        [rng.randrange(1, 10_000) for _ in range(len(prefix_lens))], dtype=torch.int64
    )
    return torch.where(
        prefix_lens > 0,
        pages * page_size + (prefix_lens - 1) % page_size,
        torch.full_like(prefix_lens, -1),
    ).to(torch.int32)


def _case(rng, batch_size, page_size, extend_choices):
    prefix = torch.tensor(
        [rng.randrange(0, 4 * page_size + 1) for _ in range(batch_size)],
        dtype=torch.int32,
    )
    extend = torch.tensor(
        [rng.choice(extend_choices) for _ in range(batch_size)], dtype=torch.int32
    )
    seq = prefix + extend
    last_loc = _last_loc_for(prefix.to(torch.int64), page_size, rng)
    num_new_pages = int(
        (
            (seq + page_size - 1) // page_size - (prefix + page_size - 1) // page_size
        ).sum()
    )
    # Non-contiguous free pages, so a wrong stride/base shows up.
    free_pages = torch.tensor(
        sorted(rng.sample(range(20_000, 60_000), num_new_pages)), dtype=torch.int32
    )
    return prefix, seq, last_loc, free_pages


@pytest.mark.parametrize("page_size", [1, 2, 8, 16, 32, 64])
@pytest.mark.parametrize("batch_size", [0, 1, 2, 7, 48, 64])
def test_matches_reference_random(page_size: int, batch_size: int) -> None:
    rng = random.Random(page_size * 1000 + batch_size)
    extend_choices = [
        0,
        1,
        15,
        16,
        17,
        page_size - 1,
        page_size,
        page_size + 1,
        3 * page_size + 5,
    ]
    extend_choices = [max(0, e) for e in extend_choices]
    for _ in range(25):
        prefix, seq, last_loc, free_pages = _case(
            rng, batch_size, page_size, extend_choices
        )
        expected = _reference(prefix, seq, last_loc, free_pages, page_size)
        actual = _vectorized(prefix, seq, last_loc, free_pages, page_size)
        assert torch.equal(actual, expected), (prefix, seq, last_loc, free_pages)


def test_documented_three_segment_example() -> None:
    prefix = torch.tensor([3, 0, 5], dtype=torch.int32)
    seq = torch.tensor([9, 2, 8], dtype=torch.int32)
    last_loc = torch.tensor([22, -1, 48], dtype=torch.int32)
    free_pages = torch.tensor([10, 27, 35, 42], dtype=torch.int32)
    expected = torch.tensor(
        [23, 40, 41, 42, 43, 108, 140, 141, 49, 50, 51], dtype=torch.int32
    )

    assert torch.equal(_reference(prefix, seq, last_loc, free_pages, 4), expected)
    assert torch.equal(_vectorized(prefix, seq, last_loc, free_pages, 4), expected)
    assert free_pages.tolist() == [10, 27, 35, 42]


def test_extend_inside_current_page_uses_no_new_pages() -> None:
    page_size = 16
    prefix = torch.tensor([3, 16, 30], dtype=torch.int32)
    seq = torch.tensor([10, 16, 32], dtype=torch.int32)  # 7 in-page, 0, 2 in-page
    last_loc = torch.tensor([5 * 16 + 2, -1, 9 * 16 + 13], dtype=torch.int32)
    free_pages = torch.empty((0,), dtype=torch.int32)
    expected = _reference(prefix, seq, last_loc, free_pages, page_size)
    assert torch.equal(
        _vectorized(prefix, seq, last_loc, free_pages, page_size), expected
    )


def test_prefix_on_page_boundary_starts_new_page() -> None:
    page_size = 8
    prefix = torch.tensor([8, 0], dtype=torch.int32)
    seq = torch.tensor([8 + 8 + 3, 5], dtype=torch.int32)  # one full page + 3; 5 fresh
    last_loc = torch.tensor([7, -1], dtype=torch.int32)
    free_pages = torch.tensor([100, 101, 102], dtype=torch.int32)
    expected = _reference(prefix, seq, last_loc, free_pages, page_size)
    actual = _vectorized(prefix, seq, last_loc, free_pages, page_size)
    assert torch.equal(actual, expected)
    assert actual[:8].tolist() == list(range(800, 808))
    assert actual[8:11].tolist() == [808, 809, 810]
    assert actual[11:].tolist() == [816, 817, 818, 819, 820]


def test_multi_token_extension_page_32() -> None:
    """Requests extend by multiple tokens across arbitrary page phases."""
    rng = random.Random(7)
    page_size = 32
    prefix = torch.tensor(
        [rng.randrange(1, 5000) for _ in range(48)], dtype=torch.int32
    )
    seq = prefix + 16
    last_loc = _last_loc_for(prefix.to(torch.int64), page_size, rng)
    num_new = int(((seq + 31) // 32 - (prefix + 31) // 32).sum())
    free_pages = torch.tensor(
        sorted(rng.sample(range(1, 100_000), num_new)), dtype=torch.int32
    )
    expected = _reference(prefix, seq, last_loc, free_pages, page_size)
    assert torch.equal(
        _vectorized(prefix, seq, last_loc, free_pages, page_size), expected
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
