# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/tuned_block_sizes.py
# Revision: 6fc30f15d40c2b5089b6605454d1045dce6b5b86 (PR #2430)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index-selection tiles. Lookups tune performance, never routing."""

from dataclasses import astuple, dataclass


@dataclass(frozen=True)
class PagedMQAConfig:
    """Paged MQA tiles and pipeline depths."""

    query_tile: int = 8
    key_tile: int = 16384
    issue_depth: int = 4
    key_buffers: int = 4

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in astuple(self)):
            raise ValueError(
                "Paged MQA tiles and issue_depth must be positive integers"
            )
        if self.key_buffers < 2:
            raise ValueError("Paged MQA needs at least two K buffers to overlap DMA")


# TPU v7, H32/D128, transposed packed FP8 pages; measured at P-state 7.
PACKED_FP8_MQA_CONFIG = PagedMQAConfig(6, 65536, 8, 4)

# Dense tiles follow configured decode width, capped at one 256-row H32 MXU tile.
PACKED_FP8_MAX_QUERY_TILE = 8

# Bound short-output padding to one tile; 4K is the smallest measured choice.
PACKED_FP8_DENSE_MIN_KEY_TILE = 4096


def paged_mqa_key_tile(width: int, page_size: int, cap: int) -> int:
    """Balance tiles at v7 MXU-column granularity instead of padding to the cap."""
    from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS

    alignment = max(page_size, 256)
    # Tiny storage pages use padded owner rows; keep the same K-buffer budget.
    page_rows = (PAGE_ROWS * page_size + PAGE_RECORDS - 1) // PAGE_RECORDS
    cap = cap * PAGE_ROWS * page_size // (PAGE_RECORDS * page_rows)
    cap = max(alignment, cap // alignment * alignment)
    tiles = max(1, (width + cap - 1) // cap)
    tile = (width + tiles * alignment - 1) // (tiles * alignment) * alignment
    return max(min(cap, PACKED_FP8_DENSE_MIN_KEY_TILE), tile)


# Query rows per score/top-k tile. One tile holds rows x score-width FP32
# scores; bound it so long contexts pipeline instead of growing the buffer.
# 96 = lcm(6 query rows, 32 SparseCore subcores).
PACKED_FP8_QUERY_ROWS = 96
PACKED_FP8_SCORE_ELEMENTS = 1 << 24

# Exact streaming selection after dense scoring; measured at 288 x 98304, P7.
PACKED_FP8_DENSE_TOPK_THRESHOLD = 2048

# (score columns, candidate block): (input DMA elements, vector-loop unroll).
PACKED_FP8_BLOCK_MAX_CONFIG = {(98816, 8): (8192, 4)}

# Rolling SparseCore gather: three buffers overlap reads and writes for 8-token blocks.
PAGED_MQA_GATHER_CONFIG = {(8, 1): (96, 3), (8, 2): (64, 3)}

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128
