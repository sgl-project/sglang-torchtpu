# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU v7 tiles and pipeline depths for the packed FP8 indexer."""

import math
from dataclasses import astuple, dataclass

from jax.experimental.pallas import tpu as pltpu


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


# TPU v7, H32/D128, transposed packed FP8 pages.
PACKED_FP8_MQA_CONFIG = PagedMQAConfig(6, 65536, 10, 4)
# Bound short-output padding to one tile; 4K is the smallest measured choice.
PACKED_FP8_DENSE_MIN_KEY_TILE = 4096


def paged_mqa_key_tile(width: int, page_size: int, cap: int) -> int:
    """Balance tiles at the device MXU-column granularity."""
    alignment = math.lcm(page_size, pltpu.get_tpu_info().mxu_column_size)
    cap = max(alignment, cap // alignment * alignment)
    tiles = max(1, (width + cap - 1) // cap)
    tile = (width + tiles * alignment - 1) // (tiles * alignment) * alignment
    return max(min(cap, PACKED_FP8_DENSE_MIN_KEY_TILE), tile)


def paged_mqa_query_tile(query_size: int, heads: int) -> int:
    """Bound queries by the MXU footprint and head-reduction sublanes."""
    info = pltpu.get_tpu_info()
    return min(query_size, info.num_sublanes, max(1, info.mxu_column_size // heads))
