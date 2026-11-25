# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/tuned_block_sizes.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming top-k and candidate publication tables for the FP8 indexer.

Only the tables the packed FP8 path reads are carried; the source's BF16
scorer tables are not ported.
"""

from dataclasses import dataclass

# (generation, queries, token-score width, block size): (rows, output blocks).
PUBLISH_BLOCK_MAX_CONFIG = {(7, 96, 98816, 8): (96, 1024)}


def publish_block_max_config(
    generation: int, rows: int, width: int, block_size: int
) -> tuple[int, int] | None:
    """Return ``(row_tile, block_tile)`` for the Pallas block-max kernel, or None.

    Falls back to general TPU v7+ tiling so 1M-context shapes (e.g. width 524288
    or 1048576, any decode/verify/prefill batch size) do not fall out of the
    Pallas kernel into TensorCore reshape + reduce.
    """
    cfg = PUBLISH_BLOCK_MAX_CONFIG.get((generation, rows, width, block_size))
    if cfg is not None:
        return cfg
    if (
        generation < 7
        or rows <= 0
        or width < 1024
        or block_size <= 0
        or (block_size & (block_size - 1)) != 0
        or 128 % block_size != 0
    ):
        return None
    if rows % 8 == 0:
        row_tile = 8
        for candidate in (64, 48, 32, 24, 16, 8):
            if rows % candidate == 0:
                row_tile = candidate
                break
    else:
        row_tile = min(64, ((rows + 7) // 8) * 8)
    blocks = (width + block_size - 1) // block_size
    min_block_tile = 128 // block_size
    block_tile = min(1024, ((blocks + min_block_tile - 1) // min_block_tile) * min_block_tile)
    return (row_tile, block_tile)


@dataclass(frozen=True)
class StreamingTopKConfig:
    loop_unroll: int = 1
    bootstrap_multiplier: int = 4
    bitonic_ids: bool = False


# (generation, score width, topk): measured scan, bootstrap and output sorting.
STREAMING_TOPK_CONFIG = {
    (7, width, 512): StreamingTopKConfig(4, 8, True)
    for width in (16384, 32768, 49408, 98816)
}


def streaming_topk_config(
    generation: int, width: int, k: int
) -> StreamingTopKConfig:
    """Return tuned streaming top-k parameters, extending v7 defaults to 1M widths."""
    cfg = STREAMING_TOPK_CONFIG.get((generation, width, k))
    if cfg is not None:
        return cfg
    if generation >= 7 and k == 512 and width >= 16384:
        return StreamingTopKConfig(4, 8, True)
    return StreamingTopKConfig()
