# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/tuned_block_sizes.py
# Revision: 42c35ec6166a19b3f0db90f31807cc3df44602ec (epic/dsv41)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming top-k and candidate publication tables for the FP8 indexer.

Only the tables the packed FP8 path reads are carried; the source's BF16
scorer tables are not ported.
"""

from dataclasses import dataclass

# (generation, queries, token-score width, block size): (rows, output blocks).
PUBLISH_BLOCK_MAX_CONFIG = {(7, 96, 98816, 8): (96, 1024)}


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
