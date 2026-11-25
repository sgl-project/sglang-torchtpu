# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU v7 tiles and pipeline depths for DeepSeek-V4.1 candidate selection."""

from sglang_tpu.kernels.index_selection.fp8.tuned import PagedMQAConfig

PACKED_FP8_PREPARED_MQA_CONFIG = PagedMQAConfig(2, 32768, 12, 6)
# Query pairs per gathered-scorer iteration; they share one MXU issue window.
PACKED_FP8_PREPARED_PAIRS = 2

# v7 H32/D128: fully static prefixes avoid dynamic per-fragment addresses.
# A 4K step bounds inactive computation while retaining a long scheduling window.
PACKED_FP8_PREPARED_PREFIX_TOKENS = 4096


# SparseCore gather, block size: (fragments per step, buffers). TC tiling needs
# whole 128-lane metadata rows; three 8-token buffers fill SparseCore VMEM.
# 128 divides tokens * 2 * 2048 blocks over 32 subcores: no 8.25 GiB output slice.
# On v7, 128 vs. 64 tied in prefill and at 128 decode groups, but was 4% slower at 32.
PAGED_MQA_GATHER_CONFIG = {8: (128, 3)}

# (generation, queries, token-score width, block size): (rows, output blocks).
PUBLISH_BLOCK_MAX_CONFIG = {(7, 96, 98816, 8): (96, 1024)}
