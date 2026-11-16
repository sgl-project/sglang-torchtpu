# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sgl-jax (Apache-2.0), revision
# c948bdda95208e68a474c8088937347d08fe8937.
# Source: python/sgl_jax/srt/kernels/radix_topk/v2/tuned_configs.py
"""Valid-prefix length buckets for exact v7 TopK V2 (k=16 and k=2048).

Use a floor bucket: an exact boundary keeps its own configuration. Below
1K use the first bucket; above 1M use the last. Input and valid lengths are unchanged.
"""

from bisect import bisect_right
from dataclasses import dataclass

SUPPORTED_K = (16, 2048)
LENGTH_BUCKETS = tuple(1024 << i for i in range(11))


@dataclass(frozen=True)
class RadixTopKV2:
    block_size: int
    reconstruct_threshold: int


DEFAULT_CONFIG = RadixTopKV2(16384, 4096)
# Selected by benchmark/kernels/radix_topk/tune_radix_topk_v2.py.
# Batch-32 synchronized call times: 5 sweep seeds x 4 distributions, then
# 5 independent validation seeds for the top 5 candidates and the incumbent.
# Replace only for >1% improvement with every validation seed faster.
# Compilation and warmup are excluded; dispatch and synchronization are included.
TUNED_CONFIGS: dict[int, tuple[RadixTopKV2, ...]] = {
    16: (
        RadixTopKV2(1024, 1024),  # 1K
        RadixTopKV2(512, 128),  # 2K
        RadixTopKV2(2048, 256),  # 4K
        RadixTopKV2(4096, 128),  # 8K
        RadixTopKV2(4096, 256),  # 16K
        RadixTopKV2(8192, 1024),  # 32K
        RadixTopKV2(8192, 2048),  # 64K
        RadixTopKV2(8192, 512),  # 128K
        RadixTopKV2(16384, 128),  # 256K
        RadixTopKV2(16384, 256),  # 512K
        RadixTopKV2(32768, 8192),  # 1024K
    ),
    2048: (
        RadixTopKV2(512, 256),  # 1K
        RadixTopKV2(2048, 128),  # 2K
        RadixTopKV2(2048, 1024),  # 4K
        RadixTopKV2(8192, 512),  # 8K
        RadixTopKV2(8192, 4096),  # 16K
        RadixTopKV2(8192, 2048),  # 32K
        RadixTopKV2(16384, 8192),  # 64K
        RadixTopKV2(16384, 4096),  # 128K
        RadixTopKV2(16384, 8192),  # 256K
        RadixTopKV2(32768, 8192),  # 512K
        RadixTopKV2(32768, 4096),  # 1024K
    ),
}


def length_bucket_index(valid_length: int) -> int:
    return max(0, bisect_right(LENGTH_BUCKETS, valid_length) - 1)


def get_config(valid_length: int, k: int = 2048) -> RadixTopKV2:
    configs = TUNED_CONFIGS.get(k)
    if not configs:
        return DEFAULT_CONFIG
    return configs[length_bucket_index(valid_length)]


def get_score_width_config(score_width: int, k: int = 2048) -> RadixTopKV2:
    """Bind one config to the score width using the existing tuning table.

    Exact table boundaries keep their own configuration. Non-table widths
    use the existing floor lookup; per-row valid lengths remain unchanged.
    """
    if score_width < k:
        raise ValueError("score_width must be at least k")
    return get_config(score_width, k)
