"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 128 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c128.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 384 (48 per rank).

Why 48 per rank: session-affinity routing makes ranks uneven. At c32 the
busiest rank peaked at 2.75x the average (11 vs 4; canonical c32
j-3e6d0e27-2566-49c7-aca4). At c128 (16 per rank on average) that is ~44; the
base cap of 16 would queue two thirds of the sessions.

Unchanged, by computation:
- KV pool: ~20.2M tokens per rank. 48 requests x ~140k average context is 6.7M
  (33%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 48 per rank (bucket 64).
- HBM: the bs 64 verify and extend programs are already compiled and resident
  at the base cap (14.7 GB free after precompile).
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(384)
