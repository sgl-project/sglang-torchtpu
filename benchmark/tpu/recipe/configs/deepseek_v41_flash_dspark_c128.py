"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 128 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c128.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 384 (48 per rank).

Why 48 per rank: session-affinity routing makes ranks uneven. At c32 the
busiest rank peaked at 2.75x the average (11 vs 4; canonical c32
j-3e6d0e27-2566-49c7-aca4). At c128 (16 per rank on average) that is ~44; the
base cap of 16 would queue two thirds of the sessions.

Capacity, computed (the boot log and smoke run check it on hardware):
- KV pool: ~20.2M tokens per rank at mem_fraction_static 0.83 and 16 requests
  per rank (canonical c32 j-3e6d0e27-2566-49c7-aca4). The request-scaled SWA
  windows, draft ring and state come out of the same budget, ~640 MiB more per
  rank at 48 requests, so the pool shrinks by a few percent. 48 requests x
  ~140k average context is 6.7M (33%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 48 per rank; no new
  compile shapes.
- HBM: 14.7 GB was free after precompile at 16 per rank, with the bs 64
  programs compiled. Headroom at 48 per rank is not measured yet.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(384)
