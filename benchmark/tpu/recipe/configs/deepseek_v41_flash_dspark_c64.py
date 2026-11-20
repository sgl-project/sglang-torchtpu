"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 64 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c64.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 256 (32 per rank).

Why 32 per rank: session-affinity routing pins each conversation and its
subagents to one rank, so ranks are uneven. At c32 the busiest rank peaked at
11 running requests vs 4 on average (2.75x; canonical c32
j-3e6d0e27-2566-49c7-aca4). At c64 (8 per rank on average) that is ~22, above
the base cap of 16, so the base would queue sessions on the busy ranks.

Unchanged, by computation:
- KV pool: ~20.2M tokens per rank at mem_fraction_static 0.83 (same run). 32
  requests x ~140k average context is 4.5M (22%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] already covers 32 per rank.
- HBM: the bs 32/64 verify and extend programs are already compiled and
  resident at the base cap (14.7 GB free after precompile). The request-scaled
  SWA window and draft ring come out of the KV pool budget.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(256)
