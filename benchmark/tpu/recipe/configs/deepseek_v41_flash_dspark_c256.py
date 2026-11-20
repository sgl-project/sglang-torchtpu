"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 256 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c256.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 512 (64 per rank,
as many sequences per engine as vLLM-TPU runs on this host).

Why 64 per rank: 32 per rank on average; with session-affinity routing the
busiest rank peaked at 2.75x the average at c32 (canonical c32
j-3e6d0e27-2566-49c7-aca4). 64 is the largest compiled bucket
(torch_compile_max_bs 64), so it is the ceiling without new compile shapes; at
this concurrency the relative imbalance should be smaller than at c32.

Unchanged, by computation:
- KV pool: ~20.2M tokens per rank. 64 requests x ~140k average context is 9.0M
  (45%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 64 per rank.
- HBM: the bs 64 verify and extend programs are already compiled and resident
  at the base cap (14.7 GB free after precompile).
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(512)
