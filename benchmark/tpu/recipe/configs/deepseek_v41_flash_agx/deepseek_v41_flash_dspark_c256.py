"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 256 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c256.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 512 (64 per rank,
as many sequences per engine as vLLM-TPU runs on this host).

Why 64 per rank: 32 per rank on average; with session-affinity routing the
busiest rank peaked at 2.75x the average at c32 (canonical c32
j-3e6d0e27-2566-49c7-aca4). 64 is the largest compiled bucket
(torch_compile_max_bs 64), so it is the ceiling without new compile shapes; at
this concurrency the relative imbalance should be smaller than at c32.

Capacity, computed (the boot log and smoke run check it on hardware):
- KV pool: ~20.2M tokens per rank at mem_fraction_static 0.83 and 16 requests
  per rank (canonical c32 j-3e6d0e27-2566-49c7-aca4). The request-scaled SWA
  windows, draft ring and state come out of the same budget, ~960 MiB more per
  rank at 64 requests, so the pool shrinks by a few percent. 64 requests x
  ~140k average context is 9.0M (45%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 64 per rank; no new
  compile shapes.
- HBM: 14.7 GB was free after precompile at 16 per rank, with the bs 64
  programs compiled. Headroom at 64 per rank is not measured yet.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(512)
# Pin each rank's scheduler process to its own CPUs (SGLANG_SET_CPU_AFFINITY): the
# MLP-sync wait across DP ranks halves, per-op costs unchanged. Paired per-request
# interactivity: c32 +2.9% (j-fff55c92 vs j-b29f6ac4), c64 canonical +2.3%
# (j-10b94a15-88dd-40a2-af97 vs j-b598a23c-9c0b-4fef-87d2); neutral at c4
# (j-6643e364-2d36-454b-ae78 vs j-eecd8be6-a630-4b97-879e), so off at c4/c16.
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
