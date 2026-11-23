"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 4 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c4.py

deepseek_v41_flash_dspark_agentx.py (128 running requests, 16 per rank) with
three low-concurrency settings, measured together:

- prefill_decode_interval_min_bs 8: under attention DP, the 4-round decode
  interval after any rank's extend made every rank skip prefill (~4 x 51 ms per
  1024-token chunk at c4). Ranks with fewer than 8 running requests now admit
  prefill every round; the interval still applies at high load.
- tpu_compile_bs from 1 (inherited [8, 16, 32, 64]): a rank holds at most ~2
  requests at c4, so every verify/draft step padded to bs 8 (48 rows per rank).
  GSM8K-1319 with bs 1 compiled and used: 0.912 (parallel 8) / 0.906 (parallel
  128), no NaN (j-516f79b6-4fb7-4e75-9ba1).
- chunked prefill 2048 tokens per rank (16384 global, max_prefill_tokens to
  match; the 2048 token bucket is already compiled).

AgentX canonical c4 on the fused-EP tree, only these settings vs the default
recipe (j-20e8c9d3-e157-40e8-9a18 vs j-a9903e34-edd8-494f-b8dc): 5,107 -> 5,600
tok/s/chip, p90 interactivity 89.1 -> 104.4, TTFT p90 1,457 -> 1,171 ms, TPOT
p50 9.39 -> 7.84 ms; paired per-request interactivity 1.207 (95% CI
1.195-1.215), TTFT 0.882 (0.864-0.893).
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
CONFIG["server_args"].update(
    {
        "prefill_decode_interval_min_bs": 8,
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32, 64],
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
    }
)
