"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 16 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c16.py

deepseek_v41_flash_dspark_agentx.py (128 running requests, 16 per rank) with
compile batch sizes from 1 and 2048-token prefill chunks per rank.

Why: at c16 a rank holds ~2 requests on average, so the inherited [8, 16, 32,
64] buckets padded most verify/draft steps to bs 8. AgentX canonical c16 on the
fused-EP tree, only tpu_compile_bs differs (j-04766568-bda9-4168-a3b9 vs
j-1805fb3a-fbc1-4bb7-8108): 17,873 -> 18,529 tok/s/chip, p90 interactivity
62.8 -> 67.0, TTFT p90 2,836 -> 2,538 ms, TPOT p50 12.11 -> 10.93 ms; paired
per-request interactivity 1.106 (95% CI 1.098-1.113), TTFT 0.947
(0.938-0.955). GSM8K with this bucket set: 0.912 / 0.906
(j-516f79b6-4fb7-4e75-9ba1). KV capacity unchanged; ~2 min more compile at boot.

Chunked prefill 2048 tokens per rank (16384 global, max_prefill_tokens to match;
the 2048 token bucket is already compiled), measured together with the buckets
(j-2995b11f-1e80-4f19-9eda): 19,008 tok/s/chip, p90 interactivity 66.1, TTFT
p90 1,749 ms; paired vs the buckets alone TTFT 0.783 (0.771-0.791),
interactivity 1.009 (1.005-1.018); vs the default recipe TTFT 0.737,
interactivity 1.131.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
CONFIG["server_args"].update(
    {
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32, 64],
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
    }
)
