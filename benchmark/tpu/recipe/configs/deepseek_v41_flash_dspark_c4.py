"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 4 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c4.py

deepseek_v41_flash_dspark_agentx.py (128 running requests, 16 per rank) with
prefill_decode_interval_min_bs 8 and compile batch sizes from 2.

Why: under attention DP, prefill_decode_interval 4 with min_bs 0 makes every rank
skip prefill for 4 decode rounds after any extend on any rank; at c4 that adds
~4 x 51 ms per 1024-token chunk. With min_bs 8, ranks with fewer than 8 running
requests admit prefill every round (the interval stays in effect at high load).
AgentX canonical c4, same payload, only this setting differs
(j-912e782e-50ae-434d-b46a vs j-66eb2c72-91fb-418f-8c08): 3,761 -> 3,892
tok/s/chip, TTFT p50 950 -> 781 ms, p90 4,654 -> 2,290 ms, TPOT p50 14.99 ->
15.02 ms, p90 interactivity within run noise.

tpu_compile_bs from 2 (inherited [8, 16, 32, 64]): at c4 a rank holds at most
~2 requests, so every verify/draft step padded to bs 8 (48 rows per rank); bs 2
runs 12. Canonical c4 on the fused-EP tree, only this setting differs
(j-a9903e34-edd8-494f-b8dc vs j-c5f58a55-08c7-4e1f-85f4): 5,107 -> 5,389
tok/s/chip, p90 interactivity 89.1 -> 101.2, p90 E2E 38.5 -> 33.3 s. GSM8K-1319
(real acceptance, parallel 8 so bs 2 runs / parallel 128): 0.912 / 0.907
(j-db72179e-f648-4514-aa10).
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
CONFIG["server_args"]["prefill_decode_interval_min_bs"] = 8
CONFIG["server_args"]["tpu_compile_bs"] = [2, 4, 8, 16, 32, 64]
