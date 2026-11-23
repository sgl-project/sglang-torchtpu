"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 32 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c32.py

Same as deepseek_v41_flash_dspark_agentx.py: 128 running requests (16 per rank).
At c32 the busiest rank peaked at 11 running requests (4 per rank on average;
canonical c32 j-3e6d0e27-2566-49c7-aca4, session-affinity routing), so 16 per
rank did not queue there. c8 and c32 are the validated canonical settings.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
# Pin each rank's scheduler process to its own CPUs (SGLANG_SET_CPU_AFFINITY): the
# MLP-sync wait across DP ranks halves, per-op costs unchanged. Paired per-request
# interactivity: c32 +2.9% (j-fff55c92 vs j-b29f6ac4), c64 canonical +2.3%
# (j-10b94a15-88dd-40a2-af97 vs j-b598a23c-9c0b-4fef-87d2); neutral at c4
# (j-6643e364-2d36-454b-ae78 vs j-eecd8be6-a630-4b97-879e), so off at c4/c16.
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
