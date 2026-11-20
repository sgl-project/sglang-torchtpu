"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 4 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c4.py

Same as deepseek_v41_flash_dspark_agentx.py: 128 running requests (16 per rank).
At c32 the busiest rank peaked at 11 running requests (4 per rank on average;
canonical c32 j-3e6d0e27-2566-49c7-aca4, session-affinity routing), so 16 per
rank never queues at c <= 32. c8 and c32 are the validated canonical settings.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark_agentx import (  # noqa: F401
    CONFIG,
)
