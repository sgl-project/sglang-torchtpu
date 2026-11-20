"""AgentX c16 alternative (not on the chart; run JSON runs/C16_S2ts_kda1400.json): the
c16 config with HBM moved from the KV pool (5.7M tokens) to 1,400 KDA slots; within the
noise of agentx_cc16.py on throughput and interactivity, prefix-hit gap to ideal 0.36 pt
instead of 0.50.

    benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 16-kda1400
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

CONFIG = agentx_config(
    num_steps=3,
    max_running=16,
    pool_tokens=9437184,  # HBM bounds the pool at 5.7M with 1,400 slots
    mamba_slots=1400,
    kv_shards=2,
    # Host KV 9.6M tokens, 2,001 KDA states: the c16 host pools at the larger slot count.
    hicache_ratio=1.7,
    state_host_ratio=1.429,
    token_split=True,
)
