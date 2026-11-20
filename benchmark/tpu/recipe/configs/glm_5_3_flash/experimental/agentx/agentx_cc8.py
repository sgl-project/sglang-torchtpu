"""AgentX c8 (chart v26 point, run JSON runs/K2ts_c8.json): latent cache split by 2,
DSA token split, host KV 1.5x, 800 KDA slots, 8 running, MTP 3 steps / 4 draft tokens
at golden AL 3.00.

    benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 8
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

CONFIG = agentx_config(
    num_steps=3,
    max_running=8,
    pool_tokens=9437184,  # HBM bounds the 2-shard pool at 8.8M
    mamba_slots=800,
    kv_shards=2,
    # Host KV 13.2M tokens and 1,201 KDA states: 2 shards keep 4 host copies of the latent.
    hicache_ratio=1.5,
    token_split=True,
)
