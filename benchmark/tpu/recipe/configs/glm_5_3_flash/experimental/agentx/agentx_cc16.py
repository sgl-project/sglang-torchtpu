"""AgentX c16 (chart v26 point, run JSON runs/C16_S2ts.json): latent cache split by 2,
DSA token split, host KV 1.07x with 2,001 host KDA states, 800 KDA slots, 16 running,
MTP 3 steps / 4 draft tokens at golden AL 3.00.

    benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 16
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

CONFIG = agentx_config(
    num_steps=3,
    max_running=16,
    pool_tokens=9437184,  # HBM bounds the 2-shard pool at 8.6M
    mamba_slots=800,
    kv_shards=2,
    # Host pools ~600 GB on the 8 ranks: KV 9.2M tokens (35.5 GB/rank), KDA 39.3 GB/rank.
    hicache_ratio=1.07,
    state_host_ratio=2.5,
    token_split=True,
)
