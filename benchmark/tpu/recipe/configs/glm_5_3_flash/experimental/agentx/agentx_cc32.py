"""AgentX c32 (chart v26 point, run JSON runs/R5_c32_ts.json): latent cache split by 8,
DSA token split, host tier 2.5x, 9.4M-token pool, 800 KDA slots, 16 running (the other
16 sessions queue), MTP 3 steps / 4 draft tokens at golden AL 3.00.

    benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 32
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

CONFIG = agentx_config(
    num_steps=3,
    max_running=16,
    pool_tokens=9437184,
    mamba_slots=800,
    kv_shards=8,
    # Host pools 604 GB on the 8 ranks; the pod peaked at 983 of ~987 GB: do not raise.
    hicache_ratio=2.5,
    token_split=True,
)
