"""AgentX c1 (chart v26 point, run JSON runs/R1b_c1.json): unsplit latent cache with SPMD
compile, host tier 2x, 3.4M-token pool, 74 KDA slots, 2 running, MTP 7 steps / 8 draft
tokens at golden AL 3.93.

    benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 1
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

CONFIG = agentx_config(num_steps=7, max_running=2, pool_tokens=3407872, mamba_slots=74)
