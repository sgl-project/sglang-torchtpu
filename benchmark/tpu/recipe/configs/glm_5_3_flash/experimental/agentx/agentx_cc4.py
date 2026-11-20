"""AgentX c4 (chart v26 point, run JSON runs/C4u_c4.json): the c1 config at 4 running
with 384 KDA slots: unsplit latent cache, host tier 2x, 3.4M-token pool, MTP 7 steps /
8 draft tokens at golden AL 3.93.

    benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 4
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

CONFIG = agentx_config(
    num_steps=7,
    max_running=4,
    pool_tokens=3407872,
    # 18.29 MB of HBM per slot and rank; 384 leaves 24.7 GB free, as the split runs had.
    mamba_slots=384,
    # Per-rank compile, as measured: the image it ran on (torch_tpu 0.2.0) has no SPMD compile.
    spmd_compile=False,
)
