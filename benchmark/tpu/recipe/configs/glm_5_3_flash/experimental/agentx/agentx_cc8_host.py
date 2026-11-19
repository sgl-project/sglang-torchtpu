"""AgentX c8 on GLM-5.3-Flash with the split cache and a host tier: MLA latent cache split
inside every page over the 8 ranks, raw ring, 9.4M-token pool, 800 KDA checkpoint slots,
8 running, HiCache at 2.5x the device pools (as agentx_cc16_host.py), MTP 8 steps / 9
draft tokens at golden AL 3.99. The decode megakernel takes steps of one running request
(9 rows of its 16-row cap); larger steps run on the model.

Server, from the repo root (AGENTX_DUMMY=1: random weights):

    AGENTX_DUMMY=1 python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc8_host.py

Client: as agentx_cc8.py (--concurrency 8).
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

# KV pages and KDA states both at this multiple of the device pool (agentx_cc16_host.py).
HOST_RATIO = 2.5

CONFIG = agentx_config(
    num_steps=8,
    max_running=8,
    pool_tokens=9437184,
    mamba_slots=800,
    kv_shards=8,
    hicache_ratio=HOST_RATIO,
)
