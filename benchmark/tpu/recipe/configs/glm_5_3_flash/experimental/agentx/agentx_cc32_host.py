"""AgentX c32 on GLM-5.3-Flash with a host tier: agentx_cc32.py (MLA latent cache split
inside every page over the 8 ranks, raw ring, model-path decode, 9.4M-token pool, 800 KDA
checkpoint slots, 16 running) plus HiCache at 2.5x the device pools. Each rank's host keeps
its 1/8 latent share and the index keys for 23.6M tokens (36.2 GB) and 2,000 KDA states
(39.3 GB): 604 GB of host RAM on the 8 ranks beside ~424 GB of serving processes (990 GB
host; 108 GB left available, measured).

Server, from the repo root (AGENTX_DUMMY=1: random weights, as measured):

    AGENTX_DUMMY=1 python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc32_host.py

Client: as agentx_cc32.py (--concurrency 32).
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

# KV pages and KDA states both at this multiple of the device pool (see the docstring).
HOST_RATIO = 2.5

CONFIG = agentx_config(
    max_running=16,
    pool_tokens=9437184,
    mamba_slots=800,
    kv_shards=8,
    hicache_ratio=HOST_RATIO,
)
