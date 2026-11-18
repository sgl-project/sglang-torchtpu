"""AgentX c32 on GLM-5.3-Flash: MLA latent cache split inside every page over the 8 ranks,
raw ring, no host tier, model-path decode, 9.4M-token pool, 800 KDA checkpoint slots,
16 running (measured 22,813 tok/s/chip).

Server, from the repo root (AGENTX_DUMMY=1: random weights, as measured):

    AGENTX_DUMMY=1 python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc32.py

Client, a 1 h window against it:

    AIPERF_HTTP_TCP_USER_TIMEOUT=900000 agentx profile --url http://127.0.0.1:30000 \\
        --endpoint /v1/chat/completions --concurrency 32 \\
        --model zai-org/GLM-5.3-Flash --tokenizer /weights/GLM-5.3-Flash \\
        --tokenizer-trust-remote-code --public-dataset semianalysis_cc_traces_weka_062126 \\
        --benchmark-duration 3600 --server-metrics http://127.0.0.1:30000/metrics \\
        --output-artifact-dir OUT/aiperf_artifacts

tok/s/chip = (prompt + completion tokens) / window / 4 chips; tokens per $ =
tok/s/chip * 3600 / 1.03 ($ per chip-hour).
"""

from benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx._base import (
    agentx_config,
)

# Without a host tier the device KDA checkpoint slots bound prefix reuse: 800, not 144.
CONFIG = agentx_config(
    max_running=16, pool_tokens=9437184, kv_shards=8, mamba_slots=800
)
