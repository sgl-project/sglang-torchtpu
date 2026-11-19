"""AgentX c8 on GLM-5.3-Flash: decode megakernel, host tier, raw ring,
3.4M-token pool, 16 running, MTP 8 steps / 9 draft tokens at golden AL 3.99
(measured 13,769 tok/s/chip with 4 draft tokens at AL 3.5). The megakernel takes
steps of one running request (9 rows of its 16-row cap); larger steps run on the model.

Server, from the repo root (AGENTX_DUMMY=1: random weights, as measured):

    AGENTX_DUMMY=1 python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc8.py

Client, a 1 h window against it:

    AIPERF_HTTP_TCP_USER_TIMEOUT=900000 agentx profile --url http://127.0.0.1:30000 \\
        --endpoint /v1/chat/completions --concurrency 8 \\
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

CONFIG = agentx_config(
    num_steps=8, max_running=16, pool_tokens=3407872, mamba_slots=144
)
