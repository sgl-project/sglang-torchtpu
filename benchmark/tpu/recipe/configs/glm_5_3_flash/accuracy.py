"""GLM-5.3-Flash on one tpu7x host (tp8), sized for the thinking-mode AIME26
nightly. DeepSeek Sparse Attention, no speculative decoding.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/accuracy.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.server import HICACHE_ARGS

CONCURRENCY = 32
# 290k thinking budget + prompt; at 128k, 3.3% of AIME26 samples truncated.
CONTEXT_LENGTH = 300000
MAX_RUNNING = CONCURRENCY + 2

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "zai-org/GLM-5.3-Flash",
        "reasoning_parser": "glm45",
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32],
        "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH,
        "max_mamba_cache_size": MAX_RUNNING + 2,
        # Radix snapshots take 5 state slots per request; these slots fit 1.
        "disable_radix_cache": True,
    }
)
# HiCache needs the radix cache.
for key in HICACHE_ARGS:
    del CONFIG["server_args"][key]
