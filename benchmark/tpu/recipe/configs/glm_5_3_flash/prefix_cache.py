"""GLM-5.3-Flash with the radix prefix cache (KDA extra_buffer snapshots), one
tpu7x host (tp8), DeepSeek Sparse Attention, no speculative decoding.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/prefix_cache.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONCURRENCY = 8
CONTEXT_LENGTH = 32768
MAX_RUNNING = CONCURRENCY + 2
# KDA state slots: the running requests' own and track slots, plus one per cached
# prefix. Too few does not fail; cache hits silently stop. Arbitrary sizing.
MAMBA_SLOTS = 5 * MAX_RUNNING + 64

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "zai-org/GLM-5.3-Flash",
        "reasoning_parser": "glm45",
        "disable_radix_cache": False,
        "mamba_radix_cache_strategy": "extra_buffer",
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [1, 2, 4, 8, 16],
        "max_total_tokens": 4 * MAX_RUNNING * CONTEXT_LENGTH,
        "max_mamba_cache_size": MAMBA_SLOTS,
        "enable_cache_report": True,
    }
)
