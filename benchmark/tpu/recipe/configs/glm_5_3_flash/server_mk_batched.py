"""GLM-5.3-Flash without speculation, the decode megakernel taking multi-request
decode steps, one tpu7x host (tp8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server_mk_batched.py

GLM_FLASH_MK_CONCURRENCY (default 4) sets the running requests.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONTEXT_LENGTH
from benchmark.tpu.recipe.configs.glm_5_3_flash.server_mk import CONFIG as MK_CONFIG

CONCURRENCY = int(os.environ.get("GLM_FLASH_MK_CONCURRENCY", "4"))
MAX_RUNNING = CONCURRENCY
COMPILE_BS = [b for b in (1, 2, 4, 8, 16) if b <= CONCURRENCY] or [1]

CONFIG = deepcopy(MK_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": COMPILE_BS,
        "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH,
        "max_mamba_cache_size": 5 * MAX_RUNNING + 64,
    }
)
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS"] = str(MAX_RUNNING)
