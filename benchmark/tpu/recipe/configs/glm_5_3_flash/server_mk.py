"""GLM-5.3-Flash with the decode megakernel, one tpu7x host (tp8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server_mk.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"]["max_running_requests"] = 1
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_MEGAKERNEL"] = "1"
