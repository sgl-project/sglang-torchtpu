"""GLM-5.3-Flash with the decode megakernel, one tpu7x host (tp8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server_mk.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"]["max_running_requests"] = 1
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1",
        # Must be a multiple of the kernel's 512-token attention tile.
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": str(
            -(-CONFIG["server_args"]["context_length"] // 512) * 512
        ),
    }
)
