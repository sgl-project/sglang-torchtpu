"""GLM-5.3-Flash MTP with the experimental decode megakernel for the target
verify, one tpu7x host (tp8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/mtp_mk.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.server_mk import CONFIG as MK_CONFIG

CONFIG = deepcopy(MTP_CONFIG)
CONFIG["server_args"].update({"max_running_requests": 1, "scheduler_recv_interval": 8})
CONFIG["runtime_env"].update(
    {k: v for k, v in MK_CONFIG["runtime_env"].items() if "GLM_FLASH_MEGAKERNEL" in k}
)
CONFIG["runtime_env"]["SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING"] = "1"
