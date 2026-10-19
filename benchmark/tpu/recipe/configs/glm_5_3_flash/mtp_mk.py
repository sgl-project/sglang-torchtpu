"""GLM-5.3-Flash MTP with the experimental decode megakernel for the target
verify, one tpu7x host (tp8).

    GLM_FLASH_MK_PACKED=<packed dir> python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/mtp_mk.py

The draft and prefill stay native. GLM_FLASH_MK_PACKED points at the
megakernel's packed weights.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.server_mk import CONFIG as MK_CONFIG

CONFIG = deepcopy(MTP_CONFIG)
CONFIG["runtime_env"].update(
    {k: v for k, v in MK_CONFIG["runtime_env"].items() if "GLM_FLASH_MEGAKERNEL" in k}
)
