"""GLM-5.3-Flash with the decode megakernel, one tpu7x host (tp8).

    GLM_FLASH_MK_PACKED=<packed dir> python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server_mk.py

Single-request decode steps run as one Pallas call per device; prefill stays
native. GLM_FLASH_MK_PACKED is the directory written by
`python -m sglang_tpu.glm_flash_megakernel.pack`.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1",
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_PACKED": os.environ.get("GLM_FLASH_MK_PACKED", ""),
        # A multiple of the kernel's 512-token attention tile, >= the context.
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": str(
            -(-CONFIG["server_args"]["context_length"] // 512) * 512
        ),
    }
)
