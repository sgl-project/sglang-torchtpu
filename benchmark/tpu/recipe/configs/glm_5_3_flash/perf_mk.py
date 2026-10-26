"""GLM-5.3-Flash decode-megakernel throughput config: server_mk.py for the 8k1k
sweep. The megakernel decodes one request at a time, so the sweep runs at c1.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/perf_mk.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server_mk import CONFIG as MK_CONFIG

CONFIG = deepcopy(MK_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "zai-org/GLM-5.3-Flash",
        # CI compiles cold: one prefill bucket.
        "tpu_compile_tokens": [MK_CONFIG["server_args"]["chunked_prefill_size"]],
    }
)
