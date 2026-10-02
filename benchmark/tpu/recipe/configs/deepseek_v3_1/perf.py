"""DeepSeek-V3.1 throughput config: the R1-0528 sweep, V3.1 checkpoint."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_r1_0528.perf import CONFIG as R1_CONFIG

CONFIG = deepcopy(R1_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-V3.1",
        "served_model_name": "deepseek-ai/DeepSeek-V3.1",
    }
)
