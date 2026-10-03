"""DeepSeek-V3.1 in thinking mode: the R1-0528 accuracy serving config, V3 parser."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_r1_0528.accuracy import CONFIG as R1_CONFIG

CONFIG = deepcopy(R1_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-V3.1",
        "reasoning_parser": "deepseek-v3",
    }
)
