"""Kimi-K3 + DSpark with the decode megakernel for the target verify, four tpu7x
hosts (tp32), one request at a time.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/dspark_mk.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3.dspark import CONFIG as DSPARK_CONFIG
from benchmark.tpu.recipe.configs.kimi_k3.server_mk import CONFIG as MK_CONFIG
from benchmark.tpu.recipe.configs.kimi_k3.server_mk import PER_OP_ONLY_ENV

CONFIG = deepcopy(DSPARK_CONFIG)
CONFIG["server_args"].update(
    {
        key: MK_CONFIG["server_args"][key]
        for key in ("tpu_compile_bs", "torch_compile_max_bs", "mem_fraction_static")
    }
)
CONFIG["runtime_env"].update(
    {k: v for k, v in MK_CONFIG["runtime_env"].items() if "MEGAKERNEL" in k}
)
for key in PER_OP_ONLY_ENV:
    CONFIG["runtime_env"].pop(key)
