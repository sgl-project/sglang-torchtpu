"""DeepSeek-V4.1-Flash on one tpu7x-8 host, compiled with the direct tracer.

Same serving setup as deepseek_v41_flash.py (attention DP8, EP8). The decoder
graphs are traced once by the direct tracer instead of Dynamo, compiled once on
an owner rank and broadcast to the others (--use-spmd-compile), and the warmup
compiles run concurrently (SGLANG_DIRECT_COMPILE_ASYNC, on by default).

Before relying on the shared compile, check that every rank traces the same
graph: VERIFY_SPMD_SAFE=1 compares them without executing.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_direct_spmd.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash import CONFIG as _V41

CONFIG = deepcopy(_V41)
CONFIG["server_args"].update(
    {
        "tpu_compile_tracer": "direct",
        "use_spmd_compile": True,
    }
)
