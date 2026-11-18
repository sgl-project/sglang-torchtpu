"""GLM-5.3-Flash text serving on one tpu7x host (tp8), compiled with the direct tracer.

Same serving setup as server.py, except that prefill keeps the residual streams
whole (no attention-TP input scattering). The decoder graphs are traced by the direct
tracer instead of Dynamo, compiled once on an owner rank and broadcast to the
others (--use-spmd-compile), and the warmup compiles run concurrently
(SGLANG_DIRECT_COMPILE_ASYNC, on by default).

Before relying on the shared compile, check that every rank traces the same
graph: VERIFY_SPMD_SAFE=1 compares them without executing.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server_direct_spmd.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "tpu_compile_tracer": "direct",
        "use_spmd_compile": True,
        # The scattered prefill picks each rank's rows at a Python-int offset
        # (layer_boundary _tp_reduce_scatter_update_read_gather): rank-dependent graphs.
        "enable_attn_tp_input_scattered": False,
    }
)
