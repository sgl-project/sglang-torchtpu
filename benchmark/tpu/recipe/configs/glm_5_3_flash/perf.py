"""GLM-5.3-Flash throughput config: server.py sized for the 8k1k sweep.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/perf.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as SERVER_CONFIG

# The largest concurrency the sweep drives; the pool holds that many full contexts.
CONCURRENCY = 16

_args = SERVER_CONFIG["server_args"]
CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "zai-org/GLM-5.3-Flash",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "max_total_tokens": CONCURRENCY * _args["context_length"],
        # Two spare KDA slots, so a replacement request never waits on a finishing one.
        "max_mamba_cache_size": CONCURRENCY + 2,
        # CI compiles cold: one prefill bucket and the two swept decode batches.
        "tpu_compile_tokens": [_args["chunked_prefill_size"]],
        "tpu_compile_bs": [1, CONCURRENCY],
    }
)
