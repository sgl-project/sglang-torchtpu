"""Gemma-4-31B-it throughput config: the serving recipe, sized for one sweep.

server.py in this directory owns HOW this model is served (tp8, bf16 KV, the
bounded hybrid pool, the watchdog); this file changes only what the measurement
needs, so a tuning change there moves the perf trend with it.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gemma4_31b/perf.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gemma4_31b.server import CONFIG as SERVER_CONFIG

# The largest concurrency the sweep drives. The pool is what caps it: the full
# (global-layer) pool below holds CONCURRENCY sequences at CONTEXT_LENGTH plus
# slack, and compile materializes that pool as graph args + outputs.
CONCURRENCY = 16
CONTEXT_LENGTH = SERVER_CONFIG["server_args"]["context_length"]

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        # The canonical name requests carry and the dashboard reports; the
        # checkpoint itself is a mount path supplied by the test's MODEL.
        "served_model_name": "google/gemma-4-31B-it",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        # 16 x 9216 = 147,456 in flight. server.py's 131,072 would leave 14 of
        # the 16 running and quietly measure a pool-bound shape as a c16 number.
        "max_total_tokens": 163840,
        # CI compiles cold every run: one prefill and one decode bucket instead
        # of the default ladders.
        "tpu_compile_tokens": [SERVER_CONFIG["server_args"]["chunked_prefill_size"]],
        "tpu_compile_bs": [CONCURRENCY],
    }
)
