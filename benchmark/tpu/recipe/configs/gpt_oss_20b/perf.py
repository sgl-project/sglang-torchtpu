"""GPT-OSS-20B (MXFP4 MoE) throughput config: tp8/ep8 on one tpu7x host, sized
for one 8k1k sweep point at concurrency 16.

accuracy.py in this directory owns HOW this model is served (tp8/ep8, the
checkpoint's own mxfp4, the dynamo tracer with the overlap scheduler off, the
SparseCore gather off); this file changes only what the measurement needs, so
the number is the accuracy-serving config's, not the 120b lane's direct-tracer
one.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/perf.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_20b.accuracy import CONFIG as ACCURACY_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 16
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 20
PREFILL_TOKENS = 8192

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(
    {
        # The canonical name requests carry and the dashboard reports; the
        # checkpoint itself is a mount path supplied by the test's MODEL.
        "served_model_name": "openai/gpt-oss-20b",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        # 16 x 9236 in flight plus one prefill chunk of slack.
        "max_total_tokens": CONCURRENCY * CONTEXT_LENGTH + PREFILL_TOKENS,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        # CI compiles cold every run: one prefill and one decode bucket.
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "tpu_compile_bs": [CONCURRENCY],
    }
)
# The eval-only knobs do not describe a throughput measurement.
for key in ("reasoning_parser", "allow_auto_truncate"):
    CONFIG["server_args"].pop(key, None)
