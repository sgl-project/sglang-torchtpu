"""GPT-OSS-120B throughput sweep: the 8k1k workload at concurrency 64.

server.py owns how this model is served and the long_8k1k_c64 profile sizes it for
the workload. Concurrency sizes the server, not just the client, so this is its own
server rather than a second sweep point on perf.py's.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/perf_c64.py
"""

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG
from benchmark.tpu.recipe.configs.profiles import PROFILES, apply_profile

CONFIG = apply_profile(SERVER_CONFIG, PROFILES["long_8k1k_c64"])
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gpt-oss-120b",
        "served_model_name": "openai/gpt-oss-120b",
        # Random-prompt throughput at InferenceX parity: a cache hit would let a
        # request skip a real prefill and drift the trend upward.
        "disable_radix_cache": True,
    }
)
