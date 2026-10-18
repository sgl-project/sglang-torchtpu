"""GPT-OSS-120B throughput sweep: the 8k1k workload at concurrency 16.

server.py owns how this model is served and the long_8k1k profile sizes it for
the workload. The one measurement departure is prefix caching, off below.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/perf.py
"""

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG
from benchmark.tpu.recipe.configs.profiles import PROFILES, apply_profile

CONFIG = apply_profile(SERVER_CONFIG, PROFILES["long_8k1k"])
CONFIG["server_args"].update(
    {
        # CI serves the Filestore mount; the dashboard keys the series on the
        # HF id, which a remount cannot split in two.
        "model_path": "/weights/gpt-oss-120b",
        "served_model_name": "openai/gpt-oss-120b",
        # Random-prompt throughput at InferenceX parity: a cache hit would let a
        # request skip a real prefill and drift the trend upward.
        "disable_radix_cache": True,
    }
)
