"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.bench_matrix.config import BENCHMARK as BASE_BENCHMARK
from benchmark.tpu.recipe.configs.bench_matrix.config import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/gcs-weights/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "context_length": 262144,
        "page_size": 128,
        "mem_fraction_static": 0.8,
        "tpu_fp8_runtime": "channelwise",
    }
)

BENCHMARK = deepcopy(BASE_BENCHMARK)
BENCHMARK.update(
    {
        "bench_model": "qwen3_coder_480b",
        "bench_recipe": "1host_pc",
        "long_target_context": 262144,
        "long_ramp_tokens": 36096,
    }
)
GROUP = {"hosts": 1, "prefill_engines": 0, "decode_engines": 0}
