"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.bench_matrix.config import BENCHMARK as BASE_BENCHMARK
from benchmark.tpu.recipe.configs.bench_matrix.config import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/gcs-weights/weights/gpt-oss-120b",
        "context_length": 131072,
        "page_size": 64,
        "mem_fraction_static": 0.6,
        "quantization": None,
        "reasoning_parser": "gpt-oss",
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})

BENCHMARK = deepcopy(BASE_BENCHMARK)
BENCHMARK.update(
    {
        "bench_model": "gpt_oss_120b",
        "bench_recipe": "1host_pc",
        "long_target_context": 120832,
        "long_ramp_tokens": 16384,
    }
)
GROUP = {"hosts": 1, "prefill_engines": 0, "decode_engines": 0}
