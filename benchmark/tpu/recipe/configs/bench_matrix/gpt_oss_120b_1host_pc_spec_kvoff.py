"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.bench_matrix.gpt_oss_120b_1host_pc import (
    BENCHMARK as BASE_BENCHMARK,
)
from benchmark.tpu.recipe.configs.bench_matrix.gpt_oss_120b_1host_pc import (
    CONFIG as BASE_CONFIG,
)

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_ratio": 1,
        "hicache_mem_layout": "page_first_fused",
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "/gcs-weights/weights/EAGLE3-gpt-oss-120b-bf16",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "watchdog_timeout": 7200,
        "sampling_backend": "pytorch",
        "tpu_compile_bs": [1, 2, 4, 16, 32],
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1"})

BENCHMARK = deepcopy(BASE_BENCHMARK)
BENCHMARK["bench_recipe"] = "1host_pc_spec_kvoff"
GROUP = {"hosts": 1, "prefill_engines": 0, "decode_engines": 0}
