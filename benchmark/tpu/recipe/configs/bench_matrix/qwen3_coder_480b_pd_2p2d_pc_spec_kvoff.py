"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.bench_matrix.qwen3_coder_480b_1host_pc_spec_kvoff import (
    BENCHMARK as BASE_BENCHMARK,
)
from benchmark.tpu.recipe.configs.bench_matrix.qwen3_coder_480b_1host_pc_spec_kvoff import (
    CONFIG as BASE_CONFIG,
)

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update({})
CONFIG["runtime_env"].update(
    {"SGLANG_RAIDEN_KV_PORT": "31000", "SGLANG_RAIDEN_RESHARD": "0"}
)

BENCHMARK = deepcopy(BASE_BENCHMARK)
BENCHMARK["bench_recipe"] = "pd_2p2d_pc_spec_kvoff"
GROUP = {"hosts": 4, "prefill_engines": 2, "decode_engines": 2}
PREFILL_CONFIG = deepcopy(CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=30001,
    disaggregation_mode="prefill",
    disaggregation_transfer_backend="raiden",
    disaggregation_bootstrap_port=8998,
)
DECODE_CONFIG = deepcopy(CONFIG)
DECODE_CONFIG["server_args"].update(
    port=30002,
    disaggregation_mode="decode",
    disaggregation_transfer_backend="raiden",
    enable_hierarchical_cache=False,
)

for name in ("hicache_io_backend", "hicache_mem_layout", "hicache_ratio"):
    DECODE_CONFIG["server_args"].pop(name, None)
