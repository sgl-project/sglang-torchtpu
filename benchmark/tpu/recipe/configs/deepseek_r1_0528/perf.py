"""DeepSeek-R1-0528 throughput config: tp8 on ONE tpu7x host, fp8 KV.

The 8k1k sweep's serving config. V3.1 imports it (configs/deepseek_v3_1/perf.py):
same 671B blockwise-fp8 backbone, same shapes, different checkpoint.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_r1_0528/perf.py

WHY tp8 AND NOT THE tp16 ACCURACY CONFIG: a perf job is one single-host slice.
642 GB of fp8 weights is ~80 GB per rank, of the 101.7 GB a rank owns, so the KV
pool is what is left over -- a ~175k-token fp8 pool, measured 2026-09-04. That,
not the client, is what caps CONCURRENCY: 16 x CONTEXT_LENGTH is 148k of it, and
64 would need 594k.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 16
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 64

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-R1-0528",
        # The name requests carry and the dashboard reports; model_path is a
        # mount, and a mount would split the trend the day the weights move.
        "served_model_name": "deepseek-ai/DeepSeek-R1-0528",
        "tp_size": 8,
        "ep_size": 8,
        # Required: the base config's rpa_v3 is not an MLA backend.
        "attention_backend": "pallas_tpu_mla",
        # The only page size with tuned MLA blocks at 16 heads (128 / tp8).
        "page_size": 128,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        # A cold 8k prompt is one chunk.
        "chunked_prefill_size": INPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        # CI compiles cold every run: one prefill and one decode bucket.
        "tpu_compile_tokens": [INPUT_TOKENS],
        "tpu_compile_bs": [CONCURRENCY],
        "mem_fraction_static": 0.90,
        # 642 GB of checkpoint pages would otherwise stay in the host page cache
        # and put the CI pod on node-memory eviction.
        "weight_loader_drop_cache_after_load": True,
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_FP8_MOE_FP32_SCALES": "1"})
