"""One-host-per-engine Raiden P/D with RAM KV offload on both engines.

Both engines keep a radix prefix cache backed by Raiden HiCache in host RAM.
The decode engine reuses its own cache, and for GPT-OSS's sliding-window
layers it keeps only full-attention KV in RAM while every request's sliding
window comes from prefill.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

GROUP = deepcopy(pd_singlehost.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _config["server_args"].update(
        # A small device pool, so the test can push prefixes out to RAM.
        max_total_tokens=8192,
        context_length=8192,
        max_running_requests=16,
        torch_compile_max_bs=16,
        enable_hierarchical_cache=True,
        hicache_ratio=2,
        hicache_io_backend="raiden",
        hicache_mem_layout="page_first_fused",
        hicache_write_policy="write_through",
        enable_cache_report=True,
        # The probe reads decode's load_back_tokens_total to see RAM restores.
        enable_metrics=True,
    )

DECODE_CONFIG["server_args"].update(
    disaggregation_decode_enable_radix_cache=True,
    # The full-only decode cache rejects host-pool retraction.
    disaggregation_decode_retraction_backup="cpu_tensor",
)
DECODE_CONFIG["runtime_envs"]["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
