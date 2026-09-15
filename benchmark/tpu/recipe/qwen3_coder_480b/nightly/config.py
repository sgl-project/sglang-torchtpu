"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "tp_size": 16,
        "ep_size": 16,
        "nnodes": 2,
        "dp_size": 2,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
        "host": "0.0.0.0",
        "context_length": 1048576,
        "max_total_tokens": 877440,
        "max_running_requests": 64,
        "torch_compile_max_bs": 32,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 4096,
        "page_size": 128,
        "mem_fraction_static": 0.6,
        "disable_radix_cache": False,
        "enable_hierarchical_cache": True,
        "hicache_ratio": 1,
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        "enable_metrics": True,
        "enable_cache_report": True,
        "weight_loader_drop_cache_after_load": True,
        "json_model_override_args": {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 262144,
                "rope_theta": 10000000,
            }
        },
    }
)

BENCHMARK = {
    "profile": "full",
    "workload": "agentic",
    "sweep_bs": [1, 16, 32],
    "sweep_long": [0, 1, 2, 4],
    "turn_input_tokens": 4096,
    "turn_output_tokens": 1024,
    "target_context": 16384,
    "long_target_context": 262144,
    "long_ramp_tokens": 4096,
    "sessions_per_slot": "auto",
    "max_sessions_per_slot": 8,
    "max_leg_seconds": 7200,
    "health_timeout": 5400,
    "min_spec_accept_length": 1.0,
    "chunked_prefill_per_rank": 4096,
    "pd_prefill_port": 30000,
    "pd_decode_port": 30001,
    "pd_bootstrap_port": 8998,
    "pd_router_port": 8000,
}
