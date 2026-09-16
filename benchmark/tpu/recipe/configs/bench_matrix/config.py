"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "tp_size": 8,
        "ep_size": 8,
        "host": "0.0.0.0",
        "attention_backend": "pallas_tpu_rpa_v3",
        "max_total_tokens": 1572864,
        "max_running_requests": 32,
        "torch_compile_max_bs": 32,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        "disable_radix_cache": False,
        "enable_metrics": True,
        "enable_cache_report": True,
    }
)

BENCHMARK = {
    "profile": "ci",
    "workload": "agentic",
    "sweep_bs": [1, 16, 32],
    "sweep_long": [1, 2, 4],
    "turn_input_tokens": 4096,
    "turn_output_tokens": 1024,
    "target_context": 8192,
    "sessions_per_slot": "auto",
    "max_sessions_per_slot": 8,
    "max_leg_seconds": 900,
    "health_timeout": 3600,
    "health_poll_interval": 10,
    "curl_max_time": 10,
    "leg_kill_grace": 60,
    "pd_prefill_port": 30001,
    "pd_decode_port": 30002,
    "pd_bootstrap_port": 8998,
    "raiden_kv_base_port": 31000,
    "hicache_ratio": 1,
}
