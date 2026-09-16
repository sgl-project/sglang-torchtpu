"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "XiaomiMiMo/MiMo-V2.5-Pro",
        "tp_size": 16,
        "ep_size": 16,
        "dp_size": 2,
        "context_length": 8192,
        "max_running_requests": 8,
        "torch_compile_max_bs": 16,
        "mem_fraction_static": 0.7,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "tpu_compile_tokens": [64, 128, 256, 512, 1024, 2048, 4096],
        "json_model_override_args": {"vision_config": None, "audio_config": None},
        "trust_remote_code": True,
        "nnodes": 2,
        "host": "0.0.0.0",
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
    }
)
