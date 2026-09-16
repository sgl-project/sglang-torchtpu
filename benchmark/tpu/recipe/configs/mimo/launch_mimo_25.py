"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "XiaomiMiMo/MiMo-V2.5",
        "tp_size": 8,
        "ep_size": 8,
        "dp_size": 2,
        "context_length": 8192,
        "max_running_requests": 8,
        "torch_compile_max_bs": 16,
        "mem_fraction_static": 0.5,
        "chunked_prefill_size": 2048,
        "max_prefill_tokens": 2048,
        "tpu_compile_tokens": [64, 128, 256, 512, 1024, 2048],
        "json_model_override_args": {"vision_config": None, "audio_config": None},
        "trust_remote_code": True,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
    }
)
