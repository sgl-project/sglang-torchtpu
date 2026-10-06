"""Shared GPT-OSS-120B accuracy serving; each benchmark derives its own sizing.

The eval harness drives one prompt per request, so prefix caching is off and the
pool is sized from the benchmark's own budget rather than the sweep's 2x rule.
Input length is 4096 for every eval; only the output budget and concurrency
differ, which is what the per-benchmark modules beside this one set.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gpt-oss-120b",
        "served_model_name": "gpt-oss-120b",
        "reasoning_parser": "gpt-oss",
        # The published scores were measured on the checkpoint's own KV dtype,
        # not the fp8 the serving base selects; fp8 KV changes what they mean.
        "kv_cache_dtype": None,
        # "unquant" is the explicit opt-out the measured runs used: it also sets
        # _quantization_explicitly_unset, which an omitted value does not.
        "quantization": "unquant",
        "page_size": 128,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.60,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "tpu_compile_tokens": [4096],
    }
)
