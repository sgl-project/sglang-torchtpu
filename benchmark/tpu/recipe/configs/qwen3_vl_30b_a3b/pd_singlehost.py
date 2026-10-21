"""Qwen3-VL-30B-A3B Raiden P/D: one TP8/EP8 host for each engine.

The serving settings are the ones the single-host MMMU-Pro gate launches through
recipe/qwen3_vl_30b_a3b/server.sh. Keep the two in sync: the P/D gate is meant
to differ from the single-host gate only in the disaggregated handoff -- the
sampler, the warmup skip, and the disaggregation_* settings below. Only model
locations, service ports, and readiness budgets come from the deployment
environment.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

# max_pixels caps one image at ~306 vision tokens (patch 16 + 2x2 merge = 1024
# px per token). The decode engine processes each request's images again to
# rebuild its prompt, so both engines need the same cap; otherwise the decode
# engine expects a different prompt length than the KV the prefill sends.
MM_PROCESS_CONFIG = {"image": {"min_pixels": 65536, "max_pixels": 313600}}

_COMMON_CONFIG = {
    "server_args": {
        **deepcopy(_FP8_BASE_CONFIG["server_args"]),
        "model_path": (
            os.environ.get("QWEN3_VL_30B_MODEL")
            or os.environ.get("MODEL")
            or "/weights/Qwen3-VL-30B-A3B-Instruct"
        ),
        "served_model_name": "qwen3-vl-30b-a3b",
        "host": "0.0.0.0",
        "trust_remote_code": True,
        "tp_size": 8,
        # Experts are split ep8 (16 per rank), as in the single-host recipe.
        "ep_size": 8,
        # bf16 checkpoint: "unquant" is the explicit opt-out of the base fp8,
        # and the checkpoint has no KV scales for an fp8 KV cache.
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        "mem_fraction_static": 0.30,
        "max_total_tokens": 262144,
        # The compile cap follows max_running_requests, so 8 keeps the bs grid
        # at [1, 2, 4, 8], matching the gate's concurrency.
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        "context_length": 9216,
        # Four token buckets bound the VL EXTEND compile grid; 8192 covers
        # max_prefill_tokens.
        "tpu_compile_tokens": [512, 1024, 2048, 8192],
        "mm_process_config": MM_PROCESS_CONFIG,
        # The ViT and decoder graph captures outlast the 300s default.
        "watchdog_timeout": 3600,
        # As in the single-host recipe: direct tracing compiles the ViT and
        # decoder warmups concurrently instead of one after another.
        "tpu_compile_tracer": "direct",
        # The sampler every TPU P/D gate pins (pd_reshard_harness.py notes the
        # distributed top-k crash behind it). Greedy decoding picks the argmax
        # token with either sampler.
        "sampling_backend": "pytorch",
        # A disaggregated engine's warmup request needs its peer and the router.
        "skip_server_warmup": True,
        "disaggregation_transfer_backend": "raiden",
    },
    "runtime_envs": {
        **_FP8_BASE_CONFIG["runtime_env"],
        # A cold start writes ~49 GB of executables to each of the tier-2 and
        # tier-3 compile caches. CI keeps both in the pod's 64Gi /dev/shm, and
        # a fresh pod has nothing to reuse.
        "ENABLE_COMPILE_CACHE": "0",
        # The base +recompiles makes dynamo reprint its guard list on every
        # recompile. Serving-time recompiles still fail the scheduler.
        "TORCH_LOGS": "-dynamo",
        "SGLANG_RAIDEN_KV_PORT": os.environ.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        "LB_PORT": os.environ.get("LB_PORT", "8000"),
        "HEALTH_TIMEOUT": os.environ.get("WAIT_SECS", "3600"),
        # Preserve QWEN3_VL_30B_MODEL precedence over the generic MODEL override.
        "MODEL": None,
    },
}

# Independent role configs: mutating one engine must not change the other.
PREFILL_CONFIG = deepcopy(_COMMON_CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
)

DECODE_CONFIG = deepcopy(_COMMON_CONFIG)
DECODE_CONFIG["server_args"].update(
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
    # CPU-tensor retraction needs no HiCache host pool. With 8 requests of at
    # most 9216 tokens, the 262144-token pool is not expected to retract.
    disaggregation_decode_retraction_backup="cpu_tensor",
)
