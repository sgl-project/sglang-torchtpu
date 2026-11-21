"""DeepSeek-V4.1-Flash text serving on one tpu7x-8 host: attention DP8, EP8.

The decoder graphs are traced once by the direct tracer, compiled once on an
owner rank and broadcast to the others (--use-spmd-compile), and the warmup
compiles run concurrently (SGLANG_DIRECT_COMPILE_ASYNC, on by default).
Before relying on the shared compile, check that every rank traces the same
graph: VERIFY_SPMD_SAFE=1 compares them without executing.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash.py
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import CONFIG as _BASE

MODEL_PATH = os.environ.get("MODEL", "/weights/DeepSeek-V4.1-Flash")

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "model_path": MODEL_PATH,
        "served_model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "trust_remote_code": True,
        # EP-local expert and TP-local Engram reads (deepseek_v41/loader.py).
        "load_format": "runai_streamer",
        "tp_size": 8,
        "attn_dp_size": 8,
        "ep_size": 8,
        "enable_dp_lm_head": True,
        "attention_backend": "dsv4",
        "page_size": 256,
        "kv_cache_dtype": "bfloat16",
        # Served AgentX traces at 1M (prompts up to 507k tokens) with a 21.1M-token
        # KV pool per worker and ~13 GB HBM left after precompile.
        "context_length": 1048576,
        # Global values, divided by attn_dp_size (8): 64 requests and 1024
        # prefill tokens per worker, as in the 1M AgentX runs.
        "max_running_requests": 512,
        "chunked_prefill_size": 8192,
        # Per scheduler step, not divided by attn_dp_size.
        "max_prefill_tokens": 8192,
        "mem_fraction_static": 0.85,
        "enable_torch_compile": True,
        "tpu_compile_tracer": "direct",
        "use_spmd_compile": True,
        "tpu_compile_tokens": [512, 1024, 2048],
        # bs 1 decode returns NaN logits after the first token.
        "tpu_compile_bs": [8, 16, 32, 64],
        "torch_compile_max_bs": 64,
        "disable_radix_cache": True,
        "random_seed": 1234,
        "watchdog_timeout": 3600,
    }
)
CONFIG["extra_args"] = ["--no-enable-multimodal"]
CONFIG["runtime_env"].update(
    {
        "RUNAI_STREAMER_CONCURRENCY": "32",
        "RUNAI_STREAMER_MEMORY_LIMIT": "34359738368",
        "SGLANG_DSV4_FP4_EXPERTS": "1",
        "SGLANG_DSV4_FP4_DEQUANT": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TPU_ENABLE_TELEMETRY": "0",
        "TPU_ENABLE_TELEMETRY_HARVESTING": "0",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
        "OMP_NUM_THREADS": "4",
        # The checker syncs before a compiled call whenever its tier-1 miss counter
        # moved, waiting out the in-flight program: 0.6-10.7 ms per DSpark decode
        # step in probes. Set it to "1" to find serving-time recompiles.
        "SGLANG_TPU_CHECK_RECOMPILATION": "0",
        # One Engram table in host memory, shared by the 8 ranks.
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
        # The SparseCore gather offload halts a SparseCore in the 512-token
        # extend graph (libtpu 0.0.50).
        "LIBTPU_INIT_ARGS": (
            "--xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_dvfs_p_state=7"
        ),
    }
)
