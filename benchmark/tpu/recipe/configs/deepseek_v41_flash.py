"""DeepSeek-V4.1-Flash text serving on one tpu7x-8 host: attention DP8, EP8.

The default launch for the V4.1 bring-up (PR1): text only, block-FP8 dense
weights, native packed-FP4 experts, EP-local RunAI reads of the checkpoint.

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
        # The name requests carry and the dashboard keys on; model_path is a mount.
        "served_model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "trust_remote_code": True,
        # EP-local expert reads (sglang_tpu/models/deepseek_v41/loader.py).
        "load_format": "runai_streamer",
        "tp_size": 8,
        "attn_dp_size": 8,
        "ep_size": 8,
        "enable_dp_lm_head": True,
        "attention_backend": "dsv4",
        "page_size": 256,
        # The pool keeps CSA2 FP8 main KV internally.
        "kv_cache_dtype": "bfloat16",
        "context_length": 9472,
        # Under attention DP the pool, the compile grid and the chunk size are
        # per DP worker; max_running_requests and chunked_prefill_size are
        # global and divided by attn_dp_size (8).
        "max_total_tokens": 262144,
        # 64 running requests per worker: the largest decode bucket, which is
        # also the batch dimension every EXTEND graph is compiled at under DP.
        "max_running_requests": 512,
        # 4096 prefill tokens per worker per step, the largest token bucket.
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 4096,
        "mem_fraction_static": 0.85,
        "enable_torch_compile": True,
        # 512/1024 keep short prompts off the 2048 graph.
        # TODO(JamesBrianD): add 8192 once its EXTEND graph fits; at bs 128 it
        # needs 95.44 GB of HLO temporaries against 94.74 GB HBM at compile time.
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
        # The smallest decode bucket stays above 1: the bs=1 decode graph
        # returns NaN logits after the first token (cause not located; the
        # CSA2 kernel alone matches the dense reference at batch 1).
        "tpu_compile_bs": [8, 16, 32, 64],
        "torch_compile_max_bs": 64,
        "disable_radix_cache": True,
        "random_seed": 1234,
        "watchdog_timeout": 3600,
    }
)
# False omits a flag (config_eval.server_args_to_cli), so the negative switch
# goes through extra_args: PR1 serves the checkpoint text only.
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
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
        # One Engram table copy in host memory shared by the 8 ranks, looked up
        # outside the compiled graph (sglang_tpu/models/deepseek_v41/engram_host.py).
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
        # XLA's SparseCore gather offload halted a SparseCore in the 512-token
        # extend graph (libtpu 0.0.50); drop once that is fixed. P-state 7 pins
        # the clock, as the vllm-torchtpu DSV4.1 launchers do.
        "LIBTPU_INIT_ARGS": (
            "--xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_dvfs_p_state=7"
        ),
    }
)
