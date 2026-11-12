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
        "served_model_name": "deepseek-v4.1-flash",
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
        "max_total_tokens": 16384,
        "max_running_requests": 8,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        "mem_fraction_static": 0.85,
        "enable_torch_compile": True,
        "tpu_compile_tokens": [128, 256, 512, 1024],
        # Decode pads to bs 8, as vllm-torchtpu's MIN_NUM_SEQS = 8 does: the
        # bs=1 decode graph returns NaN logits after the first token. The cause
        # is not located yet; the CSA2 kernel alone matches the dense reference
        # at batch 1 (fixed-tile and padded layouts), so it is not the kernel.
        "tpu_compile_bs": [8],
        "torch_compile_max_bs": 8,
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
    }
)
