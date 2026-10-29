"""Kimi-K3 (2.7T, MXFP4 experts) Raiden P/D: one 2x2x4 tpu7x slice per engine.

Each engine is a four-host TP32 / EP8 server; launch the matching config on
every host of its slice:

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/pd.py:PREFILL_CONFIG   # prefill slice
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/pd.py:DECODE_CONFIG    # decode slice

and front them with the P/D router (``sglang_router.launch_router
--pd-disaggregation --mini-lb``). At equal attention TP, Raiden pulls each
request's MLA latent pages and its KDA conv / recurrent state.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import (
    _FP8_BASE_CONFIG,
    pd_singlehost_role_configs,
)

MAX_RUNNING = 8
CONTEXT_LENGTH = 9344
PREFILL_TOKENS = 1024

_SERVER = deepcopy(_FP8_BASE_CONFIG)
_SERVER["server_args"].update(
    {
        "model_path": "/weights/Kimi-K3",
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 4,
        "tp_size": 32,
        "ep_size": 8,
        # MXFP4 experts (compressed-tensors): no online quantization.
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        "attention_backend": "pallas_tpu_mla",
        "page_size": 128,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [1, 2, 4, 8],
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "mem_fraction_static": 0.87,
        "trust_remote_code": True,
        "weight_loader_disable_mmap": True,
        "watchdog_timeout": 3600,
        "dist_timeout": 3600,
        # The sampler every TPU P/D gate pins; greedy picks the argmax either way.
        "sampling_backend": "pytorch",
        # A disaggregated engine's warmup request needs its peer and the router.
        "skip_server_warmup": True,
    }
)
_SERVER["runtime_env"]["SGLANG_TPU_CHECK_RECOMPILATION"] = "1"

# The role settings every Raiden P/D recipe shares (ports, bootstrap, backend);
# nnodes above makes each engine span its slice.
PREFILL_CONFIG, DECODE_CONFIG = pd_singlehost_role_configs(_SERVER)
