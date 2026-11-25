"""GLM-5.3-Flash text serving on one tpu7x host as a pipeline (pp8, tp1, ep1).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server_pp8.py

Each TensorCore is one stage: it owns a slice of the 45 layers with every
expert of its MoE layers, the sparse-MLA cache of its DSA layers and the state
of its KDA layers. The 1M context, 4096-token prefill chunks and the direct
tracer are the AgentX settings (agentx.py). No prefix cache yet.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as SERVER_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.server import (
    HICACHE_ARGS,
)

CONTEXT_LENGTH = 1048576
MAX_RUNNING = 32
# Stage 0 holds the three dense-MLP layers, the last stage the lm_head.
LAYER_PARTITION = "7,6,6,6,6,5,5,4"

CONFIG = deepcopy(SERVER_CONFIG)
_args = CONFIG["server_args"]
for _key in (*HICACHE_ARGS, "mamba_radix_cache_strategy"):
    _args.pop(_key, None)
_args.update(
    {
        "tp_size": 1,
        "pp_size": 8,
        "ep_size": 1,
        # Scattered prefill input is a TP feature.
        "enable_attn_tp_input_scattered": False,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "tpu_compile_tokens": [1024, 2048, 3072, 4096],
        "tpu_compile_tracer": "direct",
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        # A stage's batch holds at most max_running_requests // pp_size requests.
        "tpu_compile_bs": [1, 2, 4],
        "max_total_tokens": 4 * CONTEXT_LENGTH,
        "max_mamba_cache_size": MAX_RUNNING + 2,
        "dist_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_PP_LAYER_PARTITION": LAYER_PARTITION,
        "SGLANG_WARMUP_TIMEOUT": "3600",
        "SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM": "1",  # 1M context: page table in HBM
        "SGLANG_TPU_ENABLE_MOE_PREQUANT_DISPATCH": "1",  # fp8 rows into the MoE
        "SGLANG_TPU_ROUTER_TOPK_KERNEL_PREFILL": "1",
    }
)
