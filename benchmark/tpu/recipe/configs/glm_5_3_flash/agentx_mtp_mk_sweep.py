"""AgentX sweep point: MTP verify on the decode megakernel at AGENTX_CONC concurrent
sessions, radix cache on, no KV offload, sized like the GB300 reference run.

    AGENTX_CONC=8 python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/agentx_mtp_mk_sweep.py

Every verify step runs on the kernel: its row cap follows the running requests, and
steps beyond one call's rows run as several calls.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import NUM_STEPS

CONCURRENCY = int(os.environ["AGENTX_CONC"])
# 1M context overflows SMEM: the DSA indexer prefetches a 1M/page_size page table per row.
# 262k booted on 10-15; the client drops longer sessions (capped corpus).
CONTEXT_LENGTH = 262144
POOL_TOKENS = 1048576
# GPU reference runs --max-running-requests 32 at every point; +2 is TPU pool slack.
MAX_RUNNING = min(CONCURRENCY, 32) + 2

CONFIG = deepcopy(MTP_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "zai-org/GLM-5.3-Flash",
        "reasoning_parser": "glm45",
        "tool_call_parser": "glm47",
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [b for b in (1, 2, 4, 8, 16, 32, 64) if b < 2 * MAX_RUNNING],
        # The 1M-token device pool validated on 10-15; no host tier.
        "max_total_tokens": POOL_TOKENS,
        "max_mamba_cache_size": 5 * MAX_RUNNING + 64,
        "enable_metrics": True,
        "enable_cache_report": True,
        "scheduler_recv_interval": 8,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
        # GPU parity: forced accept length 3.5 (match-expected, fixed tokens).
        "SGLANG_SIMULATE_ACC_LEN": "3.5",
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1",
        # The kernel's page tables cover 128k; longer sessions continue on the model.
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": "131072",
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS": str(MAX_RUNNING * (NUM_STEPS + 1)),
        "SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING": "1",
    }
)
