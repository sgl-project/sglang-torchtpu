"""AgentX (SemiAnalysis cc-traces) at the model's full 1M context, tp8 on one host:
MTP verify on the decode megakernel up to 128k of context, the model path beyond.

    AGENTX_CONC=4 python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/experimental/agentx_1m.py

Then `agentx profile --concurrency $AGENTX_CONC --public-dataset
semianalysis_cc_traces_weka_062126 --benchmark-duration 3600 ...` against port 30000.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import NUM_STEPS

CONCURRENCY = int(os.environ["AGENTX_CONC"])
CONTEXT_LENGTH = int(os.environ.get("AGENTX_CTX", 1048576))
PAGE_SIZE = 128
# The GPU reference runs --max-running-requests 32 at every point. One program set serves
# every point (the compile is paid once); a point with c lanes never batches past c.
MAX_RUNNING = int(os.environ.get("AGENTX_MRR", 16))
BUCKETS = [b for b in (1, 4, 8, 16, 32) if b < MAX_RUNNING] + [MAX_RUNNING]
# Decode steps whose longest request exceeds this run on the model path; the kernel's
# page-table prefetch sizes with max_len. Measured 1 h windows (tok/s per chip, c1/c4/c8/c16):
# 4,804 / 6,707 / 7,964 / 7,040 with this cap; every step on the kernel at 1M was slower.
MEGAKERNEL_MAX_LEN = int(os.environ.get("AGENTX_MK_MAX_LEN", 131072))

CONFIG = deepcopy(MTP_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "zai-org/GLM-5.3-Flash",
        "reasoning_parser": "glm45",
        "tool_call_parser": "glm47",
        "context_length": CONTEXT_LENGTH,
        "page_size": PAGE_SIZE,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": sorted(set(BUCKETS)),
        # One prefill program: at 1M the chunk cost is the context scan, not the padding.
        "tpu_compile_tokens": [8192],
        # 1M tokens = 13 GB per chip of fp8 DSA KV; two requests' worth of pool.
        "max_total_tokens": int(os.environ.get("AGENTX_POOL_TOKENS", 2097152)),
        "max_mamba_cache_size": 5 * MAX_RUNNING + 64,
        "enable_metrics": True,
        "enable_cache_report": True,
        "scheduler_recv_interval": 8,
        # NFS checkpoint load: 454 s with prefetch against ~19 min without.
        "weight_loader_prefetch_checkpoints": True,
        # Each rank's host-tier replica plus the load's page cache evicts the pod: drop it.
        "weight_loader_drop_cache_after_load": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
        # GPU parity: forced accept length 3.5 (match-expected, fixed tokens).
        "SGLANG_SIMULATE_ACC_LEN": "3.5",
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1",
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": str(MEGAKERNEL_MAX_LEN),
        "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS": str(MAX_RUNNING * (NUM_STEPS + 1)),
        # 1M context: the StreamIndex page table stays in HBM.
        "SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM": "1",
        "SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING": "1",
    }
)
