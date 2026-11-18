"""Server config shared by the per-concurrency AgentX recipes (agentx_cc<N>.py):
GLM-5.3-Flash, tp8 on one tpu7x host (4 chips), 1M context, MTP verify.

AGENTX_DUMMY=1 serves random weights (load_format dummy), as the measured runs did.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import NUM_STEPS
from benchmark.tpu.recipe.configs.glm_5_3_flash.server import HICACHE_ARGS

CONTEXT_LENGTH = 1048576
# Decode steps whose longest request exceeds this run on the model path (1M: none do).
MEGAKERNEL_MAX_LEN = 1048576


def agentx_config(
    *,
    max_running: int,
    pool_tokens: int,
    kv_shards: int = 1,
    mamba_slots: int | None = None,
) -> dict:
    """max_running: running-request cap (the rest queue); pool_tokens: KV pool size;
    kv_shards: attention-TP ranks that split the MLA latent cache by page (>1: no
    host tier, decode on the model path); mamba_slots: device KDA checkpoint slots
    (default 5 * max_running + 64)."""
    if mamba_slots is None:
        mamba_slots = 5 * max_running + 64
    config = deepcopy(MTP_CONFIG)
    args, env = config["server_args"], config["runtime_env"]
    args.update(
        {
            "served_model_name": "zai-org/GLM-5.3-Flash",
            "reasoning_parser": "glm45",
            "tool_call_parser": "glm47",
            "context_length": CONTEXT_LENGTH,
            "page_size": 128,
            "max_running_requests": max_running,
            "torch_compile_max_bs": max_running,
            "tpu_compile_bs": [b for b in (1, 4, 8, 16, 32) if b < max_running]
            + [max_running],
            # Short turns run a small program instead of a padded 8192-token chunk.
            "tpu_compile_tokens": [1024, 2048, 4096, 8192],
            "max_total_tokens": pool_tokens,
            "max_mamba_cache_size": mamba_slots,
            "enable_metrics": True,
            "enable_cache_report": True,
            "scheduler_recv_interval": 8,
            # NFS checkpoint load: 454 s with prefetch against ~19 min without.
            "weight_loader_prefetch_checkpoints": True,
            # Each rank's host-tier replica plus the load's page cache evicts the pod: drop it.
            "weight_loader_drop_cache_after_load": True,
        }
    )
    env.update(
        {
            "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
            # GPU parity: forced accept length 3.5 (match-expected, fixed tokens).
            "SGLANG_SIMULATE_ACC_LEN": "3.5",
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1",
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": str(MEGAKERNEL_MAX_LEN),
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS": str(
                max_running * (NUM_STEPS + 1)
            ),
            # 1M context: the StreamIndex page table stays in HBM.
            "SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM": "1",
            "SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING": "1",
            # Pooled indexer: raw key|gate rows in a per-request ring, not one per pool token.
            "SGLANG_TPU_DSA_KPOOL_RAW_RING": "1",
        }
    )
    if kv_shards > 1:
        # The split latent cache refuses the host tier and the decode megakernel.
        for key in HICACHE_ARGS:
            args.pop(key)
        env["SGLANG_TPU_GLM_FLASH_MEGAKERNEL"] = "0"
        env["SGLANG_TPU_DSA_MLA_KV_SHARDS"] = str(kv_shards)
    if os.environ.get("AGENTX_DUMMY") == "1":
        args["load_format"] = "dummy"
        args.pop("weight_loader_prefetch_checkpoints")
        args.pop("weight_loader_drop_cache_after_load")
    return config
