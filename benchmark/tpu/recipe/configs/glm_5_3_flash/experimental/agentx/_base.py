"""Server config shared by the AgentX recipes (agentx_cc<N>.py): GLM-5.3-Flash, tp8 on
one tpu7x host (4 chips), 1M context, MTP verify at InferenceX's golden acceptance.

With AGENTX_DUMMY=1 a recipe renders the launch JSON of its chart point (runs/*.json)
exactly. Without it (the submission) the server loads real weights from model_path;
REAL_WEIGHT_ARGS / REAL_WEIGHT_ENV are the only other changes (check_recipes.py).
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG

CONTEXT_LENGTH = 1048576
# InferenceX golden acceptance length (bonus token included) per MTP draft length K
# (speculative_num_steps; K + 1 draft tokens), SPEED-Bench coding, thinking_on
# (golden_al_distribution/glm-5.3-flash__mtp__native__default-default.yaml).
GOLDEN_AL = {1: 1.86, 2: 2.52, 3: 3.00, 4: 3.35, 5: 3.60, 6: 3.79, 7: 3.93, 8: 3.99}
# Real weights only, as the real-weight (GSM8K) boots of these configs ran.
REAL_WEIGHT_ARGS = {
    "weight_loader_prefetch_checkpoints": True,  # NFS load: 454 s, ~19 min without
    "weight_loader_drop_cache_after_load": True,  # load page cache + host tier evict the pod
}
REAL_WEIGHT_ENV = {
    "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S": "2400",  # skewed NFS loads pass the 480 s default
}


def agentx_config(
    *,
    num_steps: int,
    max_running: int,
    pool_tokens: int,
    mamba_slots: int,
    kv_shards: int = 1,
    hicache_ratio: float | None = None,
    state_host_ratio: float | None = None,
    token_split: bool = False,
    spmd_compile: bool = True,
) -> dict:
    """One AgentX server config: MTP with K = num_steps (topk 1, K + 1 draft tokens) at
    GOLDEN_AL[K]. pool_tokens caps the KV pool (HBM may bound it lower); mamba_slots
    (device KDA checkpoints) is sized for the conversations AgentX keeps cached, not from
    max_running. kv_shards ranks split every page of the MLA latent cache (1: unsplit).
    hicache_ratio sizes the host KV tier (None: 2x the device pool); state_host_ratio
    sizes the host KDA states on their own (None: as hicache_ratio)."""
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
            # One decode program per running count: no decode step is padded.
            "tpu_compile_bs": list(range(1, max_running + 1)),
            "tpu_compile_tokens": [1024, 2048, 3072, 4096],  # prefill programs
            "chunked_prefill_size": 4096,  # one 4096-token chunk per prefill step
            "max_prefill_tokens": 4096,  # a prefill batch fits one chunk
            "tpu_compile_tracer": "direct",  # trace graphs from the module, not through Dynamo
            "max_total_tokens": pool_tokens,
            "max_mamba_cache_size": mamba_slots,
            "speculative_eagle_topk": 1,
            "speculative_num_steps": num_steps,
            "speculative_num_draft_tokens": num_steps + 1,
            "enable_metrics": True,
            "enable_cache_report": True,  # cached_tokens in usage: the prefix hit rate
            "scheduler_recv_interval": 8,
        }
    )
    if hicache_ratio is not None:
        args["hicache_ratio"] = hicache_ratio
    env.update(
        {
            "SGLANG_TIMEOUT_KEEP_ALIVE": "900",  # outlast the client's pooled connections
            # Forced acceptance at the golden AL: floor/ceil per step with the AL as mean.
            "SGLANG_SIMULATE_ACC_LEN": str(GOLDEN_AL[num_steps]),
            "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed",  # committed tokens are one fixed id
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "0",  # decode on the model path
            # Megakernel caps: inert while it is off, set as the measured runs set them.
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": str(CONTEXT_LENGTH),
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS": str(
                max_running * (num_steps + 1)
            ),
            "SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM": "1",  # 1M context: page table stays in HBM
            "SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING": "1",
            # Pooled indexer: raw key|gate rows in a per-request ring, not one per pool token.
            "SGLANG_TPU_DSA_KPOOL_RAW_RING": "1",
        }
    )
    if kv_shards == 1:
        args["use_spmd_compile"] = spmd_compile
        # Unsplit cache: scattered prefill input off (SPMD slices rows at a per-rank offset).
        args["enable_attn_tp_input_scattered"] = False
    else:
        env["SGLANG_TPU_DSA_MLA_KV_SHARDS"] = str(kv_shards)
    if token_split:
        env["SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT"] = "1"
    if state_host_ratio is not None:
        env["SGLANG_TPU_HICACHE_STATE_RATIO"] = str(state_host_ratio)
    if os.environ.get("AGENTX_DUMMY") == "1":
        args["load_format"] = "dummy"  # random weights, as the chart points ran
    else:
        args.update(REAL_WEIGHT_ARGS)
        env.update(REAL_WEIGHT_ENV)
    return config
