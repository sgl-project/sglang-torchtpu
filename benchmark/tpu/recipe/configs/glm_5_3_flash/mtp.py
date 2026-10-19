"""GLM-5.3-Flash with its MTP head as a NEXTN draft, one tpu7x host (tp8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/mtp.py

The draft layer runs its own DSA indexer each step instead of reusing the
target's top-k (index sharing is not supported on TPU).
"""

import json
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

NUM_STEPS = 3
NO_INDEX_SHARE = {"index_share_for_mtp_iteration": False}

CONFIG = deepcopy(BASE_CONFIG)
_args = CONFIG["server_args"]
_overrides = json.loads(_args.get("json_model_override_args") or "{}")
_overrides.update(NO_INDEX_SHARE)
_overrides["text_config"] = {**_overrides.get("text_config", {}), **NO_INDEX_SHARE}
_args.update(
    {
        "speculative_algorithm": "NEXTN",
        "speculative_draft_model_path": _args["model_path"],
        "speculative_eagle_topk": 1,
        "speculative_num_steps": NUM_STEPS,
        "speculative_num_draft_tokens": NUM_STEPS + 1,
        # Linear-chain KDA speculation keeps one state slot per running request.
        "max_mamba_cache_size": _args["max_running_requests"],
        "json_model_override_args": json.dumps(_overrides),
        # Young-generation GC ~70x rarer: a collection on one TP rank stalls
        # every rank at the next all-reduce.
        "gc_threshold": [50000, 100, 100],
    }
)
# Module state is fixed after warmup; skip its Dynamo guards (#1472).
CONFIG["runtime_env"]["SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER"] = "1"
