"""GLM-5.3-Flash with its MTP head as a NEXTN draft, one tpu7x host (tp8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/mtp.py

As the checkpoint config sets (index_share_for_mtp_iteration), the draft runs
its DSA indexer on the committed rows only and every draft step reuses that
top-k. GLM_FLASH_MTP_INDEX_SHARE=0 runs the draft's indexer on every step.
"""

import json
import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

NUM_STEPS = 3
INDEX_SHARE = os.environ.get("GLM_FLASH_MTP_INDEX_SHARE", "1") != "0"

CONFIG = deepcopy(BASE_CONFIG)
_args = CONFIG["server_args"]
if not INDEX_SHARE:
    no_index_share = {"index_share_for_mtp_iteration": False}
    _overrides = json.loads(_args.get("json_model_override_args") or "{}")
    _overrides.update(no_index_share)
    _overrides["text_config"] = {**_overrides.get("text_config", {}), **no_index_share}
    _args["json_model_override_args"] = json.dumps(_overrides)
_args.update(
    {
        "speculative_algorithm": "NEXTN",
        "speculative_draft_model_path": _args["model_path"],
        "speculative_eagle_topk": 1,
        "speculative_num_steps": NUM_STEPS,
        "speculative_num_draft_tokens": NUM_STEPS + 1,
        # Radix snapshot slots on top of one per running request; arbitrary.
        "max_mamba_cache_size": 4 * _args["max_running_requests"] + 16,
        # Young-generation GC ~70x rarer: a collection on one TP rank stalls
        # every rank at the next all-reduce.
        "gc_threshold": [50000, 100, 100],
    }
)
CONFIG["runtime_env"].update(
    {
        # MTP runs on the experimental chain spec worker (TpuChainSpecWorker).
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
        # Module state is fixed after warmup; skip its Dynamo guards (#1472).
        "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER": "1",
        # Greedy draft and accept agree across ranks; sync only random draws.
        "SGLANG_SPEC_TP_SYNC": "rng",
        # One scheduler per TP rank on its own NUMA-local cores.
        "SGLANG_SET_CPU_AFFINITY": "1",
    }
)
