"""One-host-per-engine Raiden P/D with DFlash and RAM KV offload on both engines.

The caches of pd_singlehost_kv_offload_spec.py with DFlash on the experimental
TPU spec worker (TpuBlockSpecWorker) instead of EAGLE3. The decode engine keeps
full-attention and DFlash draft KV in RAM, and every request's sliding window
comes from prefill.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost_kv_offload_spec

DRAFT_MODEL = (
    os.environ.get("GPT_OSS_DRAFT_MODEL")
    or os.environ.get("DRAFT_MODEL")
    or "z-lab/gpt-oss-120b-DFlash"
)

GROUP = deepcopy(pd_singlehost_kv_offload_spec.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost_kv_offload_spec.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost_kv_offload_spec.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _args = _config["server_args"]
    # The DFlash arg hook derives these from the draft config's block size.
    for _key in (
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_draft_load_format",
    ):
        _args.pop(_key, None)
    _args.update(
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_MODEL,
    )
    if DRAFT_MODEL.startswith("gs://"):
        _args["speculative_draft_load_format"] = "runai_streamer"
    _config["runtime_envs"].update(
        SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
        # Broadcast only the sites that can differ across the TP ranks.
        SGLANG_SPEC_TP_SYNC="rng",
    )

# DFlash's auto min-free-slots delay holds decode rows the batch bucket pays for
# anyway; 1 disables it.
DECODE_CONFIG["server_args"]["min_free_slots_delay"] = 1
