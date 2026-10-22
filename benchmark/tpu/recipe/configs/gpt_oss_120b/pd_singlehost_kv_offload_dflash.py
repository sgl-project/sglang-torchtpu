"""One-host-per-engine Raiden P/D with DFlash and RAM KV offload on both engines.

pd_singlehost_kv_offload_spec.py with DFlash on the experimental spec worker in
place of EAGLE3. The decode engine keeps full-attention and DFlash draft KV in
RAM while every request's sliding window comes from prefill.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost_kv_offload_spec

DRAFT_MODEL = (
    os.environ.get("GPT_OSS_DRAFT_MODEL")
    or os.environ.get("DRAFT_MODEL")
    or "z-lab/gpt-oss-120b-DFlash"
)
# Fixed by the z-lab DFlash checkpoint.
DFLASH_BLOCK_SIZE = 6

GROUP = deepcopy(pd_singlehost_kv_offload_spec.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost_kv_offload_spec.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost_kv_offload_spec.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _args = _config["server_args"]
    for _key in (
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    ):
        _args.pop(_key)
    _args.update(
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_MODEL,
        speculative_dflash_block_size=DFLASH_BLOCK_SIZE,
    )
    if DRAFT_MODEL.startswith("gs://"):
        _args["speculative_draft_load_format"] = "runai_streamer"
    else:
        _args.pop("speculative_draft_load_format", None)
    _config["runtime_envs"].update(
        SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
        SGLANG_SPEC_TP_SYNC="rng",
    )
