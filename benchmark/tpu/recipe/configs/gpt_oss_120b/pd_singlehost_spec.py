"""One-host-per-engine Raiden P/D with EAGLE3 on both engines."""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

DRAFT_MODEL = (
    os.environ.get("GPT_OSS_DRAFT_MODEL")
    or os.environ.get("DRAFT_MODEL")
    or "lmsys/EAGLE3-gpt-oss-120b-bf16"
)
GROUP = deepcopy(pd_singlehost.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _config["server_args"].update(
        kv_cache_dtype="bfloat16",
        disable_radix_cache=True,
        mem_fraction_static=0.6,
        max_total_tokens=8192,
        max_running_requests=16,
        context_length=8192,
        max_prefill_tokens=4096,
        torch_compile_max_bs=16,
        tpu_compile_bs=[1, 2, 4, 16],
        tpu_compile_tokens=[512, 2048, 4096],
        watchdog_timeout=7200,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=DRAFT_MODEL,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    if _config["server_args"]["model_path"].startswith("gs://"):
        _config["server_args"]["load_format"] = "runai_streamer"
    if DRAFT_MODEL.startswith("gs://"):
        _config["server_args"]["speculative_draft_load_format"] = "runai_streamer"
    _config["runtime_envs"].update(
        SGLANG_RAIDEN_RESHARD="0",
        HEALTH_TIMEOUT=os.environ.get("WAIT_SECS", "5400"),
    )

# Spec uses CPU-tensor retraction; host-pool I/O is tested separately.
DECODE_CONFIG["server_args"].pop("hicache_mem_layout")
DECODE_CONFIG["server_args"].pop("hicache_io_backend")
DECODE_CONFIG["server_args"]["disaggregation_decode_retraction_backup"] = "cpu_tensor"
