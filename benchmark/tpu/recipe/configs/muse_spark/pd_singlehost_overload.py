"""Muse Spark 1P1D overload stress: one tpu7x 2x2x1 TP8 host per engine.

Both engines serve the NVFP4 checkpoint at the full 256K context with fp8 KV and
multimodal input, and exchange KV over Raiden. The prefill engine keeps its
prefix cache and writes evicted KV back to a host pool, so an overload drives the
cache, the offload and the handoff at once.

A disaggregated engine does not shed load itself (max_queued_requests applies
only without disaggregation), so the test that boots this recipe launches each
engine on its own and puts a capped router in front of them:

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/pd_singlehost_overload.py:PREFILL_CONFIG
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

# Declares the P/D pair so each role can be exported on its own.
GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

MODEL_PATH = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
SERVED_MODEL_NAME = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"

_COMMON = deepcopy(SERVER_CONFIG)
_COMMON["server_args"].update(
    model_path=MODEL_PATH,
    served_model_name=SERVED_MODEL_NAME,
    host="0.0.0.0",
    enable_multimodal=True,
    enable_metrics=True,
    enable_cache_report=True,
    # The fp8 KV layout agentic.py serves Muse with (its _FP8_1M_KV): sequence-
    # along-lane pages and batched RPA prefill, without the draft settings.
    kv_cache_dtype="fp8_e4m3",
    page_size=256,
    attention_backend="pallas_tpu_rpa_batched",
    schedule_policy="fcfs",
    swa_full_tokens_ratio=0.0625,
    tpu_compile_tokens=[1024, 2048, 4096],
    # A disaggregated engine's warmup request needs its peer and the router.
    skip_server_warmup=True,
    # The sampler every TPU P/D gate pins (pd_reshard_harness.py notes the
    # distributed top-k crash behind it).
    sampling_backend="pytorch",
    disaggregation_transfer_backend="raiden",
    dist_timeout=3600,
)
_COMMON["runtime_env"].update(
    SGLANG_TPU_USE_SEQ_ALONG_LANE_KV="1",
    SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK="1",
    SGLANG_TPU_RPA_EXP2_SOFTMAX="1",
    SGLANG_TPU_RPA_ONES_V="1",
    SGLANG_TPU_RPA_TRANSPOSED="1",
    TPU_PREMAPPED_BUFFER_SIZE="4294967296",
    # A cold start writes tens of GB of executables into each of the tier-2 and
    # tier-3 compile caches, which CI keeps in the pod's 64Gi /dev/shm.
    ENABLE_COMPILE_CACHE="0",
)

# Independent role configs: mutating one engine must not change the other.
PREFILL_CONFIG = deepcopy(_COMMON)
PREFILL_CONFIG["server_args"].update(
    port=30000,
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=8998,
    # Host pool at twice the device pool, filled as the radix cache evicts: the
    # HiCache settings agentic.py runs over this fp8 layout.
    enable_hierarchical_cache=True,
    hicache_ratio=2.0,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_back",
)

DECODE_CONFIG = deepcopy(_COMMON)
DECODE_CONFIG["server_args"].update(
    port=30000,
    disaggregation_mode="decode",
    # Overload can retract decode requests, and this engine has no HiCache host
    # pool to keep their KV in.
    disaggregation_decode_retraction_backup="cpu_tensor",
)
