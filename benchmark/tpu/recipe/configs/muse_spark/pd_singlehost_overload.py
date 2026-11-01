"""Muse Spark 1P1D overload stress: one tpu7x 2x2x1 TP8 host per engine.

Both engines serve the NVFP4 checkpoint at the full 256K context with fp8 KV and
multimodal input, and exchange KV over Raiden. The prefill engine keeps its
prefix cache and writes it through to a host pool, so an overload drives the
cache, the offload and the handoff at once. The KV and HiCache settings are
those of pd_singlehost_prefix_cache_kv_offload_dflash.py, which runs on the same
hosts, with the device pools sized by HBM instead of capped.

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
    # fp8 KV on the default RPA v3 path, at the DFlash 1P1D nightly's fraction:
    # its decode pool held 1.58M tokens there, six full-context requests, and
    # the fraction leaves more activation headroom than the base 0.92.
    kv_cache_dtype="fp8_e4m3",
    mem_fraction_static=0.85,
    # A disaggregated engine's warmup request needs its peer and the router.
    skip_server_warmup=True,
    # The sampler every TPU P/D gate pins (pd_reshard_harness.py notes the
    # distributed top-k crash behind it).
    sampling_backend="pytorch",
    disaggregation_transfer_backend="raiden",
    dist_timeout=3600,
)
_COMMON["runtime_env"].update(
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
    # Every prefilled page is written through to the host pool, so the offload
    # runs under the overload, not only once the device pool fills. Ratio 2, not
    # the DFlash nightly's 3: this HBM-sized pool puts the host pools near 35 GB
    # per rank.
    enable_hierarchical_cache=True,
    hicache_ratio=2.0,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
)

DECODE_CONFIG = deepcopy(_COMMON)
DECODE_CONFIG["server_args"].update(
    port=30000,
    disaggregation_mode="decode",
    # Overload can retract decode requests, and this engine has no HiCache host
    # pool to keep their KV in.
    disaggregation_decode_retraction_backup="cpu_tensor",
)
