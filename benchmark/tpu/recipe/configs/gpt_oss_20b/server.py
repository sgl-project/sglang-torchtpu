"""GPT-OSS-20B (MXFP4 MoE) on one tpu7x host, 4 chips x 2 cores = tp8.

How this model is served. Every other recipe in this directory derives from it
and changes only what its measurement needs.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/server.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-20b",
        "served_model_name": "gpt-oss-20b",
        "tp_size": 8,
        # EP, not TP, for the 32 experts: sharding intermediate_size across
        # 8 ranks fits neither the MXFP4 32-element block nor the MXU tile.
        "ep_size": 8,
        # The checkpoint ships its own mxfp4 config; the base config's fp8
        # would re-quantize those weights.
        "quantization": None,
        # The TPU platform default, and the granularity a radix prefix match
        # is floored to.
        "page_size": 64,
        # What the single-host gates are measured at; low for a model needing
        # ~5 GB per chip, but raising it is a measured change of its own.
        "mem_fraction_static": 0.6,
        # Not "dynamo", which accuracy.py uses: under it a Raiden host KV
        # reload decodes a different continuation, failing the offload probe.
        "tpu_compile_tracer": "direct",
        # Serve with prefix caching; the base config turns the radix tree off
        # for random-prompt throughput parity, a measurement choice.
        "disable_radix_cache": False,
    }
)
# SparseCore ragged-gather MoE cannot be lowered for GPT-OSS: ragged_gather_v2
# needs a static row count, and EXTEND reaches it with a symbolic token dim.
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
