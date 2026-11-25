"""Muse Spark PP8 prefill with batched RPA and fp32 NVFP4 scales (one tpu7x host).

The production prefill configuration on top of ``server_pp8.py``:

- ``pallas_tpu_rpa_batched`` attention (128-head query tiling, #1920). It needs
  ``page_size >= 256``.
- ``SGLANG_TPU_NVFP4_MOE_FP32_SCALES=1`` (#1922): fp32 group scales let stock
  gmm_v2 load them without the bf16 zero-stride path, for about +12 GB per rank.
- ``max_running_requests`` 16, so the PP8 pipeline gets two requests per
  micro-batch. The inherited 8 gives ``pp_max_micro_batch_size`` 1.

Measured on main f4788d3ea9 (cdk j-ffd35840, j-b1a2a8b7) with these settings
plus Raiden L2 HiCache, which this recipe leaves off: 64K cached (device hit) +
8K new TTFT 1,308 ms (rpa_v3 4,055 ms, TP8 1,366 ms); 0+8K prefill throughput
saturates at about 25.7K tok/s from C=8; GSM8K 0.970.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/server_pp8_batched_rpa.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server_pp8 import CONFIG as PP8_CONFIG

CONFIG = deepcopy(PP8_CONFIG)
CONFIG["server_args"].update(
    {
        "attention_backend": "pallas_tpu_rpa_batched",
        "page_size": 256,
        # The measured KV budget: 64K cached + 8K new per request, with room for
        # concurrent 8K prompts. A larger context needs a new memory profile.
        "context_length": 81920,
        "max_total_tokens": 114688,
        "swa_full_tokens_ratio": 1.0,
        "max_running_requests": 16,
        "torch_compile_max_bs": 16,
        "tpu_compile_bs": [1, 2, 4, 8, 16],
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_NVFP4_MOE_FP32_SCALES": "1",
    }
)
