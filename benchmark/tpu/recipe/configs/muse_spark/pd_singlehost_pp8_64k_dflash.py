"""Muse Spark DFlash P/D at 64K context: PP8 batched-RPA prefill + TP8 decode.

The production variant of ``pd_singlehost_pp8_prefix_cache_kv_offload_dflash.py``
(the CI gate, which runs rpa_v3, page 128 and an 8K context):

- Prefill: ``pallas_tpu_rpa_batched`` at page 256 with
  ``SGLANG_TPU_NVFP4_MOE_FP32_SCALES=1``.
- Decode: page 256 too. Raiden moves whole pages, and the layout check refuses a
  page-size mismatch.
- Decode radix cache on, so a cached prefix is not shipped again. This halves
  64K + 8K TTFT (3,543 ms -> 1,720 ms).
- Prefill ``mem_fraction_static`` 0.94 and ``swa_full_tokens_ratio`` 0.25. The
  fp32 scales (+12 GB/rank), the DFlash2 draft (weights and KV on the last
  stage only) and the pipeline's per-slot SWA pin leave the last stage only
  15,872 full-attention tokens at 0.90, and the pool size is the minimum over
  stages, so every 73K request was rejected.
- Chunk transfer timeout 1,800 s instead of 60 s, so queued 64K transfers are
  not aborted under load.

Keep ``kv_cache_dtype`` bfloat16: the DFlash2 draft has 8 KV heads, one per TP8
decode rank, and fp8 token-major packing needs an even count
(``Partial FP8 head packs are unsupported``). Do not enable
``SGLANG_RAIDEN_PIPELINE_EARLY_PARTS``/``EARLY_CLOSE`` with the decode radix
cache: the transfer fails with ``the decode holds no full-attention page for a
window twin``.

Measured on main f4788d3ea9 (cdk j-acf906f3, arm B): 64K cached + 8K new, C=1,
router TTFT median 1,720 ms, KV transfer 249 ms.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_pp8_prefix_cache_kv_offload_dflash import (
    DECODE_CONFIG as _GATE_DECODE_CONFIG,
)
from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_pp8_prefix_cache_kv_offload_dflash import (
    GROUP,
)
from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_pp8_prefix_cache_kv_offload_dflash import (
    PREFILL_CONFIG as _GATE_PREFILL_CONFIG,
)

__all__ = ["GROUP", "PREFILL_CONFIG", "DECODE_CONFIG"]

CONTEXT_LENGTH = 81920
PAGE_SIZE = 256
_TRANSFER_ENV = {"SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S": "1800"}

PREFILL_CONFIG = deepcopy(_GATE_PREFILL_CONFIG)
PREFILL_CONFIG["server_args"].update(
    attention_backend="pallas_tpu_rpa_batched",
    page_size=PAGE_SIZE,
    context_length=CONTEXT_LENGTH,
    max_total_tokens=114688,
    mem_fraction_static=0.94,
    swa_full_tokens_ratio=0.25,
)
PREFILL_CONFIG["runtime_env"].update(
    SGLANG_TPU_NVFP4_MOE_FP32_SCALES="1",
    **_TRANSFER_ENV,
)

DECODE_CONFIG = deepcopy(_GATE_DECODE_CONFIG)
DECODE_CONFIG["server_args"].update(
    page_size=PAGE_SIZE,
    context_length=CONTEXT_LENGTH,
    disaggregation_decode_enable_radix_cache=True,
)
DECODE_CONFIG["runtime_env"].update(_TRANSFER_ENV)
