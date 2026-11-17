"""DeepSeek-V4.1-Flash, direct tracer + SPMD, real weights loaded while compiling.

deepseek_v41_flash_direct_spmd.py plus SGLANG_TPU_DSV41_DEFERRED_LOAD: boot with
dummy weights, trace and submit the async compiles, then stream the real checkpoint
(RunAI EP-local reads) while XLA compiles
(sglang_tpu/models/deepseek_v41/deferred_load.py). MODEL may be a gs:// path.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_direct_spmd import CONFIG as _BASE

CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_DSV41_DEFERRED_LOAD": "1",
        # Per-rank RunAI reads for a GCS checkpoint: more concurrency speeds the
        # tensor stream, but the Engram ranged reads slow down past 64.
        "RUNAI_STREAMER_CONCURRENCY": "64",
    }
)
