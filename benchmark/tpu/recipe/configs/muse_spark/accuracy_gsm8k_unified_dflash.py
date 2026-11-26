"""Muse Spark full GSM8K at low effort on the unified Full/SWA KV pool with DFlash.

accuracy.py owns the backbone serving shape the nightly score comes from; this file
only adds the DFlash draft and the unified pool, so the two scores are directly
comparable.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/accuracy_gsm8k_unified_dflash.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.accuracy import CONFIG as ACCURACY_CONFIG

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(
    enable_unified_memory=True,
    speculative_algorithm="DFLASH",
    speculative_draft_model_path="/weights/Muse-Spark-Open-DFlash2",
    speculative_num_draft_tokens=8,
)
CONFIG["runtime_env"].update(
    SGLANG_SPEC_TP_SYNC="rng",
    # The unified pool serves DFlash drafts only on the experimental spec worker.
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
)
