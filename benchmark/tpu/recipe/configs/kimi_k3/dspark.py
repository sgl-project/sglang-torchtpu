"""Kimi-K3 + DSpark speculative decoding on four tpu7x hosts, one request at a time.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/dspark.py

The server.py deployment plus the RadixArk/Kimi-K3-DSpark draft on the
experimental TPU spec worker. Sized by the long_8k1k_c1 profile: with
SGLANG_TPU_K3_STATIC_FP8 on, a verify batch of more than one request produces
wrong tokens, so this recipe serves a single stream.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "DSPARK",
        "speculative_draft_model_path": "RadixArk/Kimi-K3-DSpark",
        # 7 draft tokens per step: the verify runs 8 rows per request.
        "speculative_dspark_block_size": 7,
        # The draft is a dense MHA model; without this it inherits the target's
        # MLA backend.
        "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
        # One KDA state slot per running request.
        "max_mamba_cache_size": 1,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
    }
)
