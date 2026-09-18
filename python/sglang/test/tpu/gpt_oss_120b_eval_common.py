"""Shared serving configuration for GPT-OSS 120B TPU evals."""

import os
from copy import deepcopy
from typing import Any

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    model_path="/weights/gpt-oss-120b",
    served_model_name=SERVED_MODEL_NAME,
    reasoning_parser="gpt-oss",
    kv_cache_dtype=None,
    page_size=128,
    disable_radix_cache=True,
    mem_fraction_static=0.6,
    chunked_prefill_size=4096,
    max_prefill_tokens=4096,
    tpu_compile_tokens=[4096],
)


def make_model_config(
    *,
    test_name: str,
    series_id: str,
    tag: str,
    launch_config: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Build one isolated model configuration for one eval."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": test_name,
        # The dashboard's series join key: never renamed, never shown. A forced
        # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
        "id": series_id,
        # Cosmetic and free to change -- the split from `id` is what makes a
        # display rename a non-event for the trend.
        "display_name": "GPT-OSS 120B",
        "variant": "TP8+EP8",
        "dashboard": "accuracy",
        "dashboard_model": SERVED_MODEL_NAME,
        "launch": {
            **launch_config,
            "health_timeout": HEALTH_TIMEOUT,
            "tag": tag,
        },
        "gates": [gate],
    }
