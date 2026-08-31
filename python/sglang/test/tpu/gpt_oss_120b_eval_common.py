"""Shared serving configuration for GPT-OSS 120B TPU evals."""

import os
from typing import Any

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

_BASE_LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "EXTRA_ARGS": (
        f"--reasoning-parser gpt-oss --served-model-name {SERVED_MODEL_NAME}"
    ),
    "KV_CACHE_DTYPE": "",
    "PAGE_SIZE": "128",
    "DISABLE_RADIX_CACHE": "1",
    "MEM_FRACTION_STATIC": "0.60",
    "ISL": "4096",
    "TPU_COMPILE_TOKENS": "4096",
    "ENABLE_PROFILER_ENDPOINT": "0",
}


def make_launch_env(**overrides: str) -> dict[str, str]:
    """Return a fresh launch environment with eval-specific sizing."""
    env = dict(_BASE_LAUNCH_ENV)
    env.update(overrides)
    return env


def make_model_config(
    *, test_name: str, tag: str, launch_env: dict[str, str], gate: dict[str, Any]
) -> dict[str, Any]:
    """Build one isolated model configuration for one eval gate."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": test_name,
        "variant": "TP8+EP8",
        "dashboard": "accuracy",
        "launch": {
            "recipe": "recipe/gpt_oss_120b/server.sh",
            "env": launch_env,
            "health_timeout": HEALTH_TIMEOUT,
            "tag": tag,
        },
        "gates": [gate],
    }
