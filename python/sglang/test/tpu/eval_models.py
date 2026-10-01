"""Per-model serving identity for the TPU accuracy nightlies.

``model_config.py`` is the engine that turns a ``MODEL_CONFIG`` dict into tests;
this module is the data those dicts are built from. A new model is an entry
here, not a new module -- one entry serves every benchmark file for that model.

An entry says how its server starts, in one of two ways:

``config_module``
    A Python recipe the test's own ``TPULauncherTestCase`` boots. The config is
    then built without a ``launch`` key, because the fixture owns the server.
``recipe`` + ``launch_env``
    A shell recipe that ``model_config.py``'s launcher runs, with per-benchmark
    sizing layered on by ``make_launch_env`` -- for models whose server is
    resized per benchmark rather than shared across them.

``sampling`` is the model card's decoding settings, used by ``make_runner`` for
gates scored through sgl-eval directly.
"""

import os
from typing import Any
from urllib.parse import urlsplit

from sglang.test.tpu.eval_support import sgl_eval_runner

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GPT_OSS_120B = {
    "served_model_name": "gpt-oss-120b",
    "display_name": "GPT-OSS 120B",
    "variant": "TP8+EP8",
    "recipe": "recipe/gpt_oss_120b/server.sh",
    "launch_env": {
        "MODEL": "/weights/gpt-oss-120b",
        "EXTRA_ARGS": ("--reasoning-parser gpt-oss --served-model-name gpt-oss-120b"),
        "KV_CACHE_DTYPE": "",
        "PAGE_SIZE": "128",
        "DISABLE_RADIX_CACHE": "1",
        "MEM_FRACTION_STATIC": "0.60",
        "ISL": "4096",
        "TPU_COMPILE_TOKENS": "4096",
        "ENABLE_PROFILER_ENDPOINT": "0",
    },
}

# Both Gemma 4 checkpoints take the same card sampling; only the recipe differs.
_GEMMA4_SAMPLING = {
    "max_tokens": 32768,
    "num_threads": 64,
    "temperature": 1.0,
    "top_p": 0.95,
    # The Gemma 4 template reads enable_thinking; sgl-eval's --thinking sets `thinking`.
    "chat_template_kwargs": {"enable_thinking": True},
}

GEMMA4_31B = {
    "served_model_name": "gemma4-31b",
    "display_name": "Gemma 4 31B",
    "variant": "TP8",
    "config_module": "recipe/configs/gemma4_31b/accuracy.py",
    "sampling": _GEMMA4_SAMPLING,
}

GEMMA4_26B_A4B = {
    "served_model_name": "gemma4-26b-a4b",
    "display_name": "Gemma 4 26B-A4B",
    "variant": "TP8+EP8",
    "config_module": "recipe/configs/gemma4_26b_a4b/accuracy.py",
    "sampling": _GEMMA4_SAMPLING,
}


def runtime_envs(base_url: str = BASE_URL) -> dict[str, str | None]:
    """Fixture environment: MODEL=None keeps a stray job MODEL from winning."""
    return {
        "MODEL": None,
        "HOST": urlsplit(base_url).hostname,
        "PORT": str(urlsplit(base_url).port),
    }


def make_launch_env(model: dict[str, Any], **overrides: str) -> dict[str, str]:
    """A fresh launch environment for one benchmark's sizing."""
    return {**model["launch_env"], **overrides}


def make_runner(model: dict[str, Any], eval_name: str, *, n_repeats: int = 1):
    """Score one benchmark through sgl-eval with this model's card sampling."""
    return sgl_eval_runner(
        eval_name,
        model=model["served_model_name"],
        n_repeats=n_repeats,
        **model["sampling"],
    )


def make_model_config(
    model: dict[str, Any],
    *,
    test_name: str,
    series_id: str,
    gate: dict[str, Any],
    tag: str | None = None,
    launch_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one isolated model configuration for one benchmark."""
    served = model["served_model_name"]
    config = {
        "model": served,
        "test_name": test_name,
        # The dashboard's series join key: never renamed, never shown. A forced
        # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
        "id": series_id,
        # Cosmetic and free to change -- the split from `id` is what makes a
        # display rename a non-event for the trend.
        "display_name": model["display_name"],
        "variant": model["variant"],
        "dashboard": "accuracy",
        "dashboard_model": served,
        "gates": [gate],
    }
    if "recipe" in model:
        config["launch"] = {
            "recipe": model["recipe"],
            "env": launch_env if launch_env is not None else dict(model["launch_env"]),
            "health_timeout": HEALTH_TIMEOUT,
            "tag": tag,
        }
    return config
