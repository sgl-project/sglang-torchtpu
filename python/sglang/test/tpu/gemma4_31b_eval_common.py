"""Shared serving and sampling configuration for Gemma-4-31B TPU evals."""

import os
from typing import Any
from urllib.parse import urlsplit

from sglang.test.tpu.eval_support import sgl_eval_runner

SERVED_MODEL_NAME = "gemma4-31b"
CONFIG_MODULE = "recipe/configs/gemma4_31b/accuracy.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
# MODEL=None keeps a stray MODEL in the job env from overriding the recipe's path.
RUNTIME_ENVS = {
    "MODEL": None,
    "HOST": urlsplit(BASE_URL).hostname,
    "PORT": str(urlsplit(BASE_URL).port),
}


def make_runner(eval_name: str, *, n_repeats: int = 1):
    """sgl-eval with the model card's sampling and thinking on."""
    return sgl_eval_runner(
        eval_name,
        model=SERVED_MODEL_NAME,
        max_tokens=32768,
        num_threads=64,
        temperature=1.0,
        top_p=0.95,
        n_repeats=n_repeats,
        # The Gemma 4 template reads enable_thinking; sgl-eval's --thinking sets `thinking`.
        chat_template_kwargs={"enable_thinking": True},
    )


def make_model_config(
    *, test_name: str, series_id: str, gate: dict[str, Any]
) -> dict[str, Any]:
    """Build one isolated model configuration for one eval."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": test_name,
        # The dashboard's series join key: never renamed, never shown.
        "id": series_id,
        "display_name": "Gemma 4 31B",
        "variant": "TP8",
        "dashboard": "accuracy",
        "dashboard_model": SERVED_MODEL_NAME,
        "gates": [gate],
    }
