"""Gemma-4-26B-A4B-it: the Gemma-4-31B accuracy config with expert parallelism."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gemma4_31b.accuracy import CONFIG as GEMMA4_31B

CONFIG = deepcopy(GEMMA4_31B)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gemma-4-26B-A4B-it",
        "served_model_name": "gemma4-26b-a4b",
        # 704-wide experts are not lane-splittable 8 ways: each rank owns 16 whole experts.
        "ep_size": 8,
    }
)
