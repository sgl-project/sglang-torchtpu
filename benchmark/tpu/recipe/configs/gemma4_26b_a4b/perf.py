"""Gemma-4-26B-A4B-it throughput config: the 31B sweep, MoE checkpoint.

Same family, same shapes and the same single sweep point as
configs/gemma4_31b/perf.py; the deltas are the checkpoint and expert
parallelism.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gemma4_31b.perf import CONFIG as DENSE_CONFIG

CONFIG = deepcopy(DENSE_CONFIG)
CONFIG["server_args"].update(
    {
        "served_model_name": "google/gemma-4-26B-A4B-it",
        # EP, not TP, for the experts: 704-wide intermediates are not
        # lane-splittable 8 ways, so each rank owns whole experts (128/8).
        "ep_size": CONFIG["server_args"]["tp_size"],
    }
)
