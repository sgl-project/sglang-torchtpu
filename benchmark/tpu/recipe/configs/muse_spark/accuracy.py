"""Muse Spark NVFP4 backbone accuracy, without speculative or cache features."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    model_path="/weights/Muse-Spark-Open-816B-A42B-NVFP4",
    served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    disable_radix_cache=True,
)
