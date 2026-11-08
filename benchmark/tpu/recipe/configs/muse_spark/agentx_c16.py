"""Muse Spark AgentX at concurrency 16 on one tpu7x host; the profile is in agentx.py.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentx_c16.py
"""

from benchmark.tpu.recipe.configs.muse_spark.agentx import config_for

CONFIG = config_for(16)
