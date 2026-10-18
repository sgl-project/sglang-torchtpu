"""GPT-OSS-120B EAGLE3 under DP attention, single-host GSM8K gate.

singlehost_spec.py's configuration with attention data-parallel across four
groups: tp8 / dp4 gives attention_tp = 2, which the LM head has to follow.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/spec_dpa.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.singlehost_spec import (
    CONFIG as SPEC_CONFIG,
)

DP_SIZE = 4

CONFIG = deepcopy(SPEC_CONFIG)
CONFIG["server_args"].update(
    {
        "dp_size": DP_SIZE,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
        # DP-local buckets: dp4 turns the global concurrency of 16 into four
        # per rank.
        "tpu_compile_bs": [1, 2, 4],
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
        # Not the recipe's direct tracer: this gate is measured on dynamo.
        "tpu_compile_tracer": "dynamo",
    }
)
