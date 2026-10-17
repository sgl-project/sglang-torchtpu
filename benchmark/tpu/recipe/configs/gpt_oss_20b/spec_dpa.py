"""GPT-OSS-20B with EAGLE3 under DP attention, for the single-host PR gate.

spec.py's configuration with attention data-parallel across two groups:
tp8 / dp2 gives attention_tp = 4, which is why the LM head is split too.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/spec_dpa.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_20b.spec import CONFIG as SPEC_CONFIG

DP_SIZE = 2

CONFIG = deepcopy(SPEC_CONFIG)
CONFIG["server_args"].update(
    {
        "dp_size": DP_SIZE,
        "enable_dp_attention": True,
        # attention_tp = tp_size / dp_size = 4 > 1, which TPU requires the LM
        # head to follow.
        "enable_dp_lm_head": True,
    }
)
