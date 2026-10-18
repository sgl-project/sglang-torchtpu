"""GPT-OSS-120B EAGLE3 throughput sweep: the 8k1k workload at concurrency 64.

perf_c64.py's server with the EAGLE3 draft attached, so the pair measures what
speculation is worth at one shape.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/perf_eagle3_c64.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.perf_c64 import CONFIG as PERF_CONFIG

CONFIG = deepcopy(PERF_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "lmsys/EAGLE3-gpt-oss-120b-bf16",
        # Three drafts per verify, matching perf_dflash_c64.py's block size so
        # the two legs propose the same amount of work. topk=1 makes it a chain,
        # and the arg hook resolves num_draft_tokens to num_steps + 1.
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
    }
)
# random-ids prompts have no structure to predict, so real acceptance would
# collapse toward 1 and describe no served workload. Pin it near what GSM8K
# measures instead. Output text is meaningless under this knob -- never set it
# on an accuracy gate.
CONFIG["runtime_env"].update({"SGLANG_SIMULATE_ACC_LEN": "2.5"})
