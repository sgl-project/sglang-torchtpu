"""GPT-OSS-120B DFlash throughput sweep: the 8k1k workload at concurrency 64.

perf_c64.py's server with the DFlash draft attached, the counterpart to
perf_eagle3_c64.py.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/perf_dflash_c64.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.perf_c64 import CONFIG as PERF_CONFIG

CONFIG = deepcopy(PERF_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "z-lab/gpt-oss-120b-DFlash",
        # Fixed by the z-lab checkpoint; the arg hook pins num_steps and
        # eagle_topk to 1, so block size is the only speculative knob that applies.
        "speculative_dflash_block_size": 3,
        # 1 disables the delay DFlash's scheduler auto-enables (4 slots at
        # max-running 64): it holds decode rows empty that the TPU batch bucket
        # pays for anyway, and the prefill batching buys nothing at ISL 8192.
        "min_free_slots_delay": 1,
    }
)
CONFIG["runtime_env"].update(
    {
        # random-ids prompts have no structure to predict, so real acceptance
        # would collapse toward 1. Pin it near what GSM8K measures. Never set
        # this on an accuracy gate: the output text is meaningless under it.
        "SGLANG_SIMULATE_ACC_LEN": "2.5",
        # Broadcast only memory init and the RNG draws across ranks.
        "SGLANG_SPEC_TP_SYNC": "rng",
        # Route DFLASH to sglang_tpu.experimental.TpuBlockSpecWorker, the worker the
        # DFlash gates are measured on (#1413).
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
    }
)
