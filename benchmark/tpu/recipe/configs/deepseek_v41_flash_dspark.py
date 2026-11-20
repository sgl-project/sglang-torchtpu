"""DeepSeek-V4.1-Flash + its bundled DSpark draft on one tpu7x-8 host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark.py

The deepseek_v41_flash.py deployment with DSpark on the
experimental TPU spec worker: the target and draft graphs are traced by the
direct tracer and each compiled once on an owner rank (--use-spmd-compile).
The draft's MoE shards over the target's EP8 group (~1.6 GB/device).
VERIFY_SPMD_SAFE=1 passed on all 8 ranks for the target graphs; the four
draft block graphs share the compile too, and every rank's accept length
matched the per-rank compile (4.2-4.6).

Measured on one tpu7x-8 with MODEL=gs://... (weights and Engram streamed
from object storage, tokenizer from a local copy), cold compile caches:
ready in 10.5 min; 1024 in / 1024 out random-ids c64 1782 tok/s and c128
2999 tok/s at accept length 4.3-4.4; GSM8K 1319 at c128 0.970.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        # Measured at this context; not yet validated at the target's 1M.
        "context_length": 103424,
        # The draft weights are the mtp.* tensors of the target checkpoint.
        "speculative_algorithm": "DSPARK",
        # 5 draft tokens per step: the verify runs 6 rows per request.
        "speculative_dspark_block_size": 5,
        # At 0.85 the larger KV pool leaves the 2048-token extend program
        # ~280 MB short of HBM at load; 0.83 loads every graph.
        "mem_fraction_static": 0.83,
        # Under attention DP every rank runs a peer's prefill step, so lone
        # 1024-token prefills stall all 8 ranks' spec decode; batching them
        # takes c64 from 1004 to 1900 tok/s.
        "enable_prefill_delayer": True,
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1"})
