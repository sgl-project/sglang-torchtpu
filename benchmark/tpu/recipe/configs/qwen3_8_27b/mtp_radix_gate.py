"""Qwen3.8-27B MTP (3/1/4) on the experimental TPU spec worker with the GDN radix
prefix cache (extra_buffer snapshots), one tpu7x host, tp8.

mtp_gate.py with the radix cache on; extra mamba slots hold the snapshots.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/mtp_radix_gate.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.qwen3_8_27b import mtp_gate

# Four snapshot slots per running request; arbitrary, enough for the gate's reuse.
MAMBA_SLOTS = 4 * mtp_gate.CONCURRENCY

CONFIG = deepcopy(mtp_gate.CONFIG)
CONFIG["server_args"].update(
    {
        "disable_radix_cache": False,
        "mamba_radix_cache_strategy": "extra_buffer",
        "max_mamba_cache_size": MAMBA_SLOTS,
    }
)
