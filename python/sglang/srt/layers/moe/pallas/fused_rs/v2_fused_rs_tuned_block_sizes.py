"""Moved to :mod:`sglang_tpu.moe.pallas.fused_rs.v2_fused_rs_tuned_block_sizes`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.moe.pallas.fused_rs.v2_fused_rs_tuned_block_sizes"
)
