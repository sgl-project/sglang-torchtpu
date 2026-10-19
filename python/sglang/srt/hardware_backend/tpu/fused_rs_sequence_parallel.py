"""Moved to :mod:`sglang_tpu.layers.moe.fused_rs_sequence_parallel`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.moe.fused_rs_sequence_parallel"
)
