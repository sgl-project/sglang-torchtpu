"""Moved to :mod:`sglang_tpu.layers.quantization.fp8_moe_fused_rs`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.quantization.fp8_moe_fused_rs"
)
