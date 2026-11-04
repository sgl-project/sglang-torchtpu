"""Moved to :mod:`sglang_tpu.layers.attention.pp_attention_capacity`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.attention.pp_attention_capacity"
)
