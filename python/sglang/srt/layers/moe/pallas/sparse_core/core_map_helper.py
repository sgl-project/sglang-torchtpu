"""Moved to :mod:`sglang_tpu.layers.moe.pallas.sparse_core.core_map_helper`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.moe.pallas.sparse_core.core_map_helper"
)
