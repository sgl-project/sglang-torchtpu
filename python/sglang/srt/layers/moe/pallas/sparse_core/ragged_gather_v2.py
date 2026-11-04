"""Moved to :mod:`sglang_tpu.layers.moe.pallas.sparse_core.ragged_gather_v2`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.moe.pallas.sparse_core.ragged_gather_v2"
)
