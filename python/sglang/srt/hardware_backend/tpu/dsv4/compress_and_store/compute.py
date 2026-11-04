"""Moved to :mod:`sglang_tpu.layers.attention.dsv4.compress_and_store.compute`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.attention.dsv4.compress_and_store.compute"
)
