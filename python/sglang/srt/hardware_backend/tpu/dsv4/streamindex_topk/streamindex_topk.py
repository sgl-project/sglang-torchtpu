"""Moved to :mod:`sglang_tpu.dsv4.streamindex_topk.streamindex_topk`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.dsv4.streamindex_topk.streamindex_topk"
)
