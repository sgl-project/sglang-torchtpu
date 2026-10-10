"""Moved to :mod:`sglang_tpu.attention.pallas.gdn.compute_conv1d`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.attention.pallas.gdn.compute_conv1d"
)
