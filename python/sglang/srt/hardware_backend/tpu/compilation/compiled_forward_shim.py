"""Moved to :mod:`sglang_tpu.compilation.compiled_forward_shim`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.compilation.compiled_forward_shim"
)
