"""Moved to :mod:`sglang_tpu.compilation.spec_decode_compile_policy_dflash`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.compilation.spec_decode_compile_policy_dflash"
)
