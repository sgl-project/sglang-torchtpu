"""Moved to :mod:`sglang_tpu.multimodal.moonvit_tpu_compile_runner`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.multimodal.moonvit_tpu_compile_runner"
)
