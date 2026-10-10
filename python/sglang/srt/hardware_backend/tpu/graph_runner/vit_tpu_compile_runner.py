"""Moved to :mod:`sglang_tpu.graph_runner.vit_tpu_compile_runner`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.graph_runner.vit_tpu_compile_runner"
)
