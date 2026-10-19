"""Moved to :mod:`sglang_tpu.model_executor.runner.tpu_graph_runner`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.model_executor.runner.tpu_graph_runner"
)
