"""Moved to :mod:`sglang_tpu.dsv4.compress_and_store.kernel`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.dsv4.compress_and_store.kernel"
)
