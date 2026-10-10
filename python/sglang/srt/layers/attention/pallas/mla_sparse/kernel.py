"""Moved to :mod:`sglang_tpu.attention.pallas.mla_sparse.kernel`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.attention.pallas.mla_sparse.kernel"
)
