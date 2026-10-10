"""Moved to :mod:`sglang_tpu.moe.pallas.fused_rs.gmm_v2_gather_scatter`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.moe.pallas.fused_rs.gmm_v2_gather_scatter"
)
