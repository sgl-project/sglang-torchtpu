"""Moved to :mod:`sglang_tpu.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup"
)
