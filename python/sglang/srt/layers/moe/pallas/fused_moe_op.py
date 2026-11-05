"""Alias for :mod:`sglang_tpu.kernels.ops.moe.fused_moe_op`; see this package's docstring for why it stays."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.kernels.ops.moe.fused_moe_op"
)
