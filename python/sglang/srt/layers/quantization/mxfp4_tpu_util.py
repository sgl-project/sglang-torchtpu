"""Moved to :mod:`sglang_tpu.moe.mxfp4_tpu_util`; alias kept for the core TPU branches in
unquant.py and mxfp4.py, delete once those move into plugin MoE subclasses."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sglang_tpu.moe.mxfp4_tpu_util")
