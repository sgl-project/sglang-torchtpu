"""Moved to :mod:`sglang_tpu.mem_cache.pool_host.kv_pool_host`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "sglang_tpu.mem_cache.pool_host.kv_pool_host"
)
