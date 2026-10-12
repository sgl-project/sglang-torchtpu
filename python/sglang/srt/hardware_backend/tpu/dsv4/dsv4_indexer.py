"""Moved to :mod:`sglang_tpu.dsv4.dsv4_indexer`; alias kept for in-flight branches, delete once they merge."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sglang_tpu.dsv4.dsv4_indexer")
