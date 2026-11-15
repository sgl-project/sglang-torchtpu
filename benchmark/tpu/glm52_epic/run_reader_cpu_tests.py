#!/usr/bin/env python3
"""Run real reader tests on CPU images lacking unrelated compiled CPU kernels.

Only known import-boundary extension symbols are supplied. Any execution fails;
no model, reader, safetensors, torch or loader operation is mocked.
"""
import runpy
import sys
import types
from unittest.mock import patch


def unavailable(*args, **kwargs):
    raise AssertionError('reader CPU tests unexpectedly invoked compiled CPU kernel')


def main():
    import importlib.util
    if importlib.util.find_spec('sgl_kernel') is not None:
        runpy.run_path('test/registered/unit/hardware_backend/tpu/test_ep_local_expert_reader.py', run_name='__main__')
        return
    boundary = types.ModuleType('sgl_kernel')
    boundary.copy_all_layer_kv_cache_cpu = unavailable
    boundary.assign_req_to_token_pool_cpu = unavailable
    boundary.assign_extend_cache_locs_cpu = unavailable
    print('CPU_EXTENSION_BOUNDARY: 3 unused symbols fail on execution; real reader/loader tests', flush=True)
    with patch.dict(sys.modules, {'sgl_kernel': boundary}):
        runpy.run_path('test/registered/unit/hardware_backend/tpu/test_ep_local_expert_reader.py', run_name='__main__')


if __name__ == '__main__':
    main()
