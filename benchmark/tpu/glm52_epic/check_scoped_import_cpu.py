"""Real scheduler import through scoped finder; no engine or TPU initialized."""
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch


def main():
    from benchmark.tpu.glm52_epic.run_reader_cpu_tests import unavailable
    from benchmark.tpu.glm52_epic.scoped_eager_ir import install
    boundary=types.ModuleType('sgl_kernel')
    for name in ('copy_all_layer_kv_cache_cpu','assign_req_to_token_pool_cpu',
                 'assign_extend_cache_locs_cpu','fill_accept_out_cache_loc_cpu',
                 'fill_bonus_tokens_cpu','build_tree_kernel_efficient_cpu','verify_tree_greedy_cpu'):
        setattr(boundary,name,unavailable)
    assert 'sglang.srt.managers.scheduler' not in sys.modules
    with tempfile.TemporaryDirectory() as directory:
        os.environ.pop('XLA_FLAGS',None)
        install(directory)
        with patch.dict(sys.modules,{'sgl_kernel':boundary}):
            from sglang.srt.managers.scheduler import Scheduler
            for name in ('event_loop_normal','event_loop_overlap'):
                method=getattr(Scheduler,name)
                assert method.__wrapped__.__name__==name
                assert method.__wrapped__.__module__=='sglang.srt.managers.scheduler'
            assert list(Path(directory).iterdir())==[]
            assert 'xla_dump_' not in os.environ.get('XLA_FLAGS','')
    print('SCOPED_REAL_SCHEDULER_IMPORT_CPU_PASS; real importfinder/methods; no loop/nativecompiler/deviceproof',flush=True)


if __name__=='__main__':main()
