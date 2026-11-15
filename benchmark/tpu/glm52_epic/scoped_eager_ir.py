"""Benchmark-only post-initialization scheduler-loop compiler capture.

Install before scheduler import. No global dump flags; thread context spans
whole serving loop including delayed materialization. No added synchronization.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

TARGET='sglang.srt.managers.scheduler'


def wrap_loop(original, directory, context_factory=None, parallel_getter=None):
    @functools.wraps(original)
    def loop(owner,*args,**kwargs):
        if not owner.spec_algorithm.is_none():
            raise ValueError('scoped IR supports non-speculative eager loop only')
        factory=context_factory
        if factory is None:
            from torch_tpu._internal.compiler_options import custom_compiler_options
            factory=custom_compiler_options
        getter=parallel_getter
        if getter is None:
            from sglang.srt.runtime_context import get_parallel
            getter=get_parallel
        parallel=getter()
        root=Path(directory)/f'host-{os.environ.get("FALCON_RANK","unknown")}'/f'pid-{os.getpid()}'/'post-load'
        root.mkdir(parents=True,exist_ok=False)
        options=dict(xla_dump_to=str(root),xla_dump_hlo_as_text='true',xla_dump_hlo_as_proto='true')
        record=dict(pid=os.getpid(),wall_ns=time.time_ns(),tp_rank=parallel.tp_rank,
                    attn_dp_rank=parallel.attn_dp_rank,pp_rank=parallel.pp_rank,
                    phase='SCHEDULER_LOOP_AFTER_INITIALIZATION',options=options,
                    scope='options change cache fingerprint; no new drain or device fence')
        (root/'scope.json').write_text(json.dumps(record)+'\n')
        with factory(options):
            return original(owner,*args,**kwargs)
    return loop


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self,directory):self.directory=directory
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.loader is not None:
            spec.loader=Loader(spec.loader,self.directory)
        return spec


class Loader(importlib.abc.Loader):
    def __init__(self,loader,directory):self.loader,self.directory=loader,directory
    def create_module(self,spec):return self.loader.create_module(spec)
    def exec_module(self,module):
        self.loader.exec_module(module)
        for name in ('event_loop_normal','event_loop_overlap'):
            setattr(module.Scheduler,name,wrap_loop(getattr(module.Scheduler,name),self.directory))
    def __getattr__(self,name):return getattr(self.loader,name)


def install(directory):
    if TARGET in sys.modules:raise RuntimeError('scoped capture must install before scheduler import')
    if 'xla_dump_' in os.environ.get('XLA_FLAGS',''):
        raise ValueError('global dump flags would capture loading; refusing')
    if any(isinstance(f,Finder) for f in sys.meta_path):raise RuntimeError('scoped capture already installed')
    sys.meta_path.insert(0,Finder(directory))
