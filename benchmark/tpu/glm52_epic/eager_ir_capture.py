"""Explicit pre-runtime HLO capture setup; dormant until runner opts in.

Only established HLO flags. No unverified Jellyfish flags, tensors or fences.
"""
import json
import os
from pathlib import Path
import socket
import sys
import time


def install(directory):
    root=Path(directory)
    if not root.is_absolute():
        raise ValueError('absolute allocation-scoped capture directory required')
    # Initializing after backend import could capture the wrong compiler flags.
    if any(name in sys.modules for name in ('jax','torch_tpu','torch_xla')):
        raise RuntimeError('IR setup must precede backend imports')
    inherited=os.environ.get('XLA_FLAGS','')
    owned=os.environ.get('GLM52_OWNED_IR_FLAGS','')
    if owned:
        # Spawn inherits parent's env. Strip ONLY the exact suffix we emitted;
        # arbitrary external dump flags remain forbidden rather than hidden.
        if not inherited.endswith(owned):
            raise ValueError('owned dump flags changed after setup')
        inherited=inherited[:-len(owned)].strip()
    if 'xla_dump_' in inherited:
        raise ValueError('inherited dump flags/filter conflict with producer capture')
    pidroot=root/f'host-{os.environ.get("FALCON_RANK","unknown")}'/f'pid-{os.getpid()}'
    pidroot.mkdir(parents=True,exist_ok=False)
    hlo=pidroot/'hlo';hlo.mkdir()
    if any(c.isspace() for c in str(hlo)):
        raise ValueError('XLA dump directory cannot contain whitespace')
    flags=f'--xla_dump_to={hlo} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'
    os.environ['XLA_FLAGS']=(inherited+' '+flags).strip()
    os.environ['GLM52_OWNED_IR_FLAGS']=flags
    record=dict(pid=os.getpid(),ppid=os.getppid(),hostname=socket.gethostname(),
        falcon_rank=os.environ.get('FALCON_RANK'),wall_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),source_commit=os.environ.get('SOURCE_COMMIT'),
        inherited_xla_flags=inherited,final_xla_flags=os.environ['XLA_FLAGS'],
        libtpu_init_args_at_setup=os.environ.get('LIBTPU_INIT_ARGS'),
        scope='preimport setup only; final env_override/dependency/rank map still required')
    (pidroot/'capture-setup.json').write_text(json.dumps(record,indent=2)+'\n')
    # sys.setprofile observes the original override returning without replacing
    # production functions/import loaders. Remove immediately after one return.
    # Do not overwrite an existing profiler (would contaminate diagnostics).
    if sys.getprofile() is not None:
        raise RuntimeError('existing Python profiler conflicts with effective-env capture')
    def final_environment(frame, event, arg):
        if (event == 'return' and frame.f_code.co_name == 'apply_tpu_env_overrides'
                and frame.f_globals.get('__name__') == 'sglang_tpu.env_override'):
            sys.setprofile(None)
            selected={k:v for k,v in os.environ.items() if k in (
                'XLA_FLAGS','LIBTPU_INIT_ARGS','SOURCE_COMMIT','FALCON_RANK')
                or 'COMPILATION_CACHE' in k}
            (pidroot/'effective-compiler-env.json').write_text(json.dumps(dict(
                pid=os.getpid(),wall_ns=time.time_ns(),env=selected,
                scope='after base env_override; later shape-dependent overrides may append'),indent=2)+'\n')
    sys.setprofile(final_environment)
    return pidroot
