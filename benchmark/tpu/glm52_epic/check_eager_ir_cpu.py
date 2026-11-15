"""Check actual plugin environment override capture without creating devices."""
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    from benchmark.tpu.glm52_epic.eager_ir_capture import install
    with tempfile.TemporaryDirectory() as directory:
        os.environ.pop('GLM52_OWNED_IR_FLAGS',None)
        os.environ['XLA_FLAGS']=''
        root=install(directory)
        from sglang_tpu.env_override import apply_tpu_env_overrides
        apply_tpu_env_overrides()
        assert sys.getprofile() is None
        record=json.loads((root/'effective-compiler-env.json').read_text())
        assert '--xla_dump_hlo_as_proto' in record['env']['XLA_FLAGS']
        assert 'LIBTPU_INIT_ARGS' in record['env']
        print('EAGER_IR_REAL_ENV_CPU_PASS '+json.dumps(record),flush=True)
        print('CPU only; no backend initialized, no pinned compiler flag acceptance or fatal IR proof',flush=True)


if __name__=='__main__':
    main()
