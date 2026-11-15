#!/usr/bin/env python3
"""Explicit bounded TPU scoped-options microprobe; no checkpoint/server launch.

Run ONLY using verified exact bootstrap/pinned wheel on a managed TPU job.
This tests option acceptance/emission, not original eagerfatal or model accuracy.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import tempfile


def main():
    pins={'torch_tpu':'0.2.1.dev20261028201649','libtpu':'0.0.50','torch':'2.11.0'}
    versions={name:metadata.version(name) for name in pins}
    assert all(versions[name].split('+')[0]==pin for name,pin in pins.items()),versions
    assert 'xla_dump_' not in os.environ.get('XLA_FLAGS',''),'global dump flags forbidden'
    import torch
    import torch_tpu  # Register backend only AFTER preflight.
    from torch_tpu._internal.compiler_options import custom_compiler_options
    from benchmark.tpu.glm52_epic.archive_eager_ir import archive

    artifact=Path(os.environ['ARTIFACT_LOCAL_DIR'])
    assert artifact.is_absolute()
    target=artifact/'scoped-compiler-microprobe'
    target.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='glm52-scoped-probe-') as directory:
        dump=Path(directory)/'hlo';dump.mkdir()
        # Materialize loading/setup BEFORE context; tiny values have exact CPUoracle.
        x=torch.arange(16,dtype=torch.float32).reshape(4,4).to('tpu')
        outside=(x+1).cpu()
        assert torch.equal(outside,torch.arange(16,dtype=torch.float32).reshape(4,4)+1)
        assert list(dump.iterdir())==[]
        options=dict(xla_dump_to=str(dump),xla_dump_hlo_as_text='true',xla_dump_hlo_as_proto='true')
        with custom_compiler_options(options):
            # Different operation from setup avoids a setup cachehit hiding dump.
            actual=(x*x+3).cpu()  # Explicit probe materialization, NOT serving instrumentation.
        expected=torch.arange(16,dtype=torch.float32).reshape(4,4).square()+3
        assert torch.equal(actual,expected),'microprobe numerical mismatch'
        files=[p for p in dump.rglob('*') if p.is_file()]
        assert any(p.suffix=='.txt' for p in files),'no HLO text emitted'
        assert any(p.suffix=='.pb' for p in files),'no HLO proto emitted'
        assert len(files)<=1000 and sum(p.stat().st_size for p in files)<=128*1024**2,'probe artifactbudget exceeded'
        inventory=[dict(path=str(p.relative_to(dump)),size=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
        archive(dump,target/'compiler-ir.tar')
        result=dict(status='SCOPED_COMPILER_TPU_MICROPROBE_PASS',versions=versions,
                    options=options,inventory=inventory,
                    scope='tiny TPU numerical/materialization and scoped HLOemission only; not model/fatal/performance proof')
        (target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
