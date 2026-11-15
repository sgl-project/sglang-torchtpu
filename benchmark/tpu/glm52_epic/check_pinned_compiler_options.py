"""Inspect exact pinned wheel source without importing or executing TPU code."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    path=Path('/wheels/torch-tpu/torch_tpu-0.2.1.dev20261028201649-cp312-cp312-manylinux_2_31_x86_64.whl')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82'
    with zipfile.ZipFile(path) as wheel:
        candidates=[name for name in wheel.namelist() if name.endswith('compiler_options.py')]
        assert candidates,candidates
        for name in candidates:
            raw=wheel.read(name);text=raw.decode()
            print('PINNED_COMPILER_OPTIONS_SOURCE '+json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),source=text)),flush=True)
            assert 'custom_compiler_options' in text
    print('PINNED_COMPILER_OPTIONS_SOURCE_PASS; sourcecontract only no runtime/thread/materialization proof',flush=True)


if __name__=='__main__':main()
