#!/usr/bin/env python3
"""Inspect pinned installed launcher source without importing TorchTPU/TPU init."""
import hashlib
import importlib.metadata
import json


def main():
    dist = importlib.metadata.distribution('torch_tpu')
    if dist.version != '0.2.1.dev20261028201649':
        raise RuntimeError(f'unexpected TorchTPU version: {dist.version}')
    selected = [p for p in dist.files or [] if str(p) == 'torch_tpu/distributed.py'
                or (str(p).startswith('torch_tpu/distributed/') and str(p).endswith('.py'))]
    if not selected:
        raise RuntimeError('distributed launcher Python source absent from installed wheel')
    for relative in sorted(selected, key=str):
        data = dist.locate_file(relative).read_bytes()
        print(json.dumps({'path': str(relative), 'sha256': hashlib.sha256(data).hexdigest(),
                          'version': dist.version}), flush=True)
        print('SOURCE_BEGIN', str(relative), flush=True)
        print(data.decode('utf-8'), flush=True)
        print('SOURCE_END', str(relative), flush=True)


if __name__ == '__main__':
    main()
