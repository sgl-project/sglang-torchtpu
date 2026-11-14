#!/usr/bin/env python3
"""Inspect pinned installed launcher source without importing TorchTPU/TPU init."""
import hashlib
import importlib.metadata
import json


def main():
    dist = importlib.metadata.distribution('torch_tpu')
    if dist.version != '0.2.1.dev20261028201649':
        raise RuntimeError(f'unexpected TorchTPU version: {dist.version}')
    # The public invocation is torch.tpu.distributed; wheel layouts can ship
    # that namespace directly or register it from an internal TorchTPU module.
    selected = [p for p in dist.files or []
                if str(p).endswith('.py') and 'distributed' in str(p)
                and ('torch_tpu/' in str(p) or 'torch/tpu/' in str(p))]
    if not selected:
        print(json.dumps({'version': dist.version, 'inventory_candidates':
                          [str(p) for p in dist.files or []
                           if 'tpu' in str(p) and str(p).endswith('.py')]}), flush=True)
        raise RuntimeError('no launcher candidate in installed RECORD; inspect inventory before inferring absence')
    for relative in sorted(selected, key=str):
        data = dist.locate_file(relative).read_bytes()
        print(json.dumps({'path': str(relative), 'sha256': hashlib.sha256(data).hexdigest(),
                          'version': dist.version}), flush=True)
        print('SOURCE_BEGIN', str(relative), flush=True)
        print(data.decode('utf-8'), flush=True)
        print('SOURCE_END', str(relative), flush=True)


if __name__ == '__main__':
    main()
