#!/usr/bin/env python3
"""CPU-only exact-byte check of one real indexed shard; not model loading.

Read headers until a shard with numeric experts is found. Compare every retained
weight/scale/shared tensor against safetensors mmap, with gap0 and production gap.
Reports logical readinto bytes, NOT physical GCS bytes or performance.
"""
import argparse
import gc
import json
import struct
from pathlib import Path
from unittest.mock import patch

from benchmark.tpu.glm52_epic.plan_expert_reads import MAX_HEADER, plan


def header(path):
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('truncated header length')
        size, = struct.unpack('<Q', raw)
        if size > MAX_HEADER:
            raise ValueError('header exceeds limit')
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError('truncated header')
        return json.loads(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--max-shard-bytes', type=int, default=4*1024**3)
    args = p.parse_args()
    if args.max_shard_bytes <= 0:
        p.error('positive shard byte cap required')
    index = json.loads((args.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    # Choose from index keys, not whole-file tensor reads or 282 shard scans.
    candidates = sorted({v for k,v in index.items() if '.experts.0.' in k})
    if not candidates:
        raise ValueError('no expert0 indexed shard')
    shard = args.checkpoint/candidates[0]
    if shard.stat().st_size > args.max_shard_bytes:
        raise ValueError('representative shard exceeds bounded-memory cap')
    h = header(shard)
    import safetensors
    import torch
    from sglang_tpu import ep_local_expert_reader as reader

    real_open = open
    for gap in (0, reader._SPAN_MERGE_GAP_BYTES):
        counters = {'payload_readinto_bytes': 0}

        class CountingFile:
            def __init__(self, stream): self.stream = stream
            def __getattr__(self, name): return getattr(self.stream, name)
            def __enter__(self): return self
            def __exit__(self, *exc): self.stream.close()
            def readinto(self, view):
                n = self.stream.readinto(view)
                counters['payload_readinto_bytes'] += n or 0
                return n

        def counted_open(path, mode='r', *a, **kw):
            return CountingFile(real_open(path, mode, *a, **kw))

        expected = plan(h, 0, 16, gap)
        r = reader.EpLocalExpertReader([str(shard)], (0, 16))
        print('SHARD_READ_BEGIN', json.dumps(dict(shard=shard.name, gap=gap,
              useful_bytes=expected['useful_bytes'], requested_span_bytes=expected['requested_span_bytes'])), flush=True)
        with patch.object(reader, '_SPAN_MERGE_GAP_BYTES', gap), patch.object(reader, 'open', counted_open, create=True):
            loaded = r.read(str(shard))
        r.verify()
        if set(loaded) != set(expected['retained_keys']):
            raise AssertionError('retained key mismatch')
        if counters['payload_readinto_bytes'] != expected['requested_span_bytes']:
            raise AssertionError('planned/readinto byte mismatch (check dtype fallback)')
        with safetensors.safe_open(str(shard), framework='pt', device='cpu') as baseline:
            for name, actual in loaded.items():
                target = baseline.get_tensor(name)
                if actual.dtype != target.dtype or actual.shape != target.shape:
                    raise AssertionError('dtype/shape mismatch: '+name)
                a = actual.contiguous().reshape(-1).view(torch.uint8)
                b = target.contiguous().reshape(-1).view(torch.uint8)
                # Exact FP8 bits, bounded temporary comparison chunks.
                for offset in range(0, a.numel(), 8*1024**2):
                    if not torch.equal(a[offset:offset+8*1024**2], b[offset:offset+8*1024**2]):
                        raise AssertionError('payload mismatch: '+name)
        print('EXPERT_SHARD_BYTE_PASS', json.dumps(dict(shard=shard.name, gap=gap,
              keys=len(loaded), skipped_keys=expected['skipped_keys'], **counters,
              useful_bytes=expected['useful_bytes'], gap_overread_bytes=expected['gap_overread_bytes'])), flush=True)
        del loaded, r, actual, target, a, b
        gc.collect()


if __name__ == '__main__':
    main()
