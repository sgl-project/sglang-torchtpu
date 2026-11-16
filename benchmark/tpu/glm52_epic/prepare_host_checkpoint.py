#!/usr/bin/env python3
"""Pre-scheduler, once-per-host checkpoint preparation, bounded host buffers.

Default is header-only plan. --execute explicitly reads planned ranges without
materializing tensors. Run once in the host launcher, never once per TP rank.
This warms kernel page cache only; cache residency is not guaranteed.
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.tpu.glm52_epic.plan_expert_reads import plan
from benchmark.tpu.glm52_epic.verify_expert_shard import header

BUFFER_BYTES = 4 * 1024 * 1024


def read_ranges(path, spans, data_base):
    """Read only declared ranges, fail closed on short reads; 4MiB/worker."""
    buf = bytearray(BUFFER_BYTES)
    view = memoryview(buf)
    total = 0
    with path.open('rb', buffering=0) as stream:
        for start, end in spans:
            stream.seek(data_base + start)
            remaining = end-start
            while remaining:
                n = stream.readinto(view[:min(remaining, len(buf))])
                if not n:
                    raise IOError(f'truncated checkpoint preparation read: {path.name}')
                remaining -= n
                total += n
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--expert-start', type=int, required=True)
    p.add_argument('--expert-end', type=int, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--max-shards', type=int, help='Explicit slice for CPU validation')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not 1 <= args.threads <= 8:
        p.error('threads must be within [1,8]')
    if not 0 <= args.expert_start < args.expert_end <= 256:
        p.error('explicit contiguous host union must be within GLM256 experts')
    if args.max_shards is not None and args.max_shards < 1:
        p.error('max-shards must be positive')
    index_bytes = (args.checkpoint/'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)['weight_map']
    names = sorted(set(index.values()))
    # Unindexed files (e.g. MTP) must not be silently claimed as prepared.
    unindexed = sorted(x.name for x in args.checkpoint.glob('*.safetensors') if x.name not in names)
    if args.max_shards is not None:
        names = names[:args.max_shards]
    jobs = []
    useful = requested = payload = 0
    for name in names:
        if Path(name).name != name:
            raise ValueError('index shard must be basename')
        path = args.checkpoint/name
        h = header(path)
        selected = plan(h, args.expert_start, args.expert_end, 0)
        # Read exact ranges, not16MiB gaps; warmup needs no tensor assembly.
        with path.open('rb') as f:
            base = 8+int.from_bytes(f.read(8), 'little')
        end = max((m['data_offsets'][1] for k,m in h.items() if k != '__metadata__'), default=0)
        if path.stat().st_size < base+end:
            raise IOError('checkpoint shard truncated')
        jobs.append((path, selected['spans'], base))
        useful += selected['useful_bytes']
        requested += selected['requested_span_bytes']
        payload += selected['tensor_payload_bytes']
    summary = dict(checkpoint=str(args.checkpoint), index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                   host_expert_union=[args.expert_start,args.expert_end], indexed_files=len(names),
                   unindexed_not_prepared=unindexed, selected_useful_bytes=useful,
                   requested_bytes=requested, total_indexed_payload_bytes=payload,
                   threads=args.threads, buffer_bytes_per_worker=BUFFER_BYTES,
                   execute=args.execute, slice_only=args.max_shards is not None,
                   scope='numeric expert union plus all unmatched/shared keys; not physical I/O or cache residency')
    print('HOST_PREPARE_PLAN',json.dumps(summary),flush=True)
    if args.execute:
        started=time.monotonic()
        # map receives small perfile descriptors only; no tensors or payload futures.
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            actual=sum(pool.map(lambda job:read_ranges(*job),jobs))
        if actual != requested:
            raise AssertionError('planned/read byte mismatch')
        print('HOST_PREPARE_PASS',json.dumps(dict(**summary,actual_read_bytes=actual,
              elapsed_s=time.monotonic()-started,host=os.uname().nodename)),flush=True)


if __name__ == '__main__':
    main()
