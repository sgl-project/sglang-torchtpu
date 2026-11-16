#!/usr/bin/env python3
"""Header-only EP read plan; no tensor payload reads, torch, or accelerator.

Use on one representative safetensors shard before changing reader policy.
Mirrors current plugin expert-key selection and span merging. Estimated bytes
are requested ranges, NOT physical/network bytes (filesystem caching differs).
"""
import argparse
import json
import re
import struct
from pathlib import Path

EXPERT = re.compile(r'^(.*)\.experts\.(\d+)\.')
MAX_HEADER = 64 * 1024 * 1024


def plan(header, start, end, gap):
    if start < 0 or end <= start or gap < 0:
        raise ValueError('invalid expert interval or merge gap')
    kept = []
    total = skipped = 0
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        a, b = meta['data_offsets']
        if type(a) is not int or type(b) is not int or not 0 <= a <= b:
            raise ValueError('invalid data offsets')
        total += b - a
        m = EXPERT.match(name)
        if m and not start <= int(m.group(2)) < end:
            skipped += 1
            continue
        kept.append((a, b, name))
    kept.sort()
    spans = []
    for a, b, _ in kept:
        if not spans or a - spans[-1][1] > gap:
            spans.append([a, b])
        else:
            spans[-1][1] = max(spans[-1][1], b)
    useful = sum(b - a for a, b, _ in kept)
    requested = sum(b - a for a, b in spans)
    return dict(expert_range=[start, end], merge_gap_bytes=gap,
                retained_keys=[name for _, _, name in kept], skipped_keys=skipped,
                tensor_payload_bytes=total, useful_bytes=useful,
                requested_span_bytes=requested, gap_overread_bytes=requested-useful,
                requested_over_useful=requested/useful if useful else None,
                spans=spans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard', type=Path)
    parser.add_argument('--ep-size', type=int, default=16)
    parser.add_argument('--experts', type=int, default=256)
    parser.add_argument('--merge-gap', type=int, default=16*1024*1024)
    args = parser.parse_args()
    if args.ep_size <= 0 or args.experts <= 0 or args.experts % args.ep_size:
        parser.error('expert count must be positive and divisible by EP size')
    if args.merge_gap < 0:
        parser.error('merge gap must be nonnegative')
    with args.shard.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('truncated header length')
        size, = struct.unpack('<Q', raw)
        if size > MAX_HEADER:
            raise ValueError('header exceeds safety bound')
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('truncated header')
        header = json.loads(raw)
    n = args.experts // args.ep_size
    plans = [plan(header, rank*n, (rank+1)*n, args.merge_gap)
             for rank in range(args.ep_size)]
    data_end = max((m['data_offsets'][1] for k,m in header.items()
                    if k != '__metadata__'), default=0)
    if args.shard.stat().st_size < 8 + size + data_end:
        raise ValueError('file shorter than declared payload')
    print(json.dumps(dict(shard=str(args.shard), header_bytes=8+size,
                         scope='indexed shard selection only; nonindexed shards read whole in production; payload NOT read; dtype fallback not modeled',
                         plans=plans), indent=2))


if __name__ == '__main__':
    main()
