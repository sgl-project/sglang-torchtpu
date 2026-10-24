# SPDX-License-Identifier: Apache-2.0
"""Stage only one host's slice of an MoE checkpoint into local memory (sparse files).

A checkpoint larger than the host's tmpfs (Beam BF16: 1.0 TB vs ~919 GB of
``/dev/shm``) still fits when each host keeps only the routed experts its EP
ranks own. Every shard is recreated at full logical size with the header, all
non-expert tensors and the in-range experts copied to their original offsets;
the bytes of other experts stay holes, which tmpfs does not back with memory.

The EP-local reader (``ep_local_expert_reader``, enabled by
``--weight-loader-disable-mmap`` on a model with
``supports_tpu_ep_local_expert_reads``) reads exactly those tensors, so serving
from the staged copy is byte-identical to serving from the source. A gap of
other experts shorter than the reader's span-merge distance is read as zeros
and discarded. Do not point any other loader at a staged copy.

    python -m sglang_tpu.stage_ep_local_checkpoint \
        --src /gcs/Beam-1.0-501B-A23B --dst /dev/shm/Beam-1.0-501B-A23B \
        --expert-range 0 128

Re-running with the same arguments skips shards that are already staged.
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from sglang_tpu.ep_local_expert_reader import _EXPERT_ID_KEY_RE

_COPY_CHUNK_BYTES = 64 * 1024 * 1024
# Kept tensors closer than this are copied as one range (fewer, larger reads).
_MERGE_GAP_BYTES = 1024 * 1024
_MARKER_SUFFIX = ".staged.json"


def kept_ranges(header_len: int, header: dict, expert_range) -> list:
    """Absolute byte ranges of ``[0, 8 + header_len)`` and every kept tensor, merged."""
    lo, hi = expert_range
    base = 8 + header_len
    ranges = [[0, base]]
    spans = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        m = _EXPERT_ID_KEY_RE.match(name)
        if m is not None and not lo <= int(m.group(2)) < hi:
            continue
        start, end = meta["data_offsets"]
        if end > start:
            spans.append((base + start, base + end))
    for start, end in sorted(spans):
        if start - ranges[-1][1] <= _MERGE_GAP_BYTES:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    return ranges


def _read_header(path: str):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return header_len, json.loads(f.read(header_len))


def stage_shard(src: str, dst: str, expert_range) -> int:
    """Stage one shard; returns the bytes copied (0 if already staged)."""
    marker = dst + _MARKER_SUFFIX
    size = os.path.getsize(src)
    want = {"expert_range": list(expert_range), "size": size}
    if os.path.exists(marker) and os.path.exists(dst):
        with open(marker) as f:
            if json.load(f) == want and os.path.getsize(dst) == size:
                return 0
    header_len, header = _read_header(src)
    ranges = kept_ranges(header_len, header, expert_range)
    copied = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.truncate(size)
        for start, end in ranges:
            pos = start
            while pos < end:
                buf = os.pread(fin.fileno(), min(_COPY_CHUNK_BYTES, end - pos), pos)
                if not buf:
                    raise IOError(f"{src} is shorter than its header says")
                os.pwrite(fout.fileno(), buf, pos)
                pos += len(buf)
                copied += len(buf)
    with open(marker, "w") as f:
        json.dump(want, f)
    return copied


def stage_checkpoint(src: str, dst: str, expert_range, threads: int = 16) -> int:
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        path = os.path.join(src, name)
        if os.path.isfile(path) and not name.endswith(".safetensors"):
            shutil.copyfile(path, os.path.join(dst, name))
    t0, total = time.time(), 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [
            pool.submit(
                stage_shard,
                os.path.join(src, shard),
                os.path.join(dst, shard),
                expert_range,
            )
            for shard in shards
        ]
        for i, fut in enumerate(futures):
            total += fut.result()
            if (i + 1) % 16 == 0 or i + 1 == len(futures):
                rate = total / max(time.time() - t0, 1e-9) / 1e9
                print(
                    f"staged {i + 1}/{len(shards)} shards, {total / 1e9:.1f} GB "
                    f"({rate:.2f} GB/s)",
                    flush=True,
                )
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="checkpoint dir with an index")
    ap.add_argument("--dst", required=True, help="local dir, e.g. under /dev/shm")
    ap.add_argument(
        "--expert-range",
        nargs=2,
        type=int,
        required=True,
        metavar=("START", "END"),
        help="routed expert ids [START, END) owned by this host's EP ranks",
    )
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)
    stage_checkpoint(args.src, args.dst, tuple(args.expert_range), args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
