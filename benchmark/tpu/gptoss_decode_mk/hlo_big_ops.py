"""List the largest data-movement ops in XLA HLO dumps (find per-step weight/pool copies).

Dump the server's compiled programs with, e.g.
  LIBTPU_INIT_ARGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text" (or XLA_FLAGS=...)
then run
  python benchmark/tpu/gptoss_decode_mk/hlo_big_ops.py /tmp/xla [--min-mb 64]

For every *after_optimizations*.txt module it prints copy / copy-start / bitcast-convert /
convert / dynamic-update-slice / transpose / fusion / collective instructions whose result is
at least --min-mb, largest first. A healthy MK decode program moves nothing near the size of
the megakernel weights (w13q 5.2 GB, w2q 2.6 GB per rank) or of a KV pool (~0.8 GB).
"""

import argparse
import glob
import os
import re

DT = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "f8e4m3fn": 1,
    "f8e5m2": 1,
    "s16": 2,
    "u16": 2,
    "bf16": 2,
    "f16": 2,
    "s32": 4,
    "u32": 4,
    "f32": 4,
    "s64": 8,
    "u64": 8,
    "f64": 8,
}
OPS = (
    "copy",
    "copy-start",
    "bitcast-convert",
    "convert",
    "dynamic-update-slice",
    "transpose",
    "fusion",
    "all-gather",
    "all-reduce",
    "collective-permute",
    "all-to-all",
    "reduce-scatter",
)
LINE = re.compile(r"^\s*(?:ROOT )?%?([\w.\-]+) = ((?:\w+)\[[\d,]*\])\S* ([\w\-]+)\(")


def nbytes(shape: str) -> int:
    dt, dims = shape.split("[")
    n = DT.get(dt, 4)
    for d in dims.rstrip("]").split(","):
        if d:
            n *= int(d)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--min-mb", type=float, default=64.0)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    files = sorted(
        glob.glob(
            os.path.join(a.dump_dir, "**", "*after_optimizations*.txt"), recursive=True
        )
    )
    print(f"{len(files)} optimized modules under {a.dump_dir}")
    for f in files:
        hits, custom = [], 0
        for line in open(f, errors="replace"):
            if 'custom_call_target="tpu_custom_call"' in line:
                custom += 1
            m = LINE.match(line)
            if not m or m.group(3) not in OPS:
                continue
            b = nbytes(m.group(2))
            if b >= a.min_mb * 2**20:
                hits.append((b, m.group(3), m.group(1), m.group(2)))
        if hits:
            hits.sort(reverse=True)
            print(
                f"\n== {os.path.basename(f)} (tpu_custom_calls={custom}, "
                f"{len(hits)} ops >= {a.min_mb} MB, total {sum(h[0] for h in hits) / 2**30:.2f} GiB)"
            )
            for b, op, name, shape in hits[: a.top]:
                print(f"  {b / 2**20:10.1f} MB  {op:22s} {name}  {shape}")


if __name__ == "__main__":
    main()
