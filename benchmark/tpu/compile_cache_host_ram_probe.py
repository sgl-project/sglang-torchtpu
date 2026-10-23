# SPDX-License-Identifier: Apache-2.0
"""Measure host RAM growth of the TPU compile cache per new prompt length.

Each TP rank runs in its own process and keeps its own copy of every
compiled executable, so a new prompt length that compiles costs host RAM
once per rank (sgl-project/sglang-torchtpu#1429). This probe sends prompts
of increasing length to a running server, samples the RSS of every
``sglang::scheduler*`` process on this host after each one, and reports the
growth per new length, per rank and per host.

Run it on the host that serves the prefill (or the single server)::

    python3 benchmark/tpu/compile_cache_host_ram_probe.py \\
        --url http://127.0.0.1:30000 --start 1 --step 4 --count 64

Pair it with ``SGLANG_TPU_CHECK_RECOMPILATION=1`` on the server: its
``[recompilation checker]`` lines report the same growth from inside each
rank, with the peak host memory each compile took.

Only the standard library is used, so the script runs anywhere ``/proc``
exists; ``--proc-root`` and ``--pids`` exist for tests and remote samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_MIB = float(1 << 20)
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
# setproctitle on the scheduler process; TP ranks append their rank.
_SCHEDULER_TITLE = "sglang::scheduler"


@dataclass(frozen=True)
class RankProcess:
    pid: int
    title: str


@dataclass
class Sample:
    """RSS of every rank process after one prompt length was served."""

    prompt_len: int
    rss_bytes: dict[int, int]  # pid -> RSS

    @property
    def host_bytes(self) -> int:
        return sum(self.rss_bytes.values())

    @property
    def per_rank_bytes(self) -> float:
        return self.host_bytes / len(self.rss_bytes) if self.rss_bytes else 0.0


@dataclass
class Growth:
    """Least-squares RSS growth per new prompt length over a run."""

    ranks: int
    lengths: int
    per_rank_per_length_bytes: float
    host_per_length_bytes: float
    host_total_bytes: int
    samples: list[Sample] = field(default_factory=list)


def _read_title(proc_root: str, pid: int) -> str | None:
    # setproctitle rewrites argv, so cmdline carries the full title and comm
    # its first 15 bytes; either identifies a scheduler.
    try:
        with open(f"{proc_root}/{pid}/cmdline", "rb") as f:
            title = f.read().split(b"\0", 1)[0].decode(errors="replace")
    except OSError:
        title = ""
    if title.startswith(_SCHEDULER_TITLE):
        return title
    try:
        with open(f"{proc_root}/{pid}/comm") as f:
            comm = f.read().strip()
    except OSError:
        return None
    return comm if comm.startswith(_SCHEDULER_TITLE[:15]) else None


def find_rank_processes(proc_root: str = "/proc") -> list[RankProcess]:
    """Every scheduler (TP rank) process on this host, by pid."""
    found = []
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        title = _read_title(proc_root, int(name))
        if title is not None:
            found.append(RankProcess(int(name), title))
    return sorted(found, key=lambda p: p.pid)


def read_rss_bytes(pid: int, proc_root: str = "/proc") -> int:
    """RSS of ``pid`` in bytes, 0 when it is gone or unreadable."""
    try:
        with open(f"{proc_root}/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


def sample(prompt_len: int, pids: Iterable[int], proc_root: str = "/proc") -> Sample:
    return Sample(prompt_len, {pid: read_rss_bytes(pid, proc_root) for pid in pids})


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ys over xs; 0 with fewer than two points."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var


def estimate_growth(samples: Sequence[Sample]) -> Growth:
    """Fit RSS against the number of new lengths served.

    The slope is per new length, not per token: a length that compiles adds
    executables of about the same size whatever the length is.
    """
    ranks = len(samples[0].rss_bytes) if samples else 0
    xs = [float(i) for i in range(len(samples))]
    host_slope = _slope(xs, [float(s.host_bytes) for s in samples])
    return Growth(
        ranks=ranks,
        lengths=len(samples),
        per_rank_per_length_bytes=host_slope / ranks if ranks else 0.0,
        host_per_length_bytes=host_slope,
        host_total_bytes=(samples[-1].host_bytes - samples[0].host_bytes)
        if samples
        else 0,
        samples=list(samples),
    )


def format_row(previous: Sample | None, current: Sample) -> str:
    delta = current.host_bytes - previous.host_bytes if previous else 0
    return (
        f"{current.prompt_len:>8} {len(current.rss_bytes):>5} "
        f"{current.per_rank_bytes / _MIB:>12.1f} {current.host_bytes / _MIB:>12.1f} "
        f"{delta / _MIB:>+10.1f}"
    )


def format_header() -> str:
    return (
        f"{'prompt':>8} {'ranks':>5} {'rss/rank MiB':>12} {'host MiB':>12} "
        f"{'host +MiB':>10}"
    )


def format_growth(growth: Growth) -> str:
    return (
        f"{growth.lengths} new length(s) on {growth.ranks} rank(s): "
        f"{growth.per_rank_per_length_bytes / _MIB:.2f} MiB per rank per length, "
        f"{growth.host_per_length_bytes / _MIB:.2f} MiB per host per length "
        f"(x{growth.ranks} rank processes), {growth.host_total_bytes / _MIB:+.0f} "
        "MiB host total over the run."
    )


def _generate(url: str, prompt_len: int, max_new_tokens: int, timeout: float) -> None:
    body = json.dumps(
        {
            # Plain ids avoid tokenizer variance; the server pads by length.
            "input_ids": [1000 + (i % 2000) for i in range(prompt_len)],
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--start", type=int, default=1, help="first prompt length")
    parser.add_argument("--step", type=int, default=4, help="length increment")
    parser.add_argument("--count", type=int, default=64, help="number of lengths")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="seconds to wait after each response before sampling RSS",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument(
        "--pids",
        type=int,
        nargs="*",
        help="rank pids to sample instead of discovering sglang::scheduler processes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.pids:
        pids = list(args.pids)
    else:
        procs = find_rank_processes(args.proc_root)
        if not procs:
            print(
                f"no {_SCHEDULER_TITLE}* process found under {args.proc_root}; "
                "run this on the serving host or pass --pids",
                file=sys.stderr,
            )
            return 2
        pids = [p.pid for p in procs]
        for p in procs:
            print(f"rank process {p.pid}: {p.title}", file=sys.stderr)
    samples: list[Sample] = []
    print(format_header())
    for i in range(args.count):
        prompt_len = args.start + i * args.step
        _generate(args.url, prompt_len, args.max_new_tokens, args.timeout)
        time.sleep(args.settle)
        current = sample(prompt_len, pids, args.proc_root)
        print(format_row(samples[-1] if samples else None, current), flush=True)
        samples.append(current)
    if samples:
        print(format_growth(estimate_growth(samples)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
