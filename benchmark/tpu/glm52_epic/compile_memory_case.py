"""Bound one fresh-process sparse probe and retain passive memory samples.

Not compile-only: includes allocation, bridge compilation, execution and oracle.
No checkpoint or server. Run each mode separately; sampler may miss transient
children. Neither RSS sums nor cgroup current are unique compiler memory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark.tpu.glm52_epic.startup_memory import collect


def supervise(command, directory, *, timeout, interval):
    if not all(math.isfinite(x) and x > 0 for x in (timeout, interval)):
        raise ValueError('finite positive timeout and interval required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'native.log').open('w') as log, (directory / 'memory.jsonl').open('w') as memory:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        started = time.monotonic()
        timed_out = False
        try:
            while True:
                data = collect(root_pids=(child.pid,))
                data['case_root_returncode'] = child.poll()
                memory.write(json.dumps(data) + '\n')
                memory.flush()
                if child.poll() is not None:
                    break
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    break
                time.sleep(min(interval, max(0.001, timeout - (time.monotonic() - started))))
        finally:
            # The new process group belongs only to this probe. Always reap it,
            # including on sampler/serialization failure or local interruption.
            if child.poll() is None or timed_out:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            child.wait()
        result = dict(command=command, pid=child.pid, returncode=child.returncode,
                      timed_out=timed_out, elapsed_s=time.monotonic() - started,
                      scope='whole probe; not pure compile or unique compiler peak')
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('eager', 'dynamo'), required=True)
    parser.add_argument('--case', choices=('prefill', 'decode-capacity'), default='prefill')
    parser.add_argument('--output', type=Path, required=True, help='New per-case evidence directory')
    parser.add_argument('--timeout', type=float, default=400)
    parser.add_argument('--interval', type=float, default=0.5)
    args = parser.parse_args()
    queries, entry = (1, 'per-query') if args.case == 'decode-capacity' else (2048, 'prefill-long-table')
    command = [sys.executable, '-m', 'benchmark.tpu.glm52_epic.repro_sparse_step',
               '--queries', str(queries), '--selection', 'self', '--entry', entry]
    if args.case == 'decode-capacity':
        command.append('--decode-capacity')
    if args.mode == 'dynamo':
        command.append('--dynamo')
    result = supervise(command, args.output, timeout=args.timeout, interval=args.interval)
    if result['returncode'] or result['timed_out']:
        return 2
    lines = (args.output / 'native.log').read_text().splitlines()
    if args.case == 'decode-capacity':
        for marker in ('DECODE_CAPACITY page_size=64 pages_per_seq=2113 live_kv=1',
                       'SPARSE_REQUESTS count=1 queries_per_request=1'):
            if lines.count(marker) != 1:
                raise RuntimeError('missing exact decode shape marker')
    dynamo_marker = 'PRODUCTION_SPARSE_DYNAMO_ENABLED fullgraph=True dynamic=False'
    if lines.count(dynamo_marker) != int(args.mode == 'dynamo'):
        raise RuntimeError('compile mode marker mismatch')
    for i in (0, 1):
        marker = f'PRODUCTION_SPARSE_STEP_PASS iteration={i} queries={queries} selection=self entry={entry}'
        if lines.count(marker) != 1:
            raise RuntimeError('missing exact numerical/cache oracle marker')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
