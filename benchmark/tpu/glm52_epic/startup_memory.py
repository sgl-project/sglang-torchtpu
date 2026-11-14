"""Passive benchmark-only host/cgroup startup sampler (no device imports)."""
import json
import os
from pathlib import Path
import time


def collect(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    record = {'time_ns': time.time_ns(), 'pid': os.getpid()}
    for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events',
                 'memory.events.local', 'memory.stat'):
        try:
            record[name] = (cgroup / name).read_text()
        except OSError:
            pass
    try:
        record['meminfo'] = (proc / 'meminfo').read_text()
    except OSError:
        pass
    processes = []
    for p in proc.glob('[0-9]*'):
        try:
            status = (p / 'status').read_text()
            fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
            rss = int(fields.get('VmRSS', '0 kB').split()[0])
            hwm = int(fields.get('VmHWM', '0 kB').split()[0])
            # Keep surviving compiler children whose allocations have receded
            # since their peak. Exited children still require external evidence.
            if max(rss, hwm) > 100000:
                processes.append({'pid': int(p.name), 'rss_kib': rss,
                                  'rss_hwm_kib': hwm,
                                  'ppid': int(fields.get('PPid', '0').strip()),
                                  'name': fields.get('Name', '').strip()})
        except (OSError, ValueError):
            pass
    record['processes'] = sorted(processes, key=lambda p: p['rss_kib'], reverse=True)
    return record


if __name__ == '__main__':
    out = Path('/tmp/tpu_logs/glm52-startup-memory.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        data = collect()
        with out.open('a') as stream:
            stream.write(json.dumps(data) + '\n')
            stream.flush()
        # Keep anon/file/shmem and host availability in durable stdout too:
        # cgroup usage includes file cache and must not be called compiler RSS.
        print('GLM52_HOST_MEMORY ' + json.dumps(data), flush=True)
        time.sleep(30)
