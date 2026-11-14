"""Passive benchmark-only host/cgroup startup sampler (no device imports)."""
import json
import os
from pathlib import Path
import socket
import time


def log_snapshot(path, max_bytes=4096):
    """Read a bounded raw tail, not an inferred execution-stage classifier."""
    try:
        with Path(path).open('rb') as stream:
            stat = os.fstat(stream.fileno())
            offset = max(0, stat.st_size - max_bytes)
            stream.seek(offset)
            tail = stream.read(max_bytes)
        # Both newline logs and carriage-return progress bars delimit records.
        # If the bounded window contains neither, retain its raw partial tail.
        if offset:
            boundaries = [i for sep in (b'\n', b'\r')
                          if (i := tail.find(sep)) >= 0]
            if boundaries:
                tail = tail[min(boundaries) + 1:]
        return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'tail': tail.decode('utf-8', errors='replace')}
    except OSError as error:
        return {'error': type(error).__name__}


def collect(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup'), root_pids=()):
    root_pids = tuple(int(pid) for pid in root_pids)
    record = {'time_ns': time.time_ns(), 'pid': os.getpid(),
              'hostname': socket.gethostname(),
              'falcon_rank': os.environ.get('FALCON_RANK'),
              'cgroup_read_root': str(cgroup),
              'root_pids': list(root_pids)}
    # Record membership without assuming the mounted cgroup root is the task's
    # leaf. Counters can describe a wider scope; zero local events cannot rule
    # out ancestor-cgroup or host OOM.
    try:
        record['self_cgroup'] = (proc / 'self' / 'cgroup').read_text()
    except OSError:
        pass
    root_pids = set(root_pids)
    descendants = set(root_pids)
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
    statuses = {}
    for p in proc.glob('[0-9]*'):
        try:
            status = (p / 'status').read_text()
            fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
            statuses[int(p.name)] = fields
        except (OSError, ValueError):
            pass
    changed = True
    while changed:
        changed = False
        for pid, fields in statuses.items():
            if pid not in descendants and int(fields.get('PPid', '0').strip()) in descendants:
                descendants.add(pid)
                changed = True
    record['descendant_pids'] = sorted(descendants)
    for p in proc.glob('[0-9]*'):
        try:
            fields = statuses.get(int(p.name), {})
            rss = int(fields.get('VmRSS', '0 kB').split()[0])
            hwm = int(fields.get('VmHWM', '0 kB').split()[0])
            # Keep surviving compiler children whose allocations have receded
            # since their peak. Exited children still require external evidence.
            if max(rss, hwm) > 100000 or int(p.name) in descendants:
                processes.append({'pid': int(p.name), 'rss_kib': rss,
                                  'rss_hwm_kib': hwm,
                                  'ppid': int(fields.get('PPid', '0').strip()),
                                  'name': fields.get('Name', '').strip(),
                                  'tracked_descendant': int(p.name) in descendants})
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
