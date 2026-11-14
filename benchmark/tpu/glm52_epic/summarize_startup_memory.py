"""Summarize sampler JSONL or Falcon logs JSON without inferring native stage."""
import argparse
import datetime
import json
from pathlib import Path


def summarize(text):
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict) and isinstance(envelope.get('content'), str):
        text = envelope['content']
    groups = {}
    for line in text.splitlines():
        sample_kind = 'periodic'
        if line.startswith(('GLM52_HOST_MEMORY ', 'GLM52_PRE_CLEANUP_MEMORY ')):
            sample_kind = ('pre_cleanup' if line.startswith('GLM52_PRE_CLEANUP_MEMORY ')
                           else 'periodic')
            line = line.split(' ', 1)[1]
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or 'memory.current' not in row:
            continue
        key = (row.get('hostname'), row.get('falcon_rank'), row.get('pid'))
        stat = dict(item.split() for item in row.get('memory.stat', '').splitlines())
        tracking_available = 'descendant_pids' in row
        children = [p for p in row.get('processes', []) if p.get('tracked_descendant')]
        sample = {
            'sample_kind': sample_kind,
            'time_utc': datetime.datetime.fromtimestamp(row['time_ns'] / 1e9,
                        datetime.timezone.utc).isoformat(),
            'current_bytes': int(row['memory.current']),
            'lifetime_peak_bytes': int(row['memory.peak']) if 'memory.peak' in row else None,
            'anon_bytes': int(stat['anon']) if 'anon' in stat else None,
            'file_bytes': int(stat['file']) if 'file' in stat else None,
            'cgroup_read_root': row.get('cgroup_read_root'),
            'self_cgroup': row.get('self_cgroup'),
            'root_pids': row.get('root_pids'),
            'descendant_processes': children,
            'surviving_descendant_rss_kib': (
                sum(p['rss_kib'] for p in children) if tracking_available else None),
            # Max individual HWM; summing independent HWMs invents a peak.
            'max_individual_survivor_hwm_kib': max(
                (p['rss_hwm_kib'] for p in children), default=0) if tracking_available else None,
            'memory_events': row.get('memory.events'),
            'native_log': row.get('native_log'),
        }
        groups.setdefault(key, []).append(sample)
    return {'scope': 'sampled usage only; cgroup cache is not compiler RSS; summed process RSS can double-count shared mappings and is not physical usage; exited children/transient peaks may be missed',
            'hosts': [{'hostname': key[0], 'falcon_rank': key[1], 'observer_pid': key[2],
                       'samples': sorted(rows, key=lambda r: r['time_utc'])}
                      for key, rows in groups.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.input.read_text()), indent=2))


if __name__ == '__main__':
    main()
