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
        if line.startswith('GLM52_HOST_MEMORY '):
            line = line.split(' ', 1)[1]
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or 'memory.current' not in row:
            continue
        key = (row.get('hostname'), row.get('falcon_rank'), row.get('pid'))
        stat = dict(item.split() for item in row.get('memory.stat', '').splitlines())
        children = [p for p in row.get('processes', []) if p.get('tracked_descendant')]
        sample = {
            'time_utc': datetime.datetime.fromtimestamp(row['time_ns'] / 1e9,
                        datetime.timezone.utc).isoformat(),
            'current_bytes': int(row['memory.current']),
            'lifetime_peak_bytes': int(row.get('memory.peak', 0)),
            'anon_bytes': int(stat.get('anon', 0)),
            'file_bytes': int(stat.get('file', 0)),
            'surviving_descendant_rss_kib': sum(p['rss_kib'] for p in children),
            # Max individual HWM; summing independent HWMs invents a peak.
            'max_individual_survivor_hwm_kib': max(
                (p['rss_hwm_kib'] for p in children), default=0),
            'memory_events': row.get('memory.events'),
            'native_log': row.get('native_log'),
        }
        groups.setdefault(key, []).append(sample)
    return {'scope': 'sampled usage only; cgroup cache is not compiler RSS; exited children/transient peaks may be missed',
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
