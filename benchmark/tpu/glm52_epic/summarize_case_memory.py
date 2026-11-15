"""Summarize passive samples; never infer OOM causality or unique RSS."""
import argparse
import json
from pathlib import Path


def summarize(path):
    maxima = {}
    processes = {}
    minimum_available = None
    samples = 0
    counters = []
    for line in Path(path).open():
        row = json.loads(line)
        samples += 1
        for field in ('memory.current', 'memory.peak'):
            value = row.get(field, '').strip()
            if value.isdigit():
                maxima[field] = max(maxima.get(field, 0), int(value))
        stat = dict(s.split() for s in row.get('memory.stat', '').splitlines())
        for field in ('anon', 'file', 'shmem'):
            if field in stat:
                key = 'memory.stat.' + field
                maxima[key] = max(maxima.get(key, 0), int(stat[field]))
        for entry in row.get('meminfo', '').splitlines():
            if entry.startswith('MemAvailable:'):
                available = int(entry.split()[1]) * 1024
                minimum_available = available if minimum_available is None else min(minimum_available, available)
        for p in row.get('processes', []):
            if not p.get('tracked_descendant'):
                continue
            # PID reuse cannot be ruled out by this sampler; cmdline separates
            # observed exec identities, not full process lifetimes.
            key = (p['pid'], p.get('cmdline'))
            old = processes.setdefault(key, dict(pid=p['pid'], cmdline=p.get('cmdline'),
                                                 max_sampled_rss_kib=0, max_observed_hwm_kib=0))
            old['max_sampled_rss_kib'] = max(old['max_sampled_rss_kib'], p['rss_kib'])
            old['max_observed_hwm_kib'] = max(old['max_observed_hwm_kib'], p['rss_hwm_kib'])
        counters.append({k: row[k] for k in ('memory.events', 'memory.events.local') if k in row})
    return dict(samples=samples, sampled_maxima_bytes=maxima,
                minimum_host_memavailable_bytes=minimum_available,
                tracked_processes=sorted(processes.values(), key=lambda x: x['max_observed_hwm_kib'], reverse=True),
                first_counters=counters[0] if counters else {},
                last_counters=counters[-1] if counters else {},
                limitations=['Not unique compiler peak or RSS sum.',
                             'memory.peak may predate the probe; stat fields peak at different times.',
                             'file may include shmem; these fields are not additive.',
                             'Sampling can miss short-lived children; PID lifetimes unavailable.',
                             'Zero counters do not exclude ancestor or host OOM.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.samples), indent=2))
