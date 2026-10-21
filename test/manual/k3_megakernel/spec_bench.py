# SPDX-License-Identifier: Apache-2.0
"""Speculative-serving numbers from a running standalone server (server/app.py; stdlib only).
  python3 spec_bench.py force <port> [acc=6] [tokens=2000] [requests=3] [draft=dspark|zero]
      forced-acceptance timing (garbage text; the server must run with --bench-hooks); draft=zero skips
      the draft (verify + loop only)
  python3 spec_bench.py stats <server log> [first_job]
      acceptance / ms per round of the jobs in a server log
``force``: PR #8 protocol, every round emits ``acc`` tokens; prints wall tpot and ms/round = tpot * acc.
``stats``: rounds-weighted accept length (tokens per round, bonus included) and ms/round from the
``job N: {timing}`` lines (jobs >= first_job, forced jobs skipped; e.g. skip warmup)."""

import json
import re
import sys
import time
import urllib.request


def force(port, acc=6, tokens=2000, requests=3, draft="dspark"):
    body = dict(
        input_ids=[1008, 18985, 387],
        sampling_params=dict(max_new_tokens=tokens, ignore_eos=True),
        stream=True,
        k3_force_acc=acc,
        k3_draft=draft,
    )
    for i in range(requests):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        t0, first, last, n = time.time(), None, None, 0
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                now = time.time()
                first, last = first or now, now
                n = d["meta_info"]["completion_tokens"]
        tpot = 1e3 * (last - first) / max(1, n - 1)
        print(
            "FORCE "
            + json.dumps(
                dict(
                    req=i,
                    draft=draft,
                    acc=acc,
                    tokens=n,
                    ttft_s=round(first - t0, 3),
                    tpot_ms=round(tpot, 4),
                    ms_per_round=round(tpot * acc, 3),
                    tok_per_s=round(1e3 / tpot, 1),
                )
            ),
            flush=True,
        )


def stats(log, first_job=0):
    rounds = acc = ms = toks = 0.0
    jobs = 0
    for line in open(log, errors="replace"):
        m = re.search(r"\] job (\d+): (\{.*\}) reason", line)
        if not m or int(m.group(1)) < first_job:
            continue
        t = json.loads(m.group(2))
        if "rounds_run" not in t or t.get("forced") or t.get("zero_drafts"):
            continue
        n = t["rounds_run"]
        jobs += 1
        rounds += n
        acc += t["accept_length"] * n
        ms += t["ms_per_round"] * n
        toks += t["completion_tokens"]
    print(
        "STATS "
        + json.dumps(
            dict(
                jobs=jobs,
                rounds=int(rounds),
                accept_length=round(acc / max(rounds, 1), 3),
                ms_per_round=round(ms / max(rounds, 1), 3),
                completion_tokens=int(toks),
            )
        )
    )


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "force":
        force(int(a[0]), *[int(x) for x in a[1:4]], *a[4:5])
    else:
        stats(a[0], int(a[1]) if len(a) > 1 else 0)
