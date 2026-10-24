# SPDX-License-Identifier: Apache-2.0
"""Anchor-prompt check against a running standalone server (stdlib only).

  python3 client_check.py [port] [max_tokens] [prefill: rows|loop]
Prints the tokens sha (as bench/anchor.py: sha256 of int64 ids, 16 hex) of the generated ids of
prompt [1008, 18985, 387] (greedy, EOS ignored) and the wall time per token."""

import hashlib
import json
import sys
import time
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 30200
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
mode = sys.argv[3] if len(sys.argv) > 3 else "rows"
body = dict(
    input_ids=[1008, 18985, 387],
    sampling_params=dict(max_new_tokens=n, ignore_eos=True),
    stream=True,
    k3_prefill=mode,
    return_ids=True,
    k3_steps=int(sys.argv[4]) if len(sys.argv) > 4 else 0,
)
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/generate",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
t0 = time.time()
first, last, ids = None, None, None
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        now = time.time()
        first = first or now
        last = now
        ids = d.get("output_ids", ids)
        n_out = d["meta_info"]["completion_tokens"]
import numpy as np  # noqa: E402

sha = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
print(
    json.dumps(
        dict(
            mode=mode,
            tokens=n_out,
            sha=sha,
            first_ids=ids[:12],
            ttft_s=first - t0,
            tpot_ms=1e3 * (last - first) / max(1, n_out - 1),
        )
    )
)
