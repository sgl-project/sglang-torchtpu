# SPDX-License-Identifier: Apache-2.0
"""HTTP smoke test of a running standalone server (server/app.py; stdlib only; spec or plain): max_tokens,
stop strings, EOS, stream vs non-stream, repeatability; with ``--spec`` also forced acceptance (k3_force_acc)
and verify without the draft (k3_draft=zero). The server must run with ``--bench-hooks`` (EOS is exercised
through the k3_stop_ids hook). Prints PASS / FAIL lines and a final ``SMOKE {"ok": ...}`` line.
  python3 test/manual/k3_megakernel/spec_smoke.py [port] [--spec]"""

import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30210
P = [1008, 18985, 387, 4001, 512, 99, 77, 1234, 4321]


def post(path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not body.get("stream"):
            return json.loads(r.read())
        out = []
        for line in r:
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                out.append(json.loads(line[6:]))
        return out


def gen_ids(n, **kw):
    ev = post(
        "/generate",
        dict(
            input_ids=P,
            sampling_params=dict(max_new_tokens=n, ignore_eos=True),
            stream=True,
            return_ids=True,
            **kw,
        ),
    )
    return ev[-1]["output_ids"], ev[-1]["meta_info"]["finish_reason"]


def comp(n, **kw):
    r = post("/v1/completions", dict(prompt=P, max_tokens=n, ignore_eos=True, **kw))
    c = r["choices"][0]
    return c["text"], c["finish_reason"], r["usage"]["completion_tokens"]


res, ok = {}, True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    res[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


A, ra = gen_ids(60)
A2, _ = gen_ids(60)
check("generate_len60", len(A) == 60 and ra["type"] == "length", (len(A), ra))
check("repeatable", A == A2)
for n in (1, 7, 13, 33):
    ids, r = gen_ids(n)
    check(f"max_tokens_{n}", ids == A[:n] and r["type"] == "length", (len(ids), r))
T, rT, nT = comp(60)
check("completions_len60", nT == 60 and rT == "length", (nT, rT))
Ts = "".join(
    e["choices"][0]["text"]
    for e in post(
        "/v1/completions", dict(prompt=P, max_tokens=60, ignore_eos=True, stream=True)
    )
)
check("stream_eq_nonstream", Ts == T)
mid = len(T) // 2
s = T[mid : mid + 5]
t, r, n = comp(60, stop=[s])
check(
    "stop_string",
    t == T[: T.find(s)] and r == "stop",
    (repr(s), len(t), T.find(s), r, n),
)
eos = A[20]
ids, r = gen_ids(60, k3_stop_ids=[eos])
k = A.index(eos)
check("eos_token", ids == A[:k] and r["type"] == "stop", (eos, k, len(ids), r))
if "--spec" in sys.argv:  # forced-acceptance hook (spec servers only)
    ids, r = gen_ids(60, k3_force_acc=6)
    check("force_acc6", len(ids) == 60 and r["type"] == "length", (len(ids), r))
    ids, r = gen_ids(60, k3_force_acc=6, k3_draft="zero")
    check("force_zero_drafts", len(ids) == 60 and r["type"] == "length", (len(ids), r))
    A3, _ = gen_ids(60)
    check("after_force_repeatable", A3 == A)
print("SMOKE " + json.dumps(dict(ok=ok, **res)), flush=True)
sys.exit(0 if ok else 1)
