"""GPT-OSS-120B long-context multi-turn P/D stress test (1 prefill + 1 decode).

Topology: 2-host P/D disaggregation -- one single-host prefill engine and one
single-host decode engine (tp=8 each) over raiden behind the mini-LB, prefix
caching on, bf16 KV, no speculative decoding and no KV offload, so the P->D
transfer of a growing context is the variable under test.

Workload: multi-turn agentic sessions. Long sessions add ~8k new prefill
tokens per turn and grow to ~120k (GPT-OSS-120B's 131072 window); regular
sessions (4k in + 1k out, 2 rounds) keep the other slots busy.

Sweep: concurrency {1,8,16} x long sessions {1,4} (legs with more long
sessions than slots are skipped). c1-long1 is the clean turn-by-turn probe:
one session, nothing else in flight.

Recorded per leg, beyond the matrix's usual numbers: TTFT/ITL by context
length and per turn of the long sessions (client side), and the servers'
histograms -- P->D KV transfer latency/size/speed, prefill queue time, and
per-stage latency on both engines (summary.md "Server-side breakdown").
Gates: every leg completes, no failed requests, no retraction/abort, and the
prefix-cache hit rate clears its reachable floor. All gates live in
bench_matrix_base.py.
"""

import unittest

from bench_matrix_base import build_matrix_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=1800,
    suite="external-nightly-2x2x2",
    nightly=True,
)

# A light cell (~14 min sweep): discovery runs it LAST on a 2x2x2 slice that
# another matrix group already fills, with the smaller tail budget
# (bench_matrix_base.TAIL_TIMEOUT_PER_FILE_BUDGET).
MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/gpt_oss_120b_pd_1p1d_pc.sh",
    "weights": "gpt-oss-120b",
    "timeout_per_file": 5400,
    "placement": "tail",
}

TestBenchGptOss120bPd1p1dPcAgentic = build_matrix_test(
    bench_model="gpt_oss_120b",
    bench_recipe="pd_1p1d_pc",
    workload="agentic",
)

if __name__ == "__main__":
    unittest.main()
