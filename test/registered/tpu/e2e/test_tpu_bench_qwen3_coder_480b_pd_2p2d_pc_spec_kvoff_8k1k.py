"""Qwen3-Coder-480B-A35B-Instruct-FP8 serving benchmark: 4-host P/D disaggregation: two single-host prefill + two single-host
decode engines (tp=8 each) over raiden behind a mini-LB, prefix caching +
EAGLE3 spec + KV offload.

Workload: single-turn 8k-input / 1k-output serving; no long requests (the
sweep's long axis is 0 for this workload). Prefix-cache hit rate is recorded
but not gated (a single-turn session cannot re-see a prefix).

Sweep: concurrency {1,16,32}. One server serves every leg; the client's
concurrency semaphore is what varies. Pods 0-1 serve prefill, pods 2-3 decode, each a single-host engine
(the raiden-validated P/D topology); pod 0 fronts all four with the mini-LB
on :30000 (the benched endpoint).

The two workload files of a (model, recipe) pair share one launch group (same
launch_script + weights), so the JobSet brings the server up once and rank 0
runs both files against it. All gates live in bench_matrix_base.py.
"""

import unittest

from bench_matrix_base import build_matrix_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=7200,
    suite="external-nightly-2x2x4",
    nightly=True,
    disabled="raiden P/D terminalization deadlock: one failed KV transfer parks its request forever (the decode withholds KVPoll.Failed while the room still counts active IO), so the leg hangs at the tail and the mini-LB never answers; re-enable once the deadlock is fixed",
    labels=[
        "multihost",
        "qwen3-coder-480b",
        "perf",
        "pd",
        "prefix-cache",
        "spec",
        "kv-offload",
    ],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/qwen3_coder_480b_pd_2p2d_pc_spec_kvoff.sh",
    "weights": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "timeout_per_file": 8100,
}

TestBenchQwen3Coder480bPd2p2dPcSpecKvoff8k1k = build_matrix_test(
    bench_model="qwen3_coder_480b",
    bench_recipe="pd_2p2d_pc_spec_kvoff",
    workload="8k1k",
)

if __name__ == "__main__":
    unittest.main()
