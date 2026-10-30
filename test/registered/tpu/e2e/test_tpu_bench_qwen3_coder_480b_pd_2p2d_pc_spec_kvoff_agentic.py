"""Qwen3-Coder-480B-A35B-Instruct-FP8 serving benchmark: 4-host P/D disaggregation: two single-host prefill + two single-host
decode engines (tp=8 each) over raiden behind a mini-LB, prefix caching +
EAGLE3 spec + KV offload.

Workload: multi-turn agentic (4k in + 1k out per round, 2 rounds = 8k of
input per regular session; long sessions ramp to the model's
LONG_TARGET_CONTEXT: ~122k gpt-oss / ~260k qwen) -- the 480B nightly's
workload shape. Prefix-cache hit rate is gated against what the round count
makes reachable.

Sweep: concurrency {1,16,32} x long requests {1,2,4} (legs with more long
requests than slots are skipped). One server serves every leg; the client's
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
    labels=[
        "multihost",
        "qwen3-coder-480b",
        "perf",
        "pd",
        "prefix-cache",
        "spec",
        "kv-offload",
        "agentic",
    ],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/qwen3_coder_480b_pd_2p2d_pc_spec_kvoff.sh",
    "weights": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "timeout_per_file": 8100,
}

TestBenchQwen3Coder480bPd2p2dPcSpecKvoffAgentic = build_matrix_test(
    bench_model="qwen3_coder_480b",
    bench_recipe="pd_2p2d_pc_spec_kvoff",
    workload="agentic",
)

if __name__ == "__main__":
    unittest.main()
