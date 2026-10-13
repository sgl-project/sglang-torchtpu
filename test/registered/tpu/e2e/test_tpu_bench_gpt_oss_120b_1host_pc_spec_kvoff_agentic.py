"""GPT-OSS-120B serving benchmark: 1 host tp=8/ep=8, prefix caching + EAGLE3 spec + KV offload (raiden).

Workload: multi-turn agentic (4k in + 1k out per round, 2 rounds = 8k of
input per regular session; long sessions ramp to the model's
LONG_TARGET_CONTEXT: ~122k gpt-oss / ~260k qwen) -- the 480B nightly's
workload shape. Prefix-cache hit rate is gated against what the round count
makes reachable.

Sweep: concurrency {1,16,32} x long requests {1,2,4} (legs with more long
requests than slots are skipped). One server serves every leg; the client's
concurrency semaphore is what varies. Rank 0 of the 2-host slice serves single-host; the second host idles
(2x2x2 is the registered suite's smallest slice).

The two workload files of a (model, recipe) pair share one launch group (same
launch_script + weights), so the JobSet brings the server up once and rank 0
runs both files against it. All gates live in bench_matrix_base.py.
"""

import unittest

from bench_matrix_base import build_matrix_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=5400,
    suite="external-nightly-2x2x2",
    nightly=True,
    labels=[
        "multihost",
        "gpt-oss-120b",
        "perf",
        "prefix-cache",
        "spec",
        "kv-offload",
        "agentic",
    ],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/gpt_oss_120b_1host_pc_spec_kvoff.sh",
    "weights": "gpt-oss-120b",
    "timeout_per_file": 8100,
}

TestBenchGptOss120b1hostPcSpecKvoffAgentic = build_matrix_test(
    bench_model="gpt_oss_120b",
    bench_recipe="1host_pc_spec_kvoff",
    workload="agentic",
)

if __name__ == "__main__":
    unittest.main()
