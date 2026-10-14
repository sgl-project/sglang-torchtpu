"""Qwen3-Coder-480B-A35B-Instruct-FP8 serving benchmark: 1 host tp=8/ep=8, prefix caching + EAGLE3 spec + KV offload (raiden).

Workload: single-turn 8k-input / 1k-output serving; no long requests (the
sweep's long axis is 0 for this workload). Prefix-cache hit rate is recorded
but not gated (a single-turn session cannot re-see a prefix).

Sweep: concurrency {1,16,32}. One server serves every leg; the client's
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
        "qwen3-coder-480b",
        "perf",
        "prefix-cache",
        "spec",
        "kv-offload",
    ],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/qwen3_coder_480b_1host_pc_spec_kvoff.sh",
    "weights": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "timeout_per_file": 8100,
}

TestBenchQwen3Coder480b1hostPcSpecKvoff8k1k = build_matrix_test(
    bench_model="qwen3_coder_480b",
    bench_recipe="1host_pc_spec_kvoff",
    workload="8k1k",
)

if __name__ == "__main__":
    unittest.main()
