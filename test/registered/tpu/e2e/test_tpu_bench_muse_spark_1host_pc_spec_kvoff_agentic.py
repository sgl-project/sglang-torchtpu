"""Muse Spark serving benchmark: 1 host tp=8/ep=1, prefix caching + DFlash spec + KV offload (raiden).

Workload: multi-turn agentic (4k in + 1k out per round, 2 rounds = 8k of
input per regular session; long sessions ramp to the model's
LONG_TARGET_CONTEXT, ~122k on Muse) -- the 480B nightly's workload shape.
Prefix-cache hit rate is gated against what the round count makes reachable.

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
        "muse-spark-open-816b-a42b",
        "perf",
        "prefix-cache",
        "spec",
        "dflash",
        "kv-offload",
        "agentic",
    ],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/muse_spark_1host_pc_spec_kvoff.sh",
    "weights": "Muse-Spark-Open-816B-A42B-NVFP4",
    "timeout_per_file": 8100,
}

TestBenchMuseSpark1hostPcSpecKvoffAgentic = build_matrix_test(
    bench_model="muse_spark",
    bench_recipe="1host_pc_spec_kvoff",
    workload="agentic",
)

if __name__ == "__main__":
    unittest.main()
