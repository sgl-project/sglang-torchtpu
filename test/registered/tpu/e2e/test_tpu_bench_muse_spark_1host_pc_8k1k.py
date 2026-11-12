"""Muse Spark serving benchmark: 1 host tp=8/ep=1, NVFP4 weights, fp8 KV, prefix caching on.

Workload: single-turn 8k-input / 1k-output serving; no long requests (the
sweep's long axis is 0 for this workload). Prefix-cache hit rate is recorded
but not gated (a single-turn session cannot re-see a prefix).

Sweep: concurrency {1,16,32}. One server serves every leg; the client's
concurrency semaphore is what varies. Rank 0 of the 2-host slice serves single-host; the second host idles
(2x2x2 is the registered suite's smallest slice).

All gates live in bench_matrix_base.py.
"""

import unittest

from bench_matrix_base import build_matrix_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=5400,
    suite="external-nightly-2x2x2",
    nightly=True,
    labels=["multihost", "muse-spark-open-816b-a42b", "perf", "prefix-cache"],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/bench_matrix/muse_spark_1host_pc.sh",
    "weights": "Muse-Spark-Open-816B-A42B-NVFP4",
    "timeout_per_file": 8100,
}

TestBenchMuseSpark1hostPc8k1k = build_matrix_test(
    bench_model="muse_spark",
    bench_recipe="1host_pc",
    workload="8k1k",
)

if __name__ == "__main__":
    unittest.main()
