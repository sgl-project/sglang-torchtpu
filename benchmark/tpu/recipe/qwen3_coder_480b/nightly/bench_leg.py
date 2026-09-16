"""Run one leg of the Qwen3-Coder agentic nightly."""

import os

from benchmark.tpu.recipe.bench_matrix.bench_leg import run_leg, workload_settings
from benchmark.tpu.recipe.bench_matrix.serve_group import child_processes
from benchmark.tpu.recipe.configs.qwen3_coder_480b.nightly.config import (
    BENCHMARK,
    CONFIG,
)

if __name__ == "__main__":
    env = dict(os.environ)
    with child_processes():
        run_leg(
            CONFIG, workload_settings(BENCHMARK, env, nightly=True), env, nightly=True
        )
