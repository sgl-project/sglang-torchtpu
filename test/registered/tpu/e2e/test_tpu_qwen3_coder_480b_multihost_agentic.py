"""Qwen3-Coder-480B-A35B-Instruct-FP8 multi-turn agentic serving benchmark
(live-server mode), the colocated half of the team's standard 480B contract.

Its MULTIHOST_LAUNCH forms its own launch group: recipe/qwen3_coder_480b_agentic_mh.sh
on 2 hosts, i.e. tp=16 / ep=16 / dp=2 -> attention_tp=8, prefix caching + HiCache
KV offload on, 1M server context. The CI runs this file on rank-0 against the
server the JobSet brought up; it skips unless a live server is provided. DP size
is config.sh's COLOCATED_DP and is overridable, so nothing here hardcodes it --
the prefix-cache gate reads dp_size back off the run.

The gates and their rationale live in bench_matrix_base.build_bench_test (this
file predates the benchmark matrix and is its blueprint): every leg finished,
prefix caching hit what the round count makes reachable at the measured DP,
zero retract/abort, throughput recorded (never gated) and dumped for the
dashboard. The sweep harness stays this recipe's own run_nightly.sh; only the
test machinery is shared. Metric names are bare leg tags (no namespace) --
this file's dashboard trend predates the matrix and must not split.

The weights are already on the shared model-weights-filestore PVC, so no hydration
step is needed.
"""

import os
import unittest

from bench_matrix_base import build_bench_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=1800,
    suite="nightly-tpu-multihost",
    nightly=True,
    labels=["multihost", "qwen3-coder-480b", "agentic"],
)

# Own launch group: the 480B agentic recipe + the shared ROX weights volume.
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/qwen3_coder_480b_agentic_mh.sh",
    "weights_pvc": "model-weights",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
RECIPE_DIR = os.path.join(
    REPO_ROOT, "benchmark", "tpu", "recipe", "qwen3_coder_480b", "nightly"
)

TestTpuQwen3Coder480bMultihostAgentic = build_bench_test(
    class_name="TestTpuQwen3Coder480bMultihostAgentic",
    label="480b-agentic",
    # Canonical HF id, not the mount path the recipe serves from -- remounting
    # the weights elsewhere must not split the trend in two.
    model_label="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    # Stable series join key + its cosmetic display half. `id` is never
    # renamed (forced renames go through .github/tpu-ci/dashboard_aliases.json);
    # the display name is free.
    series_id="perf-qwen3-coder-480b-agentic",
    display_name="Qwen3-Coder-480B agentic",
    sweep_script=os.path.join(RECIPE_DIR, "run_nightly.sh"),
    sweep_env={
        # MODE=external means "sweep this endpoint, launch nothing" -- the
        # JobSet owns the server. The layout is the colocated one (tp=16
        # dp=2), so the results are labelled as such; P/D is a different
        # JobSet shape and is disabled here.
        "MODE": "external",
        "EXTERNAL_MODE_LABEL": "colocated",
        # The JobSet already warmed the server by serving the health probe; a
        # second compile-bucket pass would eat the file's whole budget.
        "SKIP_WARMUP": "1",
    },
    run_root_default="/tmp/qwen3-coder-480b-agentic",
    profile=os.environ.get("PROFILE", "ci"),
    # Leave headroom under the JobSet's `timeout 3600` per file so the summary
    # still gets written and printed when the sweep itself runs long.
    # HEALTH_TIMEOUT is spent before this and comes out of the same 3600,
    # hence 600 + 2700 + 300 spare. It must also clear the `ci` profile's own
    # worst case -- 4 legs x MAX_LEG_SECONDS=420 = 1680s -- or the sweep is
    # killed mid-run and the whole file fails on a budget, not on a result.
    sweep_timeout_default=2700,
    gate_prefix_hits=True,
    metric_namespace=None,
)

if __name__ == "__main__":
    unittest.main()
