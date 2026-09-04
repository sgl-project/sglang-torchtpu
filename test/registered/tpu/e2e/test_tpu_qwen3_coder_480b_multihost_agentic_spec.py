"""Qwen3-Coder-480B-A35B-Instruct-FP8 multi-turn agentic EAGLE3 benchmark.

This is the speculative-decoding counterpart to
test_tpu_qwen3_coder_480b_multihost_agentic.py. Its MULTIHOST_LAUNCH forms a
separate two-host launch group using recipe/qwen3_coder_480b_agentic_spec_mh.sh:
tp=16 / ep=16 / dp=2 -> attention_tp=8, EAGLE3 speculative decoding, prefix
caching + HiCache KV offload, and a 1M server context.

The shared benchmark gates require every leg to finish, hit the reachable
prefix-cache floor, perform speculative verification, and report an accept
length above 1.0 from every DP scheduler. Throughput is recorded, never gated.
The distinct test file and series id keep these measurements separate from the
original non-spec agentic benchmark.
"""

import os
import unittest

from bench_matrix_base import build_bench_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=1800,
    suite="nightly-tpu-multihost",
    nightly=True,
    labels=["multihost", "qwen3-coder-480b", "agentic", "spec"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/qwen3_coder_480b_agentic_spec_mh.sh",
    "weights_pvc": "model-weights-rox",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
RECIPE_DIR = os.path.join(
    REPO_ROOT, "benchmark", "tpu", "recipe", "qwen3_coder_480b", "nightly"
)

TestTpuQwen3Coder480bMultihostAgenticSpec = build_bench_test(
    class_name="TestTpuQwen3Coder480bMultihostAgenticSpec",
    label="480b-agentic-spec",
    model_label="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    series_id="perf-qwen3-coder-480b-agentic-eagle3",
    display_name="Qwen3-Coder-480B agentic EAGLE3",
    sweep_script=os.path.join(RECIPE_DIR, "run_nightly.sh"),
    sweep_env={
        "MODE": "external",
        "EXTERNAL_MODE_LABEL": "colocated",
        "SKIP_WARMUP": "1",
    },
    run_root_default="/tmp/qwen3-coder-480b-agentic-spec",
    profile=os.environ.get("PROFILE", "ci"),
    sweep_timeout_default=2700,
    gate_prefix_hits=True,
    spec_contract={
        "algorithm": "EAGLE3",
        "draft_model_suffix": (
            "SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI"
        ),
        "num_steps": 3,
        "eagle_topk": 1,
        "num_draft_tokens": 4,
        # Vocabulary-sampled inputs cannot use GSM8K's calibrated 1.90 floor.
        # >1.0 proves each scheduler accepted at least one draft token.
        "min_accept_length": float(os.environ.get("MIN_SPEC_ACCEPT_LENGTH", "1.0")),
    },
    metric_namespace=None,
)

if __name__ == "__main__":
    unittest.main()
