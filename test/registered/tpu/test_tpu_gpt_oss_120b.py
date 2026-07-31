"""gpt-oss-120b single-host e2e accuracy: GPQA Diamond, MMLU and AIME 2025.

Three gates against one server boot, all anchored on the model card's low-effort,
no-tools column (Table 3, arXiv:2508.10925): GPQA 0.671, MMLU 0.859, AIME25 0.504.

Both qualifiers are load-bearing. The card gives a different number per effort level
(GPQA 0.671 / 0.731 / 0.801) and the with-tools column is higher again (AIME25 0.729
vs 0.504) -- this server executes no tools.

GPQA uses OpenAI's own gpt_oss.evals because 0.671 is only reproducible with their
permuted-choice prompt and grader; MMLU and AIME use run_eval, itself adapted from
OpenAI's simple-evals.

All three gate at atol 0.03, capped by the tolerance contract. Headroom differs by
sample count: MMLU ~3.7 sigma, GPQA ~1.8, AIME only ~0.9 -- 30 questions at p=0.5 is
the noisiest case there is, so AIME trips ~18% of runs on sampling noise alone.

Notes:
- Weights: shared ROX PVC model-weights-rox at /weights (pvc-model-weights-rox.yaml.tmpl).
- gpt-oss is runtime-installed --no-deps so its pins never touch the pinned torch.
- CONTEXT_LEN=16384: longest GPQA prompt (~2.9k tokens) + 8k completion budget.
- SGLANG_TPU_FUSED_MOE_SC_GATHER=0: SC gather (PR #188) cannot lower gpt-oss's
  topk=4 + swigluoai fused-MoE; keep the pre-#188 path until the kernel supports it.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import dump_metric
from sglang.test.tpu import RecipeLauncher, TpuModelLaunchSettings

# est_time drives the JobSet deadline (render_jobset.py: max(sum(est_time)*1.5, 360s)),
# so an under-estimate kills the run. The workload is ~3000 generations on one boot
# (GPQA 4x198, MMLU 2000, AIME 8x30), much of it low-effort reasoning up to 8k tokens.
# weekly-update-est-time.yml refreshes this from the p90 of real runs.
register_tpu_ci(est_time=10800, suite="nightly-tpu-4-v7", nightly=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
TPU_BENCH_DIR = REPO_ROOT / "benchmark" / "tpu"
LAUNCH_SCRIPT = TPU_BENCH_DIR / "launch_server.sh"

MODEL_PATH = "/weights/gpt-oss-120b"
SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
REASONING_EFFORT = "low"

# --- model card ground truths (Table 3; low effort, no tools) ---
GPQA_GROUND_TRUTH = 0.671
MMLU_GROUND_TRUTH = 0.859
AIME25_GROUND_TRUTH = 0.504

# 198 questions -> single-pass sigma 3.3pts, so a 3pt gate on one pass is under
# 1 sigma. 4 repeats put the mean's SE at ~1.7pts, making the floor ~1.8 sigma.
# The card uses 8 passes; 4 halves CI time. Do NOT go below 4.
GPQA_ATOL = 0.03
GPQA_N_REPEATS = 4

# 2000 questions -> sigma ~0.8pts, so 3pts is ~3.7 sigma. The headroom also absorbs
# prompt-formatting differences between simple-evals and OpenAI's internal harness.
MMLU_ATOL = 0.03
MMLU_NUM_EXAMPLES = 2000

# The noisiest gate here: 30 questions (AIME2025-I + II) at p=0.504, which is the
# worst case for binomial noise since p(1-p) peaks at 0.5. Single-pass sigma is
# 9.1pts; at 8 repeats the mean's SE is 3.2pts, putting the 3pt floor 0.9 sigma below
# the reference, so it trips ~18% of runs on sampling noise alone. Reaching 2 sigma
# takes 37 repeats and 3 sigma takes 83. atol is capped at 3pts by the tolerance
# contract, so repeats are the only lever -- a red run here warrants a re-run before
# investigation.
AIME25_ATOL = 0.03
AIME25_REPEAT = 8

# The recipe derives every server flag from this environment, so no CLI flags appear
# in this file.
LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    "TP_SIZE": "8",
    "EP_SIZE": "8",
    # Empty = defeat launch_server.sh's fp8 defaults; mxfp4 is auto-detected
    # from the checkpoint and the KV cache stays bf16 (30B-recipe pattern).
    "QUANTIZATION": "",
    "KV_CACHE_DTYPE": "",
    "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
    # Harmony: analysis channel -> reasoning_content, so the harness grades
    # final-channel text only. served-model-name matches the harness client.
    "EXTRA_ARGS": (
        f"--reasoning-parser gpt-oss --served-model-name {SERVED_MODEL_NAME}"
    ),
    "MEM_FRACTION_STATIC": "0.60",
    # Longest GPQA prompt (~2.9k tokens) + 8192 completion must fit.
    "CONTEXT_LEN": "16384",
    # This workload only prefills <=4096-token chunks and decodes at bs<=32;
    # one bucket each keeps bring-up ~10 min (validated 2026-07-28).
    "TPU_COMPILE_TOKENS": "4096",
    "TPU_COMPILE_BS": "32",
    "MAX_RUNNING": "32",
    "TORCH_COMPILE_MAX_BS": "32",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "PYTHON": "python3",
}


def _gpqa_via_openai_harness(base_url: str) -> dict:
    """Score GPQA Diamond with OpenAI's own gpt_oss.evals harness."""
    # Pure-python eval code; --no-deps so its pins never touch the pinned
    # TPU torch stack. openai/pandas/blobfile are image deps (test extra).
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "gpt-oss"],
        check=True,
    )
    # The openai client insists on a key even for a local base_url.
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler
    from gpt_oss.evals.gpqa_eval import GPQAEval

    sampler = ChatCompletionsSampler(
        model=SERVED_MODEL_NAME,
        reasoning_model=True,
        reasoning_effort=REASONING_EFFORT,
        temperature=1.0,
        base_url=base_url + "/v1",
        # Their CLI uses 131072; capped to fit CONTEXT_LEN. Proven
        # non-truncating: low-effort generations complete well under 8k.
        max_tokens=8192,
    )
    result = GPQAEval(n_repeats=GPQA_N_REPEATS, n_threads=32)(sampler)
    print(
        f"gpqa_diamond (openai harness, n_repeats={GPQA_N_REPEATS}): "
        f"score={result.score:.4f} metrics={result.metrics}"
    )
    # Built-in backends dump their own metrics; a runner must do it itself to reach
    # the trend dashboards.
    dump_metric(
        "gpqa_diamond_openai_harness",
        result.score,
        labels={"model": "openai/gpt-oss-120b", "eval": "gpt_oss.evals"},
    )
    return {"score": result.score}


def tpu_launch_settings() -> TpuModelLaunchSettings:
    return TpuModelLaunchSettings(
        MODEL_PATH,
        launcher=RecipeLauncher(
            str(LAUNCH_SCRIPT),
            env=LAUNCH_ENV,
            cwd=str(TPU_BENCH_DIR),
            health_timeout=HEALTH_TIMEOUT,
            tag="gpqa-e2e",
        ),
        tp_size=8,
        variant="TP8+EP8",
    )


def accuracy_gates() -> list:
    return [
        AccuracyTestParams(
            dataset="gpqa",
            label="gpqa_diamond_low",
            runner=_gpqa_via_openai_harness,
            baseline_accuracy=GPQA_GROUND_TRUTH,
            atol=GPQA_ATOL,
        ),
        AccuracyTestParams(
            dataset="mmlu",
            label="mmlu_low",
            baseline_accuracy=MMLU_GROUND_TRUTH,
            atol=MMLU_ATOL,
            num_examples=MMLU_NUM_EXAMPLES,
            num_threads=32,
            extra_eval_args={"reasoning_effort": REASONING_EFFORT},
            max_tokens=8192,
        ),
        AccuracyTestParams(
            dataset="aime25",
            label="aime25_low",
            baseline_accuracy=AIME25_GROUND_TRUTH,
            atol=AIME25_ATOL,
            num_threads=32,
            repeat=AIME25_REPEAT,
            extra_eval_args={"reasoning_effort": REASONING_EFFORT},
            max_tokens=8192,
        ),
    ]


class TestTpuGptOss120B(unittest.TestCase):
    """gpt-oss-120b accuracy gates, all sharing one TPU server boot."""

    def test_gpt_oss_120b_accuracy(self):
        run_combined_tests(
            models=[tpu_launch_settings()],
            test_name="gpt-oss-120b",
            base_url=BASE_URL,
            accuracy_params=accuracy_gates(),
        )


if __name__ == "__main__":
    unittest.main()
