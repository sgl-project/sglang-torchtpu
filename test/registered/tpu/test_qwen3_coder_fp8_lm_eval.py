"""Qwen3-Coder-30B-A3B-Instruct-FP8 single-host e2e accuracy via lm-eval.

Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml is the single source of truth for the
mmlu_llama / mmlu_pro ground truths and tolerances. The gate runs on lm-eval because
mmlu_pro has no simple-evals equivalent, and mmlu_llama's few-shot protocol is what
the reference values were measured with.
"""

import os
import unittest
from pathlib import Path

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.tpu import RecipeLauncher, TpuModelLaunchSettings, lm_eval_runner

register_tpu_ci(
    est_time=3600,
    suite="stage-b-test-tpu-4-v7",
    disabled="TPU PR runners are not provisioned yet",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TPU_BENCH_DIR = REPO_ROOT / "benchmark" / "tpu"
LAUNCH_SCRIPT = TPU_BENCH_DIR / "launch_server.sh"

MODEL_PATH = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
LM_EVAL_CONFIG = "test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    "TP_SIZE": "8",
    "EP_SIZE": "8",
    "PAGE_SIZE": "128",
    "CONTEXT_LEN": "8192",
    "CHUNKED_PREFILL_SIZE": "4096",
    "MAX_PREFILL_TOKENS": "4096",
    "MAX_TOTAL_TOKENS": "262144",
    "MEM_FRACTION_STATIC": "0.30",
    "MAX_RUNNING": "64",
    "TORCH_COMPILE_MAX_BS": "64",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "PYTHON": "python3",
}


class TestQwen3CoderFP8TPULMEval(unittest.TestCase):
    def test_qwen3_coder_fp8_accuracy(self):
        model = TpuModelLaunchSettings(
            MODEL_PATH,
            launcher=RecipeLauncher(
                str(LAUNCH_SCRIPT),
                env=LAUNCH_ENV,
                cwd=str(TPU_BENCH_DIR),
                health_timeout=HEALTH_TIMEOUT,
            ),
            tp_size=8,
            variant="TP8+EP8",
        )

        run_combined_tests(
            models=[model],
            test_name="Qwen3-Coder-30B-A3B-Instruct-FP8",
            base_url=BASE_URL,
            accuracy_params=AccuracyTestParams(
                dataset="mmlu_llama+mmlu_pro",
                label="lm_eval",
                runner=lm_eval_runner(LM_EVAL_CONFIG),
            ),
        )


if __name__ == "__main__":
    unittest.main()
