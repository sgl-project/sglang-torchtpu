"""gpt-oss-120b single-host e2e accuracy: GPQA Diamond via OpenAI's own eval harness.

Self-launches a tp=8 server, then runs gpt_oss.evals with the model card's
methodology (permuted choices, temperature 1.0, their prompt + grader; n_repeats=4
vs the card's 8, halved for CI time) and asserts the mean score is within 3 points
of the card's 0.671 (floor 0.641). Single passes are too noisy on 198 questions for
a 3pt gate -- do NOT go below n_repeats=4.

Notes:
- Weights: ROX PVC gptoss120b-rox at /weights (hydrate-gptoss120b.yaml).
- gpt-oss is runtime-installed --no-deps so its pins never touch the pinned torch.
- CONTEXT_LEN=16384: longest GPQA prompt (~2.9k tokens) + 8k completion budget.
- SGLANG_TPU_FUSED_MOE_SC_GATHER=0: SC gather (PR #188) cannot lower gpt-oss's
  topk=4 + swigluoai fused-MoE; keep the pre-#188 path until the kernel supports it.
"""

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase, dump_metric

register_tpu_ci(est_time=3600, suite="nightly-tpu-4-v7", nightly=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
TPU_BENCH_DIR = REPO_ROOT / "benchmark" / "tpu"

GPQA_GROUND_TRUTH = 0.671  # model card: low effort, no tools, mean of 8 sampled passes
GPQA_ATOL = 0.03  # agreed tolerance -> floor 0.641
GPQA_N_REPEATS = 4  # card uses 8; 4 halves CI time and keeps the mean's SE ~1.7pts


class TestTpuGptOss120BGpqa(CustomTestCase):
    base_url = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
    health_timeout = int(os.environ.get("HEALTH_TIMEOUT", "1800"))
    process = None

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1":
            cls._wait_for_health()
            return

        env = os.environ.copy()
        env.setdefault("MODEL", "/weights/gpt-oss-120b")
        env.setdefault("TP_SIZE", "8")
        env.setdefault("EP_SIZE", "8")
        # Empty = defeat launch_server.sh's fp8 defaults; mxfp4 is auto-detected
        # from the checkpoint and the KV cache stays bf16 (30B-recipe pattern).
        env.setdefault("QUANTIZATION", "")
        env.setdefault("KV_CACHE_DTYPE", "")
        env.setdefault("SGLANG_TPU_FUSED_MOE_SC_GATHER", "0")
        # Harmony: analysis channel -> reasoning_content, so the harness grades
        # final-channel text only. served-model-name matches the harness client.
        env.setdefault(
            "EXTRA_ARGS", "--reasoning-parser gpt-oss --served-model-name gpt-oss-120b"
        )
        env.setdefault("MEM_FRACTION_STATIC", "0.60")
        # Longest GPQA prompt (~2.9k tokens) + 8192 completion must fit.
        env.setdefault("CONTEXT_LEN", "16384")
        # This workload only prefills <=4096-token chunks and decodes at bs<=32;
        # one bucket each keeps bring-up ~10 min (validated 2026-07-28).
        env.setdefault("TPU_COMPILE_TOKENS", "4096")
        env.setdefault("TPU_COMPILE_BS", "32")
        env.setdefault("MAX_RUNNING", "32")
        env.setdefault("TORCH_COMPILE_MAX_BS", "32")
        env.setdefault("ENABLE_PROFILER_ENDPOINT", "0")
        env.setdefault("PYTHON", "python3")

        cls.process = subprocess.Popen(
            ["bash", str(TPU_BENCH_DIR / "launch_server.sh")],
            cwd=TPU_BENCH_DIR,
            env=env,
        )
        cls._wait_for_health()

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid, wait_timeout=60)
            cls.process = None

    @classmethod
    def _wait_for_health(cls):
        start = time.perf_counter()
        while time.perf_counter() - start < cls.health_timeout:
            if cls.process is not None and cls.process.poll() is not None:
                raise RuntimeError(
                    f"TPU server exited with code {cls.process.returncode}"
                )
            try:
                response = requests.get(cls.base_url + "/health", timeout=10)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(
            f"{cls.base_url}/health did not return 200 within {cls.health_timeout}s"
        )

    def test_gpqa_openai_harness(self):
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
            model="gpt-oss-120b",
            reasoning_model=True,
            reasoning_effort="low",
            temperature=1.0,
            base_url=self.base_url + "/v1",
            # Their CLI uses 131072; capped to fit CONTEXT_LEN. Proven
            # non-truncating: low-effort generations complete well under 8k.
            max_tokens=8192,
        )
        result = GPQAEval(n_repeats=GPQA_N_REPEATS, n_threads=32)(sampler)

        floor = GPQA_GROUND_TRUTH - GPQA_ATOL
        print(
            f"gpqa_diamond (openai harness, n_repeats={GPQA_N_REPEATS}): "
            f"measured={result.score:.4f} | ground_truth={GPQA_GROUND_TRUTH} | "
            f"floor={floor:.3f} | metrics={result.metrics}"
        )
        dump_metric(
            "gpqa_diamond_openai_harness",
            result.score,
            labels={"model": "openai/gpt-oss-120b", "eval": "gpt_oss.evals"},
        )
        self.assertGreaterEqual(
            result.score,
            floor,
            f"GPQA {result.score:.4f} below floor {floor:.3f} "
            f"(card {GPQA_GROUND_TRUTH} - {GPQA_ATOL})",
        )


if __name__ == "__main__":
    unittest.main()
