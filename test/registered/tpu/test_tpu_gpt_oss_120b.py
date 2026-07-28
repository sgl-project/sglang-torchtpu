"""gpt-oss-120b single-host e2e lm-eval (GPQA Diamond).

Single-host on purpose: gpt-oss-120b is ~117B mxfp4 (~235GB after the TPU path
dequantizes MoE weights to bf16), which fits one tpu7x host (8 devices x ~94G HBM)
at tp=8. Running it in the single-host nightly suite keeps it OFF the single
2x2x2 multi-host slice, so it runs in parallel with (not serialized behind) the
multi-host launch groups.

Weights load from the ReadOnlyMany Hyperdisk PVC gptoss120b-rox mounted at /weights
(the nightly single-host job passes weights_pvc; one-time hydration via
.github/tpu-ci/hydrate-gptoss120b.yaml + pvc-gptoss120b-rox.yaml). Set
MODEL=openai/gpt-oss-120b to run against the HF checkpoint instead.

SGLANG_TPU_FUSED_MOE_SC_GATHER=0: PR #188 defaulted fused-MoE token gather onto the
SparseCore kernels (validated on fp8/silu/topk=8 models); gpt-oss's topk=4 +
swigluoai geometry fails SC lowering (NotImplementedError in the pltpu.async_copy
indexer). Keep the pre-#188 gather path until the SC kernel supports this config.

Asserts real accuracy: one pass of lm-eval gpqa_diamond_cot_zeroshot (full
198-question diamond set, reasoning_effort low) against
test/lm_eval_configs/gpt-oss-120b-TPU.yaml -- ground truth is the model card's
low-effort GPQA Diamond (67.1) with the agreed 3-point absolute tolerance.
⚠️ Idavidrein/gpqa is a gated HF dataset: the pod needs HF_TOKEN (see the config
yaml's note).
"""

import os
import subprocess
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=3600, suite="nightly-tpu-4-v7", nightly=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
TPU_BENCH_DIR = REPO_ROOT / "benchmark" / "tpu"


class TestTpuGptOss120BLMEval(LMEvalMixin, CustomTestCase):
    model = "openai/gpt-oss-120b"
    model_config_name = "test/lm_eval_configs/gpt-oss-120b-TPU.yaml"
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
        # Harmony output: analysis channel -> reasoning_content, so lm_eval's
        # answer extraction only sees final-channel text.
        env.setdefault("EXTRA_ARGS", "--reasoning-parser gpt-oss")
        env.setdefault("MEM_FRACTION_STATIC", "0.60")
        env.setdefault("CONTEXT_LEN", "36864")
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


if __name__ == "__main__":
    unittest.main()
