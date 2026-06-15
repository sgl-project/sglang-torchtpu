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

register_tpu_ci(
    est_time=3600,
    suite="stage-b-test-tpu-4-v7",
    disabled="TPU PR runners are not provisioned yet",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TPU_BENCH_DIR = REPO_ROOT / "benchmark" / "tpu"


class TestQwen3CoderFP8TPULMEval(LMEvalMixin, CustomTestCase):
    model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    model_config_name = "test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml"
    base_url = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
    health_timeout = int(os.environ.get("HEALTH_TIMEOUT", "1800"))
    process = None

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1":
            cls._wait_for_health()
            return

        env = os.environ.copy()
        env.setdefault("MODEL", cls.model)
        env.setdefault("TP_SIZE", "8")
        env.setdefault("EP_SIZE", "8")
        env.setdefault("PAGE_SIZE", "128")
        env.setdefault("CONTEXT_LEN", "8192")
        env.setdefault("CHUNKED_PREFILL_SIZE", "4096")
        env.setdefault("MAX_PREFILL_TOKENS", "4096")
        env.setdefault("MAX_TOTAL_TOKENS", "262144")
        env.setdefault("MEM_FRACTION_STATIC", "0.30")
        env.setdefault("MAX_RUNNING", "64")
        env.setdefault("TORCH_COMPILE_MAX_BS", "64")
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
            f"{cls.base_url}/health did not return 200 within " f"{cls.health_timeout}s"
        )


if __name__ == "__main__":
    unittest.main()
