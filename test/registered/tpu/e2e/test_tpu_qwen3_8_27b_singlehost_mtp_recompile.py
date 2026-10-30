"""Qwen3.8-27B MTP PR gate: the experimental spec worker serves with no XLA compile.

Requests of staggered lengths drain the batch through every live size, so a
program keyed on live rows instead of the batch bucket compiles mid-serving and
the recompilation checker logs it.
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=900,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3.8-27b", "spec", "compile"],
)

MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3.8-27B")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# The recipe's max_running_requests: the drain walks every live size below it.
CONCURRENCY = 16
SERVING_EVENT = "at serving time"


def _generate(index: int) -> dict:
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": f"Request {index}: count upward from {index * 7} in words.",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 16 + 24 * index,
                "ignore_eos": True,
            },
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()["meta_info"]


class TestQwen38MtpServesWithoutRecompile(TPULauncherTestCase):
    config_module = "recipe/configs/qwen3_8_27b/mtp_gate.py"
    base_url = BASE_URL
    tag = "qwen3-8-27b-mtp-recompile-e2e"
    log_path = str(Path(tempfile.gettempdir()) / "qwen3_8_27b_mtp_recompile.log")

    @classmethod
    def get_runtime_envs(cls):
        # CI-only FutureMap debug asserts scatter into live rows eagerly, one
        # program per live bs; this server measures production serving.
        return {
            **super().get_runtime_envs(),
            "MODEL": MODEL_PATH,
            "SGLANG_IS_IN_CI": None,
        }

    def test_draining_batch_compiles_nothing(self):
        log = Path(self.log_path)
        served_from = log.stat().st_size
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            metas = list(pool.map(_generate, range(CONCURRENCY)))

        verify_count = sum(meta.get("spec_verify_ct", 0) for meta in metas)
        completion_tokens = sum(meta["completion_tokens"] for meta in metas)
        self.assertGreater(verify_count, 0, "speculative verification never ran")
        # Every verify commits at least one token, so this is accept length > 1.
        self.assertGreater(completion_tokens, verify_count, "no draft was accepted")

        with log.open(errors="replace") as f:
            f.seek(served_from)
            events = [line.strip() for line in f if SERVING_EVENT in line]
        self.assertEqual(events, [], "XLA programs compiled or loaded while serving")


if __name__ == "__main__":
    import unittest

    unittest.main()
