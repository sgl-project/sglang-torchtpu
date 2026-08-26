"""End-to-end test for torch.compile bundles on TPU."""

import tempfile
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(est_time=180, suite="stage-b-test-tpu-4-v7")

MODEL = "Qwen/Qwen3-0.6B"


class TestTpuQwen306bSinglehostTorchCompileBundle(CustomTestCase):
    def test_save_then_load(self):
        common_args = [
            "--dtype",
            "bfloat16",
            "--device",
            "tpu",
            "--grammar-backend",
            "none",
            "--enable-torch-compile",
            "--tp-size",
            "1",
            "--page-size",
            "64",
            "--chunked-prefill-size",
            "512",
            "--context-length",
            "1024",
            "--max-running-requests",
            "4",
            "--torch-compile-max-bs",
            "4",
            "--tpu-compile-tokens",
            "64",
            "512",
            "--tpu-compile-bs",
            "1",
            "4",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "bundle"
            for flag in (
                "--unsafe-save-torch-compile-bundle",
                "--unsafe-load-torch-compile-bundle",
            ):
                process = popen_launch_server(
                    MODEL,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[*common_args, flag, str(bundle_dir)],
                    device="tpu",
                )
                try:
                    server_info = requests.get(
                        f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=30
                    ).json()
                    internal_state = server_info["internal_states"][0]
                    dynamo_graphs = internal_state["torch_compile_bundle_dynamo_graphs"]
                    aot_graphs = internal_state[
                        "torch_compile_bundle_aot_autograd_graphs"
                    ]
                    if flag == "--unsafe-save-torch-compile-bundle":
                        self.assertGreater(dynamo_graphs, 0)
                        self.assertGreater(aot_graphs, 0)
                    else:
                        self.assertEqual(dynamo_graphs, 0)
                        self.assertEqual(aot_graphs, 0)
                finally:
                    kill_process_tree(process.pid, wait_timeout=30)


if __name__ == "__main__":
    unittest.main()
