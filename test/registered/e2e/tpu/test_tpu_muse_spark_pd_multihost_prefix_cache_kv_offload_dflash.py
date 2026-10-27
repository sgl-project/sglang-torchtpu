"""Muse Spark two-host-per-engine DFlash P/D, prefix cache and bfloat16 KV offload."""

import os
import signal
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    MUSE_SPARK,
    RecipeLauncher,
    run_model_config,
    wait_for_health,
)
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase
from sglang.test.tpu.utils.pd_singlehost_base import peer_hostname

register_tpu_ci(
    est_time=10800,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "multihost",
        "muse-spark-open-816b-a42b",
        "pd",
        "spec",
        "dflash",
        "prefix-cache",
        "kv-offload",
    ],
)

TEST_NAME = "muse-spark-pd-multihost-dflash"


class TestMuseSparkPDDflash(TPULauncherTestCase):
    """Require actual speculative decoding and a complete host-cache reload.

    Every pod starts its slice-local engine shard. The prefill leader also runs
    the router and the gates, and the other pods wait for the JobSet teardown.
    """

    config_module = (
        "recipe/configs/muse_spark/pd_multihost_prefix_cache_kv_offload_dflash.py"
    )
    tag = "muse-spark-pd-dflash"
    health_timeout = 7200
    log_path = "/tmp/pd_multihost_server.log"
    base_url = "http://127.0.0.1:8000"

    @classmethod
    def setUpClass(cls):
        cls.role = os.environ["CI_PD_ROLE"]
        cls.rank = int(os.environ.get("TPU_WORKER_ID", "0"))
        cls.is_test_leader = cls.role == "prefill" and cls.rank == 0
        cls._launcher = cls._start_launcher(
            cls.config_module, cls.get_runtime_envs(), log_path=cls.log_path
        )
        if not cls.is_test_leader:
            return

        if str(cls.repo_root) not in sys.path:
            sys.path.insert(0, str(cls.repo_root))
        from benchmark.tpu.launch_server import router_command, wait_healthy

        prefill_host = os.environ["TPU_WORKER_HOSTNAMES"].split(",")[0]
        decode_host = peer_hostname(prefill_host, from_role="prefill", to_role="decode")
        prefill_url = f"http://{prefill_host}:30000"
        decode_url = f"http://{decode_host}:30000"

        def check_local_server():
            if cls._launcher.process.poll() is not None:
                raise RuntimeError(
                    "Prefill launcher exited before the P/D pair was ready"
                )

        for url in (prefill_url, decode_url):
            wait_healthy(url, cls.health_timeout, check_local_server)
        router = RecipeLauncher(
            command=router_command(
                [prefill_url], [decode_url], bootstrap_port=8998, port=8000
            ),
            cwd=str(cls.repo_root),
            start_new_session=True,
            tag=cls.tag + "-router",
        )
        cls.addClassCleanup(router.shutdown)
        router.start()
        wait_for_health(
            cls.base_url,
            cls.health_timeout,
            process=router.process,
            endpoint="/health_generate",
            tag=cls.tag,
        )

    def _wait_for_worker_shutdown(self):
        """Accept the JobSet's expected teardown, while surfacing engine crashes."""
        code = self._launcher.wait()
        if code in (-signal.SIGTERM, 128 + signal.SIGTERM):
            return
        if code in (-signal.SIGKILL, 128 + signal.SIGKILL):
            log = Path(self.log_path).read_text(errors="replace")
            if "Connection closed by peer" in log or "Connection reset by peer" in log:
                return
        self.fail(f"{self.role} rank {self.rank} exited before teardown: {code}")

    def test_prefix_cache_kv_offload_dflash(self):
        if not self.is_test_leader:
            self._wait_for_worker_shutdown()
            return
        run_model_config(
            {
                "model": MUSE_SPARK["served_model_name"],
                "test_name": TEST_NAME,
                "gates": [
                    {
                        "dataset": "gsm8k",
                        "baseline": 0.98,
                        "atol": 0.10,
                        "num_examples": 200,
                        "num_threads": 8,
                        "max_tokens": 4096,
                        "temperature": 0.0,
                        "reasoning_effort": "low",
                        # 1.0 means no accepted drafts and the gate fails only
                        # below it. 1.5 is arbitrary until a measured rate.
                        "spec_accept_length_threshold": 1.5,
                    },
                    {
                        "dataset": "kv_offload",
                        "label": "kv_offload_dflash",
                        "runner": kv_offload_runner(
                            test_name=TEST_NAME,
                            device_pool_tokens=8192,
                            page_size=128,
                            probe_tokens=2048,
                            hicache_ratio=3,
                            spec_decoding=True,
                        ),
                    },
                ],
            },
            base_url=self.base_url,
        )


if __name__ == "__main__":
    unittest.main()
