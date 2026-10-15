"""Unit tests for the TPU P/D router argv builder."""

import importlib.util
import os
import unittest

try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.test_utils import CustomTestCase
    from sglang.test.tpu.pd_router import (
        DEFAULT_DECODE_POLICY,
        DEFAULT_PREFILL_POLICY,
        MINI_LB,
        SMG,
        build_router_cmd,
        prefill_kv_events_args,
        resolve_mode,
        router_metrics_url,
        router_ready_url,
    )
except ModuleNotFoundError:

    def register_tpu_ci(*_args, **_kwargs):
        return None

    CustomTestCase = unittest.TestCase
    _pd_router_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            *[".."] * 4,
            "python",
            "sglang",
            "test",
            "tpu",
            "pd_router.py",
        )
    )
    _spec = importlib.util.spec_from_file_location("pd_router", _pd_router_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DEFAULT_DECODE_POLICY = _mod.DEFAULT_DECODE_POLICY
    DEFAULT_PREFILL_POLICY = _mod.DEFAULT_PREFILL_POLICY
    MINI_LB = _mod.MINI_LB
    SMG = _mod.SMG
    build_router_cmd = _mod.build_router_cmd
    prefill_kv_events_args = _mod.prefill_kv_events_args
    resolve_mode = _mod.resolve_mode
    router_metrics_url = _mod.router_metrics_url
    router_ready_url = _mod.router_ready_url

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit")


class TestPdRouterArgv(CustomTestCase):
    def test_default_mode_is_smg_and_builds_sgl_router_argv(self):
        self.assertEqual(DEFAULT_PREFILL_POLICY, "cache_aware")
        self.assertEqual(DEFAULT_DECODE_POLICY, "power_of_two")
        cmd = build_router_cmd(
            python="python3",
            prefill_url="http://10.0.0.1:8080",
            decode_url="http://10.0.0.2:8081",
            bootstrap_port=8998,
            host="0.0.0.0",
            port=8000,
            model_id="gpt-oss-120b",
            tokenizer_path="/models/gpt-oss-120b",
            env={},
        )
        self.assertEqual(
            cmd,
            (
                "sgl-router",
                "--model-id",
                "gpt-oss-120b",
                "--tokenizer-path",
                "/models/gpt-oss-120b",
                "--worker-urls",
                "http://10.0.0.1:8080",
                "http://10.0.0.2:8081",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--policy",
                "cache_aware",
                "--decode-policy",
                "power_of_two",
            ),
        )

    def test_mini_lb_mode_preserves_legacy_argv(self):
        cmd = build_router_cmd(
            python="/opt/venv/bin/python3",
            prefill_url="http://10.0.0.1:8080",
            decode_url="http://10.0.0.2:8081",
            bootstrap_port=8998,
            host="0.0.0.0",
            port=8000,
            env={"SGLANG_TPU_PD_ROUTER": "mini_lb"},
        )
        self.assertEqual(
            cmd,
            (
                "/opt/venv/bin/python3",
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                "http://10.0.0.1:8080",
                "8998",
                "--decode",
                "http://10.0.0.2:8081",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ),
        )

    def test_mini_lb_omits_bootstrap_port_when_none(self):
        cmd = build_router_cmd(
            python="python3",
            prefill_url="http://127.0.0.1:8080",
            decode_url="http://127.0.0.1:8081",
            host="0.0.0.0",
            port=8000,
            request_timeout_secs=3600,
            env={"SGLANG_TPU_PD_ROUTER": "mini_lb"},
        )
        self.assertEqual(
            cmd,
            (
                "python3",
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                "http://127.0.0.1:8080",
                "--decode",
                "http://127.0.0.1:8081",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--request-timeout-secs",
                "3600",
            ),
        )

    def test_smg_multi_worker_and_policy_env_overrides(self):
        cmd = build_router_cmd(
            python="python3",
            prefills=[
                ("http://10.0.0.1:8080", 8998),
                ("http://10.0.0.2:8082", 8999),
            ],
            decodes=["http://10.0.0.3:8081"],
            host="0.0.0.0",
            port=8000,
            request_timeout_secs=1800,
            env={
                "SGLANG_TPU_PD_ROUTER": "sgl-router",
                "SGL_ROUTER_BIN": "/usr/local/bin/sgl-router",
                "SERVED_MODEL_NAME": "qwen3-30b",
                "GPT_OSS_MODEL": "/weights/qwen3",
                "TPU_PD_PREFILL_POLICY": "round_robin",
                "TPU_PD_DECODE_POLICY": "legacy_host_affinity",
                "TPU_PD_CACHE_THRESHOLD": "0.25",
                "TPU_PD_BALANCE_ABS": "1024",
                "TPU_PD_BALANCE_REL": "1.2",
            },
        )
        self.assertEqual(
            cmd,
            (
                "/usr/local/bin/sgl-router",
                "--model-id",
                "qwen3-30b",
                "--tokenizer-path",
                "/weights/qwen3",
                "--worker-urls",
                "http://10.0.0.1:8080",
                "http://10.0.0.2:8082",
                "http://10.0.0.3:8081",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--policy",
                "round_robin",
                "--decode-policy",
                "legacy_host_affinity",
                "--cache-affinity-min-match-ratio",
                "0.25",
                "--pressure-abs-threshold-tokens",
                "1024",
                "--pressure-rel-threshold",
                "1.2",
                "--request-timeout-secs",
                "1800",
            ),
        )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            build_router_cmd(
                python="python3",
                prefill_url="http://10.0.0.1:8080",
                decode_url="http://10.0.0.2:8081",
                host="0.0.0.0",
                port=8000,
                mode=SMG,
                prefill_policy="cache-aware",
                env={},
            )

    def test_router_ready_and_metrics_url(self):
        self.assertEqual(
            router_ready_url(host="127.0.0.1", port=8000, env={}),
            "http://127.0.0.1:8000/readyz",
        )
        self.assertEqual(
            router_ready_url(
                host="127.0.0.1",
                port=8000,
                env={"SGLANG_TPU_PD_ROUTER": "mini_lb"},
            ),
            "http://127.0.0.1:8000/health",
        )
        self.assertEqual(
            router_metrics_url(host="127.0.0.1", port=8000, env={}),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertIsNone(
            router_metrics_url(
                host="127.0.0.1",
                port=8000,
                env={"SGLANG_TPU_PD_ROUTER": "mini_lb"},
            )
        )

    def test_prefill_kv_events_args(self):
        self.assertEqual(
            prefill_kv_events_args(env={}),
            (
                "--kv-events-config",
                '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}',
            ),
        )
        self.assertEqual(
            prefill_kv_events_args(env={"SGLANG_TPU_PD_ROUTER": "mini_lb"}),
            (),
        )
        self.assertEqual(
            prefill_kv_events_args(
                port=5560,
                env={"SGLANG_TPU_PD_ROUTER": "smg"},
            ),
            (
                "--kv-events-config",
                '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5560"}',
            ),
        )
        self.assertEqual(
            prefill_kv_events_args(
                env={
                    "SGLANG_TPU_PD_ROUTER": "smg",
                    "TPU_PD_KV_EVENTS_ENABLED": "0",
                }
            ),
            (),
        )

    def test_resolve_mode_rejects_unknown_value(self):
        self.assertEqual(resolve_mode(env={}), SMG)
        self.assertEqual(resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "mini_lb"}), MINI_LB)
        self.assertEqual(resolve_mode("SMG"), SMG)
        self.assertEqual(resolve_mode("sgl-router"), SMG)
        with self.assertRaises(ValueError):
            resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "nginx"})

    def test_smg_resolves_tokenizer_directory_to_tokenizer_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tok_file = os.path.join(tmpdir, "tokenizer.json")
            with open(tok_file, "w") as f:
                f.write("{}")
            cmd = build_router_cmd(
                python="python3",
                prefill_url="http://10.0.0.1:8080",
                decode_url="http://10.0.0.2:8081",
                host="0.0.0.0",
                port=8000,
                tokenizer_path=tmpdir,
                env={"SGLANG_TPU_PD_ROUTER": "smg"},
            )
            idx = cmd.index("--tokenizer-path")
            self.assertEqual(cmd[idx + 1], tok_file)

    def test_smg_sticky_policy_sets_routing_key_header(self):
        cmd = build_router_cmd(
            python="python3",
            prefill_url="http://10.0.0.1:8080",
            decode_url="http://10.0.0.2:8081",
            host="0.0.0.0",
            port=8000,
            prefill_policy="sticky",
            env={"SGLANG_TPU_PD_ROUTER": "smg"},
        )
        idx = cmd.index("--routing-key-header")
        self.assertEqual(cmd[idx + 1], "x-smg-routing-key")


if __name__ == "__main__":
    unittest.main()
