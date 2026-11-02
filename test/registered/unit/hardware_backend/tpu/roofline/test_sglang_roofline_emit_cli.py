# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for SGLang launch_server flag emission, Top-3 structural pruning, SVG report, and CLI."""

from __future__ import annotations

import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    from roofline import (
        CHIPS,
        SOLVE_SCHEMA,
        PoolMeshPlan,
        TrafficScenario,
        build_solve_payload,
        dtypes_from_hf_config,
        emit_launch_args,
        from_hf_config,
        pool_cli_args,
        render_html,
        solve_serving_plan,
        top_layouts,
    )
    from roofline.cli import main as cli_main
except ModuleNotFoundError:
    from sglang.srt.hardware_backend.tpu.roofline import (
        CHIPS,
        SOLVE_SCHEMA,
        PoolMeshPlan,
        TrafficScenario,
        build_solve_payload,
        dtypes_from_hf_config,
        emit_launch_args,
        from_hf_config,
        pool_cli_args,
        render_html,
        solve_serving_plan,
        top_layouts,
    )
    from sglang.srt.hardware_backend.tpu.roofline.cli import main as cli_main
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _qwen35_cfg() -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "quantization_config": {"quant_method": "fp8", "kv_cache_scheme": "fp8"},
    }


def _gpt_oss_120b_cfg() -> dict:
    return {
        "model_type": "gpt_oss",
        "torch_dtype": "bfloat16",
        "hidden_size": 2880,
        "num_hidden_layers": 36,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "intermediate_size": 2880,
        "vocab_size": 201088,
        "quantization_config": {"quant_method": "mxfp4"},
    }


class TestSglangRooflineEmitCli(unittest.TestCase):
    def test_sglang_emit_launch_server_flags_and_structural_top3(self) -> None:
        # 1. Attention DP inside MoE (dp > 1 and ep > 1): --tp-size {tp*dp} --attn-dp-size {dp} --ep-size {ep},
        #    never --enable-dp-attention; plus --moe-dp-size, --decode-context-parallel-size, HiCache pair, fp8_e4m3.
        moe_dp_pool = PoolMeshPlan(
            role="decode",
            instances=2,
            topology="2x2x2",
            chips_per_instance=8,
            tp=4,
            dp=4,
            ep=8,
            moe_dp=2,
            pp=1,
            pcp=1,
            kv_split=1,
            mesh_axes=(2, 2, 1, 1),
            hbm_used_gb=72.0,
            hbm_limit_gb=90.0,
            page_size=256,
            kv_offload_gb=128.0,
            max_running_requests=64,
            gpu_memory_utilization=0.90,
        )
        args_moe = pool_cli_args(moe_dp_pool, kv_dtype="fp8")
        self.assertNotIn("--enable-dp-attention", args_moe)
        self.assertEqual(args_moe[args_moe.index("--tp-size") + 1], "16")
        self.assertEqual(args_moe[args_moe.index("--attn-dp-size") + 1], "4")
        self.assertEqual(args_moe[args_moe.index("--ep-size") + 1], "8")
        self.assertEqual(args_moe[args_moe.index("--moe-dp-size") + 1], "2")
        self.assertIn("--enable-dp-lm-head", args_moe)
        self.assertEqual(
            args_moe[args_moe.index("--disaggregation-mode") + 1], "decode"
        )
        self.assertEqual(args_moe[args_moe.index("--kv-cache-dtype") + 1], "fp8_e4m3")
        self.assertIn("--enable-hierarchical-cache", args_moe)
        self.assertEqual(args_moe[args_moe.index("--hicache-size") + 1], "128")
        self.assertEqual(args_moe[args_moe.index("--max-running-requests") + 1], "64")
        self.assertIn("--nnodes", args_moe)
        self.assertEqual(args_moe[args_moe.index("--nnodes") + 1], "2")

        # 2. Whole-model DP (dp > 1 and ep == 1) + PP + PCP prefill flags.
        dense_pre_pool = PoolMeshPlan(
            role="prefill",
            instances=1,
            topology="2x2x1",
            chips_per_instance=4,
            tp=4,
            dp=1,
            ep=1,
            pp=2,
            pcp=4,
            kv_split=1,
            mesh_axes=(2, 2, 1, 1),
            hbm_used_gb=50.0,
            hbm_limit_gb=90.0,
            page_size=64,
        )
        args_pre = pool_cli_args(dense_pre_pool, kv_dtype="bfloat16")
        self.assertEqual(args_pre[args_pre.index("--pp-size") + 1], "2")
        self.assertEqual(args_pre[args_pre.index("--attn-cp-size") + 1], "4")
        self.assertIn("--enable-prefill-cp", args_pre)
        self.assertEqual(args_pre[args_pre.index("--cp-strategy") + 1], "zigzag")
        self.assertEqual(args_pre[args_pre.index("--moe-dp-size") + 1], "4")
        self.assertIn("--enable-torch-compile", args_pre)
        self.assertEqual(
            args_pre[args_pre.index("--disaggregation-mode") + 1], "prefill"
        )
        self.assertEqual(args_pre[args_pre.index("--page-size") + 1], "256")

        # 3. DCP pool + torch-compile.
        dcp_pool = PoolMeshPlan(
            role="decode",
            instances=1,
            topology="2x2x1",
            chips_per_instance=4,
            tp=4,
            dp=1,
            ep=1,
            pp=1,
            pcp=1,
            kv_split=2,
            mesh_axes=(2, 2, 1, 1),
            hbm_used_gb=60.0,
            hbm_limit_gb=90.0,
            page_size=256,
        )
        args_dcp = pool_cli_args(dcp_pool)
        self.assertIn("--enable-torch-compile", args_dcp)
        self.assertEqual(
            args_dcp[args_dcp.index("--decode-context-parallel-size") + 1], "2"
        )
        self.assertEqual(args_dcp[args_dcp.index("--dcp-comm-backend") + 1], "ag_rs")

        # 4. Speculative decoding pool.
        spec_pool = PoolMeshPlan(
            role="mixed",
            instances=1,
            topology="2x2x1",
            chips_per_instance=4,
            tp=4,
            dp=1,
            ep=1,
            pp=1,
            pcp=1,
            kv_split=1,
            mesh_axes=(2, 2, 1, 1),
            hbm_used_gb=60.0,
            hbm_limit_gb=90.0,
            page_size=256,
            spec_tokens=3,
        )
        args_spec = pool_cli_args(spec_pool)
        self.assertEqual(
            args_spec[args_spec.index("--speculative-algorithm") + 1], "NEXTN"
        )
        self.assertEqual(args_spec[args_spec.index("--speculative-num-steps") + 1], "3")
        self.assertEqual(
            args_spec[args_spec.index("--speculative-num-draft-tokens") + 1], "4"
        )

        dense_dp_pool = PoolMeshPlan(
            role="mixed",
            instances=1,
            topology="2x2x1",
            chips_per_instance=4,
            tp=2,
            dp=4,
            ep=1,
            pp=1,
            pcp=1,
            kv_split=1,
            mesh_axes=(2, 1, 1, 1),
            hbm_used_gb=40.0,
            hbm_limit_gb=90.0,
            page_size=256,
        )
        args_dp = pool_cli_args(dense_dp_pool)
        self.assertEqual(args_dp[args_dp.index("--tp-size") + 1], "2")
        self.assertEqual(args_dp[args_dp.index("--dp-size") + 1], "4")
        self.assertNotIn("--disaggregation-mode", args_dp)

        # 3. Structural Top-3 pruning produces 3 distinct roles and 3 distinct launch_server CLI command tuples.
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_dt = dtypes_from_hf_config(_gpt_oss_120b_cfg())
        scen = TrafficScenario(isl=4096, osl=512, concurrencies=(1, 8, 32, 64))
        sp = solve_serving_plan(
            oss_geom, oss_dt, CHIPS["tpu7x"], scen, mode="auto", max_chips=16
        )
        top3 = top_layouts(sp, k=3)
        self.assertEqual(len(top3), 3)
        self.assertEqual(
            [pt.frontier_role for pt in top3],
            ["max_throughput", "max_interactivity", "balanced_knee"],
        )
        self.assertEqual(top3[0].margin_vs_rank1_pct, 0.0)
        launched = emit_launch_args(sp, dtypes=oss_dt, model="openai/gpt-oss-120b")
        cmd_tuples = {tuple(entry["commands"]) for entry in launched["top_k"]}
        self.assertEqual(len(cmd_tuples), 3)
        for cmd_tuple in cmd_tuples:
            for cmd in cmd_tuple:
                self.assertTrue(
                    cmd.startswith(
                        "python3 -m sglang.launch_server --model-path openai/gpt-oss-120b "
                    )
                )

    def test_solve_v1_schema_svg_report_cli_and_import_isolation(self) -> None:
        q_cfg = _qwen35_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            html_path = Path(tmp) / "report.html"
            cfg_path.write_text(json.dumps(q_cfg), encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(
                    [
                        "solve",
                        "--config",
                        str(cfg_path),
                        "--from-bench-cmd",
                        (
                            "python3 -m sglang.bench_serving --model-path Qwen/Qwen3.5-397B-A17B-FP8 "
                            "--shared-prefix-len 6144 --random-prefix-len 2048 --random-output-len 512 "
                            "--max-concurrency 1,16 --enable-hierarchical-cache --hicache-size 64"
                        ),
                        "--max-chips",
                        "8",
                        "--html",
                        str(html_path),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["schema"], SOLVE_SCHEMA)
            self.assertEqual(payload["isl"], 8192)
            self.assertEqual(payload["scenario"]["kv_offload_gb"], 64.0)
            self.assertTrue(payload["hardware"].startswith("v7x-"))
            self.assertGreaterEqual(len(payload["top_k"]), 1)

            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("<svg", html_text)
            self.assertIn("<polyline", html_text)
            self.assertIn("sglang.srt.hardware_backend.tpu.roofline", html_text)
            direct_html = render_html(
                build_solve_payload(
                    solve_serving_plan(
                        from_hf_config(q_cfg),
                        dtypes_from_hf_config(q_cfg),
                        CHIPS["tpu7x"],
                        TrafficScenario(isl=2048, osl=256, concurrencies=(1, 8)),
                        mode="mixed",
                        max_chips=8,
                    ),
                    chip=CHIPS["tpu7x"],
                    dtypes=dtypes_from_hf_config(q_cfg),
                    model="Qwen/Qwen3.5-397B-A17B-FP8",
                )
            )
            self.assertIn("<circle", direct_html)
            self.assertIn("<rect", direct_html)

            # Verify emit, ceiling, plan, and explain CLI subcommands.
            for sub_argv in (
                ["emit", "--config", str(cfg_path), "--max-chips", "8", "--json"],
                ["ceiling", "--config", str(cfg_path), "--max-chips", "8", "--json"],
                [
                    "plan",
                    "--config",
                    str(cfg_path),
                    "--tp",
                    "8",
                    "--moe",
                    "ep",
                    "--context",
                    "2048",
                    "--json",
                ],
                [
                    "explain",
                    "--config",
                    str(cfg_path),
                    "--tp",
                    "8",
                    "--moe",
                    "ep",
                    "--context",
                    "2048",
                    "--json",
                ],
            ):
                out_buf = io.StringIO()
                with redirect_stdout(out_buf):
                    self.assertEqual(cli_main(sub_argv), 0)
                parsed = json.loads(out_buf.getvalue())
                self.assertTrue(bool(parsed))

        # Verify intra-package relative imports (zero top-level torch or sglang.srt imports in roofline/*.py).
        pkg_dir = (
            Path(__file__).resolve().parents[6]
            / "python/sglang/srt/hardware_backend/tpu/roofline"
        )
        py_files = sorted(pkg_dir.glob("*.py"))
        self.assertEqual(len(py_files), 13)
        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if type(node) is ast.Import:
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith(("torch", "sglang")))
                elif type(node) is ast.ImportFrom and node.level == 0 and node.module:
                    self.assertFalse(node.module.startswith(("torch", "sglang")))


if __name__ == "__main__":
    unittest.main()
