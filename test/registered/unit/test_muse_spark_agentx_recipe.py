"""The Muse Spark AgentX recipes render to the submitted launch (no TPU needed).

benchmark/tpu/recipe/configs/muse_spark/agentx_c<N>.py reproduce the server of the
InferenceX AgentX submission at each swept concurrency. The expected flags and
environment below are that submission's launch, rendered from its recipe script;
only where things live (model and draft paths, host, port) is left out. A change
here is a change to what the recipes reproduce.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.launch_server import load_recipe, plan_launch  # noqa: E402
from benchmark.tpu.recipe.configs.muse_spark import (  # noqa: E402
    agentx_summary,
    windowed_draft,
)

# Where things live, not how they are served.
LOCATION_FLAGS = ("--model-path", "--speculative-draft-model-path", "--host", "--port")
# Set by the caller's environment (base.py), or by the launcher for compile caches.
CALLER_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "SGLANG_PLATFORM")


def render(conc):
    plan = plan_launch(
        load_recipe(f"recipe/configs/muse_spark/agentx_c{conc}.py"), inherited={}
    )
    flags, flag = {}, None
    for arg in plan.command[plan.command.index("sglang.launch_server") + 1 :]:
        if arg.startswith("--"):
            flag = arg
            flags[flag] = []
        else:
            flags[flag].append(arg)
    for name in LOCATION_FLAGS:
        flags.pop(name)
    # The launcher's single-host mode adds ServerArgs' own default.
    assert flags.pop("--nnodes") == ["1"]
    env = {
        name: value
        for name, value in plan.env.items()
        if name not in CALLER_ENV and not name.startswith("TORCH_TPU_INTERNAL_TIER")
    }
    return flags, env


class TestAgentxRecipes(unittest.TestCase):
    def test_each_swept_point_renders_the_submitted_launch(self):
        for conc, (flags, env) in PER_CONC.items():
            with self.subTest(conc=conc):
                rendered_flags, rendered_env = render(conc)
                self.assertEqual(rendered_flags, {**COMMON_FLAGS, **flags})
                self.assertEqual(rendered_env, {**COMMON_ENV, **env})

    def test_windowed_draft_declares_the_sliding_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, out = Path(tmp, "snapshot"), Path(tmp, "out")
            snapshot.mkdir()
            config = {"num_hidden_layers": 2, "layer_types": ["sliding_attention"] * 2}
            (snapshot / "config.json").write_text(json.dumps(config))
            (snapshot / "model.safetensors").write_text("weights")
            windowed_draft.write_windowed_draft(snapshot, out)
            written = json.loads((out / "config.json").read_text())
            self.assertEqual(
                written,
                {**config, "is_hybrid_swa": True, "hybrid_layer_pattern": [1, 1]},
            )
            self.assertEqual(
                (out / "model.safetensors").resolve(),
                (snapshot / "model.safetensors").resolve(),
            )
            config["layer_types"][0] = "full_attention"
            with self.assertRaises(ValueError):
                windowed_draft.windowed_config(config)

    def test_summary_definitions(self):
        def record(start_s, e2e_s, ttft_s, prompt, completion, **extra):
            itl = (e2e_s - ttft_s) / (completion - 1)
            ms = lambda s: {"value": s * 1000, "unit": "ms"}  # noqa: E731
            return {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": int(start_s * 1e9),
                    "request_end_ns": int((start_s + e2e_s) * 1e9),
                },
                "metrics": {
                    "input_sequence_length": {"value": prompt},
                    "output_sequence_length": {"value": completion},
                    "request_latency": ms(e2e_s),
                    "time_to_first_token": ms(ttft_s),
                    "inter_token_latency": ms(itl),
                },
                **extra,
            }

        records = [
            record(0, 10, 2, 1000, 81),  # ITL 0.1 s
            record(5, 15, 5, 3000, 201),  # ITL 0.05 s, ends at 20 s
            record(1, 1, 1, 10, 2, error={"type": "timeout"}),
            {"metadata": {"benchmark_phase": "warmup"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "profile_export.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records)
            )
            summary = agentx_summary.summarize(tmp, chips=4)
        self.assertEqual((summary["requests"], summary["failed_requests"]), (2, 1))
        # (4000 prompt + 282 completion tokens) over the 0-20 s window on 4 chips.
        self.assertAlmostEqual(summary["tok_per_s_per_chip"], 4282 / 20 / 4)
        self.assertAlmostEqual(summary["p90_e2e_s"], 10 + 0.9 * 5)
        # The slow tail: 1 / (p90 ITL), between 1 / 0.1 and 1 / 0.05.
        self.assertAlmostEqual(summary["p90_interactivity"], 1 / (0.05 + 0.9 * 0.05))


# The submitted launch: what every swept point shares, then each point's own.
COMMON_FLAGS = {
    "--served-model-name": ["meta-models/Muse-Spark-Open-816B-A42B-NVFP4"],
    "--device": ["tpu"],
    "--dtype": ["bfloat16"],
    "--grammar-backend": ["none"],
    "--tp-size": ["8"],
    "--ep-size": ["1"],
    "--quantization": ["modelopt_fp4"],
    "--moe-runner-backend": ["pallas_tpu_fused_rs"],
    "--attention-backend": ["pallas_tpu_rpa_batched"],
    "--speculative-draft-attention-backend": ["pallas_tpu_rpa_v3"],
    "--page-size": ["256"],
    "--kv-cache-dtype": ["fp8_e4m3"],
    "--chunked-prefill-size": ["4096"],
    "--max-prefill-tokens": ["4096"],
    "--swa-full-tokens-ratio": ["0.0625"],
    "--mem-fraction-static": ["0.95"],
    "--enable-torch-compile": [],
    "--speculative-algorithm": ["DFLASH"],
    "--speculative-num-draft-tokens": ["8"],
    "--scheduler-recv-interval": ["8"],
    "--schedule-policy": ["fcfs"],
    "--reasoning-parser": ["muse"],
    "--tool-call-parser": ["muse"],
    "--watchdog-timeout": ["3600"],
    "--enable-cache-report": [],
    "--enable-metrics": [],
    "--enable-hierarchical-cache": [],
    "--hicache-io-backend": ["raiden"],
    "--hicache-mem-layout": ["page_first_fused"],
}

COMMON_ENV = {
    "ENABLE_TPU_TELEMETRY": "0",
    "FORCE_SINGLE_HOST": "1",
    "LIBTPU_INIT_ARGS": "--xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_latency_hiding_scheduler=true --xla_latency_hiding_scheduler_rerun=10 --xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative --xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DFLASH_CONV_MATMUL_EXPAND": "1",
    "SGLANG_DFLASH_ROW_PARALLEL_FC": "1",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
    "SGLANG_INIT_NEW_TOKEN_RATIO": "1.0",
    "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR": "1.0",
    "SGLANG_PREFILL_CHUNK_SHARE_MIN_TOKENS": "1024",
    "SGLANG_PREFILL_DECODE_INTERVAL_SKIP_IDLE": "1",
    "SGLANG_SIMULATE_ACC_LEN": "4.05",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    "SGLANG_SPEC_TP_SYNC": "rng",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
    "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
    "SGLANG_TPU_DFLASH_FP8_DRAFT": "1",
    "SGLANG_TPU_FP4_DEQUANT_VIA_FP8": "1",
    "SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE": "1",
    "SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS": "1",
    "SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE": "1",
    "SGLANG_TPU_FUSED_RS_ETP_SKIP_OUT_ZERO": "1",
    "SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE": "1",
    "SGLANG_TPU_FUSED_RS_GMM2_ROW_OUT": "1",
    "SGLANG_TPU_FUSED_RS_PAIR_LOCAL_RANKS": "1",
    "SGLANG_TPU_FUSED_RS_PAIR_RUN_ADD": "1",
    "SGLANG_TPU_FUSED_RS_PAIR_RUN_TABLE": "1",
    "SGLANG_TPU_FUSED_RS_PAIR_WINDOW_ADD": "1",
    "SGLANG_TPU_FUSED_RS_ROW_LOOP_UNROLL": "8",
    "SGLANG_TPU_FUSED_RS_STAGING_SLOTS": "10",
    "SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG": "1",
    "SGLANG_TPU_HICACHE_TRIM_BEFORE_ALLOC": "1",
    "SGLANG_TPU_HOST_PADDING": "1",
    "SGLANG_TPU_KV_PACK_HEADS": "1",
    "SGLANG_TPU_MK_FOLD_ATTN_SCALE": "1",
    "SGLANG_TPU_MK_SAL_FP8_DOT": "2",
    "SGLANG_TPU_MK_Y_BLOCK_SEND": "1",
    "SGLANG_TPU_MK_Y_EARLY": "1",
    "SGLANG_TPU_MUSE_GATHER_COLUMNS_DIM": "1",
    "SGLANG_TPU_MUSE_MEGAKERNEL": "1",
    "SGLANG_TPU_MUSE_MK_ATTN_SKIP_IDLE": "1",
    "SGLANG_TPU_MUSE_MK_KV_WRITE_PIPELINED": "1",
    "SGLANG_TPU_MUSE_MK_LARGE_STEP_ATTN_PIPE": "1",
    "SGLANG_TPU_MUSE_MK_SAL_SWA_CHUNK_TOKENS": "512",
    "SGLANG_TPU_MUSE_PREFILL_SP": "1",
    "SGLANG_TPU_OVERLAP_PROFILE": "1",
    "SGLANG_TPU_PRELOAD_TELEMETRY_STUB": "1",
    "SGLANG_TPU_RAIDEN_COALESCE_PAGES": "1",
    "SGLANG_TPU_RPA_NO_MAX": "1",
    "SGLANG_TPU_RPA_OUT_HEAD_NORM_EPS": "1e-05",
    "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK": "1",
    "SGLANG_TPU_RPA_SKIP_PAD_Q": "1",
    "SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP": "256",
    "SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS": "512:512",
    "SGLANG_TPU_RPA_TRANSPOSED": "1",
    "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1",
    "TORCH_LOGS": "+recompiles",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
    "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
    "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
    "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
    "TPU_ENABLE_TELEMETRY": "0",
    "TPU_ENABLE_TELEMETRY_HARVESTING": "0",
    "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    "TPU_TELEMETRY_INTERVAL_SEC": "86400",
    "TPU_TELEMETRY_INTERVAL_SECONDS": "86400",
}

PER_CONC = {
    1: (
        {
            "--max-running-requests": ["2"],
            "--torch-compile-max-bs": ["2"],
            "--tpu-compile-bs": ["1", "2"],
            "--tpu-compile-tokens": ["1024", "2048", "4096"],
            "--prefill-decode-interval": ["128"],
            "--hicache-ratio": ["2.0"],
            "--hicache-write-policy": ["write_back"],
        },
        {
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "131072",
            "SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY": "4:4:3.06",
        },
    ),
    7: (
        {
            "--max-running-requests": ["3"],
            "--torch-compile-max-bs": ["3"],
            "--tpu-compile-bs": ["1", "2", "3"],
            "--tpu-compile-tokens": ["1024", "2048", "4096"],
            "--prefill-decode-interval": ["128"],
            "--hicache-ratio": ["2.0"],
            "--hicache-write-policy": ["write_back"],
        },
        {
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "131072",
            "SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY": "4:4:3.06",
        },
    ),
    9: (
        {
            "--max-running-requests": ["4"],
            "--torch-compile-max-bs": ["4"],
            "--tpu-compile-bs": ["1", "2", "3", "4"],
            "--tpu-compile-tokens": ["1024", "2048", "4096"],
            "--prefill-decode-interval": ["128"],
            "--hicache-ratio": ["2.0"],
            "--hicache-write-policy": ["write_back"],
        },
        {
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "131072",
            "SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY": "4:4:3.06",
        },
    ),
    10: (
        {
            "--max-running-requests": ["6"],
            "--torch-compile-max-bs": ["6"],
            "--tpu-compile-bs": ["1", "2", "3", "4", "5", "6"],
            "--tpu-compile-tokens": ["1024", "2048", "4096"],
            "--prefill-decode-interval": ["64"],
            "--hicache-ratio": ["2.0"],
            "--hicache-write-policy": ["write_back"],
        },
        {
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "131072",
            "SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY": "4:4:3.06",
        },
    ),
    16: (
        {
            "--max-running-requests": ["16"],
            "--torch-compile-max-bs": ["16"],
            "--tpu-compile-bs": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "16",
            ],
            "--tpu-compile-tokens": ["4096"],
            "--prefill-decode-interval": ["64"],
            "--hicache-ratio": ["3.0"],
            "--hicache-write-policy": ["write_through"],
            "--hicache-size": ["56"],
            "--enable-session-radix-cache": [],
            "--session-id-header": ["X-Correlation-ID"],
            "--session-leaf-at-prompt-end": [],
            "--session-coverage-host-only": [],
        },
        {
            "SGLANG_HICACHE_SWA_HOST_EVICT_PASSED_FIRST": "1",
            "SGLANG_HICACHE_SWA_HOST_FRACTION": "0.3",
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "16384",
            "SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY": "3:4:3.06",
            "SGLANG_TPU_MUSE_MK_EXPERT_SPLIT": "4",
            "SGLANG_TPU_MUSE_MK_MAX_ROWS": "64",
        },
    ),
    22: (
        {
            "--max-running-requests": ["16"],
            "--torch-compile-max-bs": ["16"],
            "--tpu-compile-bs": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "16",
            ],
            "--tpu-compile-tokens": ["4096"],
            "--prefill-decode-interval": ["64"],
            "--hicache-ratio": ["3.0"],
            "--hicache-write-policy": ["write_through"],
            "--hicache-size": ["56"],
            "--enable-session-radix-cache": [],
            "--session-id-header": ["X-Correlation-ID"],
            "--session-leaf-at-prompt-end": [],
            "--session-coverage-host-only": [],
        },
        {
            "SGLANG_HICACHE_SWA_HOST_EVICT_PASSED_FIRST": "1",
            "SGLANG_HICACHE_SWA_HOST_FRACTION": "0.3",
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "16384",
            "SGLANG_PREFILL_DECODE_INTERVAL_ADAPTIVE": "6:8",
            "SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY": "3:4:3.06",
            "SGLANG_TPU_MUSE_MK_EXPERT_SPLIT": "4",
            "SGLANG_TPU_MUSE_MK_MAX_ROWS": "64",
        },
    ),
}

if __name__ == "__main__":
    unittest.main()
