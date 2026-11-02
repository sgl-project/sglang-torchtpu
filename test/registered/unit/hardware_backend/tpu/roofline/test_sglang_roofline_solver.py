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

"""CPU unit tests for SGLang TPU layout legality, uniform PP slicing, cache ceiling, and <50ms crossover."""

from __future__ import annotations

import sys
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

try:
    from roofline import (
        CHIPS,
        TOPOLOGIES,
        Parallel,
        TrafficScenario,
        default_page_size,
        dtypes_from_hf_config,
        from_hf_config,
        hbm_breakdown_per_core_bytes,
        is_sglang_layout_legal,
        layout_page_size,
        solve_serving_plan,
        sweep_prefix_hit_crossover,
    )
except ModuleNotFoundError:
    from sglang.srt.hardware_backend.tpu.roofline import (
        CHIPS,
        TOPOLOGIES,
        Parallel,
        TrafficScenario,
        default_page_size,
        layout_page_size,
        dtypes_from_hf_config,
        from_hf_config,
        hbm_breakdown_per_core_bytes,
        is_sglang_layout_legal,
        solve_serving_plan,
        sweep_prefix_hit_crossover,
    )

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


def _dense_bf16_cfg() -> dict:
    return {
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 35,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    }


class TestSglangRooflineSolver(unittest.TestCase):
    def test_sglang_pp_tp2_and_pcp_legality_and_uniform_pp_slice(self) -> None:
        dense_geom = from_hf_config(_dense_bf16_cfg())
        dense_dt = dtypes_from_hf_config(_dense_bf16_cfg())
        q_geom = from_hf_config(_qwen35_cfg())
        q_dt = dtypes_from_hf_config(_qwen35_cfg())
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())

        self.assertEqual(default_page_size(oss_geom), 64)
        self.assertEqual(default_page_size(q_geom), 256)
        with self.assertRaises(ValueError):
            default_page_size(q_geom, requested=64)

        topo_4 = TOPOLOGIES["2x2x1"]
        topo_16 = TOPOLOGIES["2x2x4"]

        # PP=4, TP=2 (stage_is_tp_pair) on 8 cores is legal; PP=2, TP=4 without PCP==TP is rejected.
        self.assertTrue(
            is_sglang_layout_legal(
                dense_geom,
                dense_dt,
                Parallel(tp=2, pp=4, dp=1, ep=1, pcp=1),
                topo_4,
                ctx_full=4096,
                b_local=4,
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                dense_geom,
                dense_dt,
                Parallel(tp=4, pp=2, dp=1, ep=1, pcp=1),
                topo_4,
                ctx_full=4096,
                b_local=4,
            )
        )

        # Single-host BF16 non-GDN PP=2, TP=4, PCP=4 is legal on <=8 chips for supported architectures (Qwen3Moe/GptOss),
        # rejected on >8 chips, FP8 KV, GDN, or unsupported architectures (dense LLaMA).
        qwen_moe_cfg = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 35,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "dtype": "bfloat16",
        }
        qwen_moe_geom = from_hf_config(qwen_moe_cfg)
        qwen_moe_dt = dtypes_from_hf_config(qwen_moe_cfg)
        par_pp_pcp = Parallel(
            tp=4, pp=2, dp=1, ep=1, pcp=4, moe="tp_slice", moe_dispatch="allreduce"
        )
        self.assertTrue(
            is_sglang_layout_legal(
                qwen_moe_geom, qwen_moe_dt, par_pp_pcp, topo_4, ctx_full=4096, b_local=2
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                dense_geom, dense_dt, par_pp_pcp, topo_4, ctx_full=4096, b_local=2
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                qwen_moe_geom,
                replace(qwen_moe_dt, kv=1.0),
                par_pp_pcp,
                topo_4,
                ctx_full=4096,
                b_local=2,
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                q_geom,
                replace(q_dt, kv=2.0),
                par_pp_pcp,
                topo_4,
                ctx_full=4096,
                b_local=2,
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                qwen_moe_geom,
                qwen_moe_dt,
                Parallel(
                    tp=8,
                    pp=4,
                    dp=1,
                    ep=1,
                    pcp=8,
                    moe="tp_slice",
                    moe_dispatch="allreduce",
                ),
                topo_16,
                ctx_full=4096,
                b_local=2,
            )
        )

        # DCP (kv_split > 1) rejected when pp > 1 or ctx_full < page_size * kv_split, and SMEM cap enforced.
        self.assertFalse(
            is_sglang_layout_legal(
                q_geom,
                q_dt,
                Parallel(tp=2, pp=4, kv_split=2),
                topo_4,
                ctx_full=4096,
                b_local=2,
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                q_geom,
                q_dt,
                Parallel(
                    tp=8, pp=1, kv_split=4, moe="ep", ep=8, moe_dispatch="allreduce"
                ),
                topo_4,
                ctx_full=512,
                b_local=2,
            )
        )
        self.assertFalse(
            is_sglang_layout_legal(
                dense_geom,
                dense_dt,
                Parallel(tp=8, pp=1),
                topo_4,
                ctx_full=262144,
                b_local=256,
            )
        )

        # Uniform PP stage slice on 35 layers across PP=4 yields stages of (8, 9, 9, 9) layers;
        # heaviest stage carries 9 layers + lm_head (> naive 35/4 = 8.75 average).
        bd_pp4 = hbm_breakdown_per_core_bytes(
            dense_geom, dense_dt, Parallel(tp=2, pp=4), 4, 2048
        )
        bd_pp1 = hbm_breakdown_per_core_bytes(
            dense_geom, dense_dt, Parallel(tp=2, pp=1), 4, 2048
        )
        self.assertEqual(bd_pp4.stage_layers, 9)
        self.assertGreater(bd_pp4.total, bd_pp1.total / 4.0)

    def test_sglang_endogenous_cache_ceiling_and_crossover_sweep(self) -> None:
        q_geom = from_hf_config(_qwen35_cfg())
        q_dt = dtypes_from_hf_config(_qwen35_cfg())
        chip = CHIPS["tpu7x"]

        # Non-GDN Transformer MoE allows HiCache offload
        q_cfg_no_gdn = dict(_qwen35_cfg())
        q_cfg_no_gdn["full_attention_interval"] = 0
        q_cfg_no_gdn.pop("linear_num_key_heads", None)
        q_cfg_no_gdn.pop("linear_num_value_heads", None)
        q_cfg_no_gdn.pop("linear_key_head_dim", None)
        q_cfg_no_gdn.pop("linear_value_head_dim", None)
        q_cfg_no_gdn.pop("linear_conv_kernel_dim", None)
        q_geom_no_gdn = from_hf_config(q_cfg_no_gdn)
        q_dt_no_gdn = dtypes_from_hf_config(q_cfg_no_gdn)

        # Mode A shared_prefix_len / random_prefix_len derives isl and nominal prefix_hit_rate;
        # large working set clamps effective_prefix_hit_rate endogenously, and --kv-offload-gb raises it
        # while charging the 32.0 GB/s per-TensorCore PCIe HiCache floor.
        scen_no_offload = TrafficScenario(
            shared_prefix_len=28672,
            random_prefix_len=4096,
            osl=1024,
            concurrencies=(64,),
            kv_working_set_turns=8.0,
            kv_offload_gb=0.0,
            max_running_requests=32,
        )
        self.assertEqual(scen_no_offload.isl, 32768)
        self.assertAlmostEqual(scen_no_offload.prefix_hit_rate, 0.875, places=5)

        plan_no_off = solve_serving_plan(
            q_geom_no_gdn, q_dt_no_gdn, chip, scen_no_offload, mode="mixed", max_chips=4
        )
        self.assertIsNotNone(plan_no_off.recommended)
        rec_no_off = plan_no_off.recommended
        assert rec_no_off is not None
        self.assertEqual(rec_no_off.served_concurrency, 32)
        self.assertGreater(rec_no_off.queue_wait_ms, 0.0)
        self.assertIsNotNone(rec_no_off.kv_cache_capacity_tokens)
        self.assertLess(
            rec_no_off.effective_prefix_hit_rate, scen_no_offload.prefix_hit_rate
        )

        scen_offload = replace(scen_no_offload, kv_offload_gb=512.0)
        plan_off = solve_serving_plan(
            q_geom_no_gdn, q_dt_no_gdn, chip, scen_offload, mode="mixed", max_chips=4
        )
        rec_off = plan_off.recommended
        assert rec_off is not None
        self.assertGreater(
            rec_off.effective_prefix_hit_rate, rec_no_off.effective_prefix_hit_rate
        )
        self.assertGreater(rec_off.pcie_hicache_ms, 0.0)

        # GDN model rejects HiCache offload: charges 0.0 PCIe HiCache transfer time
        gdn_plan_off = solve_serving_plan(
            q_geom, q_dt, chip, scen_offload, mode="mixed", max_chips=4
        )
        assert gdn_plan_off.recommended is not None
        self.assertEqual(gdn_plan_off.recommended.pcie_hicache_ms, 0.0)

        # Warm-cache crossover sweep across (0.0, 0.25, 0.5, 0.75, 0.9, 0.95) must run in < 50 ms on CPU.
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_dt = dtypes_from_hf_config(_gpt_oss_120b_cfg())
        scen_sweep = TrafficScenario(
            isl=8192, osl=512, concurrencies=(1, 16, 64), prefix_hit_rate=0.5
        )
        # Warm cache via solve_serving_plan, then measure standalone sweep_prefix_hit_crossover latency.
        full_plan = solve_serving_plan(
            oss_geom, oss_dt, chip, scen_sweep, mode="mixed", max_chips=8
        )
        self.assertIsNotNone(full_plan.winner_prefix_hit_range)
        _cross, _ = sweep_prefix_hit_crossover(
            oss_geom, oss_dt, chip, scen_sweep, mode="mixed", max_chips=8
        )

        t0 = time.perf_counter()
        _cross, win_rng = sweep_prefix_hit_crossover(
            oss_geom, oss_dt, chip, scen_sweep, mode="mixed", max_chips=8
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 50.0)
        self.assertIsNotNone(win_rng)

    def test_sglang_sweep_prefix_hit_crossover_3_run_transition(self) -> None:
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_dt = dtypes_from_hf_config(_gpt_oss_120b_cfg())
        chip = CHIPS["tpu7x"]
        scen = TrafficScenario(
            isl=8192, osl=512, concurrencies=(1,), prefix_hit_rate=0.85
        )

        class DummyRec:
            def __init__(self, tag: str) -> None:
                self.mode = "mixed"
                self.prefill_pool = None
                self.decode_pool = type(
                    "P",
                    (),
                    {
                        "topology": tag,
                        "instances": 1,
                        "tp": 1,
                        "dp": 1,
                        "ep": 1,
                        "pp": 1,
                        "pcp": 1,
                        "kv_split": 1,
                        "spec_tokens": 0,
                    },
                )()

        class DummyPlan:
            def __init__(self, tag: str) -> None:
                self.recommended = DummyRec(tag)

        def mock_solve_core(*args, **kwargs):
            sc = kwargs.get("scenario") or args[3]
            h = sc.prefix_hit_rate
            tag = "A" if (h < 0.3 or h >= 0.75) else "B"
            return DummyPlan(tag)

        solver_mod = sys.modules[sweep_prefix_hit_crossover.__module__]
        with patch.object(solver_mod, "_solve_core", side_effect=mock_solve_core):
            cross, (win_lo, win_hi) = sweep_prefix_hit_crossover(
                oss_geom,
                oss_dt,
                chip,
                scen,
                grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                _target_family=("mixed", None, ("A", 1, 1, 1, False, 1, 1, 1, 0)),
            )
            self.assertIsNotNone(win_lo)
            self.assertGreater(win_lo, 0.6)
            self.assertLess(win_lo, 0.8)
            self.assertEqual(win_hi, 1.0)

    def test_sglang_spec_decode_mtp_gated_clamp(self) -> None:
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_dt = dtypes_from_hf_config(_gpt_oss_120b_cfg())
        chip = CHIPS["tpu7x"]
        scen_spec = TrafficScenario(
            name="coding-agent-spec",
            isl=4096,
            osl=1024,
            concurrencies=(1, 4),
            prefix_hit_rate=0.5,
            spec_tokens=3,
        )
        plan = solve_serving_plan(oss_geom, oss_dt, chip, scen_spec, max_chips=8)
        self.assertTrue(any("MTP_UNSUPPORTED_MODEL" in w for w in plan.warnings))
        for cand in plan.candidates:
            self.assertEqual(cand.decode_pool.spec_tokens, 0)
            if cand.prefill_pool is not None:
                self.assertEqual(cand.prefill_pool.spec_tokens, 0)

    def test_sglang_layout_page_size_pcp_dcp_floor(self) -> None:
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        self.assertEqual(
            layout_page_size(oss_geom, Parallel(tp=1, pcp=1, kv_split=1)), 64
        )
        self.assertEqual(
            layout_page_size(oss_geom, Parallel(tp=2, pcp=2, kv_split=1)), 256
        )
        self.assertEqual(
            layout_page_size(oss_geom, Parallel(tp=2, pcp=1, kv_split=2)), 256
        )
        with self.assertRaises(ValueError):
            layout_page_size(oss_geom, Parallel(tp=2, pcp=2), requested=64)

    def test_sglang_pcp_kv_hbm_replicated_equality(self) -> None:
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_cfg = _gpt_oss_120b_cfg()
        dt_bf16 = replace(dtypes_from_hf_config(oss_cfg), kv=2.0)
        bd_pcp2 = hbm_breakdown_per_core_bytes(
            oss_geom, dt_bf16, Parallel(tp=2, pp=4, pcp=2, moe="tp_slice"), 1, 4096
        )
        bd_pcp1 = hbm_breakdown_per_core_bytes(
            oss_geom, dt_bf16, Parallel(tp=1, pp=4, pcp=1, moe="tp_slice"), 1, 4096
        )
        self.assertAlmostEqual(bd_pcp2.kv, bd_pcp1.kv, places=3)

        chip = CHIPS["tpu7x"]
        plan_oss = solve_serving_plan(
            oss_geom,
            dt_bf16,
            chip,
            TrafficScenario(
                name="oss_auto_page",
                isl=4096,
                osl=512,
                concurrencies=(8,),
                page_size=None,
            ),
            max_chips=8,
        )
        self.assertTrue(
            any(
                p.decode_pool.pcp > 1 and p.decode_pool.page_size == 256
                for p in plan_oss.candidates
            )
        )
        self.assertTrue(
            any(
                p.decode_pool.kv_split == 2 and p.decode_pool.page_size == 256
                for p in plan_oss.candidates
            )
        )
        self.assertTrue(
            any(
                p.decode_pool.pcp == 1
                and p.decode_pool.kv_split == 1
                and p.decode_pool.page_size == 64
                for p in plan_oss.candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
