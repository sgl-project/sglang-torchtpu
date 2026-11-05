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

"""CPU unit tests for SGLang TPU module costs, FP32 MambaPool, fused_rs, and step floors."""

from __future__ import annotations

import unittest
from dataclasses import replace

try:
    from roofline import (
        CHIPS,
        Parallel,
        StepShape,
        dtypes_from_hf_config,
        from_hf_config,
        pass_costs,
        plan,
        spec_cycle_floor,
        step_floor,
        supports_fused_rs,
    )
except ModuleNotFoundError:
    from sglang.srt.hardware_backend.tpu.roofline import (
        CHIPS,
        Parallel,
        StepShape,
        dtypes_from_hf_config,
        from_hf_config,
        pass_costs,
        plan,
        spec_cycle_floor,
        step_floor,
        supports_fused_rs,
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


class TestSglangRooflineModules(unittest.TestCase):
    def test_sglang_roofline_moe_tp_under_pcp(self) -> None:
        cfg = {
            "model_type": "qwen3_moe",
            "torch_dtype": "bfloat16",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1024,
        }
        geom = from_hf_config(cfg)
        dt = dtypes_from_hf_config(cfg)
        shape = StepShape(batch=1, rows_per_seq=1, prefill_tokens=0)
        par_pcp = Parallel(
            tp=4,
            pcp=2,
            moe="tp_slice",
            moe_dispatch="allreduce",
            residual="replicated",
        )
        costs = pass_costs(geom, par_pcp, shape, dt)
        moe_costs = [c for c in costs if c.name == "moe"]
        self.assertTrue(len(moe_costs) > 0)
        moe_ars = [c for c in moe_costs[0].collectives if c.name == "moe_ar"]
        self.assertTrue(len(moe_ars) > 0)
        self.assertEqual(moe_ars[0].group, 2)

        par_pcp_tp2 = Parallel(
            tp=2, pcp=2, moe="tp_slice", moe_dispatch="allreduce", residual="replicated"
        )
        costs_tp2 = pass_costs(geom, par_pcp_tp2, shape, dt)
        moe_costs_tp2 = [c for c in costs_tp2 if c.name == "moe"]
        moe_ars_tp2 = [c for c in moe_costs_tp2[0].collectives if c.name == "moe_ar"]
        self.assertEqual(len(moe_ars_tp2), 0)

    def test_sglang_mamba_pool_fp32_conv_and_fused_rs_256_expert_gate(self) -> None:
        q_cfg = _qwen35_cfg()
        q_geom = from_hf_config(q_cfg)
        q_dt_fp32_conv = dtypes_from_hf_config(q_cfg)
        q_dt_bf16_conv = replace(q_dt_fp32_conv, conv=2.0)

        # Verify 186.33 MiB/req (FP32 conv + FP32 recurrent) vs 183.16 MiB/req (BF16 conv) across 45 GDN layers.
        par_tp1 = Parallel(tp=1, moe="tp_slice", moe_dispatch="allreduce")
        shape_1 = StepShape(batch=1, rows_per_seq=1, context=1024, state_writes=0)
        shape_0 = StepShape(batch=0, rows_per_seq=0, context=0, state_writes=0)
        gdn_fp32 = next(
            m
            for m in pass_costs(q_geom, par_tp1, shape_1, q_dt_fp32_conv)
            if m.name == "gdn"
        )
        gdn_w_only = next(
            m
            for m in pass_costs(q_geom, par_tp1, shape_0, q_dt_fp32_conv)
            if m.name == "gdn"
        )
        gdn_bf16 = next(
            m
            for m in pass_costs(q_geom, par_tp1, shape_1, q_dt_bf16_conv)
            if m.name == "gdn"
        )

        slot_mib_fp32 = (gdn_fp32.hbm_bytes - gdn_w_only.hbm_bytes) / float(1024**2)
        slot_mib_bf16 = (gdn_bf16.hbm_bytes - gdn_w_only.hbm_bytes) / float(1024**2)
        self.assertAlmostEqual(slot_mib_fp32, 186.328125, places=4)
        self.assertAlmostEqual(slot_mib_bf16, 183.1640625, places=4)

        # Verify fused_rs <= 256 expert gate and (top_k / ep) reduce-scatter wire byte scaling.
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_dt = dtypes_from_hf_config(_gpt_oss_120b_cfg())
        par_fused_ok = Parallel(
            tp=8, ep=8, dp=1, pp=1, moe="ep", moe_dispatch="fused_rs", residual="sp"
        )
        par_ar_sp = Parallel(
            tp=8, ep=8, dp=1, pp=1, moe="ep", moe_dispatch="allreduce", residual="sp"
        )
        self.assertTrue(supports_fused_rs(oss_geom, par_fused_ok, total_cores=8))

        moe_fused = next(
            m
            for m in pass_costs(
                oss_geom,
                par_fused_ok,
                StepShape(batch=16, context=512),
                oss_dt,
                total_cores=8,
            )
            if m.name == "moe"
        )
        moe_ar = next(
            m
            for m in pass_costs(
                oss_geom,
                par_ar_sp,
                StepShape(batch=16, context=512),
                oss_dt,
                total_cores=8,
            )
            if m.name == "moe"
        )
        rs_fused = next(c for c in moe_fused.collectives if c.name == "combine_rs")
        rs_dense = next(c for c in moe_ar.collectives if c.name == "combine_rs")
        self.assertAlmostEqual(
            rs_fused.wire_bytes, rs_dense.wire_bytes * (4.0 / 8.0), places=5
        )

        # E=512 (Qwen3.5-397B), dp>1, pp>1, and top_k >= ep must all reject fused_rs.
        self.assertFalse(supports_fused_rs(q_geom, par_fused_ok, total_cores=8))
        with self.assertRaises(ValueError):
            pass_costs(
                q_geom,
                par_fused_ok,
                StepShape(batch=8, context=512),
                q_dt_fp32_conv,
                total_cores=8,
            )
        with self.assertRaises(ValueError):
            pass_costs(
                oss_geom,
                replace(par_fused_ok, dp=2),
                StepShape(batch=8, context=512),
                oss_dt,
                total_cores=16,
            )

    def test_mxu_m_tile_nope_mla_and_step_floor(self) -> None:
        oss_geom = from_hf_config(_gpt_oss_120b_cfg())
        oss_dt = dtypes_from_hf_config(_gpt_oss_120b_cfg())
        par = Parallel(tp=8, ep=8, moe="ep", moe_dispatch="allreduce")
        mods_tile256 = pass_costs(
            oss_geom, par, StepShape(batch=4, context=512), oss_dt, mxu_m_tile=256
        )
        mods_tile1 = pass_costs(
            oss_geom, par, StepShape(batch=4, context=512), oss_dt, mxu_m_tile=1
        )
        moe_256 = next(m for m in mods_tile256 if m.name == "moe")
        moe_1 = next(m for m in mods_tile1 if m.name == "moe")
        self.assertGreater(
            moe_256.fp8_flops + moe_256.bf16_flops, moe_1.fp8_flops + moe_1.bf16_flops
        )

        chip = CHIPS["tpu7x"]
        sf = step_floor(oss_geom, chip, par, StepShape(batch=8, context=1024), oss_dt)
        self.assertGreater(sf.serial_floor_s, 0.0)
        spec_sf = spec_cycle_floor(oss_geom, chip, par, 8, 1024, 3, oss_dt)
        self.assertGreater(spec_sf.serial_floor_s, sf.serial_floor_s)
        rows = plan(oss_geom, chip, par, oss_dt, context=1024, concurrencies=[1, 8])
        self.assertEqual(len(rows), 2)
        self.assertGreater(
            rows[1].output_tput_ceil_per_chip, rows[0].output_tput_ceil_per_chip
        )


if __name__ == "__main__":
    unittest.main()
