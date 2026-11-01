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

"""CPU unit tests for SGLang TPU roofline specs, torus collectives, and HF geometry."""

from __future__ import annotations

import unittest

try:
    from roofline import (
        CHIPS,
        TOPOLOGIES,
        collective_ici_hops,
        dtypes_from_hf_config,
        from_hf_config,
        submesh_for_group,
        supports,
        torus_collective_s,
    )
    from roofline.collectives import make
except ModuleNotFoundError:
    from sglang.srt.hardware_backend.tpu.roofline import (
        CHIPS,
        TOPOLOGIES,
        collective_ici_hops,
        dtypes_from_hf_config,
        from_hf_config,
        submesh_for_group,
        supports,
        torus_collective_s,
    )
    from sglang.srt.hardware_backend.tpu.roofline.collectives import make
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestSglangRooflineGeometry(unittest.TestCase):
    def test_sglang_roofline_gpt_oss_120b_and_supports_validator(self) -> None:
        gpt_oss_cfg = {
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
        self.assertTrue(supports(gpt_oss_cfg))
        geom = from_hf_config(gpt_oss_cfg)
        dt = dtypes_from_hf_config(gpt_oss_cfg)

        self.assertEqual(geom.hidden, 2880)
        self.assertEqual(geom.num_layers, 36)
        self.assertEqual(geom.model_type, "gpt_oss")
        self.assertEqual(geom.architectures, ("GptOssForCausalLM",))
        self.assertEqual(geom.num_experts, 128)
        self.assertEqual(geom.top_k, 4)
        self.assertEqual(geom.sliding_window, 128)
        self.assertTrue(geom.has_attention_sink)
        self.assertEqual(geom.layer_types.count("sliding_attention"), 18)
        self.assertEqual(geom.layer_types.count("full_attention"), 18)
        self.assertAlmostEqual(dt.w_expert, 0.53125, places=6)
        self.assertEqual(dt.state, 4.0)
        self.assertEqual(dt.conv, 4.0)

        # Preflight validator rejects malformed, vision-only, and incomplete GDN configs.
        self.assertFalse(supports({"model_type": "clip", "hidden_size": 1024}))
        self.assertFalse(
            supports(
                {"hidden_size": 0, "num_hidden_layers": 4, "num_attention_heads": 4}
            )
        )
        self.assertFalse(
            supports(
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 16,
                    "intermediate_size": 4096,
                    "layer_types": ["linear_attention"] * 4,
                }
            )
        )
        with self.assertRaises(ValueError):
            from_hf_config({"model_type": "vit", "hidden_size": 768})

    def test_sglang_roofline_mamba_fp32_state_pinning(self) -> None:
        mamba_cfg = {
            "model_type": "mamba",
            "torch_dtype": "bfloat16",
            "mamba_ssm_dtype": "bfloat16",
            "mamba_conv_dtype": "bfloat16",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
        }
        dt = dtypes_from_hf_config(mamba_cfg)
        self.assertEqual(dt.state, 4.0)
        self.assertEqual(dt.conv, 4.0)

    def test_qwen35_deepseek_and_nope_mla_geometry(self) -> None:
        qwen35_cfg = {
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
        self.assertTrue(supports(qwen35_cfg))
        q_geom = from_hf_config(qwen35_cfg)
        q_dt = dtypes_from_hf_config(qwen35_cfg)
        self.assertEqual(q_geom.layer_types.count("linear_attention"), 45)
        self.assertEqual(q_geom.layer_types.count("full_attention"), 15)
        self.assertFalse(q_geom.has_attention_sink)
        self.assertEqual(q_dt.conv, 4.0)
        self.assertEqual(q_dt.state, 4.0)
        self.assertEqual(q_dt.kv, 1.0)

        nope_mla_cfg = {
            "model_type": "deepseek_v3",
            "torch_dtype": "bfloat16",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 0,
            "v_head_dim": 128,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "first_k_dense_replace": 3,
            "intermediate_size": 18432,
            "vocab_size": 129280,
        }
        self.assertTrue(supports(nope_mla_cfg))
        m_geom = from_hf_config(nope_mla_cfg)
        self.assertEqual(m_geom.qk_rope_head_dim, 0)
        self.assertEqual(m_geom.index_n_heads, 64)
        self.assertEqual(m_geom.layer_types[0], "dsa")

    def test_tpu7x_datasheet_and_torus_wraparound(self) -> None:
        chip = CHIPS["tpu7x"]
        self.assertEqual(chip.mxu_m_tile, 256)
        self.assertEqual(chip.pcie_bw, 64_000_000_000.0)
        self.assertEqual(chip.pcie_bw_per_core, 32_000_000_000.0)
        self.assertEqual(chip.dcn_bw, 12_500_000_000.0)

        topo_open = TOPOLOGIES["2x2x4"]
        topo_torus = TOPOLOGIES["4x4x4"]
        self.assertFalse(topo_open.has_torus_wraparound)
        self.assertTrue(topo_torus.has_torus_wraparound)

        coll = make("ar_test", "ar", 1024.0 * 1024.0, 128, "two_shot", 2)
        hops_torus = collective_ici_hops(coll, topo_torus, 2)
        self.assertEqual(hops_torus, 12)
        sub = submesh_for_group(128, topo_torus, 2)
        self.assertGreater(
            torus_collective_s("ar", 1024.0 * 1024.0, sub, chip, topo_torus), 0.0
        )


if __name__ == "__main__":
    unittest.main()
