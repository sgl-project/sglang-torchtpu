"""TPU server-argument decisions made by the plugin's resolution hooks.

The resolution cases run a full `resolve_once()` on the TPU platform (mode A)
and check what the core `arg_groups` handlers resolved before these decisions
moved into `sglang_tpu.server_args_resolution`.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt import platforms
from sglang.srt.arg_groups import kimi_k3_hook
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks.experimental_spec import kimi_spec_defaults
from sglang_tpu.server_args_resolution import _reject_kda_speculative_decoding
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# index_topk puts this config on the DeepSeek sparse attention (DSA) path.
_DEEPSEEK_DSA = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 16,
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
    "kv_lora_rank": 8,
    "q_lora_rank": 8,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 8,
    "topk_method": "greedy",
    "scoring_func": "softmax",
    "index_topk": 4,
    "index_head_dim": 8,
    "index_n_heads": 2,
}

# A Kimi hybrid with one KDA and one MLA layer.
_KIMI_LINEAR = {
    "architectures": ["KimiLinearForCausalLM"],
    "model_type": "kimi_linear",
    "hidden_size": 64,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_hidden_layers": 2,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
    "kv_lora_rank": 16,
    "q_lora_rank": None,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "v_head_dim": 16,
    "mla_use_nope": True,
    "linear_attn_config": {
        "kda_layers": [1],
        "full_attn_layers": [2],
        "head_dim": 16,
        "num_heads": 4,
        "short_conv_kernel_size": 4,
    },
}

_KDA_REJECTION = r"not supported for Kimi hybrid \(KDA\) models on TPU"


class _TpuResolveCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def resolve(self, config, **fields):
        path = tempfile.mkdtemp(prefix="tpu_server_args_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(config, handle)
        if fields.get("speculative_algorithm") is not None:
            fields.setdefault("speculative_draft_model_path", path)
        original = kimi_k3_hook.apply_kimi_k3_spec_backend_defaults
        with (
            override_platform(
                is_tpu=True,
                is_cuda=False,
                is_hip=False,
                is_musa=False,
                is_npu=False,
                is_xpu=False,
                is_sm100=False,
            ),
            patch.object(platforms, "_current_platform", TpuSRTPlatform()),
            # Builds the slice environment of a real TPU worker.
            patch("sglang_tpu.patches._prepare_tpu_env"),
            # The registry hook sglang_tpu.hooks.register() installs on a TPU host.
            patch.object(
                kimi_k3_hook,
                "apply_kimi_k3_spec_backend_defaults",
                lambda server_args: kimi_spec_defaults(original, server_args),
            ),
        ):
            server_args = ServerArgs(
                model_path=path, device="tpu", random_seed=42, **fields
            )
            server_args.resolve_once()
        return server_args


class TestKimiHybridSpeculativeDecoding(_TpuResolveCase):
    def test_rejected_without_a_tpu_kda_verify_path(self):
        for algorithm in ("EAGLE", "NEXTN"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, _KDA_REJECTION):
                    self.resolve(_KIMI_LINEAR, speculative_algorithm=algorithm)

    def test_rejected_dflash_without_the_experimental_worker(self):
        with (
            tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False),
            tpu_envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(False),
        ):
            with self.assertRaisesRegex(ValueError, _KDA_REJECTION):
                self.resolve(_KIMI_LINEAR, speculative_algorithm="DFLASH")

    def test_experimental_worker_algorithms_resolve_without_gpu_pins(self):
        for algorithm in ("DSPARK", "DFLASH"):
            with self.subTest(algorithm=algorithm):
                server_args = self.resolve(
                    _KIMI_LINEAR, speculative_algorithm=algorithm
                )
                self.assertIsNone(
                    resolution_result(server_args, "linear_attn_verify_backend")
                )

    def test_without_speculative_decoding(self):
        server_args = self.resolve(_KIMI_LINEAR)
        self.assertIsNone(resolution_result(server_args, "speculative_algorithm"))

    def test_rejects_every_kimi_hybrid_architecture_before_the_step(self):
        cases = (
            ("KimiLinearForCausalLM", True),
            ("KimiK3ForConditionalGeneration", True),
            ("KimiK31ForConditionalGeneration", True),
            ("DeepseekV3ForCausalLM", False),
        )
        for architecture, rejected in cases:
            server_args = ServerArgs(model_path="dummy", speculative_algorithm="EAGLE")
            server_args._model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture])
            )
            steps = []
            with (
                self.subTest(architecture=architecture),
                override_platform(is_tpu=True),
            ):
                if rejected:
                    with self.assertRaisesRegex(ValueError, _KDA_REJECTION):
                        _reject_kda_speculative_decoding(server_args, steps.append)
                    self.assertEqual(steps, [])
                else:
                    _reject_kda_speculative_decoding(server_args, steps.append)
                    self.assertEqual(steps, [server_args])


class TestDsaSkipsGpuDefaults(_TpuResolveCase):
    def test_gpu_dsa_passes_do_not_run(self):
        server_args = self.resolve(
            _DEEPSEEK_DSA, attention_backend="pallas_tpu_mla_sparse"
        )
        # The CUDA arm picks flashmla split backends and a bf16 KV cache.
        self.assertIsNone(resolution_result(server_args, "dsa_prefill_backend"))
        self.assertIsNone(resolution_result(server_args, "dsa_decode_backend"))
        self.assertEqual(resolution_result(server_args, "kv_cache_dtype"), "fp8_e4m3")
        self.assertEqual(resolution_result(server_args, "page_size"), 64)

    def test_operator_page_size_stands(self):
        server_args = self.resolve(
            _DEEPSEEK_DSA, attention_backend="pallas_tpu_mla_sparse", page_size=128
        )
        self.assertEqual(resolution_result(server_args, "page_size"), 128)


if __name__ == "__main__":
    unittest.main()
