"""An all-sliding-window DFlash2 draft is hybrid SWA from its own config.

RadixArk/Muse-Spark-Open-DFlash2 lists every layer as sliding_attention but does not
declare is_hybrid_swa. ModelConfig must size and route its KV exactly as it would for a
copy of that config that declares is_hybrid_swa and an all-SWA hybrid_layer_pattern.
"""

import json
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_hybrid_swa_model,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# RadixArk/Muse-Spark-Open-DFlash2's config.json.
DRAFT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0,
    "bos_token_id": 200000,
    "dflash_config": {
        "attention_mode": "gqa",
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "final_logit_softcapping": 20,
        "mask_token_id": 200002,
        "output_multiplier": 0.1767766952966369,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [1, 16, 30, 44, 59],
    },
    "draft_vocab_size": 202048,
    "dtype": "bfloat16",
    "eos_token_id": [200001, 200008],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "intermediate_size": 28672,
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 64,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 62,
    "pad_token_id": 200018,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 500000, "rope_type": "default"},
    "rope_theta": 500000,
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": True,
    "vocab_size": 202048,
}
# What the AgentX recipe used to serve instead: the same config, declared hybrid SWA.
WINDOWED_CONFIG = {
    **DRAFT_CONFIG,
    "is_hybrid_swa": True,
    "hybrid_layer_pattern": [1] * 5,
}
CONFIG_OBJECTS = {"model_path", "hf_config", "hf_text_config", "hf_generation_config"}


def draft_model_config(config, tmp):
    with open(f"{tmp}/config.json", "w") as f:
        json.dump(config, f)
    return ModelConfig(tmp, is_draft_model=True, speculative_algorithm="DFLASH")


class TestDFlash2DraftHybridSwa(CustomTestCase):
    def test_original_config_matches_windowed_config(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            original = draft_model_config(DRAFT_CONFIG, a)
            windowed = draft_model_config(WINDOWED_CONFIG, b)
        self.assertTrue(original.is_hybrid_swa)
        self.assertEqual(original.swa_attention_layer_ids, [0, 1, 2, 3, 4])
        self.assertEqual(original.full_attention_layer_ids, [])
        self.assertEqual(original.sliding_window_size, 2048)
        fields = lambda mc: {
            k: repr(v) for k, v in vars(mc).items() if k not in CONFIG_OBJECTS
        }
        self.assertEqual(fields(original), fields(windowed))
        self.assertEqual(original.get_num_kv_heads(8), windowed.get_num_kv_heads(8))

    def test_only_all_sliding_dflash2_drafts(self):
        sliding = SimpleNamespace(
            num_hidden_layers=2, layer_types=["sliding_attention"] * 2
        )
        mixed = SimpleNamespace(
            num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"]
        )
        self.assertTrue(is_hybrid_swa_model(["DFlash2DraftModel"], sliding))
        self.assertEqual(
            get_hybrid_layer_ids(["DFlash2DraftModel"], sliding), ([0, 1], [])
        )
        self.assertFalse(is_hybrid_swa_model(["DFlash2DraftModel"], mixed))
        # Other architectures keep their behavior for the same layer_types.
        self.assertFalse(is_hybrid_swa_model(["Qwen3ForCausalLM"], sliding))
        self.assertEqual(
            get_hybrid_layer_ids(["Qwen3ForCausalLM"], sliding), (None, None)
        )


if __name__ == "__main__":
    unittest.main()
