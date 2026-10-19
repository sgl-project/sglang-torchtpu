# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash megakernel: kernel configs and their VMEM scratch, host only.

Builds the checkpoint's tp8 geometry from its architecture fields and checks, for
the served shapes (one query, and the MTP verify's draft tokens), that the
kernel's scratch fits the TensorCore's VMEM and that the config's ring
constraints hold.
"""

import json
import math
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)

# Per-TensorCore VMEM on tpu7x.
_VMEM_BYTES = 64 * 2**20

_LAYERS = 45
_DSA = {3 + 4 * i for i in range(11)}


def _text_config():
    return {
        "vocab_size": 154880, "hidden_size": 4096, "num_hidden_layers": _LAYERS,
        "rms_norm_eps": 1e-05, "pad_token_id": 154820,
        "eos_token_id": [154820, 154827, 154829], "tie_word_embeddings": False,
        "layer_types": ["deepseek_sparse_attention" if L in _DSA else "linear_attention"
                        for L in range(_LAYERS)],
        "mlp_layer_types": ["dense" if L < 3 else "sparse" for L in range(_LAYERS)],
        "indexer_types": ["full"] * _LAYERS,
        "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20,
        "num_attention_heads": 64, "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_head_dim": 256, "qk_rope_head_dim": 0, "v_head_dim": 256,
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
        "index_kpool": 4, "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "linear_attn_config": {"num_heads": 64, "head_dim": 128,
                               "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "n_routed_experts": 288, "n_shared_experts": 1, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
        "topk_method": "noaux_tc", "swiglu_limit": 10.0,
        "first_k_dense_replace": 3, "num_nextn_predict_layers": 1,
    }


def _spec():
    from sglang_tpu.glm_flash_megakernel.config import Glm53FlashSpec

    raw = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": _text_config(),
        "quantization_config": {"fmt": "e4m3", "weight_block_size": [128, 128],
                                "modules_to_not_convert": []},
    }
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(raw, f)
        return Glm53FlashSpec.from_checkpoint(d)


def _scratch_bytes(spec, cfg) -> int:
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.glm_flash_megakernel.kernel import decode as D
    from sglang_tpu.glm_flash_megakernel.layout import KernelLayout

    total = 0
    for s in D.scratch_shapes(spec, KernelLayout(spec, 8), cfg):
        if getattr(s, "memory_space", None) == pltpu.VMEM:   # not the semaphores
            total += math.prod(s.shape) * jnp.dtype(s.dtype).itemsize
    return total


class TestGlmFlashMegakernelConfig(CustomTestCase):
    def test_served_shapes_fit_vmem(self):
        from sglang_tpu.glm_flash_megakernel.kernel import decode as D

        spec = _spec()
        for n_query in (1, 4):
            with self.subTest(n_query=n_query):
                cfg = D.KernelConfig(max_len=16384, n_query=n_query, embed_2d=True)
                self.assertLessEqual(_scratch_bytes(spec, cfg), _VMEM_BYTES)

    def test_ring_constraints(self):
        from sglang_tpu.glm_flash_megakernel.kernel import decode as D

        for n_query in (1, 4):
            cfg = D.KernelConfig(max_len=16384, n_query=n_query)
            self.assertLessEqual(cfg.km_issue, cfg.km_slots - 1)
            self.assertLessEqual(cfg.prefetch, cfg.slots - 2)
        self.assertEqual(D.KernelConfig(max_len=16384, n_query=4).km_cols, 256)
        with self.assertRaises(ValueError):
            D.KernelConfig(max_len=1000)                 # not a multiple of attn_tile


if __name__ == "__main__":
    unittest.main()
