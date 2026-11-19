# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash megakernel: served configs fit VMEM at tp8, and the split latent cache's
ownership math matches serving's layout (host only)."""

import itertools
import json
import math
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)

_VMEM_BYTES = 64 * 2**20  # per TensorCore on tpu7x

_LAYERS = 45
_DSA = {3 + 4 * i for i in range(11)}


def _text_config():
    return {
        "vocab_size": 154880,
        "hidden_size": 4096,
        "num_hidden_layers": _LAYERS,
        "rms_norm_eps": 1e-05,
        "pad_token_id": 154820,
        "eos_token_id": [154820, 154827, 154829],
        "tie_word_embeddings": False,
        "layer_types": [
            "deepseek_sparse_attention" if L in _DSA else "linear_attention"
            for L in range(_LAYERS)
        ],
        "mlp_layer_types": ["dense" if L < 3 else "sparse" for L in range(_LAYERS)],
        "indexer_types": ["full"] * _LAYERS,
        "hc_mult": 4,
        "hc_eps": 1e-06,
        "hc_sinkhorn_iters": 20,
        "num_attention_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "n_routed_experts": 288,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "intermediate_size": 12288,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "swiglu_limit": 10.0,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
    }


def _spec():
    from sglang_tpu.glm_flash_megakernel.config import Glm53FlashSpec

    raw = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": _text_config(),
        "quantization_config": {
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [],
        },
    }
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(raw, f)
        return Glm53FlashSpec.from_checkpoint(d)


class TestGlmFlashMegakernelConfig(CustomTestCase):
    def test_served_configs(self):
        import jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.glm_flash_megakernel.kernel import decode as D
        from sglang_tpu.glm_flash_megakernel.layout import KernelLayout

        spec = _spec()
        # The served buckets: plain decode and the NEXTN-3 verify, up to 64 rows; each
        # with the latent cache whole or split over tp8.
        cases = ((1, 1), (1, 16), (4, 1), (4, 16))
        for (n_query, n_req), shards in itertools.product(cases, (1, 8)):
            with self.subTest(n_query=n_query, n_req=n_req, shards=shards):
                cfg = D.KernelConfig(
                    max_len=16384,
                    n_query=n_query,
                    n_req=n_req,
                    embed_2d=True,
                    context_parallel=shards > 1,
                    mla_shards=shards,
                )
                vmem = sum(
                    math.prod(s.shape) * jnp.dtype(s.dtype).itemsize
                    for s in D.scratch_shapes(spec, KernelLayout(spec, 8), cfg)
                    if getattr(s, "memory_space", None) == pltpu.VMEM
                )
                self.assertLessEqual(vmem, _VMEM_BYTES)
                self.assertLessEqual(cfg.km_issue, cfg.km_slots - 1)
                self.assertLessEqual(cfg.prefetch, cfg.slots - 2)
        with self.assertRaises(ValueError):
            D.KernelConfig(max_len=1000)  # not a multiple of attn_tile

    def test_split_ownership_matches_serving(self):
        """A pool's owner and gather-block row (layout.py, read by the kernel) agree with
        where serving stores each of its tokens (kv_shard.own_page_slots)."""
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention.pallas.mla_sparse.kv_shard import own_page_slots
        from sglang_tpu.glm_flash_megakernel.layout import pool_block_row, pool_owner

        ps, kp, block = 128, 4, 16
        for shards in (1, 2, 4, 8):
            sub = ps // shards
            pools = np.arange(3 * ps // kp, dtype=np.int32)  # three pages of pools
            owner = pool_owner(pools, ps // kp, sub // kp)
            np.testing.assert_array_equal(
                owner, pool_owner(jnp.asarray(pools), ps // kp, sub // kp)
            )
            for pool in pools.tolist():
                row0 = pool_block_row(pool, kp, ps, sub, block)
                for t in range(pool * kp, (pool + 1) * kp):
                    page, slot = t // ps, t % ps
                    # The rank whose store keeps this token, and the row it keeps it at.
                    roles = [
                        r
                        for r in range(shards)
                        if int(
                            own_page_slots(
                                jnp.int32(page),
                                jnp.int32(slot),
                                sub_page=sub,
                                role=r,
                                oob_page=-1,
                            )[0]
                        )
                        == page
                    ]
                    _, row = own_page_slots(
                        jnp.int32(page),
                        jnp.int32(slot),
                        sub_page=sub,
                        role=0,
                        oob_page=-1,
                    )
                    self.assertEqual(roles, [int(owner[pool])])
                    # The kernel gathers the block at row0; the token is row t % block of it.
                    self.assertEqual(row0 + t % block, int(row))


if __name__ == "__main__":
    unittest.main()
