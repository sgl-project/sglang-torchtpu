# SPDX-License-Identifier: Apache-2.0
"""Host-side Kimi-K3.1 layout of the K3 megakernel (no kernel launch).

* config: DSA layers from ``dsa_config`` (must be the MLA layers), the
  unsupported-config refusals, and EP-local checkpoint detection;
* stack: the DSA inputs, state inputs and index-key pool shapes;
* weights: a DSA layer's w1 carries the indexer wk / weights_proj blocks with
  g_proj moved behind them, every other MLA tensor unchanged, and each lane gets
  its 4 index heads of wq_b;
* expert-block remap: kernel expert e reads checkpoint expert ``ckpt_expert(e)``.

Usage:
    python test/registered/tpu/unit/test_k3_megakernel_k31_layout.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

import json
import os
import tempfile
import unittest

import ml_dtypes
import numpy as np

from sglang_tpu.k3_megakernel import model_io
from sglang_tpu.k3_megakernel.config import K3Dims, MoeLayout, Topology
from sglang_tpu.k3_megakernel.kernel import stack
from sglang_tpu.k3_megakernel.kernel.stages import mla
from sglang_tpu.k3_megakernel.weights import load_mla_layer

LAYERS = 13
MLA = (3, 7, 11)  # 0-based full-attention layers of the 13-layer proxy
DSA_CONFIG = dict(
    dsa_layers=[l + 1 for l in MLA],  # 1-based in the config
    index_topk=2048,
    index_topk_freq=1,
    index_num_heads=32,
    index_head_dim=128,
    index_rope_dim=0,
    index_k_norm_type="layernorm",
)


def _write_ckpt(root, *, dsa_config=None, experts=None):
    """config.json (+ a safetensors index listing ``experts`` of layer 1) in ``root``."""
    text = dict(
        num_hidden_layers=LAYERS,
        linear_attn_config=dict(
            full_attn_layers=[l + 1 for l in MLA],
            kda_layers=[l + 1 for l in range(LAYERS) if l not in MLA],
        ),
    )
    if dsa_config is not None:
        text["dsa_config"] = dsa_config
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(dict(text_config=text), f)
    if experts is not None:
        names = {
            f"language_model.model.layers.1.block_sparse_moe.experts.{e}.w1.weight": "a"
            for e in experts
        }
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
            json.dump(dict(weight_map=names), f)


class _Ckpt:
    """``Checkpoint.get`` over in-memory tensors."""

    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, name):
        return self.tensors[name]


def _mla_tensors(rng, layer, hidden, heads=96):
    bf = lambda *s: rng.standard_normal(s).astype(ml_dtypes.bfloat16)  # noqa: E731
    pre = f"layers.{layer}.self_attn."
    t = {
        "q_a_proj.weight": bf(1536, hidden),
        "kv_a_proj_with_mqa.weight": bf(576, hidden),
        "q_b_proj.weight": bf(heads * 192, 1536),
        "kv_b_proj.weight": bf(heads * 256, 512),
        "g_proj.weight": bf(heads * 128, hidden),
        "o_proj.weight": bf(hidden, heads * 128),
        "q_a_layernorm.weight": bf(1536),
        "kv_a_layernorm.weight": bf(512),
        "indexer.wk.weight": bf(128, hidden),
        "indexer.weights_proj.weight": bf(32, hidden),
        "indexer.wq_b.weight": bf(32 * 128, 1536),
        "indexer.k_norm.weight": bf(128),
        "indexer.k_norm.bias": bf(128),
    }
    return {pre + k: v for k, v in t.items()}


class TestK31Config(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def test_dsa_layers(self):
        _write_ckpt(self.root, dsa_config=DSA_CONFIG)
        self.assertEqual(model_io.mla_layers(LAYERS, self.root), MLA)
        self.assertEqual(model_io.dsa_layers(LAYERS, self.root), MLA)
        self.assertEqual(model_io.dsa_topk(self.root), 2048)
        _write_ckpt(self.root)  # Kimi-K3: no dsa_config
        self.assertEqual(model_io.dsa_layers(LAYERS, self.root), ())
        self.assertEqual(model_io.dsa_topk(self.root), 0)

    def test_refuses_unsupported_dsa(self):
        for bad in (
            dict(DSA_CONFIG, index_topk_freq=2),
            dict(DSA_CONFIG, index_rope_dim=64),
            dict(DSA_CONFIG, dsa_layers=[4, 8]),
        ):
            with self.subTest(bad=bad):
                _write_ckpt(self.root, dsa_config=bad)
                with self.assertRaises(ValueError):
                    model_io.dsa_layers(LAYERS, self.root)

    def test_local_expert_block(self):
        _write_ckpt(self.root, experts=range(896))
        self.assertIsNone(model_io.local_expert_block(self.root, 224, 896))
        _write_ckpt(self.root, experts=range(448, 672))
        self.assertEqual(model_io.local_expert_block(self.root, 224, 896), 2)
        _write_ckpt(self.root, experts=range(100, 324))
        with self.assertRaises(ValueError):
            model_io.local_expert_block(self.root, 224, 896)


class TestK31Stack(unittest.TestCase):
    def test_spec_and_inputs(self):
        topo = Topology(1, 8)
        spec = stack.StackSpec(topo, LAYERS, MLA, dsa_layers=MLA)
        with self.assertRaises(ValueError):
            stack.StackSpec(topo, LAYERS, MLA, dsa_layers=(3,))
        names = stack.input_names(spec, fused_head=True)
        self.assertIn("mla_idx_qb", names)
        self.assertNotIn(
            "mla_idx_qb", stack.input_names(stack.StackSpec(topo, LAYERS, MLA))
        )
        self.assertEqual(
            stack.state_inputs(spec)[-3:], ("mla_idx_0", "mla_idx_1", "mla_idx_2")
        )

        class _Kernel:  # what state_shapes reads of a StackKernel
            class kda:
                hd, heads = 1536, 12

            idx_page_shape, idx_pool_multiple = stack.IDX_PAGE_SHAPE, 1

        shapes = model_io.state_shapes(_Kernel, spec, 16384, kda_rows=2)
        self.assertEqual(shapes["mla_idx_0"], ((128, 32, 4, 256), np.uint8))
        _Kernel.idx_page_shape, _Kernel.idx_pool_multiple = (33, 4, 128), 16
        shapes = model_io.state_shapes(_Kernel, spec, 16384 + 128, kda_rows=2)
        self.assertEqual(shapes["mla_idx_0"], ((144, 33, 4, 128), np.uint8))

    def test_w1_split_blocks(self):
        dsa = mla.W1Split(8, mla.REPLICATED_DSA)
        self.assertEqual(dsa.per, mla.W1Split(8).per)
        self.assertEqual(len(dsa.owners()), mla.REPLICATED_DSA // 128)


class TestK31Weights(unittest.TestCase):
    def test_dsa_mla_layer(self):
        rng = np.random.default_rng(0)
        hidden, layer, ranks = 256, 3, [0, 5]
        topo = Topology(1, 8)
        ck = _Ckpt(_mla_tensors(rng, layer, hidden))
        w = load_mla_layer(ck, layer, topo, ranks, dsa=True)
        w0 = load_mla_layer(ck, layer, topo, ranks, dsa=False)
        hd = 12 * 128
        self.assertEqual(w["mla_w1"].shape, (2, hidden, mla.REPLICATED_DSA + hd))
        pre = f"layers.{layer}.self_attn.indexer."
        wk, wp = ck.get(pre + "wk.weight"), ck.get(pre + "weights_proj.weight")
        wqb = ck.get(pre + "wq_b.weight")
        for i, r in enumerate(ranks):
            a, a0 = w["mla_w1"][i], w0["mla_w1"][i]
            np.testing.assert_array_equal(
                a[:, : mla.REPLICATED], a0[:, : mla.REPLICATED]
            )
            np.testing.assert_array_equal(
                a[:, mla.REPLICATED_DSA :], a0[:, mla.REPLICATED :]
            )
            np.testing.assert_array_equal(a[:, mla.IDX_K : mla.IDX_K + 128], wk.T)
            np.testing.assert_array_equal(a[:, mla.IDX_W : mla.IDX_W + 32], wp.T)
            self.assertFalse(
                np.any(a[:, mla.IDX_W + 32 : mla.REPLICATED_DSA].astype(np.float32))
            )
            lane = r % topo.lanes
            np.testing.assert_array_equal(
                w["mla_idx_qb"][i], wqb[512 * lane : 512 * (lane + 1)].T
            )
        for k in ("mla_qb", "mla_wkc", "mla_wvc", "mla_oproj"):
            np.testing.assert_array_equal(w[k], w0[k], err_msg=k)
        np.testing.assert_array_equal(
            w["mla_idx_knorm_w"][0, 0], ck.get(pre + "k_norm.weight").astype(np.float32)
        )

    def test_expert_block_remap(self):
        d, topo = K3Dims(), Topology(2, 4)
        lay = MoeLayout(d, topo, ckpt_blocks=(1, 0))
        perm = lay.expert_perm()
        half = d.num_experts // 2
        self.assertEqual(perm[:half], list(range(half, d.num_experts)))
        self.assertEqual(perm[half:], list(range(half)))
        self.assertEqual(lay.ckpt_expert(0), half)
        self.assertIsNone(MoeLayout(d, topo).expert_perm())
        with self.assertRaises(ValueError):
            MoeLayout(d, topo, ckpt_blocks=(0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
