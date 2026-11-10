"""Kimi-K3.1 on the K3 programs: the DSA indexer in the model config, its
weight families in the kernels' per-rank layout, and the pre-shard shapes."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs.kimi_k3.kernels import Config
from sglang_tpu.k3_programs.kimi_k3.kernels import load

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FULL = [4 * i for i in range(1, 24)] + [93]  # 1-based, as the HF config
DSA = {
    "dsa_layers": FULL,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}


def _text_config(dsa=None):
    c = {
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "latent_moe_use_norm": True,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "hidden_act": "situ",
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_expert_group": 1,
        "routed_scaling_factor": 1.0,
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "intermediate_size": 33792,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "vocab_size": 163840,
        "attn_res_block_size": 12,
        "rms_norm_eps": 1e-5,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "linear_attn_config": {
            "full_attn_layers": FULL,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    }
    if dsa is not None:
        c["dsa_config"] = dsa
    return {"text_config": c}


def _config(dsa=None):
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(_text_config(dsa)))
        return Config.from_checkpoint(directory)


class TestK31Config(CustomTestCase):
    def test_k3_has_no_indexer_and_k31_has_its_dsa(self):
        k3 = _config()
        self.assertEqual(k3, Config())
        self.assertFalse(k3.dsa)
        k31 = _config(DSA)
        self.assertTrue(k31.dsa)
        self.assertEqual((k31.index_topk, k31.index_heads, k31.index_dim), (2048, 32, 128))
        self.assertEqual(k31.without_dsa(), Config())

    def test_unsupported_dsa_settings_are_refused(self):
        for name, value in (
            ("index_rope_dim", 64),
            ("index_topk_freq", 2),
            ("dsa_layers", FULL[:-1]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _config(dict(DSA, **{name: value}))


class _Checkpoint(load.Checkpoint):
    """``load.Checkpoint`` over in-memory tensors (no shard files)."""

    def __init__(self, tensors, config):
        self.tensors, self.config = tensors, config

    def read(self, key, selection=None):
        value = self.tensors[key]
        return value if selection is None else value[selection]


class TestIndexerFamilies(CustomTestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        base = f"{load.PREFIX}layers.3.self_attn.indexer."
        self.raw = {
            "wq_b": rng.standard_normal((32 * 128, 1536)).astype(np.float32),
            "wk": rng.standard_normal((128, 7168)).astype(np.float32),
            "weights_proj": rng.standard_normal((32, 7168)).astype(np.float32),
            "k_norm.weight": rng.standard_normal(128).astype(np.float32),
            "k_norm.bias": rng.standard_normal(128).astype(np.float32),
        }
        tensors = {
            base + (k if k.startswith("k_norm") else k + ".weight"): v
            for k, v in self.raw.items()
        }
        self.ckpt = _Checkpoint(tensors, _config(DSA))

    def test_rank_layouts(self):
        for rank in (0, 5, 31):
            with self.subTest(rank=rank):
                wqb = self.ckpt.dense("d_wqb", 3, rank)
                np.testing.assert_array_equal(
                    wqb, self.raw["wq_b"][rank * 128 : (rank + 1) * 128].T
                )
                np.testing.assert_array_equal(self.ckpt.dense("d_wk", 3, rank), self.raw["wk"].T)
                proj = self.ckpt.dense("d_wproj", 3, rank)
                self.assertEqual(proj.shape, (7168, 128))
                np.testing.assert_array_equal(proj[:, :32], self.raw["weights_proj"].T)
                self.assertFalse(proj[:, 32:].any())
                norm = self.ckpt.dense("d_knorm", 3, rank).astype(np.float32)
                np.testing.assert_allclose(norm[0], self.raw["k_norm.weight"], rtol=1e-2)
                np.testing.assert_allclose(norm[1], self.raw["k_norm.bias"], rtol=1e-2)

    def test_shapes_and_layers(self):
        shapes = load.selected_local_shapes(93, 7168, dsa=True)
        self.assertEqual(shapes["d_wqb"], (24, 1536, 128))
        self.assertEqual(shapes["d_knorm"], (24, 2, 128))
        self.assertNotIn("d_wqb", load.selected_local_shapes(93, 7168))
        layers = load._family_layers(self.ckpt.config, "d_wk", 93)
        self.assertEqual(layers, list(self.ckpt.config.full_attention))
        for name in ("d_wqb", "d_wk", "d_wproj", "d_knorm"):
            value = self.ckpt.dense(name, 3, 0)
            self.assertEqual(tuple(value.shape), shapes[name][1:])


if __name__ == "__main__":
    unittest.main()


class TestDsaReference(CustomTestCase):
    """The host reference of the DSA selection, including the distributed
    exact top-k the DCP kernels implement."""

    def test_quantization_and_scores(self):
        from sglang_tpu.k3_programs.kimi_k3 import dsa

        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 128)).astype(np.float32) * 3
        q8, exp = dsa.quantize_fp8(x)
        self.assertTrue(np.all(np.abs(q8) <= 448))
        np.testing.assert_allclose(q8 * np.exp2(exp), x, rtol=0.07, atol=1e-6)
        q, w = dsa.index_query(rng.standard_normal(32 * 128), rng.standard_normal(32))
        keys, kexp = dsa.quantize_fp8(rng.standard_normal((50, 128)))
        s = dsa.scores(q, w, keys, kexp)
        ref = np.array([
            sum(max(float(keys[n] @ q[h]), 0.0) * w[h] for h in range(32)) * 2.0 ** kexp[n, 0]
            for n in range(50)
        ])
        np.testing.assert_allclose(s, ref, rtol=1e-4)

    def test_distributed_topk_matches_the_global_selection(self):
        from sglang_tpu.k3_programs.kimi_k3 import dsa

        rng = np.random.default_rng(2)
        cases = [
            ("random", lambda n: rng.standard_normal(n)),
            ("zeros", lambda n: np.where(rng.random(n) < 0.7, 0.0, rng.random(n))),
            ("negative", lambda n: -np.abs(rng.standard_normal(n))),
            ("ties", lambda n: rng.integers(0, 5, n).astype(np.float64)),
        ]
        for name, make in cases:
            for total, topk in ((1500, 2048), (9000, 2048), (40000, 2048), (3000, 64)):
                with self.subTest(case=name, total=total, topk=topk):
                    s = make(total).astype(np.float32)
                    keys = dsa.monotone_key(s)
                    positions = rng.permutation(total)
                    want = dsa.topk_mask(keys, positions, topk)
                    owner = positions // 512 % 32  # DCP32: 512-token chunks round robin
                    parts = [np.nonzero(owner == r)[0] for r in range(32)]
                    masks, rounds = dsa.distributed_topk(
                        [keys[i] for i in parts], [positions[i] for i in parts], topk
                    )
                    got = np.zeros(total, bool)
                    for idx, m in zip(parts, masks):
                        got[idx] = m
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(int(got.sum()), min(total, topk))
                    self.assertLessEqual(rounds, 7)  # 4 key digits + 3 position digits


class TestIndexPools(CustomTestCase):
    """Kimi-K3.1's DSA index-key pools beside the unified pools: one per MLA
    layer, rows aligned with the latent blocks, and in every list the P/D
    transfer and the host tier read."""

    def test_pools_transfer_and_layer_ids(self):
        from types import SimpleNamespace
        from unittest import mock

        import torch

        from sglang_tpu.hooks import k3_program as hooks
        from sglang_tpu.k3_programs import sglang_pools
        from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS

        model = mock.patch(
            "sglang.srt.runtime_context.get_model",
            return_value=SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
        )
        with model:
            state = sglang_pools.K3StatePool(size=4, mamba_layer_ids=KDA, device="cpu")
            kv = sglang_pools.K3UnifiedKVPool(
                size=6 * 16384, page_size=16384, dtype=torch.float8_e4m3fn,
                layer_num=len(MLA), device="cpu", index_head_dim=128,
            )
        self.assertEqual(len(kv.index_pools), len(MLA))
        self.assertEqual(
            tuple(kv.index_pools[0].shape), (kv.plan.num_blocks, 4 * 33, 4, 128)
        )
        self.assertEqual(kv.index_pools[0].dtype, torch.uint8)
        self.assertEqual(len(kv.raw_pools()), len(MLA) + 5 + len(MLA))
        ptrs, lens, items = kv.get_contiguous_buf_infos()
        ids = kv.get_kv_layer_ids()
        self.assertEqual(len(ptrs), len(ids))
        self.assertEqual(set(items[-len(MLA):]), {4 * 33 * 4 * 128})
        self.assertEqual(ids[-len(MLA):], list(range(NUM_LAYERS + 5, NUM_LAYERS + 5 + len(MLA))))
        self.assertEqual(len(set(ids)), len(ids))
        # Kimi-K3 builds none.
        with model:
            sglang_pools.K3StatePool(size=4, mamba_layer_ids=KDA, device="cpu")
            k3 = sglang_pools.K3UnifiedKVPool(
                size=6 * 16384, page_size=16384, dtype=torch.float8_e4m3fn,
                layer_num=len(MLA), device="cpu",
            )
        self.assertEqual(k3.index_pools, [])
        self.assertEqual(len(k3.get_kv_layer_ids()), len(MLA) + 5)
        del hooks, state


class TestIndexCommit(CustomTestCase):
    """New tokens' index keys land on their DCP owner, in the block their
    latents go to, and read back as the reference quantization."""

    def test_records_round_trip_on_their_owners(self):
        from unittest import mock

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        rng = np.random.default_rng(3)
        tokens, group, blocks, layers = 512, 32, 6, 2
        positions = np.array([5, 511, 512 * 33 + 130, 512 * 40 + 511], np.int32)
        keys = rng.standard_normal((len(positions), layers, 128)).astype(np.float32)
        records = dsa_dcp.key_records(jnp.asarray(keys))
        # Local tables: entry e covers global pages e*32 .. e*32+31; block ids 1.. per row.
        tables = np.zeros((len(positions), 4), np.int32)
        for b in range(len(positions)):
            tables[b] = np.arange(1, 5) + 4 * b
        for member in range(group):
            pools = tuple(jnp.zeros((blocks * 4, 4 * 33, 4, 128), jnp.uint8) for _ in range(layers))
            with mock.patch.object(mla_dcp, "owner_index", return_value=member):
                pools = dsa_dcp.commit_index_rows(
                    pools, records, jnp.asarray(tables), jnp.asarray(positions), group=group
                )
            for b, p in enumerate(positions):
                page = p // tokens
                block = tables[b, page // group]
                for m in range(layers):
                    got, gexp = dsa_dcp.read_records(pools[m][block], tokens)
                    t = p % tokens
                    if page % group == member:
                        want, wexp = dsa.quantize_fp8(keys[b, m][None])
                        np.testing.assert_array_equal(np.asarray(got[t]), want[0])
                        self.assertEqual(int(gexp[t, 0]), int(wexp[0, 0]))
                    else:
                        self.assertFalse(np.asarray(pools[m][block]).any())
