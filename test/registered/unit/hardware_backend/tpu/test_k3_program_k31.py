"""Kimi-K3.1 on the K3 programs: the DSA indexer in the model config, its
weight families in the kernels' per-rank layout, and the pre-shard shapes."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs.kimi_k3.kernels import Config, load

register_cpu_ci(est_time=2300, suite="base-a-test-cpu")

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
        self.assertEqual(
            (k31.index_topk, k31.index_heads, k31.index_dim), (2048, 32, 128)
        )
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
                np.testing.assert_array_equal(
                    self.ckpt.dense("d_wk", 3, rank), self.raw["wk"].T
                )
                proj = self.ckpt.dense("d_wproj", 3, rank)
                self.assertEqual(proj.shape, (7168, 128))
                np.testing.assert_array_equal(proj[:, :32], self.raw["weights_proj"].T)
                self.assertFalse(proj[:, 32:].any())
                norm = self.ckpt.dense("d_knorm", 3, rank).astype(np.float32)
                np.testing.assert_allclose(
                    norm[0], self.raw["k_norm.weight"], rtol=1e-2
                )
                np.testing.assert_allclose(norm[1], self.raw["k_norm.bias"], rtol=1e-2)

    def test_shapes_and_layers(self):
        shapes = load.selected_local_shapes(93, 7168, dsa=True)
        self.assertEqual(shapes["d_wqb"], (24, 1536, 128))
        self.assertEqual(shapes["d_knorm"], (24, 2, 128))
        self.assertNotIn("d_wqb", load.selected_local_shapes(93, 7168))
        layers = load._family_layers(self.ckpt.config, "d_wk", 93)
        self.assertEqual(layers, list(self.ckpt.config.full_attention))
        # Every pre-shard family comes from the checkpoint (the decode's d_wks /
        # d_wpt are derived at load: rank_loader.decode_indexer_families).
        names = [n for n in shapes if n.startswith("d_")]
        self.assertEqual(sorted(names), ["d_knorm", "d_wk", "d_wproj", "d_wqb"])
        for name in names:
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
        ref = np.array(
            [
                sum(max(float(keys[n] @ q[h]), 0.0) * w[h] for h in range(32))
                * 2.0 ** kexp[n, 0]
                for n in range(50)
            ]
        )
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
                size=6 * 16384,
                page_size=16384,
                dtype=torch.float8_e4m3fn,
                layer_num=len(MLA),
                device="cpu",
                index_head_dim=128,
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
        self.assertEqual(set(items[-len(MLA) :]), {4 * 33 * 4 * 128})
        self.assertEqual(
            ids[-len(MLA) :], list(range(NUM_LAYERS + 5, NUM_LAYERS + 5 + len(MLA)))
        )
        self.assertEqual(len(set(ids)), len(ids))
        # Kimi-K3 builds none.
        with model:
            sglang_pools.K3StatePool(size=4, mamba_layer_ids=KDA, device="cpu")
            k3 = sglang_pools.K3UnifiedKVPool(
                size=6 * 16384,
                page_size=16384,
                dtype=torch.float8_e4m3fn,
                layer_num=len(MLA),
                device="cpu",
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
            pools = tuple(
                jnp.zeros((blocks * 4, 4 * 33, 4, 128), jnp.uint8)
                for _ in range(layers)
            )
            with mock.patch.object(mla_dcp, "owner_index", return_value=member):
                pools = dsa_dcp.commit_index_rows(
                    pools,
                    records,
                    jnp.asarray(tables),
                    jnp.asarray(positions),
                    group=group,
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


class TestFusedDsaCommit(CustomTestCase):
    """The fused K3.1 epilogue commit (``paged._commit_dsa_rows_fused``, Pallas
    interpret mode) writes the bytes of the latent commit kernel and the XLA
    index scatters: runs inside a sub-page and across sub-pages and pages,
    other ranks' rows, two sequences, full waves, out-of-range and negative
    blocks; repeated null-block targets land in row order."""

    def _commit(self, positions, tables, members, layers=3, blocks=6, cache="fp8"):
        import functools
        from unittest import mock

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp, paged
        from sglang_tpu.k3_programs.kimi_k3.layout import (
            cache_layout,
            index_block_shape,
        )

        layout = cache_layout(cache)
        rng = np.random.default_rng(len(positions) + int(np.sum(positions)) % 997)
        batch, packing = len(positions), layout.block_shape[1]
        page_tokens = layout.tokens_per_block
        dtype, bits = (
            (jnp.bfloat16, np.uint16)
            if layout.dtype == "bfloat16"
            else (jnp.float8_e4m3fn, np.uint8)
        )
        pool_map = tuple(int(i) for i in rng.permutation(layers + 1)[:layers])
        # Random bytes; finite BF16 values (interpret-mode BF16 selects
        # canonicalize NaN payloads, which the TPU's do not).
        lat = [
            rng.integers(0, 256, (blocks, *layout.block_shape), dtype=np.uint8)
            if bits is np.uint8
            else np.asarray(
                jnp.asarray(
                    rng.standard_normal((blocks, *layout.block_shape)), jnp.bfloat16
                )
            ).view(np.uint16)
            for _ in range(layers + 1)
        ]
        idx = [
            rng.integers(0, 256, (blocks, *index_block_shape(layout)), dtype=np.uint8)
            for _ in range(layers)
        ]
        latents = jnp.asarray(rng.standard_normal((batch, layers, 640)), jnp.bfloat16)
        records = rng.integers(0, 256, (batch, layers, 2, 128)).astype(np.int32)
        cast = lambda x: jax.lax.bitcast_convert_type(jnp.asarray(x), dtype)
        raw = lambda x: np.asarray(
            jax.lax.bitcast_convert_type(x, bits) if x.dtype == dtype else x
        )
        row_bytes = raw(latents.astype(dtype))
        tables_, positions_ = jnp.asarray(tables, jnp.int32), jnp.asarray(positions)
        interpret = functools.partial(
            pl.pallas_call, interpret=pltpu.InterpretParams()
        )
        for member in members:
            want_lat = [x.copy() for x in lat]
            want_idx = [x.copy() for x in idx]
            for b, p in enumerate(positions):  # row order
                page, t = p // page_tokens, p % page_tokens
                block = tables[b][page // 32]
                block = block + blocks if block < 0 else block
                if page % 32 != member or not 0 <= block < blocks:
                    continue
                (key_row, sub), (scale_row, lane) = dsa_dcp.record_place(t)
                for m in range(layers):
                    want_lat[pool_map[m]][block, t // packing, t % packing] = (
                        row_bytes[b, m]
                    )
                    want_idx[m][block, key_row, sub] = records[b, m, 0]
                    want_idx[m][block, scale_row, 0, lane] = records[b, m, 1, 0]
            with (
                mock.patch.object(mla_dcp, "owner_index", return_value=member),
                mock.patch.object(paged.pl, "pallas_call", interpret),
            ):
                pools = tuple(cast(x) for x in lat)
                index_pools = tuple(jnp.asarray(x) for x in idx)
                fused = paged._commit_dsa_rows_fused(
                    pools,
                    index_pools,
                    latents,
                    jnp.asarray(records),
                    tables_,
                    positions_,
                    pool_map,
                )
                old = paged.commit_latent_rows(
                    pools, latents, tables_, positions_, pool_map
                ) + dsa_dcp.commit_index_rows(
                    index_pools,
                    jnp.concatenate(
                        (records[:, :, 0], records[:, :, 1, :1]), axis=-1
                    ).astype(jnp.uint8),
                    tables_,
                    positions_,
                    group=32,
                )
            for got, base, want in zip(fused, old, want_lat + want_idx, strict=True):
                np.testing.assert_array_equal(raw(got), want)
                np.testing.assert_array_equal(raw(got), raw(base))

    @staticmethod
    def _interpretable():
        try:
            from jax.experimental.pallas import tpu as pltpu

            pltpu.InterpretParams()
        except (ImportError, AttributeError):
            return False
        return True

    def test_verify_runs(self):
        """One sequence's eight verify rows: inside a sub-page, across a
        sub-page, across a page (the next page's owner commits the rest)."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        tables = np.tile(np.array([[3, 1, 4, 2]], np.int32), (8, 1))
        for start in (0, 3, 29, 116, 125, 508):
            with self.subTest(start=start):
                base = 512 * 37 + start  # page 37: member 5's local entry 1
                positions = base + np.arange(8, dtype=np.int32)
                members = (5, 6, 7) if start == 508 else (5,)
                self._commit(positions, tables, members=members)

    def test_two_sequences_and_full_waves(self):
        """B2x7 (two sequences at one context: one owner, two blocks), then
        sixteen rows on sixteen blocks of one owner (two full waves)."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        positions = np.tile(512 * 70 + 250 + np.arange(8, dtype=np.int32), 2)
        tables = np.repeat(np.array([[0, 0, 2, 0], [0, 0, 5, 0]], np.int32), 8, axis=0)
        self._commit(positions, tables, members=(6, 7))
        positions = (512 * 32 * np.arange(16) + 512 * 3 + 7).astype(np.int32)
        tables = np.zeros((16, 16), np.int32)
        tables[np.arange(16), np.arange(16)] = np.arange(16)
        self._commit(positions, tables, members=(3,), blocks=17)

    def test_plain_rows_padding_and_bounds(self):
        """Plain decode rows: rows sharing blocks out of order, padding rows
        on null block 0, an out-of-range block, a negative block next to its
        normalized twin; then the BF16 layout's runs."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        positions = np.array(
            [512 * 33 + 5, 0, 512 * 33 + 9, 0, 512 * 65 + 130, 512 + 511]
            + [512 * 65 + 131, 512 * 33 + 6],
            np.int32,
        )
        tables = np.zeros((8, 4), np.int32)
        tables[[0, 2, 7], 1] = 4
        tables[[4, 6], 2] = -1, 5
        tables[5, 0] = 9
        self._commit(positions, tables, members=(0, 1))
        tables = np.tile(np.array([[3, 1, 4, 2]], np.int32), (8, 1))
        for start in (2, 124, 252):
            with self.subTest(cache="bf16", start=start):
                positions = 256 * 37 + start + np.arange(8, dtype=np.int32)
                self._commit(positions, tables, members=(5,), cache="bfloat16")

    def test_mla_layers_and_dispatch(self):
        """All 24 MLA layers; the decode step takes the fused commit only
        with ``dsa_fused_commit``."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        from unittest import mock

        import jax
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.config import KernelConfig
        from sglang_tpu.k3_programs.kimi_k3.kernels import paged

        positions = 512 * 40 + 61 + np.arange(8, dtype=np.int32)
        tables = np.tile(np.array([[0, 2, 0]], np.int32), (8, 1))
        self._commit(positions, tables, members=(8,), layers=24, blocks=3)
        pools = (jax.ShapeDtypeStruct((4, 128, 4, 640), jnp.float8_e4m3fn),) * 24
        index_pools = (jax.ShapeDtypeStruct((4, 132, 4, 128), jnp.uint8),) * 24
        rows = jax.ShapeDtypeStruct((8, 24, 640), jnp.bfloat16)
        records = jax.ShapeDtypeStruct((8, 24, 2, 128), jnp.int32)
        tables = jax.ShapeDtypeStruct((8, 64), jnp.int32)
        for enabled in (False, True):
            with mock.patch.object(
                paged, "_config", return_value=KernelConfig(dsa_fused_commit=enabled)
            ):
                self.assertEqual(
                    paged._dsa_commit_fusable(
                        pools, index_pools, rows, records, tables, tuple(range(24))
                    ),
                    enabled,
                )


class TestIndexSelect(CustomTestCase):
    """The decode selection kernel (``dsa_dcp.select_call``) in Pallas
    interpret mode on a 4-rank DCP group of host devices: every rank's keys
    match the reference scores, the thresholds select exactly the top-k
    (lowest positions among tied keys) and agree on every rank, in-flight
    tokens are scored from their records (the pool holds garbage there), and
    the own-token flags follow the selection."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T = 4, 512
        def case(firsts, S, topk, ties, seed):
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            seq_keys = []
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                if ties:  # five distinct keys: ties at the threshold
                    raw = rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]
                k8, kexp = dsa.quantize_fp8(raw)
                seq_keys.append((k8, kexp))
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            tied = 0
            for b in range(B):
                L = positions[b] + 1
                k8, kexp = seq_keys[b // S]
                t = np.arange(L)
                member = (t // T) % G
                kk = keys[member, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                ref = dsa.monotone_key(dsa.scores(q8[b], w[b], k8[:L], kexp[:L]).astype(np.float32))
                unkey = lambda k: (k ^ ((k >> 31) & 0x7FFFFFFF)).astype(np.int32).view(np.float32)
                err = np.abs(unkey(kk) - unkey(ref)).max() / np.abs(unkey(ref)).max()
                assert err < 1e-5, (b, err)
                for name, v in (("tau", tau), ("p_tie", ptie), ("own", own)):
                    assert len(set(v[:, b])) == 1, (name, b, v[:, b])
                got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])
                np.testing.assert_array_equal(got, dsa.topk_mask(kk, t, topk))
                assert bool(own[0, b]) == bool(got[-1]), b
                tied += int(L > topk and np.sum(kk == tau[0, b]) > 1)
            return tied

        case([300, 2500], 1, 256, False, 0)
        assert case([1500, 2600], 2, 300, True, 1) > 0  # position tie-breaks exercised
        print("OK")
    """

    def test_selection_matches_the_reference_on_a_dcp_group(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestIndexSelectBins(CustomTestCase):
    """Bin selection (``KernelConfig.dsa_select_rounds`` < 8): the kernel's
    threshold is the lower bound of the ``4 * rounds``-bit key prefix bin that
    holds the ``topk``-th key (every key at or above it selected), on every
    rank alike, whatever digit widths its exchange rounds use
    (``dsa_dcp.digit_widths``)."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        ROUNDS = int(os.environ["ROUNDS"])
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        kc.install(kc.KernelConfig(dsa_select_rounds=ROUNDS))
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T, INT_MIN = 4, 512, -(2**31)
        def case(firsts, S, topk, seed):
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            print("widths", dsa_dcp.digit_widths(B, ROUNDS))
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                k8, kexp = dsa.quantize_fp8(raw)
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            extra = []
            for b in range(B):
                L = positions[b] + 1
                t = np.arange(L)
                kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                for name, v in (("tau", tau), ("p_tie", ptie), ("own", own)):
                    assert len(set(v[:, b])) == 1, (name, b, v[:, b])
                if L <= topk:
                    want_tau = INT_MIN
                    want = kk > INT_MIN
                else:
                    kth = np.sort(kk)[::-1][topk - 1]
                    drop = 32 - 4 * ROUNDS
                    unsigned = (int(kth) ^ INT_MIN) & 0xFFFFFFFF
                    want_tau = ((unsigned >> drop) << drop) ^ (1 << 31)
                    want_tau = want_tau - (1 << 32) if want_tau >= (1 << 31) else want_tau
                    want = kk >= want_tau
                    extra.append(int(want.sum()) - topk)
                assert tau[0, b] == want_tau, (b, tau[0, b], want_tau)
                got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])
                np.testing.assert_array_equal(got, want)
                assert np.all(got[dsa.topk_mask(kk, t, topk)]), b  # a superset of the top-k
                assert bool(own[0, b]) == bool(got[-1]), b
            print("superset extra tokens", extra)

        case([300, 2500], 1, 256, 0)
        case([1500, 2600], 2, 300, 1)
        case([3100], 8, 400, 2)
        print("OK")
    """

    def _run(self, rounds):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), ROUNDS=str(rounds))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )

    def test_five_rounds(self):
        self._run(5)

    def test_four_rounds(self):
        self._run(4)


class TestIndexSelectGatherFinish(CustomTestCase):
    """``KernelConfig.dsa_gather_finish``: once every row's threshold bin is
    small the ranks gather its keys and finish the rounds locally -- the same
    thresholds as the distributed rounds, with and without position ties."""

    @staticmethod
    def script(knobs=""):
        import textwrap

        script = textwrap.dedent(TestIndexSelect.SCRIPT)
        script = script.replace(
            "from sglang_tpu.k3_programs.kimi_k3 import dsa\n",
            "from sglang_tpu.k3_programs.kimi_k3 import config as kc\n"
            f"kc.install(kc.KernelConfig(dsa_gather_finish=True{knobs}))\n"
            "from sglang_tpu.k3_programs.kimi_k3 import dsa\n",
        )
        script = script.replace(
            "from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp\n",
            "from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp\n"
            "from jax.experimental import pallas as pl\n"
            "_finish = dsa_dcp.IndexSelect._gather_finish\n"
            "def _traced(self, first_round):\n"
            "    pl.debug_print('GATHER at round {}', first_round)\n"
            "    return _finish(self, first_round)\n"
            "dsa_dcp.IndexSelect._gather_finish = _traced\n",
        )
        script = script.replace(
            "def case(firsts, S, topk, ties, seed):",
            "def case(firsts, S, topk, ties, seed, distinct=5):",
        )
        script = script.replace(
            "rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]",
            "rng.standard_normal((distinct, 128)).astype(np.float32)[rng.integers(0, distinct, L)]",
        )
        tail = script[script.index("case([300, 2500], 1, 256, False, 0)") :]
        return script.replace(
            tail,
            "\n".join(
                [
                    "case([300, 2500], 1, 256, False, 0)",
                    "assert case([2600, 3100], 2, 256, True, 2, distinct=350) > 0  # ties inside the gathered bin",
                    "case([3500], 4, 512, False, 3)",
                    'print("OK")',
                    "",
                ]
            ),
        )

    def test_gather_finish_matches_the_reference(self):
        self._run()

    def test_gather_finish_with_odd_stride(self):
        # KernelConfig.dsa_odd_stride: the resident scan reads the padded rows.
        self._run(", dsa_odd_stride=True")

    def _run(self, knobs=""):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.script(knobs)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])
        self.assertGreaterEqual(
            result.stdout.count("GATHER at round"), 3 * 4, result.stdout[-2000:]
        )


class TestIndexSelectScoringKnobs(CustomTestCase):
    """The scoring loop's KernelConfig knobs (``dsa_stage_slots``,
    ``dsa_page_slots``, ``dsa_resident_keys``, ``dsa_patch_loop``) leave the
    selection bit for bit unchanged: ``select_call`` in interpret mode on a
    4-rank group, once with the defaults and once per knob set, must give the
    same thresholds, own-token flags and keys of every row's tokens (each run
    also checks the reference). The cases cover in-flight tokens across a
    page (and rank) boundary, eight one-row sequences, two two-row sequences,
    keys over both resident chunks (the second one clamped and not), and
    exact rounds with position ties."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        knobs = dict(kv.split("=") for kv in os.environ["KNOBS"].split(",") if kv)
        kc.install(kc.KernelConfig().replace(**knobs))
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T, INT_MIN = 4, 512, -(2**31)
        ROUNDS = kc.active().dsa_select_rounds
        outputs = {}

        def case(name, firsts, S, topk, seed, chunk=None, ties=False):
            if chunk:  # narrower count chunks: the keys span both resident chunks
                dsa_dcp.WIDE_CHUNK_ROWS = chunk
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            print(name, "resident", dsa_dcp.resident_keys(B, dsa_dcp.key_rows_for(pages, T), S))
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                if ties:  # five distinct keys: ties at the threshold
                    raw = rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]
                k8, kexp = dsa.quantize_fp8(raw)
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            row_keys = []
            for b in range(B):
                L = positions[b] + 1
                t = np.arange(L)
                kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                row_keys.append(kk)
                for v in (tau, ptie, own):
                    assert len(set(v[:, b])) == 1, (name, b)
                if ROUNDS == 8:
                    want = dsa.topk_mask(kk, t, topk)
                elif L <= topk:
                    want = kk > INT_MIN
                else:
                    kth = np.sort(kk)[::-1][topk - 1]
                    drop = 32 - 4 * ROUNDS
                    unsigned = (int(kth) ^ INT_MIN) & 0xFFFFFFFF
                    bound = ((unsigned >> drop) << drop) ^ (1 << 31)
                    want = kk >= (bound - (1 << 32) if bound >= (1 << 31) else bound)
                np.testing.assert_array_equal(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b]), want)
                assert bool(own[0, b]) == bool(want[-1]), (name, b)
            for key, v in (("tau", tau), ("ptie", ptie), ("own", own), ("keys", np.concatenate(row_keys))):
                outputs[f"{name}_{key}"] = v

        if ROUNDS == 8:
            case("ties", [1500, 2600], 2, 300, 1, ties=True)
            case("ties8", [3069], 8, 400, 6, ties=True)
        else:
            case("b8s8", [6 * T - 3], 8, 400, 2)  # in-flight tokens on two pages (ranks)
            case("b8s1", [300, 2500, 1100, 3600, 700, 2049, 1500, 3000], 1, 256, 3)
            case("b4s2", [1500, 2600], 2, 300, 1)
            case("two24", [9500], 8, 300, 4, chunk=16)  # the second chunk clamped
            case("two32", [13000], 8, 300, 5, chunk=16)
        np.savez(os.environ["OUT"], **outputs)
        print("OK")
    """

    # Rings of other depths than the defaults (not powers of two), the patch
    # loop; then the resident keys (no stage ring) with an eight-page ring.
    KNOBS = "dsa_stage_slots=3,dsa_page_slots=6,dsa_patch_loop=1"
    RESIDENT = "dsa_page_slots=8,dsa_resident_keys=1,dsa_patch_loop=1"
    # dsa_odd_stride: the count buffer's rows an odd number of tiles apart
    # (these cases' 16-row chunks: 24 rows), copied in and resident.
    ODD = "dsa_odd_stride=1"

    def _runs(self, rounds, knob_sets):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            for i, knobs in enumerate(("", *knob_sets)):
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(sys.path),
                    KNOBS=",".join(filter(None, (f"dsa_select_rounds={rounds}", knobs))),
                    OUT=f"{tmp}/run{i}.npz",
                )
                result = subprocess.run(
                    [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1500,
                )
                self.assertIn(
                    "OK",
                    result.stdout,
                    f"{knobs}: {result.stdout[-2000:]}{result.stderr[-3000:]}",
                )
            base = np.load(f"{tmp}/run0.npz")
            for i, knobs in enumerate(knob_sets, start=1):
                other = np.load(f"{tmp}/run{i}.npz")
                self.assertEqual(sorted(base.files), sorted(other.files))
                for name in base.files:
                    np.testing.assert_array_equal(
                        base[name], other[name], err_msg=f"{knobs}: {name}"
                    )

    def test_bin_selection_is_unchanged(self):
        self._runs(
            5,
            (
                self.KNOBS,
                self.RESIDENT,
                f"{self.KNOBS},{self.ODD}",
                f"{self.RESIDENT},{self.ODD}",
            ),
        )

    def test_exact_selection_with_ties_is_unchanged(self):
        self._runs(8, (self.RESIDENT, f"{self.RESIDENT},{self.ODD}"))


class TestSelectedAttention(CustomTestCase):
    """The DCP attention with a DSA selection (``mla_dcp.DcpSelection``) in
    Pallas interpret mode on a 4-rank group: each row attends exactly its
    selected tokens (pool tokens, cached verify tokens on their owners, its
    own token through the merge), at the FP8 attention's accuracy, and far
    from dense attention. Interpret mode has no ref bitcasts, so the FP8 page
    staging (``_pages``, ``_clean_step``, ``_cache_verify_row``) is replaced
    by value-level equivalents."""

    SCRIPT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        G, T, H, FP8 = 4, 512, 3, jnp.float8_e4m3fn
        mla_dcp.GROUPS = (G, 8, 16, 32)

        def tokens(x):
            return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                    + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

        def _pages(self, slot):
            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                return x.reshape(x.shape[0] * x.shape[1], x.shape[2])[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            keep = tokens(x) < cut
            if pad:  # compact steps (pad=False) leave the stale lanes 576:640 to their gathered tiles
                keep = keep & (lane < 576)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _cache_verify_row(self, slot, local_step, row):
            position = self.position_ref[row]
            page = position // self.page_tokens
            local_page = page // self.group - local_step * self.span

            @pl.when((page % self.group == self.owner) & (local_page >= 0) & (local_page < self.span))
            def write():
                if mla_dcp.chunked(self.batch):
                    chunk = mla_dcp.chunk_layout(self.batch, self.heads)[0]
                    latent = self.ws.new_rows[row // chunk, self.slots * (row % chunk)]
                else:
                    latent = self.ws.new_rows[self.slots * row]
                x = self.ws.pages[slot]
                offset = local_page * self.page_tokens + position % self.page_tokens
                self.ws.pages[slot] = jnp.where(tokens(x) == offset, latent.astype(x.dtype)[None, None, :], x)

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            x = self.ws.pages[slot]
            self.ws.pages[slot] = jnp.zeros(x.shape, x.dtype)

        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes
        mla_dcp.DcpAttention._cache_verify_row = _cache_verify_row

        # Every step activity entry the attention reads, recorded (the reads of
        # KernelConfig.dsa_live_activity's entries: dsa_dcp.live_activity_steps).
        READS = []
        _active = mla_dcp.DcpAttention._active

        def _recorded_active(self, row, local_step):
            if READS and self.selection is not None and self.selection.active is not None:
                READS[0][row * self.selection.steps + local_step] = jnp.int32(1)
            return _active(self, row, local_step)

        mla_dcp.DcpAttention._active = _recorded_active

        import dataclasses
        base_config = mla_dcp._config()

        def case(firsts, S, topk, seed, grouped=False, activity="exact", poison=False, compact=0, expect=None,
                 window=0, wide=0):
            config = dataclasses.replace(base_config, dsa_compact_tokens=compact, dsa_compact_window=window)
            mla_dcp._config = lambda: config
            mla_dcp._grouped_spec = ((lambda b, g, h, rows: rows in (4, 8) and b % 4 == 0) if grouped
                                     else (lambda *a: False))
            rng = np.random.default_rng(seed)
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            B = len(positions)
            persistent = mla_dcp.persistent_scratch(B, G, cache_dtype=FP8, sequence_rows=S)
            scoped = mla_dcp.scoped_scratch(B, group=G, cache_dtype=FP8, page_tokens=T, sequence_rows=S)
            selection = mla_dcp.selection_scratch(B, page_tokens=T, sequence_rows=S)
            span = selection[0].shape[2] * 128 // T
            pages = -(-(max(-(-(int(p) + T) // T) // G for p in positions) + 1) // span) * span
            pages += wide * span  # a table wider than the context (serving's 1M-context tables)
            key_rows = dsa_dcp.key_rows_for(pages, T, span)
            lat = (rng.standard_normal((len(firsts), pages * G * T, 640)) * 0.5).astype(np.float32)
            lat[..., 576:] = 0
            lat = lat.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
            pool = rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, 640)).astype(ml_dtypes.float8_e4m3fn)
            tables = np.zeros((G, B, pages), np.int32)
            keys = np.full((G, B, key_rows, 128), -2**31, np.int32)
            tok = rng.integers(-2**31 + 1, 2**31 - 1, (B, pages * G * T), dtype=np.int64)
            for b in range(B):
                s, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s * pages + np.arange(pages)
                for t in range(positions[b]):
                    page = t // T; g, j = page % G, page // G
                    if t < p0:
                        pool[g, tables[g, b, j], (t % T) // 4, t % 4] = lat[s, t].astype(ml_dtypes.float8_e4m3fn)
                    keys[g, b, j * (T // 128) + (t % T) // 128, t % 128] = tok[b, t]
            tau = np.zeros(B, np.int32); ptie = np.zeros(B, np.int32); own = np.zeros(B, np.int32); masks = []
            for b in range(B):
                L = positions[b] + 1
                kk = tok[b, :L]
                m = dsa.topk_mask(kk, np.arange(L), topk)
                order = np.lexsort((np.arange(L), -kk))
                tau[b], ptie[b] = (-2**31, -1) if L <= topk else (kk[order[topk - 1]], 2**31 - 1)
                own[b] = int(m[-1]); masks.append(m)
            step_rows = selection[0].shape[2]
            steps = key_rows // step_rows
            active = np.zeros((G, B * steps), np.int32)
            for b in range(B):
                first = b - b % S
                for t in np.flatnonzero(masks[b]):
                    page = t // T; g, j = page % G, page // G
                    active[g, first * steps + (j * (T // 128) + (t % T) // 128) // step_rows] = 1
            if activity == "none":
                active[:] = 0
            if expect is not None:  # which compact steps the case exercises (per sequence, rank, step)
                counts = []
                for s in range(len(firsts)):
                    union = np.zeros(pages * G * T, bool)
                    for r in range(S):
                        union[:positions[s * S + r] + 1] |= masks[s * S + r]
                        union[positions[s * S + r]] = False  # its own token goes through the merge
                    for g in range(G):
                        for j in range(pages // span):
                            toks = [((j * span + k) * G + g) * T + t for k in range(span) for t in range(T)]
                            counts.append(int(union[toks].sum()))
                kinds = {"compact" if 0 < c <= compact else "dense" if c > compact else "empty" for c in counts}
                assert set(expect) <= kinds, (expect, sorted(counts))
            q = (rng.standard_normal((G, B, H, 640)) * 0.05).astype(np.float32)
            q[..., 576:] = 0
            new = np.broadcast_to(lat[np.arange(B) // S, positions], (G, B, 640)).astype(np.float32)
            n_p, n_s = len(persistent), len(scoped)

            def kernel(pos, table, tau_r, ptie_r, own_r, act_r, q_r, new_r, pool_r, keys_r, out_r, reads_r,
                       *scratch):
                for i in range(B * steps):
                    reads_r[i] = jnp.int32(0)
                READS[:] = [reads_r]
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                sel = mla_dcp.DcpSelection(keys_r, tau_r, ptie_r, own_r, *scratch[n_p + n_s:],
                                           active=act_r, steps=steps)
                att = mla_dcp.DcpAttention(cache, ws, pos, B, selection=sel, sequence_rows=S,
                                           cache_verify_rows=S > 1)
                if poison:  # another scoped region's bytes: NaN where no step loads a page
                    ws.pages[...] = jnp.full(ws.pages.shape, jnp.nan, ws.pages.dtype)
                att.prime()
                out_r[...] = att.attend(q_r[...], new_r[...], 0)
                READS[:] = []

            def body(pos, table, tau_, ptie_, own_, act_, q_, new_, pool_, keys_):
                vmem = pltpu.VMEM
                out, reads = pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                                                               jax.ShapeDtypeStruct((B * steps,), jnp.int32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=6,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,
                        out_specs=(pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=pltpu.SMEM)),
                        scratch_shapes=[*persistent, *scoped, *selection]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], tau_, ptie_, own_, act_[0], q_[0], new_[0],
                                                       pool_[0], keys_[0])
                return out[None], reads[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P()) + (P("tp"),) * 5,
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, reads = (np.asarray(x) for x in f(positions, tables.reshape(G, -1), tau, ptie, own, active, q, new,
                                                   pool, keys))
            # The attention reads only each sequence's first row's entries below
            # dsa_dcp.live_activity_steps (the selection's longest row's steps and one more).
            for g in range(G):
                longest = max(max(-(-(int(p) + 1) // T) - g + G - 1, 0) // G for p in positions)
                live = min(-(-longest * (T // 128) // step_rows) + 1, steps)
                domain = np.zeros(B * steps, bool)
                for first in range(0, B, S):
                    domain[first * steps:first * steps + live] = True
                assert not reads[g][~domain].any(), (g, np.flatnonzero(reads[g]), live, steps)
            if activity == "none":  # no step is read: only the rows' own tokens (where selected) remain
                L = positions[-1] + 1
                vals = lat[(B - 1) // S, :L]
                s_ = vals @ q[0, B - 1, 0]
                p = np.exp(np.where(masks[-1], s_, -np.inf) - s_[masks[-1]].max()); p /= p.sum()
                ref = p @ vals[:, :512]
                assert np.abs(out[0, B - 1, 0] - ref).max() / np.abs(ref).max() > 0.3
                return
            for b in range(B):
                L = positions[b] + 1
                vals = lat[b // S, :L]
                for d in range(G):
                    for j in range(H):
                        s_ = vals @ q[d, b, j]
                        errs = []
                        for m in (masks[b], np.ones(L, bool)):
                            p = np.exp(np.where(m, s_, -np.inf) - s_[m].max()); p /= p.sum()
                            ref = p @ vals[:, :512]
                            errs.append(np.abs(out[d, b, j] - ref).max() / np.abs(ref).max())
                        assert errs[0] < 6e-2, (b, d, j, errs)
                        assert L <= topk or errs[1] > 0.3, (b, d, j, errs)  # the mask matters

        case([700, 2000], 1, 64, 0)
        case([1500, 3000], 4, 200, 1, grouped=True)
        case([2047, 600, 64, 3000], 2, 128, 2)
        case([700, 5000], 1, 64, 3, activity="none")
        case([700, 9000], 1, 8, 4, poison=True)  # skipped steps on never-loaded slots stay finite
        # Compact verify steps (KernelConfig.dsa_compact_tokens): few selected tokens a step attend
        # only those, gathered; busier steps fall back to the dense masked step.
        case([1500, 3000], 4, 24, 5, grouped=True, compact=64, expect=("compact",))
        case([1500, 3000], 4, 200, 1, grouped=True, compact=128, expect=("compact", "dense"))
        case([900, 2600], 8, 40, 6, grouped=True, compact=64, expect=("compact",))
        # Windowed compact steps (KernelConfig.dsa_compact_window): gathered tokens of several steps fold
        # once; a small window flushes when the next step's slots would not fit.
        case([1500, 3000], 4, 200, 1, grouped=True, compact=128, expect=("compact", "dense"), window=512)
        case([900, 2600], 8, 40, 6, grouped=True, compact=64, expect=("compact",), window=64)
        case([700, 9000], 4, 8, 4, grouped=True, compact=64, poison=True, window=256)
        # Tables wider than the contexts: the reads stay within the live steps.
        case([700, 2000], 1, 64, 0, wide=3)
        case([1500, 3000], 4, 200, 1, grouped=True, compact=128, expect=("compact", "dense"), window=512,
             wide=3)
        case([3069], 8, 400, 7, grouped=True, compact=128, window=512, wide=3)
        print("OK")
    """

    def test_rows_attend_their_selected_tokens(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestPairCombine(CustomTestCase):
    """``KernelConfig.dcp_pair_combine`` (the two-stage DCP combine) in Pallas
    interpret mode on a 4-rank group whose chips are ranks 0/1 and 2/3, with
    a DSA selection: ``attend`` and ``send_queries`` + ``attend(queries_sent=
    True)`` agree bitwise, and every output is within one BF16 rounding of the
    one-stage combine's: |pair - one-stage| <= 2^-8 sum_i softmax(lse)_i |o_i|
    over the owner's received partials. A rank whose mate holds no selected
    tokens passes its partial through unchanged (the skewed case is bitwise
    the one-stage result)."""

    SCRIPT = """
        import dataclasses, os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        G, T, H, FP8 = 4, 512, 3, jnp.float8_e4m3fn
        mla_dcp.GROUPS = (G, 8, 16, 32)

        def tokens(x):
            return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                    + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

        def _pages(self, slot):  # interpret mode has no ref bitcasts (see TestSelectedAttention)
            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                return x.reshape(x.shape[0] * x.shape[1], x.shape[2])[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _cache_verify_row(self, slot, local_step, row):
            position = self.position_ref[row]
            page = position // self.page_tokens
            local_page = page // self.group - local_step * self.span

            @pl.when((page % self.group == self.owner) & (local_page >= 0) & (local_page < self.span))
            def write():
                latent = self.ws.new_rows[self.slots * row]
                x = self.ws.pages[slot]
                offset = local_page * self.page_tokens + position % self.page_tokens
                self.ws.pages[slot] = jnp.where(tokens(x) == offset, latent.astype(x.dtype)[None, None, :], x)

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes
        mla_dcp.DcpAttention._cache_verify_row = _cache_verify_row
        base_config = mla_dcp._config()

        def run(firsts, S, topk, seed, pair, early, compact=0, window=0, favor=None):
            config = dataclasses.replace(base_config, dsa_compact_tokens=compact, dsa_compact_window=window,
                                         dcp_pair_combine=pair)
            mla_dcp._config = lambda: config
            mla_dcp._grouped_spec = lambda b, g, h, rows: rows in (4, 8) and b % 4 == 0
            rng = np.random.default_rng(seed)
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            B = len(positions)
            persistent = mla_dcp.persistent_scratch(B, G, cache_dtype=FP8, sequence_rows=S)
            scoped = mla_dcp.scoped_scratch(B, group=G, cache_dtype=FP8, page_tokens=T, sequence_rows=S)
            selection = mla_dcp.selection_scratch(B, page_tokens=T, sequence_rows=S)
            span = selection[0].shape[2] * 128 // T
            pages = -(-(max(-(-(int(p) + T) // T) // G for p in positions) + 1) // span) * span
            key_rows = dsa_dcp.key_rows_for(pages, T, span)
            lat = (rng.standard_normal((len(firsts), pages * G * T, 640)) * 0.5).astype(np.float32)
            lat[..., 576:] = 0
            lat = lat.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
            pool = rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, 640)).astype(ml_dtypes.float8_e4m3fn)
            tables = np.zeros((G, B, pages), np.int32)
            keys = np.full((G, B, key_rows, 128), -2**31, np.int32)
            tok = rng.integers(-2**31 + 1, 2**31 - 1, (B, pages * G * T), dtype=np.int64)
            if favor is not None:  # rank `favor`'s tokens hold the top keys: only it attends pages
                hot = (np.arange(pages * G * T) // T) % G == favor
                tok[:, hot] = 2**31 - 2 - rng.integers(0, 2**20, (B, int(hot.sum())))
            for b in range(B):
                s, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s * pages + np.arange(pages)
                for t in range(positions[b]):
                    page = t // T; g, j = page % G, page // G
                    if t < p0:
                        pool[g, tables[g, b, j], (t % T) // 4, t % 4] = lat[s, t].astype(ml_dtypes.float8_e4m3fn)
                    keys[g, b, j * (T // 128) + (t % T) // 128, t % 128] = tok[b, t]
            tau = np.zeros(B, np.int32); ptie = np.zeros(B, np.int32); own = np.zeros(B, np.int32)
            steps = key_rows // selection[0].shape[2]
            active = np.zeros((G, B * steps), np.int32)
            for b in range(B):
                L = positions[b] + 1
                kk = tok[b, :L]
                m = dsa.topk_mask(kk, np.arange(L), topk)
                order = np.lexsort((np.arange(L), -kk))
                tau[b], ptie[b] = (-2**31, -1) if L <= topk else (kk[order[topk - 1]], 2**31 - 1)
                own[b] = int(m[-1])
                for t in np.flatnonzero(m):
                    page = t // T; g, j = page % G, page // G
                    active[g, (b - b % S) * steps + (j * (T // 128) + (t % T) // 128) // selection[0].shape[2]] = 1
            q = (rng.standard_normal((G, B, H, 640)) * 0.05).astype(np.float32)
            q[..., 576:] = 0
            new = np.broadcast_to(lat[np.arange(B) // S, positions], (G, B, 640)).astype(np.float32)
            n_p, n_s, rows = len(persistent), len(scoped), persistent[2].shape[2]

            def kernel(pos, table, tau_r, ptie_r, own_r, act_r, q_r, new_r, pool_r, keys_r, out_r, recv_r, *scratch):
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                sel = mla_dcp.DcpSelection(keys_r, tau_r, ptie_r, own_r, *scratch[n_p + n_s:],
                                           active=act_r, steps=steps)
                att = mla_dcp.DcpAttention(cache, ws, pos, B, selection=sel, sequence_rows=S,
                                           cache_verify_rows=S > 1)
                if early:  # the queries leave first (as under the indexer); work fills the combine
                    mla_dcp.DcpAttention(cache, ws, pos, B, sequence_rows=S, cache_verify_rows=S > 1
                                         ).send_queries(q_r[...], new_r[...], 0)
                    att.prime()

                    def while_combining():
                        out_r[...] = jnp.full(out_r.shape, 3.0, jnp.float32)

                    out_r[...] = att.attend(None, new_r[...], 0, queries_sent=True,
                                            while_combining=while_combining)
                else:
                    att.prime()
                    out_r[...] = att.attend(q_r[...], new_r[...], 0)
                recv_r[...] = ws.receive[0].astype(jnp.float32)

            def body(pos, table, tau_, ptie_, own_, act_, q_, new_, pool_, keys_):
                vmem = pltpu.VMEM
                out, recv = pl.pallas_call(kernel,
                    out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                               jax.ShapeDtypeStruct((G, rows, 640), jnp.float32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=6,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,
                        out_specs=(pl.BlockSpec(memory_space=vmem),) * 2,
                        scratch_shapes=[*persistent, *scoped, *selection]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], tau_, ptie_, own_, act_[0], q_[0], new_[0],
                                                       pool_[0], keys_[0])
                return out[None], recv[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P()) + (P("tp"),) * 5,
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, recv = f(positions, tables.reshape(G, -1), tau, ptie, own, active, q, new, pool, keys)
            return np.asarray(out), np.asarray(recv), mla_dcp.packet_layout(B)[0]

        def case(*args, exact=False, **kwargs):
            one, recv, slots = run(*args, pair=False, early=False, **kwargs)
            two, _, _ = run(*args, pair=True, early=False, **kwargs)
            two_early, _, _ = run(*args, pair=True, early=True, **kwargs)
            assert np.array_equal(two, two_early), "queries_sent changes the pair combine's outputs"
            lse = recv[..., 512] + recv[..., 513]  # [owner, source, packet row]
            w = np.exp(lse - lse.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            bound = np.einsum("dir,dire->dre", w, np.abs(recv[..., :512]))
            bound = np.stack([bound[:, slots * b:slots * b + H] for b in range(one.shape[1])], axis=1)
            diff = np.abs(two.astype(np.float64) - one)
            assert np.isfinite(two).all()
            assert (diff <= 2.0**-8 * bound * (1 + 2.0**-6) + 1e-7 * bound.max()).all(), (
                args, (diff / np.maximum(2.0**-8 * bound, 1e-30)).max())
            assert not exact or np.array_equal(one, two), args
            print(args, "max |pair - one-stage| / (2^-8 bound):", (diff / np.maximum(2.0**-8 * bound, 1e-30)).max())

        # One 8-row verify (the B1 x7 shape) on compact steps with a 512-slot window; two 4-row
        # sequences dense; plain decode; ranks skewed so that only rank 1 attends pages (its mate and
        # the other chip reach the combine first; the mates' partials pass through).
        case([3000], 8, 200, 11, compact=128, window=512)
        case([1500, 3000], 4, 200, 1)
        case([2047, 600, 64, 3000], 1, 128, 2)
        case([3000], 8, 200, 13, favor=1, exact=True)
        print("OK")
    """

    def test_pair_combine_within_one_bf16_rounding(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestDraftPairCombine(CustomTestCase):
    """``KernelConfig.draft_pair_combine``: the MLA draft's attention
    (explicit keys, its lse in the padding row, FP8 pages) through the host
    combine, its last chunk with the pair stage, in Pallas interpret mode at
    DCP16 with ``HOST_LANES`` 4 (four hosts of two chips; the draft runs four
    of four), one- and two-chunk calls: finite, an earlier chunk's rows the
    knob-off bits, the last chunk's within the combine's BF16 roundings of
    them -- |on - off| <= 3 2^-8 sum_i softmax(lse)_i |o_i| over every rank's
    partial of the row (the pair merge's rounding, plus the host merge's
    re-rounding of a value that moved)."""

    SCRIPT = """
        import os, time
        G = 16
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp

        # The MLA draft's attention (explicit keys, lse in the padding row, FP8 pages) through the host combine,
        # with and without its pair stage. HOST_LANES 4 at DCP16: 4 hosts of 2 chips (the draft: 4 hosts of 4 chips).
        T, H, S, FP8 = 512, 3, 8, jnp.float8_e4m3fn
        mla_dcp.HOST_LANES = G // 4
        mla_dcp.GROUPS = (G, 8, 16, 32)
        LANES = mla_dcp.LANES

        def tokens(x):
            return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                    + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

        def _pages(self, slot):  # interpret mode has no ref bitcasts (see TestSelectedAttention)
            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                if hasattr(lanes, "size"):  # pl.ds
                    return jax.lax.dynamic_slice_in_dim(x, lanes.start, lanes.size, 1)
                return x[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

        _pack = mla_dcp.DcpAttention._pack_partials

        def _pack_partials(self, first, end, base=0):  # also keep each chunk's packets (every owner's partials)
            _pack(self, first, end, base)
            capture, counter = self._capture
            capture[counter[0]] = self.ws.send[...].astype(jnp.float32)
            counter[0] += 1

        mla_dcp.DcpAttention._pack_partials = _pack_partials
        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes

        def run(firsts, counts, seed, pairs):
            # Outputs [G, B, H, 512] and every rank's packets of each chunk [G, chunks, G, H S, 640].
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            plan = mla_dcp.page_plan(B, H, S, FP8, force_chunks=True)
            persistent = mla_dcp.persistent_scratch(B, G, H, cache_dtype=FP8, sequence_rows=S, fp8_wire=True,
                                                    force_chunks=True, chunk_rows=S)
            scoped = mla_dcp.scoped_scratch(B, plan["span"], plan["buffers"], heads=H, group=G, cache_dtype=FP8,
                                            page_tokens=T, sequence_rows=S, force_chunks=True, chunk_rows=S)
            ring, hosts = persistent[0].shape[0], G // mla_dcp.HOST_LANES
            stage2 = (pltpu.VMEM((ring, hosts, H * S, LANES), jnp.bfloat16),
                      pltpu.VMEM((hosts, H * S, LANES), jnp.bfloat16), pltpu.SemaphoreType.DMA((2, ring)))
            extra = (pltpu.SemaphoreType.DMA((2, ring)),) if pairs else ()
            pages = max(-(-(int(p0) + T) // T) // G for p0 in firsts) + 1
            lat = (rng.standard_normal((len(firsts), pages * G * T, LANES)) * 0.5).astype(np.float32)
            lat[..., 576:] = 0
            pool = rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, LANES)).astype(ml_dtypes.float8_e4m3fn)
            tables = np.zeros((G, B, pages), np.int32)
            for b in range(B):
                s_, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s_ * pages + np.arange(pages)
                for t in range(p0):
                    page = t // T; g, j = page % G, page // G
                    pool[g, tables[g, b, j], (t % T) // 4, t % 4] = lat[s_, t].astype(ml_dtypes.float8_e4m3fn)
            q = (rng.standard_normal((G, B, H, LANES)) * 0.08).astype(np.float32)
            q[..., 576:] = 0
            keys = (rng.standard_normal((len(firsts), 2 * S, LANES)) * 0.5).astype(np.float32)
            keys[..., 576:] = 0
            lane = np.arange(2 * S)
            valid = np.stack([((lane < S) & (lane <= c)) | ((lane >= S) & (lane < 2 * S - 1)) for c in counts])
            valid = valid[:, None, :].astype(np.int32)
            n_p, n_s = len(persistent), len(scoped)
            rows = persistent[1].shape[1]

            def kernel(pos, table, q_r, keys_r, valid_r, pool_r, out_r, send_r, *scratch):
                counter = scratch[-1]
                counter[0] = 0
                scratch = scratch[:-1]
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                att = mla_dcp.DcpAttention(cache, ws, pos, B, sequence_rows=S, heads=H, chunk_rows=S)
                att.lse_in_padding = True
                att.group_fold = True
                att.force_chunked = True
                att.host_combine = tuple(scratch[n_p + n_s : n_p + n_s + 3])
                att.host_pairs = scratch[n_p + n_s + 3] if pairs else None
                att._capture = (send_r, counter)
                att.prime()
                out_r[...] = att.attend(q_r[...], jnp.zeros((B, LANES), jnp.float32), 0, keys=keys_r, key_valid=valid_r)

            def body(pos, table, q_, keys_, valid_, pool_):
                vmem = pltpu.VMEM
                out, send = pl.pallas_call(kernel,
                    out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                               jax.ShapeDtypeStruct((B // S, G, rows, LANES), jnp.float32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)],
                        out_specs=(pl.BlockSpec(memory_space=vmem),) * 2,
                        scratch_shapes=[*persistent, *scoped, *stage2, *extra, pltpu.SMEM((1,), jnp.int32)]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], q_[0], keys_, valid_, pool_[0])
                return out[None], send[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P("tp"), P(), P(), P("tp")),
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, send = f(positions, tables.reshape(G, -1), q, keys, valid, pool)
            return np.asarray(out), np.asarray(send)

        def bound(send):
            # 2^-8 sum_i softmax(lse)_i |o_i| over every rank's partial of each owner's real rows ([G, S - 1, H, 512];
            # row S - 1 is the padding row, whose packet rows carry the lse's).
            hl = mla_dcp.HOST_LANES
            hosts = G // hl
            out = np.zeros((G, S, H, 512))
            for m in range(G):
                idx = (m % hl) * hosts + m // hl  # the owner's packet in the lane-major send buffer
                part = send[:, idx]  # [G ranks, H S, 640]
                lse = part[..., 512] + part[..., 513]  # [G, rows]
                w = np.exp(lse - lse.max(axis=0, keepdims=True)); w /= w.sum(axis=0, keepdims=True)
                b = np.einsum("gr,gre->re", w, np.abs(part[..., :512]))  # [rows, 512]
                out[m] = b.reshape(S, H, 512)
            return 2.0 ** -8 * out[:, : S - 1]

        t0 = time.time()
        ok = True
        for firsts, counts, seed in (([3000], [5], 1), ([5200], [7], 2), ([1500, 4100], [3, 7], 3)):
            off, send = run(firsts, counts, seed, False)
            on, _ = run(firsts, counts, seed, True)
            B = len(firsts) * S
            diff = np.abs(on.astype(np.float64) - off)  # [G, B, H, 512]
            finite = bool(np.isfinite(on).all())
            good, ratio, rel = finite, 0.0, 0.0
            # Only the last chunk takes the pair stage: the earlier ones' outputs are the knob-off bits.
            good &= bool(np.array_equal(on[:, : B - S], off[:, : B - S]))
            for c in range(len(firsts)):  # each chunk's real rows against its partials' bound
                bnd = bound(send[:, c])
                d = diff[:, c * S : c * S + S - 1]
                good &= bool((d <= 3 * bnd * (1 + 2.0**-6) + 1e-7 * bnd.max()).all())
                ratio = max(ratio, float((d / np.maximum(bnd, 1e-30)).max()))
                rel = max(rel, float(d.max() / np.abs(off[:, c * S : c * S + S - 1]).max()))
            print(f"firsts {firsts} counts {counts}: identical {np.array_equal(on, off)} max |on - off| / (2^-8 bound) "
                  f"{ratio:.3f} max rel {rel:.2e} finite {finite} -> {'ok' if good else 'FAIL'} ({time.time() - t0:.0f}s)",
                  flush=True)
            ok &= good
        print("OK" if ok else "FAIL")
    """

    def test_pair_stage_within_its_roundings(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        # 16 interpret devices block in callbacks: give the CPU client enough
        # threads (XLA reads NPROC for its pool size).
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestPrefillIndexRows(CustomTestCase):
    """A prefill window's index keys laid out as pool rows match the decode's
    per-token commit of the same keys, byte for byte."""

    def test_rows_match_the_token_commit(self):
        from unittest import mock

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        rng = np.random.default_rng(5)
        tokens, pages = 512, 3
        keys = rng.standard_normal((tokens * pages, 128)).astype(np.float32)
        rows = np.asarray(
            dsa_dcp.index_rows(jnp.asarray(keys).astype(jnp.bfloat16), pages)
        )
        self.assertEqual(rows.shape, (pages, 4 * 33, 4, 128))
        # The same keys, token by token, into blocks 1..3 of one rank's pool
        # (a table whose entry e is global page e: every page on this rank).
        records = dsa_dcp.key_records(jnp.asarray(keys).astype(jnp.bfloat16))[
            :, None, :
        ]
        positions = np.arange(tokens * pages, dtype=np.int32)
        tables = np.tile(np.arange(1, pages + 1, dtype=np.int32), (len(positions), 1))
        pool = (jnp.zeros((pages + 1, 4 * 33, 4, 128), jnp.uint8),)
        with mock.patch.object(mla_dcp, "owner_index", return_value=0):
            (pool,) = dsa_dcp.commit_index_rows(
                pool,
                records,
                jnp.asarray(tables),
                jnp.asarray(positions),
                group=1,
                global_pages=True,
            )
        np.testing.assert_array_equal(np.asarray(pool)[1:], rows)


class TestDecodeIndexer(CustomTestCase):
    """The decode megakernel's indexer (``dsa_decode.DecodeIndexer``) in
    Pallas interpret mode on a 4-rank DCP group (8 index heads a rank): the
    rows' new key records, every rank's keys, the selection and the
    attention steps holding selected tokens match a numpy reference of the
    whole indexer (projections with sglang's native rounding, FP8 queries and
    head weights, scores, top-k), plain and verify rows."""

    SCRIPT = 'import os, sys, time\nG = 4\nos.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"\nos.environ["JAX_PLATFORMS"] = "cpu"\nimport ml_dtypes, numpy as np, jax, jax.numpy as jnp\nfrom jax.experimental import pallas as pl\nfrom jax.experimental.pallas import tpu as pltpu\nfrom jax.sharding import Mesh, PartitionSpec as P\nif os.environ.get("GATHER") == "1":\n    from sglang_tpu.k3_programs.kimi_k3 import config as kc\n    kc.install(kc.KernelConfig(dsa_gather_finish=True))\nelif os.environ.get("ROUNDS"):\n    from sglang_tpu.k3_programs.kimi_k3 import config as kc\n    knobs = dict(kv.split("=") for kv in os.environ.get("KNOBS", "").split(",") if kv)\n    kc.install(kc.KernelConfig(dsa_select_rounds=int(os.environ["ROUNDS"])).replace(**knobs))\nfrom sglang_tpu.k3_programs.kimi_k3 import dsa\nfrom sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, dsa_decode, mla_dcp\nif os.environ.get("GATHER") == "1":\n    _finish = dsa_dcp.IndexSelect._gather_finish\n    def _traced(self, first_round):\n        pl.debug_print("GATHER at round {}", first_round)\n        return _finish(self, first_round)\n    dsa_dcp.IndexSelect._gather_finish = _traced\nmla_dcp.GROUPS = (G, 8, 16, 32)\ndsa_dcp.ACTIVITY_MIN_STEPS = int(os.environ.get("ACTIVITY_MIN_STEPS", "0"))\nBF = ml_dtypes.bfloat16\nbf = lambda x: np.asarray(x, np.float32).astype(BF).astype(np.float32)\n\nS = int(os.environ["S"]); TOPK = int(os.environ["TOPK"])\nfirsts = [int(x) for x in os.environ["POS"].split(",")]\nH, QL, T, PER = 1024, 1536, 512, 32 // G\npositions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)\nB = len(positions)\nrng = np.random.default_rng(0)\npages = max(-(-((p + T) // T) // G) for p in positions) + 1\npages += int(os.environ.get("WIDE", "0"))  # a table wider than the context (as the 1M-context serving tables)\nkey_rows = dsa_dcp.key_rows_for(pages, T)\nhidden = bf(rng.standard_normal((B, H)))\nqlat = bf(rng.standard_normal((B, QL)))\nwqb = bf(rng.standard_normal((QL, 32 * 128)) * QL**-0.5)\nwk = bf(rng.standard_normal((H, 128)) * H**-0.5)\nwproj = np.zeros((H, 128), np.float32); wproj[:, :32] = bf(rng.standard_normal((H, 32)) * H**-0.5)\nknorm = bf(np.stack([1 + 0.1 * rng.standard_normal(128), 0.1 * rng.standard_normal(128)]))\n\n# Reference projections (native rounding).\ndef layer_norm(x, w, b, eps=1e-5):\n    mean = x.mean(-1, keepdims=True); var = (x * x).mean(-1, keepdims=True) - mean * mean\n    return (x - mean) / np.sqrt(var + eps) * w + b\nkey = bf(layer_norm(bf(hidden @ wk), knorm[0], knorm[1]))\nnew_k8, new_e = dsa.quantize_fp8(key)\ngate = bf(bf(bf(hidden @ wproj)[:, :32]) * float(np.float32(BF(32**-0.5))))\nq_full = bf(qlat @ wqb)  # [B, 4096]\nq8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)\nfor b in range(B):\n    q8[b], e = dsa.quantize_fp8(q_full[b].reshape(32, 128))\n    w[b] = gate[b] * np.exp2(e[:, 0].astype(np.float32)) * 128**-0.5\n\n# Pool: committed tokens\' records (random keys), garbage past them.\npool = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)\ntables = np.zeros((G, B, pages), np.int32)\nseq_k8, seq_e = [], []\nfor s_, p0 in enumerate(firsts):\n    raw = rng.standard_normal((p0, 128)).astype(np.float32)\n    k8, ke = dsa.quantize_fp8(raw)\n    seq_k8.append(k8); seq_e.append(ke)\n    rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))\n    for r in range(S):\n        tables[:, s_ * S + r] = 1 + s_ * pages + np.arange(pages)\n    for t in range(p0):\n        page = t // T; g = page % G\n        blk = tables[g, s_ * S, page // G]\n        (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)\n        pool[g, blk, row, sub] = rec[t, :128]; pool[g, blk, srow, 0, lane] = rec[t, 128]\n\nwqb_ranks = np.stack([wqb[:, g * PER * 128:(g + 1) * PER * 128] for g in range(G)]).astype(BF)\nwks_ranks = np.stack([dsa_decode.key_weight(wk, g, G) for g in range(G)])\nwpt = dsa_decode.gate_weight(wproj)\npersistent = dsa_decode.persistent_scratch(B, G)\nscoped = dsa_decode.scoped_scratch(B, hidden=H, index_block_shape=(132, 4, 128), key_rows=key_rows,\n                                   page_tokens=T, sequence_rows=S,\n                                   page_plan=mla_dcp.page_plan(B, 3, S, jnp.float8_e4m3fn, dsa=True))\nnames = list(scoped)\nSTEP_ROWS = scoped["attend_keys"].shape[2]\nSTEPS = key_rows // STEP_ROWS\n\ndef kernel(pos_ref, table_ref, hidden_ref, qlat_ref, wqb_ref, wks_ref, wpt_ref, knorm_ref, pool_ref, lat_ref,\n           tau_out, ptie_out, own_out, act_out, rec_out, keys_out, *scratch):\n    cache = mla_dcp.DcpCache((lat_ref,), table_ref, pages, 0, group=G, global_pages=False)\n    sc = dict(zip(names, scratch[len(persistent):]))\n    indexer = dsa_decode.DecodeIndexer(\n        weights={"d_wqb": wqb_ref, "d_wks": wks_ref, "d_wpt": wpt_ref, "d_knorm": knorm_ref},\n        pools=(pool_ref,), layer=0, keys=keys_out, records_out=rec_out, persistent=scratch[:len(persistent)],\n        scoped=sc, topk=TOPK, sequence_rows=S)\n    indexer.start()\n    sel = indexer.selection(cache, pos_ref, hidden_ref[...], qlat_ref[...], 0)\n    for b in range(B):\n        tau_out[b] = sel.tau[b]; ptie_out[b] = sel.p_tie[b]; own_out[b] = sel.own[b]\n    for i in range(B * STEPS):\n        act_out[i] = sel.active[i]\n\ndef body(pos, table, hid, ql, wqb_r, wks_r, wpt_r, kn_r, pool_r, lat_r):\n    vmem, smem = pltpu.VMEM, pltpu.SMEM\n    out = pl.pallas_call(kernel,\n        out_shape=(jax.ShapeDtypeStruct((B,), jnp.int32),) * 3 + (jax.ShapeDtypeStruct((B * STEPS,), jnp.int32),\n            \n            jax.ShapeDtypeStruct((B, 24, 2, 128), jnp.int32), jax.ShapeDtypeStruct((B, key_rows, 128), jnp.int32)),\n        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,\n            in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 6,\n            out_specs=[pl.BlockSpec(memory_space=smem)] * 4 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,\n            scratch_shapes=[*persistent, *scoped.values()]),\n        compiler_params=pltpu.CompilerParams(has_side_effects=True),\n        interpret=pltpu.InterpretParams())(pos, table[0], hid, ql, wqb_r[0], wks_r[0], wpt_r, kn_r, pool_r[0], lat_r)\n    return tuple(x[None] for x in out)\n\nmesh = Mesh(np.asarray(jax.devices()), ("tp",))\nf = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P("tp"), P("tp"), P(), P(), P("tp"), P()),\n                          out_specs=(P("tp"),) * 6, check_vma=False))\nt0 = time.time()\ntau, ptie, own, act, rec, keys = (np.asarray(x) for x in f(\n    positions, tables.reshape(G, -1), hidden.astype(BF), qlat.astype(BF), wqb_ranks, wks_ranks,\n    wpt, knorm.astype(BF), pool, jnp.zeros((1, 128, 4, 640), jnp.float8_e4m3fn)))\n# Records: the rows\' new keys (every rank the same).\nwant_rec = np.asarray(dsa_dcp.key_records(jnp.asarray(key)))\ngot_rec = np.concatenate([rec[0, :, 0, 0, :], rec[0, :, 0, 1, :1]], axis=1)\nrec_match = np.mean(got_rec == want_rec)\nprint(f"records match {rec_match:.4f}; identical on ranks {all(np.array_equal(rec[0], rec[g]) for g in range(G))}")\nok = rec_match > 0.99\nunkey = lambda k: (k ^ ((k >> 31) & 0x7FFFFFFF)).astype(np.int32).view(np.float32)\nfor b in range(B):\n    s_, p0, L = b // S, firsts[b // S], positions[b] + 1\n    t = np.arange(L)\n    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)\n    k8 = np.concatenate([seq_k8[s_], new_k8[s_ * S:s_ * S + (b - s_ * S) + 1]])\n    ke = np.concatenate([seq_e[s_], new_e[s_ * S:s_ * S + (b - s_ * S) + 1]])\n    ref = dsa.monotone_key(dsa.scores(q8[b], w[b], k8, ke).astype(np.float32))\n    err = np.abs(unkey(kk) - unkey(ref)).max() / np.abs(unkey(ref)).max()\n    got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])\n    want = dsa.topk_mask(kk, t, TOPK)\n    if os.environ.get("ROUNDS") and L > TOPK:  # bin selection: keys at or above the prefix bin\n        kth = np.sort(kk)[::-1][TOPK - 1]; drop = 32 - 4 * int(os.environ["ROUNDS"])\n        u = (int(kth) ^ -(2**31)) & 0xFFFFFFFF; tb = ((u >> drop) << drop) ^ (1 << 31)\n        want = kk >= (tb - (1 << 32) if tb >= (1 << 31) else tb)\n    ref_sel = dsa.topk_mask(ref, t, TOPK)\n    agree = np.mean(got == ref_sel)\n    uniform = all(len(set(v[:, b])) == 1 for v in (tau, ptie, own))\n    print(f"row {b} L={L} key err {err:.1e} exact {np.array_equal(got, want)} vs-ref-selection {agree:.4f} "\n          f"own {own[0, b]}/{bool(own[0, b]) == bool(got[-1])} uniform {uniform}")\n    ok &= bool(err < 1e-2 and np.array_equal(got, want) and uniform and bool(own[0, b]) == bool(got[-1]) and agree > 0.99)\n# Step activity: per rank, row (its sequence\'s first row), step: any selected token of the sequence\'s rows.\nwant_act = np.zeros_like(act)\nfor b in range(B):\n    L, first = positions[b] + 1, b - b % S\n    t = np.arange(L)\n    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)\n    for i in np.flatnonzero(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])):\n        g, j = (i // T) % G, (i // T) // G\n        want_act[g, first * STEPS + (j * (T // 128) + (i % T) // 128) // STEP_ROWS] = 1\nif dsa_dcp.ACTIVITY_MIN_STEPS:\n    want_act = np.ones_like(act)\n# KernelConfig.dsa_live_activity: only the entries the attention reads are written (per rank, each\n# sequence first row: the steps of the longest row and one more); the rest are stale.\ndomain = np.ones_like(act, bool)\nif "dsa_live_activity=1" in os.environ.get("KNOBS", ""):\n    domain[:] = False\n    for g in range(G):\n        longest = max(max(-(-(int(p) + 1) // T) - g + G - 1, 0) // G for p in positions)\n        live = min(-(-longest * (T // 128) // STEP_ROWS) + 1, STEPS)\n        for f in range(0, B, S):\n            domain[g, f * STEPS:f * STEPS + live] = True\nact_ok = np.array_equal(act[domain], want_act[domain])\nprint(f"activity exact {act_ok} active steps {int(act[domain].sum())} of {int(domain.sum())}")\nok &= bool(act_ok)\nprint("OK" if ok else "FAIL")\n'

    def test_indexer_matches_the_reference(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        # The scoring knobs (TestIndexSelectScoringKnobs) inside the decode
        # indexer: resident keys (the step activity scans the HBM keys their
        # copies fill), the early plan and page copies under the query gather.
        SCORING = (
            "dsa_stage_slots=4,dsa_page_slots=8,dsa_resident_keys=1,"
            "dsa_patch_loop=1,dsa_early_plan=1"
        )
        LIVE = SCORING + ",dsa_live_activity=1,dsa_odd_stride=1"

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        for case in (
            {"S": "1", "TOPK": "128", "POS": "700,2100"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "ACTIVITY_MIN_STEPS": "16"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "GATHER": "1"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "ROUNDS": "5"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "ROUNDS": "5", "KNOBS": SCORING},
            {"S": "8", "TOPK": "400", "POS": "3069", "ROUNDS": "5", "KNOBS": SCORING},
            # KernelConfig.dsa_live_activity: the entries the attention reads,
            # scanned and dense, verify sequences and plain rows, on tables
            # wider than the contexts; with dsa_odd_stride (resident chunks of
            # 48 rows at 56 rows apart, and copied ones).
            {
                "S": "4",
                "TOPK": "200",
                "POS": "1500,3000",
                "ROUNDS": "5",
                "KNOBS": LIVE,
                "WIDE": "8",
            },
            {
                "S": "4",
                "TOPK": "200",
                "POS": "1500,3000",
                "ROUNDS": "5",
                "KNOBS": LIVE,
                "ACTIVITY_MIN_STEPS": "16",
                "WIDE": "8",
            },
            {
                "S": "8",
                "TOPK": "400",
                "POS": "3069",
                "ROUNDS": "5",
                "KNOBS": LIVE,
                "WIDE": "8",
            },
            {
                "S": "1",
                "TOPK": "128",
                "POS": "700,2100",
                "ROUNDS": "5",
                "KNOBS": "dsa_live_activity=1,dsa_odd_stride=1",
                "WIDE": "8",
            },
        ):
            with self.subTest(**case):
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **case)
                result = subprocess.run(
                    [sys.executable, "-c", self.SCRIPT],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=900,
                )
                self.assertIn(
                    "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
                )
                if case.get("GATHER"):  # the gather finish ran (on every rank)
                    self.assertGreaterEqual(
                        result.stdout.count("GATHER at round"), 4, result.stdout[-2000:]
                    )


class TestRouterBothHalves(CustomTestCase):
    """The decode MoE router with both router halves on every rank
    (``KernelConfig.decode_router_both_halves``) in Pallas interpret mode on a
    rank pair of host devices: on both lanes the raw score rows assembled from
    the own and peer half projections, and the final score rows, selected
    experts, probabilities, route counts and probability totals, are
    bit-identical to the pair's score-half exchange, with stale NaN bytes in
    the router's padding columns."""

    SCRIPT = """
        import dataclasses, os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        H, E, K = 7168, 896, 16
        FIELDS = [f.name for f in dataclasses.fields(dmk.MixtureOfExpertsWeights)]
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))

        def run(slots, hidden, router, bias, live):
            B = hidden.shape[0]

            def kernel(live_ref, hidden_ref, bias_ref, router_ref, raw_out, scores_out,
                       experts_out, probabilities_out, counts_out, totals_out,
                       router_vmem, router_semaphores, scores_vmem, sends, receives):
                weights = dmk.MixtureOfExpertsWeights(**dict(
                    dict.fromkeys(FIELDS), routing_projection=router_ref,
                    routing_projection_vmem=router_vmem,
                    routing_copy_semaphores=router_semaphores, routing_bias=bias_ref))
                rank = jax.lax.axis_index("tp")
                wait_half = weights.wait_for_routing_projection_half if slots == 2 else None

                def load_router():
                    # Stale arena bytes (NaN) under every router column, then the
                    # arena request's copies; one slot is waited for up front.
                    router_vmem[...] = jnp.full(router_vmem.shape, jnp.nan, jnp.bfloat16)
                    scores_vmem[...] = jnp.full(scores_vmem.shape, jnp.nan, jnp.float32)
                    dmk._router_copy(router_ref, router_vmem, router_semaphores, False,
                                     initialize_padding=False)
                    if slots == 1:
                        weights.wait_for_routing_projection_copy()

                load_router()
                dmk._project_routing_scores(rank, hidden_ref[...], router_vmem, scores_vmem,
                                            sends.at[0], receives.at[0],
                                            wait_for_projection_half=wait_half)
                raw_out[...] = scores_vmem[...]
                load_router()
                live_rows = (jnp.stack([live_ref[b] for b in range(B)]) != 0) if live else None
                experts, probabilities, counts, totals = dmk._route_hidden_states(
                    hidden_ref[...], router_vmem, bias_ref, scores_vmem, sends.at[0],
                    receives.at[0], selected_expert_count=K, local_expert_count=112,
                    expert_tensor_parallel=1, on_local_selection=lambda *unused: None,
                    after_selection=lambda unused: None, live_rows=live_rows,
                    wait_for_projection_half=wait_half)
                scores_out[...] = scores_vmem[...]
                experts_out[...] = experts
                probabilities_out[...] = probabilities
                counts_out[...] = jnp.broadcast_to(counts[:, None], counts_out.shape)
                totals_out[...] = jnp.broadcast_to(totals[:, None], totals_out.shape)

            vmem = pltpu.VMEM
            shapes = ((B, 1024, jnp.float32), (B, 1024, jnp.float32), (B, K, jnp.int32),
                      (B, K, jnp.float32), (B, 128, jnp.int32), (B, 128, jnp.float32))

            def body(live_rows, hidden, bias, router):
                outputs = pl.pallas_call(
                    kernel,
                    out_shape=tuple(jax.ShapeDtypeStruct(s[:2], s[2]) for s in shapes),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=1,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2
                        + [pl.BlockSpec(memory_space=pl.ANY)],
                        out_specs=[pl.BlockSpec(memory_space=vmem)] * len(shapes),
                        scratch_shapes=[vmem((slots, H, 512), jnp.bfloat16),
                                        pltpu.SemaphoreType.DMA((2,)),
                                        vmem((B, 1024), jnp.float32),
                                        pltpu.SemaphoreType.DMA((1,)),
                                        pltpu.SemaphoreType.DMA((1,))]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams(),
                )(live_rows, hidden, bias, router)
                return tuple(x[None] for x in outputs)

            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(),) * 4,
                                      out_specs=(P("tp"),) * len(shapes), check_vma=False))
            live_rows = (np.arange(B) % 3 != 1) if live else np.ones(B, bool)
            return [np.asarray(x) for x in f(live_rows.astype(np.int32), hidden, bias, router)]

        rng = np.random.default_rng(0)
        for B, live in ((1, False), (8, False), (16, True)):
            hidden = rng.standard_normal((B, H)).astype(ml_dtypes.bfloat16)
            router = (rng.standard_normal((H, E)) * H**-0.5).astype(ml_dtypes.bfloat16)
            bias = (0.1 * rng.standard_normal((1, E))).astype(np.float32)
            exchanged = run(1, hidden, router, bias, live)
            projected = run(2, hidden, router, bias, live)
            names = ("raw scores", "final scores", "experts", "probabilities", "counts",
                     "totals")
            for name, old, new in zip(names, exchanged, projected):
                assert old.tobytes() == new.tobytes(), (B, name)
            raw = exchanged[0]
            assert np.array_equal(raw[0], raw[1]), B  # both lanes hold the same rows
            assert not np.isnan(raw).any() and not raw[..., 896:].any(), B
            want = hidden.astype(np.float32) @ router.astype(np.float32)
            assert np.abs(raw[0, :, :896] - want).max() < 1e-3 * np.abs(want).max(), B
            assert exchanged[4][..., 0].sum(axis=1).all(), B  # both lanes own routes
        print("OK")
    """

    def test_both_halves_match_the_score_exchange(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestManyOperandOps(CustomTestCase):
    """Kimi-K3.1's decode steps take more operands than a torch op schema holds (64):
    neighbouring operands with one spec and donation become one ``Tensor[]`` argument."""

    SCRIPT = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax, numpy as np, torch
from jax.sharding import Mesh, PartitionSpec as P
from sglang_tpu.k3_programs import bridge

mesh = Mesh(np.array(jax.devices("cpu")[:4]), ("x",))
W, POOLS = 40, 50
def body(*args):
    acc = sum(a.sum() for a in args[:W])
    return (args[-1] + acc, *(p + 1 for p in args[W:W + POOLS]))
in_specs = (P("x"),) * (W + POOLS) + (P(),)
donated = tuple(range(W, W + POOLS))
op = bridge.register_op("pallas::k31_runs_check", mesh, body, in_specs,
                        (P(),) + (P("x"),) * POOLS, donate_argnums=donated)
assert op.runs == [(0, W), (W, W + POOLS), (W + POOLS, W + POOLS + 1)], op.runs
assert str(op.op._schema).startswith("(Tensor[] a0, Tensor[] a1, Tensor a2)"), op.op._schema
init = op._init_fn
assert init.donate_argnums in ([1], (1,)), init.donate_argnums
flat = [torch.zeros(2, 8)] * W + [torch.zeros(3, 8)] * POOLS + [torch.zeros(5)]
grouped = [flat[s] if e - s == 1 else list(flat[s:e]) for s, e in op.runs]
args, kwargs = init._get_jax_placeholders(*init._normalize_invocation_args(grouped, {}))
with jax._src.config.export_ignore_forward_compatibility(True):
    lowered = init.exported(*args, **kwargs)
assert len(lowered.in_avals) == W + POOLS + 1 and lowered.in_avals[0].shape == (8, 8)
assert len(lowered.out_avals) == POOLS + 1 and lowered.out_avals[1].shape == (12, 8)
small = bridge.register_op("pallas::k31_runs_small", mesh, lambda a, b: (a + b,),
                           (P("x"), P("x")), (P("x"),))
assert not isinstance(small, bridge._RunOp)
print("OK")
"""

    def test_operand_runs(self):
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_programs import bridge

        specs = (P("x"),) * 3 + (P(),)
        self.assertEqual(
            bridge._operand_runs(specs, (1,)), [(0, 1), (1, 2), (2, 3), (3, 4)]
        )
        many = (P("x"),) * 70 + (P(),)
        self.assertEqual(
            bridge._operand_runs(many, tuple(range(40, 70))),
            [(0, 40), (40, 70), (70, 71)],
        )

    def test_registers_and_traces_with_flat_operands(self):
        import os
        import subprocess
        import sys

        env = dict(
            os.environ, PYTHONPATH=os.pathsep.join(sys.path), JAX_PLATFORMS="cpu"
        )
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestK31Draft(CustomTestCase):
    """Kimi-K3.1's DSpark draft (``Inferact/Kimi-K3.1-Pro-DSpark-preview``)
    in the MLA draft's config: three heads a rank, causal block rows, YaRN's
    attention factor (the rotary scale is then exactly one), its aux layers;
    the reference attention's block rows see only earlier block rows."""

    CONFIG = {
        "model_type": "k3_dspark",
        "hidden_size": 7168,
        "num_hidden_layers": 5,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-5,
        "target_layer_ids": [75, 79, 83, 87, 91],
        "mask_token_id": 163839,
        "markov_rank": 512,
        "block_size": 7,
        "rope_interleave": True,
        "full_attention_causal": True,
        "dflash_config": {"causal": True},
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "attn_factor": 0.7426255848312643,
        },
    }

    def test_config_and_causal_reference(self):
        import dataclasses

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla as mla

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps(self.CONFIG))
            c = mla.MlaDraftConfig.from_checkpoint(Path(d))
        self.assertEqual(
            (c.heads_per_rank, c.causal, c.markov_rank, c.mask_token),
            (3, True, 512, 163839),
        )
        self.assertEqual(c.aux_layers, (76, 80, 84, 88, 92))
        self.assertAlmostEqual(mla.rope_mscale(c), 1.0, places=6)
        self.assertAlmostEqual(c.softmax_scale, 192**-0.5, places=9)
        small = dataclasses.replace(
            c,
            hidden=256,
            heads=4,
            q_lora=128,
            kv_lora=64,
            nope=32,
            rope=16,
            v_head=32,
            intermediate=128,
            vocab=512,
            layers=1,
        )
        rng = np.random.default_rng(0)
        w = {
            k: jnp.asarray(rng.standard_normal(s).astype(np.float32) * 0.05)
            for k, s in mla.weight_shapes(small).items()
        }
        h = jnp.asarray(rng.standard_normal((4, small.hidden)).astype(np.float32))
        positions, context = (
            jnp.arange(10, 14),
            jnp.asarray(rng.standard_normal((6, small.latent)).astype(np.float32)),
        )
        valid = jnp.ones((6,), bool)
        causal = mla.reference_attention(w, small, 0, h, positions, context, valid)
        full = mla.reference_attention(
            w, dataclasses.replace(small, causal=False), 0, h, positions, context, valid
        )
        alone = mla.reference_attention(
            w, small, 0, h[:1], positions[:1], context, valid
        )
        np.testing.assert_allclose(
            causal[-1], full[-1], atol=1e-5
        )  # the last row sees every block row
        np.testing.assert_allclose(
            causal[0], alone[0], atol=1e-5
        )  # the first sees only itself
        self.assertFalse(np.allclose(causal[0], full[0], atol=1e-5))


class TestExpertSchedule(CustomTestCase):
    """The grouped routed-expert stream (``_run_routed_expert_stream``) at B8
    and B16 in Pallas interpret mode, on random MXFP4 experts and routings
    (single-row and multi-row experts, more rows than a compact tile, odd
    expert counts, empty rows, expert 0, no experts): reading each pair from
    the SMEM schedule (``KernelConfig.expert_schedule``) gives bit-identical
    outputs to reducing the pending routes every iteration."""

    SCRIPT = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        from types import SimpleNamespace
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        E, K, IN, WIDTH, OUT = 112, 16, 512, 768, 128

        def stream(B, scheduled):
            rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)
            compact = dmk._compact_rows(B)

            def kernel(sel_ref, prob_ref, count_ref, total_ref, source_ref, gu_hbm, gus_hbm,
                       down_hbm, ds_hbm, out_ref, gu, gus, down, ds, semaphores, gate_up_acc,
                       down_acc, processed, first_compact, second_compact, table, table_semaphore):
                selected, counts = sel_ref[:, :K], count_ref[:, 0]

                def copy_expert(expert, slot, *, gate_up=True, down=True):
                    def copy(hbm, vmem, kind, wanted):
                        return (pltpu.make_async_copy(hbm.at[expert], vmem.at[slot],
                                                      semaphores.at[slot, kind]) if wanted else None)
                    return SimpleNamespace(
                        gate_up=copy(gu_hbm, gu, 0, gate_up), gate_up_scales=copy(gus_hbm, gus, 1, gate_up),
                        down=copy(down_hbm, down_slots, 2, down), down_scales=copy(ds_hbm, ds, 3, down))

                down_slots = down
                dmk._start_initial_expert_weight_copies(
                    selected, counts, copy_expert, slot_count=4, already_started=jnp.int32(0),
                    group_pending_routes=True)
                schedule = None
                if scheduled:
                    schedule = dmk._expert_schedule(processed.at[:8], table, table_semaphore)
                    dmk._start_expert_schedule(schedule, selected, counts)
                dmk._run_routed_expert_stream(
                    source_ref, selected, prob_ref[:, :K], counts, total_ref[:, 0], gu, gus, down_slots,
                    ds, out_ref, gate_up_acc, down_acc, processed, first_compact, second_compact,
                    copy_expert, expert_input_size=IN, expert_width=WIDTH, expert_output_size=OUT,
                    group_pending_routes=True, expert_schedule=schedule)

            sources = ((pltpu.VMEM((compact, IN), jnp.float32),) * 2 if compact
                       else (pltpu.VMEM((1, IN), jnp.bfloat16),) * 2)
            return jax.jit(pl.pallas_call(
                kernel, out_shape=jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 5
                + [pl.BlockSpec(memory_space=pl.ANY)] * 4,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((4, IN // 8, 2 * WIDTH), jnp.uint32),
                    pltpu.VMEM((4, IN // 32, 2 * WIDTH), jnp.uint8),
                    pltpu.VMEM((4, WIDTH // 8, OUT), jnp.uint32),
                    pltpu.VMEM((4, WIDTH // 32, OUT), jnp.uint8),
                    pltpu.SemaphoreType.DMA((4, 4)),
                    pltpu.VMEM((rows[0], 4 * WIDTH), jnp.float32),
                    pltpu.VMEM((rows[1], OUT), jnp.float32),
                    pltpu.VMEM((B, 128), jnp.int32), *sources,
                    pltpu.SMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA],
                interpret=pltpu.InterpretParams()))

        rng = np.random.default_rng(0)
        weights = (rng.integers(0, 2**32, (E, IN // 8, 2 * WIDTH), dtype=np.uint32),
                   rng.integers(122, 128, (E, IN // 32, 2 * WIDTH), dtype=np.uint8),
                   rng.integers(0, 2**32, (E, WIDTH // 8, OUT), dtype=np.uint32),
                   rng.integers(122, 128, (E, WIDTH // 32, OUT), dtype=np.uint8))

        def routing(B, seed, pool, most):
            # Rows pick up to `most` distinct experts of a pool holding expert 0
            # (row 0 the most, others 0..most); slots past a row's count hold
            # expert 0, as after routing.
            r = np.random.default_rng(seed)
            experts = r.choice(E, pool, replace=False)
            experts[0] = 0
            sel = np.zeros((B, 128), np.int32)
            prob = np.zeros((B, 128), np.float32)
            cnt = np.zeros((B, 128), np.int32)
            for b in range(B):
                n = int(r.integers(0, min(most, pool) + 1)) if b else min(most, pool)
                sel[b, :n] = r.choice(experts, n, replace=False)
                prob[b, :n] = r.uniform(0.01, 1.0, n)
                cnt[b] = n
            total = np.broadcast_to(prob.sum(1, keepdims=True) + r.uniform(0.5, 2.0, (B, 1)), (B, 128))
            source = r.standard_normal((B, IN)).astype(jnp.bfloat16)
            return sel, prob, cnt, np.ascontiguousarray(total, np.float32), source

        CASES = {8: [(1, 112, 4), (3, 9, 5), (4, 1, 1), (8, 5, 0)],
                 16: [(11, 112, 3), (12, 12, 12), (13, 20, 6)]}
        for B, cases in CASES.items():
            old, new = stream(B, False), stream(B, True)
            for seed, pool, most in cases:
                sel, prob, cnt, total, source = routing(B, seed, pool, most)
                a = np.asarray(old(sel, prob, cnt, total, source, *weights))
                b = np.asarray(new(sel, prob, cnt, total, source, *weights))
                used = {int(sel[r, s]) for r in range(B) for s in range(cnt[r, 0])}
                rows_per_expert = [sum(e in sel[r, :cnt[r, 0]] for r in range(B)) for e in sorted(used)]
                assert np.isfinite(a).all() and (np.abs(a).sum(1) > 0).sum() == (cnt[:, 0] > 0).sum()
                assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (B, seed)
                print(f"B{B} seed {seed}: rows per expert {rows_per_expert} bit-identical")
        print("OK")
    """

    def test_schedule_matches_the_pending_route_loop(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestK31DraftWholeWeights(CustomTestCase):
    """``KernelConfig.draft_whole_weights``: Kimi-K3.1's draft (three heads a
    rank) holds o whole at up to 8 rows and down at up to 16 instead of
    streaming them one 896-column chunk at a time (other drafts and shapes as
    before), and the chunks ``all_reduce`` multiplies
    (``dspark_mla_kernel._ChunkedWeight``) give bit-identical products either
    way (Pallas interpret mode)."""

    SCRIPT = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla as mla
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla_kernel as kern

        for heads in (64, 96):
            c = mla.MlaDraftConfig(heads=heads)
            for rows in (4, 8, 12, 16, 24, 32, 48, 64):
                for dp in (1, 2):
                    old = rows > 32 or heads > 64  # before the knob: down; o also under DP
                    assert kern.streamed_weights(c, rows, dp) == (old or dp > 1, old)
                    if heads > 64 and dp == 1:
                        want = (rows > 8, rows > 16)
                    else:
                        want = (old or dp > 1, old)
                    assert kern.streamed_weights(c, rows, dp, whole=True) == want, (heads, rows, dp)

        LAYERS, CHUNK = 2, kern.CHUNK
        def products(lhs, weight, streamed):
            T, K = lhs.shape
            def kernel(lhs_ref, weight_hbm, out_ref, buffer, sems):
                copy = lambda src, dst, sem: pltpu.make_async_copy(src, dst, sems.at[sem])
                chunked = kern._ChunkedWeight(weight_hbm, buffer, (0, 1), copy)
                assert chunked.streamed == streamed
                for layer in range(LAYERS):
                    chunked.load(layer).wait()
                    for k, operand in enumerate(chunked.chunks(layer)):
                        out_ref[layer, k] = kern._dot(lhs_ref[...], operand)
            buffer = (2, K, CHUNK) if streamed else (K, 8 * CHUNK)
            return np.asarray(pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((LAYERS, 8, T, CHUNK), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[pltpu.VMEM(buffer, jnp.bfloat16), pltpu.SemaphoreType.DMA((2,))],
                interpret=pltpu.InterpretParams(),
            )(lhs, weight))

        rng = np.random.default_rng(0)
        for T, K in ((8, 384), (16, 512)):  # o (three heads) at 8 rows, down at 16
            lhs = jnp.asarray(rng.standard_normal((T, K)), jnp.bfloat16)
            weight = jnp.asarray(rng.standard_normal((LAYERS, K, 8 * CHUNK)), jnp.bfloat16)
            streamed, whole = products(lhs, weight, True), products(lhs, weight, False)
            assert np.array_equal(streamed.view(np.uint32), whole.view(np.uint32)), (T, K)
            ref = np.asarray(lhs, np.float32) @ np.asarray(weight, np.float32)
            ref = ref.reshape(LAYERS, T, 8, CHUNK).transpose(0, 2, 1, 3)
            np.testing.assert_allclose(whole, ref, rtol=1e-4, atol=1e-3)
        print("OK")
    """

    def test_whole_and_streamed_chunks_match(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestK31DraftFusedEpilogue(CustomTestCase):
    """``KernelConfig.draft_fused_epilogue`` and ``draft_lm_prefetch`` in
    Pallas interpret mode, against the XLA code they replace, bitwise:

    * ``paged.commit_draft_rows`` (one tile read / write per touched tile)
      lands the bytes of ``commit_latent_rows``'s scatter on a 4-rank group:
      rows within one tile, across tiles, across a page (two owners), into
      the next table entry, a negative block, a dropped block and entries
      past the table.
    * ``dspark_mla_kernel.single_epilogue`` gives ``_build_single``'s emitted
      tokens, next ids and control state, slot indices that wrap or drop
      included, and (``draft_verify_packet``) the packing op's next verify
      operand, bank writes that land in the plan row included.
    * ``dspark_mla_kernel.stream_chunks``: the LM head through six slots
      (four started early) gives the two-slot ring's logits, race-free.
    * ``MlaDsparkDraft._build_single``'s op with the knob (the fused kernel
      stubbed) returns the knob-off op's vector, pools, emitted tokens, next
      ids and control state; with ``draft_verify_packet`` (either epilogue)
      also ``DecodeOps.pack_verify_metadata`` of its next ids, control state
      and verify mode 1."""

    COMMIT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import dataclasses
        import numpy as np, ml_dtypes, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import paged

        G, ROWS, PAGES, L, BLOCKS, PT = 4, 8, 6, 5, 7, 512
        base_config = paged._config()
        scatter_config = dataclasses.replace(base_config, fused_cache_commit=False)
        stock = pl.pallas_call
        paged.pl.pallas_call = lambda *a, **k: stock(*a, **{"interpret": pltpu.InterpretParams(), **k})
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))

        def new(pools, latents, ids):
            return paged.commit_draft_rows(tuple(pools), latents.astype(jnp.float8_e4m3fn), ids, ROWS, PAGES,
                                           tuple(range(L)), group=G, global_pages=False)

        def old(pools, latents, ids):
            paged._config = lambda: scatter_config
            try:
                table = ids[3 * ROWS + 1 : 3 * ROWS + 1 + PAGES]
                positions = ids[ROWS] + jnp.arange(ROWS, dtype=jnp.int32)
                return paged.commit_latent_rows(tuple(pools), latents.transpose(1, 0, 2),
                                                jnp.broadcast_to(table[None], (ROWS, PAGES)), positions,
                                                tuple(range(L)), group=G, global_pages=False)
            finally:
                paged._config = lambda: base_config

        def run(base, table, seed):
            rng = np.random.default_rng(seed)
            pools = (rng.integers(0, 256, (L, G * BLOCKS, 128, 4, 640), dtype=np.uint8) & 0x7E).view(
                ml_dtypes.float8_e4m3fn)
            latents = (rng.standard_normal((L, ROWS, 640)) * 2).astype(ml_dtypes.bfloat16)
            ids = rng.integers(0, 100, 3 * ROWS + 1 + ROWS * PAGES).astype(np.int32)
            ids[ROWS] = base
            ids[3 * ROWS + 1 : 3 * ROWS + 1 + PAGES] = table
            outs = []
            for fn in (new, old):
                f = jax.jit(jax.shard_map(lambda *a, fn=fn: fn(a[:L], *a[L:]), mesh=mesh,
                                          in_specs=(P("tp"),) * L + (P(), P()), out_specs=(P("tp"),) * L,
                                          check_vma=False))
                outs.append([np.asarray(x).view(np.uint8) for x in f(*map(jnp.asarray, pools),
                                                                    jnp.asarray(latents), jnp.asarray(ids))])
            assert all(np.array_equal(a, b) for a, b in zip(*outs)), (base, table)
            return sum(int((a != p.view(np.uint8)).sum()) for a, p in zip(outs[1], pools))

        written = [
            run(5, [1, 2, 3, 4, 5, 6], 0),  # one tile
            run(28, [3, 1, 2, 4, 5, 6], 1),  # two tiles
            run(PT - 3, [2, 5, 1, 1, 1, 1], 2),  # two pages, two owners
            run(4 * PT - 4, [1, 6, 2, 3, 4, 5], 3),  # the next table entry
            run(3 * PT + 100, [-1, 2, 3, 4, 5, 6], 4),  # a negative block
            run(30 * PT + 500, [1, 2, 3, 4, 5, 6], 6),  # entries past the table (clamped)
        ]
        assert min(written) > 0
        assert run(2 * PT + 7, [G * BLOCKS + 3, 2, 3, 1, 1, 1], 5) == 0  # out of range: dropped
        print("OK")
    """

    EPILOGUE = """
        import itertools, os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        kc.install(kc.KernelConfig(draft_epilogue_unroll=int(os.environ.get("UNROLL", "0"))))
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla_kernel as kern
        from sglang_tpu.k3_programs.kimi_k3.kernels.dspark import OUT_BONUS, OUT_COUNT, OUT_DRAFTS

        ROWS, FIELDS = 8, 8

        def reference(vector, ids, state, slots, reset):  # MlaDsparkDraft._build_single's XLA epilogue
            rows = ROWS
            tokens, base, slot = ids[:rows], ids[rows], ids[2 * rows]
            accepted, bonus = vector[0, OUT_COUNT], vector[0, OUT_BONUS]
            row = jnp.arange(rows, dtype=jnp.int32)
            emitted = jnp.where(row < accepted, tokens[jnp.minimum(row + 1, rows - 1)],
                                jnp.where(row == accepted, bonus, -1)).astype(jnp.int32)
            following = jnp.concatenate((bonus[None], vector[0, OUT_DRAFTS : OUT_DRAFTS + rows - 1]))
            following_ids = jnp.concatenate((following, base + accepted + 1 + row, ids[2 * rows : 3 * rows],
                                             accepted[None], ids[3 * rows + 1 :])).astype(jnp.int32)
            state = state.at[slot].set(bonus)
            state = state.at[slots + slot].set(base + accepted + 1)
            state = state.at[2 * slots + slot].set(accepted)
            state = state.at[reset].set(jnp.where(slot != 0, 0, state[reset]))
            # The packing op's next verify operand (verify mode 1).
            packet = jnp.concatenate((following_ids, state[3 * slots : 3 * slots + FIELDS], jnp.ones(1, jnp.int32)))
            return emitted[None], following_ids, state, packet

        def kernel_outputs(spec, vector, ids, state):
            def kernel(ids_ref, state_ref, vector_ref, *outs):
                kern.single_epilogue(spec, ROWS, ids_ref, state_ref, vector_ref, *outs)
            sizes = [spec.ids_size, spec.state_size] + [spec.ids_size + spec.fields + 1] * (spec.fields > 0)
            return pl.pallas_call(
                kernel,
                out_shape=(jax.ShapeDtypeStruct((1, ROWS), jnp.int32),
                           *(jax.ShapeDtypeStruct((size,), jnp.int32) for size in sizes)),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=3, in_specs=[],
                    out_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * (1 + len(sizes))),
                interpret=pltpu.InterpretParams(),
            )(ids, state, vector)

        rng = np.random.default_rng(0)
        for (slots, plan_rows, pages), fields in itertools.product(((3, 1, 26), (9, 4, 7)), (0, FIELDS)):
            size = 3 * slots + plan_rows * (8 + pages)
            spec = kern.SingleEpilogue(slots=slots, reset=3 * slots + 1, state_size=size,
                                       ids_size=3 * ROWS + 1 + ROWS * pages, plan=3 * slots, fields=fields)
            for accepted in (-1, 0, 3, 7, 8):
                # slots + 1 / + 2: the row bank write lands in plan row 0 (the packet carries it)
                for slot in (0, 1, -1, slots + 1, slots + 2, size + 5):
                    vector = rng.integers(-2**31, 2**31 - 1, (8, 128), dtype=np.int64).astype(np.int32)
                    vector[0, OUT_COUNT] = accepted
                    ids = rng.integers(0, 200000, spec.ids_size).astype(np.int32)
                    ids[2 * ROWS] = slot
                    state = rng.integers(-1000, 1000, size).astype(np.int32)
                    got = kernel_outputs(spec, *map(jnp.asarray, (vector, ids, state)))
                    want = reference(jnp.asarray(vector[:1]), jnp.asarray(ids), jnp.asarray(state), slots,
                                     spec.reset)
                    assert len(got) == (4 if fields else 3)
                    for g, w in zip(got, want):
                        assert np.array_equal(np.asarray(g), np.asarray(w)), (slots, accepted, slot, fields)
        print("OK")
    """

    LM_RING = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla_kernel as kern

        H, CH, N, T = 512, kern.LM_CHUNK, 9, 8

        def logits(hidden, lm_head, early):
            def kernel(hidden_ref, lm_hbm, out_ref, gate, up, a, b, sems):
                def copy(k, slot, slots, sem_of):
                    return pltpu.make_async_copy(lm_hbm.at[:, pl.ds(k * CH, CH)], slots[slot],
                                                 sems.at[sem_of[slot]])
                if early:  # as the kernel: four gate / up halves started first, then the two scoped slots
                    slots = [buf.at[:, pl.ds(h * CH, CH)] for buf in (gate, up) for h in range(2)] + [a, b]
                    sem_of = (0, 1, 2, 3, 4, 5)
                    for k in range(4):
                        copy(k, k, slots, sem_of).start()
                else:
                    slots, sem_of = [a, b], (4, 5)
                chunks = kern.stream_chunks(lambda k, s: copy(k, s, slots, sem_of), slots, N,
                                            started=4 if early else 0)
                for k, slot in chunks:
                    out_ref[:, pl.ds(k * CH, CH)] = kern._f32(kern._bf16(kern._dot(hidden_ref[...], slot[...])))
            return np.asarray(pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((T, N * CH), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[pltpu.VMEM((H, 2 * CH), jnp.bfloat16), pltpu.VMEM((H, 2 * CH), jnp.bfloat16),
                                pltpu.VMEM((H, CH), jnp.bfloat16), pltpu.VMEM((H, CH), jnp.bfloat16),
                                pltpu.SemaphoreType.DMA((6,))],
                interpret=pltpu.InterpretParams(detect_races=True),
            )(hidden, lm_head))

        rng = np.random.default_rng(0)
        hidden = jnp.asarray(rng.standard_normal((T, H)), jnp.bfloat16)
        lm_head = jnp.asarray(rng.standard_normal((H, N * CH)), jnp.bfloat16)
        two, six = logits(hidden, lm_head, False), logits(hidden, lm_head, True)
        assert not interpreter.races.races_found
        assert np.array_equal(two.view(np.uint32), six.view(np.uint32))
        print("OK")
    """

    OP = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import contextlib, dataclasses, sys, types
        import numpy as np, ml_dtypes, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P

        # The op bodies as MlaDsparkDraft._build_single registers them (bridge stubbed), the fused kernel a stub
        # returning one vector and latents: knob on and off must return the same vector, pools, tokens, ids, state.
        captured = []
        bridge = types.ModuleType("sglang_tpu.k3_programs.bridge")
        bridge.register_op = lambda name, mesh, body, ins, outs, donate_argnums=(), export_key=None: (
            captured.append((body, ins, outs)) or body)
        for name in ("hbm", "kernel_mesh", "materialize", "upload", "report_load_stats", "stream_to_device"):
            setattr(bridge, name, lambda *a, **k: None)
        sys.modules["sglang_tpu.k3_programs.bridge"] = bridge
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig, draft_mla
        from sglang_tpu.k3_programs.kimi_k3.decode_ops import Control, DecodeOps
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark, dspark_mla_kernel as kern, mla_dcp, paged

        G, ROWS, PAGES, L, BLOCKS, PT, SLOTS = 4, 8, 6, 5, 7, 512, 3
        mla_dcp.GROUPS = (G, 8, 16, 32)
        stock = pl.pallas_call
        paged.pl.pallas_call = lambda *a, **k: stock(*a, **{"interpret": pltpu.InterpretParams(), **k})
        base_config = kconfig.active()
        for fn in (paged.commit_latent_rows, paged.commit_draft_rows):
            fn.__kwdefaults__["group"] = G

        @dataclasses.dataclass(frozen=True)
        class DraftConfig:
            layers: int = L

        def run(knob, vector, latents, pools, ids, state, decisions, packet=False):
            # The scatter commit as the reference (interpret mode has no aliased HBM refs).
            config = dataclasses.replace(base_config, draft_fused_epilogue=knob, draft_verify_packet=packet,
                                         fused_cache_commit=False)
            kconfig.active = paged._config = lambda: config
            stub = types.SimpleNamespace(**{n: getattr(kern, n) for n in (
                "WEIGHT_NAMES", "LEADING_NAMES", "SingleEpilogue", "single_epilogue", "copy_words")})
            stub.fused_draft_local = lambda *a, **k: (lambda *args: (jnp.asarray(vector), jnp.asarray(latents)))
            d = object.__new__(draft_mla.MlaDsparkDraft)
            d.dspark, d.kernel, d.paged, d.config = dspark, stub, paged, DraftConfig()
            d.attention_dp, d.max_position, d.owner_table, d._ops, d.mesh = 1, 1 << 20, None, {}, None
            d.names = ("ln1", "ln2", "q_norm", "kv_norm", "rot")
            captured.clear()
            d._build_single(Control(SLOTS, 1, PAGES))
            body, ins, outs = captured[0]
            weights = [np.zeros((G, L, 1, 64), ml_dtypes.bfloat16)] * 4 + [np.zeros((G, 64, 64), ml_dtypes.bfloat16)]
            rest = (np.zeros((3, ROWS, 64), ml_dtypes.bfloat16), np.zeros((G, ROWS, 128), np.float32),
                    np.zeros((1, 128), np.int32), ids, state, decisions)
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=ins, out_specs=outs, check_vma=False))
            out = f(*map(jnp.asarray, (*weights, *pools, *rest)))
            return [np.asarray(x).view(np.uint8) if x.dtype == jnp.float8_e4m3fn else np.asarray(x) for x in out]

        rng = np.random.default_rng(0)
        for base, accepted, slot in ((5, 3, 1), (PT - 3, 7, 2), (28, 0, 0), (4 * PT - 4, 5, -1), (40, 2, SLOTS + 2)):
            vector = rng.integers(0, 160000, (1, 128)).astype(np.int32)
            vector[0, dspark.OUT_COUNT] = accepted
            latents = (rng.standard_normal((L, ROWS, 640)) * 2).astype(ml_dtypes.bfloat16)
            pools = [(rng.integers(0, 256, (G * BLOCKS, 128, 4, 640), dtype=np.uint8) & 0x7E).view(
                ml_dtypes.float8_e4m3fn) for _ in range(L)]
            ids = rng.integers(1, BLOCKS, 3 * ROWS + 1 + ROWS * PAGES).astype(np.int32)
            ids[ROWS], ids[2 * ROWS] = base, slot
            state = rng.integers(-50, 50, 3 * SLOTS + 8 + PAGES).astype(np.int32)
            decisions = np.array([[-1, -1]], np.int32)
            args = (vector, latents, pools, ids, state, decisions)
            old = run(False, *args)
            new = run(True, *args)
            assert len(old) == len(new) == 1 + L + 3
            for k, (a, b) in enumerate(zip(old, new)):
                assert a.shape == b.shape and np.array_equal(a, b), (base, k)
            assert any((o != p.view(np.uint8)).any() for o, p in zip(old[1 : 1 + L], pools))
            # KernelConfig.draft_verify_packet: the same outputs, then the packing op's next operand.
            want = DecodeOps.pack_verify_metadata(Control(SLOTS, 1, PAGES), 1, jnp.asarray(old[-2]),
                                                  jnp.asarray(old[-1]), jnp.ones(1, jnp.int32))
            for knob in (False, True):
                *outs, packet = run(knob, *args, packet=True)
                assert all(np.array_equal(a, b) for a, b in zip(old, outs, strict=True)), (base, knob)
                assert packet.dtype == np.int32 and np.array_equal(packet, np.asarray(want)), (base, knob)
        print("OK")
    """

    def _run(self, script, **env_vars):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **env_vars)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])

    def test_single_op_matches_the_xla_op(self):
        self._run(self.OP)

    def test_commit_draft_rows_matches_the_scatter(self):
        self._run(self.COMMIT)

    def test_single_epilogue_matches_the_xla_epilogue(self):
        self._run(self.EPILOGUE)

    def test_single_epilogue_with_unrolled_copies(self):
        # KernelConfig.draft_epilogue_unroll: blocks of 8 and 3 words, then the
        # rest (these sizes leave 0-7 words past the last block).
        for unroll in ("8", "3"):
            with self.subTest(unroll=unroll):
                self._run(self.EPILOGUE, UNROLL=unroll)

    def test_lm_ring_matches_two_slots(self):
        self._run(self.LM_RING)


class TestK31DraftVerifyPacket(CustomTestCase):
    """``KernelConfig.draft_verify_packet`` in the decoder (the packets
    themselves: ``TestK31DraftFusedEpilogue``): ``KimiDecoder._speculate``
    verifies from the draft op's packet exactly when it chains (same
    request, unchanged plan) at verify mode 1, and ``_enter_batched`` /
    ``_verify_and_draft`` hand the verify the packet the packing op
    (``DecodeOps.pack_verify_metadata``) builds from the same control and
    modes."""

    HOST = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import sys, types
        import numpy as np, jax, jax.numpy as jnp, torch

        bridge = types.ModuleType("sglang_tpu.k3_programs.bridge")
        bridge.register_op = bridge.register_global_op = (
            lambda name, mesh, fn, ins, outs, donate_argnums=(), export_key=None: fn)
        for name in ("hbm", "kernel_mesh", "materialize", "upload", "report_load_stats", "stream_to_device"):
            setattr(bridge, name, lambda *a, **k: None)
        sys.modules["sglang_tpu.k3_programs.bridge"] = bridge
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig
        kconfig.install(kconfig.KernelConfig().replace(draft_verify_packet=True))
        from sglang_tpu.k3_programs.kimi_k3 import decode
        from sglang_tpu.k3_programs.kimi_k3.decode_ops import Control, DecodeOps

        class Ops:
            mesh = None
            def __init__(self):
                self.calls = []
            def cached_op(self, key, build):
                return build()
            def verify_packed(self, ids, aux_layers, control, state, modes, rates=None, packet=None):
                self.calls.append((ids, int(modes[0]), packet))
                return "sharded", "aux", "decisions"
            def verify_batched(self, ids, aux_layers, requests, rows_per, control, state, modes, rates=None,
                               packet=None):
                self.calls.append((ids, modes, packet))
                return "sharded", "aux", "decisions"

        class Draft:  # step_single's outputs tagged with the step number
            aux_layers = (1,)
            def __init__(self, packets):
                self.packets, self.steps = packets, 0
            def _synthetic_rates(self, drafts):
                return None
            def step_single(self, state, aux, logits, ids, control_state, decisions, control):
                self.steps += 1
                out = (state, "emitted", ("ids", self.steps), ("control", self.steps))
                return out + ((("packet", self.steps),) if self.packets else ())
            def step_batched(self, states, aux, logits, ids, control_state, decisions, control, rows):
                return states, "emitted", control_state

        def decoder(packets):
            d = object.__new__(decode.KimiDecoder)
            d.ops, d.draft = Ops(), Draft(packets)
            d.layout, d.control, d.device_plan = Control(5, 4, 3), ("control", 0), None
            d.spec_req = d.draft_state = d.next_ids = d.next_packet = None
            d._drafts, d._fresh, d._offered, d._group, d.tracked = {}, {}, {}, {}, set()
            d.slot_map = types.SimpleNamespace(release=lambda finished: None)
            d._modes = {0: torch.zeros(1, dtype=torch.int32), 1: torch.ones(1, dtype=torch.int32)}
            d.stats = dict.fromkeys(("plans", "enters", "chained", "packets", "plain", "batched"), 0)
            d._take_state = lambda req_id: types.SimpleNamespace(vector=("vector", req_id))
            d._enter = lambda: (lambda control, vector, keep: (("entered", int(keep[0])),))
            return d

        def step(d, req_id, reset, scheduled, changed):
            d._sync_plan = lambda entries, tables: changed
            request = types.SimpleNamespace(req_id=req_id, scheduled=scheduled)
            d._speculate(request, [(1, reset, 0, 0, None)], [None], False)
            return d.ops.calls[-1]

        d = decoder(True)
        assert step(d, "a", True, 8, True) == (("entered", 0), 0, None)  # a new request: placeholders
        assert step(d, "a", False, 8, False) == (("ids", 1), 1, ("packet", 1))  # chained: the draft op's packet
        assert step(d, "a", False, 5, False) == (("ids", 2), 0, None)  # chained, fewer rows: mode 0, packed
        assert step(d, "a", False, 8, True) == (("entered", 1), 1, None)  # a new plan: the operand rebuilt
        assert step(d, "a", False, 8, False) == (("ids", 4), 1, ("packet", 4))
        assert step(d, "b", False, 8, False) == (("entered", 0), 0, None)  # another request
        assert d.next_packet == ("packet", 6)
        d.forget(["b"])
        assert d.next_ids is None and d.next_packet is None
        assert step(d, "b", False, 8, False) == (("entered", 0), 0, None)
        d._park_single()
        assert d.next_packet is None
        assert d.stats["packets"] == 2 and d.stats["chained"] == 3 and d.stats["enters"] == 4, d.stats
        d = decoder(False)  # a draft without packets (or the knob off): always packed
        step(d, "a", True, 8, True)
        assert step(d, "a", False, 8, False) == (("ids", 1), 1, None) and d.stats["packets"] == 0

        # The batched enter op's packet is the packing op's, from the same control and modes.
        rng = np.random.default_rng(3)
        d = decoder(True)
        layout = d.layout
        for requests, rows_per in ((2, 4), (4, 8)):
            control = rng.integers(0, 200, layout.size).astype(np.int32)
            fields = control[layout.plan_start : layout.tables_start].reshape(layout.rows, layout.FIELDS)
            fields[:, layout.SLOT] = rng.integers(0, layout.slots, layout.rows)
            fields[:, layout.RESET] = rng.integers(0, 2, layout.rows)
            control[layout.plan_start : layout.tables_start] = fields.reshape(-1)
            control = jnp.asarray(control)
            vectors = [jnp.asarray(rng.integers(0, 160000, (1, 128)).astype(np.int32)) for _ in range(requests)]
            modes = jnp.asarray(rng.integers(0, 2, requests).astype(np.int32))
            (ids,) = d._enter_batched(requests, rows_per)(control, *vectors, modes)
            ids2, packet = d._enter_batched(requests, rows_per, packet=True)(control, *vectors, modes)
            want = DecodeOps.pack_verify_metadata(layout, requests, ids, control, modes)
            assert np.array_equal(np.asarray(ids), np.asarray(ids2))
            assert np.asarray(packet).dtype == np.int32 and np.array_equal(np.asarray(packet), np.asarray(want))
            d.control = control
            states = [types.SimpleNamespace(vector=v) for v in vectors]
            d._verify_and_draft(requests, rows_per, states, modes)
            got_ids, _, got_packet = d.ops.calls[-1]
            assert np.array_equal(np.asarray(got_ids), np.asarray(ids))
            assert np.array_equal(np.asarray(got_packet), np.asarray(want))
        print("OK")
    """

    def _run(self, script):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])

    def test_decoder_verifies_from_the_packet(self):
        self._run(self.HOST)


class TestB16OneHotStaging(CustomTestCase):
    """KernelConfig.b16_one_hot_staging: the B16 routed-expert stream stages
    its compact rows with a one-hot product instead of a scalar row scan. In
    Pallas interpret mode both give bitwise-identical outputs over routes that
    take every pair and tail path, with NaN/inf latents in the rows without
    routes (null slots; the one-hot path zeroes them) and signed zeros in
    routed rows."""

    SCRIPT = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from types import SimpleNamespace
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

B, C, K, W, OUT, SLOTS = 16, 8, 512, 256, 128, 4
NAMES = ("gate_up", "gate_up_scales", "down", "down_scales")


def stream(one_hot, latents, experts, probabilities, counts, totals, weights):
    dmk.B16_ONE_HOT_STAGING = one_hot  # read when the stream is traced
    k = experts.shape[1]

    def kernel(latent_ref, expert_ref, probability_ref, count_ref, total_ref, *refs):
        hbm, (out_ref, source_ref), vmem = refs[:4], refs[4:6], refs[6:10]
        gate_up_acc, down_acc, processed, first, second, semaphores = refs[10:]
        source_ref[...] = latent_ref[...]

        def copy_expert(expert, slot, *, gate_up=True, down=True):
            wanted = (gate_up, gate_up, down, down)
            return SimpleNamespace(**{
                name: pltpu.make_async_copy(hbm[i].at[expert], vmem[i].at[slot],
                                            semaphores.at[slot, i]) if wanted[i] else None
                for i, name in enumerate(NAMES)})

        selected, route_counts = expert_ref[:, :k], count_ref[:, 0]
        dmk._start_initial_expert_weight_copies(
            selected, route_counts, copy_expert, slot_count=SLOTS, already_started=0,
            group_pending_routes=True)
        dmk._run_routed_expert_stream(
            source_ref, selected, probability_ref[:, :k], route_counts, total_ref[:, 0], *vmem,
            out_ref, gate_up_acc, down_acc, processed, first, second, copy_expert,
            expert_input_size=K, expert_width=W, expert_output_size=OUT,
            group_pending_routes=True)

    vmem, hbm = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)
    wide = lambda x: np.pad(x.reshape(B, -1), ((0, 0), (0, 128 - x.size // B)), mode="edge")
    out, source = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                   jax.ShapeDtypeStruct(latents.shape, jnp.bfloat16)),
        in_specs=[vmem] * 5 + [hbm] * 4,
        out_specs=(vmem, vmem),
        scratch_shapes=[*(pltpu.VMEM((SLOTS, *w.shape[1:]), w.dtype) for w in weights),
                        pltpu.VMEM((2 * C, 4 * W), jnp.float32), pltpu.VMEM((C, OUT), jnp.float32),
                        pltpu.VMEM((B, 128), jnp.int32), pltpu.VMEM((C, K), jnp.float32),
                        pltpu.VMEM((C, K), jnp.float32), pltpu.SemaphoreType.DMA((SLOTS, 4))],
        interpret=pltpu.InterpretParams(),
    )(latents, wide(experts), wide(probabilities), wide(counts), wide(totals), *weights)
    return np.asarray(out), np.asarray(source)


def case(seed, live, local_experts, k):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((B, K + 128)).astype(np.float32)
    rows = rng.choice(B, live, replace=False)
    dead = np.setdiff1d(np.arange(B), rows)
    if dead.size:
        latents[dead[:, None], rng.integers(0, K, 3)[None, :]] = (np.nan, np.inf, -np.inf)
        latents[dead[0]] = np.nan
    for r in rows[:3]:
        latents[r, rng.integers(0, K, 6)] = (0.0, -0.0, -0.0, 1e-30, -1e-30, 3e4)
    popularity = rng.dirichlet(np.full(local_experts, 0.6))
    experts, probabilities = np.zeros((B, k), np.int32), np.zeros((B, k), np.float32)
    counts = np.zeros(B, np.int32)
    for r in rows:
        n = counts[r] = rng.integers(1, k + 1)
        experts[r, :n] = rng.choice(local_experts, n, replace=False, p=popularity)
        probabilities[r, :n] = rng.uniform(0.02, 0.2, n)
    totals = np.where(counts > 0, probabilities.sum(1) + rng.uniform(0.5, 1, B), 1)
    bits = lambda *shape: rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    exponents = lambda *shape: rng.integers(119, 123, shape).astype(np.uint8)
    weights = (bits(local_experts, K // 8, 2 * W), exponents(local_experts, K // 32, 2 * W),
               bits(local_experts, W // 8, OUT), exponents(local_experts, W // 32, OUT))
    return (latents.astype(ml_dtypes.bfloat16), experts, probabilities, counts,
            totals.astype(np.float32), weights)


def paths(experts, counts):
    # Distinct experts run in pairs in id order; their row counts pick the path.
    routed = (np.arange(experts.shape[1]) < counts[:, None])
    sizes = [n for e in range(experts.max() + 1) if (n := int(np.sum(routed & (experts == e))))]
    kinds = set()
    for i in range(0, len(sizes), 2):
        pair = sizes[i:i + 2]
        size = "pass" if max(pair) > C else "single" if max(pair) == 1 else "multi"
        kinds.add(f"{size}_{'pair' if len(pair) == 2 else 'tail'}")
    return kinds


covered = set()
for seed, live, local_experts, k in ((11, 13, 9, 4), (10, 14, 5, 3), (1, 3, 5, 3)):
    latents, experts, probabilities, counts, totals, weights = case(seed, live, local_experts, k)
    covered |= paths(experts, counts)
    old, old_source = stream(False, latents, experts, probabilities, counts, totals, weights)
    new, new_source = stream(True, latents, experts, probabilities, counts, totals, weights)
    routed = counts > 0
    assert np.array_equal(old.view(np.uint32), new.view(np.uint32)), seed
    assert np.all(np.isfinite(new)) and np.all(np.any(old[routed] != 0, axis=1)), seed
    assert np.array_equal(old_source.view(np.uint16), latents.view(np.uint16)), seed
    assert np.array_equal(new_source[routed].view(np.uint16), latents[routed].view(np.uint16))
    assert np.all(new_source[~routed, :K].astype(np.float32) == 0), seed
assert covered == {f"{s}_{p}" for s in ("single", "multi", "pass") for p in ("pair", "tail")}, covered
print("OK")
"""

    def test_matches_the_row_scan(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestSharedExpertPasses(CustomTestCase):
    """KernelConfig.shared_expert_passes: the routed-expert stream with compact
    tiles (B16, B32) runs an iteration's noncompact pair or tail (experts with
    more rows than a compact tile) from one copy of the eight-row passes, a
    loop over its one or two experts, instead of three unrolled copies. In
    Pallas interpret mode both give bitwise-identical outputs over routes that
    take every pair and tail path, with and without the SMEM expert schedule
    and B16 one-hot staging (NaN/inf latents in rows without routes, signed
    zeros in routed rows)."""

    SCRIPT = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from types import SimpleNamespace
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

C, K, W, OUT, SLOTS = 8, 512, 256, 128, 4
NAMES = ("gate_up", "gate_up_scales", "down", "down_scales")


def stream(B, shared, scheduled, one_hot, latents, experts, probabilities, counts, totals, weights):
    dmk.SHARED_EXPERT_PASSES = shared  # read when the stream is traced
    dmk.B16_ONE_HOT_STAGING = one_hot
    k = experts.shape[1]
    rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)

    def kernel(latent_ref, expert_ref, probability_ref, count_ref, total_ref, *refs):
        hbm, (out_ref, source_ref), vmem = refs[:4], refs[4:6], refs[6:10]
        gate_up_acc, down_acc, processed, first, second, semaphores, table, table_semaphore = refs[10:]
        source_ref[...] = latent_ref[...]

        def copy_expert(expert, slot, *, gate_up=True, down=True):
            wanted = (gate_up, gate_up, down, down)
            return SimpleNamespace(**{
                name: pltpu.make_async_copy(hbm[i].at[expert], vmem[i].at[slot],
                                            semaphores.at[slot, i]) if wanted[i] else None
                for i, name in enumerate(NAMES)})

        selected, route_counts = expert_ref[:, :k], count_ref[:, 0]
        dmk._start_initial_expert_weight_copies(
            selected, route_counts, copy_expert, slot_count=SLOTS, already_started=0,
            group_pending_routes=True)
        schedule = None
        if scheduled:
            schedule = dmk._expert_schedule(processed.at[:8], table, table_semaphore)
            dmk._start_expert_schedule(schedule, selected, route_counts)
        dmk._run_routed_expert_stream(
            source_ref, selected, probability_ref[:, :k], route_counts, total_ref[:, 0], *vmem,
            out_ref, gate_up_acc, down_acc, processed, first, second, copy_expert,
            expert_input_size=K, expert_width=W, expert_output_size=OUT,
            group_pending_routes=True, expert_schedule=schedule)

    vmem, hbm = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)
    wide = lambda x: np.pad(x.reshape(B, -1), ((0, 0), (0, 128 - x.size // B)), mode="edge")
    out, source = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                   jax.ShapeDtypeStruct(latents.shape, jnp.bfloat16)),
        in_specs=[vmem] * 5 + [hbm] * 4,
        out_specs=(vmem, vmem),
        scratch_shapes=[*(pltpu.VMEM((SLOTS, *w.shape[1:]), w.dtype) for w in weights),
                        pltpu.VMEM((rows[0], 4 * W), jnp.float32),
                        pltpu.VMEM((rows[1], OUT), jnp.float32),
                        pltpu.VMEM((B, 128), jnp.int32), pltpu.VMEM((C, K), jnp.float32),
                        pltpu.VMEM((C, K), jnp.float32), pltpu.SemaphoreType.DMA((SLOTS, 4)),
                        pltpu.SMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(),
    )(latents, wide(experts), wide(probabilities), wide(counts), wide(totals), *weights)
    return np.asarray(out), np.asarray(source)


def case(B, seed, live, local_experts, k):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((B, K + 128)).astype(np.float32)
    rows = rng.choice(B, live, replace=False)
    dead = np.setdiff1d(np.arange(B), rows)
    if dead.size:
        latents[dead[:, None], rng.integers(0, K, 3)[None, :]] = (np.nan, np.inf, -np.inf)
        latents[dead[0]] = np.nan
    for r in rows[:3]:
        latents[r, rng.integers(0, K, 6)] = (0.0, -0.0, -0.0, 1e-30, -1e-30, 3e4)
    popularity = rng.dirichlet(np.full(local_experts, 0.6))
    experts, probabilities = np.zeros((B, k), np.int32), np.zeros((B, k), np.float32)
    counts = np.zeros(B, np.int32)
    for r in rows:
        n = counts[r] = rng.integers(1, k + 1)
        experts[r, :n] = rng.choice(local_experts, n, replace=False, p=popularity)
        probabilities[r, :n] = rng.uniform(0.02, 0.2, n)
    totals = np.where(counts > 0, probabilities.sum(1) + rng.uniform(0.5, 1, B), 1)
    bits = lambda *shape: rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    exponents = lambda *shape: rng.integers(119, 123, shape).astype(np.uint8)
    weights = (bits(local_experts, K // 8, 2 * W), exponents(local_experts, K // 32, 2 * W),
               bits(local_experts, W // 8, OUT), exponents(local_experts, W // 32, OUT))
    return (latents.astype(ml_dtypes.bfloat16), experts, probabilities, counts,
            totals.astype(np.float32), weights)


def paths(experts, counts):
    # Distinct experts run in pairs in id order; their row counts pick the path.
    routed = (np.arange(experts.shape[1]) < counts[:, None])
    sizes = [n for e in range(experts.max() + 1) if (n := int(np.sum(routed & (experts == e))))]
    kinds = set()
    for i in range(0, len(sizes), 2):
        pair = sizes[i:i + 2]
        size = "pass" if max(pair) > C else "single" if max(pair) == 1 else "multi"
        kinds.add(f"{size}_{'pair' if len(pair) == 2 else 'tail'}")
    return kinds


# (batch, schedule, one-hot staging): B16 as before and as served; B32 (no schedule).
CASES = {(16, False, False): ((11, 13, 9, 4), (10, 14, 5, 3), (1, 3, 5, 3), (22, 16, 4, 4)),
         (16, True, True): ((11, 13, 9, 4), (10, 14, 5, 3), (1, 3, 5, 3), (22, 16, 4, 4)),
         (32, False, False): ((37, 20, 7, 3), (32, 28, 6, 4))}
for (B, scheduled, one_hot), cases in CASES.items():
    covered = set()
    for seed, live, local_experts, k in cases:
        latents, experts, probabilities, counts, totals, weights = case(B, seed, live, local_experts, k)
        covered |= paths(experts, counts)
        inputs = (latents, experts, probabilities, counts, totals, weights)
        old, old_source = stream(B, False, scheduled, one_hot, *inputs)
        new, new_source = stream(B, True, scheduled, one_hot, *inputs)
        routed = counts > 0
        assert np.array_equal(old.view(np.uint32), new.view(np.uint32)), (B, seed)
        assert np.array_equal(old_source.view(np.uint16), new_source.view(np.uint16)), (B, seed)
        assert np.all(np.isfinite(new)) and np.all(np.any(old[routed] != 0, axis=1)), (B, seed)
    want = {"pass_pair", "pass_tail", "multi_pair"}
    if B == 16:
        want = {f"{s}_{p}" for s in ("single", "multi", "pass") for p in ("pair", "tail")}
    assert want <= covered, (B, covered)
print("OK")
"""

    def test_one_copy_matches_the_unrolled_passes(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestSingletonRowStaging(CustomTestCase):
    """A singleton route's row staging (``_stage_singleton_row``) in Pallas
    interpret mode: at 8 rows, ``KernelConfig.b8_singleton_select`` (one select
    over the row tile, no branch) stages the bits of the per-row copies for
    every active row and every BF16 bit pattern (signed zeros, subnormals,
    infinities, NaN payloads); other row counts keep the per-row copies."""

    def test_select_matches_the_row_copies(self):
        import jax
        import jax.numpy as jnp
        import ml_dtypes
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        width = 3584  # expert input columns of the [rows, 4096] latent tile
        # -0, +0, -/+ smallest subnormal, -/+ largest finite, -/+ infinity,
        # NaNs (quiet, signalling, payloads, negative), +/-1.
        special = np.array(
            [0x8000, 0x0000, 0x8001, 0x0001, 0xFF7F, 0x7F7F, 0xFF80, 0x7F80]
            + [0x7FC0, 0x7F81, 0x7FC1, 0xFFC0, 0xFFFF, 0x7FFF, 0x3F80, 0xBF80],
            np.uint16,
        )

        def staging(rows, row_select):
            def kernel(routes_ref, source_ref, out_ref):
                for case in range(rows):
                    dmk._stage_singleton_row(
                        source_ref,
                        jnp.any(routes_ref[case] != 0, axis=1),
                        out_ref.at[case],
                        expert_input_size=width,
                        block_select=False,
                        row_select=row_select,
                    )

            return pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((rows, 1, width), jnp.bfloat16),
                interpret=pltpu.InterpretParams(),
            )

        def inputs(rows):
            bits = rng.integers(0, 1 << 16, (rows, 4096), dtype=np.uint16)
            # Each row holds every special pattern; in each of those columns
            # the other rows hold other ones.
            for column in range(0, width, 448):
                for row in range(rows):
                    bits[row, column : column + 16] = np.roll(special, row)
            # Case r: row r is the only row with a route to the expert.
            routes = np.zeros((rows, rows, 16), np.int32)
            for row in range(rows):
                routes[row, row, rng.integers(0, 16)] = 1 + rng.integers(0, 128)
            return bits, routes

        def branches(call, *args):
            return str(jax.make_jaxpr(call)(*args)).count("cond[")

        rng = np.random.default_rng(0)
        bits, routes = inputs(8)
        source = bits.view(ml_dtypes.bfloat16)
        copy, select = staging(8, False), staging(8, True)
        self.assertEqual(branches(copy, routes, source), 8 * 8)
        self.assertEqual(branches(select, routes, source), 0)
        for call in (copy, select):
            staged = np.asarray(call(routes, source)).view(np.uint16)
            np.testing.assert_array_equal(staged[:, 0], bits[:, :width])
        # Other row counts keep the per-row copies.
        bits, routes = inputs(12)
        source = bits.view(ml_dtypes.bfloat16)
        self.assertEqual(branches(staging(12, True), routes, source), 12 * 12)


class TestAttentionHostBf16(CustomTestCase):
    """The token-scattered attention output projection (``_project_attention_2d``)
    at B8, B16 and B24 in Pallas interpret mode on 32 host devices (4 hosts x 4 rank
    pairs x 2 lanes), with stale NaN bytes wherever a round lands. With FP32
    host sums and with BF16 host sums (``KernelConfig.attention_host_bf16``)
    every rank's owned rows equal, bit for bit, the reduction's rounding points
    applied to the ranks' own FP32 partials: BF16 pair partials, FP32 sums of
    each host's four pairs in pair order, the host sums kept in FP32 or rounded
    to BF16, an FP32 sum of the four hosts in host order, a BF16 output. The four
    hosts' copies of every row are bitwise equal."""

    # The interpret harness (shared with TestAttentionRankRows): bitcast and
    # reshape views, ``project`` and the rounding-point reference.
    PRELUDE = r"""
import math, os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.state import indexing, types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec as P
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

BF16 = ml_dtypes.bfloat16
H, HALF, W = 7168, 3584, 384


# TPU interpret mode reads and writes NDIndexer views only; give its loads,
# stores and DMAs the kernel's bitcast and reshape views too (TPU semantics: a
# 32-bit row r is 16-bit rows 2r and 2r + 1; dmk._PaddedBf16Rows is the leading
# rows of its backing).
def view_index(content, transforms):
    units = content.dtype.itemsize // 2
    index = np.arange(content.size * units).reshape(*content.shape, units)
    dtype = content.dtype
    for t in transforms:
        if isinstance(t, indexing.NDIndexer):
            index = index[interpret_utils.to_range((t,))]
        elif isinstance(t, state_types.BitcastTransform):
            dtype = np.dtype(t.dtype)
            new, (*lead, rows, cols, old) = dtype.itemsize // 2, index.shape
            if new < old:
                index = index.swapaxes(-1, -2).reshape(*lead, rows * old, cols, 1)
            elif new > old:
                index = index.reshape(*lead, rows // new, new, cols).swapaxes(-1, -2)
        elif isinstance(t, state_types.ReshapeTransform):
            shape = tuple(t.shape)
            if math.prod(shape) == math.prod(index.shape[:-1]):
                index = index.reshape(*shape, index.shape[-1])
            else:
                index = index.reshape(-1, shape[-1], index.shape[-1])[: shape[0]]
        else:
            raise NotImplementedError(t)
    return index, dtype


def access(device_id, local_core_id, memory_space, buffer_id, transforms, val=None):
    memory = ipc._get_shared_memory()
    space = ipc.TPU_MEMORY_SPACE_NAMES[int(memory_space)]
    core = ipc._local_core_id_or_zero_if_hbm(int(local_core_id), space)
    content = memory.mem[(space, int(buffer_id), int(device_id), core)].content
    with memory.lock:
        index, dtype = view_index(content, transforms)
        flat = content.reshape(-1).view(np.uint16)
        old = np.ascontiguousarray(flat[index]).view(dtype)[..., 0]
        if val is not None:
            new = np.ascontiguousarray(np.asarray(val, dtype)[..., None]).view(np.uint16)
            flat[index] = new.reshape(index.shape)
    return old


ORIGINAL = {name: getattr(ipc, name).__wrapped__ for name in ("get", "store", "swap")}
plain = lambda transforms: all(isinstance(t, indexing.NDIndexer) for t in transforms)


@ipc.fail_on_exception
def get(token, device_id, core, space, buffer_id, transforms, *args, **kwargs):
    transforms = jax.tree.map(int, transforms)
    if plain(transforms):
        return ORIGINAL["get"](token, device_id, core, space, buffer_id, transforms, *args, **kwargs)
    return token, access(device_id, core, space, buffer_id, transforms)


@ipc.fail_on_exception
def store(token, device_id, core, space, buffer_id, transforms, val, *args, **kwargs):
    transforms = jax.tree.map(int, transforms)
    if plain(transforms):
        return ORIGINAL["store"](token, device_id, core, space, buffer_id, transforms, val, *args, **kwargs)
    access(device_id, core, space, buffer_id, transforms, val)
    return token


@ipc.fail_on_exception
def swap(token, device_id, core, space, buffer_id, transforms, val, mask, **kwargs):
    transforms = jax.tree.map(int, transforms)
    if plain(transforms) or mask is not None:
        return ORIGINAL["swap"](token, device_id, core, space, buffer_id, transforms, val, mask, **kwargs)
    return token, access(device_id, core, space, buffer_id, transforms, val)


ipc.get, ipc.store, ipc.swap = get, store, swap
mesh = Mesh(np.asarray(jax.devices()), ("tp",))


def project(attended, weight, host_bf16, rank_rows=False):
    # Every rank's owned rows (native 16-row backing) and FP32 partial.
    B = attended.shape[1]
    dmk.ATTENTION_HOST_BF16 = host_bf16  # read when the kernel is traced
    scratch = dmk.communication_scratch(B)
    backing = max(16, -(-B // 16) * 16)

    def kernel(attended_ref, weight_ref, owned_out, partial_out, out, comm, local, host,
               features, outputs, sends, recvs, phase):
        phase[0] = 0
        local[...] = jnp.full(local.shape, jnp.nan, jnp.float32)
        host[...] = jnp.full(host.shape, jnp.nan, jnp.float32)
        features[...] = jnp.full(features.shape, jnp.nan, jnp.bfloat16)
        outputs[...] = jnp.full(outputs.shape, jnp.nan, jnp.bfloat16)
        dmk._project_attention_2d(
            attended_ref[...], weight_ref, out, comm, local, host, features, outputs,
            sends, recvs, phase, dmk._dot, token_scatter_output=True,
            rank_rows=rank_rows)
        owned_out[...] = outputs[...]
        partial_out[...] = comm[...]

    vmem = pltpu.VMEM

    def body(attended, weight):
        owned, partial = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((backing, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, 8192), jnp.float32)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((B, 8192), jnp.float32),
                            scratch[0], scratch[2], vmem((backing, 1024), jnp.bfloat16),
                            vmem((backing, H), jnp.bfloat16), pltpu.SemaphoreType.DMA((16,)),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SMEM((1,), jnp.int32)],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(attended[0], weight[0])
        return owned[None], partial[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"), P("tp")),
                              out_specs=(P("tp"), P("tp")), check_vma=False))
    return [np.asarray(x) for x in f(attended, weight)]


f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(BF16)


def ordered_sum(parts):
    total = parts[0]
    for part in parts[1:]:
        total = f32(total + part)
    return total


def reference(partial, B, host_bf16):
    tpp = B // 4
    out = np.zeros((32, tpp, H), BF16)
    for t in range(4):
        rows = slice(t * tpp, (t + 1) * tpp)
        halves = []
        for q in range(2):
            hosts = [ordered_sum([f32(bf(partial[8 * h + 2 * p + q, rows, :HALF]))
                                  for p in range(4)]) for h in range(4)]
            if host_bf16:
                hosts = [f32(bf(s)) for s in hosts]
            halves.append(bf(ordered_sum(hosts)))
        for h in range(4):
            out[8 * h + 2 * t: 8 * h + 2 * t + 2] = np.concatenate(halves, axis=1)
    return out
"""

    SCRIPT = PRELUDE + r"""
rng = np.random.default_rng(0)
for B in (8, 16, 24):
    tpp = B // 4
    attended = (rng.standard_normal((32, B, W)) * rng.uniform(0.2, 2.0, (32, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, B, W))))).astype(BF16)
    full = (rng.standard_normal((32 * W, H)) * 0.01).astype(BF16)
    # Rank r's shard: its pair's 768 features x its lane's output half.
    weight = np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                       for r in range(32)])
    rows = {}
    for host_bf16 in (False, True):
        owned, partial = project(attended, weight, host_bf16)
        got = owned[:, :tpp]
        want = reference(partial, B, host_bf16)
        assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), (B, host_bf16)
        for r in range(32):  # the four hosts' copies
            for offset in (8, 16, 24):
                assert np.array_equal(got[r].view(np.uint16), got[r ^ offset].view(np.uint16))
        rows[host_bf16] = np.concatenate([got[2 * t] for t in range(4)])
        print(f"B{B} host_bf16={host_bf16}: bit-exact")
    changed = np.mean(rows[False] != rows[True])
    assert 0.2 < changed < 0.6, changed  # the extra rounding moves a share of outputs by an ulp
print("OK")
"""

    def test_host_sums_match_their_rounding_points(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        # 32 interpret devices block in callbacks: give the CPU client enough
        # threads (XLA reads NPROC for its pool size).
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestAttentionRankRows(CustomTestCase):
    """``KernelConfig.attention_rank_rows`` in the token-scattered attention
    output projection (``_project_attention_2d(rank_rows=True)``) at B8 and
    B16 in Pallas interpret mode on 32 host devices (TestAttentionHostBf16's
    harness, stale NaN bytes wherever a round lands). With FP32 and with BF16
    host sums, rank 2p + q's rows (one at B8, two at B16) equal, bit for bit,
    rows q r .. q r + r - 1 of the rows its pair holds without the knob (the
    same inputs) and the reduction's rounding points; the hosts' copies are
    bitwise equal. The MoE input gather of those rows (eight local owners on
    one semaphore, one-row wires in private storage) delivers every local
    rank's rows in owner order."""

    SCRIPT = TestAttentionHostBf16.PRELUDE + r"""
rng = np.random.default_rng(1)
for B in (8, 16):
    tpp, rows = B // 4, B // 8
    attended = (rng.standard_normal((32, B, W)) * rng.uniform(0.2, 2.0, (32, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, B, W))))).astype(BF16)
    full = (rng.standard_normal((32 * W, H)) * 0.01).astype(BF16)
    weight = np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                       for r in range(32)])
    for host_bf16 in (False, True):
        pair_owned, partial = project(attended, weight, host_bf16)
        rank_owned, rank_partial = project(attended, weight, host_bf16, rank_rows=True)
        assert np.array_equal(partial.view(np.uint32), rank_partial.view(np.uint32))
        want = reference(partial, B, host_bf16)
        for r in range(32):
            mine = slice((r % 2) * rows, (r % 2 + 1) * rows)
            got = rank_owned[r, :rows].view(np.uint16)
            assert np.array_equal(got, want[r, mine].view(np.uint16)), (B, host_bf16, r)
            assert np.array_equal(got, pair_owned[r, mine].view(np.uint16)), (B, host_bf16, r)
            for offset in (8, 16, 24):
                assert np.array_equal(got, rank_owned[r ^ offset, :rows].view(np.uint16))
        print(f"B{B} host_bf16={host_bf16}: rank rows bit-exact")


def gather(rows_in):
    # The MoE input gather of rank rows: eight local owners, one semaphore,
    # one-row wires in private storage (B8) or in-place copies (B16).
    rows = rows_in.shape[1]

    def kernel(rows_ref, out_ref, wires, dest, sends, recvs):
        wires[...] = jnp.full(wires.shape, jnp.nan, jnp.float32)
        dest[...] = jnp.full(dest.shape, jnp.nan, jnp.bfloat16)
        dmk._gather_local_pair_rows(
            rows_ref, dest, sends, recvs, wires, owners=8,
            semaphore=dmk.RANK_ROWS_GATHER_SEMAPHORE, transpose_wires=True)
        out_ref[...] = dest[...]

    vmem = pltpu.VMEM

    def body(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8 * rows, H), jnp.bfloat16),
            in_specs=[pl.BlockSpec(memory_space=vmem)],
            out_specs=pl.BlockSpec(memory_space=vmem),
            scratch_shapes=[vmem((8, 32, 128), jnp.float32), vmem((8 * rows, H), jnp.bfloat16),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SemaphoreType.DMA((16,))],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(x[0])[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"),
                              check_vma=False))
    return np.asarray(f(rows_in))


for rows in (1, 2):
    rows_in = rng.standard_normal((32, rows, H)).astype(BF16)
    got = gather(rows_in)
    for r in range(32):
        want = rows_in[8 * (r // 8):8 * (r // 8) + 8].reshape(8 * rows, H)
        assert np.array_equal(got[r].view(np.uint16), want.view(np.uint16)), (rows, r)
    print(f"gather of {rows}-row owners bit-exact")
print("OK")
"""

    def test_rank_rows_match_the_pair_rows(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestFirstPairGateUpCopies(CustomTestCase):
    """The initial routed-expert copies with ``gate_up_first_pair``
    (``KernelConfig.decode_late_shared_down``) in Pallas interpret mode: on
    random grouped and per-slot routings at B8 and B16 (with prefetched slots
    and rows without routes), every slot copies the same expert's same weight
    families as the slot-by-slot order, and the copies run in the order
    first two slots' gate/up and scales, their down and scales, then the
    remaining slots."""

    SCRIPT = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        from types import SimpleNamespace
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        FAMILIES = ("gate_up", "gate_up_scales", "down", "down_scales")

        class Copy:
            def __init__(self, experts, order, counter, slot, family, expert):
                self.refs = experts, order, counter
                self.at = slot, family
                self.expert = expert

            def start(self):
                experts, order, counter = self.refs
                slot, family = self.at
                experts[slot, family] = self.expert + 1
                order[slot, family] = counter[0]
                counter[0] = counter[0] + 1

        def run(experts_in, counts_in, grouped, started, first_pair):
            def kernel(experts_ref, counts_ref, experts_out, order_out, counter):
                experts_out[...] = jnp.zeros(experts_out.shape, jnp.int32)
                order_out[...] = jnp.full(order_out.shape, -1, jnp.int32)
                counter[0] = 0

                def copy_expert(expert, slot, *, gate_up=True, down=True):
                    wanted = (gate_up, gate_up, down, down)
                    return SimpleNamespace(**{
                        name: Copy(experts_out, order_out, counter, slot, family, expert)
                        if wanted[family] else None
                        for family, name in enumerate(FAMILIES)})

                dmk._start_initial_expert_weight_copies(
                    experts_ref[...], counts_ref[...][:, 0], copy_expert, slot_count=4,
                    already_started=started, group_pending_routes=grouped,
                    gate_up_first_pair=first_pair)

            smem = pltpu.SMEM
            return [np.asarray(x) for x in pl.pallas_call(
                kernel,
                out_shape=(jax.ShapeDtypeStruct((4, 4), jnp.int32),
                           jax.ShapeDtypeStruct((4, 4), jnp.int32)),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
                out_specs=[pl.BlockSpec(memory_space=smem)] * 2,
                scratch_shapes=[smem((1,), jnp.int32)],
                interpret=pltpu.InterpretParams(),
            )(experts_in, counts_in)]

        rng = np.random.default_rng(0)
        cases = 0
        for B in (8, 16):
            for trial in range(6):
                pool = rng.choice(112, size=rng.integers(2, 12), replace=False)
                experts = rng.choice(pool, size=(B, 16)).astype(np.int32)
                counts = rng.integers(0, 4, size=B).astype(np.int32)
                counts[rng.integers(B)] = 0
                counts = np.broadcast_to(counts[:, None], (B, 128)).copy()
                for grouped in (True, False):
                    for started in (0, 1):
                        plain = run(experts, counts, grouped, started, False)
                        paired = run(experts, counts, grouped, started, True)
                        assert np.array_equal(plain[0], paired[0]), (B, trial, grouped)
                        live = paired[1] >= 0
                        assert np.array_equal(live, plain[1] >= 0)
                        # Each slot copies all four families or none.
                        assert all(row.all() or not row.any() for row in live)
                        slots = [s for s in range(4) if live[s].all()]
                        want = [(s, f) for s in slots[:2] if s < 2 for f in (0, 1)]
                        want += [(s, f) for s in slots[:2] if s < 2 for f in (2, 3)]
                        want += [(s, f) for s in slots if s >= 2 for f in range(4)]
                        got = sorted(want, key=lambda sf: paired[1][sf])
                        assert got == want, (got, want)
                        assert sorted(paired[1][live]) == list(range(len(want)))
                        cases += 1
        print("cases", cases)
        print("OK")
    """

    def test_same_copies_first_pair_gate_up_first(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestKdaFp8Families(CustomTestCase):
    """KernelConfig.kda_fp8_projections at load: each column of the decode's
    KDA input and output projections becomes FP8 e4m3 of the column divided by
    its absolute maximum / 448 (rounded to nearest), with that FP32 scale."""

    def test_columns_scale_to_448_and_round_to_nearest(self):
        import ml_dtypes

        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        rng = np.random.default_rng(0)
        weight = (rng.standard_normal((256, 384)) * rng.uniform(1e-3, 1, 384)).astype(
            ml_dtypes.bfloat16
        )
        weight[:, 5] = 0
        values, scale = rank_loader.quantize_columns_fp8(weight)
        self.assertEqual(values.dtype, ml_dtypes.float8_e4m3fn)
        self.assertEqual((values.shape, scale.shape), ((256, 384), (1, 384)))
        self.assertEqual(scale.dtype, np.float32)
        exact = weight.astype(np.float32)
        peak = np.abs(exact).max(axis=0)
        np.testing.assert_array_equal(scale[0, 5], 1)
        np.testing.assert_array_equal(values[:, 5].astype(np.float32), 0)
        live = peak > 0
        np.testing.assert_array_equal(scale[0, live], peak[live] / np.float32(448))
        # The column maximum lands on 448; every value is the nearest FP8.
        decoded = values.astype(np.float32)
        np.testing.assert_array_equal(np.abs(decoded[:, live]).max(axis=0), 448)
        np.testing.assert_array_equal(
            decoded, (exact / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
        )
        # Normal-range values are within half an FP8 ulp (2^-4 relative).
        normal = np.abs(exact / scale) >= 2.0**-6
        error = np.abs(decoded * scale - exact)[normal] / np.abs(exact)[normal]
        self.assertLessEqual(error.max(), 2.0**-4 + 1e-6)

    def test_families_replace_both_projections_per_layer(self):
        import ml_dtypes

        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        rng = np.random.default_rng(1)
        families = {
            "k_projection": (rng.standard_normal((3, 64, 256)) * 0.02).astype(
                ml_dtypes.bfloat16
            ),
            "k_o": (rng.standard_normal((3, 96, 128)) * 0.02).astype(
                ml_dtypes.bfloat16
            ),
            "k_gate": np.ones((3, 64, 128), ml_dtypes.bfloat16),
        }
        sources = {name: families[name].copy() for name in ("k_projection", "k_o")}
        converted = rank_loader.kda_fp8_families(families)
        self.assertEqual(sorted(families), ["k_gate"])  # BF16 sources popped
        self.assertEqual(
            sorted(converted),
            ["k_o", "k_o_scale", "k_projection", "k_projection_scale"],
        )
        for name, source in sources.items():
            values, scale = converted[name], converted[f"{name}_scale"]
            self.assertEqual(values.dtype, ml_dtypes.float8_e4m3fn)
            self.assertEqual(scale.shape, (3, 1, source.shape[-1]))
            for layer in range(3):
                want_values, want_scale = rank_loader.quantize_columns_fp8(
                    source[layer]
                )
                np.testing.assert_array_equal(
                    values[layer].view(np.uint8), want_values.view(np.uint8)
                )
                np.testing.assert_array_equal(scale[layer], want_scale)

    def test_fp8_families_upload_as_torch_fp8(self):
        import ml_dtypes
        import torch

        from sglang_tpu.k3_programs import bridge

        values = np.linspace(-448, 448, 64, dtype=np.float32).astype(
            ml_dtypes.float8_e4m3fn
        )
        tensor = bridge.host_tensor(values.reshape(8, 8))
        self.assertEqual(tensor.dtype, torch.float8_e4m3fn)
        np.testing.assert_array_equal(
            tensor.view(torch.uint8).numpy().reshape(-1), values.view(np.uint8)
        )
        halves = np.ones(4, ml_dtypes.bfloat16)
        self.assertEqual(bridge.host_tensor(halves).dtype, torch.bfloat16)

    def test_colocated_prefill_keeps_its_bf16_input_projection(self):
        from types import SimpleNamespace
        from unittest import mock

        from sglang_tpu.k3_programs import sglang_colocated as colocated

        for fp8 in (False, True):
            config = SimpleNamespace(attention_dp=1, kda_fp8_projections=fp8)
            with mock.patch(
                "sglang_tpu.k3_programs.kimi_k3.config.active", return_value=config
            ):
                kda = colocated.shared_dense()["kda"]
            self.assertEqual("k_projection" in kda, not fp8)
            self.assertIn("k_gate", kda)


class TestKdaFp8OutputProjection(CustomTestCase):
    """The token-scattered attention output projection
    (``_project_attention_2d``) with FP8 weights and FP32 column scales
    (KernelConfig.kda_fp8_projections, rank_loader.quantize_columns_fp8) at B8
    and B16 in Pallas interpret mode on 32 host devices: every rank's FP32
    partial is the FP64 product of the dequantized weights (own and peer
    halves, columns scaled ~10x apart) to FP32 accuracy, and its owned rows
    are, bit for bit, the reduction's rounding points applied to those
    partials (as TestAttentionHostBf16 checks for BF16 weights)."""

    MAIN = r"""

def project_scaled(attended, weight, scale):
    B = attended.shape[1]
    dmk.ATTENTION_HOST_BF16 = False
    scratch = dmk.communication_scratch(B)
    backing = max(16, -(-B // 16) * 16)

    def kernel(attended_ref, weight_ref, scale_ref, owned_out, partial_out, out, comm,
               local, host, features, outputs, sends, recvs, phase):
        phase[0] = 0
        local[...] = jnp.full(local.shape, jnp.nan, jnp.float32)
        host[...] = jnp.full(host.shape, jnp.nan, jnp.float32)
        features[...] = jnp.full(features.shape, jnp.nan, jnp.bfloat16)
        outputs[...] = jnp.full(outputs.shape, jnp.nan, jnp.bfloat16)
        dmk._project_attention_2d(
            attended_ref[...], weight_ref, out, comm, local, host, features, outputs,
            sends, recvs, phase, dmk._dot, token_scatter_output=True,
            weight_scale=scale_ref)
        owned_out[...] = outputs[...]
        partial_out[...] = comm[...]

    vmem = pltpu.VMEM

    def body(attended, weight, scale):
        owned, partial = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((backing, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, 8192), jnp.float32)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((B, 8192), jnp.float32),
                            scratch[0], scratch[2], vmem((backing, 1024), jnp.bfloat16),
                            vmem((backing, H), jnp.bfloat16), pltpu.SemaphoreType.DMA((16,)),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SMEM((1,), jnp.int32)],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(attended[0], weight[0], scale[0])
        return owned[None], partial[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"),) * 3,
                              out_specs=(P("tp"), P("tp")), check_vma=False))
    return [np.asarray(x) for x in f(attended, weight, scale)]


F8 = ml_dtypes.float8_e4m3fn
rng = np.random.default_rng(0)
for B in (8, 16):
    tpp = B // 4
    attended = (rng.standard_normal((32, B, W)) * rng.uniform(0.2, 2.0, (32, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, B, W))))).astype(BF16)
    full = (rng.standard_normal((32 * W, H)) * 0.01 * rng.uniform(0.3, 3, H)).astype(BF16)
    # Rank r's shard: its pair's 768 features x its lane's output half.
    weight = np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                       for r in range(32)]).astype(np.float32)
    peak = np.abs(weight).max(axis=1, keepdims=True)
    scale = (peak / np.float32(448)).astype(np.float32)
    values = (weight / scale).astype(F8)
    owned, partial = project_scaled(attended, values, scale)
    a = attended.astype(np.float64)
    for r in range(32):
        w = values[r].astype(np.float64) * scale[r].astype(np.float64)
        own, peer = (r % 2) * W, (r % 2 ^ 1) * W
        want = a[r] @ w[own:own + W] + a[r ^ 1] @ w[peer:peer + W]
        err = np.abs(partial[r, :, :HALF] - want).max() / np.abs(want).max()
        assert err < 1e-5, (B, r, err)
    got = owned[:, :tpp]
    assert np.array_equal(got.view(np.uint16), reference(partial, B, False).view(np.uint16)), B
    for r in range(32):  # the four hosts' copies
        for offset in (8, 16, 24):
            assert np.array_equal(got[r].view(np.uint16), got[r ^ offset].view(np.uint16))
    print(f"B{B}: FP64 partials, bit-exact rows")
print("OK")
"""

    def test_fp8_partials_and_rows_match_the_dequantized_weights(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = TestAttentionHostBf16.SCRIPT.split("rng = np.random.default_rng(0)")[0]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", harness + self.MAIN],
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )



class TestKdaFp8Layer(CustomTestCase):
    """A KDA decode layer (``kda``: gate, input projection, short conv,
    delta-rule recurrence, gated norm, token-scattered output projection) at
    B8, as one 8-row verify sequence and as 8 plain rows, in Pallas interpret
    mode on 32 host devices, with FP8 input and output projections and
    power-of-two column scales (KernelConfig.kda_fp8_projections): every
    rank's owned output rows and post-states equal, bit for bit, the layer
    with FP32 copies of the dequantized weights (the interpret mode's
    mixed-dtype products run in FP32 for both, and each scaled product is
    exact)."""

    MAIN = r"""
from sglang_tpu.k3_programs.kimi_k3.layout import CONV_HEAD_ROWS

F8 = ml_dtypes.float8_e4m3fn
B = 8


def layer(hidden, gate, inputs, fb, output, aux, history, state, scales, sequence_rows):
    comm = dmk.communication_scratch(B)
    in_dtype, out_dtype = inputs.dtype, output.dtype

    def kernel(hidden_ref, gate_ref, input_ref, fb_ref, output_ref, conv_ref, decay_ref,
               bias_ref, norm_ref, history_ref, state_ref, input_scale_ref, output_scale_ref,
               history_out, state_out, owned_out, out, gate_vmem, input_vmem, fb_vmem,
               output_vmem, copy_sems, *workspace):
        ws = dmk.AttentionCommunicationWorkspace(*workspace)
        ws.reduction_phase[0] = 0
        for buffer in workspace[:4]:
            buffer[...] = jnp.full(buffer.shape, jnp.nan, jnp.float32)
        ws.attention_features_vmem[...] = jnp.full(ws.attention_features_vmem.shape, jnp.nan, jnp.bfloat16)
        ws.attention_output_vmem[...] = jnp.full(ws.attention_output_vmem.shape, jnp.nan, jnp.bfloat16)
        dmk.kda(
            hidden_ref, out, history_out, state_out,
            weights=dmk.KimiDeltaAttentionWeights(
                gate_ref, gate_vmem, input_ref, input_vmem, fb_ref, fb_vmem, output_ref,
                output_vmem, conv_ref, decay_ref, bias_ref, norm_ref, copy_sems,
                input_projection_scale=input_scale_ref if scales else None,
                output_projection_scale=output_scale_ref if scales else None),
            convolution_history_ref=history_ref, recurrent_state_ref=state_ref,
            communication_workspace=ws, synchronize_devices=False,
            weights_are_prefetched=False, token_scatter_output=True,
            sequence_rows=sequence_rows)
        owned_out[...] = ws.attention_output_vmem[...]

    vmem, anywhere = pltpu.VMEM, pl.ANY
    spaces = [vmem, anywhere, anywhere, anywhere, anywhere] + [vmem] * 8

    def body(*args):
        outs = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct(history.shape[1:], jnp.bfloat16),
                       jax.ShapeDtypeStruct(state.shape[1:], jnp.float32),
                       jax.ShapeDtypeStruct((16, H), jnp.bfloat16)),
            in_specs=[pl.BlockSpec(memory_space=s) for s in spaces],
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((H, 128), jnp.bfloat16),
                            vmem((H, 1664), in_dtype), vmem((128, 384), jnp.bfloat16),
                            vmem((768, HALF), out_dtype), pltpu.SemaphoreType.DMA((4,)), *comm],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(*(a[0] for a in args))
        return tuple(o[None] for o in outs)

    operands = (hidden, gate, inputs, fb, output, *aux, history, state, *scales_or_ones(scales))
    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"),) * len(operands),
                              out_specs=(P("tp"),) * 3, check_vma=False))
    return [np.asarray(x) for x in f(*operands)]


def scales_or_ones(scales):
    if scales:
        return scales
    return (np.ones((32, 1, 1664), np.float32), np.ones((32, 1, HALF), np.float32))


def pow2_fp8(weight):
    # FP8 values and power-of-two column scales; values x scales is exact in FP32.
    exponent = np.floor(np.log2(np.abs(weight).max(axis=-2, keepdims=True) / 448)) + 1
    scale = np.exp2(exponent).astype(np.float32)
    values = (weight / scale).astype(F8)
    exact = values.astype(np.float32) * scale
    assert np.array_equal(exact.astype(np.float64), values.astype(np.float64) * scale)
    return values, scale, exact


rng = np.random.default_rng(0)
rep = lambda x: np.broadcast_to(x, (32, *x.shape)).copy()
hidden = rep((rng.standard_normal((B, H)) * rng.uniform(0.3, 3, H)).astype(BF16))
gate = (rng.standard_normal((32, H, 128)) * 0.02).astype(BF16)
inputs = (rng.standard_normal((32, H, 1664)) * 0.02 * rng.uniform(0.2, 5, (32, 1, 1664))).astype(np.float32)
fb = (rng.standard_normal((32, 128, 384)) * 0.1).astype(BF16)
output = (rng.standard_normal((32, 768, HALF)) * 0.02 * rng.uniform(0.2, 5, (32, 1, HALF))).astype(np.float32)
aux = ((rng.standard_normal((32, 4, 3, 3, 128)) * 0.5).astype(BF16),
       rng.uniform(-1, 1, (32, 3, 1)).astype(np.float32),
       rng.uniform(-2, 2, (32, 3, 128)).astype(np.float32),
       (1 + 0.1 * rng.standard_normal((32, 8, 128))).astype(BF16))
history = (rng.standard_normal((32, B, 3, 3, CONV_HEAD_ROWS, 128))).astype(BF16)
state = (rng.standard_normal((32, B, 3, 128, 128)) * 0.1).astype(np.float32)
in8, in_scale, in32 = pow2_fp8(inputs)
out8, out_scale, out32 = pow2_fp8(output)
for sequence_rows in (8, 1):
    got = layer(hidden, gate, in8, fb, out8, aux, history, state, (in_scale, out_scale), sequence_rows)
    want = layer(hidden, gate, in32, fb, out32, aux, history, state, None, sequence_rows)
    for g, w, name in zip(got, want, ("history", "state", "rows")):
        if name == "history":  # the three real heads of the four head rows
            g, w = g[..., :3, :], w[..., :3, :]
        if name == "rows":
            g, w = g[:, :B // 4], w[:, :B // 4]
        assert np.isfinite(g.astype(np.float32)).all(), name
        bits = np.uint16 if g.dtype.itemsize == 2 else np.uint32
        assert np.array_equal(g.view(bits), w.view(bits)), (sequence_rows, name)
    # Without the scales the same FP8 values give a different layer.
    unscaled = layer(hidden, gate, in8, fb, out8, aux, history, state, None, sequence_rows)
    assert not np.array_equal(unscaled[2][:, :B // 4].view(np.uint16), want[2][:, :B // 4].view(np.uint16))
    print(f"sequence_rows={sequence_rows}: bit-exact")
print("OK")
"""

    def test_fp8_layer_matches_the_dequantized_weights(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = TestAttentionHostBf16.SCRIPT.split("rng = np.random.default_rng(0)")[0]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", harness + self.MAIN],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )
