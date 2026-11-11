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

register_cpu_ci(est_time=150, suite="base-a-test-cpu")

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
            capture_output=True, text=True, env=env, timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


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

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where((tokens(x) < cut) & (lane < 576), x, jnp.zeros((), x.dtype))

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

        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._cache_verify_row = _cache_verify_row

        def case(firsts, S, topk, seed, grouped=False, activity="exact"):
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
            q = (rng.standard_normal((G, B, H, 640)) * 0.05).astype(np.float32)
            q[..., 576:] = 0
            new = np.broadcast_to(lat[np.arange(B) // S, positions], (G, B, 640)).astype(np.float32)
            n_p, n_s = len(persistent), len(scoped)

            def kernel(pos, table, tau_r, ptie_r, own_r, act_r, q_r, new_r, pool_r, keys_r, out_r, *scratch):
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                sel = mla_dcp.DcpSelection(keys_r, tau_r, ptie_r, own_r, *scratch[n_p + n_s:],
                                           active=act_r, steps=steps)
                att = mla_dcp.DcpAttention(cache, ws, pos, B, selection=sel, sequence_rows=S,
                                           cache_verify_rows=S > 1)
                att.prime()
                out_r[...] = att.attend(q_r[...], new_r[...], 0)

            def body(pos, table, tau_, ptie_, own_, act_, q_, new_, pool_, keys_):
                vmem = pltpu.VMEM
                return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=6,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,
                        out_specs=pl.BlockSpec(memory_space=vmem),
                        scratch_shapes=[*persistent, *scoped, *selection]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], tau_, ptie_, own_, act_[0], q_[0], new_[0],
                                                       pool_[0], keys_[0])[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P()) + (P("tp"),) * 5,
                                      out_specs=P("tp"), check_vma=False))
            out = np.asarray(f(positions, tables.reshape(G, -1), tau, ptie, own, active, q, new, pool, keys))
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
            capture_output=True, text=True, env=env, timeout=900,
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
        rows = np.asarray(dsa_dcp.index_rows(jnp.asarray(keys).astype(jnp.bfloat16), pages))
        self.assertEqual(rows.shape, (pages, 4 * 33, 4, 128))
        # The same keys, token by token, into blocks 1..3 of one rank's pool
        # (a table whose entry e is global page e: every page on this rank).
        records = dsa_dcp.key_records(jnp.asarray(keys).astype(jnp.bfloat16))[:, None, :]
        positions = np.arange(tokens * pages, dtype=np.int32)
        tables = np.tile(np.arange(1, pages + 1, dtype=np.int32), (len(positions), 1))
        pool = (jnp.zeros((pages + 1, 4 * 33, 4, 128), jnp.uint8),)
        with mock.patch.object(mla_dcp, "owner_index", return_value=0):
            (pool,) = dsa_dcp.commit_index_rows(
                pool, records, jnp.asarray(tables), jnp.asarray(positions), group=1,
                global_pages=True)
        np.testing.assert_array_equal(np.asarray(pool)[1:], rows)


class TestDecodeIndexer(CustomTestCase):
    """The decode megakernel's indexer (``dsa_decode.DecodeIndexer``) in
    Pallas interpret mode on a 4-rank DCP group (8 index heads a rank): the
    rows' new key records, every rank's keys, the selection and the
    attention steps holding selected tokens match a numpy reference of the
    whole indexer (projections with sglang's native rounding, FP8 queries and
    head weights, scores, top-k), plain and verify rows."""

    SCRIPT = 'import os, sys, time\nG = 4\nos.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"\nos.environ["JAX_PLATFORMS"] = "cpu"\nimport ml_dtypes, numpy as np, jax, jax.numpy as jnp\nfrom jax.experimental import pallas as pl\nfrom jax.experimental.pallas import tpu as pltpu\nfrom jax.sharding import Mesh, PartitionSpec as P\nfrom sglang_tpu.k3_programs.kimi_k3 import dsa\nfrom sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, dsa_decode, mla_dcp\nmla_dcp.GROUPS = (G, 8, 16, 32)\ndsa_dcp.ACTIVITY_MIN_STEPS = int(os.environ.get("ACTIVITY_MIN_STEPS", "0"))\nBF = ml_dtypes.bfloat16\nbf = lambda x: np.asarray(x, np.float32).astype(BF).astype(np.float32)\n\nS = int(os.environ["S"]); TOPK = int(os.environ["TOPK"])\nfirsts = [int(x) for x in os.environ["POS"].split(",")]\nH, QL, T, PER = 1024, 1536, 512, 32 // G\npositions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)\nB = len(positions)\nrng = np.random.default_rng(0)\npages = max(-(-((p + T) // T) // G) for p in positions) + 1\nkey_rows = dsa_dcp.key_rows_for(pages, T)\nhidden = bf(rng.standard_normal((B, H)))\nqlat = bf(rng.standard_normal((B, QL)))\nwqb = bf(rng.standard_normal((QL, 32 * 128)) * QL**-0.5)\nwk = bf(rng.standard_normal((H, 128)) * H**-0.5)\nwproj = np.zeros((H, 128), np.float32); wproj[:, :32] = bf(rng.standard_normal((H, 32)) * H**-0.5)\nknorm = bf(np.stack([1 + 0.1 * rng.standard_normal(128), 0.1 * rng.standard_normal(128)]))\n\n# Reference projections (native rounding).\ndef layer_norm(x, w, b, eps=1e-5):\n    mean = x.mean(-1, keepdims=True); var = (x * x).mean(-1, keepdims=True) - mean * mean\n    return (x - mean) / np.sqrt(var + eps) * w + b\nkey = bf(layer_norm(bf(hidden @ wk), knorm[0], knorm[1]))\nnew_k8, new_e = dsa.quantize_fp8(key)\ngate = bf(bf(bf(hidden @ wproj)[:, :32]) * float(np.float32(BF(32**-0.5))))\nq_full = bf(qlat @ wqb)  # [B, 4096]\nq8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)\nfor b in range(B):\n    q8[b], e = dsa.quantize_fp8(q_full[b].reshape(32, 128))\n    w[b] = gate[b] * np.exp2(e[:, 0].astype(np.float32)) * 128**-0.5\n\n# Pool: committed tokens\' records (random keys), garbage past them.\npool = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)\ntables = np.zeros((G, B, pages), np.int32)\nseq_k8, seq_e = [], []\nfor s_, p0 in enumerate(firsts):\n    raw = rng.standard_normal((p0, 128)).astype(np.float32)\n    k8, ke = dsa.quantize_fp8(raw)\n    seq_k8.append(k8); seq_e.append(ke)\n    rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))\n    for r in range(S):\n        tables[:, s_ * S + r] = 1 + s_ * pages + np.arange(pages)\n    for t in range(p0):\n        page = t // T; g = page % G\n        blk = tables[g, s_ * S, page // G]\n        (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)\n        pool[g, blk, row, sub] = rec[t, :128]; pool[g, blk, srow, 0, lane] = rec[t, 128]\n\nwqb_ranks = np.stack([wqb[:, g * PER * 128:(g + 1) * PER * 128] for g in range(G)]).astype(BF)\nwks_ranks = np.stack([dsa_decode.key_weight(wk, g, G) for g in range(G)])\nwpt = dsa_decode.gate_weight(wproj)\npersistent = dsa_decode.persistent_scratch(B, G)\nscoped = dsa_decode.scoped_scratch(B, hidden=H, index_block_shape=(132, 4, 128), key_rows=key_rows,\n                                   page_tokens=T, sequence_rows=S,\n                                   page_plan=mla_dcp.page_plan(B, 3, S, jnp.float8_e4m3fn, dsa=True))\nnames = list(scoped)\nSTEP_ROWS = scoped["attend_keys"].shape[2]\nSTEPS = key_rows // STEP_ROWS\n\ndef kernel(pos_ref, table_ref, hidden_ref, qlat_ref, wqb_ref, wks_ref, wpt_ref, knorm_ref, pool_ref, lat_ref,\n           tau_out, ptie_out, own_out, act_out, rec_out, keys_out, *scratch):\n    cache = mla_dcp.DcpCache((lat_ref,), table_ref, pages, 0, group=G, global_pages=False)\n    sc = dict(zip(names, scratch[len(persistent):]))\n    indexer = dsa_decode.DecodeIndexer(\n        weights={"d_wqb": wqb_ref, "d_wks": wks_ref, "d_wpt": wpt_ref, "d_knorm": knorm_ref},\n        pools=(pool_ref,), layer=0, keys=keys_out, records_out=rec_out, persistent=scratch[:len(persistent)],\n        scoped=sc, topk=TOPK, sequence_rows=S)\n    indexer.start()\n    sel = indexer.selection(cache, pos_ref, hidden_ref[...], qlat_ref[...], 0)\n    for b in range(B):\n        tau_out[b] = sel.tau[b]; ptie_out[b] = sel.p_tie[b]; own_out[b] = sel.own[b]\n    for i in range(B * STEPS):\n        act_out[i] = sel.active[i]\n\ndef body(pos, table, hid, ql, wqb_r, wks_r, wpt_r, kn_r, pool_r, lat_r):\n    vmem, smem = pltpu.VMEM, pltpu.SMEM\n    out = pl.pallas_call(kernel,\n        out_shape=(jax.ShapeDtypeStruct((B,), jnp.int32),) * 3 + (jax.ShapeDtypeStruct((B * STEPS,), jnp.int32),\n            \n            jax.ShapeDtypeStruct((B, 24, 2, 128), jnp.int32), jax.ShapeDtypeStruct((B, key_rows, 128), jnp.int32)),\n        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,\n            in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 6,\n            out_specs=[pl.BlockSpec(memory_space=smem)] * 4 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,\n            scratch_shapes=[*persistent, *scoped.values()]),\n        compiler_params=pltpu.CompilerParams(has_side_effects=True),\n        interpret=pltpu.InterpretParams())(pos, table[0], hid, ql, wqb_r[0], wks_r[0], wpt_r, kn_r, pool_r[0], lat_r)\n    return tuple(x[None] for x in out)\n\nmesh = Mesh(np.asarray(jax.devices()), ("tp",))\nf = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P("tp"), P("tp"), P(), P(), P("tp"), P()),\n                          out_specs=(P("tp"),) * 6, check_vma=False))\nt0 = time.time()\ntau, ptie, own, act, rec, keys = (np.asarray(x) for x in f(\n    positions, tables.reshape(G, -1), hidden.astype(BF), qlat.astype(BF), wqb_ranks, wks_ranks,\n    wpt, knorm.astype(BF), pool, jnp.zeros((1, 128, 4, 640), jnp.float8_e4m3fn)))\n# Records: the rows\' new keys (every rank the same).\nwant_rec = np.asarray(dsa_dcp.key_records(jnp.asarray(key)))\ngot_rec = np.concatenate([rec[0, :, 0, 0, :], rec[0, :, 0, 1, :1]], axis=1)\nrec_match = np.mean(got_rec == want_rec)\nprint(f"records match {rec_match:.4f}; identical on ranks {all(np.array_equal(rec[0], rec[g]) for g in range(G))}")\nok = rec_match > 0.99\nunkey = lambda k: (k ^ ((k >> 31) & 0x7FFFFFFF)).astype(np.int32).view(np.float32)\nfor b in range(B):\n    s_, p0, L = b // S, firsts[b // S], positions[b] + 1\n    t = np.arange(L)\n    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)\n    k8 = np.concatenate([seq_k8[s_], new_k8[s_ * S:s_ * S + (b - s_ * S) + 1]])\n    ke = np.concatenate([seq_e[s_], new_e[s_ * S:s_ * S + (b - s_ * S) + 1]])\n    ref = dsa.monotone_key(dsa.scores(q8[b], w[b], k8, ke).astype(np.float32))\n    err = np.abs(unkey(kk) - unkey(ref)).max() / np.abs(unkey(ref)).max()\n    got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])\n    want = dsa.topk_mask(kk, t, TOPK)\n    ref_sel = dsa.topk_mask(ref, t, TOPK)\n    agree = np.mean(got == ref_sel)\n    uniform = all(len(set(v[:, b])) == 1 for v in (tau, ptie, own))\n    print(f"row {b} L={L} key err {err:.1e} exact {np.array_equal(got, want)} vs-ref-selection {agree:.4f} "\n          f"own {own[0, b]}/{bool(own[0, b]) == bool(got[-1])} uniform {uniform}")\n    ok &= bool(err < 1e-2 and np.array_equal(got, want) and uniform and bool(own[0, b]) == bool(got[-1]) and agree > 0.99)\n# Step activity: per rank, row (its sequence\'s first row), step: any selected token of the sequence\'s rows.\nwant_act = np.zeros_like(act)\nfor b in range(B):\n    L, first = positions[b] + 1, b - b % S\n    t = np.arange(L)\n    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)\n    for i in np.flatnonzero(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])):\n        g, j = (i // T) % G, (i // T) // G\n        want_act[g, first * STEPS + (j * (T // 128) + (i % T) // 128) // STEP_ROWS] = 1\nif dsa_dcp.ACTIVITY_MIN_STEPS:\n    want_act = np.ones_like(act)\nact_ok = np.array_equal(act, want_act)\nprint(f"activity exact {act_ok} active steps {int(act.sum())} of {act.size}")\nok &= bool(act_ok)\nprint("OK" if ok else "FAIL")\n'

    def test_indexer_matches_the_reference(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        for case in ({"S": "1", "TOPK": "128", "POS": "700,2100"},
                     {"S": "4", "TOPK": "200", "POS": "1500,3000"},
                     {"S": "4", "TOPK": "200", "POS": "1500,3000", "ACTIVITY_MIN_STEPS": "16"}):
            with self.subTest(**case):
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **case)
                result = subprocess.run(
                    [sys.executable, "-c", self.SCRIPT], capture_output=True, text=True, env=env,
                    timeout=900,
                )
                self.assertIn("OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:])


class TestK31Draft(CustomTestCase):
    """Kimi-K3.1's DSpark draft (``Inferact/Kimi-K3.1-Pro-DSpark-preview``)
    in the MLA draft's config: three heads a rank, causal block rows, YaRN's
    attention factor (the rotary scale is then exactly one), its aux layers;
    the reference attention's block rows see only earlier block rows."""

    CONFIG = {
        "model_type": "k3_dspark", "hidden_size": 7168, "num_hidden_layers": 5,
        "num_attention_heads": 96, "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "intermediate_size": 14336, "vocab_size": 163840, "rms_norm_eps": 1e-5,
        "target_layer_ids": [75, 79, 83, 87, 91], "mask_token_id": 163839, "markov_rank": 512,
        "block_size": 7, "rope_interleave": True, "full_attention_causal": True,
        "dflash_config": {"causal": True},
        "rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 32.0,
                            "original_max_position_embeddings": 32768, "beta_fast": 32, "beta_slow": 1,
                            "mscale": 1.0, "mscale_all_dim": 0.0, "attn_factor": 0.7426255848312643},
    }

    def test_config_and_causal_reference(self):
        import dataclasses

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla as mla

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps(self.CONFIG))
            c = mla.MlaDraftConfig.from_checkpoint(Path(d))
        self.assertEqual((c.heads_per_rank, c.causal, c.markov_rank, c.mask_token), (3, True, 512, 163839))
        self.assertEqual(c.aux_layers, (76, 80, 84, 88, 92))
        self.assertAlmostEqual(mla.rope_mscale(c), 1.0, places=6)
        self.assertAlmostEqual(c.softmax_scale, 192**-0.5, places=9)
        small = dataclasses.replace(c, hidden=256, heads=4, q_lora=128, kv_lora=64, nope=32, rope=16,
                                    v_head=32, intermediate=128, vocab=512, layers=1)
        rng = np.random.default_rng(0)
        w = {k: jnp.asarray(rng.standard_normal(s).astype(np.float32) * 0.05)
             for k, s in mla.weight_shapes(small).items()}
        h = jnp.asarray(rng.standard_normal((4, small.hidden)).astype(np.float32))
        positions, context = jnp.arange(10, 14), jnp.asarray(rng.standard_normal((6, small.latent)).astype(np.float32))
        valid = jnp.ones((6,), bool)
        causal = mla.reference_attention(w, small, 0, h, positions, context, valid)
        full = mla.reference_attention(w, dataclasses.replace(small, causal=False), 0, h, positions, context, valid)
        alone = mla.reference_attention(w, small, 0, h[:1], positions[:1], context, valid)
        np.testing.assert_allclose(causal[-1], full[-1], atol=1e-5)  # the last row sees every block row
        np.testing.assert_allclose(causal[0], alone[0], atol=1e-5)  # the first sees only itself
        self.assertFalse(np.allclose(causal[0], full[0], atol=1e-5))
