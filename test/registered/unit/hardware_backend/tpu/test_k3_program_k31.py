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

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

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

        def case(firsts, S, topk, seed, grouped=False):
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
            q = (rng.standard_normal((G, B, H, 640)) * 0.05).astype(np.float32)
            q[..., 576:] = 0
            new = np.broadcast_to(lat[np.arange(B) // S, positions], (G, B, 640)).astype(np.float32)
            n_p, n_s = len(persistent), len(scoped)

            def kernel(pos, table, tau_r, ptie_r, own_r, q_r, new_r, pool_r, keys_r, out_r, *scratch):
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                sel = mla_dcp.DcpSelection(keys_r, tau_r, ptie_r, own_r, *scratch[n_p + n_s:])
                att = mla_dcp.DcpAttention(cache, ws, pos, B, selection=sel, sequence_rows=S,
                                           cache_verify_rows=S > 1)
                att.prime()
                out_r[...] = att.attend(q_r[...], new_r[...], 0)

            def body(pos, table, tau_, ptie_, own_, q_, new_, pool_, keys_):
                vmem = pltpu.VMEM
                return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=5,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,
                        out_specs=pl.BlockSpec(memory_space=vmem),
                        scratch_shapes=[*persistent, *scoped, *selection]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], tau_, ptie_, own_, q_[0], new_[0],
                                                       pool_[0], keys_[0])[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P()) + (P("tp"),) * 4,
                                      out_specs=P("tp"), check_vma=False))
            out = np.asarray(f(positions, tables.reshape(G, -1), tau, ptie, own, q, new, pool, keys))
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
