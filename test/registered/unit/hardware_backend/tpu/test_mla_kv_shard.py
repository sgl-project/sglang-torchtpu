# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA latent cache split inside every page (host): the shift-network
compaction is a stable partition, merged partials equal one unsharded attention
(the JAX merge and the serving merge, which hands each rank its own heads),
every offset of a page has exactly one owner and row, every pool buffer is
addressed by page id (host-tier page copies stay rank-local) with host bytes
per token equal to the device's, and the validator admits the host tier and
the token split but refuses a shared SPMD program. The token split's owned-row
attention (each rank gathers its own entries in fixed-width slices) equals one
attention over the rank's entries, with the kernels replaced by references."""

import importlib.util
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

if importlib.util.find_spec("torch_tpu") is None:
    # The kernel module imports torch_tpu's Pallas bridge; the references below
    # replace every kernel it would reach.
    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return mock.MagicMock(name=f"{self.__name__}.{name}")

    for _name in ("torch_tpu", "torch_tpu._internal", "torch_tpu._internal.pallas"):
        sys.modules[_name] = _Stub(_name)
        sys.modules[_name].__path__ = []

import jax.numpy as jnp
import numpy as np
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_sparse import kv_shard
from sglang_tpu.attention_pallas_mla_kv_shard import merge_kv_shards

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


def _attend(q, keys, rows):
    """Normalized output and natural-log sum over each row's -1-padded keys."""
    tokens, heads, dim = q.shape
    out = np.zeros((tokens, heads, dim), np.float32)
    lse = np.full((tokens, heads), -np.inf, np.float32)
    for t, row in enumerate(rows):
        live = [p for p in row if p >= 0]
        if not live:
            continue
        k = keys[t][live]
        s = q[t] @ k.T
        m = s.max(-1, keepdims=True)
        e = np.exp(s - m)
        out[t] = (e @ k) / e.sum(-1, keepdims=True)
        lse[t] = (m[:, 0] + np.log(e.sum(-1))).astype(np.float32)
    return out, lse


class _FakeGroup:
    """One rank of a merge group: all_gather stacks every rank's lse. The first
    pass over the ranks records each one's reduce-scatter input, the second
    returns this rank's chunk summed over them."""

    def __init__(self, rank: int, lses: list, fed: list):
        self.world_size, self.rank_in_group = len(lses), rank
        self.lses, self.fed = lses, fed

    def all_gather(self, x, dim):
        return torch.cat([s.unsqueeze(0) for s in self.lses], dim=dim)

    def reduce_scatter_tensor(self, out, inp):
        if len(self.fed) < self.world_size:
            self.fed.append(inp)
        else:
            out.copy_(sum(f[self.rank_in_group] for f in self.fed).unsqueeze(0))


def _merge_inputs(shards, *, seed, tokens=37, heads=2, dim=128):
    """Each rank's bf16 partial over all group heads and its lse (spread over
    ~30 nats, ranks with no candidate, padding tokens); the f64 merge of those
    partials (own heads in rank order) and its per-element sum of |terms|."""
    rng = np.random.default_rng(seed)
    group_heads = shards * heads
    out = rng.normal(size=(shards, tokens, group_heads, dim)).astype(np.float32)
    out = torch.from_numpy(out).bfloat16().float().numpy()  # the kernel's rounding
    lse = rng.normal(0.0, 4.0, (shards, tokens, group_heads)) + rng.normal(
        0.0, 8.0, (1, tokens, 1)
    )
    lse = lse.astype(np.float32)
    lse[rng.random((shards, tokens, group_heads)) < 0.2] = -np.inf  # empty owner
    lse[:, -3:] = -np.inf  # padding tokens: no rank has a candidate
    held = np.isfinite(lse)
    l64 = lse.astype(np.float64)
    m = np.where(held.any(0), np.where(held, l64, -np.inf).max(0), 0.0)
    w = np.where(held, np.exp(l64 - m), 0.0)
    w = w / np.maximum(w.sum(0), 1e-300)
    terms = np.where(held[..., None], out.astype(np.float64), 0.0) * w[..., None]
    want, scale = terms.sum(0), np.abs(terms).sum(0)
    out[~held] = np.nan  # the lse must mask it
    outs = [torch.from_numpy(out[r]).bfloat16() for r in range(shards)]
    lses = [torch.from_numpy(lse[r]) for r in range(shards)]
    return outs, lses, want, scale


def _bf16_half_step(x):
    """Half the bf16 spacing at |x| (the rounding bound of one cast)."""
    mag = np.maximum(np.abs(x), np.finfo(np.float32).tiny)
    return np.exp2(np.floor(np.log2(mag)) - 8)


class TestMlaKvShard(CustomTestCase):
    def test_compact_left_is_a_stable_partition(self):
        # Shifts applied bit by bit must never land two kept entries on one slot.
        rng = np.random.default_rng(1)
        for groups in (1, 2, 7, 64, 513, 700):
            for p in (0.0, 0.1, 0.5, 0.9, 1.0):
                keep = rng.random((3, groups)) < p
                vals = rng.integers(0, 1 << 20, (3, groups, 8)).astype(np.int32)
                vals = np.where(keep[..., None], vals, -1)
                got = np.asarray(
                    kv_shard.compact_left(jnp.asarray(vals), jnp.asarray(keep))
                )
                for t in range(3):
                    want = np.concatenate(
                        [vals[t][keep[t]], np.full((groups - keep[t].sum(), 8), -1)]
                    )
                    np.testing.assert_array_equal(got[t], want, f"{groups} {p}")

    def test_merged_partials_equal_unsharded_attention(self):
        rng = np.random.default_rng(0)
        tokens, heads, dim, width, n_keys = 9, 3, 8, 12, 40
        q = rng.normal(size=(tokens, heads, dim)).astype(np.float32)
        keys = rng.normal(size=(tokens, n_keys, dim)).astype(np.float32)
        rows = np.full((tokens, width), -1, np.int32)
        for t in range(tokens - 1):  # the last row is padding: no candidate
            n = rng.integers(0, width + 1)
            rows[t, :n] = rng.choice(n_keys, size=n, replace=False)
        owner = rng.integers(0, 2, n_keys)  # each key lives on one rank
        want, _ = _attend(q, keys, rows)
        outs, lses = [], []
        for role in range(2):
            mine = np.where((rows >= 0) & (owner[rows] == role), rows, -1)
            o, s = _attend(q, keys, mine)
            outs.append(o)
            lses.append(s)
        got = np.asarray(
            kv_shard.merge_partials(
                jnp.asarray(np.stack(outs)), jnp.asarray(np.stack(lses))
            )
        )
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_serving_merge_hands_each_rank_its_heads(self):
        # Every rank attends all group heads; rank r keeps heads [r * h, (r + 1) * h).
        rng = np.random.default_rng(2)
        shards, heads, tokens, dim, width, n_keys = 4, 2, 9, 8, 12, 40
        q = rng.normal(size=(tokens, shards * heads, dim)).astype(np.float32)
        keys = rng.normal(size=(tokens, n_keys, dim)).astype(np.float32)
        rows = np.full((tokens, width), -1, np.int32)
        for t in range(tokens - 1):  # the last row is padding: no candidate
            n = rng.integers(0, width + 1)
            rows[t, :n] = rng.choice(n_keys, size=n, replace=False)
        owner = rng.integers(0, shards, n_keys)
        want, _ = _attend(q, keys, rows)
        outs, lses = [], []
        for role in range(shards):
            mine = np.where((rows >= 0) & (owner[rows] == role), rows, -1)
            o, s = _attend(q, keys, mine)
            o[~np.isfinite(s)] = np.nan  # no candidate: the lse must mask it
            outs.append(torch.from_numpy(o))
            lses.append(torch.from_numpy(s))
        fed = []
        for _ in range(2):  # record every rank's partial, then reduce
            got = [
                merge_kv_shards(outs[r], lses[r], _FakeGroup(r, lses, fed))
                for r in range(shards)
            ]
        np.testing.assert_allclose(
            torch.cat(got, dim=1).numpy(), want, rtol=1e-5, atol=1e-6
        )

    def test_merge_rounding(self):
        # The FP32 merge rounds once: within half a bf16 step of the f64 merge
        # (plus FP32 slack). The bf16 sum rounds each weighted term and each
        # partial sum (2 * shards roundings, each <= 2^-9 of at most sum|terms|),
        # so it only keeps that looser bound; it must still mask empty ranks.
        for shards in (2, 4, 8):
            outs, lses, want, scale = _merge_inputs(shards, seed=shards)
            for bf16_sum in (False, True):
                with self.subTest(shards=shards, bf16_sum=bf16_sum):
                    fed = []
                    for _ in range(2):  # record every rank's input, then reduce
                        got = [
                            merge_kv_shards(
                                outs[r], lses[r], _FakeGroup(r, lses, fed),
                                bf16_sum=bf16_sum,
                            )
                            for r in range(shards)
                        ]  # fmt: skip
                    got = torch.cat(got, dim=1).double().numpy()
                    bound = _bf16_half_step(want) + 1e-6 * scale
                    if bf16_sum:
                        bound = bound + 2.0**-8 * shards * scale
                    self.assertTrue(np.all(np.abs(got - want) <= bound))
                    np.testing.assert_array_equal(got[-3:], 0.0)  # padding tokens

    def test_own_page_slots_route_every_offset_to_one_rank(self):
        shards, page_size, pages = 8, 128, 5
        sub = page_size // shards
        page = jnp.repeat(jnp.arange(1, pages + 1, dtype=jnp.int32), page_size)
        slot = jnp.tile(jnp.arange(page_size, dtype=jnp.int32), pages)
        hits = np.zeros(page.shape[0], np.int32)
        seen = set()
        for role in range(shards):
            got_page, got_slot = map(
                np.asarray,
                kv_shard.own_page_slots(
                    page, slot, sub_page=sub, role=role, oob_page=10**6
                ),
            )
            mine = got_page < 10**6
            hits += mine
            np.testing.assert_array_equal(
                np.asarray(slot)[mine], role * sub + got_slot[mine]
            )
            np.testing.assert_array_equal(got_page[mine], np.asarray(page)[mine])
            seen |= {
                (role, int(p), int(r)) for p, r in zip(got_page[mine], got_slot[mine])
            }
        np.testing.assert_array_equal(hits, 1)
        self.assertEqual(len(seen), page.shape[0])  # no two tokens share a row


def _pooled_rows(rng, positions, *, kpool, picks, width):
    """Pooled-indexer rows: ``picks`` random closed pools (all while fewer are
    closed), each as ``kpool`` tokens, then the open pool's tokens, then -1."""
    rows = np.full((len(positions), width), -1, np.int32)
    for t, pos in enumerate(positions):
        closed = (pos + 1) // kpool
        pick = rng.choice(closed, size=min(closed, picks), replace=False)
        toks = [p * kpool + j for p in pick for j in range(kpool)]
        toks += list(range(closed * kpool, pos + 1))
        rows[t, : len(toks)] = toks
    return rows


def _ref_gather_page_ids(page_indices, seq_page_ids, seq_ids_segment, max_num_seqs):
    table = page_indices.reshape(max_num_seqs, -1)
    logical = jnp.clip(seq_page_ids, 0, table.shape[1] - 1)
    return table[seq_ids_segment[:, None], logical]


def _ref_chunk(q, nope, rope, kv_lens, physical, lse, *, start_pos, end_pos,
               token_end, sm_scale, k_scale, **_):  # fmt: skip
    """sparse_attention_chunk on a float cache of whole rows: attends rows
    [start_pos, min(end_pos, token_end)) over their first kv_lens slots."""
    keys = nope.reshape(-1, nope.shape[-1])[physical[start_pos:end_pos]]
    qc = q[start_pos:end_pos].astype(jnp.float32)
    s = jnp.einsum("chd,cwd->chw", qc, keys) * sm_scale * k_scale
    live = jnp.arange(keys.shape[1])[None, :] < kv_lens[start_pos:end_pos, None]
    s = jnp.where(live[:, None, :], s, -jnp.inf)
    m = jnp.max(s, -1, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.where(live[:, None, :], jnp.exp(s - m), 0.0)
    total = jnp.sum(p, -1, keepdims=True)
    out = jnp.einsum("chw,cwd->chd", p, keys) / jnp.maximum(total, 1e-30)
    part_lse = jnp.where(
        total > 0, m + jnp.log(jnp.maximum(total, 1e-30)), jnp.finfo(jnp.float32).min
    )
    ok = (jnp.arange(start_pos, end_pos) < token_end)[:, None, None]
    q = q.at[start_pos:end_pos].set(jnp.where(ok, out, qc).astype(q.dtype))
    lse = lse.at[start_pos:end_pos].set(jnp.where(ok, part_lse, lse[start_pos:end_pos]))
    return q, lse


class TestOwnedRows(CustomTestCase):
    def test_owned_rows_keep_own_pools_in_order(self):
        rng = np.random.default_rng(3)
        positions = list(rng.integers(0, 5000, 40)) + [0, 1, 2, 3, 130]
        rows = _pooled_rows(rng, positions, kpool=4, picks=48, width=200)
        for shards in (2, 8):
            sub = 128 // shards
            for role in range(shards):
                got = np.asarray(
                    kv_shard.owned_rows(
                        jnp.asarray(rows), page_size=128, shards=shards, role=role,
                        group=4, out_width=256,
                    )
                )  # fmt: skip
                for t, row in enumerate(rows):
                    mine = [p for p in row if p >= 0 and (p % 128) // sub == role]
                    want = mine + [-1] * (256 - len(mine))
                    np.testing.assert_array_equal(got[t], want, f"{shards} {role}")

    def test_merge_two_equals_one_attention(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(6, 3, 8)).astype(np.float32)
        keys = rng.normal(size=(6, 20, 8)).astype(np.float32)
        rows = np.tile(np.arange(20), (6, 1))
        rows[5] = -1  # no candidate
        want, _ = _attend(q, keys, rows)
        split = rng.integers(0, 21, 6)
        parts = []
        for side in (0, 1):
            mine = np.where((np.arange(20) < split[:, None]) == (side == 0), rows, -1)
            o, s = _attend(q, keys, mine)
            s = np.where(np.isfinite(s), s, np.finfo(np.float32).min)
            parts += [jnp.asarray(o), jnp.asarray(s)[..., None]]
        got, _ = kv_shard.merge_two(*parts)
        np.testing.assert_allclose(np.asarray(got)[:5], want[:5], rtol=1e-5, atol=1e-6)

    def test_owned_row_slices_match_one_attention(self):
        from sglang_tpu.attention.pallas.mla_sparse import kernel

        rng = np.random.default_rng(5)
        shards, role, page, kpool, heads = 8, 5, 128, 4, 3
        sub = page // shards
        # Seq 0: a 3000-token prefix and 150 new tokens (partial pages at both
        # ends); seq 1: 70 new tokens from 0; 36 padding tokens.
        starts, lens = [3000, 0], [150, 70]
        num_tokens, chunk, pps = 256, 64, 32
        positions = [s + i for s, n in zip(starts, lens) for i in range(n)]
        seq_of = [r for r, n in enumerate(lens) for _ in range(n)]
        rows = _pooled_rows(rng, positions, kpool=kpool, picks=64, width=260)
        rows = np.pad(
            rows, ((0, num_tokens - len(positions)), (0, 0)), constant_values=-1
        )
        num_pages = 2 * pps + 1
        page_indices = rng.permutation(np.arange(1, num_pages))[: 2 * pps]
        table = page_indices.reshape(2, pps)
        # Logical keys; this rank's share holds its offsets of every page.
        keys = rng.normal(size=(2, pps * page, 576)).astype(np.float32)
        cache = np.zeros((num_pages, sub, 640), np.float32)
        for seq in range(2):
            for pos in range(pps * page):
                if (pos % page) // sub == role:
                    cache[table[seq, pos // page], pos % sub, :576] = keys[seq, pos]
        ql_nope = rng.normal(size=(num_tokens, heads, 512)).astype(np.float32)
        q_pe = rng.normal(size=(num_tokens, heads, 64)).astype(np.float32)
        q = np.concatenate([ql_nope, q_pe], -1)
        cu = np.array([0, 150, 220], np.int32)
        dist = np.array([0, 2, 2], np.int32)
        with (
            mock.patch.object(kernel, "sparse_attention_chunk", _ref_chunk),
            mock.patch.object(kernel, "gather_page_ids", _ref_gather_page_ids),
        ):
            for width in (32, 64, 512):
                mine = kv_shard.owned_rows(
                    jnp.asarray(rows), page_size=page, shards=shards, role=role,
                    group=kpool, out_width=-(-260 // width) * width,
                )  # fmt: skip
                out, lse = kernel.sparse_attention_owned_rows(
                    jnp.asarray(ql_nope), jnp.asarray(q_pe), jnp.asarray(cache),
                    jnp.zeros((num_pages, 1, 4, 128), jnp.uint8), mine,
                    jnp.asarray(page_indices, jnp.int32), jnp.asarray(cu),
                    jnp.asarray(dist), sm_scale=0.1, k_scale=1.0, shards=shards,
                    width=width, chunk=chunk,
                )  # fmt: skip
                out, lse = np.asarray(out), np.asarray(lse)
                for t in range(len(positions)):
                    own = [p for p in rows[t] if p >= 0 and (p % page) // sub == role]
                    if not own:
                        continue
                    k = keys[seq_of[t], own]
                    s = 0.1 * q[t] @ k.T
                    m = s.max(-1, keepdims=True)
                    e = np.exp(s - m)
                    np.testing.assert_allclose(
                        out[t], (e @ k) / e.sum(-1, keepdims=True), rtol=1e-4,
                        atol=1e-5, err_msg=f"width {width} token {t}",
                    )  # fmt: skip
                    np.testing.assert_allclose(
                        lse[t], m[:, 0] + np.log(e.sum(-1)), rtol=1e-5, atol=1e-5
                    )
                self.assertGreater(
                    int(np.max(np.sum(np.asarray(mine) >= 0, -1))), 32
                )  # the 32-slot run needs further slices


def _split_pool(
    *, shards: int, role: int, page_size: int = 128, rope: int = 0, kpool: int = 4
):
    from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

    with (
        patch("sglang_tpu.kv_pool.mla_kv_shards", return_value=shards),
        patch("sglang_tpu.kv_pool.mla_kv_shard_role", return_value=role),
    ):
        return TpuSplitMLATokenToKVPool(
            size=4 * page_size,
            page_size=page_size,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=512,
            qk_rope_head_dim=rope,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            index_key_layers=[True, False, True],
            index_kpool=kpool,
        )


class TestSplitPoolLayout(CustomTestCase):
    def test_every_page_buffer_is_addressed_by_page_id(self):
        """A host page then holds this rank's share of every buffer of the page."""
        from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool
        from sglang_tpu.kv_pool_host import TpuDMAHostKVPool
        from sglang_tpu.tpu_dma_hicache import page_major_tensors

        for rope in (0, 64):
            with self.subTest(rope=rope):
                pool = _split_pool(shards=8, role=3, rope=rope)
                self.assertEqual(
                    pool.nope_buffer[0].shape, (pool.num_pages, 16, 4, 128)
                )
                if rope:
                    self.assertEqual(
                        pool.rope_buffer[0].shape, (pool.num_pages, 4, 4, 128)
                    )
                tensors = page_major_tensors(pool)
                # NoPE (+ RoPE) on every layer; index keys + raw key|gate on two.
                self.assertEqual(len(tensors), 3 * (2 if rope else 1) + 2 * 2)
                self.assertEqual({t.shape[0] for t in tensors}, {pool.num_pages})
                moved = [buf for _, buf in pool.transfer_page_buffers()]
                self.assertEqual({t.shape[0] for t in moved}, {pool.num_pages})
                host = object.__new__(TpuDMAHostKVPool)
                host.device_pool, host.page_size, host.draft_pools = pool, 128, ()
                per_token = TpuSplitMLATokenToKVPool.bytes_per_token(
                    index_head_dim=128,
                    index_key_layers=[True, False, True],
                    index_kpool=4,
                    mla_shards=8,
                    qk_rope_head_dim=rope,
                )
                self.assertEqual(host.get_size_per_token(), per_token)
                # 1/8 of the latent, every index key and raw row.
                whole = _split_pool(shards=1, role=0, rope=rope)
                latent = 3 * (512 + (128 if rope else 0))
                host.device_pool = whole
                self.assertEqual(host.get_size_per_token() - per_token, latent * 7 // 8)

    def test_a_share_must_hold_whole_rope_rows_and_pools(self):
        with self.assertRaisesRegex(ValueError, "split the latent cache"):
            _split_pool(shards=8, role=0, page_size=16)  # 2-token shares
        _split_pool(shards=8, role=0, page_size=32)  # one RoPE row, one pool
        with self.assertRaisesRegex(ValueError, "split the latent cache"):
            # 6-token shares: one pool but 1.5 RoPE rows.
            _split_pool(shards=8, role=0, page_size=48, kpool=6)


class TestSplitValidator(CustomTestCase):
    @staticmethod
    def _cfg(**overrides):
        cfg = dict(
            tp_size=8,
            attn_cp_size=1,
            attn_dp_size=1,
            pp_size=1,
            page_size=128,
            enable_hierarchical_cache=True,
            disaggregation_mode="null",
            disaggregation_transfer_backend="raiden",
            enable_prefill_cp=False,
            use_spmd_compile=False,
            hicache_storage_backend=None,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def _validate(
        self,
        cfg,
        kpool=4,
        shards=8,
        megakernel=False,
        token_split=False,
        pipeline_transfer=False,
    ):
        from sglang_tpu.tpu_platform import _validate_mla_kv_shards

        env = {
            "SGLANG_TPU_DSA_MLA_KV_SHARDS": str(shards),
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1" if megakernel else "0",
            "SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT": "1" if token_split else "0",
            "SGLANG_RAIDEN_PIPELINE_TRANSFER": "1" if pipeline_transfer else "0",
        }
        with patch.dict(os.environ, env):
            _validate_mla_kv_shards(cfg, SimpleNamespace(index_kpool=kpool))

    def test_admits_the_host_tier(self):
        self._validate(self._cfg())

    def test_admits_a_decode_over_the_raiden_pipeline_transfer(self):
        self._validate(self._cfg(disaggregation_mode="decode"), pipeline_transfer=True)

    def test_token_split_admits_every_group_size(self):
        for shards in (2, 4, 8):
            self._validate(self._cfg(), shards=shards, token_split=True)

    def test_megakernel_needs_every_rank_and_16_token_shares(self):
        self._validate(self._cfg(), megakernel=True)  # host tier on
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_GLM_FLASH_MEGAKERNEL"):
            self._validate(self._cfg(), shards=4, megakernel=True)
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_GLM_FLASH_MEGAKERNEL"):
            self._validate(self._cfg(page_size=64), megakernel=True)  # 8-token shares

    def test_refuses_a_shared_spmd_program(self):
        with self.assertRaisesRegex(ValueError, "--use-spmd-compile"):
            self._validate(self._cfg(use_spmd_compile=True))

    def test_refuses_storage_that_writes_rank_0s_share_only(self):
        with self.assertRaisesRegex(ValueError, "--hicache-storage-backend"):
            self._validate(self._cfg(hicache_storage_backend="file"))

    def test_refuses_pd_and_a_page_that_does_not_split(self):
        with self.assertRaisesRegex(ValueError, "--disaggregation-mode"):
            self._validate(self._cfg(disaggregation_mode="prefill"))
        for mode in ("prefill", "decode"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "--disaggregation-mode"),
            ):
                self._validate(
                    self._cfg(disaggregation_mode=mode),
                    pipeline_transfer=mode == "prefill",
                )
        with self.assertRaisesRegex(ValueError, "--page-size a multiple of 64"):
            self._validate(self._cfg(page_size=96), kpool=8)
        # kpool 6: shares need lcm(4, 6) = 12 tokens, not max(4, 6) = 6.
        with self.assertRaisesRegex(ValueError, "--page-size a multiple of 96"):
            self._validate(self._cfg(page_size=48), kpool=6)


class _Mode:
    def __init__(self, extend=True):
        self.extend = extend

    def is_extend_without_speculative(self):
        return self.extend

    def is_extend_or_draft_extend_or_mixed(self):
        return self.extend


class TestOwnedRowsRouting(CustomTestCase):
    """Prefill batches the token split would split take the owned-rows op;
    decode and short batches keep the whole-row op."""

    def _backend(self, *, owned=True, zero_rope=True):
        from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._idx_split_group = SimpleNamespace(world_size=8)
        backend._idx_split_size = 8
        backend._idx_split_min_tokens = 1024
        backend._attn_split_world = 0
        backend._zero_rope = zero_rope
        backend._kpool = 4
        backend.mla_shards, backend.mla_role = 8, 3
        backend.forward_metadata = SimpleNamespace(decode_query_size=1)
        backend._kv_shard_op = mock.MagicMock(name="whole", return_value=(1, 2))
        backend._kv_shard_owned_op = (
            mock.MagicMock(name="owned", return_value=(1, 2)) if owned else None
        )
        backend._mla_group = SimpleNamespace(
            world_size=8,
            all_gather=lambda t, dim: torch.cat([t] * 8, dim=dim),
        )
        return backend

    def test_routing(self):
        backend = self._backend()
        prefill = SimpleNamespace(forward_mode=_Mode())
        decode = SimpleNamespace(forward_mode=_Mode(extend=False))
        self.assertTrue(backend._kv_shard_owned_rows(prefill, 4096))
        self.assertFalse(backend._kv_shard_owned_rows(prefill, 512))  # unsplit indexer
        self.assertFalse(backend._kv_shard_owned_rows(decode, 4096))
        self.assertFalse(self._backend(owned=False)._kv_shard_owned_rows(prefill, 4096))
        # The indexer returns whole rows: every rank needs its own entries of all.
        self.assertEqual(backend._attention_split_rows(prefill, 4096), 0)

    def test_owned_op_keeps_a_zero_rope_lane_home(self):
        args = [torch.ones(16, 2, 512), torch.zeros(16, 2, 64)] + [None] * 12
        args[11], args[12] = 0.1, None  # sm_scale, k_scale
        for owned in (True, False):
            backend = self._backend()
            with patch(
                "sglang_tpu.attention_pallas_mla_kv_shard.merge_kv_shards",
                return_value="merged",
            ):
                backend._forward_mla_kv_shard(tuple(args), "rows", owned=owned)
            op = backend._kv_shard_owned_op if owned else backend._kv_shard_op
            other = backend._kv_shard_op if owned else backend._kv_shard_owned_op
            op.assert_called_once()
            other.assert_not_called()
            q_nope, q_pe = op.call_args.args[:2]
            self.assertEqual(tuple(q_nope.shape), (16, 16, 512))
            self.assertEqual(tuple(q_pe.shape), (16, 16, 64))
            self.assertEqual(op.call_args.args[10], "rows")


if __name__ == "__main__":
    unittest.main()
