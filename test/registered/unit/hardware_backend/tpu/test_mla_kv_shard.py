# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA latent cache split inside every page (host): the shift-network
compaction is a stable partition, merged partials equal one unsharded attention
(the JAX merge and the serving merge, which hands each rank its own heads),
every offset of a page has exactly one owner and row, every pool buffer is
addressed by page id (host-tier page copies stay rank-local) with host bytes
per token equal to the device's, and the validator admits the host tier but
refuses a shared SPMD program."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
            enable_prefill_cp=False,
            use_spmd_compile=False,
            hicache_storage_backend=None,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def _validate(self, cfg, kpool=4, shards=8, megakernel=False):
        from sglang_tpu.tpu_platform import _validate_mla_kv_shards

        env = {
            "SGLANG_TPU_DSA_MLA_KV_SHARDS": str(shards),
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "1" if megakernel else "0",
            "SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT": "0",
        }
        with patch.dict(os.environ, env):
            _validate_mla_kv_shards(cfg, SimpleNamespace(index_kpool=kpool))

    def test_admits_the_host_tier(self):
        self._validate(self._cfg())

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
        with self.assertRaisesRegex(ValueError, "--page-size a multiple of 64"):
            self._validate(self._cfg(page_size=96), kpool=8)
        # kpool 6: shares need lcm(4, 6) = 12 tokens, not max(4, 6) = 6.
        with self.assertRaisesRegex(ValueError, "--page-size a multiple of 96"):
            self._validate(self._cfg(page_size=48), kpool=6)


if __name__ == "__main__":
    unittest.main()
