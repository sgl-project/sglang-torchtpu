"""TpuHiCacheL3Backend moves one host-pool page as one opaque blob.

The backend addresses pages by their first token slot
(``page_id = slot // page_size``), hands the engine a writable view of
``kv_buffer[page_id]`` with no copy, and round-trips bytes through the
posix engine on disk. These cases pin:

* v1 zero-copy set/get round-trips a page bit-exactly;
* ``batch_exists`` counts the longest stored prefix of the key list;
* missing keys report False without touching the destination page;
* the legacy (generic) set/get path stays functional for controllers that
  do not select the v1 interface.
"""

import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.storage.tpu_l3 import TpuHiCacheL3Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
LAYER_NUM = 3
PAGE_NUM = 8
PAGE_SHAPE = (LAYER_NUM, PAGE_SIZE, 16)


def _make_pool(dtype=torch.uint8):
    pool = SimpleNamespace()
    pool.page_size = PAGE_SIZE
    pool.kv_buffer = torch.zeros((PAGE_NUM,) + PAGE_SHAPE, dtype=dtype)

    def get_dummy_flat_data_page():
        return torch.zeros((LAYER_NUM, PAGE_SIZE * 16), dtype=dtype)

    pool.get_dummy_flat_data_page = get_dummy_flat_data_page
    return pool


def _make_backend(root_dir, tp_rank=0, tp_size=1):
    cfg = SimpleNamespace(
        extra_config={"engine": "posix", "engine_config": {"root_dir": root_dir}},
        model_name="Qwen/Qwen3-8B",
        is_mla_model=False,
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    return TpuHiCacheL3Backend(cfg)


def _host_indices(page_ids):
    idx = torch.zeros(len(page_ids) * PAGE_SIZE, dtype=torch.int64)
    for i, pid in enumerate(page_ids):
        idx[i * PAGE_SIZE] = pid * PAGE_SIZE
    return idx


class TestTpuL3Backend(CustomTestCase):
    def test_v1_roundtrip_bit_exact(self):
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            backend = _make_backend(d)
            backend.register_mem_pool_host(pool)

            payload = torch.randint(0, 256, PAGE_SHAPE, dtype=torch.uint8)
            pool.kv_buffer[2].copy_(payload)
            pool.kv_buffer[5].copy_(payload * 3)

            ok = backend.batch_set_v1(["k0", "k1"], _host_indices([2, 5]))
            self.assertEqual(ok, [True, True])
            self.assertEqual(backend.batch_exists(["k0", "k1", "k2"]), 2)

            pool.kv_buffer.zero_()
            ok = backend.batch_get_v1(["k0", "k1"], _host_indices([2, 5]))
            self.assertEqual(ok, [True, True])
            self.assertTrue(torch.equal(pool.kv_buffer[2], payload))
            self.assertTrue(torch.equal(pool.kv_buffer[5], payload * 3))
            backend.close()

    def test_missing_key_leaves_destination_untouched(self):
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            backend = _make_backend(d)
            backend.register_mem_pool_host(pool)

            sentinel = torch.full(PAGE_SHAPE, 7, dtype=torch.uint8)
            pool.kv_buffer[1].copy_(sentinel)
            ok = backend.batch_get_v1(["absent"], _host_indices([1]))
            self.assertEqual(ok, [False])
            self.assertTrue(torch.equal(pool.kv_buffer[1], sentinel))
            self.assertEqual(backend.batch_exists(["absent"]), 0)
            backend.close()

    def test_tp_ranks_do_not_share_keys(self):
        """Each TP rank holds a different KV shard of the same logical page;
        ranks writing the same keys must not overwrite each other."""
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            rank0 = _make_backend(d, tp_rank=0, tp_size=2)
            rank1 = _make_backend(d, tp_rank=1, tp_size=2)
            rank0.register_mem_pool_host(pool)
            rank1.register_mem_pool_host(pool)

            shard0 = torch.full(PAGE_SHAPE, 11, dtype=torch.uint8)
            shard1 = torch.full(PAGE_SHAPE, 22, dtype=torch.uint8)
            idx = _host_indices([3])

            pool.kv_buffer[3].copy_(shard0)
            self.assertEqual(rank0.batch_set_v1(["k"], idx), [True])
            pool.kv_buffer[3].copy_(shard1)
            self.assertEqual(rank1.batch_set_v1(["k"], idx), [True])

            pool.kv_buffer.zero_()
            self.assertEqual(rank0.batch_get_v1(["k"], idx), [True])
            self.assertTrue(torch.equal(pool.kv_buffer[3], shard0))
            pool.kv_buffer.zero_()
            self.assertEqual(rank1.batch_get_v1(["k"], idx), [True])
            self.assertTrue(torch.equal(pool.kv_buffer[3], shard1))
            rank0.close()
            rank1.close()

    def test_legacy_generic_path_roundtrip(self):
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            backend = _make_backend(d)
            backend.register_mem_pool_host(pool)

            value = torch.randint(0, 256, (LAYER_NUM, PAGE_SIZE * 16), dtype=torch.uint8)
            self.assertTrue(backend.set("k", value=value))
            self.assertTrue(backend.exists("k"))
            out = backend.get("k")
            self.assertIsNotNone(out)
            self.assertTrue(torch.equal(out, value))
            backend.close()


if __name__ == "__main__":
    unittest.main()
