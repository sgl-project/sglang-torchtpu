"""TpuHiCacheL3Backend moves one host-pool page as one opaque blob.

The backend addresses pages by their first token slot
(``page_id = slot // page_size``), hands the engine a writable view of
``kv_buffer[page_id]`` with no copy, and round-trips bytes through the
posix engine on disk. These cases pin:

* v1 zero-copy set/get round-trips a page bit-exactly;
* TP ranks never share keys or directories (cross-rank shard isolation);
* a size-mismatched stored page is a miss, never a partial read;
* files written by another process sharing the namespace are found via
  the stat fallback (MLA single-writer layout);
* ``clear()`` deletes only the calling rank's namespace;
* non-page_first_fused pools are rejected;
* the documented CLI flag parses.
"""

import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.storage.tpu_l3 import TpuHiCacheL3Backend

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
LAYER_NUM = 3
PAGE_NUM = 8
PAGE_SHAPE = (LAYER_NUM, PAGE_SIZE, 16)


def _make_pool(dtype=torch.uint8, layout="page_first_fused"):
    pool = SimpleNamespace()
    pool.page_size = PAGE_SIZE
    pool.layout = layout
    pool.kv_buffer = torch.zeros((PAGE_NUM,) + PAGE_SHAPE, dtype=dtype)

    def get_dummy_flat_data_page():
        return torch.zeros((LAYER_NUM, PAGE_SIZE * 16), dtype=dtype)

    pool.get_dummy_flat_data_page = get_dummy_flat_data_page
    return pool


def _make_backend(root_dir, tp_rank=0, tp_size=1, is_mla_model=False):
    cfg = SimpleNamespace(
        extra_config={"engine": "posix", "engine_config": {"root_dir": root_dir}},
        model_name="Qwen/Qwen3-8B",
        is_mla_model=is_mla_model,
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

    def test_clear_deletes_only_own_namespace(self):
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            rank0 = _make_backend(d, tp_rank=0, tp_size=2)
            rank1 = _make_backend(d, tp_rank=1, tp_size=2)
            rank0.register_mem_pool_host(pool)
            rank1.register_mem_pool_host(pool)
            idx = _host_indices([0])

            pool.kv_buffer[0].fill_(5)
            rank0.batch_set_v1(["k"], idx)
            rank1.batch_set_v1(["k"], idx)
            self.assertTrue(rank0.clear())
            self.assertEqual(rank0.batch_exists(["k"]), 0)
            # The other rank's namespace is untouched.
            self.assertEqual(rank1.batch_exists(["k"]), 1)
            pool.kv_buffer.zero_()
            self.assertEqual(rank1.batch_get_v1(["k"], idx), [True])
            self.assertTrue(torch.all(pool.kv_buffer[0] == 5))
            rank0.close()
            rank1.close()

    def test_size_mismatched_page_is_a_miss(self):
        """A stale file from an incompatible layout must not load into KV."""
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            backend = _make_backend(d)
            backend.register_mem_pool_host(pool)
            engine = backend._engine
            suffix = backend.key_suffix

            # Write a file under this rank's key that is larger than a page.
            key = "stale" + suffix
            rel = engine._relpath(key)
            full = os.path.join(engine.root_dir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(b"\xff" * (pool.kv_buffer[0].nbytes + 128))

            idx = _host_indices([4])
            sentinel = torch.full(PAGE_SHAPE, 9, dtype=torch.uint8)
            pool.kv_buffer[4].copy_(sentinel)
            self.assertEqual(backend.batch_get_v1(["stale"], idx), [False])
            self.assertTrue(torch.equal(pool.kv_buffer[4], sentinel))
            backend.close()

    def test_stat_fallback_sees_other_process_files(self):
        """MLA layout: one writer, many readers sharing a namespace."""
        pool = _make_pool()
        with tempfile.TemporaryDirectory() as d:
            # MLA: identical suffixes -> shared namespace directory.
            writer = _make_backend(d, tp_rank=0, tp_size=2, is_mla_model=True)
            reader = _make_backend(d, tp_rank=1, tp_size=2, is_mla_model=True)
            self.assertEqual(writer.key_suffix, reader.key_suffix)
            writer.register_mem_pool_host(pool)
            reader.register_mem_pool_host(pool)

            payload = torch.full(PAGE_SHAPE, 42, dtype=torch.uint8)
            idx = _host_indices([6])
            pool.kv_buffer[6].copy_(payload)
            self.assertEqual(writer.batch_set_v1(["k"], idx), [True])

            # The reader never wrote: its in-memory index is empty, but the
            # stat fallback must find the writer's file.
            self.assertEqual(reader.batch_exists(["k"]), 1)
            pool.kv_buffer.zero_()
            self.assertEqual(reader.batch_get_v1(["k"], idx), [True])
            self.assertTrue(torch.equal(pool.kv_buffer[6], payload))
            writer.close()
            reader.close()

    def test_rejects_non_fused_pool_layout(self):
        pool = _make_pool(layout="page_first")
        with tempfile.TemporaryDirectory() as d:
            backend = _make_backend(d)
            with self.assertRaises(ValueError):
                backend.register_mem_pool_host(pool)

    def test_documented_cli_flag_parses(self):
        from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
        from sglang.srt.arg_groups.fields.memory import Memory

        parser = argparse.ArgumentParser()
        add_cli_args_from_dataclass(parser, Memory)
        args = parser.parse_args(["--hicache-storage-backend", "tpu_l3"])
        self.assertEqual(args.hicache_storage_backend, "tpu_l3")

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
