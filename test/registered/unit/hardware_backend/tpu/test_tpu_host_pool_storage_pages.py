"""The TPU fused host pool's L3 storage seams move one page as one opaque row.

``TpuFusedMHATokenToKVPoolHost`` keeps the device page verbatim, page-major:
``kv_buffer[page_num, layer_num, *device_page_shape]``. A storage backend
therefore reads and writes ``kv_buffer[page_id]`` as a flat blob through
``get_data_page`` / ``get_dummy_flat_data_page`` / ``set_from_flat_data_page``,
the three methods the HiCache controller's generic page path calls
(``_generic_page_set`` / ``_generic_page_get``). These cases pin:

* the row is addressed by the page's first token slot, like every other host
  entry point on this pool;
* the flat page is a view of the host buffer (no copy on the backup path);
* a dummy page has the same element count and dtype as a real one, which is
  what ``HiCacheFile.get`` sizes its read against;
* an fp8 page round-trips bit-exactly through a real ``HiCacheFile`` on disk,
  under both device page layouts.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_pool_host import TpuFusedMHATokenToKVPoolHost

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
LAYER_NUM = 3
PAGE_NUM = 8
# Same element count either way; see test_tpu_host_pool_kv_layout.py.
TOKEN_MAJOR_PAGE = (PAGE_SIZE, 4, 4, 128)  # page, heads_packed, packing, head_dim
SEQ_ALONG_LANE_PAGE = (16, 32, 4, PAGE_SIZE)  # 2H, head_dim_lanes/p, p, page
LAYOUTS = (("token_major", TOKEN_MAJOR_PAGE), ("seq_along_lane", SEQ_ALONG_LANE_PAGE))


class _FakeDevicePool:
    def __init__(self, page_shape, dtype):
        self._fused = torch.empty((2,) + page_shape, dtype=dtype)
        self.layer_num = LAYER_NUM
        self.start_layer = 0
        # init_kv_buffer picks the host allocator by the DEVICE pool's device
        # ("tpu" -> alloc_tpu_host, plain unpinned CPU memory).
        self.device = "tpu"

    def get_fused_kv_buffer(self, layer_id):
        return self._fused


def _host_pool(page_shape, dtype=torch.float32):
    pool = TpuFusedMHATokenToKVPoolHost.__new__(TpuFusedMHATokenToKVPoolHost)
    pool.device_pool = _FakeDevicePool(page_shape, dtype)
    pool.page_size = PAGE_SIZE
    pool.dtype = dtype
    pool.page_num = PAGE_NUM
    pool.pin_memory = False
    pool.device = "cpu"
    pool.allocator = None
    pool.size_per_token = pool.get_size_per_token()
    pool.kv_buffer = pool.init_kv_buffer()
    pool.kv_buffer.zero_()
    return pool


def _fill_page(pool, page_id, seed):
    """Give page ``page_id`` a per-element pattern no other page shares."""
    numel = pool.kv_buffer[page_id].numel()
    pattern = (torch.arange(numel, dtype=torch.int64) * 7 + seed) % 251
    pool.kv_buffer[page_id] = (
        pattern.to(torch.float32).to(pool.dtype).reshape(pool.kv_buffer.shape[1:])
    )


def _as_bytes(t):
    return t.contiguous().view(torch.uint8)


def _file_backend(tmpdir, model_name="unit/tpu-host-pool"):
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=model_name,
    )
    # The env var overrides file_path; make the test independent of the shell.
    with patch.dict(os.environ, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": ""}):
        return HiCacheFile(config, file_path=tmpdir)


class TestTpuHostPoolStoragePages(CustomTestCase):
    def test_get_data_page_is_a_view_of_the_host_row(self):
        for name, page_shape in LAYOUTS:
            with self.subTest(layout=name):
                pool = _host_pool(page_shape)
                _fill_page(pool, 3, seed=11)
                flat = pool.get_data_page(3 * PAGE_SIZE)
                self.assertEqual(flat.shape, (pool.kv_buffer[3].numel(),))
                self.assertTrue(torch.equal(flat, pool.kv_buffer[3].flatten()))
                # Backup serializes straight out of the pool: no copy.
                self.assertEqual(flat.data_ptr(), pool.kv_buffer[3].data_ptr())
                self.assertEqual(
                    pool.get_data_page(3 * PAGE_SIZE, flat=False).shape,
                    (LAYER_NUM,) + page_shape,
                )

    def test_dummy_page_matches_a_real_page(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = _host_pool(TOKEN_MAJOR_PAGE, dtype=dtype)
                dummy = pool.get_dummy_flat_data_page()
                real = pool.get_data_page(0)
                self.assertEqual(dummy.numel(), real.numel())
                self.assertEqual(dummy.dtype, real.dtype)
                self.assertEqual(dummy.dim(), 1)
                self.assertTrue(dummy.is_contiguous())
                self.assertEqual(dummy.device.type, "cpu")

    def test_set_from_flat_data_page_writes_only_that_row(self):
        for name, page_shape in LAYOUTS:
            with self.subTest(layout=name):
                pool = _host_pool(page_shape)
                _fill_page(pool, 3, seed=11)
                _fill_page(pool, 4, seed=23)
                before_4 = pool.kv_buffer[4].clone()
                pool.set_from_flat_data_page(
                    5 * PAGE_SIZE, pool.get_data_page(3 * PAGE_SIZE).clone()
                )
                self.assertTrue(torch.equal(pool.kv_buffer[5], pool.kv_buffer[3]))
                self.assertTrue(torch.equal(pool.kv_buffer[4], before_4))
                self.assertEqual(pool.kv_buffer[6].abs().sum().item(), 0)

    def test_index_is_the_pages_first_token_slot(self):
        # Allocator output is page-aligned, so the token slot maps to its page;
        # a sub-page offset is dropped the way _to_page_ids drops it.
        pool = _host_pool(TOKEN_MAJOR_PAGE)
        _fill_page(pool, 2, seed=5)
        row = pool.kv_buffer[2].flatten()
        for index in (2 * PAGE_SIZE, torch.tensor(2 * PAGE_SIZE), 2 * PAGE_SIZE + 7):
            with self.subTest(index=index):
                self.assertTrue(torch.equal(pool.get_data_page(index), row))

    def test_fp8_page_roundtrips_through_hicache_file(self):
        for name, page_shape in LAYOUTS:
            with self.subTest(layout=name), tempfile.TemporaryDirectory() as tmp:
                pool = _host_pool(page_shape, dtype=torch.float8_e4m3fn)
                storage = _file_backend(tmp)
                _fill_page(pool, 1, seed=3)
                original = _as_bytes(pool.kv_buffer[1]).clone()

                # Backup: the controller hands the backend get_data_page(flat).
                self.assertTrue(storage.set("page-1", pool.get_data_page(PAGE_SIZE)))
                files = [f for f in os.listdir(tmp) if f.endswith(".bin")]
                self.assertEqual(len(files), 1)
                page_bytes = LAYER_NUM * torch.Size(page_shape).numel()
                self.assertEqual(
                    os.path.getsize(os.path.join(tmp, files[0])), page_bytes
                )

                # Prefetch into a different, empty page: dummy -> readinto -> set.
                fetched = storage.get("page-1", pool.get_dummy_flat_data_page())
                self.assertIsNotNone(fetched)
                pool.set_from_flat_data_page(6 * PAGE_SIZE, fetched)
                self.assertTrue(torch.equal(_as_bytes(pool.kv_buffer[6]), original))
                self.assertIsNone(
                    storage.get("missing", pool.get_dummy_flat_data_page())
                )

    def test_controller_generic_page_helpers_against_file_backend(self):
        # Drive the controller's own helper bodies, unbound, with the exact
        # argument shapes the backup / prefetch threads use: a page-aligned
        # host_indices tensor indexed at i * page_size (a 0-d tensor).
        from sglang.srt.managers.cache_controller import HiCacheController

        with tempfile.TemporaryDirectory() as tmp:
            pool = _host_pool(TOKEN_MAJOR_PAGE, dtype=torch.bfloat16)
            storage = _file_backend(tmp)
            for page_id, seed in ((2, 1), (3, 2), (4, 3)):
                _fill_page(pool, page_id, seed)
            src_rows = [pool.kv_buffer[p].clone() for p in (2, 3, 4)]
            hashes = ["h2", "h3", "h4"]
            src_indices = torch.arange(2 * PAGE_SIZE, 5 * PAGE_SIZE)
            dst_indices = torch.arange(5 * PAGE_SIZE, 8 * PAGE_SIZE)

            ctl = SimpleNamespace(
                storage_host_pool=pool, page_size=PAGE_SIZE, storage_backend=storage
            )
            self.assertTrue(
                HiCacheController._generic_page_set(ctl, hashes, src_indices)
            )
            op = SimpleNamespace(request_id="req", is_terminated=lambda: False)
            self.assertEqual(
                HiCacheController._generic_page_get(ctl, op, hashes, dst_indices), 3
            )
            for row, page_id in zip(src_rows, (5, 6, 7)):
                self.assertTrue(torch.equal(pool.kv_buffer[page_id], row))

            # A missing page stops the prefix at that page, as upstream does.
            self.assertEqual(
                HiCacheController._generic_page_get(
                    ctl, op, ["h2", "nope", "h4"], dst_indices
                ),
                1,
            )

    def test_storage_format_tag_separates_layouts_and_dtypes(self):
        # Token-major and seq-along-lane pages have identical byte counts; the
        # format tag in storage_model_name keeps one layout from reading back a
        # page persisted by the other or by a different store dtype.
        tm_bf16 = _host_pool(TOKEN_MAJOR_PAGE, dtype=torch.bfloat16)
        sal_bf16 = _host_pool(SEQ_ALONG_LANE_PAGE, dtype=torch.bfloat16)
        tm_fp8 = _host_pool(TOKEN_MAJOR_PAGE, dtype=torch.float8_e4m3fn)

        tags = {
            tm_bf16.storage_format_tag,
            sal_bf16.storage_format_tag,
            tm_fp8.storage_format_tag,
        }
        self.assertEqual(len(tags), 3)
        self.assertEqual(
            storage_model_name("org/model", tm_bf16),
            f"org/model-{tm_bf16.storage_format_tag}",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tm_store = _file_backend(tmp, storage_model_name("org/model", tm_bf16))
            sal_store = _file_backend(tmp, storage_model_name("org/model", sal_bf16))
            fp8_store = _file_backend(tmp, storage_model_name("org/model", tm_fp8))
            untagged_store = _file_backend(tmp, "org/model")

            _fill_page(tm_bf16, 0, seed=17)
            self.assertTrue(tm_store.set("page-0", tm_bf16.get_data_page(0)))
            self.assertTrue(tm_store.exists("page-0"))
            self.assertFalse(sal_store.exists("page-0"))
            self.assertFalse(fp8_store.exists("page-0"))
            self.assertFalse(untagged_store.exists("page-0"))


if __name__ == "__main__":
    unittest.main()
