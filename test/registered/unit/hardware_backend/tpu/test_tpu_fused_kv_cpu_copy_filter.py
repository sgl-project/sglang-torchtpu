"""Unit tests for row-filtering host KV copies before loading them back.

``SWAKVPool.load_cpu_copy`` keeps only the rows of a host copy whose tokens
still map to a sliding-window slot at the resume indices. The filtering is
done by the sub-pool so it follows that pool's host payload format: ``[K, V]``
pairs per chunk for ``MHATokenToKVPool``, one fused KV tensor per chunk for
the TPU pool. Everything here runs on CPU with no TPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except Exception:
    _HAS_JAX = False

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_pool import (
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
)

HEAD_NUM = 1
HEAD_DIM = 128
CHUNK_SIZE = 3
# Row values are encoded as offset + layer * LAYER_STRIDE + row and must stay
# below 256 so they are exact in bfloat16.
LAYER_STRIDE = 40
SWA_OFFSET = 128


def _make_fused_pool(size, page_size, layer_num):
    pool = TpuFusedMHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.cpu_offloading_chunk_size = CHUNK_SIZE
    return pool


def _make_mha_pool(size, layer_num):
    pool = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=torch.float32,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.cpu_offloading_chunk_size = CHUNK_SIZE
    return pool


def _make_seq_along_lane_pool(middle_shape, page_size=4, layer_num=2):
    pool = object.__new__(TpuSeqAlongLaneKVPool)
    pool.page_size = page_size
    pool.layer_num = layer_num
    pool.cpu_offloading_chunk_size = CHUNK_SIZE
    pool.device_module = torch.get_device_module("cpu")
    pool._raiden_backup_obj = None
    pool.kv_buffer = [
        torch.empty((5, *middle_shape, page_size), dtype=torch.bfloat16)
        for _ in range(layer_num)
    ]
    return pool


def _fill_rows(buffers, offset=0.0):
    """Write ``offset + layer * LAYER_STRIDE + row`` into every element of a row."""
    for layer, buf in enumerate(buffers):
        rows = buf.shape[0]
        values = torch.arange(rows, dtype=torch.float32) + layer * LAYER_STRIDE + offset
        buf.copy_(values.view(rows, *([1] * (buf.dim() - 1))).to(buf.dtype))


def _fused_rows(pool):
    return [buf.flatten(0, 1) for buf in pool.kv_buffer]


def _row_values(buf, rows):
    rows = torch.as_tensor(rows, dtype=torch.long)
    return buf[rows].flatten(1)[:, 0].float().tolist()


class TestFusedPoolFilterCpuCopy(CustomTestCase):
    def test_fused_pool_filter_round_trip(self):
        pool = _make_fused_pool(size=32, page_size=4, layer_num=2)
        _fill_rows(_fused_rows(pool))

        indices = torch.arange(5, 15)
        row_mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=torch.bool)
        kept = indices[row_mask]
        self.assertGreater(len(indices), 3 * CHUNK_SIZE)

        kv_cpu = pool.get_cpu_copy(indices)
        filtered = pool.filter_cpu_copy(kv_cpu, row_mask)

        self.assertEqual(len(filtered), pool.layer_num)
        for layer_chunks in filtered:
            self.assertEqual([len(c) for c in layer_chunks], [3, 3])
            for chunk in layer_chunks:
                self.assertIsInstance(chunk, torch.Tensor)
                self.assertEqual(chunk.shape[1:], kv_cpu[0][0].shape[1:])

        new_indices = torch.arange(20, 20 + len(kept))
        pool.load_cpu_copy(filtered, new_indices)
        for layer, buf in enumerate(_fused_rows(pool)):
            self.assertEqual(
                _row_values(buf, new_indices),
                [layer * LAYER_STRIDE + r for r in kept.tolist()],
            )

    def test_fused_pool_filter_keeps_single_chunk_layout(self):
        pool = _make_fused_pool(size=8, page_size=4, layer_num=1)
        _fill_rows(_fused_rows(pool))

        indices = torch.tensor([1, 2])
        kv_cpu = pool.get_cpu_copy(indices)
        filtered = pool.filter_cpu_copy(kv_cpu, torch.tensor([False, True]))
        self.assertEqual(len(filtered[0]), 1)
        self.assertEqual(_row_values(filtered[0][0], [0]), [2.0])

    def test_fused_pool_load_partial_pages_keeps_neighbours(self):
        pool = _make_fused_pool(size=32, page_size=4, layer_num=2)
        _fill_rows(_fused_rows(pool))
        # Rows 6..13 span a partial page (6, 7), a full page (8..11) and
        # another partial page (12, 13).
        src = torch.arange(6, 14)
        kv_cpu = pool.get_cpu_copy(src)

        _fill_rows(_fused_rows(pool), offset=SWA_OFFSET)
        dst = torch.arange(18, 26)
        pool.load_cpu_copy(kv_cpu, dst)
        for layer, buf in enumerate(_fused_rows(pool)):
            self.assertEqual(
                _row_values(buf, dst),
                [layer * LAYER_STRIDE + r for r in src.tolist()],
            )
            untouched = [16, 17, 26, 27]
            self.assertEqual(
                _row_values(buf, untouched),
                [SWA_OFFSET + layer * LAYER_STRIDE + r for r in untouched],
            )

    def test_fused_pool_load_unordered_indices(self):
        pool = _make_fused_pool(size=16, page_size=4, layer_num=1)
        _fill_rows(_fused_rows(pool))
        src = torch.tensor([9, 2, 11, 3])
        kv_cpu = pool.get_cpu_copy(src)
        _fill_rows(_fused_rows(pool), offset=SWA_OFFSET)
        dst = torch.tensor([5, 13, 4, 12])
        pool.load_cpu_copy(kv_cpu, dst)
        buf = _fused_rows(pool)[0]
        self.assertEqual(_row_values(buf, dst), [9.0, 2.0, 11.0, 3.0])
        self.assertEqual(
            _row_values(buf, [6, 7, 14, 15]), [SWA_OFFSET + r for r in [6, 7, 14, 15]]
        )


class TestSeqAlongLanePoolCpuCopy(CustomTestCase):
    def test_round_trip_preserves_unselected_lanes(self):
        for middle_shape in ((2, 3), (2, 3, 2)):
            with self.subTest(middle_shape=middle_shape):
                pool = _make_seq_along_lane_pool(middle_shape)
                for layer, buffer in enumerate(pool.kv_buffer):
                    values = (
                        torch.arange(buffer.shape[0] * pool.page_size)
                        + layer * LAYER_STRIDE
                    )
                    buffer.copy_(
                        values.reshape(
                            buffer.shape[0],
                            *([1] * len(middle_shape)),
                            pool.page_size,
                        ).expand_as(buffer)
                    )

                source = [buffer.clone() for buffer in pool.kv_buffer]
                source_indices = torch.tensor([0, 1, 2, 3, 5, 11])
                payload = pool.get_cpu_copy(source_indices)
                self.assertEqual(
                    [[chunk.shape[0] for chunk in chunks] for chunks in payload],
                    [[CHUNK_SIZE, CHUNK_SIZE]] * pool.layer_num,
                )

                for buffer in pool.kv_buffer:
                    buffer.add_(SWA_OFFSET)
                expected = [buffer.clone() for buffer in pool.kv_buffer]
                destination_indices = torch.tensor([12, 13, 14, 15, 9, 6])
                for layer in range(pool.layer_num):
                    for source_index, destination_index in zip(
                        source_indices.tolist(), destination_indices.tolist()
                    ):
                        source_block, source_offset = divmod(
                            source_index, pool.page_size
                        )
                        destination_block, destination_offset = divmod(
                            destination_index, pool.page_size
                        )
                        expected[layer][destination_block, ..., destination_offset] = (
                            source[layer][source_block, ..., source_offset]
                        )

                pool.load_cpu_copy(payload, destination_indices)
                for actual, wanted in zip(pool.kv_buffer, expected):
                    self.assertTrue(torch.equal(actual, wanted))


class TestKvPageWriteKernel(CustomTestCase):
    @unittest.skipUnless(_HAS_JAX, "jax not installed")
    def test_kernel_writes_only_listed_blocks(self):
        import jax.numpy as jnp

        from sglang_tpu.kv_page_write import write_pages

        kv = jnp.arange(6 * 4 * 2 * 1 * 8, dtype=jnp.float32).reshape(6, 4, 2, 1, 8)
        kv = kv.astype(jnp.bfloat16)
        # Distinct values per source page, so a swapped or repeated page write
        # is caught.
        pages = jnp.stack(
            [
                -jnp.ones((4, 2, 1, 8), jnp.bfloat16),
                -2 * jnp.ones((4, 2, 1, 8), jnp.bfloat16),
            ]
        )
        block_ids = jnp.array([4, 1], jnp.int32)
        out = write_pages(block_ids, pages, kv, interpret=True)
        self.assertTrue(bool((out[4] == pages[0]).all()))
        self.assertTrue(bool((out[1] == pages[1]).all()))
        for b in (0, 2, 3, 5):
            self.assertTrue(bool((out[b] == kv[b]).all()))


class _FakeTransfer:
    """Copies pages between the pool tensor and an in-memory host store, the
    way raiden's D2H/H2D moves them, so the whole backup contract runs."""

    def __init__(self, device_pool, num_host_pages):
        self.pool = device_pool
        self.host = {}

    def _pages(self, layer_id):
        return self.pool.get_fused_kv_buffer(self.pool.start_layer + layer_id)

    def backup(self, device_page_ids, host_page_ids):
        for d, h in zip(device_page_ids, host_page_ids):
            self.host[h] = [
                self._pages(l)[d].clone() for l in range(self.pool.layer_num)
            ]
        return _Ready()

    def load(self, host_page_ids, device_page_ids):
        for h, d in zip(host_page_ids, device_page_ids):
            for l, page in enumerate(self.host[h]):
                self._pages(l)[d] = page
        return _Ready()


class _Ready:
    def synchronize(self):
        pass


class TestRaidenRetractionBackup(CustomTestCase):
    def _pool_with_backup(self, size=64 * 8, page=8, layers=2, pages=16):
        from sglang_tpu import raiden_kv_backup

        pool = _make_fused_pool(size=size, page_size=page, layer_num=layers)
        backup = raiden_kv_backup.RaidenKVBackup.__new__(
            raiden_kv_backup.RaidenKVBackup
        )
        backup._transfer = _FakeTransfer(pool, pages)
        backup._page_size = page
        backup._free = list(range(pages))
        pool._raiden_backup_obj = backup
        return pool, backup

    def test_sequential_page_runs(self):
        from sglang_tpu.raiden_kv_backup import (
            sequential_page_runs,
        )

        # A retracted request: page-aligned sequential rows, partial tail.
        self.assertEqual(
            sequential_page_runs(torch.arange(16, 16 + 11), 8), ([2, 3], 0, 11)
        )
        # Full single page.
        self.assertEqual(sequential_page_runs(torch.arange(8), 8), ([0], 0, 8))
        # A gap makes it irregular.
        self.assertIsNone(
            sequential_page_runs(torch.tensor([0, 1, 2, 4, 5, 6, 7, 8]), 8)
        )
        # Sequential across a page gap is fine (pages need not be adjacent),
        # but starting mid-page is reported so the caller can fall back.
        self.assertEqual(
            sequential_page_runs(torch.tensor([6, 7, 16, 17]), 8), ([0, 2], 6, 4)
        )
        # A page change before the boundary is irregular.
        self.assertIsNone(sequential_page_runs(torch.tensor([5, 6, 16, 17]), 8))
        self.assertIsNone(sequential_page_runs(torch.tensor([], dtype=torch.long), 8))

    def test_round_trip_into_new_pages(self):
        pool, backup = self._pool_with_backup()
        _fill_rows(_fused_rows(pool))
        saved_indices = torch.arange(16, 16 + 21)  # pages 2..4, tail of 5 rows
        rows_before = [
            _fused_rows(pool)[l].flatten(0, 1)[saved_indices].clone()
            for l in range(pool.layer_num)
        ]
        handle = pool.get_cpu_copy(saved_indices)
        from sglang_tpu.raiden_kv_backup import (
            RaidenKVBackupHandle,
        )

        self.assertIsInstance(handle, RaidenKVBackupHandle)
        self.assertEqual(len(handle.host_pages), 3)

        # The pool moves on while the request is retracted.
        _fill_rows(_fused_rows(pool), offset=SWA_OFFSET)

        # Resume lands at different, freshly allocated pages.
        new_indices = torch.arange(40, 40 + 21)  # pages 5..7
        pool.load_cpu_copy(handle, new_indices)
        for l in range(pool.layer_num):
            rows_now = _fused_rows(pool)[l].flatten(0, 1)[new_indices]
            self.assertTrue(torch.equal(rows_now, rows_before[l]))
        # The handle returned its host pages.
        self.assertEqual(sorted(backup._free), list(range(16)))

    def test_irregular_rows_fall_back_to_the_row_path(self):
        pool, backup = self._pool_with_backup()
        _fill_rows(_fused_rows(pool))
        irregular = torch.tensor([3, 4, 5, 11, 12, 19])
        payload = pool.get_cpu_copy(irregular)
        self.assertIsInstance(payload, list)  # the row path's format
        _fill_rows(_fused_rows(pool), offset=SWA_OFFSET)
        pool.load_cpu_copy(payload, irregular)
        buf = _fused_rows(pool)[0].flatten(0, 1)
        self.assertEqual(_row_values(_fused_rows(pool)[0], [3, 12])[0], 3.0)

    def test_exhausted_pool_falls_back(self):
        pool, backup = self._pool_with_backup(pages=1)
        _fill_rows(_fused_rows(pool))
        payload = pool.get_cpu_copy(torch.arange(0, 24))  # needs 3 pages
        self.assertIsInstance(payload, list)

    def test_restore_shape_mismatch_raises(self):
        pool, backup = self._pool_with_backup()
        _fill_rows(_fused_rows(pool))
        handle = pool.get_cpu_copy(torch.arange(16, 16 + 21))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            pool.load_cpu_copy(handle, torch.arange(40, 40 + 20))


class TestMhaPoolFilterCpuCopy(CustomTestCase):
    def test_mha_pool_filter_round_trip(self):
        pool = _make_mha_pool(size=32, layer_num=2)
        _fill_rows(pool.k_buffer)
        _fill_rows(pool.v_buffer, offset=0.5)

        indices = torch.arange(5, 15)
        row_mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=torch.bool)
        kept = indices[row_mask]

        kv_cpu = pool.get_cpu_copy(indices)
        filtered = pool.filter_cpu_copy(kv_cpu, row_mask)

        self.assertEqual(len(filtered), pool.layer_num)
        for layer_chunks in filtered:
            self.assertEqual([len(c[0]) for c in layer_chunks], [3, 3])
            for chunk in layer_chunks:
                self.assertEqual(len(chunk), 2)
                self.assertEqual(chunk[0].shape[0], chunk[1].shape[0])

        new_indices = torch.arange(20, 20 + len(kept))
        pool.load_cpu_copy(filtered, new_indices)
        for layer in range(pool.layer_num):
            self.assertEqual(
                _row_values(pool.k_buffer[layer], new_indices),
                [layer * LAYER_STRIDE + r for r in kept.tolist()],
            )
            self.assertEqual(
                _row_values(pool.v_buffer[layer], new_indices),
                [layer * LAYER_STRIDE + r + 0.5 for r in kept.tolist()],
            )


class TestSwaPoolResumeWithFusedSubPool(CustomTestCase):
    """Retract-then-resume of a request whose SWA rows partially remap."""

    def _make_swa_pool(self):
        pool = SWAKVPool(
            size=64,
            size_swa=32,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        pool.full_kv_pool.cpu_offloading_chunk_size = CHUNK_SIZE
        pool.swa_kv_pool.cpu_offloading_chunk_size = CHUNK_SIZE
        _fill_rows(_fused_rows(pool.full_kv_pool))
        _fill_rows(_fused_rows(pool.swa_kv_pool), offset=SWA_OFFSET)
        return pool

    def test_resume_drops_rows_without_swa_slot(self):
        pool = self._make_swa_pool()
        # Slot 0 of the SWA pool is the dummy slot; unmapped full indices map to it.
        mapping = torch.zeros(pool.size + 1, dtype=torch.int64)
        mapping[-1] = -1
        pool.register_mapping(mapping)

        # Before retraction: full indices 1..10, only the last eight hold an
        # SWA slot (1..8).
        old_indices = torch.arange(1, 11)
        mapping[old_indices[2:]] = torch.arange(1, 9)
        kv_cpu = pool.get_cpu_copy(old_indices)
        self.assertEqual(kv_cpu["swa_mask"].tolist(), [False] * 2 + [True] * 8)

        # On resume: new full indices 21..30, and only five of the eight
        # previously copied SWA rows get an SWA slot again (11..15).
        new_indices = torch.arange(21, 31)
        remapped_positions = [2, 3, 6, 7, 9]
        mapping[new_indices[remapped_positions]] = torch.arange(11, 16)
        pool.load_cpu_copy(kv_cpu, new_indices)

        for layer, buf in enumerate(_fused_rows(pool.full_kv_pool)):
            self.assertEqual(
                _row_values(buf, new_indices),
                [layer * LAYER_STRIDE + r for r in old_indices.tolist()],
            )
        old_swa_slots = [p - 2 + 1 for p in remapped_positions]
        for layer, buf in enumerate(_fused_rows(pool.swa_kv_pool)):
            self.assertEqual(
                _row_values(buf, torch.arange(11, 16)),
                [SWA_OFFSET + layer * LAYER_STRIDE + s for s in old_swa_slots],
            )

    def test_resume_with_all_rows_remapped(self):
        pool = self._make_swa_pool()
        mapping = torch.zeros(pool.size + 1, dtype=torch.int64)
        mapping[-1] = -1
        pool.register_mapping(mapping)

        old_indices = torch.arange(1, 6)
        mapping[old_indices] = torch.arange(1, 6)
        kv_cpu = pool.get_cpu_copy(old_indices)

        new_indices = torch.arange(21, 26)
        mapping[new_indices] = torch.arange(11, 16)
        pool.load_cpu_copy(kv_cpu, new_indices)

        for layer, buf in enumerate(_fused_rows(pool.swa_kv_pool)):
            self.assertEqual(
                _row_values(buf, torch.arange(11, 16)),
                [SWA_OFFSET + layer * LAYER_STRIDE + s for s in range(1, 6)],
            )


if __name__ == "__main__":
    unittest.main()
