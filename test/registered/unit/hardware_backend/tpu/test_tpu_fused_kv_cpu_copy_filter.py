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

from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase

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
