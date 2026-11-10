"""Compare Raiden reshard byte ranges with the actual fused KV pool layout."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.kv_pool import (
    TpuBlockMajorMHATokenToKVPool,
    TpuFusedMHATokenToKVPool,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer(dtype, heads):
    pool = TpuFusedMHATokenToKVPool(
        size=128,
        page_size=64,
        dtype=dtype,
        head_num=heads,
        head_dim=128,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return pool.get_fused_kv_buffer(0)


class TestTpuReshardLayout(unittest.TestCase):
    def test_bf16_writer_ranges_cover_each_destination_token(self):
        for total_heads in (4, 8, 16, 32):
            for swa in (False, True):
                with self.subTest(total_heads=total_heads, swa=swa):
                    self._check_layout(torch.bfloat16, total_heads, swa=swa)

    def _check_layout(self, dtype, total_heads, *, swa=False):
        src_tp, dst_tp = 8, 4
        source = _layer(dtype, max(1, total_heads // src_tp))
        destination = _layer(dtype, max(1, total_heads // dst_tp))
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._kv_caches = [destination]
        mgr._swa_kv_caches = [destination] if swa else []
        mgr.kv_args = SimpleNamespace(total_kv_head_num=total_heads)
        geometry = mgr._make_geometry(src_tp, dst_tp, swa=swa)
        source_width = source.stride(1) * source.element_size()
        destination_width = destination.stride(1) * destination.element_size()
        for dst_rank in range(dst_tp):
            written = set()
            for src_rank in range(dst_rank * 2, dst_rank * 2 + 2):
                src_off, size, src_stride, dst_off, dst_stride = geometry.entry_bytes(
                    src_rank, dst_rank
                )
                self.assertEqual(src_stride, source_width)
                self.assertEqual(dst_stride, destination_width)
                self.assertLessEqual(src_off + size, source_width)
                self.assertLessEqual(dst_off + size, destination_width)
                written.update(range(dst_off, dst_off + size))
            self.assertEqual(written, set(range(destination_width)))

    def test_block_major_bundle_refuses_unequal_tp(self):
        """A bundle's axis 2 is page_size, not the kv-head axis the reshard
        geometry reads, so resharding a bundle must fail instead of computing
        wrong byte ranges."""
        pool = TpuBlockMajorMHATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
        )
        mgr = object.__new__(RaidenKVManager)
        mgr._kv_caches = pool.page_buffers()
        mgr._swa_kv_caches = pool.page_buffers()
        mgr.kv_args = SimpleNamespace(total_kv_head_num=8)
        for swa in (False, True):
            with self.subTest(swa=swa):
                with self.assertRaisesRegex(
                    ValueError, "SGLANG_TPU_USE_BLOCK_MAJOR_KV.*TP 8 on prefill and 4"
                ):
                    mgr._make_geometry(8, 4, swa=swa)


if __name__ == "__main__":
    unittest.main()
