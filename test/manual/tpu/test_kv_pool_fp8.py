"""Allocation tests for the fused TPU KV pool, focused on fp8 storage.

:class:`TpuFusedMHATokenToKVPool` keeps fp8 as the *storage* dtype (unlike the
CUDA pools, which remap fp8 -> uint8 for ``index_put``): the Pallas RPA kernel
validates that the cache dtype is floating and TPU supports fp8 storage
natively, and the kernel performs the cache writes itself. ``torch.zeros``
cannot materialize fp8 on TPU, so the pool allocates a zeroed uint8 buffer and
reinterprets it as fp8 via ``view``.

These tests assert:

* fp8 (e4m3 / e5m2) buffers are genuinely fp8-typed (not uint8) and zeroed,
* the packing / fused shape is computed from the fp8 dtype (4 elems per lane),
* bf16 is unchanged (``store_dtype == dtype``, packing 2).

They need a TPU device (``torch.zeros(..., device="tpu")``) and are skipped
otherwise.
"""

import importlib.util
import unittest

from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


@unittest.skipIf(_skip_if_no_tpu(), "Requires torch_tpu / a TPU device")
class TestTpuFusedKVPoolFp8(CustomTestCase):
    SIZE = 64
    PAGE_SIZE = 32
    HEAD_NUM = 2
    HEAD_DIM = 128
    LAYER_NUM = 2

    def _make_pool(self, dtype):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        return TpuFusedMHATokenToKVPool(
            size=self.SIZE,
            page_size=self.PAGE_SIZE,
            dtype=dtype,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            device="tpu",
            enable_memory_saver=False,
        )

    def test_bf16_unchanged(self):
        import torch

        pool = self._make_pool(torch.bfloat16)
        self.assertEqual(pool.store_dtype, torch.bfloat16)
        self.assertEqual(pool.kv_packing, 2)
        buf = pool.get_fused_kv_buffer(0)
        self.assertEqual(buf.dtype, torch.bfloat16)
        # num_kv_heads_x2 = 2*2 = 4; packed by 2 -> 2 along the packed axis.
        self.assertEqual(buf.shape[2], 4 // 2)
        self.assertEqual(buf.shape[3], 2)

    def _check_fp8(self, dtype):
        import torch

        pool = self._make_pool(dtype)
        # Storage dtype stays fp8, NOT uint8.
        self.assertEqual(pool.store_dtype, dtype)
        self.assertEqual(pool.kv_packing, 4)
        for layer_id in range(self.LAYER_NUM):
            buf = pool.get_fused_kv_buffer(layer_id)
            self.assertEqual(buf.dtype, dtype)
            # num_kv_heads_x2 = 2*2 = 4; packed by 4 -> 1 along the packed axis.
            self.assertEqual(buf.shape[2], 4 // 4)
            self.assertEqual(buf.shape[3], 4)
            # Freshly allocated cache must be all zeros (all-zero bytes == +0.0).
            self.assertEqual(float(buf.to(torch.float32).abs().sum().item()), 0.0)

    def test_e4m3_stored_as_float8(self):
        import torch

        self._check_fp8(torch.float8_e4m3fn)

    def test_e5m2_stored_as_float8(self):
        import torch

        self._check_fp8(torch.float8_e5m2)

    def test_logical_kv_views_are_fp8(self):
        """``get_key_buffer`` / ``get_value_buffer`` (used by radix/HiCache)
        return fp8 views, not uint8 — they reinterpret the fused buffer."""
        import torch

        pool = self._make_pool(torch.float8_e4m3fn)
        k = pool.get_key_buffer(0)
        v = pool.get_value_buffer(0)
        self.assertEqual(k.dtype, torch.float8_e4m3fn)
        self.assertEqual(v.dtype, torch.float8_e4m3fn)
        self.assertEqual(k.shape[1], self.HEAD_NUM)
        self.assertEqual(k.shape[2], self.HEAD_DIM)


if __name__ == "__main__":
    unittest.main()
