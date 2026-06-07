"""Unit tests for the TPU HiCache KV-offload data plane (layer-wise overlapped).

Mirrors the other ``test/manual/tpu`` suites: ``CustomTestCase`` classes guarded
by ``@unittest.skipUnless(_has_tpu_via_jax(), ...)`` so they no-op cleanly off
TPU. Requires torch_tpu + JAX with a TPU backend visible to ``jax.devices()``
(the Pallas gather/scatter kernels JIT-export at first call).

Coverage:
  * Pallas scatter -> gather round-trip (layout + kernel + 128-align + pow2 pad).
  * D2H -> host -> H2D round-trip incl. the PCIe ``copy_`` legs.
  * Real ``TpuFusedMHATokenToKVPool`` + ``TpuFusedMHATokenToKVPoolHost``
    backup -> evict -> async-load round-trip (the overlapped load issues the
    scatter without a blocking sync; the ``.cpu()`` readback forces
    materialization, so this asserts correctness of the overlapped path).
  * The HBM staging reserve estimate equals the actual ``device_stage`` buffer
    (guards drift between ``fused_block_nbytes`` and the real allocation that
    the KV-cache sizing path relies on).

Whether the overlapped load DMA actually *overlaps* compute is an
integration/profiler question, not asserted here.
"""

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu + JAX with a TPU backend visible to jax.devices()"

DEV = "tpu"
# Fused per-layer block shape: [page_size, kv_heads_packed, kv_packing, padded_hd].
PAGE_SIZE, KV_HEADS_PACKED, KV_PACKING, PADDED_HD, NUM_BLOCKS = 16, 2, 2, 128, 64
BLOCK = (PAGE_SIZE, KV_HEADS_PACKED, KV_PACKING, PADDED_HD)


def _rand_blocks(n):
    return torch.randn((n,) + BLOCK, dtype=torch.float32).to(torch.bfloat16)


def _pad_src(src, p):
    n = src.shape[0]
    if p <= n:
        return src
    return torch.cat([src, src[-1:].expand((p - n,) + tuple(src.shape[1:]))], dim=0)


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuKvOffloadKernels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.tpu.kernels.kv_offload import (
            _gather_fn,
            _scatter_fn,
            _tpu_sync,
            pad_to_pow2,
        )

        cls.gather = staticmethod(_gather_fn)
        cls.scatter = staticmethod(_scatter_fn)
        cls.sync = staticmethod(_tpu_sync)
        cls.pad_to_pow2 = staticmethod(pad_to_pow2)

    def test_scatter_gather_roundtrip(self):
        cache = torch.zeros((NUM_BLOCKS,) + BLOCK, dtype=torch.bfloat16, device=DEV)
        block_ids = np.array([3, 7, 1, 18, 9, 40, 2], dtype=np.int64)
        n = len(block_ids)
        src = _rand_blocks(n)
        padded, _ = self.pad_to_pow2(block_ids)
        p = len(padded)
        ids = torch.from_numpy(padded.astype(np.int32)).to(DEV)

        cache[:] = self.scatter(cache, _pad_src(src, p).to(DEV), ids)
        self.sync([cache], wait=True)
        gathered = self.gather(cache, ids)
        self.sync([gathered], wait=True)

        self.assertTrue(torch.equal(gathered[:n].cpu(), src.cpu()))

    def test_d2h_h2d_pcie_roundtrip(self):
        cache = torch.zeros((NUM_BLOCKS,) + BLOCK, dtype=torch.bfloat16, device=DEV)
        block_ids = np.array([5, 11, 30], dtype=np.int64)
        n = len(block_ids)
        src = _rand_blocks(n)
        padded, _ = self.pad_to_pow2(block_ids)
        p = len(padded)
        ids = torch.from_numpy(padded.astype(np.int32)).to(DEV)

        cache[:] = self.scatter(cache, _pad_src(src, p).to(DEV), ids)
        self.sync([cache], wait=True)

        # D2H: gather -> host staging.
        gathered = self.gather(cache, ids)
        self.sync([gathered], wait=False)
        host = torch.empty(gathered.shape, dtype=torch.bfloat16, device="cpu")
        host.copy_(gathered)

        # Wipe the device pages to prove the reload restores them.
        cache[:] = self.scatter(
            cache, torch.zeros((p,) + BLOCK, dtype=torch.bfloat16, device=DEV), ids
        )
        self.sync([cache], wait=True)

        # H2D: host staging -> device staging -> scatter back.
        dev_stage = torch.empty(host.shape, dtype=torch.bfloat16, device=DEV)
        dev_stage.copy_(host)
        cache[:] = self.scatter(cache, dev_stage, ids)
        self.sync([cache], wait=True)
        restored = self.gather(cache, ids)
        self.sync([restored], wait=True)

        self.assertTrue(torch.equal(restored[:n].cpu(), src.cpu()))


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuFusedHostPool(CustomTestCase):
    SIZE, PAGE_SIZE, LAYER_NUM, HEAD_NUM, HEAD_DIM = 256, 16, 2, 2, 64

    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.tpu.kv_pool import (
            TpuFusedMHATokenToKVPool,
            fused_block_nbytes,
            hicache_tpu_stage_blocks,
        )
        from sglang.srt.mem_cache.memory_pool_host import (
            TpuFusedMHATokenToKVPoolHost,
        )

        cls.fused_block_nbytes = staticmethod(fused_block_nbytes)
        cls.hicache_tpu_stage_blocks = staticmethod(hicache_tpu_stage_blocks)
        cls.device_pool = TpuFusedMHATokenToKVPool(
            size=cls.SIZE,
            page_size=cls.PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=cls.HEAD_NUM,
            head_dim=cls.HEAD_DIM,
            layer_num=cls.LAYER_NUM,
            device=DEV,
            enable_memory_saver=False,
        )
        cls.host_pool = TpuFusedMHATokenToKVPoolHost(
            cls.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=cls.PAGE_SIZE,
            layout="page_first_fused",
            device="cpu",
        )

    def test_backup_evict_load_roundtrip(self):
        dev, host = self.device_pool, self.host_pool
        dev_pages = [1, 2]
        orig = {}
        for layer in range(self.LAYER_NUM):
            r = (
                torch.randn((len(dev_pages),) + tuple(dev.kv_buffer[layer].shape[1:]))
                .to(torch.bfloat16)
                .to(DEV)
            )
            dev.kv_buffer[layer][dev_pages] = r
            orig[layer] = r.cpu().clone()

        device_indices = torch.arange(
            dev_pages[0] * self.PAGE_SIZE,
            (dev_pages[-1] + 1) * self.PAGE_SIZE,
            device=DEV,
        )
        host_indices = self.host_pool.alloc(len(device_indices))
        self.assertIsNotNone(host_indices)

        host.backup_from_device_all_layer(
            dev, host_indices, device_indices, "pallas_tpu"
        )
        for layer in range(self.LAYER_NUM):  # evict
            dev.kv_buffer[layer][dev_pages] = 0
            self.assertEqual(
                dev.kv_buffer[layer][dev_pages].cpu().abs().sum().item(), 0.0
            )
        for layer in range(self.LAYER_NUM):  # async load (no blocking sync)
            host.load_to_device_per_layer(
                dev, host_indices, device_indices, layer, "pallas_tpu"
            )
        for layer in range(self.LAYER_NUM):  # .cpu() forces materialization
            self.assertTrue(
                torch.equal(dev.kv_buffer[layer][dev_pages].cpu(), orig[layer])
            )

    def test_staging_reserve_matches_buffer(self):
        """The HBM reserve estimate used by the KV-cache sizing path must equal
        the actual device_stage allocation, or the KV cache is mis-sized."""
        reserve = (
            self.LAYER_NUM
            * self.hicache_tpu_stage_blocks()
            * self.fused_block_nbytes(
                self.HEAD_NUM, self.HEAD_DIM, torch.bfloat16, self.PAGE_SIZE
            )
        )
        actual = (
            self.host_pool.device_stage.numel()
            * self.host_pool.device_stage.element_size()
        )
        self.assertEqual(reserve, actual)


if __name__ == "__main__":
    unittest.main(verbosity=3)
