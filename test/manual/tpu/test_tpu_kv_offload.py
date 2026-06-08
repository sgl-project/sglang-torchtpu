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


def _make_pools(size=256, layer_num=2, head_num=2, head_dim=64, stage_blocks=None):
    """Build a (device_pool, host_pool) pair on TPU.

    If ``stage_blocks`` is given, override ``SGLANG_HICACHE_TPU_STAGE_BLOCKS``
    for that pool's dedicated staging buffers (used to exercise the
    overflow/fallback path with a small cap).
    """
    import os
    from unittest.mock import patch

    from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
    from sglang.srt.mem_cache.memory_pool_host import TpuFusedMHATokenToKVPoolHost

    dev = TpuFusedMHATokenToKVPool(
        size=size,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=DEV,
        enable_memory_saver=False,
    )
    env = (
        {}
        if stage_blocks is None
        else {"SGLANG_HICACHE_TPU_STAGE_BLOCKS": str(stage_blocks)}
    )
    with patch.dict(os.environ, env):
        host = TpuFusedMHATokenToKVPoolHost(
            dev,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="page_first_fused",
            device="cpu",
        )
    return dev, host


def _assert_backup_load_roundtrip(
    test, dev, host, dev_page_ids, host_indices=None, free=True
):
    """Seed ``dev_page_ids`` with random KV, backup -> evict -> async-load, and
    assert bit-exact restoration on every layer. ``dev_page_ids`` may be in any
    order / non-consecutive. Returns the host indices used."""
    page_size = dev.page_size
    orig = {}
    for layer in range(dev.layer_num):
        r = (
            torch.randn((len(dev_page_ids),) + tuple(dev.kv_buffer[layer].shape[1:]))
            .to(torch.bfloat16)
            .to(DEV)
        )
        dev.kv_buffer[layer][dev_page_ids] = r
        orig[layer] = r.cpu().clone()

    # device_indices = concat of per-page token ranges, preserving page order.
    device_indices = torch.cat(
        [
            torch.arange(pid * page_size, (pid + 1) * page_size, device=DEV)
            for pid in dev_page_ids
        ]
    )
    if host_indices is None:
        host_indices = host.alloc(len(device_indices))
    test.assertIsNotNone(host_indices)

    host.backup_from_device_all_layer(dev, host_indices, device_indices, "pallas_tpu")
    for layer in range(dev.layer_num):
        dev.kv_buffer[layer][dev_page_ids] = 0
    for layer in range(dev.layer_num):
        host.load_to_device_per_layer(
            dev, host_indices, device_indices, layer, "pallas_tpu"
        )
    for layer in range(dev.layer_num):
        test.assertTrue(
            torch.equal(dev.kv_buffer[layer][dev_page_ids].cpu(), orig[layer])
        )
    if free:
        host.free(host_indices)
    return host_indices


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

    def _kernel_roundtrip(self, block_ids, block=BLOCK):
        cache = torch.zeros((NUM_BLOCKS,) + block, dtype=torch.bfloat16, device=DEV)
        block_ids = np.asarray(block_ids, dtype=np.int64)
        n = len(block_ids)
        src = torch.randn((n,) + block, dtype=torch.float32).to(torch.bfloat16)
        padded, _ = self.pad_to_pow2(block_ids)
        p = len(padded)
        ids = torch.from_numpy(padded.astype(np.int32)).to(DEV)
        cache[:] = self.scatter(cache, _pad_src(src, p).to(DEV), ids)
        self.sync([cache], wait=True)
        g = self.gather(cache, ids)
        self.sync([g], wait=True)
        self.assertTrue(torch.equal(g[:n].cpu(), src.cpu()))

    def test_single_block(self):
        # n == 1 -> p == 1: smallest case, no padding.
        self._kernel_roundtrip([4])

    def test_power_of_2_count(self):
        # n already a power of two -> pad_to_pow2 is a no-op (no padded rows).
        self._kernel_roundtrip([0, 1, 2, 3, 4, 5, 6, 7])

    def test_unaligned_last_dim(self):
        # last dim 64 is NOT a multiple of 128 -> exercises the kernel's internal
        # pad-to-128 + strip path (needs_pad=True), unused by the fused layout
        # but part of the ported kernel contract.
        self._kernel_roundtrip([2, 6, 1], block=(PAGE_SIZE, 2, 2, 64))

    def test_padding_does_not_corrupt_other_pages(self):
        # n=3 -> p=4: the padded id repeats the last real id, so the padding
        # only rewrites the last real page (idempotent) and must not touch any
        # page outside block_ids.
        cache = torch.zeros((NUM_BLOCKS,) + BLOCK, dtype=torch.bfloat16, device=DEV)
        block_ids = np.array([3, 7, 1], dtype=np.int64)
        n = len(block_ids)
        src = _rand_blocks(n)
        padded, _ = self.pad_to_pow2(block_ids)
        p = len(padded)
        self.assertGreater(p, n)  # padding actually happened
        ids = torch.from_numpy(padded.astype(np.int32)).to(DEV)
        cache[:] = self.scatter(cache, _pad_src(src, p).to(DEV), ids)
        self.sync([cache], wait=True)

        g = self.gather(cache, ids)
        self.sync([g], wait=True)
        self.assertTrue(torch.equal(g[:n].cpu(), src.cpu()))

        # Pages never named in block_ids must remain zero.
        untouched = torch.tensor([0, 2, 4, 5, 8], dtype=torch.int32, device=DEV)
        gu = self.gather(cache, untouched)
        self.sync([gu], wait=True)
        self.assertEqual(gu.cpu().abs().sum().item(), 0.0)


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

    def test_scattered_nonconsecutive_pages(self):
        # Non-sorted, non-consecutive device pages exercise the indexed
        # gather/scatter and the host page-id bridge under arbitrary page order.
        _assert_backup_load_roundtrip(self, self.device_pool, self.host_pool, [5, 1, 9])

    def test_empty_transfer_is_noop(self):
        # n == 0 must be a harmless no-op, not a crash, and must not touch KV.
        dev, host = self.device_pool, self.host_pool
        snapshot = [
            dev.kv_buffer[layer].cpu().clone() for layer in range(self.LAYER_NUM)
        ]
        empty = torch.empty((0,), dtype=torch.int64, device=DEV)
        host.backup_from_device_all_layer(dev, empty.cpu(), empty, "pallas_tpu")
        host.load_to_device_per_layer(dev, empty.cpu(), empty, 0, "pallas_tpu")
        for layer in range(self.LAYER_NUM):
            self.assertTrue(torch.equal(dev.kv_buffer[layer].cpu(), snapshot[layer]))

    def test_staging_overflow_fallback(self):
        # stage_blocks=2 but the transfer is 4 pages -> the per-call fallback
        # alloc path in load/backup must still round-trip correctly.
        dev, host = _make_pools(stage_blocks=2)
        self.assertEqual(host.stage_blocks, 2)
        _assert_backup_load_roundtrip(self, dev, host, [1, 2, 3, 4])

    def test_alloc_free_realloc_roundtrip(self):
        # Host-allocator fragmentation: alloc -> free -> realloc must still hand
        # out page-aligned contiguous blocks, or the page-id bridge breaks.
        dev, host = _make_pools()
        ps = self.PAGE_SIZE
        first = host.alloc(2 * ps)
        self.assertIsNotNone(first)
        host.free(first)
        reused = host.alloc(3 * ps)
        self.assertIsNotNone(reused)
        _assert_backup_load_roundtrip(
            self, dev, host, [2, 5, 8], host_indices=reused, free=False
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
