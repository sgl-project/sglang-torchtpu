"""Unit tests for the TPU HiCache KV-offload data plane (native torch).

Requires torch_tpu + JAX with a TPU backend visible to ``jax.devices()``; classes
skip cleanly off TPU. The reload issues its scatter without a blocking sync, so
the ``.cpu()`` readbacks force materialization to assert correctness. Whether the
load DMA actually *overlaps* compute is a profiler question, not asserted here.
"""

import unittest

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
PAGE_SIZE = 16


def _make_pools(size=256, layer_num=2, head_num=2, head_dim=64):
    """Build a (device_pool, host_pool) pair on TPU."""
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
    """Seed ``dev_page_ids`` with random KV, backup -> evict -> reload, and
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

    # Per-page token ranges concatenated in page order.
    device_indices = torch.cat(
        [
            torch.arange(pid * page_size, (pid + 1) * page_size, device=DEV)
            for pid in dev_page_ids
        ]
    )
    if host_indices is None:
        host_indices = host.alloc(len(device_indices))
    test.assertIsNotNone(host_indices)

    host.backup_from_device_all_layer(dev, host_indices, device_indices, "torch_tpu")
    for layer in range(dev.layer_num):
        dev.kv_buffer[layer][dev_page_ids] = 0
    host.load_to_device_all(dev, host_indices, device_indices, "torch_tpu")
    for layer in range(dev.layer_num):
        test.assertTrue(
            torch.equal(dev.kv_buffer[layer][dev_page_ids].cpu(), orig[layer])
        )
    if free:
        host.free(host_indices)
    return host_indices


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuFusedHostPool(CustomTestCase):
    SIZE, PAGE_SIZE, LAYER_NUM, HEAD_NUM, HEAD_DIM = 256, 16, 2, 2, 64

    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import TpuFusedMHATokenToKVPoolHost

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
            dev, host_indices, device_indices, "torch_tpu"
        )
        for layer in range(self.LAYER_NUM):  # evict
            dev.kv_buffer[layer][dev_pages] = 0
            self.assertEqual(
                dev.kv_buffer[layer][dev_pages].cpu().abs().sum().item(), 0.0
            )
        host.load_to_device_all(dev, host_indices, device_indices, "torch_tpu")
        for layer in range(self.LAYER_NUM):  # .cpu() forces materialization
            self.assertTrue(
                torch.equal(dev.kv_buffer[layer][dev_pages].cpu(), orig[layer])
            )

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
        host.backup_from_device_all_layer(dev, empty.cpu(), empty, "torch_tpu")
        host.load_to_device_all(dev, empty.cpu(), empty, "torch_tpu")
        for layer in range(self.LAYER_NUM):
            self.assertTrue(torch.equal(dev.kv_buffer[layer].cpu(), snapshot[layer]))

    def test_padded_nonpow2_count(self):
        # n=3 -> p=4: the pow2-padded id repeats the last real page, so the
        # padding rewrites it idempotently and must not corrupt any other page.
        dev, host = _make_pools()
        _assert_backup_load_roundtrip(self, dev, host, [1, 2, 3])

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
