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


def _seed_pages(dev, dev_page_ids):
    """Fill ``dev_page_ids`` with random KV on every layer; return CPU copies."""
    orig = {}
    for layer in range(dev.layer_num):
        r = (
            torch.randn((len(dev_page_ids),) + tuple(dev.kv_buffer[layer].shape[1:]))
            .to(torch.bfloat16)
            .to(DEV)
        )
        dev.kv_buffer[layer][dev_page_ids] = r
        orig[layer] = r.cpu().clone()
    return orig


def _make_controller(target_dev, target_host, draft_dev=None, draft_host=None):
    """Minimal TpuHiCacheController carrying only the data-plane state the
    write/load rounds touch (the full __init__ wires storage/prefetch threads
    irrelevant to these tests)."""
    from sglang.srt.hardware_backend.tpu.kv_copy_worker import TpuKvCopyWorker
    from sglang.srt.managers.cache_controller import (
        BatchDoneCounter,
        TpuHiCacheController,
    )

    controller = TpuHiCacheController.__new__(TpuHiCacheController)
    controller.io_backend = "torch_tpu"
    controller.mem_pool_device = target_dev
    controller.mem_pool_host = target_host
    controller.has_draft = False
    controller.mem_pool_device_draft = None
    controller.mem_pool_host_draft = None
    controller.layer_done_counter = BatchDoneCounter()
    target_dev.register_layer_transfer_counter(controller.layer_done_counter)
    controller._load_worker = TpuKvCopyWorker("test-kv-offload-h2d")
    controller._backup_worker = TpuKvCopyWorker("test-kv-offload-d2h")
    controller._backup_inflight = None
    controller.write_queue = []
    controller.load_queue = []
    controller.ack_write_queue = []
    controller.ack_load_queue = []
    if draft_dev is not None:
        controller.set_draft_kv_pool(draft_dev, draft_host)
    return controller


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


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuDraftKvOffload(CustomTestCase):
    """Draft (speculative) KV piggyback: the controller reuses the target's
    host/device index pair for a 1-layer draft pool on every round."""

    def test_draft_single_layer_roundtrip(self):
        # EAGLE3 draft = 1 layer; exercises the fused host pool at layer_num=1
        # (sizing, pow2 bucketing, gather/scatter) independent of the controller.
        dev, host = _make_pools(layer_num=1)
        _assert_backup_load_roundtrip(self, dev, host, [3, 0, 6])

    def test_draft_pool_gates_on_shared_counter(self):
        # set_draft_kv_pool must hand the whole-batch gate to the draft device
        # pool: the draft forward waits via its OWN pool's counter.
        target_dev, target_host = _make_pools(layer_num=2)
        draft_dev, draft_host = _make_pools(layer_num=1)
        controller = _make_controller(target_dev, target_host, draft_dev, draft_host)
        self.assertTrue(controller.has_draft)
        self.assertIs(draft_dev.layer_transfer_counter, controller.layer_done_counter)
        self.assertIs(target_dev.layer_transfer_counter, controller.layer_done_counter)

    def test_controller_round_piggybacks_draft(self):
        # One write + one load round through the controller must round-trip
        from sglang.srt.managers.cache_controller import CacheOperation

        target_dev, target_host = _make_pools(layer_num=2)
        draft_dev, draft_host = _make_pools(layer_num=1)
        controller = _make_controller(target_dev, target_host, draft_dev, draft_host)

        pages = [1, 4, 2]
        orig_target = _seed_pages(target_dev, pages)
        orig_draft = _seed_pages(draft_dev, pages)
        device_indices = torch.cat(
            [
                torch.arange(pid * PAGE_SIZE, (pid + 1) * PAGE_SIZE, device=DEV)
                for pid in pages
            ]
        )
        host_indices = target_host.alloc(len(device_indices))
        self.assertIsNotNone(host_indices)

        controller.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id=0)
        )
        controller.start_writing()
        controller._backup_inflight.synchronize()

        for dev in (target_dev, draft_dev):  # evict
            for layer in range(dev.layer_num):
                dev.kv_buffer[layer][pages] = 0

        controller.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id=0)
        )
        producer_id = controller.start_loading()
        self.assertGreaterEqual(producer_id, 0)
        controller.layer_done_counter.set_consumer(producer_id)
        controller.layer_done_counter.wait()

        for layer in range(target_dev.layer_num):
            self.assertTrue(
                torch.equal(
                    target_dev.kv_buffer[layer][pages].cpu(), orig_target[layer]
                )
            )
        self.assertTrue(torch.equal(draft_dev.kv_buffer[0][pages].cpu(), orig_draft[0]))

    def test_controller_round_without_draft_unchanged(self):
        # No-draft config: the piggyback branches must stay dormant.
        from sglang.srt.managers.cache_controller import CacheOperation

        target_dev, target_host = _make_pools(layer_num=2)
        controller = _make_controller(target_dev, target_host)

        pages = [5, 3]
        orig = _seed_pages(target_dev, pages)
        device_indices = torch.cat(
            [
                torch.arange(pid * PAGE_SIZE, (pid + 1) * PAGE_SIZE, device=DEV)
                for pid in pages
            ]
        )
        host_indices = target_host.alloc(len(device_indices))
        controller.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id=0)
        )
        controller.start_writing()
        controller._backup_inflight.synchronize()
        for layer in range(target_dev.layer_num):
            target_dev.kv_buffer[layer][pages] = 0
        controller.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id=0)
        )
        producer_id = controller.start_loading()
        controller.layer_done_counter.set_consumer(producer_id)
        controller.layer_done_counter.wait()
        for layer in range(target_dev.layer_num):
            self.assertTrue(
                torch.equal(target_dev.kv_buffer[layer][pages].cpu(), orig[layer])
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
