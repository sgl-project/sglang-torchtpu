"""Unit tests for the TPU HiCache KV-offload data plane (native torch).

Requires torch_tpu + JAX with a TPU backend visible to ``jax.devices()``; classes
skip cleanly off TPU. The reload issues its scatter without a blocking sync, so
the ``.cpu()`` readbacks force materialization to assert correctness. Whether the
load DMA actually *overlaps* compute is a profiler question, not asserted here.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

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


def _has_tpu_via_torch() -> bool:
    return hasattr(torch, "accelerator") and torch.accelerator.is_available()


_SKIP_REASON = "Requires torch_tpu + JAX with a TPU backend visible to jax.devices()"
_TORCH_TPU_SKIP_REASON = "Requires a TPU backend visible to torch.tpu"

DEV = "tpu"
PAGE_SIZE = 16


def _make_pools(size=256, layer_num=2, head_num=2, head_dim=64):
    """Build a (device_pool, host_pool) pair on TPU."""
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
    from sglang_tpu.kv_pool_host import TpuFusedMHATokenToKVPoolHost

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


def _make_controller(target_dev, target_host):
    """Minimal TpuHiCacheController carrying only the data-plane state the
    write/load rounds touch (the full __init__ wires storage/prefetch threads
    irrelevant to these tests)."""
    from sglang.srt.managers.cache_controller import (
        BatchDoneCounter,
        TpuHiCacheController,
    )
    from sglang_tpu.kv_copy_worker import TpuKvCopyWorker

    controller = TpuHiCacheController.__new__(TpuHiCacheController)
    controller.io_backend = "torch_tpu"
    controller.device = target_dev.device  # move_indices re-pins onto this
    controller.mem_pool_device = target_dev
    controller.mem_pool_host = target_host
    controller.layer_done_counter = BatchDoneCounter()
    target_dev.register_layer_transfer_counter(controller.layer_done_counter)
    controller._load_worker = TpuKvCopyWorker("test-kv-offload-h2d")
    controller._backup_worker = TpuKvCopyWorker("test-kv-offload-d2h")
    controller._backup_inflight = None
    controller.write_queue = []
    controller.load_queue = []
    controller.ack_write_queue = []
    controller.ack_load_queue = []
    # Storage (L3) is off in these data-plane tests, but the merged base
    # HiCacheController write/load paths read these flags (set in __init__, which
    # this __new__ fixture bypasses), so mirror their defaults.
    controller.enable_storage = False
    controller.storage_backend = None
    controller.storage_backend_type = None
    controller.enable_storage_metrics = False
    return controller


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuFusedHostPool(CustomTestCase):
    SIZE, PAGE_SIZE, LAYER_NUM, HEAD_NUM, HEAD_DIM = 256, 16, 2, 2, 64

    @classmethod
    def setUpClass(cls):
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang_tpu.kv_pool_host import TpuFusedMHATokenToKVPoolHost

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


@unittest.skipUnless(_has_tpu_via_torch(), _TORCH_TPU_SKIP_REASON)
class TestTpuSeqAlongLaneCpuCopy(CustomTestCase):
    PAGE_SIZE = 256

    def _make_pool(self):
        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool

        pool = TpuSeqAlongLaneKVPool(
            size=self.PAGE_SIZE * 4,
            page_size=self.PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=2,
            device=DEV,
            enable_memory_saver=False,
        )
        pool.cpu_offloading_chunk_size = 3
        for buf in pool.kv_buffer:
            buf.copy_(torch.randn(buf.shape, dtype=torch.bfloat16).to(DEV))
        return pool

    def test_partial_and_full_page_round_trip(self):
        cases = (
            (
                "partial",
                torch.tensor([3, 258, 5, 513]),
                torch.tensor([516, 7, 257, 514]),
                (6, 8, 256, 258, 513, 515, 517),
            ),
            (
                "full",
                torch.arange(256, 512),
                torch.arange(512, 768),
                (256, 511, 768, 1023),
            ),
        )
        for name, source, destination, untouched in cases:
            with self.subTest(name=name):
                pool = self._make_pool()
                payload = pool.get_cpu_copy(source)
                expected = [torch.cat(chunks) for chunks in payload]

                for buf in pool.kv_buffer:
                    buf.fill_(-1)
                pool.load_cpu_copy(payload, destination)

                destination_device = destination.to(DEV)
                blocks = destination_device // self.PAGE_SIZE
                offsets = destination_device % self.PAGE_SIZE
                for layer, buf in enumerate(pool.kv_buffer):
                    restored = buf[blocks, ..., offsets].cpu()
                    self.assertTrue(torch.equal(restored, expected[layer]))
                    for index in untouched:
                        block, offset = divmod(index, self.PAGE_SIZE)
                        self.assertTrue(
                            bool(torch.all(buf[block, ..., offset].cpu() == -1))
                        )


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuHiCacheControllerRound(CustomTestCase):
    """Write + load rounds through TpuHiCacheController on the target pool."""

    def test_single_layer_pool_roundtrip(self):
        # 1-layer pool (the EAGLE3 draft shape): host pool sizing, gather/scatter.
        dev, host = _make_pools(layer_num=1)
        _assert_backup_load_roundtrip(self, dev, host, [3, 0, 6])

    def test_controller_write_load_round(self):
        # write to host, evict on device, load back.
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
        target_dev.tpu_commit_pending_loads()  # the forward's commit point
        controller.layer_done_counter.set_consumer(producer_id)
        controller.layer_done_counter.wait()
        for layer in range(target_dev.layer_num):
            self.assertTrue(
                torch.equal(target_dev.kv_buffer[layer][pages].cpu(), orig[layer])
            )


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestTpuKvOffloadConcurrency(CustomTestCase):
    """Guards the stage/commit split against the XLA buffer-donation race."""

    def test_reload_commit_thread_ownership(self):
        # Deterministic invariant check: staging runs on the worker thread,
        # the kv_buffer scatter is deferred past staging-done, and the commit
        # runs on the thread that drains (the forward-dispatch thread in
        # production). Fails immediately if the scatter ever moves back into
        # the worker's reload task.
        import threading

        from sglang.srt.managers.cache_controller import CacheOperation

        target_dev, target_host = _make_pools(layer_num=2)
        controller = _make_controller(target_dev, target_host)

        pages = [1, 4, 2]
        orig = _seed_pages(target_dev, pages)
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
        for layer in range(target_dev.layer_num):  # evict
            target_dev.kv_buffer[layer][pages] = 0

        # Record which thread runs each half of the reload.
        threads = {}
        orig_stage, orig_commit = target_host.load_stage, target_host.load_commit

        def _stage(*args, **kwargs):
            threads["stage"] = threading.current_thread().name
            return orig_stage(*args, **kwargs)

        def _commit(*args, **kwargs):
            threads["commit"] = threading.current_thread().name
            return orig_commit(*args, **kwargs)

        target_host.load_stage, target_host.load_commit = _stage, _commit
        try:
            controller.load_queue.append(
                CacheOperation(host_indices, device_indices, node_id=0)
            )
            producer_id = controller.start_loading()
            self.assertGreaterEqual(producer_id, 0)

            # Staging done (worker event set) but NOT yet drained: the
            # scatter must not have run — kv_buffer still holds the eviction
            # zeros, and load_commit has not been called from any thread.
            controller.layer_done_counter.events[producer_id].synchronize()
            self.assertEqual(threads.get("stage"), "test-kv-offload-h2d")
            self.assertNotIn("commit", threads)
            for layer in range(target_dev.layer_num):
                self.assertEqual(
                    target_dev.kv_buffer[layer][pages].cpu().abs().sum().item(),
                    0.0,
                    "kv_buffer scatter ran before the commit drain — the "
                    "worker thread touched a donatable buffer",
                )

            # The production commit point (init_forward_metadata ->
            # tpu_commit_pending_loads) drains on the dispatch thread; wait()
            # is only the consumer gate.
            target_dev.tpu_commit_pending_loads()
            controller.layer_done_counter.set_consumer(producer_id)
            controller.layer_done_counter.wait()
            self.assertEqual(threads.get("commit"), threading.current_thread().name)
            for layer in range(target_dev.layer_num):
                self.assertTrue(
                    torch.equal(target_dev.kv_buffer[layer][pages].cpu(), orig[layer])
                )
        finally:
            target_host.load_stage, target_host.load_commit = (
                orig_stage,
                orig_commit,
            )

    def test_reload_survives_concurrent_donation_churn(self):
        # Race miniature: donating dispatches churn the kv_buffer on this
        # thread (as the attention op does) while reloads stage on the
        # worker; drain, then assert bit-exact restoration. The pre-split
        # design crashes or loses scatters here.
        from torch_tpu._internal import pallas as _torch_pallas

        from sglang.srt.managers.cache_controller import CacheOperation

        target_dev, target_host = _make_pools(layer_num=2)
        controller = _make_controller(target_dev, target_host)

        # Identity kernel that donates its input, mirroring the RPA op's
        # donate_argnums/alias pattern: every call invalidates + rebinds the
        # buffer's PJRT handle, then the alias copies the result back so the
        # contents survive for the readback asserts.
        def _donating_identity(x):
            return x + 0

        donate = _torch_pallas.custom_jax_kernel(
            _donating_identity,
            name="test_kv_offload_donating_identity",
            donate_argnums=[0],
            input_output_aliases={0: 0},
            warn_deprecated=False,
        )
        for layer in range(target_dev.layer_num):  # compile before timing races
            donate(target_dev.kv_buffer[layer])

        pages = [5, 1, 3]
        device_indices = torch.cat(
            [
                torch.arange(pid * PAGE_SIZE, (pid + 1) * PAGE_SIZE, device=DEV)
                for pid in pages
            ]
        )
        for _round in range(4):
            orig = _seed_pages(target_dev, pages)
            host_indices = target_host.alloc(len(device_indices))
            self.assertIsNotNone(host_indices)
            controller.write_queue.append(
                CacheOperation(host_indices, device_indices, node_id=0)
            )
            controller.start_writing()
            controller._backup_inflight.synchronize()
            for layer in range(target_dev.layer_num):  # evict
                target_dev.kv_buffer[layer][pages] = 0

            controller.load_queue.append(
                CacheOperation(host_indices, device_indices, node_id=0)
            )
            producer_id = controller.start_loading()

            # Donation churn concurrent with the in-flight staging: 64
            # dispatches x layer_num handle invalidations per round.
            for _ in range(64):
                for layer in range(target_dev.layer_num):
                    donate(target_dev.kv_buffer[layer])

            target_dev.tpu_commit_pending_loads()  # the forward's commit point
            controller.layer_done_counter.set_consumer(producer_id)
            controller.layer_done_counter.wait()
            for layer in range(target_dev.layer_num):
                self.assertTrue(
                    torch.equal(target_dev.kv_buffer[layer][pages].cpu(), orig[layer]),
                    f"round {_round} layer {layer}: reload lost or garbled "
                    "under concurrent buffer donation",
                )
            target_host.free(host_indices)


if __name__ == "__main__":
    unittest.main(verbosity=3)
