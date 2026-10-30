"""Unit tests for the TPU raiden hybrid (full + SWA) KV-offload path.

Covers the hardware-agnostic contracts with fake pools/futures and a CPU model
of raiden (no TPU): the hybrid assembler consults the platform host-pool hook
(the bug that routed TPU hybrid models to the CUDA host pool), each transfer
issues one all-layer DMA per pool and its ack completes only when every pool's
DMA has landed, load-back keeps index tensors host-resident by rewriting the
tree-owned PoolTransfer in place, a transfer with SWA pages but no Full pages
leaves the Full pool's KV untouched, and speculative-decoding draft sidecars
ride the same rounds (draft DMA issued per transfer, reload ack gates on it,
draft device pool wired to the whole-batch counter, host pool picked via the
platform hook).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter, LayerDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_free_fence import KVFreeFence, fence_hicache_backups
from sglang_tpu.kv_pool_host import (
    TPU_DMA_LAYOUT,
    TpuDMAHostKVPool,
)
from sglang_tpu.raiden_transfer import RaidenKVTransfer
from sglang_tpu.tpu_dma_hybrid_cache_controller import (
    TpuDMAHybridCacheController,
)


class _FakeFuture:
    def __init__(self):
        self.ready = False
        self.synchronized = False

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronized = True
        self.ready = True


class _FakeHostPool:
    """Records all-layer transfer calls; returns controllable futures."""

    def __init__(self, size_per_token: int = 2, page_size: int = 4):
        self.size_per_token = size_per_token
        self.page_size = page_size
        self.backups = []
        self.loads = []
        self.backup_futures: List[_FakeFuture] = []
        self.load_futures: List[_FakeFuture] = []

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self.backups.append((device_pool, host_indices, device_indices))
        fut = _FakeFuture()
        self.backup_futures.append(fut)
        return fut

    def load_all_layer(self, device_pool, host_indices, device_indices):
        self.loads.append((device_pool, host_indices, device_indices))
        fut = _FakeFuture()
        self.load_futures.append(fut)
        return fut


class _FakeEntry:
    def __init__(self, name, host_pool, device_pool):
        self.name = name
        self.host_pool = host_pool
        self.device_pool = device_pool
        self.device_indices_from_anchor_fn = None


class _FakeGroup:
    def __init__(self, anchor: _FakeEntry, extra: List[_FakeEntry]):
        self.anchor_entry = anchor
        self.entries = [anchor, *extra]
        self.entry_map = {anchor.name: anchor}
        for entry in extra:
            self.entry_map[entry.name] = entry


def _make_controller() -> TpuDMAHybridCacheController:
    ctrl = object.__new__(TpuDMAHybridCacheController)
    full_pool = _FakeHostPool()
    swa_pool = _FakeHostPool()
    anchor = _FakeEntry(PoolName.KV, full_pool, device_pool=object())
    swa = _FakeEntry(PoolName.SWA, swa_pool, device_pool=object())
    ctrl.mem_pool_host = _FakeGroup(anchor, [swa])
    ctrl.io_backend = "raiden"
    ctrl.write_queue = []
    ctrl.load_queue = []
    ctrl.ack_write_queue = []
    ctrl.ack_load_queue = []
    ctrl.layer_done_counter = BatchDoneCounter()
    return ctrl


def _op(n_kv: int = 8, n_swa: int = 4) -> CacheOperation:
    return CacheOperation(
        torch.arange(n_kv, dtype=torch.int64),
        torch.arange(n_kv, dtype=torch.int64),
        node_id=1,
        pool_transfers=[
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.arange(n_swa, dtype=torch.int64),
                device_indices=torch.arange(n_swa, dtype=torch.int64),
            )
        ],
    )


class TestPlainKvController(CustomTestCase):
    def test_raiden_plain_kv_supports_forward_reload_wait(self):
        """Plain KV offload must provide the whole-batch forward reload gate."""
        device_pool = MagicMock()

        def init_base(controller, *args, **kwargs):
            controller.mem_pool_device = device_pool
            controller.mem_pool_host = _FakeGroup(
                _FakeEntry(PoolName.KV, _FakeHostPool(), device_pool), []
            )
            controller.layer_done_counter = object.__new__(LayerDoneCounter)

        with (
            patch.object(HybridCacheController, "__init__", init_base),
            patch.object(hybrid_pool_assembler, "build_kv_only_group"),
            patch.object(hybrid_pool_assembler, "get_memory") as memory,
        ):
            memory.return_value.hicache_io_backend = "raiden"
            _, controller = hybrid_pool_assembler.build_kv_only_stack(
                params=MagicMock(),
                kv_pool=device_pool,
                full_layer_mapping={0: 0},
                load_cache_event=None,
                storage_backend=None,
                use_mla=False,
            )

        counter = controller.layer_done_counter
        counter.wait()
        event = _FakeFuture()
        slot = counter.update_producer()
        counter.set_event(slot, event)
        counter.set_consumer(slot)
        counter.wait()
        self.assertTrue(event.synchronized)
        device_pool.register_layer_transfer_counter.assert_called_once_with(counter)


class TestHybridWrite(CustomTestCase):
    def test_write_issues_one_backup_per_pool(self):
        ctrl = _make_controller()
        ctrl.write_queue.append(_op())
        ctrl.start_writing()

        full_pool = ctrl.mem_pool_host.anchor_entry.host_pool
        swa_pool = ctrl.mem_pool_host.entry_map[PoolName.SWA].host_pool
        self.assertEqual(len(full_pool.backups), 1)
        self.assertEqual(len(swa_pool.backups), 1)
        # Each pool's DMA targets its own entry's device pool.
        self.assertIs(
            full_pool.backups[0][0], ctrl.mem_pool_host.anchor_entry.device_pool
        )
        self.assertIs(
            swa_pool.backups[0][0],
            ctrl.mem_pool_host.entry_map[PoolName.SWA].device_pool,
        )

    def test_write_ack_completes_only_when_both_pools_land(self):
        ctrl = _make_controller()
        ctrl.write_queue.append(_op())
        ctrl.start_writing()

        (ack,) = ctrl.ack_write_queue
        full_fut = ctrl.mem_pool_host.anchor_entry.host_pool.backup_futures[0]
        swa_fut = ctrl.mem_pool_host.entry_map[PoolName.SWA].host_pool.backup_futures[0]
        self.assertFalse(ack.finish_event.query())
        full_fut.ready = True
        self.assertFalse(ack.finish_event.query())
        swa_fut.ready = True
        self.assertTrue(ack.finish_event.query())

    def test_write_ack_reports_per_pool_token_counts(self):
        ctrl = _make_controller()
        ctrl.write_queue.append(_op(n_kv=8, n_swa=4))
        ctrl.start_writing()
        (ack,) = ctrl.ack_write_queue
        self.assertEqual(
            ack.num_tokens_by_pool,
            {PoolName.KV.value: 8, PoolName.SWA.value: 4},
        )


class TestWriteThroughBehindInFlightForward(CustomTestCase):
    def _setup(self):
        self.syncs = []
        self.ctrl = _make_controller()
        self.full_pool = self.ctrl.mem_pool_host.anchor_entry.host_pool
        self.fence = KVFreeFence(
            synchronize_device=lambda: self.syncs.append(len(self.full_pool.backups))
        )
        fence_hicache_backups(self.ctrl, self.fence)

    def _launch(self):
        result = SimpleNamespace(copy_done=None)
        self.fence.on_batch_formed()
        self.fence.on_forward_launched(result)
        return result

    def test_backup_waits_for_the_in_flight_forward(self):
        """A request that finished at N-1 is still in batch N, so its inserted
        tail page holds the slot forward N writes; a D2H issued before forward
        N finishes copies that slot stale."""
        self._setup()
        result = self._launch()
        self.ctrl.write_queue.append(_op())
        self.ctrl.start_writing()

        (ack,) = self.ctrl.ack_write_queue
        self.assertEqual(self.full_pool.backups, [])
        self.assertFalse(ack.finish_event.query())

        self.fence.on_forward_done(result)
        self.assertEqual(len(self.full_pool.backups), 1)
        self.full_pool.backup_futures[0].ready = True
        swa_pool = self.ctrl.mem_pool_host.entry_map[PoolName.SWA].host_pool
        swa_pool.backup_futures[0].ready = True
        self.assertTrue(ack.finish_event.query())

    def test_blocking_wait_syncs_the_device_before_issuing(self):
        """The write-back path blocks on the ack; it must wait for the
        in-flight forward rather than spin on a backup never issued."""
        self._setup()
        self._launch()
        self.ctrl.write_queue.append(_op())
        self.ctrl.start_writing()

        (ack,) = self.ctrl.ack_write_queue
        ack.finish_event.synchronize()
        self.assertEqual(self.syncs, [0])
        self.assertEqual(len(self.full_pool.backups), 1)
        self.assertTrue(self.full_pool.backup_futures[0].synchronized)

    def test_blocking_wait_before_a_formed_batch_launches_still_completes(self):
        """A device sync cannot cover a batch formed but not launched; a
        blocking wait in that window must still return, not stall the
        scheduler."""
        self._setup()
        self.fence.on_batch_formed()
        self.ctrl.write_queue.append(_op())
        self.ctrl.start_writing()
        (ack,) = self.ctrl.ack_write_queue
        ack.finish_event.synchronize()
        self.assertEqual(len(self.full_pool.backups), 1)

    def test_backup_issues_at_once_without_an_in_flight_forward(self):
        self._setup()
        self.ctrl.write_queue.append(_op())
        self.ctrl.start_writing()
        self.assertEqual(len(self.full_pool.backups), 1)
        self.assertEqual(self.syncs, [])


class TestHybridLoad(CustomTestCase):
    def test_load_issues_both_pools_and_gates_forward_on_both(self):
        ctrl = _make_controller()
        ctrl.load_queue.append(_op())
        producer_id = ctrl.start_loading()

        full_pool = ctrl.mem_pool_host.anchor_entry.host_pool
        swa_pool = ctrl.mem_pool_host.entry_map[PoolName.SWA].host_pool
        self.assertEqual(len(full_pool.loads), 1)
        self.assertEqual(len(swa_pool.loads), 1)
        # start_loading returns with the DMAs still in flight ...
        self.assertFalse(full_pool.load_futures[0].synchronized)
        self.assertFalse(swa_pool.load_futures[0].synchronized)
        (ack,) = ctrl.ack_load_queue
        self.assertFalse(ack.finish_event.query())
        # ... and the whole-batch gate the consumer waits on lands both.
        ctrl.layer_done_counter.set_consumer(producer_id)
        ctrl.layer_done_counter.wait()
        self.assertTrue(full_pool.load_futures[0].synchronized)
        self.assertTrue(swa_pool.load_futures[0].synchronized)
        self.assertTrue(ack.finish_event.query())

    def test_load_rewrites_tree_owned_transfer_in_place(self):
        # The SWA component's commit slices the same PoolTransfer object it
        # handed to load(); the controller must fill (and keep) its device
        # indices on that object, host-resident.
        ctrl = _make_controller()
        swa_entry = ctrl.mem_pool_host.entry_map[PoolName.SWA]
        swa_alloc = MagicMock(return_value=torch.arange(4, dtype=torch.int64))
        swa_entry.device_alloc_fn = swa_alloc
        swa_entry.device_free_fn = MagicMock()
        swa_entry.host_evict_fn = None
        swa_entry.device_evict_fn = None

        # Hybrid load reads allocator.full_attn_allocator (the SWA composite
        # allocator shape) and falls back to the allocator itself.
        full_allocator = MagicMock(kv_free_fence=None)
        full_allocator.full_attn_allocator.alloc = MagicMock(
            return_value=torch.arange(8, dtype=torch.int64)
        )
        ctrl.mem_pool_device_allocator = full_allocator
        ctrl.device = "cpu"

        tree_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(4, dtype=torch.int64),
            device_indices=None,
        )
        host_indices = torch.arange(8, dtype=torch.int64)
        device_indices = ctrl.load(
            host_indices=host_indices, node_id=1, extra_pools=[tree_transfer]
        )

        self.assertIsNotNone(device_indices)
        self.assertEqual(device_indices.device.type, "cpu")
        # In-place rewrite: the tree-owned object now carries CPU device ids.
        self.assertIsNotNone(tree_transfer.device_indices)
        self.assertEqual(tree_transfer.device_indices.device.type, "cpu")
        swa_alloc.assert_called_once_with(4)


class _FakeReadyFuture:
    def is_ready(self) -> bool:
        return True

    def ok(self) -> bool:
        return True


class _FakeRaidenManager:
    """CPU model of raiden's ``d2h`` / ``h2d`` over one sub-pool's layers.

    A non-empty page list copies those pages in every layer. An empty list
    selects raiden's whole-buffer mode: each layer's entire device buffer is
    copied to (``d2h``) or from (``h2d``) the start of its host buffer.
    """

    def __init__(self, layers: int, device_pages: int, host_pages: int):
        self.device = torch.arange(layers * device_pages * 2.0).view(
            layers, device_pages, 2
        )
        self.host = -1 - torch.arange(layers * host_pages * 2.0).view(
            layers, host_pages, 2
        )

    def d2h(self, device_pages, host_pages, block_counts):
        if not device_pages:
            self.host[:, : self.device.shape[1]] = self.device
        else:
            self.host[:, host_pages] = self.device[:, device_pages]
        return _FakeReadyFuture()

    def h2d(self, host_pages, device_pages, block_counts):
        if not host_pages:
            self.device[:] = self.host[:, : self.device.shape[1]]
        else:
            self.device[:, device_pages] = self.host[:, host_pages]
        return _FakeReadyFuture()


def _raiden_host_pool(mgr: _FakeRaidenManager, page_size: int):
    """A real raiden host pool and transfer driving ``mgr``."""
    pool = object.__new__(TpuDMAHostKVPool)
    pool.page_size = page_size
    pool.page_num = mgr.host.shape[1]
    pool.size = pool.page_num * page_size
    pool.layout = TPU_DMA_LAYOUT
    pool.device = "cpu"
    pool.size_per_token = 1
    pool.lock = threading.RLock()
    pool.transfer = object.__new__(RaidenKVTransfer)
    pool.transfer._mgr = mgr
    pool.clear()
    return pool


class TestHybridTransferWithoutFullPages(CustomTestCase):
    """A hybrid transfer can carry SWA pages but no Full pages, under either
    write policy: load-back when a node's Full KV is still on device but its
    SWA KV was evicted, backup when its Full KV is already on host. The
    controller still issues the Full pool's DMA, and raiden must not see the
    empty page list as the whole buffer."""

    PAGE_SIZE = 2

    def setUp(self):
        self.full_mgr = _FakeRaidenManager(layers=2, device_pages=4, host_pages=8)
        self.swa_mgr = _FakeRaidenManager(layers=2, device_pages=4, host_pages=8)
        self.ctrl = _make_controller()
        self.ctrl.mem_pool_host = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=_raiden_host_pool(self.full_mgr, self.PAGE_SIZE),
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.SWA,
                    host_pool=_raiden_host_pool(self.swa_mgr, self.PAGE_SIZE),
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                ),
            ]
        )
        self.ctrl.mem_pool_device_allocator = SimpleNamespace(kv_free_fence=None)
        self.ctrl.device = "cpu"

    def _tokens(self, pages: List[int]) -> torch.Tensor:
        pages = torch.tensor(pages, dtype=torch.int64)
        return (pages[:, None] * self.PAGE_SIZE + torch.arange(self.PAGE_SIZE)).view(-1)

    def test_swa_only_load_leaves_full_device_kv_intact(self):
        full_device = self.full_mgr.device.clone()
        swa = PoolTransfer(
            name=PoolName.SWA,
            host_indices=self._tokens([5]),
            device_indices=self._tokens([1]),
        )
        self.ctrl.load(host_indices=self._tokens([]), node_id=1, extra_pools=[swa])
        self.ctrl.start_loading()
        self.ctrl.ack_load_queue[0].finish_event.synchronize()

        torch.testing.assert_close(self.swa_mgr.device[:, 1], self.swa_mgr.host[:, 5])
        torch.testing.assert_close(self.full_mgr.device, full_device)

    def test_swa_only_backup_leaves_full_host_kv_intact(self):
        full_host = self.full_mgr.host.clone()
        swa = PoolTransfer(name=PoolName.SWA, device_indices=self._tokens([2]))
        self.ctrl.write(device_indices=self._tokens([]), node_id=1, extra_pools=[swa])
        self.ctrl.ack_write_queue[0].finish_event.synchronize()

        (host_page,) = (swa.host_indices[:: self.PAGE_SIZE] // self.PAGE_SIZE).tolist()
        torch.testing.assert_close(
            self.swa_mgr.host[:, host_page], self.swa_mgr.device[:, 2]
        )
        torch.testing.assert_close(self.full_mgr.host, full_host)


class TestAssemblerPlatformHook(CustomTestCase):
    def test_build_kv_host_pool_consults_platform_hook(self):
        # The hybrid assembler previously hard-coded the CUDA host-pool picker,
        # so TPU hybrid models got a CUDA host pool that died on the
        # page_first_fused layout. It must go through the platform hook.
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        built = {}

        class _RecordingPool:
            def __init__(self, *args, **kwargs):
                built["args"] = args
                built["kwargs"] = kwargs

        fake_platform = MagicMock()
        fake_platform.get_mha_kv_pool_host_cls = MagicMock(return_value=_RecordingPool)

        args = ServerArgs(model_path="dummy")
        kv_pool = object()
        reset_context()
        publish(args, role="scheduler")
        try:
            with patch("sglang.srt.platforms.current_platform", fake_platform):
                # The assembler reads hicache sizing off the memory bag; the publish above configures it.
                pool = hybrid_pool_assembler.build_kv_host_pool(
                    kv_pool=kv_pool,
                    page_size=64,
                    use_mla=False,
                    pool_label="swa",
                )
        finally:
            reset_context()

        fake_platform.get_mha_kv_pool_host_cls.assert_called_once_with(kv_pool)
        self.assertIsInstance(pool, _RecordingPool)
        self.assertEqual(built["kwargs"]["pool_label"], "swa")


class TestDraftSidecar(CustomTestCase):
    def _controller_with_draft(self):
        ctrl = _make_controller()
        draft = _FakeEntry(PoolName.DRAFT, _FakeHostPool(), device_pool=object())
        ctrl.mem_pool_host.entry_map[PoolName.DRAFT] = draft
        return ctrl, draft

    def _op_with_draft(self):
        # The draft sidecar's indices follow the full pool
        # (SidecarPoolSpec(indices_from_pool=PoolName.KV)).
        op = _op(n_kv=8, n_swa=4)
        op.pool_transfers.append(
            PoolTransfer(
                name=PoolName.DRAFT,
                host_indices=torch.arange(8, dtype=torch.int64),
                device_indices=torch.arange(8, dtype=torch.int64),
            )
        )
        return op

    def test_write_piggybacks_draft_pool(self):
        # Guards the draft rejection this path used to raise: a spec-decoding
        # write-back must issue the draft pool's DMA alongside full + SWA.
        ctrl, draft = self._controller_with_draft()
        ctrl.write_queue.append(self._op_with_draft())
        ctrl.start_writing()
        self.assertEqual(len(draft.host_pool.backups), 1)
        self.assertIs(draft.host_pool.backups[0][0], draft.device_pool)

    def test_load_ack_gates_on_draft_dma(self):
        # The whole-batch reload gate must not release before the draft KV
        # has landed: a target-only conjunction would let the draft forward
        # read stale draft KV.
        ctrl, draft = self._controller_with_draft()
        ctrl.load_queue.append(self._op_with_draft())
        producer_id = ctrl.start_loading()
        self.assertGreaterEqual(producer_id, 0)
        (ack,) = ctrl.ack_load_queue
        self.assertEqual(len(draft.host_pool.loads), 1)
        # Target pools landed, draft still in flight: the ack must stay closed.
        anchor_fut = ctrl.mem_pool_host.anchor_entry.host_pool.load_futures[0]
        swa_fut = ctrl.mem_pool_host.entry_map[PoolName.SWA].host_pool.load_futures[0]
        anchor_fut.ready = True
        swa_fut.ready = True
        self.assertFalse(ack.finish_event.query())
        draft.host_pool.load_futures[0].ready = True
        self.assertTrue(ack.finish_event.query())


class TestDraftCounterRegistration(CustomTestCase):
    def _register(self, name):
        ctrl = _make_controller()
        entry = _FakeEntry(name, _FakeHostPool(), device_pool=MagicMock())
        with patch.object(
            HybridCacheController, "register_host_pool_entry"
        ) as base_register:
            ctrl.register_host_pool_entry(entry)
        base_register.assert_called_once_with(entry)
        return ctrl, entry

    def test_draft_entry_gates_on_batch_counter(self):
        # The draft model's forward gates on its own pool's transfer counter;
        # without this registration the counter is None and the draft forward
        # consumes reloaded draft KV without waiting for the DMA.
        for name in (PoolName.DRAFT, PoolName.DRAFT_SWA):
            ctrl, entry = self._register(name)
            entry.device_pool.register_layer_transfer_counter.assert_called_once_with(
                ctrl.layer_done_counter
            )

    def test_non_draft_entry_keeps_its_counter_wiring(self):
        # Target pools get the counter through the SWAKVPool wrapper at
        # controller init; re-registering per entry would be wrong.
        ctrl, entry = self._register(PoolName.SWA)
        entry.device_pool.register_layer_transfer_counter.assert_not_called()


class TestKVFreeFenceCoverage(CustomTestCase):
    def test_pool_not_indexed_by_kv_pages_is_rejected(self):
        """A load into a Mamba state slot bypasses the KV free fence, so a
        forward still writing the freed slot could overwrite the loaded state."""
        mamba = _FakeEntry(PoolName.MAMBA, _FakeHostPool(), device_pool=MagicMock())
        with self.assertRaises(NotImplementedError):
            _make_controller().register_host_pool_entry(mamba)

        def init_base(controller, *args, **kwargs):
            controller.mem_pool_host = _FakeGroup(
                _FakeEntry(PoolName.KV, _FakeHostPool(), MagicMock()), [mamba]
            )

        with patch.object(HybridCacheController, "__init__", init_base):
            with self.assertRaises(NotImplementedError):
                TpuDMAHybridCacheController()

    def test_load_inside_forward_write_target_is_rejected(self):
        """A raiden load inside forward_write_target() could be given a page an
        in-flight forward still writes, since only forward allocations may borrow."""
        ctrl = _make_controller()
        allocator = MagicMock()
        allocator.full_attn_allocator.alloc.return_value = torch.arange(8)
        allocator.kv_free_fence = KVFreeFence(synchronize_device=MagicMock())
        ctrl.mem_pool_device_allocator = allocator
        with allocator.kv_free_fence.forward_write_target():
            with self.assertRaises(AssertionError):
                ctrl.load(host_indices=torch.arange(8))
        allocator.full_attn_allocator.alloc.assert_not_called()

        self.assertIsNotNone(ctrl.load(host_indices=torch.arange(8)))


class TestDraftSidecarPlatformHook(CustomTestCase):
    def test_draft_host_pool_consults_platform_hook(self):
        # Same bug class as build_kv_host_pool above, on the draft-sidecar
        # path: hard-coding the CUDA host-pool picker gives spec-decoding
        # draft sidecars a CUDA host pool on TPU raiden.
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

        built = {}

        class _RecordingPool:
            def __init__(self, *args, **kwargs):
                built["args"] = args
                built["kwargs"] = kwargs

        class _FakeMHAPool:
            pass

        fake_platform = MagicMock()
        fake_platform.get_mha_kv_pool_host_cls = MagicMock(return_value=_RecordingPool)

        pool = _FakeMHAPool()
        with (
            patch("sglang.srt.platforms.current_platform", fake_platform),
            patch("sglang.srt.mem_cache.memory_pool.MHATokenToKVPool", _FakeMHAPool),
        ):
            host_pool = hybrid_pool_assembler._build_mha_mla_host_pool(
                pool=pool,
                host_to_device_ratio=2.0,
                page_size=64,
                layout="page_first_fused",
                allocator_type="default",
                pool_label="draft",
            )

        fake_platform.get_mha_kv_pool_host_cls.assert_called_once_with(pool)
        self.assertIsInstance(host_pool, _RecordingPool)
        self.assertEqual(built["kwargs"]["pool_label"], "draft")


class TestDispatch(CustomTestCase):
    def test_raiden_io_backend_selects_tpu_hybrid_controller(self):
        with patch.object(
            TpuDMAHybridCacheController, "__init__", return_value=None
        ) as init:
            ctrl = HybridCacheController.create_for_backend(io_backend="raiden")
        self.assertIsInstance(ctrl, TpuDMAHybridCacheController)
        init.assert_called_once()


if __name__ == "__main__":
    unittest.main()
