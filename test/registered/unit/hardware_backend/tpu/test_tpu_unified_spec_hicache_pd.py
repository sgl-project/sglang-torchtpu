"""CPU unit tests for TPU unified KV pool speculative draft HiCache and equal-TP Raiden P/D."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache import hicache_auto_size, kv_cache_builder
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import CacheOperation
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
    HiCacheDraftPlan,
    _can_pack_hicache_mtp,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.common import (
    DRAFT_FULL_PORT_OFFSET,
    DRAFT_SWA_PORT_OFFSET,
    PORTS_PER_RANK,
    get_raiden_kv_caches,
)
from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import UNIFIED_SWA_KV_LAYOUT, ReshardReq
from sglang_tpu.hooks.draft_kv_pool import (
    bind_unified_swa_draft_host_pool,
    build_unified_hicache_draft_plan,
)
from sglang_tpu.kv_pool import (
    TpuBlockMajorMHATokenToKVPool,
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
)
from sglang_tpu.kv_pool_host import (
    TpuDMAUnifiedHostKVPool,
    _unified_swa_layer_mappings,
    register_unified_hicache_strategy,
    tpu_dma_host_kv_pool_cls,
)
from sglang_tpu.tpu_dma_hicache import page_major_tensors
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "_disagg_raiden_transfer_harness_spec",
    pathlib.Path(__file__).resolve().parents[3]
    / "tpu"
    / "unit"
    / "test_disagg_raiden_transfer.py",
)
_harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _harness
_spec.loader.exec_module(_harness)
RaidenRendezvousTestCase = _harness.RaidenRendezvousTestCase
_fake_tpu_sync = _harness._fake_tpu_sync


class _DoneEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _InMemoryDMATransfer:
    """CPU tensor-copying stand-in for ``TpuDMAKVTransfer``."""

    def __init__(self, io_backend: str, device_pool, page_num: int, *, draft_pools=()):
        self.io_backend = io_backend
        self.page_num = page_num
        self.dev_tensors = [
            t for p in (device_pool, *draft_pools) for t in page_major_tensors(p)
        ]
        self.host_tensors = [
            torch.zeros((page_num, *t.shape[1:]), dtype=t.dtype)
            for t in self.dev_tensors
        ]
        self.backups: list[tuple[list[int], list[int]]] = []
        self.loads: list[tuple[list[int], list[int]]] = []

    def backup(self, device_pages, host_pages):
        dp = list(device_pages)
        hp = list(host_pages)
        self.backups.append((dp, hp))
        for dev, host in zip(self.dev_tensors, self.host_tensors, strict=True):
            host[hp] = dev[dp]
        return _DoneEvent()

    def load(self, host_pages, device_pages):
        hp = list(host_pages)
        dp = list(device_pages)
        self.loads.append((hp, dp))
        for dev, host in zip(self.dev_tensors, self.host_tensors, strict=True):
            dev[dp] = host[hp]
        return _DoneEvent()

    def destroy(self) -> None:
        return None


def _make_target(
    *,
    size: int = 64,
    page_size: int = 4,
    full_layers: tuple[int, ...] = (0, 3),
    swa_layers: tuple[int, ...] = (1, 2, 4, 5, 6),
    head_num: int = 2,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    sliding_window_size: int = 8,
    pool_cls: type = TpuFusedMHATokenToKVPool,
) -> tuple[TpuUnifiedSWAKVPool, TpuUnifiedSWATokenToKVPoolAllocator]:
    pool = TpuUnifiedSWAKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        full=AttnLayerSpec(head_num=head_num, head_dim=head_dim, layer_ids=full_layers),
        swa=AttnLayerSpec(head_num=head_num, head_dim=head_dim, layer_ids=swa_layers),
        device="cpu",
        token_to_kv_pool_class=pool_cls,
    )
    alloc = TpuUnifiedSWATokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=dtype,
        device="cpu",
        kvcache=pool,
        need_sort=False,
        sliding_window_size=sliding_window_size,
    )
    return pool, alloc


def _attach_draft(
    target_alloc: TpuUnifiedSWATokenToKVPoolAllocator,
    *,
    full_layers: tuple[int, ...],
    swa_layers: tuple[int, ...],
    head_num: int = 4,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    sliding_window_size: int = 8,
    pool_cls: type = TpuFusedMHATokenToKVPool,
):
    is_hybrid_swa = len(swa_layers) > 0
    draft_kvc = SimpleNamespace(
        is_hybrid_swa=is_hybrid_swa,
        sliding_window_size=sliding_window_size if is_hybrid_swa else None,
        draft_swa_full_capacity=False,
        model_config=SimpleNamespace(sliding_window_size=None),
        layer_info=SimpleNamespace(
            full_attention_layer_ids=list(full_layers),
            swa_attention_layer_ids=list(swa_layers),
        ),
    )
    full_toks, d_full, d_swa = target_alloc.derive_draft_pool_sizes(
        draft_kvc,
        max_total_num_tokens=target_alloc.size_full,
        full_max_total_num_tokens=target_alloc.size_full,
        swa_max_total_num_tokens=target_alloc.size_swa,
    )
    if not is_hybrid_swa:
        draft_pool = pool_cls(
            size=full_toks,
            page_size=target_alloc.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(full_layers),
            device="cpu",
            enable_memory_saver=False,
        )
    else:
        assert d_full is not None and d_swa is not None
        draft_pool = SWAKVPool(
            size=d_full,
            size_swa=d_swa,
            page_size=target_alloc.page_size,
            dtype=dtype,
            swa_attention_layer_ids=list(swa_layers),
            full_attention_layer_ids=list(full_layers),
            token_to_kv_pool_class=pool_cls,
            enable_memory_saver=False,
            head_num=head_num,
            head_dim=head_dim,
            device="cpu",
        )
    target_alloc.attach_draft_kv_pool(draft_kvc, draft_pool)
    return draft_kvc, draft_pool


class TestUnifiedSpecHiCacheSidecars(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)
        register_unified_hicache_strategy()
        orig_plan = BaseSpecWorker._build_hicache_draft_plan
        self.enterContext(
            patch.object(
                BaseSpecWorker,
                "_build_hicache_draft_plan",
                lambda worker: build_unified_hicache_draft_plan(orig_plan, worker),
            )
        )
        orig_swa_draft = hybrid_pool_assembler.build_swa_draft_pools
        self.enterContext(
            patch.object(
                hybrid_pool_assembler,
                "build_swa_draft_pools",
                lambda *a, **kw: bind_unified_swa_draft_host_pool(
                    orig_swa_draft(*a, **kw), *a, **kw
                ),
            )
        )

    def _build_hicache_group_and_controller(
        self,
        target_pool: TpuUnifiedSWAKVPool,
        target_alloc: TpuUnifiedSWATokenToKVPoolAllocator,
        draft_pool,
        *,
        hicache_ratio: float = 2.0,
        hicache_size: float = 0.0,
    ):
        kv_host, swa_host = TpuDMAUnifiedHostKVPool.build_pair(
            target_pool,
            io_backend="raiden",
            host_to_device_ratio=hicache_ratio,
            host_size=hicache_size,
            page_size=target_pool.page_size,
        )
        full_map, swa_map = _unified_swa_layer_mappings(target_pool)
        max_lid = max(full_map.keys() | swa_map.keys()) + 1
        host_group = HostPoolGroup(
            [
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.KV,
                    host_pool=kv_host,
                    device_pool=target_pool,
                    layer_mapping=full_map,
                    transfer_layer_id_max=max_lid,
                    is_anchor=True,
                ),
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.SWA,
                    host_pool=swa_host,
                    device_pool=target_pool,
                    layer_mapping=swa_map,
                    transfer_layer_id_max=max_lid,
                    device_alloc_fn=target_alloc.swa_attn_allocator.alloc,
                    device_free_fn=target_alloc.swa_attn_allocator.free,
                ),
            ]
        )
        ctrl = object.__new__(TpuDMAHybridCacheController)
        ctrl.mem_pool_device_allocator = target_alloc
        ctrl.mem_pool_host = host_group
        ctrl.page_size = target_pool.page_size
        ctrl.io_backend = "raiden"
        ctrl.device = "cpu"
        ctrl.write_queue = []
        ctrl.load_queue = []
        ctrl.ack_write_queue = []
        ctrl.ack_load_queue = []
        ctrl.layer_done_counter = BatchDoneCounter()

        tree_cache = SimpleNamespace(
            cache_controller=ctrl,
            register_sidecar_pools=lambda specs: setattr(
                ctrl, "_registered_specs", list(specs)
            ),
        )
        fake_mem = SimpleNamespace(
            hicache_io_backend="raiden",
            hicache_mem_layout="page_first",
            hicache_ratio=hicache_ratio,
            hicache_size=hicache_size,
        )
        with (
            patch.object(hybrid_pool_assembler, "get_memory", return_value=fake_mem),
            patch(
                "sglang.srt.platforms.current_platform.get_mha_kv_pool_host_cls",
                side_effect=lambda *_, **__: tpu_dma_host_kv_pool_cls("raiden"),
            ),
        ):
            specs, entries = hybrid_pool_assembler.build_hicache_draft_sidecars(
                draft_device_pools=(draft_pool,),
                tree_cache=tree_cache,
            )
        for entry in entries:
            host_group.add_entry(entry)
        tree_cache.register_sidecar_pools(specs)
        return host_group, ctrl, specs

    def test_hicache_backup_evict_reload_data_integrity(self):
        cases = (
            ("fa_only", (0, 1), (), 8),
            ("swa_only", (), (0, 1), 8),
            ("mixed_compact", (0, 2), (1, 3), 8),
        )
        for label, d_full_ids, d_swa_ids, d_window in cases:
            with self.subTest(case=label):
                with (
                    patch(
                        "sglang_tpu.kv_pool_host.create_tpu_dma_kv_transfer",
                        side_effect=lambda io_b, pool, pg, draft_pools=(): (
                            _InMemoryDMATransfer(
                                io_b, pool, pg, draft_pools=draft_pools
                            )
                        ),
                    ),
                    patch(
                        "sglang_tpu.kv_pool_host.host_memory_budget_bytes",
                        return_value=1 << 40,
                    ),
                ):
                    target_pool, target_alloc = _make_target(size=128, page_size=4)
                    _, draft_pool = _attach_draft(
                        target_alloc,
                        full_layers=d_full_ids,
                        swa_layers=d_swa_ids,
                        sliding_window_size=d_window,
                    )
                    host_group, ctrl, specs = self._build_hicache_group_and_controller(
                        target_pool, target_alloc, draft_pool
                    )

                    # Allocate 2 full pages + 2 SWA pages on device.
                    dev_full = target_alloc.full_attn_allocator.alloc(8)
                    dev_swa = target_alloc.swa_attn_allocator.alloc(8)
                    self.assertIsNotNone(dev_full)
                    self.assertIsNotNone(dev_swa)
                    target_alloc.set_full_to_swa_mapping(dev_full, dev_swa)

                    # Populate deterministic KV values on Target Full, Target SWA (all 3 groups),
                    # Draft Full, and Draft SWA.
                    full_pages = (dev_full[::4] // 4).tolist()
                    swa_prim_pages = (dev_swa[::4] // 4).tolist()
                    swa_expanded_pages = target_pool.expand_swa_pages(swa_prim_pages)

                    expected_t_full = []
                    for lid, buf in enumerate(target_pool.kv_pool.page_buffers()):
                        vals = torch.randn(
                            (len(full_pages), *buf.shape[1:]), dtype=buf.dtype
                        )
                        buf[full_pages] = vals
                        expected_t_full.append(vals.clone())

                    expected_t_swa = []
                    for lid, buf in enumerate(target_pool.kv_pool.page_buffers()):
                        vals = torch.randn(
                            (len(swa_expanded_pages), *buf.shape[1:]), dtype=buf.dtype
                        )
                        buf[swa_expanded_pages] = vals
                        expected_t_swa.append(vals.clone())

                    d_full_sub = (
                        draft_pool.full_kv_pool
                        if isinstance(draft_pool, SWAKVPool)
                        else draft_pool
                    )
                    expected_d_full = []
                    if d_full_ids:
                        for buf in page_major_tensors(d_full_sub):
                            vals = torch.randn(
                                (len(full_pages), *buf.shape[1:]), dtype=buf.dtype
                            )
                            buf[full_pages] = vals
                            expected_d_full.append(vals.clone())

                    expected_d_swa = []
                    if d_swa_ids:
                        d_swa_sub = draft_pool.swa_kv_pool
                        d_swa_pages = target_alloc.draft_swa_sidecar.translate_pages(
                            torch.tensor(swa_prim_pages)
                        ).tolist()
                        for buf in page_major_tensors(d_swa_sub):
                            vals = torch.randn(
                                (len(d_swa_pages), *buf.shape[1:]), dtype=buf.dtype
                            )
                            buf[d_swa_pages] = vals
                            expected_d_swa.append(vals.clone())

                    # Allocate host indices and resolve derived sidecar transfers.
                    host_full = host_group.alloc(8, pool=PoolName.KV)
                    self.assertIsNotNone(host_full)
                    xfers = [PoolTransfer(name=PoolName.SWA, device_indices=dev_swa)]
                    for sp in specs:
                        xfers.append(
                            PoolTransfer(
                                name=sp.pool_name,
                                indices_from_pool=sp.indices_from_pool,
                            )
                        )
                    resolved = host_group.resolve_host_transfers(
                        xfers,
                        primary_device_indices=dev_full,
                        primary_host_indices=host_full,
                    )
                    self.assertIsNotNone(resolved)
                    host_swa = next(
                        t.host_indices for t in resolved if t.name == PoolName.SWA
                    )

                    # Execute D2H backup across all pools.
                    ctrl.write_queue.append(
                        CacheOperation(
                            host_full, dev_full, node_id=10, pool_transfers=resolved
                        )
                    )
                    ctrl.start_writing()
                    self.assertEqual(ctrl.ack_write_queue[0].node_ids, [10])
                    ctrl.ack_write_queue[0].finish_event.synchronize()

                    # Evict and scrub old device buffers, then allocate fresh device pages.
                    for buf in target_pool.kv_pool.page_buffers():
                        buf.zero_()
                    if d_full_ids:
                        for buf in page_major_tensors(d_full_sub):
                            buf.zero_()
                    if d_swa_ids:
                        for buf in page_major_tensors(draft_pool.swa_kv_pool):
                            buf.zero_()

                    # Allocate different physical device pages while keeping old ones live
                    # so reload lands in completely distinct physical page IDs.
                    new_dev_full = target_alloc.full_attn_allocator.alloc(8)
                    new_dev_swa = target_alloc.swa_attn_allocator.alloc(8)
                    self.assertIsNotNone(new_dev_full)
                    self.assertIsNotNone(new_dev_swa)
                    target_alloc.set_full_to_swa_mapping(new_dev_full, new_dev_swa)

                    reload_xfers = [
                        PoolTransfer(
                            name=PoolName.SWA,
                            host_indices=host_swa,
                            device_indices=new_dev_swa,
                        )
                    ]
                    for sp in specs:
                        reload_xfers.append(
                            PoolTransfer(
                                name=sp.pool_name,
                                indices_from_pool=sp.indices_from_pool,
                            )
                        )
                    # Resolve derived device/host indices exactly as HybridCacheController does.
                    for t in reload_xfers:
                        if t.indices_from_pool == PoolName.KV:
                            t.host_indices = host_full
                            t.device_indices = new_dev_full
                        elif t.indices_from_pool == PoolName.SWA:
                            t.host_indices = host_swa
                            t.device_indices = new_dev_swa

                    ctrl.load_queue.append(
                        CacheOperation(
                            host_full,
                            new_dev_full,
                            node_id=11,
                            pool_transfers=reload_xfers,
                        )
                    )
                    ctrl.start_loading()
                    self.assertEqual(ctrl.ack_load_queue[0].node_ids, [11])
                    ctrl.ack_load_queue[0].finish_event.synchronize()

                    # Verify reloaded tensors match expected byte-for-byte on the new pages!
                    new_full_pages = (new_dev_full[::4] // 4).tolist()
                    new_swa_prim = (new_dev_swa[::4] // 4).tolist()
                    new_swa_expanded = target_pool.expand_swa_pages(new_swa_prim)
                    self.assertNotEqual(new_full_pages, full_pages)
                    self.assertNotEqual(new_swa_prim, swa_prim_pages)

                    for buf, exp in zip(
                        target_pool.kv_pool.page_buffers(), expected_t_full, strict=True
                    ):
                        self.assertTrue(torch.equal(buf[new_full_pages], exp))
                    for buf, exp in zip(
                        target_pool.kv_pool.page_buffers(), expected_t_swa, strict=True
                    ):
                        self.assertTrue(torch.equal(buf[new_swa_expanded], exp))
                    if d_full_ids:
                        for buf, exp in zip(
                            page_major_tensors(d_full_sub), expected_d_full, strict=True
                        ):
                            self.assertTrue(torch.equal(buf[new_full_pages], exp))
                    if d_swa_ids:
                        new_d_swa_pages = (
                            target_alloc.draft_swa_sidecar.translate_pages(
                                torch.tensor(new_swa_prim)
                            ).tolist()
                        )
                        # Verify draft SWA translated slots through sidecar.mapping match!
                        mapped_slots = target_alloc.draft_swa_sidecar.mapping[
                            new_dev_full
                        ]
                        self.assertEqual(
                            (mapped_slots[::4] // 4).tolist(), new_d_swa_pages
                        )
                        for buf, exp in zip(
                            page_major_tensors(draft_pool.swa_kv_pool),
                            expected_d_swa,
                            strict=True,
                        ):
                            self.assertTrue(torch.equal(buf[new_d_swa_pages], exp))

                    # Verify host free releases compact SWA host page IDs cleanly.
                    state = host_group.entry_map[PoolName.SWA].host_pool._state
                    compact_cap = state.page_num // state.num_swa_groups
                    self.assertEqual(
                        len(state._compact_swa_free), compact_cap - len(swa_prim_pages)
                    )
                    host_group.release_transfers(resolved)
                    host_group.free(host_full, pool=PoolName.KV)
                    self.assertEqual(len(state._compact_swa_free), compact_cap)
                    self.assertEqual(state._host_to_compact_swa, {})

    def test_hicache_fixed_size_copricing_and_auto_size(self):
        with (
            patch(
                "sglang_tpu.kv_pool_host.create_tpu_dma_kv_transfer",
                side_effect=lambda io_b, pool, pg, draft_pools=(): _InMemoryDMATransfer(
                    io_b, pool, pg, draft_pools=draft_pools
                ),
            ),
            patch(
                "sglang_tpu.kv_pool_host.host_memory_budget_bytes",
                return_value=1 << 40,
            ),
            patch(
                "sglang_tpu.kv_pool_host.sync_fixed_hicache_size",
                side_effect=lambda tokens, _gb: tokens,
            ),
        ):
            p_plain, _ = _make_target(size=64, page_size=4)
            kv_plain, _ = TpuDMAUnifiedHostKVPool.build_pair(
                p_plain,
                io_backend="raiden",
                host_to_device_ratio=2.0,
                host_size=0.01,  # 10 MB
                page_size=4,
            )

            p_spec, a_spec = _make_target(size=64, page_size=4)
            _, draft_pool = _attach_draft(
                a_spec, full_layers=(0, 2), swa_layers=(1, 3), head_num=4, head_dim=128
            )
            kv_spec, _ = TpuDMAUnifiedHostKVPool.build_pair(
                p_spec,
                io_backend="raiden",
                host_to_device_ratio=2.0,
                host_size=0.01,  # 10 MB
                page_size=4,
            )
            self.assertLess(kv_spec.page_num, kv_plain.page_num)

            # Verify hicache_auto_size._draft_bytes handles mixed SWAKVPool + MHA draft on unified target.
            d_bytes = hicache_auto_size._draft_bytes(p_spec, draft_pool)
            expected = (
                hicache_auto_size._pool_bytes(draft_pool.full_kv_pool)
                * p_spec.size
                // draft_pool.full_kv_pool.size
                + hicache_auto_size._pool_bytes(draft_pool.swa_kv_pool)
            )
            self.assertEqual(d_bytes, expected)

            p_fa, a_fa = _make_target(size=64, page_size=4)
            _, d_fa = _attach_draft(a_fa, full_layers=(0, 1), swa_layers=())
            self.assertEqual(
                hicache_auto_size._draft_bytes(p_fa, d_fa),
                hicache_auto_size._pool_bytes(d_fa) * p_fa.size // d_fa.size,
            )

    def test_mtp_hicache_plan_forces_sidecar_and_rejects_multi_runner(self):
        target_pool, target_alloc = _make_target(size=64, page_size=4)
        _, draft_pool = _attach_draft(target_alloc, full_layers=(0,), swa_layers=())

        def _runner(pool):
            return SimpleNamespace(
                token_to_kv_pool=pool,
                model_config=SimpleNamespace(
                    num_nextn_predict_layers=1,
                    hf_config=SimpleNamespace(architectures=["EagleLlamaForCausalLM"]),
                ),
            )

        r1 = _runner(draft_pool)
        algo = SpeculativeAlgorithm.EAGLE
        # Default target (e.g. MHATokenToKVPool / DeepSeekV4TokenToKVPool) still packs MTP:
        self.assertTrue(_can_pack_hicache_mtp(algo, (r1,)))
        default_worker = object.__new__(BaseSpecWorker)
        default_worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(),
                mtp_draft_device_pools=(),
                spec_algorithm=algo,
            )
        )
        default_worker._draft_worker = SimpleNamespace(draft_runners=[r1])

        fake_worker = object.__new__(BaseSpecWorker)
        fake_worker.server_args = SimpleNamespace(
            speculative_algorithm="EAGLE",
            enable_hierarchical_cache=True,
        )
        fake_worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=target_pool,
                mtp_draft_device_pools=(),
                spec_algorithm=algo,
            )
        )
        fake_worker._draft_worker = SimpleNamespace(draft_runners=[r1])
        fake_mem = SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_unified_cache_external_linker=False,
        )
        with (
            patch(
                "sglang.srt.speculative.base_spec_worker.get_memory",
                return_value=fake_mem,
            ),
            patch(
                "sglang.srt.runtime_context.get_memory",
                return_value=fake_mem,
            ),
        ):
            default_plan = default_worker._build_hicache_draft_plan()
            self.assertEqual(default_plan.mode, HiCacheDraftMode.PACKED)

            plan = fake_worker._build_hicache_draft_plan()
            self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
            self.assertEqual(plan.device_pools, (draft_pool,))

            fake_worker._draft_worker = SimpleNamespace(
                draft_runners=[r1, _runner(draft_pool)]
            )
            with self.assertRaisesRegex(
                NotImplementedError, "Multi-runner MTP HiCache offload"
            ):
                fake_worker._build_hicache_draft_plan()

    def test_prepare_staging_and_full_only_pd_decode_host_cache(self):
        target_pool, target_alloc = _make_target(size=64, page_size=4)
        _, draft_pool = _attach_draft(target_alloc, full_layers=(0,), swa_layers=(1,))
        plan = HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR, device_pools=(draft_pool,)
        )
        tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=target_pool, mtp_draft_device_pools=()
            )
        )
        prepared = []
        with (
            patch.object(
                kv_cache_builder,
                "get_memory",
                return_value=SimpleNamespace(
                    enable_hierarchical_cache=True,
                    hicache_write_policy="write_back",
                    hicache_mem_layout="page_first",
                    hicache_size=0,
                    hicache_ratio=2.0,
                ),
            ),
            patch.object(
                kv_cache_builder,
                "get_schedule",
                return_value=SimpleNamespace(page_size=4),
            ),
            patch.object(
                kv_cache_builder,
                "prepare_mha_write_back_staging",
                side_effect=lambda pool, **kw: prepared.append(pool) or None,
            ),
            patch.object(
                kv_cache_builder, "MHATokenToKVPool", TpuFusedMHATokenToKVPool
            ),
        ):
            kv_cache_builder.prepare_hicache_staging(
                tp_worker=tp_worker,
                draft_plan=plan,
            )
        self.assertEqual(prepared, [draft_pool.full_kv_pool, draft_pool.swa_kv_pool])

        # Full-only PD decode host cache (no PoolName.SWA entry) builds PoolName.DRAFT
        # and cleanly skips PoolName.DRAFT_SWA.
        with (
            patch(
                "sglang_tpu.kv_pool_host.create_tpu_dma_kv_transfer",
                side_effect=lambda io_b, pool, pg, draft_pools=(): _InMemoryDMATransfer(
                    io_b, pool, pg, draft_pools=draft_pools
                ),
            ),
            patch(
                "sglang_tpu.kv_pool_host.host_memory_budget_bytes",
                return_value=1 << 40,
            ),
        ):
            kv_host, _ = TpuDMAUnifiedHostKVPool.build_pair(
                target_pool,
                io_backend="raiden",
                host_to_device_ratio=2.0,
                host_size=0.0,
                page_size=4,
            )
            full_only_group = HostPoolGroup(
                [
                    hybrid_pool_assembler.build_pool_entry(
                        name=PoolName.KV,
                        host_pool=kv_host,
                        device_pool=target_pool,
                        layer_mapping={0: 0, 3: 1},
                        transfer_layer_id_max=7,
                        is_anchor=True,
                    )
                ]
            )
            fake_tree = SimpleNamespace(
                cache_controller=SimpleNamespace(
                    mem_pool_host=full_only_group, page_size=4
                )
            )
            fake_mem = SimpleNamespace(
                hicache_io_backend="raiden",
                hicache_mem_layout="page_first",
                hicache_ratio=2.0,
                hicache_size=0.0,
            )
            with (
                patch.object(
                    hybrid_pool_assembler, "get_memory", return_value=fake_mem
                ),
                patch(
                    "sglang.srt.platforms.current_platform.get_mha_kv_pool_host_cls",
                    side_effect=lambda *_, **__: tpu_dma_host_kv_pool_cls("raiden"),
                ),
            ):
                specs, entries = hybrid_pool_assembler.build_hicache_draft_sidecars(
                    draft_device_pools=(draft_pool,), tree_cache=fake_tree
                )
            self.assertEqual([s.pool_name for s in specs], [PoolName.DRAFT])
            self.assertEqual([e.name for e in entries], [PoolName.DRAFT])

        # SWA-only block-major drafter (0 full layers) never exposes 0-layer bundles.
        bm_t_pool, bm_t_alloc = _make_target(size=64, page_size=4)
        _, bm_draft_pool = _attach_draft(
            bm_t_alloc,
            full_layers=(),
            swa_layers=(0, 1),
            pool_cls=TpuBlockMajorMHATokenToKVPool,
        )
        self.assertEqual(bm_draft_pool.get_contiguous_buf_infos(), ([], [], []))
        bm_groups = get_raiden_kv_caches(bm_t_pool, draft_pool=bm_draft_pool)
        self.assertEqual(bm_groups.draft_full, [])
        self.assertEqual(len(bm_groups.draft_swa), 1)


class _CopyingRaidenEngine:
    """In-memory Raiden engine executing real page-major tensor copies on start_read."""

    registry: dict[str, _CopyingRaidenEngine] = {}

    def __init__(
        self,
        *,
        kv_caches,
        node_id: int,
        local_control_port: int,
        max_blocks: int,
        num_slots: int,
        timeout_s: float,
        parallelism: int = 1,
        **kwargs,
    ):
        self.kv_caches = list(kv_caches)
        self.port = local_control_port
        self.max_blocks = max_blocks
        self.num_slots = num_slots
        self.endpoint = f"127.0.0.1:{local_control_port}"
        self.registered: dict[tuple[str, int], list[int]] = {}
        self.started: list[tuple[str, int, str, list[int], list[int]]] = []
        self._sent: list[str] = []
        self._recv: list[str] = []
        _CopyingRaidenEngine.registry[self.endpoint] = self

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> bool:
        self.registered[(req_id, uuid)] = list(block_ids)
        return True

    def start_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote: list[int],
        local: list[int],
    ) -> None:
        assert len(remote) == len(local), f"mismatched lengths: {remote} vs {local}"
        peer = _CopyingRaidenEngine.registry[endpoint]
        allowed = set(peer.registered[(req_id, uuid)])
        missing = [blk for blk in remote if blk not in allowed]
        assert not missing, (
            f"Read {req_id} requested unregistered blocks {missing}; "
            f"registered={peer.registered[(req_id, uuid)]}"
        )
        for src_buf, dst_buf in zip(peer.kv_caches, self.kv_caches, strict=True):
            dst_buf[local] = src_buf[remote]
        self.started.append((req_id, uuid, endpoint, list(remote), list(local)))
        self._recv.append(req_id)
        peer._sent.append(req_id)

    def poll_stats_with_details(self):
        sent = [
            SimpleNamespace(
                req_id=r,
                completed_ns=1_000_000_000,
                elapsed_ns=None,
                peer=None,
                code=None,
                reason=None,
                phase=None,
            )
            for r in self._sent
        ]
        recv = [
            SimpleNamespace(
                req_id=r,
                completed_ns=1_000_000_000,
                elapsed_ns=None,
                peer=None,
                code=None,
                reason=None,
                phase=None,
            )
            for r in self._recv
        ]
        self._sent.clear()
        self._recv.clear()
        return SimpleNamespace(sent=sent, received=recv, failed=[])


class TestUnifiedSpecRaidenPD(RaidenRendezvousTestCase):
    def setUp(self) -> None:
        super().setUp()
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)
        _CopyingRaidenEngine.registry.clear()
        p = patch(
            "sglang_tpu.kv_pool._seq_along_lane_head_dim",
            side_effect=lambda *, head_dim, kv_packing: (
                ((head_dim + 8 * kv_packing - 1) // (8 * kv_packing)) * (8 * kv_packing)
            ),
        )
        p.start()
        self.addCleanup(p.stop)

    def _pump(self, *mgrs) -> None:
        progressed = True
        while progressed:
            progressed = False
            for m in mgrs:
                frames = self.inboxes[m.endpoint].frames
                while frames:
                    m._dispatch(frames.pop(0))
                    progressed = True

    def _setup_mgr(
        self,
        mgr,
        target_pool: TpuUnifiedSWAKVPool,
        draft_pool,
        *,
        local_rank: int,
        chunked: bool,
    ):
        mgr.server_args.nnodes = 1
        mgr.is_dummy_cp_rank = False
        mgr._engine = None
        mgr._swa_engine = None
        mgr._state_engine = None
        mgr._draft_full_engine = None
        mgr._draft_swa_engine = None
        mgr._local_rank = local_rank
        mgr._control_port = 41000 + PORTS_PER_RANK * local_rank
        mgr.local_ip = "127.0.0.1"
        mgr.kv_args.state_types = [StateType.SWA]
        mgr.kv_args.kv_item_lens = target_pool.kv_pool.get_contiguous_buf_infos()[2]
        mgr.kv_item_lens_sum = sum(mgr.kv_args.kv_item_lens)
        mgr._max_request_blocks = lambda: 16
        mgr._swa_max_request_blocks = lambda: 4
        mgr._listener_start_read = False
        groups = get_raiden_kv_caches(target_pool, draft_pool=draft_pool)
        with patch.dict(sys.modules, _fake_tpu_sync(_CopyingRaidenEngine)):
            mgr.set_kv_caches(groups)
        mgr.chunked_transfer = chunked
        return mgr

    def _drain_all(self, *mgrs) -> None:
        for m in mgrs:
            m._drain_engine()

    def test_equal_tp_pd_transfer_all_drafter_topologies_and_modes(self):
        self.assertEqual(PORTS_PER_RANK, 10)
        topologies = (
            (
                "fa_only_m3",
                (0, 3),
                (1, 2, 4, 5, 6),
                (0, 1),
                (),
                8,
                TpuFusedMHATokenToKVPool,
            ),
            (
                "swa_only_m3",
                (0, 3),
                (1, 2, 4, 5, 6),
                (),
                (0, 1),
                8,
                TpuFusedMHATokenToKVPool,
            ),
            (
                "mixed_m3",
                (0, 3),
                (1, 2, 4, 5, 6),
                (0, 2),
                (1, 3, 4),
                8,
                TpuFusedMHATokenToKVPool,
            ),
            ("mixed_m1", (0, 3), (1, 2), (0,), (1,), 8, TpuFusedMHATokenToKVPool),
            (
                "mixed_seq_lane",
                (0, 3),
                (1, 2, 4, 5, 6),
                (0,),
                (1,),
                8,
                TpuSeqAlongLaneKVPool,
            ),
        )
        modes = (
            ("chunked_normal", True, False, 4, 2),
            ("chunked_cover_all", True, True, 2, 2),
            ("whole_pull", False, False, 4, 2),
            (
                "partial_hit",
                True,
                False,
                2,
                2,
            ),  # prefill has 4 full pages, decode pulls last 2
            ("swa_only_hit", True, False, 0, 2),  # full hit, SWA miss
            ("full_100pct_hit", True, False, 0, 0),  # 100% prefix + SWA hit
        )

        room_base = 9000
        for t_idx, (
            t_name,
            t_full_ids,
            t_swa_ids,
            d_full_ids,
            d_swa_ids,
            d_win,
            pool_cls,
        ) in enumerate(topologies):
            for m_idx, (m_name, chunked, cover_all, dec_n_full, dec_n_swa) in enumerate(
                modes
            ):
                with self.subTest(topology=t_name, mode=m_name):
                    _CopyingRaidenEngine.registry.clear()
                    p_target, p_alloc = _make_target(
                        size=128,
                        page_size=4,
                        full_layers=t_full_ids,
                        swa_layers=t_swa_ids,
                        pool_cls=pool_cls,
                    )
                    _, p_draft = _attach_draft(
                        p_alloc,
                        full_layers=d_full_ids,
                        swa_layers=d_swa_ids,
                        sliding_window_size=d_win,
                        pool_cls=pool_cls,
                    )
                    d_target, d_alloc = _make_target(
                        size=128,
                        page_size=4,
                        full_layers=t_full_ids,
                        swa_layers=t_swa_ids,
                        pool_cls=pool_cls,
                    )
                    _, d_draft = _attach_draft(
                        d_alloc,
                        full_layers=d_full_ids,
                        swa_layers=d_swa_ids,
                        sliding_window_size=d_win,
                        pool_cls=pool_cls,
                    )

                    pre = self._setup_mgr(
                        self._prefill(),
                        p_target,
                        p_draft,
                        local_rank=0,
                        chunked=chunked,
                    )
                    dec = self._setup_mgr(
                        self._decode(),
                        d_target,
                        d_draft,
                        local_rank=1,
                        chunked=chunked,
                    )
                    self.assertEqual(pre.kv_layout, UNIFIED_SWA_KV_LAYOUT)

                    has_dfull_engine = bool(d_full_ids)
                    has_dswa_engine = bool(d_swa_ids)
                    if has_dfull_engine:
                        self.assertIsNotNone(pre._draft_full_engine)
                        self.assertEqual(
                            pre._draft_full_engine.port,
                            pre._control_port + DRAFT_FULL_PORT_OFFSET,
                        )
                    else:
                        self.assertIsNone(pre._draft_full_engine)
                    if has_dswa_engine:
                        self.assertIsNotNone(pre._draft_swa_engine)
                        self.assertEqual(
                            pre._draft_swa_engine.port,
                            pre._control_port + DRAFT_SWA_PORT_OFFSET,
                        )
                    else:
                        self.assertIsNone(pre._draft_swa_engine)

                    # Allocate prefill pages (4 full pages, 3 SWA logical pages)
                    p_full_toks = p_alloc.full_attn_allocator.alloc(16)
                    p_swa_toks = p_alloc.swa_attn_allocator.alloc(12)
                    p_alloc.set_full_to_swa_mapping(p_full_toks[-12:], p_swa_toks)
                    p_full_pages = (p_full_toks[::4] // 4).tolist()
                    p_swa_prim = (p_swa_toks[::4] // 4).tolist()
                    p_swa_expanded = p_target.expand_swa_pages(p_swa_prim)

                    # Allocate decode pages on DIFFERENT physical IDs (consume dummy pages first)
                    _ = d_alloc.full_attn_allocator.alloc(8)
                    d_full_toks = (
                        d_alloc.full_attn_allocator.alloc(dec_n_full * 4)
                        if dec_n_full > 0
                        else torch.empty(0, dtype=torch.int64)
                    )
                    d_swa_toks = (
                        d_alloc.swa_attn_allocator.alloc(dec_n_swa * 4)
                        if dec_n_swa > 0
                        else torch.empty(0, dtype=torch.int64)
                    )
                    if dec_n_full > 0 and dec_n_swa > 0:
                        k = min(len(d_full_toks), len(d_swa_toks))
                        d_alloc.set_full_to_swa_mapping(
                            d_full_toks[-k:], d_swa_toks[-k:]
                        )
                    d_full_pages = (
                        (d_full_toks[::4] // 4).tolist() if dec_n_full > 0 else []
                    )
                    d_swa_prim = (
                        (d_swa_toks[::4] // 4).tolist() if dec_n_swa > 0 else []
                    )

                    # Populate deterministic prefill KV tensors across Target + Draft
                    for buf in p_target.kv_pool.page_buffers():
                        buf[p_full_pages] = torch.randn(
                            (len(p_full_pages), *buf.shape[1:]), dtype=buf.dtype
                        )
                        buf[p_swa_expanded] = torch.randn(
                            (len(p_swa_expanded), *buf.shape[1:]), dtype=buf.dtype
                        )
                    p_dfull_sub = (
                        p_draft.full_kv_pool
                        if isinstance(p_draft, SWAKVPool)
                        else p_draft
                    )
                    d_dfull_sub = (
                        d_draft.full_kv_pool
                        if isinstance(d_draft, SWAKVPool)
                        else d_draft
                    )
                    if d_full_ids:
                        for buf in page_major_tensors(p_dfull_sub):
                            buf[p_full_pages] = torch.randn(
                                (len(p_full_pages), *buf.shape[1:]), dtype=buf.dtype
                            )
                    if d_swa_ids:
                        if p_alloc.draft_swa_sidecar is not None:
                            p_dswa_pages = p_alloc.draft_swa_sidecar.translate_pages(
                                torch.tensor(p_swa_prim)
                            ).tolist()
                        else:
                            p_dswa_pages = p_swa_prim
                        for buf in page_major_tensors(p_draft.swa_kv_pool):
                            buf[p_dswa_pages] = torch.randn(
                                (len(p_dswa_pages), *buf.shape[1:]), dtype=buf.dtype
                            )

                    room = room_base + t_idx * 20 + m_idx
                    pre.request_status[room] = KVPoll.Bootstrapping
                    dec.request_status[room] = KVPoll.Bootstrapping

                    # Drive receiver.send_metadata so _draft_swa_blocks_from_state_indices runs!
                    recv_obj = object.__new__(RaidenKVReceiver)
                    recv_obj.kv_mgr = dec
                    recv_obj.bootstrap_room = room
                    recv_obj.bootstrap_infos = self._bootstrap_info(pre)
                    recv_obj.conclude_state = None
                    recv_obj.send_metadata(
                        np.asarray(d_full_pages, dtype=np.int32),
                        aux_index=None,
                        state_indices=[np.asarray(d_swa_prim, dtype=np.int32)],
                    )
                    self._pump(pre, dec)

                    # Drive sender.send so both chunked and closing paths + byte metrics run!
                    send_obj = object.__new__(RaidenKVSender)
                    send_obj.kv_mgr = pre
                    send_obj.bootstrap_room = room
                    send_obj.aux_index = None
                    send_obj._transfer_start_time = None
                    send_obj._transfer_metric = KVTransferMetric()
                    send_obj._transfer_num_kv_indices = 0
                    send_obj._transfer_num_state_indices = 0
                    send_obj._started = False
                    send_obj.conclude_state = None
                    pre.get_kv_replica_factor = lambda: 1

                    if chunked and cover_all:
                        pre._prefill.announce_pages = 1
                        send_obj.num_kv_indices = 2
                        send_obj._last_chunk = False
                        send_obj.send(
                            np.asarray(p_full_pages[:2], dtype=np.int32),
                            state_indices=None,
                        )
                        send_obj._last_chunk = True
                        send_obj.send(
                            np.asarray([], dtype=np.int32),
                            state_indices=[np.asarray(p_swa_prim, dtype=np.int32)],
                        )
                    elif chunked:
                        pre._prefill.announce_pages = 2
                        send_obj.num_kv_indices = len(p_full_pages)
                        send_obj._last_chunk = False
                        send_obj.send(
                            np.asarray(p_full_pages[:2], dtype=np.int32),
                            state_indices=None,
                        )
                        self._pump(pre, dec)
                        dec.maybe_start_pull(room)
                        self._drain_all(pre, dec)
                        send_obj._last_chunk = True
                        send_obj.send(
                            np.asarray(p_full_pages[2:], dtype=np.int32),
                            state_indices=[np.asarray(p_swa_prim, dtype=np.int32)],
                        )
                    else:
                        send_obj.num_kv_indices = len(p_full_pages)
                        send_obj._last_chunk = True
                        send_obj.send(
                            np.asarray(p_full_pages, dtype=np.int32),
                            state_indices=[np.asarray(p_swa_prim, dtype=np.int32)],
                        )

                    # Pump & drain until settled (bounded loop).
                    for _ in range(6):
                        self._pump(pre, dec)
                        dec.maybe_start_pull(room)
                        self._drain_all(pre, dec)
                        if pre.is_send_done(room) and dec.is_recv_done(room):
                            break

                    self.assertTrue(pre.is_send_done(room))
                    self.assertTrue(dec.is_recv_done(room))
                    self.assertFalse(pre.sender_has_active_io(room))
                    self.assertFalse(dec.room_has_active_io(room))

                    # Verify exact byte-for-byte KV data on decode!
                    if dec_n_full > 0:
                        src_tail = (
                            p_full_pages[:dec_n_full]
                            if (chunked and cover_all)
                            else p_full_pages[-dec_n_full:]
                        )
                        for s_buf, r_buf in zip(
                            p_target.kv_pool.page_buffers(),
                            d_target.kv_pool.page_buffers(),
                            strict=True,
                        ):
                            self.assertTrue(
                                torch.equal(r_buf[d_full_pages], s_buf[src_tail])
                            )
                        if d_full_ids:
                            for s_buf, r_buf in zip(
                                page_major_tensors(p_dfull_sub),
                                page_major_tensors(d_dfull_sub),
                                strict=True,
                            ):
                                self.assertTrue(
                                    torch.equal(r_buf[d_full_pages], s_buf[src_tail])
                                )

                    if dec_n_swa > 0:
                        m_t = p_target.layout.num_swa_groups
                        src_swa_exp = p_swa_expanded[-dec_n_swa * m_t :]
                        dst_swa_exp = d_target.expand_swa_pages(d_swa_prim)
                        for s_buf, r_buf in zip(
                            p_target.kv_pool.page_buffers(),
                            d_target.kv_pool.page_buffers(),
                            strict=True,
                        ):
                            self.assertTrue(
                                torch.equal(r_buf[dst_swa_exp], s_buf[src_swa_exp])
                            )
                        if d_swa_ids:
                            src_dswa = p_dswa_pages[-dec_n_swa:]
                            dst_dswa = (
                                d_alloc.draft_swa_sidecar.translate_pages(
                                    torch.tensor(d_swa_prim)
                                ).tolist()
                                if d_alloc.draft_swa_sidecar is not None
                                else d_swa_prim
                            )
                            for s_buf, r_buf in zip(
                                page_major_tensors(p_draft.swa_kv_pool),
                                page_major_tensors(d_draft.swa_kv_pool),
                                strict=True,
                            ):
                                self.assertTrue(
                                    torch.equal(r_buf[dst_dswa], s_buf[src_dswa])
                                )

                    # Verify transfer byte metric accounts for Target + Draft Full & SWA!
                    n_kv_sent = 2 if (chunked and cover_all) else len(p_full_pages)
                    expected_bytes = (
                        n_kv_sent * pre.kv_item_lens_sum
                        + len(p_swa_prim) * pre.state_item_lens_sum
                    )
                    self.assertEqual(
                        send_obj.get_transfer_metric().transfer_total_bytes,
                        expected_bytes,
                    )

                    pre.clear_room(room)
                    dec.clear_room(room)

    def test_asymmetric_tp_raises_not_implemented_for_unified_spec_draft(self):
        for d_full_ids, d_swa_ids in (((0,), ()), ((), (0,)), ((0,), (1,))):
            with self.subTest(d_full=d_full_ids, d_swa=d_swa_ids):
                t_pool, t_alloc = _make_target(size=64, page_size=4)
                _, d_pool = _attach_draft(
                    t_alloc, full_layers=d_full_ids, swa_layers=d_swa_ids
                )
                pre = self._setup_mgr(
                    self._prefill(attn_tp_rank=0, attn_tp_size=4),
                    t_pool,
                    d_pool,
                    local_rank=0,
                    chunked=True,
                )
                dec = self._setup_mgr(
                    self._decode(attn_tp_rank=0, attn_tp_size=2),
                    t_pool,
                    d_pool,
                    local_rank=1,
                    chunked=True,
                )
                with self.assertRaisesRegex(NotImplementedError, "equal attention TP"):
                    dec.register_decode_request(
                        9901,
                        [1, 2],
                        [4, 5, 6],
                        None,
                        [
                            {"rank_ip": "127.0.0.1", "rank_port": 40001},
                            {"rank_ip": "127.0.0.1", "rank_port": 40002},
                        ],
                    )
                with self.assertRaisesRegex(NotImplementedError, "equal attention TP"):
                    pre._on_reshard_req(
                        ReshardReq(
                            room=9902,
                            decode_ip="127.0.0.1",
                            decode_port=40010,
                            push_endpoint="127.0.0.1:41010",
                            decode_blocks=[1],
                            dst_attn_tp=2,
                            decode_swa_blocks=[],
                            swa_push_endpoint="",
                        ).to_frames()
                    )


if __name__ == "__main__":
    unittest.main()
