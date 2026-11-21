"""Unit tests for unified KV pool (--enable-unified-memory, TpuUnifiedSWAKVPool
+ TpuDMAUnifiedHostKVPool) support in prefill KV offload to decode DRAM."""

from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of
from sglang_tpu import pd_decode_swa_hicache
from sglang_tpu.disaggregation.decode_dram_offload import radix_host_ops
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_victim_receiver import (
    DecodeDramVictimReceiver,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_prefix_puller import (
    PrefillDramPrefixPuller,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    PrefillDramVictimPusher,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.disaggregation.raiden.serde import VictimPush
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.kv_pool_host import TpuDMAUnifiedHostKVPool
from sglang_tpu.raiden_transfer import NOOP_EVENT
from sglang_tpu.testing.decode_dram_offload import (
    FULL,
    FakeFuture,
    FakeHostTransfer,
    _bare_decode_manager,
    _bare_prefill_manager,
    host_hit,
    pushed_req,
    sender_on,
)
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

PAGE_SIZE = 64
WINDOW_TOKENS = 128


class _StrictSharedManagerTransfer(FakeHostTransfer):
    """Simulates the single shared KVCacheManager backing TpuDMAUnifiedHostKVPool:
    rejects duplicate active UUIDs and copies physical host blocks byte-for-byte."""

    _ENDPOINTS: dict[str, _StrictSharedManagerTransfer] = {}

    def __init__(self, endpoint: str, num_pages: int, page_bytes: int = 32) -> None:
        super().__init__()
        self._endpoint = endpoint
        self.num_host_pages = num_pages
        self.page_bytes = page_bytes
        self.active_uuids: dict[int, str] = {}
        self.blocks: dict[int, bytes] = {
            i: b"\x00" * page_bytes for i in range(num_pages)
        }
        self._ENDPOINTS[endpoint] = self

    def host_control_address(self, local_ip: str) -> str:
        return self._endpoint

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> None:
        if uuid in self.active_uuids:
            raise RuntimeError(f"uuid {uuid} already has an active registered read")
        super().register_read(req_id, uuid, block_ids)
        self.active_uuids[uuid] = req_id

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint,
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> None:
        assert len(remote_block_ids) == len(local_block_ids), (
            f"remote ({len(remote_block_ids)}) != local ({len(local_block_ids)})"
        )
        super().start_read(
            req_id, uuid, remote_endpoint, remote_block_ids, local_block_ids
        )
        peer = self._ENDPOINTS[remote_endpoint]
        assert uuid in peer.active_uuids, (
            f"uuid {uuid} not registered on {remote_endpoint}"
        )
        reg_id = peer.active_uuids.pop(uuid)
        _, reg_blocks = peer.registrations[reg_id]
        assert list(remote_block_ids) == list(reg_blocks) or set(
            remote_block_ids
        ).issubset(set(reg_blocks))
        for src, dst in zip(remote_block_ids, local_block_ids, strict=True):
            self.blocks[dst] = peer.blocks[src]
        peer.results[reg_id] = "sent"
        self.results[req_id] = "received"

    def backup(self, device_pages, host_pages):
        assert len(device_pages) == len(host_pages)
        return FakeFuture()

    def load(self, host_pages, device_pages):
        assert len(host_pages) == len(device_pages)
        return FakeFuture()


def _build_unified_engine(
    *,
    endpoint: str,
    num_full_layers: int,
    num_swa_layers: int,
    host_ratio: float = 2.0,
    device_tokens: int = 512,
):
    """Build a real UnifiedRadixCache over TpuUnifiedSWAKVPool + TpuDMAUnifiedHostKVPool."""
    full_ids = list(range(num_full_layers))
    swa_ids = list(range(num_full_layers, num_full_layers + num_swa_layers))
    unified_pool = TpuUnifiedSWAKVPool(
        size=device_tokens,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        full=AttnLayerSpec(head_num=2, head_dim=32, layer_ids=full_ids),
        swa=AttnLayerSpec(head_num=2, head_dim=32, layer_ids=swa_ids),
        device="cpu",
        token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
    )

    def make_transfer(io_backend, backing_pool, page_num, **kwargs):
        return _StrictSharedManagerTransfer(endpoint, page_num, page_bytes=32)

    with (
        patch(
            "sglang_tpu.kv_pool_host.host_memory_preflight",
            side_effect=lambda reqs: (sum(reqs), 1 << 40),
        ),
        patch(
            "sglang_tpu.kv_pool_host.create_tpu_dma_kv_transfer",
            side_effect=make_transfer,
        ),
    ):
        full_host, swa_host = TpuDMAUnifiedHostKVPool.build_pair(
            unified_pool,
            io_backend="raiden",
            host_to_device_ratio=host_ratio,
            host_size=0.0,
            page_size=PAGE_SIZE,
        )

    assert full_host.transfer is swa_host.transfer
    assert full_host.shared_allocation_domain is swa_host.shared_allocation_domain

    allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
    allocator.device = torch.device("cpu")
    allocator.page_size = PAGE_SIZE
    allocator._swa_req_ring = False
    allocator.translate_loc_from_full_to_swa.side_effect = lambda idx: idx

    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=MagicMock(spec=ReqToTokenPool),
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE_SIZE,
            sliding_window_size=WINDOW_TOKENS,
            tree_slots=tree_slots_of((FULL, ComponentType.SWA), window=WINDOW_TOKENS),
            tree_core_backend="python",
        )
    )
    controller = MagicMock(spec=TpuDMAHybridCacheController)
    controller.write_policy = "write_back"
    controller.enable_storage = False
    controller.mem_pool_host = SimpleNamespace(
        anchor_entry=SimpleNamespace(host_pool=full_host, host_evict_fn=None),
        available_size=full_host.available_size,
        clear=lambda: (full_host.clear(), swa_host.clear()),
    )
    controller.append_host_mem_release = MagicMock(
        side_effect=AssertionError("released through undrained L3 queue")
    )
    cache.cache_controller = controller
    cache.components[FULL]._full_kv_pool_host = full_host
    cache.components[ComponentType.SWA]._swa_kv_pool_host = swa_host
    pools = {PoolName.KV: full_host, PoolName.SWA: swa_host}
    cache.host_pool_group = SimpleNamespace(
        get_pool=pools.get,
        free=lambda indices, pool=PoolName.KV: pools[pool].free(indices),
    )
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = True
    cache.tree_core.is_write_back = cache.is_write_back = True
    return cache, full_host, swa_host, unified_pool


class TestUnifiedPoolOffloadAndRecall(CustomTestCase):
    def test_push_and_recall_preserve_all_companion_swa_pages_without_uuid_collision(
        self,
    ):
        """Test both M_swa=1 (2 full + 2 SWA layers) and M_swa=3 (2 full + 6 SWA
        layers): every primary and companion SWA physical page must round-trip
        through the shared KVCacheManager without UUID collisions."""
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            for num_swa_layers, expected_m_swa in ((2, 1), (6, 3)):
                with self.subTest(m_swa=expected_m_swa):
                    self._check_unified_round_trip(
                        num_full_layers=2,
                        num_swa_layers=num_swa_layers,
                        expected_m_swa=expected_m_swa,
                    )

    def _check_unified_round_trip(
        self, *, num_full_layers: int, num_swa_layers: int, expected_m_swa: int
    ) -> None:
        p_cache, p_full, p_swa, _ = _build_unified_engine(
            endpoint="10.0.0.1:9100",
            num_full_layers=num_full_layers,
            num_swa_layers=num_swa_layers,
            host_ratio=2.0,
            device_tokens=512,
        )
        d_cache, d_full, d_swa, _ = _build_unified_engine(
            endpoint="10.0.0.2:9200",
            num_full_layers=num_full_layers,
            num_swa_layers=num_swa_layers,
            host_ratio=3.0,
            device_tokens=512,
        )
        self.assertEqual(p_swa._pages_per_logical, expected_m_swa)
        self.assertEqual(d_swa._pages_per_logical, expected_m_swa)

        endpoints: dict[str, any] = {}

        def send_locked(endpoint: str, frames: list[bytes]) -> None:
            endpoints[endpoint]._dispatch(frames)

        pmgr = _bare_prefill_manager()
        pmgr.local_ip, pmgr.rank_port = "10.0.0.1", 4100
        pmgr._send_multipart_locked = Mock(side_effect=send_locked)
        pmgr.dram_offload_channel = DramOffloadChannel(pmgr)
        pmgr._message_handlers.update(pmgr.dram_offload_channel.message_handlers())
        pmgr.dram_offload_channel.bind_host_pools(p_full, p_swa)
        endpoints["tcp://10.0.0.1:4100"] = pmgr

        dmgr = _bare_decode_manager()
        dmgr.kv_args = SimpleNamespace(page_size=PAGE_SIZE, engine_rank=0)
        dmgr.local_ip, dmgr.rank_port = "10.0.0.2", 5100
        dmgr._deferred_ack_targets = {}
        dmgr._deferred_ack_poisoned_rooms = set()
        dmgr._staging_outstanding = {}
        dmgr._send_multipart_locked = Mock(side_effect=send_locked)
        dmgr.dram_offload_channel = DramOffloadChannel(dmgr)
        dmgr._message_handlers.update(dmgr.dram_offload_channel.message_handlers())
        dmgr.dram_offload_channel.bind_host_pools(d_full, d_swa)
        endpoints["tcp://10.0.0.2:5100"] = dmgr

        p_ctx = DramOffloadWorkerContext(
            cache=p_cache,
            host_pool=p_full,
            page_size=PAGE_SIZE,
            channel=pmgr.dram_offload_channel,
            lockstep=RankLockstep(None),
            swa_host_pool=p_swa,
            window_tokens=WINDOW_TOKENS,
        )
        pusher = PrefillDramVictimPusher(p_ctx)
        puller = PrefillDramPrefixPuller(p_ctx)
        puller._max_inflight_tokens = 1 << 20
        puller._max_inflight_swa_tokens = 1 << 20

        d_queue = SimpleNamespace(queue=[], _deferred_releases=[])
        d_ctx = DramOffloadWorkerContext(
            cache=d_cache,
            host_pool=d_full,
            page_size=PAGE_SIZE,
            channel=dmgr.dram_offload_channel,
            lockstep=RankLockstep(None),
            swa_host_pool=d_swa,
            window_tokens=WINDOW_TOKENS,
        )
        receiver = DecodeDramVictimReceiver(d_ctx)
        offerer = DecodeDramPrefixOfferer(d_ctx, d_queue)

        # Insert a 3-page prompt + 2-page SWA window on prefill and stamp distinct bytes
        # on EVERY physical page (including all M_swa - 1 companion pages!).
        tokens = list(range(3000, 3000 + 3 * PAGE_SIZE))
        full_idx = p_full.alloc(3 * PAGE_SIZE)
        swa_idx = p_swa.alloc(WINDOW_TOKENS)
        p_full_phys = radix_host_ops.host_page_ids(p_full, full_idx)
        p_swa_phys = radix_host_ops.host_page_ids(p_swa, swa_idx)
        self.assertEqual(len(p_full_phys), 3)
        self.assertEqual(len(p_swa_phys), 2 * expected_m_swa)

        expected_full_bytes = []
        for i, pid in enumerate(p_full_phys):
            b = f"U_FULL_{i}".encode().ljust(32, b"F")
            p_full.transfer.blocks[pid] = b
            expected_full_bytes.append(b)
        expected_swa_bytes = []
        for i, pid in enumerate(p_swa_phys):
            b = f"U_SWA_GRP_{i}".encode().ljust(32, b"S")
            p_swa.transfer.blocks[pid] = b
            expected_swa_bytes.append(b)

        room1 = 7
        sender_on(pmgr)
        dmgr.register_decode_request(
            room1, [1, 2, 3], [1, 2], None, [{"rank_ip": "10.0.0.1", "rank_port": 4100}]
        )
        node_id, _ = radix_host_ops.insert_host_prefix(
            p_cache,
            RadixKey(array("q", tokens)),
            full_idx,
            PAGE_SIZE,
            swa_indices=swa_idx,
        )
        req1 = pushed_req(room=room1, tokens=tokens)
        req1.last_node = node_id
        pusher.note_decode_of(req1, recalled=False)

        # Push victim from prefill unified arena -> decode unified arena.
        pusher.advance(serving=True)
        self.assertEqual(pusher.stats["started"], 1)
        for _ in range(16):
            receiver.advance()
        self.assertEqual(receiver.stats["landed"], 1)
        pusher.advance(serving=False)
        self.assertEqual(pusher.stats["stored"], 1)

        # Evict prefill host cache completely so prefill holds 0 pages of the session.
        p_cache.evict_host(3 * PAGE_SIZE, ComponentType.FULL)
        p_cache.evict_host(WINDOW_TOKENS, ComponentType.SWA)
        self.assertEqual(host_hit(p_cache, tokens), 0)

        # Turn 2: decode offers host prefix (naming all 2 * M_swa physical SWA pages),
        # prefill recalls into its unified host arena, and every companion page matches!
        room2 = 8
        turn2_tokens = tokens + list(range(4000, 4000 + PAGE_SIZE))
        req2 = SimpleNamespace(
            bootstrap_room=room2,
            origin_input_ids=turn2_tokens,
            extra_key=None,
            cache_salt=None,
            last_node=None,
        )
        pmgr.request_status[room2] = KVPoll.Bootstrapping
        decode_req2 = SimpleNamespace(req=req2)
        d_queue.queue.append(decode_req2)
        offerer.offer_host_prefix(decode_req2)
        self.assertIn(room2, offerer.offers)
        dmgr.register_decode_request(
            room2, [1, 2, 3], [1, 2], None, [{"rank_ip": "10.0.0.1", "rank_port": 4100}]
        )

        puller.advance([req2], waiting_reqs=[])
        puller.advance([req2], waiting_reqs=[])
        puller.advance([req2], waiting_reqs=[])
        self.assertTrue(puller.insert_recalled_host_prefix(req2))
        self.assertEqual(puller.stats["pulled_tokens"], 3 * PAGE_SIZE)
        self.assertEqual(puller.stats["pulled_swa_tokens"], WINDOW_TOKENS)

        leaf = next(iter(p_cache.tree_core.root_node.children.values()))
        while leaf.children:
            leaf = next(iter(leaf.children.values()))
        recalled_full = radix_host_ops.host_pages(p_cache, p_full, leaf)
        recalled_swa = radix_host_ops.window_host_pages(
            p_cache, p_swa, leaf, tokens=len(tokens), window_tokens=WINDOW_TOKENS
        )
        self.assertEqual(
            [p_full.transfer.blocks[pid] for pid in recalled_full], expected_full_bytes
        )
        self.assertEqual(
            [p_swa.transfer.blocks[pid] for pid in recalled_swa], expected_swa_bytes
        )

        d_queue.queue.clear()
        offerer.settle_offers()
        self.assertFalse(bool(offerer.offers))

    def test_shared_arena_eviction_reclaims_combined_physical_shortfall(self):
        """When len(free_pages)=5 and M_swa=3, a push needing 4 FULL + 1 SWA
        logical page requires 4 + 3 = 7 physical pages even though 4 <= 5 and
        3 <= 5 individually."""
        d_cache, d_full, d_swa, _ = _build_unified_engine(
            endpoint="10.0.0.2:9300",
            num_full_layers=2,
            num_swa_layers=6,
            host_ratio=1.5,
            device_tokens=512,
        )
        domain = d_full.shared_allocation_domain
        # Occupy arena with an evictable cached branch so exactly 5 physical pages remain free.
        keep_free = 5
        total_phys = domain.page_num
        need_occupy = total_phys - keep_free
        assert need_occupy >= 4, f"expected >=4 pages to occupy, got {need_occupy}"
        old_tokens = list(range(100, 100 + need_occupy * PAGE_SIZE))
        old_idx = d_full.alloc(need_occupy * PAGE_SIZE)
        radix_host_ops.insert_host_prefix(
            d_cache, RadixKey(array("q", old_tokens)), old_idx, PAGE_SIZE
        )
        self.assertEqual(len(domain.free_pages), keep_free)
        # Individually both look available (4 <= 5, and 1 logical SWA = 3 <= 5),
        # but together they need 7 > 5 physical pages!
        self.assertGreaterEqual(d_full.available_size(), 4 * PAGE_SIZE)
        self.assertGreaterEqual(d_swa.available_size(), 1 * PAGE_SIZE)

        producer = _StrictSharedManagerTransfer("10.0.0.1:9301", 16, page_bytes=32)
        producer.register_read("vf", 101, [0, 1, 2, 3])
        producer.register_read("vs", 102, [4, 5, 6])

        dmgr = _bare_decode_manager()
        dmgr.local_ip, dmgr.rank_port = "10.0.0.2", 5300
        dmgr._send_multipart_locked = Mock()
        dmgr.dram_offload_channel = DramOffloadChannel(dmgr)
        dmgr.dram_offload_channel.bind_host_pools(d_full, d_swa)
        d_ctx = DramOffloadWorkerContext(
            cache=d_cache,
            host_pool=d_full,
            page_size=PAGE_SIZE,
            channel=dmgr.dram_offload_channel,
            lockstep=RankLockstep(None),
            swa_host_pool=d_swa,
            window_tokens=PAGE_SIZE,
        )
        receiver = DecodeDramVictimReceiver(d_ctx)
        dmgr.dram_offload_channel._pushes[("pf", 1)] = VictimPush(
            prefill="pf",
            seq=1,
            ack_endpoint="tcp://10.0.0.1:4300",
            tokens=list(range(500, 500 + 4 * PAGE_SIZE)),
            blocks=[0, 1, 2, 3],
            uuid=101,
            address="10.0.0.1:9301",
            swa_blocks=[4, 5, 6],
            swa_uuid=102,
            swa_address="10.0.0.1:9301",
        )
        for _ in range(16):
            receiver.advance()
        self.assertEqual(receiver.stats["no_room"], 0)
        self.assertEqual(receiver.stats["landed"], 1)

    def test_pd_decode_swa_hicache_builds_unified_full_and_swa_host_stack(self):
        reset_context()
        publish(
            ServerArgs(
                model_path="dummy",
                page_size=PAGE_SIZE,
                enable_hierarchical_cache=True,
                hicache_io_backend="raiden",
                hicache_ratio=2.0,
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
            ),
            role="scheduler",
        )
        try:
            unified_pool = TpuUnifiedSWAKVPool(
                size=512,
                page_size=PAGE_SIZE,
                dtype=torch.bfloat16,
                full=AttnLayerSpec(head_num=2, head_dim=32, layer_ids=[0, 1]),
                swa=AttnLayerSpec(head_num=2, head_dim=32, layer_ids=[2, 3, 4, 5]),
                device="cpu",
                token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
            )
            req_pool = ReqToTokenPool(
                size=4, max_context_len=512, device="cpu", enable_memory_saver=False
            )
            allocator = TpuUnifiedSWATokenToKVPoolAllocator(
                size=512,
                page_size=PAGE_SIZE,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=unified_pool,
                need_sort=False,
                req_to_token_pool=req_pool,
            )
            cache = MagicMock(spec=UnifiedRadixCache)
            params = SimpleNamespace(
                page_size=PAGE_SIZE,
                mtp_draft_device_pools=(),
                token_to_kv_pool_allocator=allocator,
                tp_cache_group=None,
                attn_cp_cache_group=None,
                attn_tp_cache_group=None,
                pp_cache_group=None,
            )
            with (
                tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True),
                patch(
                    "sglang_tpu.kv_pool_host.host_memory_preflight",
                    side_effect=lambda reqs: (sum(reqs), 1 << 40),
                ),
                patch(
                    "sglang_tpu.kv_pool_host.create_tpu_dma_kv_transfer",
                    side_effect=lambda io, bp, pn, **kw: _StrictSharedManagerTransfer(
                        "10.0.0.2:9400", pn
                    ),
                ),
                patch.object(
                    pd_decode_swa_hicache,
                    "_create_cache_controller",
                    return_value=MagicMock(),
                ),
            ):
                stack = pd_decode_swa_hicache._build_full_swa_host_stack(
                    cache=cache,
                    kvcache=unified_pool,
                    params=params,
                    load_cache_event=NOOP_EVENT,
                    storage_backend=None,
                    storage_backend_extra_config=None,
                    prefetch_threshold=256,
                    model_name="dummy",
                    enable_storage_metrics=False,
                )
            self.assertIsInstance(
                stack.component_host_pools[ComponentType.FULL], TpuDMAUnifiedHostKVPool
            )
            self.assertIsInstance(
                stack.component_host_pools[ComponentType.SWA], TpuDMAUnifiedHostKVPool
            )
            self.assertIs(
                stack.component_host_pools[ComponentType.FULL].shared_allocation_domain,
                stack.component_host_pools[ComponentType.SWA].shared_allocation_domain,
            )
        finally:
            reset_context()

    def test_shared_arena_caps_combined_full_and_swa_inflight_recall_budget(self):
        """With M_swa=3 and 24 shared physical pages, _MAX_INFLIGHT_POOL_FRACTION=0.5
        caps total in-flight physical pages to 12: a recall needing 8 full + 2*3=6 SWA
        = 14 physical pages exceeds the combined cap even though 8 <= 12 and 2 <= 4
        individually."""
        p_cache, p_full, p_swa, _ = _build_unified_engine(
            endpoint="10.0.0.1:9500",
            num_full_layers=2,
            num_swa_layers=6,
            host_ratio=3.0,  # 24+ physical pages on 512-token (8-page) unified pool
            device_tokens=512,
        )
        self.assertEqual(p_swa._pages_per_logical, 3)
        total_phys = p_full.shared_allocation_domain.page_num
        max_phys = total_phys // 2
        pmgr = _bare_prefill_manager()
        pmgr.local_ip, pmgr.rank_port = "10.0.0.1", 4500
        pmgr._send_multipart_locked = Mock()
        pmgr.dram_offload_channel = DramOffloadChannel(pmgr)
        pmgr.dram_offload_channel.bind_host_pools(p_full, p_swa)
        p_ctx = DramOffloadWorkerContext(
            cache=p_cache,
            host_pool=p_full,
            page_size=PAGE_SIZE,
            channel=pmgr.dram_offload_channel,
            lockstep=RankLockstep(None),
            swa_host_pool=p_swa,
            window_tokens=WINDOW_TOKENS,
        )
        puller = PrefillDramPrefixPuller(p_ctx)
        # Pick `want` <= max_phys so full budget alone allows it, and window=2 <= swa_budget,
        # while want + 2 * 3 > max_phys!
        want = max_phys - 2
        self.assertGreater(want + 2 * 3, max_phys)
        room = 88
        pmgr._prefill.decode_addr[room] = ("10.0.0.2", 5500)
        pmgr.dram_offload_channel.accept_prefix_offer_locked(
            room, list(range(want)), list(range(2 * 3))
        )
        req = SimpleNamespace(
            bootstrap_room=room,
            origin_input_ids=list(range((want + 1) * PAGE_SIZE)),
            extra_key=None,
            cache_salt=None,
            last_node=None,
        )
        puller.advance([req], waiting_reqs=[])
        self.assertEqual(puller.stats["over_budget"], 1)
        self.assertNotIn(room, puller._plans)

    def test_host_page_metadata_handles_expanded_physical_swa_pages_above_logical_cap(
        self,
    ):
        """With M_swa=3, physical companion page IDs lie in [swa.page_num, state.page_num).
        HostPageMetadata must validate against transfer_page_limit, not swa.page_num."""
        from sglang_tpu.disaggregation.decode_dram_offload.host_page_metadata import (
            HostPageMetadata,
            discard_host_page_metadata,
            register_host_page_metadata,
        )

        _, full_pool, swa_pool, _ = _build_unified_engine(
            endpoint="10.0.0.1:9501",
            num_full_layers=2,
            num_swa_layers=6,
            host_ratio=3.0,
            device_tokens=512,
        )
        self.assertEqual(swa_pool._pages_per_logical, 3)
        self.assertGreater(swa_pool.transfer_page_limit, swa_pool.page_num)
        # Allocate full pages first so SWA primary + companions sit >= swa_pool.page_num.
        held_full = full_pool.alloc(swa_pool.page_num * PAGE_SIZE)
        swa_indices = swa_pool.alloc(PAGE_SIZE)
        self.assertIsNotNone(held_full)
        self.assertIsNotNone(swa_indices)
        phys_swa = swa_pool.transfer_page_ids(swa_indices)
        self.assertEqual(len(phys_swa), 3)
        self.assertTrue(any(p >= swa_pool.page_num for p in phys_swa))

        full_table = HostPageMetadata(full_pool, retain=True)
        swa_table = HostPageMetadata(swa_pool, retain=True)
        tags = [f"swa-{p}".encode() for p in phys_swa]
        swa_table.restore(phys_swa, tags)
        self.assertEqual(swa_table.export(phys_swa), tags)

        # Out-of-arena index is still rejected.
        with self.assertRaisesRegex(ValueError, "outside the pool"):
            swa_table.export([swa_pool.transfer_page_limit])

        # Sibling pool invalidation: restore full tag on shared page, free via SWA.
        target_phys = phys_swa[1]
        full_table.restore([target_phys], [b"stale-full"])
        self.assertEqual(swa_table.export([target_phys]), [])
        swa_table.restore(phys_swa, tags)
        self.assertEqual(full_table.export([target_phys]), [])

        # Releasing SWA clears both primary and companion tags on both tables.
        full_pool.free(held_full)
        swa_pool.free(swa_indices)
        self.assertEqual(swa_table.export(phys_swa), [])
        self.assertEqual(full_table.export(phys_swa), [])

        # Re-populate via codec and verify export works on expanded physical pages.
        codec_tags = {p: f"codec-{p}".encode() for p in phys_swa}
        register_host_page_metadata(
            swa_pool,
            export=lambda pages: [codec_tags[p] for p in pages],
            restore=lambda pages, rows: None,
        )
        self.assertEqual(swa_table.export(phys_swa), [codec_tags[p] for p in phys_swa])
        discard_host_page_metadata(full_pool)

    def test_shared_arena_clear_invalidates_reclaimed_sibling_without_wiping_live_pages(
        self,
    ):
        from sglang_tpu.disaggregation.decode_dram_offload.host_page_metadata import (
            HostPageMetadata,
        )

        _, full_pool, swa_pool, _ = _build_unified_engine(
            endpoint="10.0.0.1:9502",
            num_full_layers=2,
            num_swa_layers=6,
            host_ratio=3.0,
            device_tokens=512,
        )
        full_idx = full_pool.alloc(PAGE_SIZE)
        swa_idx = swa_pool.alloc(PAGE_SIZE)
        full_phys = full_pool.transfer_page_ids(full_idx)
        swa_phys = swa_pool.transfer_page_ids(swa_idx)
        full_table = HostPageMetadata(full_pool, retain=True)
        swa_table = HostPageMetadata(swa_pool, retain=True)
        full_table.restore(full_phys, [b"f0"])
        swa_table.restore(swa_phys, [b"s0", b"s1", b"s2"])

        # Seed a stale copy of a swa-owned physical page in full_table (and verify
        # clearing swa drops it across the domain while keeping full's live page).
        full_table._rows[swa_phys[2]] = b"stale-cross"
        swa_pool.clear()
        self.assertEqual(swa_table.export(swa_phys), [])
        self.assertEqual(full_table.export([swa_phys[2]]), [])
        self.assertEqual(full_table.export(full_phys), [b"f0"])


if __name__ == "__main__":
    unittest.main()
