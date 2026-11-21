"""Unit tests for symmetric decode KV caching & recall
(SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE=1) in prefill KV offload to decode DRAM."""

from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of
from sglang_tpu.disaggregation.decode_dram_offload import (
    dram_offload_hooks,
    radix_host_ops,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_victim_receiver import (
    DecodeDramVictimReceiver,
)
from sglang_tpu.disaggregation.decode_dram_offload.dram_offload_hooks import (
    DecodeDramOffloadExtension,
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
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.testing.decode_dram_offload import (
    FULL,
    HYBRID_PAGE,
    FakeHostPool,
    FakeHostTransfer,
    FakeRaidenController,
    _bare_decode_manager,
    _bare_prefill_manager,
    full_cache,
    host_hit,
    insert_tokens,
    pushed_req,
    sender_on,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _BytePool(FakeHostTransfer):
    """Host transfer double copying real byte blocks between prefill and decode."""

    _PEERS: dict[str, _BytePool] = {}

    def __init__(self, endpoint: str, capacity_pages: int, page_bytes: int = 32):
        super().__init__()
        self._endpoint = endpoint
        self.page_bytes = page_bytes
        self.blocks: dict[int, bytes] = {
            i: b"\x00" * page_bytes for i in range(capacity_pages)
        }
        self._PEERS[endpoint] = self

    def host_control_address(self, local_ip: str) -> str:
        return self._endpoint

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint,
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> None:
        super().start_read(
            req_id, uuid, remote_endpoint, remote_block_ids, local_block_ids
        )
        peer = self._PEERS[remote_endpoint]
        matched = [
            rid for rid, (ruuid, _) in peer.registrations.items() if ruuid == uuid
        ]
        assert len(matched) == 1
        reg_id = matched[0]
        _, reg_blocks = peer.registrations[reg_id]
        reg_set = set(reg_blocks)
        for src, dst in zip(remote_block_ids, local_block_ids, strict=True):
            assert src in reg_set
            self.blocks[dst] = peer.blocks[src]
        peer.results[reg_id] = "sent"
        self.results[req_id] = "received"


def _real_hybrid_cache(*, window: int = 128, write_back: bool = True):
    """Real paged FULL+SWA UnifiedRadixCache over real CPU device & host pools."""
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", page_size=HYBRID_PAGE)
    )
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=1024,
        size_swa=512,
        page_size=HYBRID_PAGE,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=32,
        swa_attention_layer_ids=[1, 3],
        full_attention_layer_ids=[0, 2],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=1024,
        size_swa=512,
        page_size=HYBRID_PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=HYBRID_PAGE,
            sliding_window_size=window,
            tree_slots=tree_slots_of((FULL, ComponentType.SWA), window=window),
            tree_core_backend="python",
        )
    )
    controller = FakeRaidenController(host_capacity=4096)
    controller.host.page_size = HYBRID_PAGE
    controller.write_policy = "write_back" if write_back else "write_through"
    swa_host = FakeHostPool(2048, page_size=HYBRID_PAGE)
    controller.extra_pools[PoolName.SWA] = swa_host
    controller.append_host_mem_release = MagicMock(
        side_effect=AssertionError("released through undrained L3 queue")
    )
    cache.cache_controller = controller
    cache.components[FULL]._full_kv_pool_host = controller.host
    cache.components[ComponentType.SWA]._swa_kv_pool_host = swa_host
    pools = {PoolName.KV: controller.host, PoolName.SWA: swa_host}
    cache.host_pool_group = SimpleNamespace(
        get_pool=pools.get,
        free=lambda indices, pool=PoolName.KV: pools[pool].free(indices),
    )
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = True
    cache.tree_core.is_write_back = cache.is_write_back = write_back
    if not write_back:
        cache.tree_core.write_through_threshold = 1
    cache._write_back_copy_ahead = None
    return cache, allocator, req_to_token_pool


def _make_finished_decode_req(
    *,
    cache: UnifiedRadixCache,
    allocator,
    req_to_token_pool: ReqToTokenPool,
    prompt_tokens: list[int],
    generated_tokens: list[int],
):
    """Allocate KV slots on decode for ``prompt_tokens + generated_tokens`` and
    build a finished Req ready for ``UnifiedRadixCache.checkpoint``."""
    full_tokens = prompt_tokens + generated_tokens
    row_idx = 1
    if isinstance(allocator, SWATokenToKVPoolAllocator):
        full_kv = allocator.full_attn_allocator.alloc(len(full_tokens))
        swa_kv = allocator.swa_attn_allocator.alloc(len(full_tokens))
        allocator.full_to_swa_index_mapping[full_kv] = swa_kv
        kv_indices = full_kv
    else:
        kv_indices = allocator.alloc(len(full_tokens))
    req_to_token_pool.write((row_idx, slice(0, len(full_tokens))), kv_indices)
    return SimpleNamespace(
        rid="dec-turn1",
        origin_input_ids=array("q", prompt_tokens),
        output_ids=list(generated_tokens) + [999999],
        full_untruncated_fill_ids=array("q", full_tokens),
        extra_key=None,
        cache_salt=None,
        priority=0,
        session=None,
        session_id=None,
        kv_rotation_base=None,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        last_node=cache.root_node_handle(),
        lock=cache.lock(cache.root_node_handle()),
        swa_prefix_lock_released=False,
        swa_stale_window_floor=0,
        swa_media_end=0,
        swa_branching_seqlen=None,
        multimodal_inputs=None,
        kv=ReqKvInfo(
            req_pool_idx=row_idx,
            cache_protected_len=0,
            cache_inserted_len=0,
        ),
        finished=lambda: True,
        refresh_fill_ids=lambda: None,
    )


class TestSymmetricDecodeCacheHooks(CustomTestCase):
    def test_decode_add_respects_symmetric_flag_and_preserves_warmup_skip(self):
        upstream = MagicMock(return_value="ok")
        with tpu_envs.SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE.override(False):
            req = SimpleNamespace(skip_radix_cache_insert=False)
            dram_offload_hooks._decode_add(upstream, "q", req)
            self.assertTrue(req.skip_radix_cache_insert)

        with tpu_envs.SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE.override(True):
            req_real = SimpleNamespace(skip_radix_cache_insert=False)
            dram_offload_hooks._decode_add(upstream, "q", req_real)
            self.assertFalse(req_real.skip_radix_cache_insert)

            req_fake = SimpleNamespace(skip_radix_cache_insert=True)
            dram_offload_hooks._decode_add(upstream, "q", req_fake)
            self.assertTrue(req_fake.skip_radix_cache_insert)

    def test_send_kv_metadata_offers_when_symmetric_cache_holds_short_swa_prompt(self):
        """On a hybrid SWA decode, prompts shorter than sliding_window_size
        arrive with decode_prefix_len=0; symmetric caching still offers host KV."""
        prealloc = SimpleNamespace(tree_cache=MagicMock())
        req = SimpleNamespace(origin_input_ids=list(range(128)))
        decode_req = SimpleNamespace(req=req)
        offload = SimpleNamespace(source=MagicMock())
        upstream = MagicMock()
        with (
            tpu_envs.SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE.override(True),
            patch.object(dram_offload_hooks, "_extension", return_value=offload),
        ):
            dram_offload_hooks._send_kv_metadata(
                upstream, prealloc, decode_req, [1, 2], 64, decode_prefix_len=0
            )
        offload.source.offer_host_prefix.assert_called_once_with(decode_req)

    def test_send_kv_metadata_suppresses_offer_on_host_receive_destination(self):
        """A host-staged decode receive (_pre_alloc_host) passes
        destination=KVTransferDestination.HOST and must offer nothing."""
        prealloc = SimpleNamespace(tree_cache=MagicMock())
        req = SimpleNamespace(origin_input_ids=list(range(128)))
        decode_req = SimpleNamespace(req=req)
        offload = SimpleNamespace(source=MagicMock())
        upstream = MagicMock()
        with (
            tpu_envs.SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE.override(True),
            patch.object(dram_offload_hooks, "_extension", return_value=offload),
        ):
            dram_offload_hooks._send_kv_metadata(
                upstream,
                prealloc,
                decode_req,
                [1, 2],
                64,
                decode_prefix_len=0,
                destination=KVTransferDestination.HOST,
            )
        offload.source.offer_host_prefix.assert_not_called()


class TestSymmetricMultiTurnRecall(CustomTestCase):
    def _wire_pair(self, p_cache, d_cache, page_size: int):
        p_host = radix_host_ops.anchor_host_pool(p_cache)
        p_swa = radix_host_ops.swa_host_pool(p_cache)
        d_host = radix_host_ops.anchor_host_pool(d_cache)
        d_swa = radix_host_ops.swa_host_pool(d_cache)

        p_host.transfer = _BytePool("10.0.0.1:7100", 128)
        d_host.transfer = _BytePool("10.0.0.2:8100", 128)
        if p_swa is not None:
            p_swa.transfer = _BytePool("10.0.0.1:7101", 128)
            d_swa.transfer = _BytePool("10.0.0.2:8101", 128)

        endpoints: dict[str, any] = {}

        def send_locked(endpoint: str, frames: list[bytes]) -> None:
            endpoints[endpoint]._dispatch(frames)

        pmgr = _bare_prefill_manager()
        pmgr.local_ip, pmgr.rank_port = "10.0.0.1", 4200
        pmgr._send_multipart_locked = Mock(side_effect=send_locked)
        pmgr.dram_offload_channel = DramOffloadChannel(pmgr)
        pmgr._message_handlers.update(pmgr.dram_offload_channel.message_handlers())
        pmgr.dram_offload_channel.bind_host_pools(p_host, p_swa)
        endpoints["tcp://10.0.0.1:4200"] = pmgr

        dmgr = _bare_decode_manager()
        dmgr.kv_args = SimpleNamespace(page_size=page_size, engine_rank=0)
        dmgr.local_ip, dmgr.rank_port = "10.0.0.2", 5200
        dmgr._deferred_ack_targets = {}
        dmgr._deferred_ack_poisoned_rooms = set()
        dmgr._staging_outstanding = {}
        dmgr._send_multipart_locked = Mock(side_effect=send_locked)
        dmgr.dram_offload_channel = DramOffloadChannel(dmgr)
        dmgr._message_handlers.update(dmgr.dram_offload_channel.message_handlers())
        dmgr.dram_offload_channel.bind_host_pools(d_host, d_swa)
        endpoints["tcp://10.0.0.2:5200"] = dmgr

        p_ctx = DramOffloadWorkerContext(
            cache=p_cache,
            host_pool=p_host,
            page_size=page_size,
            channel=pmgr.dram_offload_channel,
            lockstep=RankLockstep(None),
            swa_host_pool=p_swa,
            window_tokens=radix_host_ops.window_tokens(p_cache, page_size),
        )
        d_queue = SimpleNamespace(queue=[], _deferred_releases=[])
        d_ctx = DramOffloadWorkerContext(
            cache=d_cache,
            host_pool=d_host,
            page_size=page_size,
            channel=dmgr.dram_offload_channel,
            lockstep=RankLockstep(None),
            swa_host_pool=d_swa,
            window_tokens=radix_host_ops.window_tokens(d_cache, page_size),
        )
        puller = PrefillDramPrefixPuller(p_ctx)
        puller._max_inflight_tokens = 1 << 20
        puller._max_inflight_swa_tokens = 1 << 20
        pusher = PrefillDramVictimPusher(p_ctx)
        receiver = DecodeDramVictimReceiver(d_ctx)
        offerer = DecodeDramPrefixOfferer(d_ctx, d_queue)
        d_cache.cache_controller.decode_dram_extension = DecodeDramOffloadExtension(
            landing=receiver, source=offerer
        )
        return pmgr, dmgr, puller, pusher, receiver, offerer, d_queue

    def test_multi_turn_decode_generated_kv_recalls_delta_on_write_back_and_write_through(
        self,
    ):
        """Turn 1 computes P0 on prefill and generates G0 on decode; Turn 2
        sends P0 + G0 + U1 and recalls only the missing G0 pages from decode DRAM
        when prefill still holds P0 (and P0 + G0 when prefill evicted P0), with
        zero host page leaks when prefill pushes an overlapping P0 victim."""
        with (
            tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True),
            tpu_envs.SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE.override(True),
        ):
            for policy in ("write_back", "write_through", "write_through_selective"):
                for prefill_kept_p0 in (True, False):
                    with self.subTest(policy=policy, prefill_kept_p0=prefill_kept_p0):
                        self._check_full_attention_multi_turn(
                            policy=policy, prefill_kept_p0=prefill_kept_p0
                        )

    def _check_full_attention_multi_turn(
        self, *, policy: str, prefill_kept_p0: bool
    ) -> None:
        write_back = policy == "write_back"
        p_cache, p_alloc = full_cache(kv_size=64, host_capacity=64)
        d_cache, d_alloc = full_cache(kv_size=64, host_capacity=64)
        for c in (p_cache, d_cache):
            c.tree_core.is_write_back = c.is_write_back = write_back
            if not write_back:
                c.tree_core.write_through_threshold = (
                    2 if policy == "write_through_selective" else 1
                )
                c.cache_controller.write_policy = policy
        pmgr, dmgr, puller, pusher, receiver, offerer, d_queue = self._wire_pair(
            p_cache, d_cache, page_size=1
        )
        p_host = radix_host_ops.anchor_host_pool(p_cache)
        d_host = radix_host_ops.anchor_host_pool(d_cache)

        p0 = list(range(100, 108))  # 8 prompt tokens
        g0 = list(range(200, 206))  # 6 decode-generated assistant tokens
        u1 = list(range(300, 304))  # 4 Turn-2 user follow-up tokens

        # Turn 1 on prefill: prefill caches P0 and evicts it to prefill DRAM.
        room1 = 5
        sender_on(pmgr)
        dmgr.register_decode_request(
            room1, [1, 2], [], None, [{"rank_ip": "10.0.0.1", "rank_port": 4200}]
        )
        insert_tokens(p_cache, p_alloc, p0)
        if policy == "write_through_selective":
            # Second hit on prefill promotes P0 across write_through_threshold=2.
            insert_tokens(p_cache, p_alloc, p0)
        pusher.note_decode_of(pushed_req(room=room1, tokens=p0), recalled=False)
        if not write_back:
            p_cache.cache_controller.start_writing()
            p_cache.writing_check()
        p_cache.evict(EvictParams(num_tokens=len(p0)))
        if not prefill_kept_p0:
            p_cache.evict_host(len(p0))
            self.assertEqual(host_hit(p_cache, p0), 0)

        # Turn 1 finishes on decode: decode checkpoints P0 + G0 while HBM is hot!
        d_req1 = _make_finished_decode_req(
            cache=d_cache,
            allocator=d_alloc,
            req_to_token_pool=d_cache.req_to_token_pool,
            prompt_tokens=p0,
            generated_tokens=g0,
        )
        dram_offload_hooks._checkpoint(
            UnifiedRadixCache.checkpoint, d_cache, d_req1, up_to=len(p0) + len(g0)
        )
        d_cache.writing_check()

        # Stamp distinct bytes on decode's host pages for P0 + G0.
        d_leaf = d_cache.tree_core.node_by_id(d_req1.last_node)
        d_pages = radix_host_ops.host_pages(d_cache, d_host, d_leaf)
        self.assertEqual(len(d_pages), len(p0) + len(g0))
        for idx, pid in enumerate(d_pages):
            d_host.transfer.blocks[pid] = f"DEC:{idx}".encode().ljust(32, b".")
        if prefill_kept_p0:
            p_leaf = next(iter(p_cache.tree_core.root_node.children.values()))
            for idx, pid in enumerate(
                radix_host_ops.host_pages(p_cache, p_host, p_leaf)
            ):
                p_host.transfer.blocks[pid] = f"DEC:{idx}".encode().ljust(32, b".")

        # Turn 2 arrives: prompt is P0 + G0 + U1!
        room2 = 6
        turn2_tokens = p0 + g0 + u1
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
            room2, [1, 2], [], None, [{"rank_ip": "10.0.0.1", "rank_port": 4200}]
        )

        puller.advance([req2], waiting_reqs=[])
        plan = puller._plans[room2]
        if prefill_kept_p0:
            # Only G0's 6 delta pages are pulled over the network!
            self.assertEqual((plan.first_page, plan.num_pages), (len(p0), len(g0)))
        else:
            self.assertEqual((plan.first_page, plan.num_pages), (0, len(p0) + len(g0)))
        puller.advance([req2], waiting_reqs=[])
        puller.advance([req2], waiting_reqs=[])
        self.assertTrue(puller.insert_recalled_host_prefix(req2))
        self.assertEqual(host_hit(p_cache, turn2_tokens), len(p0) + len(g0))

        # Verify overlapping victim push of P0 into decode (which already holds P0 + G0)
        # deduplicates cleanly without leaking host pages!
        avail_before = d_host.available_size()
        p_host.used = p_host.capacity  # force pressure on prefill
        if prefill_kept_p0:
            pusher.advance(serving=True)
            if pusher.stats["started"]:
                for _ in range(16):
                    receiver.advance()
                self.assertEqual(d_host.available_size(), avail_before)

        d_queue.queue.clear()
        offerer.settle_offers()
        self.assertFalse(bool(offerer.offers))

    def test_hybrid_swa_symmetric_decode_checkpoints_and_recalls_generated_window(self):
        for write_back in (True, False):
            with self.subTest(write_back=write_back):
                self._check_hybrid_swa_symmetric_recall(write_back=write_back)

    def _check_hybrid_swa_symmetric_recall(self, *, write_back: bool) -> None:
        with (
            tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True),
            tpu_envs.SGLANG_PD_OFFLOAD_SYMMETRIC_DECODE_CACHE.override(True),
        ):
            p_cache, _, _ = _real_hybrid_cache(window=128, write_back=write_back)
            d_cache, d_alloc, d_r2t = _real_hybrid_cache(
                window=128, write_back=write_back
            )
            pmgr, dmgr, puller, _, _, offerer, d_queue = self._wire_pair(
                p_cache, d_cache, page_size=HYBRID_PAGE
            )
            p_swa = radix_host_ops.swa_host_pool(p_cache)
            d_swa = radix_host_ops.swa_host_pool(d_cache)

            p0 = list(range(1000, 1000 + 2 * HYBRID_PAGE))
            g0 = list(range(2000, 2000 + HYBRID_PAGE))
            u1 = list(range(3000, 3000 + HYBRID_PAGE))

            d_req1 = _make_finished_decode_req(
                cache=d_cache,
                allocator=d_alloc,
                req_to_token_pool=d_r2t,
                prompt_tokens=p0,
                generated_tokens=g0,
            )
            dram_offload_hooks._checkpoint(
                UnifiedRadixCache.checkpoint, d_cache, d_req1, up_to=len(p0) + len(g0)
            )
            d_cache.writing_check()

            d_leaf = d_cache.tree_core.node_by_id(d_req1.last_node)
            d_swa_pages = radix_host_ops.window_host_pages(
                d_cache,
                d_swa,
                d_leaf,
                tokens=len(p0) + len(g0),
                window_tokens=128,
            )
            self.assertEqual(len(d_swa_pages), 2)
            for i, pid in enumerate(d_swa_pages):
                d_swa.transfer.blocks[pid] = f"GEN_SWA_{i}".encode().ljust(32, b"W")

            room2 = 9
            turn2_tokens = p0 + g0 + u1
            req2 = SimpleNamespace(
                bootstrap_room=room2,
                origin_input_ids=turn2_tokens,
                extra_key=None,
                cache_salt=None,
                last_node=None,
            )
            sender_on(pmgr)
            pmgr.request_status[room2] = KVPoll.Bootstrapping
            decode_req2 = SimpleNamespace(req=req2)
            d_queue.queue.append(decode_req2)
            offerer.offer_host_prefix(decode_req2)
            self.assertIn(room2, offerer.offers)
            dmgr.register_decode_request(
                room2,
                [1, 2, 3],
                [1, 2],
                None,
                [{"rank_ip": "10.0.0.1", "rank_port": 4200}],
            )

            puller.advance([req2], waiting_reqs=[])
            puller.advance([req2], waiting_reqs=[])
            puller.advance([req2], waiting_reqs=[])
            self.assertTrue(puller.insert_recalled_host_prefix(req2))
            self.assertEqual(puller.stats["pulled_tokens"], 3 * HYBRID_PAGE)
            self.assertEqual(puller.stats["pulled_swa_tokens"], 128)

            p_leaf = next(iter(p_cache.tree_core.root_node.children.values()))
            while p_leaf.children:
                p_leaf = next(iter(p_leaf.children.values()))
            p_swa_pages = radix_host_ops.window_host_pages(
                p_cache,
                p_swa,
                p_leaf,
                tokens=len(p0) + len(g0),
                window_tokens=128,
            )
            self.assertEqual(
                [p_swa.transfer.blocks[pid] for pid in p_swa_pages],
                [f"GEN_SWA_{i}".encode().ljust(32, b"W") for i in range(2)],
            )

    def test_backup_decode_leaf_path_recovers_missing_trailing_swa_when_full_backed(
        self,
    ):
        """If a hybrid node's FULL KV is already backed up on host while its
        trailing SWA window lacks a host copy, backup_decode_leaf_path must
        still flush the trailing SWA node rather than returning early."""
        d_cache, d_alloc, d_r2t = _real_hybrid_cache(write_back=True, window=128)
        d_full = d_cache.cache_controller.host
        d_swa = d_cache.cache_controller.extra_pools[PoolName.SWA]
        p0 = list(range(100, 100 + 2 * HYBRID_PAGE))
        g0 = list(range(500, 500 + HYBRID_PAGE))
        d_req = _make_finished_decode_req(
            cache=d_cache,
            allocator=d_alloc,
            req_to_token_pool=d_r2t,
            prompt_tokens=p0,
            generated_tokens=g0,
        )
        d_cache.checkpoint(d_req, up_to=len(p0) + len(g0))
        d_leaf = d_cache.tree_core.node_by_id(d_req.last_node)
        d_lock = SimpleNamespace(
            agree=lambda vals: True,
            max=lambda v: v,
        )
        # First backup populates both FULL and SWA on host.
        radix_host_ops.backup_decode_leaf_path(
            d_cache,
            d_lock,
            d_req.last_node,
            host_pool=d_full,
            swa_host_pool=d_swa,
            window_tokens=128,
        )
        d_cache.writing_check()
        self.assertIsNotNone(d_leaf.component_data[FULL].host_value)
        self.assertIsNotNone(d_leaf.component_data[ComponentType.SWA].host_value)

        # Simulate host-side SWA eviction while keeping FULL backed up and SWA on device.
        old_swa_host = d_leaf.component_data[ComponentType.SWA].host_value
        d_swa.free(old_swa_host)
        d_leaf.component_data[ComponentType.SWA].host_value = None

        staged = radix_host_ops.backup_decode_leaf_path(
            d_cache,
            d_lock,
            d_req.last_node,
            host_pool=d_full,
            swa_host_pool=d_swa,
            window_tokens=128,
        )
        d_cache.writing_check()
        self.assertIsNotNone(d_leaf.component_data[ComponentType.SWA].host_value)
        # SWA-only recovery reports the SWA tokens it staged, not 0.
        self.assertGreater(staged, 0)

    def test_backup_decode_leaf_path_shared_domain_uses_pages_per_logical_and_evicts_swa(
        self,
    ):
        """Shared unified host pools expose ``_pages_per_logical`` (M_swa > 1);
        backup_decode_leaf_path must scale SWA demand by M_swa and evict SWA as
        well as FULL when draining a shared allocation domain."""
        domain = SimpleNamespace(free_pages=[0, 1])  # 2 free physical pages = 8 tokens
        page_size = 4
        evictions = []

        node = SimpleNamespace(
            id=1,
            parent=None,
            key=list(range(8)),  # 2 logical pages = 8 tokens
            write_through_pending_id=None,
            component_data={
                FULL: SimpleNamespace(
                    value=list(range(8)),  # 2 physical pages
                    host_value=None,
                ),
                ComponentType.SWA: SimpleNamespace(
                    value=list(range(8)),  # 2 logical * M_swa=3 -> 6 physical pages
                    host_value=None,
                ),
            },
        )
        root = SimpleNamespace(id=0, parent=None)
        node.parent = root

        def fake_evict(target, comp):
            evictions.append((comp, target))
            if comp == ComponentType.SWA:
                # Reclaim 6 physical pages from SWA side.
                domain.free_pages.extend(range(10, 16))

        fake_cache = SimpleNamespace(
            tree_core=SimpleNamespace(root_node=root, node_by_id=lambda nid: node),
            components={FULL: object(), ComponentType.SWA: object()},
            cache_controller=SimpleNamespace(
                write_queue=[], start_writing=lambda: None
            ),
            evict_host=fake_evict,
            _execute_and_commit_kv_backup=lambda action, write_back=False: 8,
        )
        full_pool = SimpleNamespace(
            shared_allocation_domain=domain,
            page_size=page_size,
            available_size=lambda: len(domain.free_pages) * page_size,
        )
        swa_pool = SimpleNamespace(
            shared_allocation_domain=domain,
            page_size=page_size,
            _pages_per_logical=3,
            available_size=lambda: (len(domain.free_pages) // 3) * page_size,
        )
        lockstep = SimpleNamespace(agree=lambda vals: True, max=lambda v: v)

        # Needed physical tokens: full(8) + swa(8) * 3 = 32 (8 physical pages).
        # Available: 2 free physical pages (8 tokens) -> shortfall 24 tokens (6 pages).
        written = radix_host_ops.backup_decode_leaf_path(
            fake_cache,
            lockstep,
            1,
            host_pool=full_pool,
            swa_host_pool=swa_pool,
            window_tokens=4,  # smaller than node len=8 to verify oversized-node sizing
        )
        self.assertEqual(written, 8)
        # SWA shortfall = ceil(6 / 3) * 4 = 8 logical tokens (3x multiplier honored).
        self.assertEqual(evictions, [(ComponentType.SWA, 8)])

        # Also test non-multiple shortfall and fallback from SWA to FULL eviction:
        # 5 free pages (20 tokens), need 32 -> shortfall 12 tokens = 3 pages ->
        # ceil(3/3)*4 = 4 logical SWA tokens; fake SWA reclaims 0, FULL reclaims 3 pages.
        domain.free_pages[:] = [0, 1, 2, 3, 4]
        evictions.clear()

        def fake_evict_full_fallback(target, comp):
            evictions.append((comp, target))
            if comp == FULL:
                domain.free_pages.extend(range(20, 23))

        fake_cache.evict_host = fake_evict_full_fallback
        radix_host_ops.backup_decode_leaf_path(
            fake_cache,
            lockstep,
            1,
            host_pool=full_pool,
            swa_host_pool=swa_pool,
            window_tokens=4,
        )
        self.assertEqual(
            evictions,
            [(ComponentType.SWA, 4), (FULL, 12)],
        )


if __name__ == "__main__":
    unittest.main()
