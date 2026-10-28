"""Unit tests for victim push (sglang_tpu/pd_victim_push.py).

The prefill offers the host-backed session with the oldest path end to the
decode that served it, registers its pinned host pages for a native host pull,
and makes its own copy the next eviction victim once the transfer landed.
The decode inserts the session as a host-only node. Real
full-attention unified trees with a fake raiden controller and host transfer;
the raiden manager is a fake.
"""

import heapq
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_host_prefix_recall
from sglang_tpu import pd_victim_push as victim_push
from sglang_tpu.disaggregation.raiden.serde import VictimAck, VictimDone, VictimPush
from sglang_tpu.raiden_transfer import NOOP_EVENT
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FULL = ComponentType.FULL


class _FakeFuture:
    def __init__(self):
        self.ready = True

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.ready = True


class _FakeHostTransfer:
    """Native transfer boundary; tests explicitly drive asynchronous completion."""

    def __init__(self):
        self.results: dict[str, str | None] = {}
        self.registrations: dict[str, tuple[int, list[int], str]] = {}
        self.reads: dict[str, dict] = {}
        self.registration_error: Exception | None = None
        self.read_error: Exception | None = None
        self.registration_result = True

    def host_control_address(self, local_ip: str) -> str:
        return "10.0.0.5:9000"

    def register_read(
        self, req_id: str, uuid: int, block_ids: list[int], *, source: str = "device"
    ) -> bool:
        if self.registration_error is not None:
            raise self.registration_error
        self.registrations[req_id] = (uuid, list(block_ids), source)
        self.results[req_id] = None
        return self.registration_result

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint,
        remote_block_ids: list[int],
        local_block_ids: list[int] | None = None,
        *,
        local_host_block_ids: list[int] | None = None,
        destination: str = "device",
    ) -> None:
        if self.read_error is not None:
            raise self.read_error
        self.reads[req_id] = dict(
            uuid=uuid,
            remote_endpoint=remote_endpoint,
            remote_block_ids=list(remote_block_ids),
            local_block_ids=local_block_ids,
            local_host_block_ids=list(local_host_block_ids or []),
            destination=destination,
        )
        self.results[req_id] = None

    def poll_transfer(self, req_id: str) -> str | None:
        return self.results.get(req_id)

    def forget_transfer(self, req_id: str) -> None:
        assert self.results.pop(req_id) is not None

    def complete(self, status: str) -> None:
        for req_id in self.results:
            self.results[req_id] = status


class _FakeHostPool:
    """Page size 1: token indices are page ids."""

    shared_allocation_domain = None

    def __init__(self, capacity: int, page_size=1):
        self.page_size = page_size
        self.capacity = capacity
        self.size = capacity
        self.used = 0
        self.transfer = _FakeHostTransfer()

    def _to_page_ids(self, indices):
        return [int(i) // self.page_size for i in indices[:: self.page_size]]

    def available_size(self) -> int:
        return self.capacity - self.used

    def alloc(self, num_tokens: int):
        if num_tokens > self.available_size():
            return None
        indices = torch.arange(self.used, self.used + num_tokens, dtype=torch.int64)
        self.used += num_tokens
        return indices

    def free(self, indices) -> None:
        self.used -= len(indices)

    def clear(self) -> None:
        self.used = 0


class _FakeRaidenController(TpuDMAHybridCacheController):
    """The controller surface the cache's write-back paths touch."""

    def __init__(self, *, host_capacity: int):
        self.write_policy = "write_back"
        self.host = _FakeHostPool(host_capacity)
        self.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(host_pool=self.host, host_evict_fn=None),
            available_size=self.host.available_size,
            clear=self.host.clear,
        )
        self.write_queue: list = []
        self.ack_write_queue: list = []
        self.ack_load_queue: list = []
        self.extra_host_mem_release_queues: dict = {}

    def write(
        self, device_indices, priority=None, node_id=-1, extra_pools=None, flush=True
    ):
        host_indices = self.host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append((host_indices, device_indices, node_id))
        if flush:
            self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        node_ids = [node_id for _, _, node_id in self.write_queue]
        num_tokens = sum(len(d) for _, d, _ in self.write_queue)
        self.write_queue.clear()
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=_FakeFuture(),
                node_ids=node_ids,
                num_tokens=num_tokens,
            )
        )

    def reset(self) -> None:
        self.write_queue.clear()
        self.ack_write_queue.clear()


def full_cache(*, kv_size=64, host_capacity=64):
    """A real FULL-only write-back tree over CPU pools, page size 1."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=kv_size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size,
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
            page_size=1,
            tree_components=(FULL,),
            tree_core_backend="python",
        )
    )
    controller = _FakeRaidenController(host_capacity=host_capacity)
    cache.cache_controller = controller
    cache.components[FULL]._full_kv_pool_host = controller.host
    cache.host_pool_group = SimpleNamespace(
        free=lambda indices, pool=None: controller.host.free(indices)
    )
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.is_write_back = True
    cache.is_write_back = True
    cache._write_back_copy_ahead = None
    return cache, allocator


def hybrid_cache(*, is_eagle=False, window=128):
    """Real paged FULL+SWA tree; mock only the device allocator/transport."""
    allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
    allocator.device = torch.device("cpu")
    allocator.page_size = 64
    allocator._swa_req_ring = False
    allocator.translate_loc_from_full_to_swa.side_effect = lambda indices: indices
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=MagicMock(spec=ReqToTokenPool),
            token_to_kv_pool_allocator=allocator,
            page_size=64,
            sliding_window_size=window,
            is_eagle=is_eagle,
            tree_components=(FULL, ComponentType.SWA),
            tree_core_backend="python",
        )
    )
    controller = _FakeRaidenController(host_capacity=4096)
    controller.host.page_size = 64
    swa = _FakeHostPool(1024, page_size=64)
    cache.cache_controller = controller
    # No asynchronous host users in this fixture: release duplicate pages now.
    controller.append_host_mem_release = lambda *, extra_pools: [
        swa.free(xfer.host_indices) for xfer in extra_pools
    ]
    cache.components[FULL]._full_kv_pool_host = controller.host
    cache.components[ComponentType.SWA]._swa_kv_pool_host = swa
    pools = {PoolName.KV: controller.host, PoolName.SWA: swa}
    cache.host_pool_group = SimpleNamespace(
        get_pool=pools.get,
        free=lambda indices, pool=PoolName.KV: pools[pool].free(indices),
    )
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = True
    cache.tree_core.is_write_back = cache.is_write_back = True
    return cache, allocator


def insert_tokens(cache, allocator, tokens):
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)), value=allocator.alloc(len(tokens))
        )
    )


def device_hit(cache, tokens) -> int:
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    return len(match.device_indices)


def root_node(cache, first_token):
    return next(
        c
        for c in cache.tree_core.root_node.children.values()
        if c.key.token_ids[0] == first_token
    )


def host_hit(cache, tokens) -> int:
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    return match.host_hit_length


class TestWire(CustomTestCase):
    def test_push_ack_done_round_trip(self):
        for msg in (
            VictimPush(
                prefill="ab12",
                seq=3,
                ack_endpoint="tcp://10.0.0.9:7000",
                tokens=[1, 2],
                blocks=[4, 5],
                swa_blocks=[6],
                uuid=9,
                address="10.0.0.1:9000",
                swa_address="10.0.0.1:9001",
                source="host",
            ),
            VictimAck(seq=3, stored=True),
            VictimAck(seq=4, stored=False),
            VictimDone(prefill="ab12", seq=3, ok=True),
        ):
            self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)


class TestPrefillPusher(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator = full_cache(host_capacity=32)
        self.host = self.cache.cache_controller.host
        self.mgr = MagicMock()
        self.mgr.local_ip = "10.0.0.5"
        self.acks = {}
        self.mgr.take_victim_ack.side_effect = lambda seq: self.acks.pop(seq, None)
        self.peers = {}
        self.mgr.decode_peer.side_effect = lambda room: self.peers.get(room)
        self.pusher = victim_push.PrefillPusher(
            SimpleNamespace(
                _mgr=self.mgr,
                _tree_cache=self.cache,
                _allocator=self.allocator,
                _page_size=1,
                _groups=[],
                _window_pages=None,
            )
        )
        self.a, self.b = list(range(100, 108)), list(range(200, 208))
        for room, tokens, decode in (
            (1, self.a, ("10.0.0.1", 5001)),
            (2, self.b, ("10.0.0.2", 5002)),
        ):
            insert_tokens(self.cache, self.allocator, tokens)
            self.peers[room] = decode
            self.pusher.record(
                SimpleNamespace(bootstrap_room=room, origin_input_ids=tokens)
            )
        self.cache.evict(EvictParams(num_tokens=16))

    def offer_once(self):
        self.pusher.step()
        self.mgr.send_victim_push.assert_called_once()
        return self.mgr.send_victim_push.call_args.kwargs

    def test_stored_victim_path_evicts_in_the_same_order_on_every_rank(self):
        """A stored victim's path must take unique access stamps, each older
        than every live node. Stamping the whole path with one value tied the
        eviction heap, whose ties pop in set-iteration (per-rank memory) order,
        so TP ranks evicted different nodes after the first stored push."""
        c = list(range(300, 308))
        insert_tokens(self.cache, self.allocator, c[:4])  # a two-node path
        insert_tokens(self.cache, self.allocator, c)
        leaf = root_node(self.cache, c[0])
        while leaf.children:
            leaf = next(iter(leaf.children.values()))
        self.pusher._make_next_victim(leaf.id)
        tree_core = self.cache.tree_core
        victims, node = [], leaf
        while node is not tree_core.root_node:
            victims.append(node)
            node = node.parent
        self.assertGreaterEqual(len(victims), 2)
        live, stack = [], list(tree_core.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node not in victims:
                live.append(node)
        self.assertTrue(live)
        stamps = [node.last_access_time for node in victims]
        self.assertEqual(len(set(stamps)), len(stamps), "victim nodes tie")
        self.assertLess(max(stamps), min(node.last_access_time for node in live))
        # The eviction heap pops the same nodes whichever order a rank's leaf
        # set yields them in.
        entries = [(node.last_access_time, node) for node in victims + live]
        orders = []
        for candidates in (entries, entries[::-1]):
            heap = list(candidates)
            heapq.heapify(heap)
            orders.append([heapq.heappop(heap)[1].id for _ in range(len(heap))])
        self.assertEqual(orders[0], orders[1])

    def test_host_source_stays_pinned_until_native_sent_then_is_next_victim(self):
        sent = self.offer_once()
        self.assertEqual(sent["decode"], ("10.0.0.1", 5001))
        self.assertEqual(sent["tokens"], self.a)
        self.assertEqual(sent["address"], "10.0.0.5:9000")
        registrations = list(self.host.transfer.registrations.values())
        self.assertEqual(registrations, [(sent["uuid"], sent["blocks"], "host")])
        self.acks[sent["seq"]] = VictimAck(seq=sent["seq"], stored=True)
        self.pusher.step()
        self.cache.evict_host(8, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.mgr.send_victim_done.assert_not_called()
        self.host.transfer.complete("sent")
        self.pusher.step()
        self.mgr.send_victim_done.assert_called_once_with(
            decode=("10.0.0.1", 5001), seq=sent["seq"], ok=True
        )
        self.assertEqual(self.pusher.stats["stored"], 1)
        self.cache.evict_host(8, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_declined_offer_retains_registration_pin_until_native_failed(self):
        sent = self.offer_once()
        self.acks[sent["seq"]] = VictimAck(seq=sent["seq"], stored=False)
        self.pusher.step()
        self.assertIn(sent["seq"], self.pusher._pushes)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.step()
        self.assertNotIn(sent["seq"], self.pusher._pushes)
        self.assertTrue(
            all(
                not call.kwargs["ok"]
                for call in self.mgr.send_victim_done.call_args_list
            )
        )
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_native_failed_reports_failure_and_unlocks_host_source(self):
        sent = self.offer_once()
        self.acks[sent["seq"]] = VictimAck(seq=sent["seq"], stored=True)
        self.host.transfer.complete("failed")
        self.pusher.step()
        self.mgr.send_victim_done.assert_called_once_with(
            decode=("10.0.0.1", 5001), seq=sent["seq"], ok=False
        )
        self.assertEqual(self.pusher.stats["declined"], 1)
        self.assertEqual(host_hit(self.cache, self.a), 8)

    def test_ack_timeout_does_not_release_pages_still_registered_with_native(self):
        sent = self.offer_once()
        with patch.object(victim_push, "_ACK_TIMEOUT_S", -1):
            self.pusher.step()
        self.assertIn(sent["seq"], self.pusher._pushes)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.step()
        self.assertNotIn(sent["seq"], self.pusher._pushes)

    def test_cache_flush_refuses_to_release_a_registered_source(self):
        self.offer_once()
        with patch.object(victim_push, "_DRAIN_TIMEOUT_S", 0):
            with self.assertRaises(RuntimeError):
                self.pusher.release_all()
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.release_all()
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_false_registration_result_is_polled_to_terminal(self):
        """Native false can mean an already-ACKed registration, not rejection."""
        self.host.transfer.registration_result = False
        sent = self.offer_once()
        self.acks[sent["seq"]] = VictimAck(seq=sent["seq"], stored=True)
        self.host.transfer.complete("sent")
        self.pusher.step()
        self.assertEqual(self.pusher.stats["stored"], 1)

    def test_device_only_session_is_not_offered(self):
        c = list(range(300, 308))
        insert_tokens(self.cache, self.allocator, c)
        self.peers[3] = ("10.0.0.3", 5003)
        self.pusher.record(SimpleNamespace(bootstrap_room=3, origin_input_ids=c))
        for _ in range(3):
            self.pusher.step()
            self.host.transfer.complete("failed")
            seq = self.mgr.send_victim_push.call_args.kwargs["seq"]
            self.acks[seq] = VictimAck(seq=seq, stored=False)
        offered = [c.kwargs["tokens"] for c in self.mgr.send_victim_push.call_args_list]
        self.assertNotIn(c, offered)

    def test_no_offer_without_host_pressure(self):
        with patch.object(victim_push, "_PUSH_FULL_FREE_TOKENS", 0):
            for _ in range(3):
                self.pusher.step()
        self.mgr.send_victim_push.assert_not_called()
        self.assertFalse(self.host.transfer.registrations)


class TestDecodeReceiver(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator = full_cache(host_capacity=64)
        self.host = self.cache.cache_controller.host
        self.mgr = MagicMock()
        self.pushes, self.dones = {}, {}
        self.mgr.victim_pushes.side_effect = lambda: dict(self.pushes)
        self.mgr.take_victim_push.side_effect = lambda key: self.pushes.pop(key, None)
        self.mgr.victim_dones.side_effect = lambda: dict(self.dones)
        self.mgr.take_victim_done.side_effect = lambda key: self.dones.pop(key, None)
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            attn_tp_cpu_group=None,
            attn_cp_cpu_group=None,
        )
        self.receiver = victim_push.DecodeReceiver(
            queue=SimpleNamespace(scheduler=scheduler, tree_cache=self.cache),
            mgr=self.mgr,
        )
        self.c, self.key = list(range(300, 308)), ("ab12", 1)
        self.pushes[self.key] = VictimPush(
            prefill="ab12",
            seq=1,
            ack_endpoint="tcp://p",
            tokens=self.c,
            blocks=list(range(8)),
            uuid=9,
            address="producer:9000",
            source="host",
        )

    def round(self):
        for _ in range(victim_push._CADENCE):
            self.receiver.step()

    def granted(self):
        self.round()
        self.assertTrue(self.receiver.holds_pages())
        ack = self.mgr.send_victim_ack.call_args.kwargs["ack"]
        self.assertTrue(ack.stored)
        read = next(iter(self.host.transfer.reads.values()))
        self.assertEqual(read["remote_endpoint"], "producer:9000")
        self.assertEqual(read["remote_block_ids"], list(range(8)))
        self.assertIsNone(read["local_block_ids"])
        self.assertEqual(read["destination"], "host")
        self.assertEqual(len(read["local_host_block_ids"]), len(self.c))
        return ack

    def done(self, ok=True):
        self.dones[self.key] = VictimDone(prefill="ab12", seq=1, ok=ok)

    def test_host_receive_waits_for_native_completion_even_after_done(self):
        self.granted()
        self.done()
        self.round()
        self.assertTrue(self.receiver.holds_pages())
        self.assertEqual(host_hit(self.cache, self.c), 0)
        self.host.transfer.complete("received")
        self.round()
        self.assertFalse(self.receiver.holds_pages())
        self.assertEqual(device_hit(self.cache, self.c), 0)
        self.assertEqual(host_hit(self.cache, self.c), 8)
        self.assertEqual(self.receiver.stats["landed_tokens"], 8)

    def test_native_receive_waits_for_producer_done_before_publication(self):
        self.granted()
        self.host.transfer.complete("received")
        self.round()
        self.assertEqual(host_hit(self.cache, self.c), 0)
        self.assertTrue(self.receiver.holds_pages())
        self.done()
        self.round()
        self.assertEqual(host_hit(self.cache, self.c), 8)

    def test_failed_transfer_frees_only_after_local_native_has_drained(self):
        self.granted()
        self.done(ok=False)
        self.round()
        self.assertTrue(self.receiver.holds_pages())
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.round()
        self.assertFalse(self.receiver.holds_pages())
        self.assertEqual(self.host.used, 0)
        self.assertEqual(host_hit(self.cache, self.c), 0)

    def test_landing_timeout_does_not_reuse_pages_a_remote_writer_may_hold(self):
        self.granted()
        with patch.object(victim_push, "_LANDING_TIMEOUT_S", -1):
            self.round()
        self.assertTrue(self.receiver.holds_pages())
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.round()
        self.assertFalse(self.receiver.holds_pages())
        self.assertEqual(self.host.used, 0)

    def test_full_host_pool_refuses(self):
        self.host.capacity = 4
        self.round()
        ack = self.mgr.send_victim_ack.call_args.kwargs["ack"]
        self.assertFalse(ack.stored)
        self.assertFalse(self.receiver.holds_pages())
        self.assertFalse(self.host.transfer.reads)

    def test_device_source_cannot_be_mistaken_for_host_pool_ids(self):
        self.pushes[self.key].source = "device"
        self.round()
        self.assertFalse(self.mgr.send_victim_ack.call_args.kwargs["ack"].stored)
        self.assertFalse(self.host.transfer.reads)

    def test_cache_flush_refuses_to_reuse_a_pending_host_landing(self):
        self.granted()
        with patch.object(victim_push, "_DRAIN_TIMEOUT_S", 0):
            with self.assertRaises(RuntimeError):
                self.receiver.drain()
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.receiver.drain()
        self.assertEqual(self.host.used, 0)


class TestHybridVictims(CustomTestCase):
    def make_receiver(self, *, is_eagle=False, window=128):
        cache, allocator = hybrid_cache(is_eagle=is_eagle, window=window)
        mgr = MagicMock()
        mgr.local_ip = "10.0.0.5"
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            attn_tp_cpu_group=None,
            attn_cp_cpu_group=None,
        )
        receiver = victim_push.DecodeReceiver(
            queue=SimpleNamespace(scheduler=scheduler, tree_cache=cache), mgr=mgr
        )
        return receiver, mgr

    def push(self, count, seq=1, *, is_eagle=False, window=128):
        return VictimPush(
            prefill="hybrid",
            seq=seq,
            ack_endpoint="tcp://p",
            tokens=list(range(count + int(is_eagle))),
            blocks=list(range(count // 64)),
            swa_blocks=list(range(min(count, -(-window // 64) * 64) // 64)),
            is_eagle=is_eagle,
            uuid=seq + 9,
            address="full:9000",
            swa_address="swa:9001",
            source="host",
        )

    def complete(self, receiver, *, full="received", swa="received"):
        receiver._host_pool.transfer.complete(full)
        receiver._swa_host_pool.transfer.complete(swa)

    def seeded_pusher(self):
        receiver, _ = self.make_receiver()
        offer = self.push(256)
        key = (offer.prefill, offer.seq)
        self.assertTrue(receiver._grant(key, offer))
        self.complete(receiver)
        receiver._insert(receiver._landings.pop(key))
        mgr = MagicMock()
        mgr.local_ip = "10.0.0.5"
        acks = {}
        mgr.decode_peer.return_value = ("10.0.0.1", 5001)
        mgr.take_victim_ack.side_effect = lambda seq: acks.pop(seq, None)
        pusher = victim_push.PrefillPusher(
            SimpleNamespace(
                _mgr=mgr, _tree_cache=receiver._cache, _page_size=64, _groups=[]
            )
        )
        pusher.record(SimpleNamespace(bootstrap_room=1, origin_input_ids=offer.tokens))
        return receiver, pusher, mgr, acks

    def test_window_roundtrip_splits_and_duplicate_pages_are_freed(self):
        """A window inside a FULL node and a duplicate FULL still preserve SWA."""
        for count, window, eagle in (
            (64, 128, False),
            (256, 128, False),
            (256, 65, True),
        ):
            with self.subTest(count=count, window=window, eagle=eagle):
                receiver, mgr = self.make_receiver(is_eagle=eagle, window=window)
                core = receiver._cache.tree_core
                for seq in (1, 2, 3):
                    push = self.push(count, seq, is_eagle=eagle, window=window)
                    key = (push.prefill, seq)
                    self.assertTrue(receiver._grant(key, push))
                    self.assertTrue(mgr.send_victim_ack.call_args.kwargs["ack"].stored)
                    read = next(
                        reversed(receiver._swa_host_pool.transfer.reads.values())
                    )
                    landing = receiver._landings.pop(key)
                    self.complete(receiver)
                    receiver._insert(landing)
                    self.assertEqual(receiver._host_pool.used, count)
                    self.assertEqual(
                        receiver._swa_host_pool.used, len(push.swa_blocks) * 64
                    )
                    leaf = core.node_by_id(
                        core.insert_host(
                            core.root_node.id,
                            RadixKey(array("q", push.tokens), is_bigram=eagle),
                            landing.indices,
                            [""] * len(push.blocks),
                        ).inserted_host_node
                    )
                    actual_swa = victim_push._host_pages(
                        core.root_node,
                        receiver._swa_host_pool,
                        leaf,
                        component=ComponentType.SWA,
                        tail_tokens=min(count, receiver._window_tokens),
                    )
                    if seq < 3:
                        self.assertEqual(actual_swa, read["local_host_block_ids"])
                        retained_swa = actual_swa
                    else:
                        self.assertEqual(actual_swa, retained_swa)
                    if seq == 1:
                        receiver._cache.evict_host(4096, ComponentType.SWA)
                        self.assertEqual(receiver._swa_host_pool.used, 0)
                receiver._cache.evict_host(4096, FULL)
                self.assertEqual(receiver._host_pool.used, 0)
                self.assertEqual(receiver._swa_host_pool.used, 0)

    def test_prefill_exports_host_window_and_locks_both_until_terminal(self):
        receiver, pusher, mgr, acks = self.seeded_pusher()
        pusher.step()
        offered = mgr.send_victim_push.call_args.kwargs
        self.assertEqual(len(offered["blocks"]), 4)
        self.assertEqual(len(offered["swa_blocks"]), 2)
        self.assertEqual(len(receiver._host_pool.transfer.registrations), 1)
        self.assertEqual(len(receiver._swa_host_pool.transfer.registrations), 1)
        self.assertEqual(receiver._cache.evict_host(4096, FULL), 0)
        self.assertEqual(receiver._cache.evict_host(4096, ComponentType.SWA), 0)
        self.complete(receiver, full="failed", swa="failed")
        pusher.release_all()
        receiver._cache.evict_host(4096, ComponentType.SWA)
        self.assertEqual(receiver._swa_host_pool.used, 0)
        pusher._pushed_len.clear()
        mgr.send_victim_push.reset_mock()
        pusher.step()
        mgr.send_victim_push.assert_not_called()

    def test_partial_allocation_and_failed_transfer_release_both_pools(self):
        receiver, _ = self.make_receiver()
        push = self.push(256)
        key = (push.prefill, push.seq)
        receiver._swa_host_pool.capacity = 64
        self.assertFalse(receiver._grant(key, push))
        self.assertEqual(receiver._host_pool.used, 0)
        receiver._swa_host_pool.capacity = 1024
        self.assertTrue(receiver._grant(key, push))
        self.complete(receiver, swa="failed")
        receiver._mgr.victim_dones.return_value = {
            key: VictimDone(prefill=push.prefill, seq=1, ok=False)
        }
        receiver._finish_agreed()
        self.assertEqual(receiver._host_pool.used, 0)
        self.assertEqual(receiver._swa_host_pool.used, 0)

    def test_swa_start_failure_retains_full_landing_until_native_drains(self):
        receiver, mgr = self.make_receiver()
        push = self.push(256)
        key = (push.prefill, push.seq)
        receiver._swa_host_pool.transfer.read_error = RuntimeError("cannot start SWA")
        self.assertTrue(receiver._grant(key, push))
        self.assertFalse(mgr.send_victim_ack.call_args.kwargs["ack"].stored)
        receiver._finish_agreed()
        self.assertTrue(receiver.holds_pages())
        self.assertEqual(receiver._host_pool.used, 256)
        receiver._host_pool.transfer.complete("failed")
        receiver._finish_agreed()
        self.assertFalse(receiver.holds_pages())
        self.assertEqual(receiver._host_pool.used, 0)
        self.assertEqual(receiver._swa_host_pool.used, 0)

    def test_timeout_retains_both_until_native_received_and_done(self):
        receiver, _ = self.make_receiver()
        push = self.push(256)
        key = (push.prefill, push.seq)
        self.assertTrue(receiver._grant(key, push))
        receiver._mgr.victim_dones.return_value = {}
        with patch.object(victim_push, "_LANDING_TIMEOUT_S", -1):
            receiver._finish_agreed()
            receiver._finish_agreed()
        self.assertEqual(receiver._host_pool.used, 256)
        self.assertEqual(receiver._swa_host_pool.used, 128)
        self.assertEqual(receiver.stats["timed_out"], 1)
        self.complete(receiver)
        receiver._mgr.victim_dones.return_value = {
            key: VictimDone(prefill=push.prefill, seq=1, ok=True)
        }
        receiver._finish_agreed()
        self.assertFalse(receiver.holds_pages())
        self.assertEqual(receiver.stats["landed"], 1)

    def test_missing_or_wrong_window_is_declined(self):
        receiver, _ = self.make_receiver()
        push = self.push(256)
        for pages in ([], [1], [1, 2, 3]):
            push.swa_blocks = pages
            self.assertFalse(receiver._grant((push.prefill, push.seq), push))
        self.assertEqual(receiver._host_pool.used, 0)

    def test_full_received_and_done_cannot_publish_without_swa_received(self):
        receiver, _ = self.make_receiver()
        push = self.push(256)
        key = (push.prefill, push.seq)
        self.assertTrue(receiver._grant(key, push))
        receiver._mgr.victim_dones.return_value = {
            key: VictimDone(prefill=push.prefill, seq=push.seq, ok=True)
        }
        receiver._host_pool.transfer.complete("received")
        receiver._finish_agreed()
        self.assertTrue(receiver.holds_pages())
        self.assertEqual(host_hit(receiver._cache, push.tokens), 0)
        receiver._swa_host_pool.transfer.complete("received")
        receiver._finish_agreed()
        self.assertFalse(receiver.holds_pages())
        self.assertEqual(host_hit(receiver._cache, push.tokens), 256)

    def test_success_is_published_only_after_both_native_sends(self):
        receiver, pusher, mgr, acks = self.seeded_pusher()
        pusher.step()
        seq = mgr.send_victim_push.call_args.kwargs["seq"]
        acks[seq] = VictimAck(seq=seq, stored=True)
        receiver._host_pool.transfer.complete("sent")
        pusher.step()
        self.assertEqual(receiver._cache.evict_host(4096, FULL), 0)
        self.assertEqual(receiver._cache.evict_host(4096, ComponentType.SWA), 0)
        mgr.send_victim_done.assert_not_called()
        receiver._swa_host_pool.transfer.complete("sent")
        pusher.step()
        mgr.send_victim_done.assert_called_once_with(
            decode=("10.0.0.1", 5001), seq=seq, ok=True
        )
        self.assertEqual(pusher.stats["stored"], 1)

    def test_partial_source_registration_stays_pinned_until_full_failed(self):
        receiver, pusher, mgr, _ = self.seeded_pusher()
        receiver._swa_host_pool.transfer.registration_error = RuntimeError(
            "cannot register SWA"
        )
        pusher.step()
        mgr.send_victim_push.assert_not_called()
        self.assertEqual(receiver._cache.evict_host(4096, FULL), 0)
        self.assertEqual(receiver._cache.evict_host(4096, ComponentType.SWA), 0)
        receiver._host_pool.transfer.complete("failed")
        pusher.step()
        receiver._cache.evict_host(4096, FULL)
        self.assertEqual(receiver._host_pool.used, 0)
        self.assertEqual(receiver._swa_host_pool.used, 0)

    def test_swa_native_failure_waits_for_full_terminal_before_source_release(self):
        receiver, pusher, mgr, acks = self.seeded_pusher()
        pusher.step()
        seq = mgr.send_victim_push.call_args.kwargs["seq"]
        acks[seq] = VictimAck(seq=seq, stored=True)
        receiver._swa_host_pool.transfer.complete("failed")
        pusher.step()
        self.assertEqual(receiver._cache.evict_host(4096, FULL), 0)
        mgr.send_victim_done.assert_not_called()
        receiver._host_pool.transfer.complete("sent")
        pusher.step()
        self.assertFalse(mgr.send_victim_done.call_args.kwargs["ok"])
        self.assertEqual(pusher.stats["stored"], 0)


if __name__ == "__main__":
    unittest.main()


class TestDecodeFlushWaitsForDeferredReleases(CustomTestCase):
    """A decode cache reset must wait for pages the transfer queue still owns."""

    def _scheduler(self, *, deferred: bool, mode=DisaggregationMode.DECODE):
        queue = SimpleNamespace(has_pending_deferred_releases=lambda: deferred)
        return SimpleNamespace(
            disaggregation_mode=mode, disagg_decode_transfer_queue=queue
        )

    def test_pinned_offers_keep_the_decode_non_idle_for_a_flush(self):
        upstream = MagicMock(return_value=True)
        scheduler = self._scheduler(deferred=True)
        self.assertFalse(victim_push._is_fully_idle(upstream, scheduler))
        # Health checks only ask about running work, as upstream does.
        self.assertTrue(
            victim_push._is_fully_idle(upstream, scheduler, for_health_check=True)
        )
        upstream.assert_called_with(
            scheduler, for_health_check=True, ignore_waiting=False
        )

    def test_released_pages_and_other_roles_defer_to_upstream(self):
        upstream = MagicMock(return_value=True)
        self.assertTrue(
            victim_push._is_fully_idle(upstream, self._scheduler(deferred=False))
        )
        self.assertTrue(
            victim_push._is_fully_idle(
                upstream,
                self._scheduler(deferred=True, mode=DisaggregationMode.PREFILL),
            )
        )
        upstream.return_value = False
        self.assertFalse(
            victim_push._is_fully_idle(upstream, self._scheduler(deferred=False))
        )

    def test_scheduler_idle_check_is_wrapped(self):
        self.assertIn(
            (Scheduler, "is_fully_idle", victim_push._is_fully_idle),
            victim_push._patch_targets(),
        )


class TestRecallLandingReclaimsHostDuplicates(CustomTestCase):
    """A recall lands in the prefill's RAM; under write-through that RAM is
    full of copies of pages the device still holds, which host eviction
    never takes. The puller reclaims those copies for the landing."""

    def _write_through_cache(self):
        cache, allocator = full_cache(kv_size=64, host_capacity=16)
        cache.tree_core.is_write_back = cache.is_write_back = False
        tokens = list(range(100, 116))
        insert_tokens(cache, allocator, tokens)
        node = root_node(cache, 100)
        host = cache.cache_controller.host
        cache.tree_core.commit_backup(node.id, host.alloc(len(node.key)), {})
        # The write-through ack registers the duplicate in production.
        cache.tree_core._update_duplicate_tracking(node)
        return cache, host, tokens

    def test_host_only_eviction_frees_nothing_but_the_reclaim_does(self):
        cache, host, tokens = self._write_through_cache()
        self.assertEqual(host.available_size(), 0)
        self.assertEqual(cache.evict_host(8, FULL), 0)
        freed = pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8)
        self.assertGreaterEqual(freed, 8)
        self.assertGreaterEqual(host.available_size(), 8)
        # The device copy is untouched: the prompt still hits in HBM.
        self.assertEqual(device_hit(cache, tokens), 16)

    def test_a_duplicate_whose_children_are_on_device_is_taken(self):
        cache, host, tokens = self._write_through_cache()
        _, allocator = cache, cache.token_to_kv_pool_allocator
        insert_tokens(cache, allocator, tokens + [8] * 8)  # a device child
        self.assertGreaterEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8), 8
        )

    def test_a_duplicate_below_another_duplicate_is_taken_tip_first(self):
        # Write-through host prefixes stay closed under ancestors: the child's
        # copy goes first, the parent's only once no child has a host copy.
        cache, host, tokens = self._write_through_cache()
        _, allocator = cache, cache.token_to_kv_pool_allocator
        host.capacity += 8
        child_tokens = tokens + [8] * 8
        insert_tokens(cache, allocator, child_tokens)
        child = root_node(cache, 100).children[
            next(iter(root_node(cache, 100).children))
        ]
        cache.tree_core.commit_backup(child.id, host.alloc(len(child.key)), {})
        cache.tree_core._update_duplicate_tracking(child)
        self.assertEqual(host.available_size(), 0)
        freed = pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8)
        self.assertEqual(freed, 8)  # the child's copy, not the parent's
        parent = root_node(cache, 100)
        self.assertTrue(parent.backuped)
        self.assertFalse(child.backuped)
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 16), 16
        )  # now the parent is a host tip and goes too

    def test_a_duplicate_with_a_host_only_child_is_left_alone(self):
        # Deleting its device leaf later would orphan the subtree below it.
        cache, host, tokens = self._write_through_cache()
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey
        from sglang_tpu.pd_host_cache import insert_host_prefix

        host.capacity += 8
        insert_host_prefix(
            cache, _RadixKey(_array("q", tokens + [7] * 8)), host.alloc(8), None, 1
        )
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8), 0
        )

    def test_reclaim_is_a_no_op_without_pressure_or_the_walk(self):
        cache, host, _ = self._write_through_cache()
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 0), 0
        )
        self.assertEqual(host.available_size(), 0)
        bare = SimpleNamespace(tree_core=SimpleNamespace())
        self.assertEqual(pd_host_prefix_recall.reclaim_full_host_duplicates(bare, 8), 0)


class TestEvictionWalkSkipsDeletedLeaves(CustomTestCase):
    """A leaf the eviction walk deleted earlier in the same walk may be
    yielded again; the wrapper skips it instead of raising KeyError."""

    def test_missing_leaf_is_skipped_and_present_leaf_is_evicted(self):
        cache, allocator = full_cache(kv_size=32, host_capacity=32)
        insert_tokens(cache, allocator, list(range(100, 108)))
        node = root_node(cache, 100)
        upstream = MagicMock(return_value="evicted")
        self.assertEqual(
            victim_push._evict_device_leaf(upstream, cache, node.id, {}), "evicted"
        )
        upstream.assert_called_once_with(cache, node.id, {})
        upstream.reset_mock()
        self.assertIsNone(victim_push._evict_device_leaf(upstream, cache, 10**9, {}))
        upstream.assert_not_called()

    def test_unbacked_parent_of_a_host_only_child_stays_on_device(self):
        cache, allocator = full_cache(kv_size=32, host_capacity=32)
        tokens = list(range(100, 108))
        insert_tokens(cache, allocator, tokens)
        node = root_node(cache, 100)
        host = cache.cache_controller.host
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey
        from sglang_tpu.pd_host_cache import insert_host_prefix

        insert_host_prefix(
            cache, _RadixKey(_array("q", tokens + [7] * 8)), host.alloc(8), None, 1
        )
        self.assertFalse(node.backuped)
        upstream = MagicMock(return_value="evicted")
        self.assertIsNone(victim_push._evict_device_leaf(upstream, cache, node.id, {}))
        upstream.assert_not_called()

    def test_walk_is_wrapped(self):
        self.assertIn(
            (UnifiedRadixCache, "_evict_device_leaf", victim_push._evict_device_leaf),
            victim_push._patch_targets(),
        )
