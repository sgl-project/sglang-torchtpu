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

from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
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


if __name__ == "__main__":
    unittest.main()
