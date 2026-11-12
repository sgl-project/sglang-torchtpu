"""Test doubles and fixtures for the offload of prefill KV to decode DRAM.

Shared by the unit tests of ``sglang_tpu/disaggregation/decode_dram_offload``,
of the raiden channel (``raiden/dram_offload_channel.py``) and of the host-to-host reads on
the HiCache raiden transfer.
"""

from __future__ import annotations

import threading
from array import array
from collections import deque
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.cache_controller import HiCacheAck
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
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang_tpu.disaggregation.decode_dram_offload import (
    decode_dram_victim_receiver,
    prefill_dram_victim_pusher,
    radix_host_ops,
    startup_validation,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_prefix_puller import (
    PrefillDramPrefixPuller,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVSender
from sglang_tpu.disaggregation.raiden.dram_offload_channel import (
    DramOffloadChannel,
    PrefixRegistration,
)
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    PrefixReady,
    PullReq,
    ReshardReq,
    RoomFailed,
    VictimPush,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.raiden_transfer import NOOP_EVENT, RaidenKVTransfer
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM

# ---------------------------------------------------------------------------
# A real radix cache on fake host pools and transfers
# ---------------------------------------------------------------------------

FULL = ComponentType.FULL


class FakeFuture:
    def __init__(self):
        self.ready = True

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.ready = True


class FakeHostTransfer:
    """Native transfer boundary; tests explicitly drive asynchronous completion."""

    def __init__(self):
        self.results: dict[str, str | None] = {}
        self.registrations: dict[str, tuple[int, list[int], str]] = {}
        self.reads: dict[str, dict] = {}
        self.registration_error: Exception | None = None
        self.read_error: Exception | None = None

    def host_control_address(self, local_ip: str) -> str:
        return "10.0.0.5:9000"

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> None:
        if self.registration_error is not None:
            raise self.registration_error
        self.registrations[req_id] = (uuid, list(block_ids))
        self.results[req_id] = None

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint,
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> None:
        if self.read_error is not None:
            raise self.read_error
        self.reads[req_id] = dict(
            uuid=uuid,
            remote_endpoint=remote_endpoint,
            remote_block_ids=list(remote_block_ids),
            local_block_ids=list(local_block_ids),
        )
        self.results[req_id] = None

    def poll_transfer(self, req_id: str) -> str | None:
        return self.results.get(req_id)

    def forget_transfer(self, req_id: str) -> None:
        assert self.results.pop(req_id) is not None

    def complete(self, status: str) -> None:
        for req_id in self.results:
            self.results[req_id] = status


class FakeHostPool:
    """Page size 1: token indices are page ids."""

    shared_allocation_domain = None

    def __init__(self, capacity: int, page_size=1):
        self.page_size = page_size
        self.capacity = capacity
        self.size = capacity
        self.used = 0
        self.transfer = FakeHostTransfer()

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


class FakeRaidenController(TpuDMAHybridCacheController):
    """The controller surface the cache's write-back paths touch."""

    def __init__(self, *, host_capacity: int):
        self.write_policy = "write_back"
        self.host = FakeHostPool(host_capacity)
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
                finish_event=FakeFuture(),
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
    controller = FakeRaidenController(host_capacity=host_capacity)
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


HYBRID_PAGE = 64


def hybrid_cache(*, window=128):
    """A real paged FULL+SWA tree with a host pool for each; only the device
    allocator and the transfers are doubles. Pages hold HYBRID_PAGE tokens."""
    allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
    allocator.device = torch.device("cpu")
    allocator.page_size = HYBRID_PAGE
    allocator._swa_req_ring = False
    allocator.translate_loc_from_full_to_swa.side_effect = lambda indices: indices
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=MagicMock(spec=ReqToTokenPool),
            token_to_kv_pool_allocator=allocator,
            page_size=HYBRID_PAGE,
            sliding_window_size=window,
            tree_components=(FULL, ComponentType.SWA),
            tree_core_backend="python",
        )
    )
    controller = FakeRaidenController(host_capacity=4096)
    controller.host.page_size = HYBRID_PAGE
    swa = FakeHostPool(1024, page_size=HYBRID_PAGE)
    cache.cache_controller = controller
    # As on an engine without a storage backend, nothing drains the L3 release
    # queue: a page handed to it would never return to its pool.
    controller.append_host_mem_release = MagicMock(
        side_effect=AssertionError("released through the undrained L3 queue")
    )
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


# ---------------------------------------------------------------------------
# Push: the prefill pusher and the decode receiver
# ---------------------------------------------------------------------------


def offload_context(*, cache, channel, page_size: int = 1) -> DramOffloadWorkerContext:
    """One engine's worker context on a fixture cache, as a single rank."""
    return DramOffloadWorkerContext(
        cache=cache,
        host_pool=radix_host_ops.anchor_host_pool(cache),
        page_size=page_size,
        channel=channel,
        lockstep=RankLockstep(),
        swa_host_pool=radix_host_ops.swa_host_pool(cache),
        window_tokens=radix_host_ops.window_tokens(cache, page_size),
    )


def push_channel(host, swa_host=None):
    """A channel double for the push: messages are recorded, and native host
    reads go to the fake transfer of ``host`` (of ``swa_host`` for the window
    of a hybrid model), which the test completes."""
    channel = MagicMock()
    # The transfer of each read, by request id.
    transfers = {}

    def register(req_id, uuid, pages, *, swa=False):
        transfer = (swa_host if swa else host).transfer
        try:
            transfer.register_read(req_id, uuid, pages)
        except RuntimeError:
            return False
        transfers[req_id] = transfer
        return True

    def start(req_id, uuid, endpoint, remote, local, *, swa=False):
        transfer = (swa_host if swa else host).transfer
        try:
            transfer.start_read(req_id, uuid, endpoint, remote, local)
        except RuntimeError:
            return False
        transfers[req_id] = transfer
        return True

    def poll(req_ids):
        statuses = [transfers[req_id].poll_transfer(req_id) for req_id in req_ids]
        return None if None in statuses else "failed" not in statuses

    def forget(req_ids):
        for req_id in req_ids:
            transfers.pop(req_id).forget_transfer(req_id)

    channel.register_host_read.side_effect = register
    channel.start_host_read.side_effect = start
    channel.poll_host_reads.side_effect = poll
    channel.forget_host_reads.side_effect = forget
    return channel


def pushed_req(*, room: int, tokens):
    """A request at its bootstrap commit, before admission set its last node."""
    return SimpleNamespace(
        bootstrap_room=room,
        origin_input_ids=tokens,
        last_node=None,
        extra_key=None,
        cache_salt=None,
    )


class PrefillPusherFixture:
    def setUp(self):
        self.cache, self.allocator = full_cache(host_capacity=32)
        self.host = self.cache.cache_controller.host
        self.mgr = push_channel(self.host)
        # The decode's answer by seq; None until the test gives one.
        self.replies = {}
        self.mgr.victim_reply.side_effect = self.replies.get
        self.peers = {}
        self.mgr.decode_peer.side_effect = lambda room: self.peers.get(room)
        self.pusher = prefill_dram_victim_pusher.PrefillDramVictimPusher(
            offload_context(cache=self.cache, channel=self.mgr)
        )
        self.a, self.b = list(range(100, 108)), list(range(200, 208))
        for room, tokens, decode in (
            (1, self.a, ("10.0.0.1", 5001)),
            (2, self.b, ("10.0.0.2", 5002)),
        ):
            insert_tokens(self.cache, self.allocator, tokens)
            self.peers[room] = decode
            self.pusher.note_decode_of(
                pushed_req(room=room, tokens=tokens), recalled=False
            )
        self.cache.evict(EvictParams(num_tokens=16))

    def offer_once(self):
        self.pusher.advance(serving=True)
        self.mgr.send_victim_push.assert_called_once()
        return self.mgr.send_victim_push.call_args.kwargs


class DecodeReceiverFixture:
    def setUp(self):
        self.cache, self.allocator = full_cache(host_capacity=64)
        self.host = self.cache.cache_controller.host
        self.mgr = push_channel(self.host)
        self.pushes = {}
        self.mgr.received_pushes.side_effect = lambda: dict(self.pushes)
        self.mgr.take_push.side_effect = lambda key: self.pushes.pop(key, None)
        self.receiver = decode_dram_victim_receiver.DecodeDramVictimReceiver(
            offload_context(cache=self.cache, channel=self.mgr)
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
        )

    def round(self):
        for _ in range(decode_dram_victim_receiver._CADENCE):
            self.receiver.advance()

    def granted(self):
        """The push is admitted: its read runs, and nothing is answered yet."""
        self.round()
        self.assertTrue(self.receiver.has_host_landings())
        self.mgr.send_victim_ack.assert_not_called()
        read = next(iter(self.host.transfer.reads.values()))
        self.assertEqual(read["remote_endpoint"], "producer:9000")
        self.assertEqual(read["remote_block_ids"], list(range(8)))
        self.assertEqual(len(read["local_block_ids"]), len(self.c))

    def answered(self):
        """Whether the prefill was told the push was stored."""
        self.mgr.send_victim_ack.assert_called_once()
        return self.mgr.send_victim_ack.call_args.kwargs["stored"]


# ---------------------------------------------------------------------------
# Recall: the prefill puller and the decode queue dram_offload_hooks
# ---------------------------------------------------------------------------

ROOM = 7


def recall_channel(*, offer_pages: int, refused: bool = False, window_pages: int = 0):
    """A channel double whose decode offered ``offer_pages`` pages (and, on a
    hybrid model, the ``window_pages`` that end there) for every room. The
    test sets ``ready`` (the decode's READY, uuid 0 refuses) and ``read_done``
    (None while the reads run) as the handshake proceeds."""
    channel = MagicMock()
    channel.taken = set()
    channel.ready = PrefixRegistration(
        uuid=0 if refused else 11,
        endpoint="10.0.0.2:5",
        swa_uuid=12 if window_pages and not refused else 0,
        swa_endpoint="10.0.0.2:6" if window_pages else "",
    )
    channel.read_done = False

    def take(room):
        if room in channel.taken:
            return None
        channel.taken.add(room)
        return list(range(offer_pages)), list(range(100, 100 + window_pages))

    channel.prefix_ready.side_effect = lambda room: channel.ready
    channel.take_prefix_offer.side_effect = take
    channel.ask_prefix_pull.return_value = True
    channel.prefix_req_id.side_effect = lambda room, swa=False: (
        f"{room}#pxs" if swa else f"{room}#px"
    )
    channel.start_host_read.return_value = True
    channel.poll_host_reads.side_effect = lambda req_ids: channel.read_done
    return channel


class HostRecallEdgeCasesFixture:
    def _build(self, *, host_capacity: int, mgr):
        cache, allocator = full_cache(kv_size=64, host_capacity=host_capacity)
        cache.tree_core.is_write_back = cache.is_write_back = False
        puller = PrefillDramPrefixPuller(offload_context(cache=cache, channel=mgr))
        puller._max_inflight_tokens = 1 << 20
        return cache, allocator, puller

    def _fill_host_with_a_device_copy(self, cache, allocator, *, lock: bool):
        tokens = list(range(100, 116))
        insert_tokens(cache, allocator, tokens)
        node = root_node(cache, 100)
        host = cache.cache_controller.host
        cache.tree_core.commit_backup(node.id, host.alloc(len(node.key)), {})
        cache.tree_core._update_duplicate_tracking(node)
        if lock:
            cache.inc_host_lock_ref(node.id)
        self.assertEqual(host.available_size(), 0)
        return host

    def _req(self):
        return SimpleNamespace(
            bootstrap_room=ROOM,
            origin_input_ids=list(range(200, 232)),
            extra_key=None,
            cache_salt=None,
        )


class RecallLifecycleFixture:
    def _scripted(self):
        """A channel that offers 16 pages once and lets the test set the READY
        and the read's outcome as the handshake proceeds."""
        mgr = recall_channel(offer_pages=16)
        mgr.ready = None
        return mgr

    def _advance_to_pulling(self, puller, mgr, req):
        puller.advance([req], waiting_reqs=[])
        plan = puller._plans[ROOM]
        self.assertEqual((plan.started, plan.new_indices), (False, None))
        mgr.ask_prefix_pull.assert_called_once()
        # Host pages are reserved only once READY is in.
        puller.advance([req], waiting_reqs=[])
        self.assertIsNone(plan.new_indices)
        mgr.ready, mgr.read_done = (
            PrefixRegistration(uuid=11, endpoint="10.0.0.2:5"),
            None,
        )
        puller.advance([req], waiting_reqs=[])
        self.assertEqual(plan.read_ids, [f"{ROOM}#px"])
        self.assertEqual(len(plan.new_indices), 16)
        return plan


class DecodeQueueHooksFixture:
    def _queue(self, *rids):
        reqs = [
            SimpleNamespace(req=SimpleNamespace(rid=rid, bootstrap_room=n))
            for n, rid in enumerate(rids)
        ]
        return SimpleNamespace(queue=reqs, tree_cache=None), reqs


# ---------------------------------------------------------------------------
# Startup refusals
# ---------------------------------------------------------------------------


def _tree_cache():
    cache = MagicMock(spec=UnifiedRadixCache)
    cache.tree_core = MagicMock(spec=UnifiedTreeCore)
    cache.tree_core.has_swa_host_pool = False
    return cache


def supported_engine():
    """A prefill or decode engine in a setting the offload supports."""
    return SimpleNamespace(
        scheduler=SimpleNamespace(
            server_args=SimpleNamespace(
                disable_radix_cache=False,
                disaggregation_transfer_backend="raiden",
                enable_hierarchical_cache=True,
                pp_size=1,
                dp_size=1,
                attn_dp_size=1,
                page_size=64,
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            tree_cache=_tree_cache(),
            model_config=SimpleNamespace(
                is_hybrid_swa=False, is_hybrid_swa_compress=False
            ),
        ),
        memory=SimpleNamespace(
            hicache_io_backend="raiden",
            hicache_storage_backend=None,
            enable_unified_memory=False,
            hicache_write_policy="write_back",
        ),
        disagg=SimpleNamespace(
            disaggregation_decode_enable_radix_cache=True,
            optimistic_prefill_attempts=0,
        ),
        device=SimpleNamespace(sleep_on_idle=False),
        parallel=SimpleNamespace(attn_cp_size=1),
        # What the pool configurator's GDN/Mamba predicate returns.
        mambaish=None,
        missing_api=[],
    )


def validate_engine_as(engine, role="prefill"):
    with ExitStack() as stack:
        for name in ("memory", "disagg", "device", "parallel"):
            stack.enter_context(
                patch(
                    f"sglang.srt.runtime_context.get_{name}",
                    return_value=getattr(engine, name),
                )
            )
        stack.enter_context(
            patch(
                "sglang.srt.configs.hybrid_arch.mambaish_config",
                return_value=engine.mambaish,
            )
        )
        stack.enter_context(
            patch.object(
                startup_validation,
                "_missing_tpu_sync_host_api",
                return_value=engine.missing_api,
            )
        )
        startup_validation.validate_engine_args(role=role, scheduler=engine.scheduler)


# ---------------------------------------------------------------------------
# The raiden channel on bare managers
# ---------------------------------------------------------------------------

PAGE = 4
# Request id of the prefill's read of the decode-held prefix.
PX = f"{ROOM}#px"
PREFILL_INFO = {"rank_ip": "10.0.0.1", "rank_port": 4444}
PREFILL_ADDR = _tcp_endpoint(ip="10.0.0.1", port=4444)
DECODE_ADDR = _tcp_endpoint(ip="10.0.0.2", port=5555)
# Control endpoint of the decode's HiCache host pool.
DECODE_RAIDEN = "10.0.0.2:7001"


def _bare_prefill_manager():
    """A prefill-mode RaidenKVManager with its room tables, no sockets or
    engine, on which a real RaidenKVSender can be built."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr.attn_tp_rank = 0
    mgr._prefill = PrefillState(announce_pages=1)
    mgr._decode = None
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._engine = None
    mgr._swa_engine = None
    mgr._reshard_slot_rooms = [None]
    mgr._reshard_slot_of = {}
    mgr._aborted = {}
    mgr._pulling = None
    mgr._pulling_cleared = set()
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr.chunked_transfer = False
    mgr._send_reads = {}
    mgr._superseded_sender_reads = set()
    mgr._timeout_s = 120.0
    mgr._reshard_pushes_pending = {}
    mgr.is_dummy_cp_rank = False
    mgr.bootstrap_timeout = 300
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr.req_to_decode_prefix_len = {}
    mgr._message_handlers = {
        PullReq.TAG: mgr._on_pull_req,
        ReshardReq.TAG: mgr._on_reshard_req,
    }
    return mgr


def _bare_decode_manager():
    """A decode-mode RaidenKVManager with its room tables, no sockets or engine."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr._engine = None
    mgr._swa_engine = None
    mgr._pulling = None
    mgr.attn_tp_rank = 0
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._aborted = {}
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._completed_recv = set()
    mgr.chunked_transfer = False
    mgr.waiting_timeout = 300
    mgr._message_handlers = {RoomFailed.TAG: mgr._on_room_failed}
    return mgr


def sent_frames(mgr):
    """(endpoint, frames) of every message the manager sent."""
    return [call.args[:2] for call in mgr._send_multipart_locked.call_args_list]


def with_offload(mgr):
    """Switch the offload on for a fixture manager: a channel whose host pool
    has DECODE_RAIDEN as its control endpoint, as the manager's __init__ and
    the L2 extension set it up."""
    transfer = Mock()
    transfer.host_control_address.side_effect = lambda ip: f"{ip}:7001"
    transfer.poll_transfer.return_value = None
    mgr.local_ip = "10.0.0.2"
    mgr.rank_port = 5555
    mgr._send_multipart_locked = Mock()
    mgr.dram_offload_channel = DramOffloadChannel(mgr)
    mgr._message_handlers.update(mgr.dram_offload_channel.message_handlers())
    mgr.dram_offload_channel.bind_host_pools(SimpleNamespace(transfer=transfer))
    return mgr


def host_transfer_of(mgr):
    return mgr.dram_offload_channel._host_pools[0].transfer


def prefill_manager():
    return with_offload(_bare_prefill_manager())


def sender_on(mgr):
    with get_context().override_server_args(dp_size=1):
        return RaidenKVSender(mgr, "unused", ROOM)


def decode_manager():
    mgr = with_offload(_bare_decode_manager())
    mgr.kv_args = SimpleNamespace(page_size=PAGE)
    return mgr


def pull_req(prefix_blocks=(3, 4, 5, 6), kv_offload=False):
    return PullReq(
        room=ROOM,
        decode_ip="10.0.0.2",
        decode_port=5555,
        decode_pages=4,
        prefix_blocks=list(prefix_blocks),
        kv_offload=kv_offload,
    ).to_frames()


def register_decode(mgr, offered=()):
    """Send the room's PULL_REQ, after the decode offered ``offered`` pages."""
    if mgr.dram_offload_channel is not None:
        mgr.dram_offload_channel.offer_resident_prefix(ROOM, list(offered))
    mgr.register_decode_request(ROOM, [21, 22, 23], [], None, [PREFILL_INFO])


class DecodeOffersItsPrefixFixture:
    def _served(self, mgr, first_page, num_pages):
        """The decode's PREFIX_READY to a PREFIX_PULL of the given range."""
        mgr._dispatch(
            PrefixPull(
                room=ROOM, first_page=first_page, num_pages=num_pages
            ).to_frames()
        )
        endpoint, frames = sent_frames(mgr)[-1]
        self.assertEqual(endpoint, PREFILL_ADDR)
        return PrefixReady.from_frames(frames)


class PrefillRecallsIntoItsHostPagesFixture:
    def _armed(self):
        """A prefill manager whose decode offered pages [3, 4, 5, 6] for ROOM."""
        mgr = prefill_manager()
        sender_on(mgr)
        mgr._dispatch(pull_req())
        self.assertEqual(
            mgr.dram_offload_channel.take_prefix_offer(ROOM), ([3, 4, 5, 6], [])
        )
        return mgr

    @staticmethod
    def _ready():
        return PrefixReady(room=ROOM, uuid=99, endpoint=DECODE_RAIDEN)


# ---------------------------------------------------------------------------
# Host-to-host reads on the HiCache raiden transfer
# ---------------------------------------------------------------------------


class FakeNativeManager:
    """External boundary recording the native memory-domain contract."""

    def __init__(self) -> None:
        self.registered: tuple | None = None
        self.started: tuple | None = None
        self.reports: deque[SimpleNamespace] = deque()

    def register_read(self, req_id, uuid, block_ids, *, source):
        self.registered = (req_id, uuid, block_ids, source)

    def start_read(
        self,
        req_id,
        uuid,
        endpoint,
        remote_ids,
        local_ids,
        *,
        parallelism,
        destination,
    ):
        self.started = (
            req_id,
            uuid,
            endpoint,
            remote_ids,
            local_ids,
            destination,
            parallelism,
        )

    def poll_stats_with_details(self):
        if self.reports:
            return self.reports.popleft()
        return SimpleNamespace(sent=[], received=[], failed=[])


def native_report(*, sent=(), received=(), failed=()):
    def details(req_id):
        return SimpleNamespace(
            req_id=req_id, code="UNAVAILABLE", phase="push", reason="peer lost"
        )

    return SimpleNamespace(
        sent=[details(req_id) for req_id in sent],
        received=[details(req_id) for req_id in received],
        failed=[details(req_id) for req_id in failed],
    )


class RaidenHostTransferFixture:
    def setUp(self) -> None:
        self.native = FakeNativeManager()
        self.transfer = RaidenKVTransfer.__new__(RaidenKVTransfer)
        self.transfer._mgr = self.native
        self.transfer._host_transfers = {}
        self.transfer._host_control_port = 12345
