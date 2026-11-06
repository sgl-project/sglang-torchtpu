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
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
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
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVSender
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.disaggregation.raiden.serde import (
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
        local_block_ids: list[int],
        *,
        destination: str = "device",
    ) -> None:
        if self.read_error is not None:
            raise self.read_error
        self.reads[req_id] = dict(
            uuid=uuid,
            remote_endpoint=remote_endpoint,
            remote_block_ids=list(remote_block_ids),
            local_block_ids=list(local_block_ids),
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
    )


def push_channel(host):
    """A channel double for the push: messages are recorded, and native host
    reads go to ``host``'s fake transfer, which the test completes."""
    transfer = host.transfer
    channel = MagicMock()

    def register(req_id, uuid, pages):
        try:
            transfer.register_read(req_id, uuid, pages, source="host")
        except RuntimeError:
            return False
        return True

    def start(req_id, uuid, endpoint, remote, local):
        try:
            transfer.start_read(
                req_id, uuid, endpoint, remote, local, destination="host"
            )
        except RuntimeError:
            return False
        return True

    def poll(req_id):
        status = transfer.poll_transfer(req_id)
        return None if status is None else status != "failed"

    channel.register_host_read.side_effect = register
    channel.start_host_read.side_effect = start
    channel.poll_host_read.side_effect = poll
    channel.forget_host_read.side_effect = transfer.forget_transfer
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
            self.pusher.note_decode_of(pushed_req(room=room, tokens=tokens))
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
        self.assertEqual(read["destination"], "host")
        self.assertEqual(len(read["local_block_ids"]), len(self.c))

    def answered(self):
        """Whether the prefill was told the push was stored."""
        self.mgr.send_victim_ack.assert_called_once()
        return self.mgr.send_victim_ack.call_args.kwargs["stored"]


ROOM = 7


# ---------------------------------------------------------------------------
# Startup refusals
# ---------------------------------------------------------------------------


def _engine_manager():
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr.pp_size = mgr.attn_cp_size = mgr.attn_dp_size = mgr.system_dp_size = 1
    mgr.attn_tp_size = 8
    mgr._pipeline_enabled = False
    mgr._state_kv_caches = ()
    mgr._swa_kv_caches = []
    mgr.kv_args = SimpleNamespace(page_size=64, kv_cache_dtype_str="fp8_e4m3")
    return mgr


def _tree_cache():
    cache = MagicMock(spec=UnifiedRadixCache)
    cache.tree_core = MagicMock(spec=UnifiedTreeCore)
    cache.cache_controller = object()
    return cache


def supported_engine():
    """A prefill or decode engine in a setting the offload supports."""
    return SimpleNamespace(
        kv_manager=_engine_manager(),
        tree_cache=_tree_cache(),
        scheduler=SimpleNamespace(
            server_args=SimpleNamespace(disable_radix_cache=False),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
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
        missing_api=[],
    )


def validate_engine_as(engine, role="prefill", switch=True):
    with ExitStack() as stack:
        # Not SWITCH.override: it does not restore the variable when the body
        # raises, and a refusal raises.
        stack.enter_context(patch.dict("os.environ", {SWITCH.name: str(int(switch))}))
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_memory", return_value=engine.memory)
        )
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_disagg", return_value=engine.disagg)
        )
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_device", return_value=engine.device)
        )
        stack.enter_context(
            patch.object(
                startup_validation,
                "missing_tpu_sync_host_api",
                return_value=engine.missing_api,
            )
        )
        startup_validation.validate_engine(
            role=role,
            scheduler=engine.scheduler,
            kv_manager=engine.kv_manager,
            tree_cache=engine.tree_cache,
        )


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
    mgr.dram_offload_channel.bind_host_pool(SimpleNamespace(transfer=transfer))
    return mgr


def host_transfer_of(mgr):
    return mgr.dram_offload_channel._host_pool.transfer


def prefill_manager():
    return with_offload(_bare_prefill_manager())


def sender_on(mgr):
    with get_context().override_server_args(dp_size=1):
        return RaidenKVSender(mgr, "unused", ROOM)


def decode_manager():
    mgr = with_offload(_bare_decode_manager())
    mgr.kv_args = SimpleNamespace(page_size=PAGE)
    return mgr


def pull_req(kv_offload=False):
    return PullReq(
        room=ROOM,
        decode_ip="10.0.0.2",
        decode_port=5555,
        decode_pages=4,
        kv_offload=kv_offload,
    ).to_frames()


# ---------------------------------------------------------------------------
# Host-to-host reads on the HiCache raiden transfer
# ---------------------------------------------------------------------------


class FakeNativeManager:
    """External boundary recording the native memory-domain contract."""

    def __init__(self) -> None:
        self.registered: tuple | None = None
        self.started: tuple | None = None
        self.registration_needed = True
        self.reports: deque[SimpleNamespace] = deque()

    def register_read(self, req_id, uuid, block_ids, *, source):
        self.registered = (req_id, uuid, block_ids, source)
        return self.registration_needed

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
        self.transfer._parallelism = 4
