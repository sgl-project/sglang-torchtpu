"""Raiden transfer into a decode-context-parallel (DCP) decode.

The decode keeps the prefill's attention TP and splits each request's
full-attention pages across its DCP group by page interleave. Its page table
is global with logical pages of ``dcp_size`` physical pages: position ``p``
belongs to rank ``(p // page) % dcp_size`` at local position ``(p // (dcp_size
* page)) * page + p % page``, so the prefill's physical page ``k`` of a request
lands whole on decode rank ``k % dcp_size`` at that rank's local page ``k //
dcp_size`` of the logical page, and each rank stores its group's kv heads for
the pages it holds. On the 512-granular logical page ids the decode lists for
the transfer (``l = loc // page``) that is owner ``l % dcp_size`` and local
page ``l // dcp_size``. Decode rank ``u`` therefore takes its pages from
prefill ranks ``u // dcp_size * dcp_size ..`` (a ``src_tp -> src_tp //
dcp_size`` fan-in of their heads) and its sliding-window pages, one head per
rank over every page, from prefill rank ``u`` alone.

These tests drive real ``RaidenKVManager`` pairs over an in-process loopback
with the native engine faked (as ``test_disagg_raiden_transfer`` does), and
check the plans both sides register byte for byte against a simulated push;
no TPU, tpu_sync or torch_tpu is needed. With ``dcp_size == 1`` the same
code paths produce the single-peer protocol the existing reshard tests pin.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.separate_buffer_allocator_double import bind_separate_buffer_capacity
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden import reshard_schedule as rs
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    ReshardBlocks,
    ReshardChunk,
    ReshardReady,
    ReshardReq,
)
from sglang_tpu.raiden_kv_backup import sequential_page_runs

# A small head-major (rank-4 RPA-seq) pool: prefill attention TP 4 with one
# kv head per rank; the decode keeps TP 4 and runs DCP 2, so a DCP pair
# holds two heads per full-attention page and one head per window page.
ATTN_TP = 4
DCP = 2
TOTAL_HEADS = 4
PAGE = 8
LANES = 4
ELEM = 2
RUN = 2 * LANES * PAGE * ELEM  # one head's K+V run of a page


# ---- mocks: protobuf request, KV layer tensors, native engine ---------------
class _Entry:
    pass


class _EntryList(list):
    def add(self):
        entry = _Entry()
        self.append(entry)
        return entry


class _Schedule:
    def __init__(self):
        self.entries = _EntryList()


class _ScheduleMap(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = _Schedule()
        return dict.__getitem__(self, key)


class _Req:
    def __init__(self):
        self.shard_push_schedules = _ScheduleMap()


class _Pb2:
    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return _Req()


def _pb2_modules():
    rpc = types.ModuleType("tpu_sync.rpc")
    rpc.raiden_service_pb2 = _Pb2
    pkg = types.ModuleType("tpu_sync")
    pkg.rpc = rpc
    return {
        "tpu_sync": pkg,
        "tpu_sync.rpc": rpc,
        "tpu_sync.rpc.raiden_service_pb2": _Pb2,
    }


class _Tensor:
    def __init__(self, shape, itemsize=ELEM):
        self.shape = shape
        self._itemsize = itemsize

    def element_size(self):
        return self._itemsize


def _head_major(blocks, heads):
    return _Tensor((blocks, 2 * heads, LANES, PAGE))


class _ReadyFuture:
    def is_ready(self):
        return True

    def ok(self):
        return True


class _FakeEngine:
    """Records the plans the manager registers and the pushes it issues."""

    def __init__(self, address):
        self.transfer_address = address
        self.plans = {}
        self.sender_plans = {}
        self.pushes = []
        self.staged = []
        self._stats = ([], [], [])

    def register_active_plan(self, uuid, req, is_sender):
        table = self.sender_plans if is_sender else self.plans
        if uuid in table:
            raise RuntimeError(f"plan {uuid} already registered")
        table[uuid] = req

    def unregister_active_plan(self, uuid):
        self.sender_plans.pop(uuid, None)
        self.plans.pop(uuid, None)

    def plan_host_blocks(self, uuid, blocks):
        return list(blocks)

    def d2h(self, blocks, host_blocks, counts):
        self.staged.append(list(blocks))
        return _ReadyFuture()

    def push_registered_plan(self, uuid, peer, src, dst, layer_idx, parallelism=1):
        self.pushes.append((uuid, peer, list(src), list(dst), layer_idx))

    def poll_stats_with_details(self):
        sent, received, failed = self._stats
        self._stats = ([], [], [])

        def settled(ids):
            return [SimpleNamespace(req_id=r, completed_ns=None) for r in ids]

        return SimpleNamespace(
            sent=settled(sent), received=settled(received), failed=settled(failed)
        )


class _InlineExecutor:
    def submit(self, fn, *args):
        fn(*args)


class _Inbox:
    def __init__(self):
        self.frames = []

    def deliver(self, frames):
        self.frames.append(frames)

    def take(self, tag):
        if not self.frames:
            raise AssertionError(f"expected {tag.name}, got nothing")
        frames = self.frames.pop(0)
        if frames[0] != tag:
            raise AssertionError(f"expected {tag.name}, got {frames[0]!r}")
        return frames

    def take_all(self, tag):
        taken = []
        while self.frames and self.frames[0][0] == tag:
            taken.append(self.frames.pop(0))
        return taken


class _LoopbackSender:
    def __init__(self, endpoint, inboxes):
        self.endpoint = endpoint
        self._inboxes = inboxes

    def send_multipart(self, msg, copy=True, track=False):
        self._inboxes[self.endpoint].deliver([bytes(part) for part in msg])


def _page_bytes(block, heads):
    """Deterministic contents of one head-major page: ``heads`` kv heads of
    ``block`` as rows [K rows of head 0, V rows of head 0, K rows of head 1,
    ...], each row ``PAGE`` lanes of ``ELEM`` bytes."""
    out = bytearray()
    for head in heads:
        for kv in (0, 1):
            for lane in range(LANES):
                for token in range(PAGE):
                    for byte in range(ELEM):
                        out.append(
                            (block * 17 + head * 37 + kv * 43 + lane * 61 + token * 29)
                            % 251
                            + byte
                        )
    return out


def _apply(entries, source_pages, destination_pages):
    """Simulate the receiver's strided copies from the senders' staged pages."""
    for entry in entries:
        source = source_pages[entry.src_block_id]
        target = destination_pages[entry.dst_block_id]
        for repeat in range(entry.count):
            src = entry.src_offset_bytes + repeat * entry.src_stride_bytes
            dst = entry.dst_offset_bytes + repeat * entry.dst_stride_bytes
            size = entry.size_bytes
            assert src + size <= len(source) and dst + size <= len(target)
            target[dst : dst + size] = source[src : src + size]


class RaidenDcpTestCase(CustomTestCase):
    def setUp(self):
        self.inboxes = {}
        self._next_port = 42000

    def _manager(self, mode, *, attn_tp_rank, dcp_size=1):
        port = self._next_port
        self._next_port += 1
        endpoint = f"tcp://127.0.0.1:{port}"
        inbox = _Inbox()
        self.inboxes[endpoint] = inbox
        inboxes = self.inboxes
        decode = mode == DisaggregationMode.DECODE

        def _fake_common_init(m, args, disaggregation_mode, server_args, is_mla=False):
            m.kv_args = args
            m.server_args = server_args
            m.disaggregation_mode = disaggregation_mode
            m.is_mla_backend = False
            m.is_hybrid_mla_backend = False
            m.attn_tp_size = ATTN_TP
            m.attn_tp_rank = attn_tp_rank
            m.attn_dp_rank = m.attn_cp_rank = m.pp_rank = 0
            m.attn_dp_size = m.attn_cp_size = m.pp_size = m.system_dp_size = 1
            m.system_dp_rank = 0
            m.dcp_size = dcp_size if decode else 1
            m.dcp_rank = attn_tp_rank % dcp_size if decode else 0
            m.enable_all_cp_ranks_for_transfer = False
            m.local_ip = "127.0.0.1"
            m.rank_port = port
            m.server_socket = inbox
            m._socket_cache = {}
            m._socket_lock = threading.Lock()
            m._socket_send_locks = {}
            m.request_status = {}
            m.failure_records = {}
            m.failure_lock = threading.Lock()
            m._deferred_ack_targets = {}
            m._deferred_ack_poisoned_rooms = set()
            m._staging_outstanding = {}
            m.deferred_bootstrap = None

            def _connect(ep, is_ipv6=False):
                m._socket_send_locks.setdefault(ep, threading.Lock())
                return m._socket_cache.setdefault(ep, _LoopbackSender(ep, inboxes))

            m._connect = _connect

        kv_args = SimpleNamespace(
            page_size=PAGE,
            total_kv_head_num=TOTAL_HEADS,
            sliding_window_size=2 * PAGE,
            engine_rank=attn_tp_rank,
            aux_data_ptrs=[],
            aux_item_lens=[],
        )
        with (
            patch.object(CommonKVManager, "__init__", _fake_common_init),
            patch.object(RaidenKVManager, "_listen", lambda self: None),
            patch.object(
                CommonKVManager, "_start_heartbeat_checker_thread", lambda self: None
            ),
        ):
            m = RaidenKVManager(kv_args, mode, SimpleNamespace(context_length=1024))
        heads = TOTAL_HEADS // ATTN_TP * (dcp_size if decode else 1)
        m._kv_caches = [_head_major(64, heads), _head_major(64, heads)]
        m._swa_kv_caches = [_head_major(64, TOTAL_HEADS // ATTN_TP)]
        m._engine = _FakeEngine(f"raiden://127.0.0.1:{port + 1000}")
        m._swa_engine = _FakeEngine(f"raiden://127.0.0.1:{port + 2000}")
        m._full_engine_max_blocks = m._swa_engine_max_blocks = 256
        m._pulling = StagingAdmission(256, 256)
        m.chunked_transfer = True
        if not decode:
            m._push_executor = _InlineExecutor()
            m._prefill.announce_pages = 1  # every offered chunk is announced
        m.endpoint = endpoint
        return m

    def _prefill(self, rank):
        return self._manager(DisaggregationMode.PREFILL, attn_tp_rank=rank)

    def _decode(self, rank, dcp_size=DCP):
        return self._manager(
            DisaggregationMode.DECODE, attn_tp_rank=rank, dcp_size=dcp_size
        )

    @staticmethod
    def _infos(prefills):
        return [{"rank_ip": "127.0.0.1", "rank_port": p.rank_port} for p in prefills]

    def _pair_transfer(self, room, prefill_pages, decode_pages, swa_pages, *, group=1):
        """Run one room through a DCP pair: decode ranks ``2g``, ``2g + 1`` fed
        by prefill ranks ``2g``, ``2g + 1``. Returns the four managers."""
        prefills = [self._prefill(group * DCP + k) for k in range(DCP)]
        decodes = [self._decode(group * DCP + k) for k in range(DCP)]
        with patch.dict(sys.modules, _pb2_modules()):
            for decode in decodes:
                decode.register_decode_request(
                    room, decode_pages, swa_pages, None, self._infos(prefills)
                )
            for prefill in prefills:
                for _ in decodes:
                    prefill._on_reshard_req(
                        self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_REQ)
                    )
                prefill.append_send_blocks(
                    room,
                    prefill_pages,
                    swa_block_ids=list(range(90, 90 + len(swa_pages))),
                )
                prefill.finalize_send(room, None)
            for decode in decodes:
                for _ in prefills:
                    decode._on_reshard_blocks(
                        self.inboxes[decode.endpoint].take(MessageTag.RESHARD_BLOCKS)
                    )
            for prefill in prefills:
                for _ in decodes:
                    prefill._on_reshard_ready(
                        self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_READY)
                    )
        return prefills, decodes


class TestPagePlacement(unittest.TestCase):
    def test_dcp_page_split_filters_by_owner_and_localizes(self):
        pages = [2, 5, 7, 8, 11, 12]
        self.assertEqual(rs.dcp_page_split(pages, 2, 0), ([0, 3, 5], [1, 4, 6]))
        self.assertEqual(rs.dcp_page_split(pages, 2, 1), ([1, 2, 4], [2, 3, 5]))
        # A one-page request lives on one rank; a straddling request leaves
        # the ranks one page apart.
        self.assertEqual(rs.dcp_page_split([9], 2, 0), ([], []))
        self.assertEqual(rs.dcp_page_split([9], 2, 1), ([0], [4]))
        self.assertEqual(rs.dcp_page_split([4, 7, 10], 2, 0), ([0, 2], [2, 5]))
        self.assertEqual(rs.dcp_page_split([4, 7, 10], 2, 1), ([1], [3]))

    def test_consecutive_position_pages_alternate_ranks(self):
        # A request's position pages 0..4 sit in logical (1024-position)
        # pages 5, 9, 2: 512-page ids 2j, 2j + 1, so the ranks take alternate
        # prefill pages into the same local page ids, and an odd page count
        # leaves the lone last page on rank 0.
        pages = [10, 11, 18, 19, 4]
        self.assertEqual(rs.dcp_page_split(pages, 2, 0), ([0, 2, 4], [5, 9, 2]))
        self.assertEqual(rs.dcp_page_split(pages, 2, 1), ([1, 3], [5, 9]))

    def test_dcp_size_one_is_the_identity(self):
        self.assertEqual(rs.dcp_page_split([3, 9, 4], 1, 0), ([0, 1, 2], [3, 9, 4]))
        self.assertEqual(rs.dcp_page_split([], 2, 1), ([], []))

    def test_retraction_backup_sees_sequential_local_pages(self):
        # Position p of a request belongs to rank (p // page) % 2 at local
        # position (p // (2 * page)) * page + p % page. Of a 5-page request at
        # page 4, rank 0 holds position pages 0, 2, 4 as its local pages 0,
        # 1, 2 and rank 1 holds pages 1, 3 as local pages 0, 1: page-aligned
        # and whole, so the raiden retraction backup stages them (the pages
        # need not be adjacent, only the rows within them).
        ps, n = 4, 2
        pos = torch.arange(5 * ps)
        owner = (pos // ps) % n
        local = (pos // (n * ps)) * ps + pos % ps
        rows = local[owner == 0]
        self.assertEqual(rows.tolist(), list(range(12)))
        self.assertEqual(sequential_page_runs(rows, ps), ([0, 1, 2], 0, 12))
        rows = local[owner == 1]
        self.assertEqual(sequential_page_runs(rows, ps), ([0, 1], 0, 8))


class TestDecodeTransferPageSizes(unittest.TestCase):
    """The decode lists its transfer pages at the device pools' page under
    DCP, where the allocator's page spans ``dcp_size`` pool pages."""

    def setUp(self):
        # pop_preallocated reads its config from the published bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @staticmethod
    def _queue(allocator_page, pool_page, swa_page, state_types):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=allocator_page)
        queue.token_to_kv_pool = SimpleNamespace(
            page_size=pool_page, swa_kv_pool=SimpleNamespace(page_size=swa_page)
        )
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=list(state_types))
        )
        return queue

    def test_dcp_lists_use_the_pools_pages(self):
        queue = self._queue(1024, 512, 512, [StateType.SWA])
        with (
            get_parallel().override(attn_dcp_size=2),
            patch.object(
                type(current_platform), "dcp_shards_every_kv_pool", return_value=True
            ),
        ):
            self.assertEqual(queue._transfer_page_sizes(), (512, 512))
            # A model without a window pool pages everything at the full pool.
            queue = self._queue(1024, 512, 4096, [])
            self.assertEqual(queue._transfer_page_sizes(), (512, 512))

    @staticmethod
    def _preallocate(dcp_shards_every_kv_pool):
        """Run one 20-token request through pop_preallocated on a DCP2 decode
        whose pools page at 4 and whose allocator pages at 8, with the
        request's rows hand-built in the widened location space: logical
        (8-position) pages 5, 9, 2. Returns what reached send_metadata."""
        page, dcp = 4, 2
        req = SimpleNamespace(
            rid="dcp-lists",
            origin_input_ids=[0] * 20,
            output_ids=[],
            finished_reason=None,
            sampling_params=SimpleNamespace(max_new_tokens=1),
            kv=ReqKvInfo(),
            time_stats=MagicMock(),
        )
        receiver = MagicMock(spec=["send_metadata", "abort", "clear"])
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True, is_rebootstrap=False
        )
        rows = torch.cat(
            [torch.arange(j * page * dcp, j * page * dcp + page * dcp) for j in (5, 9)]
            + [torch.arange(2 * page * dcp, 2 * page * dcp + page)]
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue._num_published_destinations = 0
        queue.transfer_queue = SimpleNamespace(enable_staging=False)
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[StateType.SWA])
        )
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = page
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=16384)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(16384, 8192)
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(return_value=8192)
        queue._radix_full_available = MagicMock(return_value=16384)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        # The SWA pool mirrors the owner's local rows at an offset, physical
        # page 4, as the unsplit window pool does.
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=page * dcp,
            translate_kv_indices_for_transfer=lambda indices: indices,
            translate_swa_indices_for_transfer=lambda loc: (
                (loc // (page * dcp)) * page + loc % page + 400
            ),
        )
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.token_to_kv_pool = SimpleNamespace(
            page_size=page, swa_kv_pool=SimpleNamespace(page_size=page)
        )
        queue.tree_cache = MagicMock()
        queue.tree_cache.supports_mamba.return_value = False
        queue._match_prefix_and_lock = MagicMock(
            return_value=SimpleNamespace(
                prefix_indices=[],
                l1_prefix_len=0,
                decode_prefix_len=0,
                restore_token_count=0,
            )
        )

        def allocate(req, _prefix_indices, prefix_len, total_prefix_len):
            req.kv.req_pool_idx = 0
            return rows

        queue._pre_alloc = MagicMock(side_effect=allocate)
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_token_pool.req_to_token = rows.reshape(1, -1)
        queue.tree_cache.req_to_token_pool = queue.req_to_token_pool
        queue.req_to_token_pool.mamba_allocator.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.enable_decode_hicache = False
        scheduler.sliding_window_size = 2 * page
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler
        with (
            get_parallel().override(attn_dcp_size=dcp),
            patch.object(
                type(current_platform),
                "dcp_shards_every_kv_pool",
                return_value=dcp_shards_every_kv_pool,
            ),
        ):
            preallocated, failed = queue.pop_preallocated()
        assert (preallocated, failed) == ([decode_req], [])
        page_indices, _, state_indices = receiver.send_metadata.call_args.args
        return page_indices.tolist(), state_indices[0].tolist()

    def test_hand_built_request_lists_one_id_per_prefill_page(self):
        # Rows 40..47, 72..79, 16..19: the full list names the five
        # 4-token pages 10, 11, 18, 19, 4 (owners 0 1 0 1 0, local pages
        # 5, 5, 9, 9, 2), and the 8-token window (rows 76..79 and 16..19)
        # is two SWA-physical pages aligned at the pool's page, like the
        # prefill's own window list.
        full, swa = self._preallocate(dcp_shards_every_kv_pool=True)
        self.assertEqual(full, [10, 11, 18, 19, 4])
        self.assertEqual(swa, [(9 * 4 + 400) // 4, (2 * 4 + 400) // 4])

    def test_relayout_platforms_keep_the_allocator_page(self):
        # The MLA relayout path lists the allocator's (widened) pages: one
        # per 8 positions, the window aligned at 8.
        full, swa = self._preallocate(dcp_shards_every_kv_pool=False)
        self.assertEqual(full, [5, 9, 2])
        self.assertEqual(len(swa), 2)

    def test_without_dcp_or_widening_the_allocator_page_is_kept(self):
        queue = self._queue(1024, 512, 512, [StateType.SWA])
        with (
            get_parallel().override(attn_dcp_size=1),
            patch.object(
                type(current_platform), "dcp_shards_every_kv_pool", return_value=True
            ),
        ):
            self.assertEqual(queue._transfer_page_sizes(), (1024, 1024))
        # A platform whose DCP relays tokens into the decode's wider pages
        # (the MLA path) keeps listing the allocator's pages.
        with (
            get_parallel().override(attn_dcp_size=2),
            patch.object(
                type(current_platform), "dcp_shards_every_kv_pool", return_value=False
            ),
        ):
            self.assertEqual(queue._transfer_page_sizes(), (1024, 1024))


class TestHeadMajorGeometry(unittest.TestCase):
    def test_fused_page_entry_is_the_per_token_entry(self):
        # The default (fused) geometry is unchanged: an entry is the per-token
        # granule repeated for the page's live tokens.
        geom = rs.ReshardGeometry(8, 4, 512, 8)
        for prefill_rank, decode_rank in [(0, 0), (1, 0), (7, 3)]:
            self.assertEqual(
                geom.page_entry(prefill_rank, decode_rank, 300),
                (*geom.entry_bytes(prefill_rank, decode_rank), 300),
            )

    def test_head_major_fan_in_lands_whole_head_runs(self):
        geom = rs.ReshardGeometry(
            ATTN_TP, ATTN_TP // DCP, RUN, TOTAL_HEADS, page_tokens=PAGE
        )
        # Prefill ranks 2 and 3 feed decode pair 1: head 2 at run 0, head 3
        # at run 1 of the destination page, one contiguous copy each.
        self.assertEqual(geom.page_entry(2, 1, PAGE), (0, RUN, 0, 0, 0, 1))
        self.assertEqual(geom.page_entry(3, 1, PAGE), (0, RUN, 0, RUN, 0, 1))
        # A partial page copies the live lanes of every K/V row.
        row = PAGE * ELEM
        self.assertEqual(
            geom.page_entry(3, 1, 3), (0, 3 * ELEM, row, RUN, row, 2 * LANES)
        )
        with self.assertRaises(ValueError):
            geom.entry_bytes(2, 1)

    def test_head_major_identity_for_the_window(self):
        geom = rs.ReshardGeometry(ATTN_TP, ATTN_TP, RUN, TOTAL_HEADS, page_tokens=PAGE)
        for rank in range(ATTN_TP):
            self.assertEqual(geom.page_entry(rank, rank, PAGE), (0, RUN, 0, 0, 0, 1))

    def test_head_major_requires_whole_rows(self):
        with self.assertRaises(ValueError):
            rs.ReshardGeometry(4, 2, RUN + 2, 4, page_tokens=PAGE)


class TestDcpRankMapping(RaidenDcpTestCase):
    def test_decode_rank_takes_its_pair_of_prefill_ranks(self):
        info = SimpleNamespace(attn_tp_size=ATTN_TP, attn_cp_size=1, pp_size=1)
        for rank, expected in [(0, [0, 1]), (1, [0, 1]), (2, [2, 3]), (3, [2, 3])]:
            decode = self._decode(rank)
            decode._resolve_rank_mapping(info)
            self.assertEqual(info.target_tp_ranks, expected)
            # The window comes from the prefill rank of the same index.
            self.assertEqual(info.target_tp_rank, rank)
            self.assertTrue(decode.supports_dcp_decode())

    def test_without_dcp_the_mapping_is_the_single_source(self):
        info = SimpleNamespace(attn_tp_size=ATTN_TP, attn_cp_size=1, pp_size=1)
        decode = self._decode(3, dcp_size=1)
        decode._resolve_rank_mapping(info)
        self.assertEqual(info.target_tp_ranks, [3])

    def test_unequal_attention_tp_is_refused(self):
        info = SimpleNamespace(attn_tp_size=2 * ATTN_TP, attn_cp_size=1, pp_size=1)
        with self.assertRaisesRegex(RuntimeError, "equal attention TP"):
            self._decode(1)._resolve_rank_mapping(info)

    def test_wrong_pool_layout_is_refused_at_startup(self):
        decode = self._decode(1)
        decode._kv_caches = [_head_major(64, 1)]  # one head: not the pair layout
        with self.assertRaisesRegex(RuntimeError, "4 K/V rows per page"):
            decode._make_geometry(ATTN_TP, ATTN_TP // DCP)
        decode._kv_caches = [_head_major(64, DCP)]
        decode._make_geometry(ATTN_TP, ATTN_TP // DCP)
        # A window pool paged at dcp_size times the transfer page (what a
        # DCP-widened allocator would give it) cannot take whole-page pushes.
        decode._swa_kv_caches = [_Tensor((64, 2, LANES, DCP * PAGE))]
        with self.assertRaisesRegex(RuntimeError, "paged at the transfer page"):
            decode._make_geometry(ATTN_TP, ATTN_TP, swa=True)


class TestDcpReshardHandshake(RaidenDcpTestCase):
    ROOM = 61
    # The request's six position pages sit in the decode's logical
    # (1024-position) pages 5, 9, 2, listed as 512-page ids 2j, 2j + 1: ranks
    # 0 1 0 1 0 1 own them, at local pages 5 5 9 9 2 2.
    DECODE_PAGES = [10, 11, 18, 19, 4, 5]
    PREFILL_PAGES = [30, 31, 32, 33, 34, 35]
    SWA_PAGES = [50, 51]

    def test_request_names_the_dcp_fan_in(self):
        prefills = [self._prefill(2), self._prefill(3)]
        decode = self._decode(3)
        decode.register_decode_request(
            self.ROOM, self.DECODE_PAGES, self.SWA_PAGES, None, self._infos(prefills)
        )
        for prefill in prefills:
            req = ReshardReq.from_frames(
                self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_REQ)
            )
            self.assertEqual((req.dst_attn_tp, req.dcp_size, req.dcp_rank), (2, 2, 1))
            self.assertEqual(req.decode_blocks, self.DECODE_PAGES)
            self.assertEqual(req.decode_swa_blocks, self.SWA_PAGES)
        meta = decode._decode.reshard_meta[self.ROOM]
        self.assertEqual(meta["source_ranks"], [2, 3])
        self.assertEqual(meta["swa_source_ranks"], [3])
        self.assertEqual((meta["src_attn_tp"], meta["dst_attn_tp"]), (4, 2))
        self.assertEqual(meta["decode_rank"], 1)
        self.assertNotEqual(meta["uuid"], self.ROOM)

    def test_without_dcp_the_request_is_unchanged(self):
        prefills = [self._prefill(0), self._prefill(1)]
        decode = self._decode(0, dcp_size=1)
        decode.attn_tp_size = ATTN_TP // 2  # a plain 4 -> 2 attention-TP reshard
        decode.register_decode_request(
            self.ROOM, self.DECODE_PAGES, [], None, self._infos(prefills)
        )
        req = ReshardReq.from_frames(
            self.inboxes[prefills[0].endpoint].take(MessageTag.RESHARD_REQ)
        )
        self.assertEqual((req.dst_attn_tp, req.dcp_size, req.dcp_rank), (2, 1, 0))
        meta = decode._decode.reshard_meta[self.ROOM]
        self.assertEqual((meta["src_attn_tp"], meta["uuid"]), (4, self.ROOM))

    def test_each_rank_receives_its_pages_with_both_heads(self):
        prefills, decodes = self._pair_transfer(
            self.ROOM, self.PREFILL_PAGES, self.DECODE_PAGES, self.SWA_PAGES
        )
        expected = {0: ([0, 2, 4], [5, 9, 2]), 1: ([1, 3, 5], [5, 9, 2])}
        for decode in decodes:
            dcp_rank = decode.dcp_rank
            positions, local = expected[dcp_rank]
            meta = decode._decode.reshard_meta[self.ROOM]
            closing_id = f"{self.ROOM}#dcp{dcp_rank}"
            plan = decode._engine.plans[meta["uuid"]]
            self.assertEqual(plan.req_id, closing_id)
            self.assertEqual(set(plan.shard_push_schedules), {2, 3})
            self.assertEqual(plan.expected_block_count, 2 * len(positions))
            for sender, offset in ((2, 0), (3, RUN)):
                entries = plan.shard_push_schedules[sender].entries
                self.assertEqual(
                    [e.src_block_id for e in entries],
                    [self.PREFILL_PAGES[i] for i in positions],
                )
                self.assertEqual([e.dst_block_id for e in entries], local)
                self.assertEqual({e.dst_offset_bytes for e in entries}, {offset})
                self.assertEqual({(e.size_bytes, e.count) for e in entries}, {(RUN, 1)})
            self.assertEqual(decode._decode.chunk_reads[self.ROOM], [closing_id])
            self.assertIn(self.ROOM, decode._decode.final_seen)
        # Each prefill asked both decode ranks for a push and told each its
        # own pages.
        for prefill in prefills:
            readies = [
                ReshardReady.from_frames(f)
                for f in self.inboxes[prefill.endpoint].take_all(
                    MessageTag.RESHARD_READY
                )
            ]
            self.assertEqual(readies, [])  # consumed by the handshake
            peers = prefill._prefill.reshard_req[self.ROOM]
            self.assertEqual(set(peers), {0, 1})
            self.assertEqual(
                prefill._prefill.reshard_ready_pages[self.ROOM],
                {0: [0, 2, 4], 1: [1, 3, 5]},
            )

    def test_sender_plans_mirror_the_receiver_plans(self):
        prefills, decodes = self._pair_transfer(
            self.ROOM, self.PREFILL_PAGES, self.DECODE_PAGES, self.SWA_PAGES
        )
        for decode in decodes:
            meta = decode._decode.reshard_meta[self.ROOM]
            receiver = decode._engine.plans[meta["uuid"]]
            for prefill in prefills:
                sent = [
                    (uuid, peer, src, dst)
                    for uuid, peer, src, dst, _ in prefill._engine.pushes
                    if peer == decode._engine.transfer_address
                ]
                self.assertEqual(len(sent), 2)  # one push per full layer
                (uuid, _, src, dst) = sent[0]
                self.assertEqual(uuid, meta["uuid"])
                entries = receiver.shard_push_schedules[prefill.attn_tp_rank].entries
                self.assertEqual(src, [e.src_block_id for e in entries])
                self.assertEqual(dst, [e.dst_block_id for e in entries])

    def test_odd_page_count_and_prefix_tail_pair_by_position(self):
        """A lone last page goes to one rank only; a decode that already holds
        a prefix pairs the prefill's list from the tail, so the positions in
        READY are the prefill's, not the decode's."""
        cases = [
            # five pages: rank 0 takes 0, 2, 4 and rank 1 takes 1, 3
            (
                self.PREFILL_PAGES[:5],
                self.DECODE_PAGES[:5],
                0,
                {0: [0, 2, 4], 1: [1, 3]},
            ),
            # prefix hit: the decode holds the last four of six prefill pages
            (self.PREFILL_PAGES, self.DECODE_PAGES[2:], 2, {0: [2, 4], 1: [3, 5]}),
        ]
        for prefill_pages, decode_pages, base, positions in cases:
            with self.subTest(pages=len(decode_pages), base=base):
                self.setUp()
                prefills, decodes = self._pair_transfer(
                    self.ROOM, prefill_pages, decode_pages, self.SWA_PAGES
                )
                for prefill in prefills:
                    self.assertEqual(
                        prefill._prefill.reshard_ready_pages[self.ROOM], positions
                    )
                for decode in decodes:
                    plan = decode._engine.plans[
                        decode._decode.reshard_meta[self.ROOM]["uuid"]
                    ]
                    owned = positions[decode.dcp_rank]
                    for sender in (2, 3):
                        entries = plan.shard_push_schedules[sender].entries
                        self.assertEqual(
                            [e.src_block_id for e in entries],
                            [prefill_pages[pg] for pg in owned],
                        )
                        self.assertEqual(
                            [e.dst_block_id for e in entries],
                            [decode_pages[pg - base] // DCP for pg in owned],
                        )
                    # The prefill pushed exactly those pages to this rank.
                    for prefill in prefills:
                        pushed = [
                            (src, dst)
                            for _, peer, src, dst, layer in prefill._engine.pushes
                            if peer == decode._engine.transfer_address and layer == 0
                        ]
                        self.assertEqual(
                            pushed,
                            [
                                (
                                    [prefill_pages[pg] for pg in owned],
                                    [decode_pages[pg - base] // DCP for pg in owned],
                                )
                            ],
                        )

    def test_pushed_bytes_assemble_each_ranks_pages(self):
        prefills, decodes = self._pair_transfer(
            self.ROOM, self.PREFILL_PAGES, self.DECODE_PAGES, self.SWA_PAGES
        )
        staged = {
            prefill.attn_tp_rank: {
                page: _page_bytes(page, [prefill.attn_tp_rank])
                for page in self.PREFILL_PAGES
            }
            for prefill in prefills
        }
        blank = bytearray(2 * RUN)
        for decode in decodes:
            meta = decode._decode.reshard_meta[self.ROOM]
            plan = decode._engine.plans[meta["uuid"]]
            destination = {page: bytearray(blank) for page in range(16)}
            for sender in (2, 3):
                _apply(
                    plan.shard_push_schedules[sender].entries,
                    staged[sender],
                    destination,
                )
            positions, local = rs.dcp_page_split(
                self.DECODE_PAGES, DCP, decode.dcp_rank
            )
            for position, page in zip(positions, local):
                self.assertEqual(
                    destination[page],
                    _page_bytes(self.PREFILL_PAGES[position], [2, 3]),
                    f"decode rank {decode.attn_tp_rank} local page {page}",
                )
            untouched = set(range(16)) - set(local)
            self.assertTrue(all(destination[page] == blank for page in untouched))

    def test_window_moves_from_the_matching_prefill_rank_only(self):
        prefills, decodes = self._pair_transfer(
            self.ROOM, self.PREFILL_PAGES, self.DECODE_PAGES, self.SWA_PAGES
        )
        for decode in decodes:
            plan = decode._swa_engine.plans[self.ROOM]
            self.assertEqual(plan.req_id, f"swa:{self.ROOM}")
            # One source, prefill rank u, landing head u whole at offset 0.
            self.assertEqual(set(plan.shard_push_schedules), {decode.attn_tp_rank})
            entries = plan.shard_push_schedules[decode.attn_tp_rank].entries
            self.assertEqual([e.src_block_id for e in entries], [90, 91])
            self.assertEqual([e.dst_block_id for e in entries], self.SWA_PAGES)
            self.assertEqual({e.dst_offset_bytes for e in entries}, {0})
        for prefill in prefills:
            peers = {peer for _, peer, *_ in prefill._swa_engine.pushes}
            partner = next(d for d in decodes if d.attn_tp_rank == prefill.attn_tp_rank)
            self.assertEqual(peers, {partner._swa_engine.transfer_address})

    def test_room_settles_only_after_every_peer(self):
        prefills, decodes = self._pair_transfer(
            self.ROOM, self.PREFILL_PAGES, self.DECODE_PAGES, self.SWA_PAGES
        )
        for prefill in prefills:
            self.assertTrue(prefill.is_send_done(self.ROOM))
            self.assertEqual(prefill._prefill.reshard_final_pushed[self.ROOM], {0, 1})
        for decode in decodes:
            self.assertFalse(decode.is_recv_done(self.ROOM))
            decode._engine._stats = ([], [f"{self.ROOM}#dcp{decode.dcp_rank}"], [])
            decode._swa_engine._stats = ([], [f"swa:{self.ROOM}"], [])
            decode._drain_engine()
            self.assertTrue(decode.is_recv_done(self.ROOM))
            self.assertFalse(decode.room_has_active_io(self.ROOM))

    def test_prefill_waits_for_the_second_peer(self):
        """The first peer's closing push must not settle the room: the
        other rank's pages are still to be pushed from the same send."""
        prefills = [self._prefill(2), self._prefill(3)]
        decodes = [self._decode(2), self._decode(3)]
        prefill = prefills[0]
        with patch.dict(sys.modules, _pb2_modules()):
            for decode in decodes:
                decode.register_decode_request(
                    self.ROOM,
                    self.DECODE_PAGES,
                    self.SWA_PAGES,
                    None,
                    self._infos(prefills),
                )
            first = self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_REQ)
            second = self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_REQ)
            prefill._on_reshard_req(first)
            prefill.append_send_blocks(
                self.ROOM, self.PREFILL_PAGES, swa_block_ids=[90, 91]
            )
            prefill.finalize_send(self.ROOM, None)
            # Only the peer that asked gets the blocks; it registers and
            # clears the push, which runs and completes.
            decode = next(
                d
                for d in decodes
                if d.dcp_rank == ReshardReq.from_frames(first).dcp_rank
            )
            decode._on_reshard_blocks(
                self.inboxes[decode.endpoint].take(MessageTag.RESHARD_BLOCKS)
            )
            self.assertEqual(self.inboxes[prefill.endpoint].frames, [])
            # The decode needs its other source before it publishes a plan.
            prefills[1]._on_reshard_req(
                self.inboxes[prefills[1].endpoint].take(MessageTag.RESHARD_REQ)
            )
            prefills[1].append_send_blocks(
                self.ROOM, self.PREFILL_PAGES, swa_block_ids=[90, 91]
            )
            prefills[1].finalize_send(self.ROOM, None)
            decode._on_reshard_blocks(
                self.inboxes[decode.endpoint].take(MessageTag.RESHARD_BLOCKS)
            )
            prefill._on_reshard_ready(
                self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_READY)
            )
            self.assertEqual(
                prefill._prefill.reshard_final_pushed[self.ROOM], {decode.dcp_rank}
            )
            self.assertFalse(prefill.is_send_done(self.ROOM))

            # The second peer arrives after the first push: it is replied to
            # at once, and its push settles the room.
            prefill._on_reshard_req(second)
            other = next(d for d in decodes if d is not decode)
            other._on_reshard_blocks(
                self.inboxes[other.endpoint].take(MessageTag.RESHARD_BLOCKS)
            )
            prefills[1]._on_reshard_req(
                self.inboxes[prefills[1].endpoint].take(MessageTag.RESHARD_REQ)
            )
            other._on_reshard_blocks(
                self.inboxes[other.endpoint].take(MessageTag.RESHARD_BLOCKS)
            )
            prefill._on_reshard_ready(
                self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_READY)
            )
        self.assertEqual(prefill._prefill.reshard_final_pushed[self.ROOM], {0, 1})
        self.assertTrue(prefill.is_send_done(self.ROOM))

    def test_dcp_rooms_are_not_chunked(self):
        prefills = [self._prefill(0), self._prefill(1)]
        decode = self._decode(1)
        prefill = prefills[0]
        # Chunks committed before the decode asked are held, then folded
        # into the closing push rather than announced; later ones too.
        prefill.append_send_blocks(self.ROOM, self.PREFILL_PAGES[:2])
        prefill.offer_chunk(self.ROOM, 0, self.PREFILL_PAGES[:2], 6)
        decode.register_decode_request(
            self.ROOM, self.DECODE_PAGES, self.SWA_PAGES, None, self._infos(prefills)
        )
        prefill._on_reshard_req(
            self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_REQ)
        )
        prefill.append_send_blocks(self.ROOM, self.PREFILL_PAGES[2:4])
        prefill.offer_chunk(self.ROOM, 2, self.PREFILL_PAGES[2:4], 6)
        self.assertEqual(self.inboxes[decode.endpoint].frames, [])
        self.assertNotIn(self.ROOM, prefill._prefill.reshard_chunks)
        # A chunk that reaches a DCP decode anyway fails the room loudly.
        decode._on_reshard_chunk(
            ReshardChunk(self.ROOM, 0, 0, 6, self.PREFILL_PAGES[:2], 0).to_frames()
        )
        self.assertTrue(decode.is_failed(self.ROOM))
        self.assertIn("closing push alone", decode.failure_records[self.ROOM])

    def test_chunks_offered_after_every_peer_asked_ride_the_closing_push(self):
        """Under load the decode pair asks before the prefill runs a single
        chunk; the chunk offers that follow are neither announced nor fatal
        (the device crash: a one-peer unpack of a two-peer room)."""
        prefills = [self._prefill(0), self._prefill(1)]
        decodes = [self._decode(0), self._decode(1)]
        with patch.dict(sys.modules, _pb2_modules()):
            for decode in decodes:
                decode.register_decode_request(
                    self.ROOM,
                    self.DECODE_PAGES,
                    self.SWA_PAGES,
                    None,
                    self._infos(prefills),
                )
            for prefill in prefills:
                for _ in decodes:
                    prefill._on_reshard_req(
                        self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_REQ)
                    )
                self.assertEqual(len(prefill._prefill.reshard_req[self.ROOM]), 2)
                prefill.append_send_blocks(self.ROOM, self.PREFILL_PAGES[:2])
                prefill.offer_chunk(self.ROOM, 0, self.PREFILL_PAGES[:2], 6)
                prefill.append_send_blocks(self.ROOM, self.PREFILL_PAGES[2:4])
                prefill.offer_chunk(self.ROOM, 2, self.PREFILL_PAGES[2:4], 6)
                self.assertNotIn(self.ROOM, prefill._prefill.reshard_chunks)
                prefill.append_send_blocks(
                    self.ROOM, self.PREFILL_PAGES[4:], swa_block_ids=[90, 91]
                )
                prefill.finalize_send(self.ROOM, None)
            for decode in decodes:
                self.assertEqual(
                    [f[0] for f in self.inboxes[decode.endpoint].frames],
                    [MessageTag.RESHARD_BLOCKS] * len(prefills),
                )
                for _ in prefills:
                    decode._on_reshard_blocks(
                        self.inboxes[decode.endpoint].take(MessageTag.RESHARD_BLOCKS)
                    )
            for prefill in prefills:
                for _ in decodes:
                    prefill._on_reshard_ready(
                        self.inboxes[prefill.endpoint].take(MessageTag.RESHARD_READY)
                    )
                self.assertEqual(
                    prefill._prefill.reshard_final_pushed[self.ROOM], {0, 1}
                )
                self.assertTrue(prefill.is_send_done(self.ROOM))

    def test_closing_plans_wait_for_host_staging(self):
        """A rank seats a room's closing plan through the staging admission
        like a pull: a room that does not fit waits (no plan, no READY), is
        seated on a later poll once an earlier room settles, in arrival
        order, and a cleared waiter gives up its place in line."""
        prefills = [self._prefill(0), self._prefill(1)]
        decodes = [self._decode(0), self._decode(1)]
        decode = decodes[1]
        # Each rank's share of a six-page room is three pages: two rooms do
        # not fit a four-page pool together.
        for d in decodes:
            d._pulling = StagingAdmission(4, 256)
        rooms = {
            61: (self.PREFILL_PAGES, self.DECODE_PAGES, self.SWA_PAGES),
            62: ([36, 37, 38, 39, 40, 41], [20, 21, 22, 23, 6, 7], [52, 53]),
            63: ([42, 43, 44, 45, 46, 47], [24, 25, 26, 27, 8, 9], [54, 55]),
        }

        ready = {p.endpoint: [] for p in prefills}

        def drain_ready():
            for p in prefills:
                frames = self.inboxes[p.endpoint].take_all(MessageTag.RESHARD_READY)
                ready[p.endpoint] += [ReshardReady.from_frames(f).room for f in frames]

        def ready_rooms(prefill):
            drain_ready()
            rooms, ready[prefill.endpoint] = sorted(ready[prefill.endpoint]), []
            return rooms

        with patch.dict(sys.modules, _pb2_modules()):
            for room, (prefill_pages, decode_pages, swa_pages) in rooms.items():
                for d in decodes:
                    d.register_decode_request(
                        room, decode_pages, swa_pages, None, self._infos(prefills)
                    )
                drain_ready()  # the previous room's READYs precede these asks
                for p in prefills:
                    for _ in decodes:
                        p._on_reshard_req(
                            self.inboxes[p.endpoint].take(MessageTag.RESHARD_REQ)
                        )
                    p.append_send_blocks(room, prefill_pages, swa_block_ids=[90, 91])
                    p.finalize_send(room, None)
                for d in decodes:
                    for _ in prefills:
                        d._on_reshard_blocks(
                            self.inboxes[d.endpoint].take(MessageTag.RESHARD_BLOCKS)
                        )
            # Room 61 is seated on both ranks (one READY per rank per source);
            # 62 holds the line and 63 waits behind it.
            self.assertEqual(ready_rooms(prefills[0]), [61, 61])
            self.assertEqual(sorted(decode._reshard_plans), [61])
            self.assertEqual(decode._decode.reshard_waiting, {62, 63})
            self.assertEqual(decode._pulling.waiting, "62#dcp1")
            self.assertFalse(decode.maybe_start_pull(63))
            self.assertEqual(sorted(decode._reshard_plans), [61])
            # 61 settles on this rank: 63 is still behind 62, which the next
            # poll seats and tells the sources about.
            decode._engine._stats = ([], ["61#dcp1"], [])
            decode._swa_engine._stats = ([], ["swa:61"], [])
            decode._drain_engine()
            self.assertNotIn("61#dcp1", decode._pulling)
            decode.maybe_start_pull(63)
            self.assertEqual(sorted(decode._reshard_plans), [61])
            decode.maybe_start_pull(62)
            self.assertEqual(sorted(decode._reshard_plans), [61, 62])
            self.assertEqual(ready_rooms(prefills[0]), [62])
            self.assertIn("62#dcp1", decode._pulling)
            self.assertEqual(decode._decode.reshard_waiting, {63})
            # 63 now holds the line; clearing it frees the line.
            decode.maybe_start_pull(63)
            self.assertEqual(decode._pulling.waiting, "63#dcp1")
            decode.clear_room(63)
            self.assertIsNone(decode._pulling.waiting)
            self.assertEqual(decode._decode.reshard_waiting, set())

    def test_prefill_refuses_a_dcp_decode_of_another_attention_tp(self):
        prefill = self._prefill(0)
        decode = self._decode(1)
        req = ReshardReq(
            room=self.ROOM,
            decode_ip=decode.local_ip,
            decode_port=decode.rank_port,
            push_endpoint="raiden://127.0.0.1:1",
            decode_blocks=self.DECODE_PAGES,
            dst_attn_tp=1,  # attention TP 2 under DCP 2, prefill is TP 4
            decode_swa_blocks=[],
            swa_push_endpoint="",
            dcp_size=2,
            dcp_rank=1,
        )
        prefill._on_reshard_req(req.to_frames())
        self.assertNotIn(self.ROOM, prefill._prefill.reshard_req)
        self.inboxes[decode.endpoint].take(MessageTag.ROOM_FAILED)
        self.assertIn("cannot take KV", prefill.failure_records[self.ROOM])

    def test_wire_messages_round_trip(self):
        for message in (
            ReshardReq(1, "127.0.0.1", 1234, "p", [2, 3], 2, [4], "s", 2, 1),
            ReshardReq(1, "127.0.0.1", 1234, "p", [2, 3], 2, [4], "s"),
            ReshardReady(1, [0, 2], 1, 99),
            ReshardReady(1),
        ):
            self.assertEqual(type(message).from_frames(message.to_frames()), message)
        with self.assertRaisesRegex(ValueError, "dcp_rank/dcp_size"):
            ReshardReq(1, "127.0.0.1", 1234, "p", [2, 3], 2, [4], "s", 2, 2)
        self.assertEqual(ReshardBlocks(1, [2], 0, b"", [3]).prefill_swa_blocks, [3])


if __name__ == "__main__":
    unittest.main()
