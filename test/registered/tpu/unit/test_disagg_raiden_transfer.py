"""Protocol tests for the raiden P/D rendezvous (``srt/disaggregation/raiden``).

The raiden backend is consumer-pull: decode asks prefill for a room's pages,
prefill replies with its block ids plus a raiden endpoint, and decode then issues
the read. Raiden's DMA engine moves the bytes; SGLang only owns the rendezvous.
These tests cover the part SGLang owns -- the handshake, its ordering, the aux
sidecar, and what gets registered with the engine -- with the engine faked, so
they need no TPU and no tpu_raiden. The on-device transfer itself is covered by
``test/manual/tpu/test_tpu_raiden_roundtrip_ondevice.py``.

The companion file ``test_disagg_tpu_transfer.py`` covers the ZMQ (tpu)
connector, which raiden is replacing. The two share very little: that one drives
a push protocol whose frames carry the KV bytes, this one a pull handshake whose
frames only carry block ids.

Deliberately not duplicated here, because they already have dedicated tests:
reshard geometry and placement (``test/manual/kv_transfer/test_reshard_schedule``),
the byte-level wire helpers (``test_raiden_serde``), and host staging admission
(``test_raiden_staging_admission``).
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from google.protobuf.wrappers_pb2 import Int64Value

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import (
    RaidenKVCacheGroups,
    get_raiden_kv_caches,
)
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import (
    Blocks,
    Chunk,
    MessageTag,
    PullReq,
    ReshardBlocks,
    ReshardChunk,
    ReshardChunkReady,
    ReshardReady,
    ReshardReq,
)
from sglang_tpu.mamba_pool import TpuMambaPool


def _polled(sent=(), received=(), failed=(), completed_ns=None):
    """Fake poll_stats_with_details() result for the given req ids."""

    def settled(ids):
        return [
            SimpleNamespace(
                req_id=r,
                completed_ns=completed_ns,
                code=None,
                phase=None,
                peer=None,
                elapsed_ns=None,
                reason=None,
            )
            for r in ids
        ]

    return SimpleNamespace(
        sent=settled(sent), received=settled(received), failed=settled(failed)
    )


class _Inbox:
    """Stand-in for a manager's bound PULL socket.

    Frames are collected rather than dispatched: the tests call the handlers
    directly so ordering is explicit, instead of racing the listener thread.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.frames: list[list[bytes]] = []

    def deliver(self, frames: list[bytes]) -> None:
        self.frames.append(frames)

    def take(self, tag: MessageTag) -> list[bytes]:
        """Pop the oldest frame list, asserting it carries ``tag``."""
        if not self.frames:
            raise AssertionError(f"expected a {tag.name} on {self.endpoint}, got none")
        frames = self.frames.pop(0)
        if frames[0] != tag:
            raise AssertionError(f"expected {tag.name}, got {frames[0]!r}")
        return frames


class _LoopbackSender:
    """Stand-in for a PUSH socket. Resolves its peer per send, so managers can
    be built in any order."""

    def __init__(self, endpoint: str, inboxes: dict[str, _Inbox]):
        self.endpoint = endpoint
        self._inboxes = inboxes

    def send_multipart(self, msg, copy: bool = True, track: bool = False):
        inbox = self._inboxes.get(self.endpoint)
        if inbox is None:
            raise AssertionError(f"no receiver bound to {self.endpoint}")
        inbox.deliver([bytes(part) for part in msg])
        return None


class _FakeRaidenEngine:
    """Records the six engine calls the manager makes, and reports nothing
    complete unless a test says so."""

    def __init__(self):
        self.reads: list[tuple[str, int, list[int]]] = []
        self.started: list[tuple] = []
        self.plans: list[tuple] = []
        self.pushes: list[tuple] = []
        self._stats: list = []

    def register_read(self, req_id, uuid, block_ids):
        self.reads.append((req_id, uuid, list(block_ids)))
        return True

    def start_read(self, *args, **kwargs):
        self.started.append((args, kwargs))

    def register_active_plan(self, *args, **kwargs):
        self.plans.append((args, kwargs))

    def unregister_active_plan(self, *args, **kwargs):
        self.plans.append((args, kwargs))

    def push_registered_plan(self, *args, **kwargs):
        self.pushes.append((args, kwargs))

    def poll_stats_with_details(self):
        stats, self._stats = self._stats, []
        return _polled(*stats)


class _AuxBuffers:
    """Host-resident aux slots, addressed the way KVArgs exposes them."""

    def __init__(self, rows: int = 8, cols: int = 16):
        self.tensor = torch.zeros(rows, cols, dtype=torch.int32)

    @property
    def ptrs(self) -> list[int]:
        return [self.tensor.data_ptr()]

    @property
    def item_lens(self) -> list[int]:
        return [self.tensor[0].nbytes]


class RaidenRendezvousTestCase(CustomTestCase):
    """Builds real RaidenKVManagers over an in-process loopback.

    Only ``CommonKVManager.__init__`` is stubbed -- raiden's own ``__init__``
    runs for real, so the ~60 fields it sets stay in sync with production
    without this file enumerating them.
    """

    def setUp(self):
        self.inboxes: dict[str, _Inbox] = {}
        self._next_port = 41000

    def _manager(self, mode, *, attn_tp_rank=0, attn_tp_size=1, page_size=4):
        port = self._next_port
        self._next_port += 1
        endpoint = f"tcp://127.0.0.1:{port}"
        inbox = _Inbox(endpoint)
        self.inboxes[endpoint] = inbox
        aux = _AuxBuffers()
        inboxes = self.inboxes

        def _fake_common_init(m, args, disaggregation_mode, server_args, is_mla=False):
            m.kv_args = args
            m.server_args = server_args
            m.disaggregation_mode = disaggregation_mode
            m.attn_tp_size = attn_tp_size
            m.attn_tp_rank = attn_tp_rank
            m.attn_dp_rank = 0
            m.attn_dp_size = 1
            m.attn_cp_rank = 0
            m.attn_cp_size = 1
            m.system_dp_rank = 0
            m.system_dp_size = 1
            m.pp_rank = 0
            m.pp_size = 1
            m.local_ip = "127.0.0.1"
            m.rank_port = port
            m.server_socket = inbox
            m._socket_cache = {}
            m._socket_lock = threading.Lock()
            m._socket_send_locks = {}
            m.request_status = {}
            m.failure_records = {}
            m.failure_lock = threading.Lock()

            def _connect(ep, is_ipv6=False):
                # The real _connect guarantees a cached socket always has its
                # endpoint send lock, and the send path indexes the lock.
                m._socket_send_locks.setdefault(ep, threading.Lock())
                return m._socket_cache.setdefault(ep, _LoopbackSender(ep, inboxes))

            m._connect = _connect

        kv_args = SimpleNamespace(
            aux_data_ptrs=aux.ptrs,
            aux_item_lens=aux.item_lens,
            page_size=page_size,
            total_kv_head_num=8,
            sliding_window_size=None,
            engine_rank=attn_tp_rank,
            state_types=[],
        )
        server_args = SimpleNamespace(context_length=1024)

        # _listen is stubbed to a no-op: its daemon thread would otherwise race
        # the tests for the inbox. Handlers are driven directly instead. The
        # decode heartbeat checker is stubbed the same way -- the stubbed
        # common init carries none of the fields its thread reads.
        with (
            patch.dict(os.environ, {"SGLANG_RAIDEN_KV_PORT": "32000"}),
            patch.object(CommonKVManager, "__init__", _fake_common_init),
            patch.object(RaidenKVManager, "_listen", lambda self: None),
            patch.object(
                CommonKVManager, "_start_heartbeat_checker_thread", lambda self: None
            ),
        ):
            m = RaidenKVManager(kv_args, mode, server_args)

        m._engine = _FakeRaidenEngine()
        m._full_engine_max_blocks = 256
        m._pulling = StagingAdmission(256)
        m.raiden_endpoint = f"raiden://127.0.0.1:{port + 1000}"
        m.aux = aux
        m.endpoint = endpoint
        self.addCleanup(self._release, m)
        return m

    @staticmethod
    def _release(m) -> None:
        # __init__ creates _push_executor only on the prefill branch, so a
        # decode manager genuinely has no such attribute to shut down.
        executor = m.__dict__.get("_push_executor")
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _prefill(self, **kw):
        return self._manager(DisaggregationMode.PREFILL, **kw)

    def _decode(self, **kw):
        return self._manager(DisaggregationMode.DECODE, **kw)

    @staticmethod
    def _bootstrap_info(prefill):
        return [{"rank_ip": "127.0.0.1", "rank_port": prefill.rank_port}]

    @staticmethod
    def _pull_req_frames(decode, room, pages, swa_pages=0):
        """Build the typed PULL_REQ frames sent by a decode rank."""
        return PullReq(
            room=room,
            decode_ip=decode.local_ip,
            decode_port=decode.rank_port,
            decode_pages=pages,
            decode_swa_pages=swa_pages,
        ).to_frames()


class TestRaidenPullHandshake(RaidenRendezvousTestCase):
    def test_chunk_transfer_id_fits_positive_int64(self):
        for upper_bound in (False, True):
            with self.subTest(upper_bound=upper_bound):
                prefill, decode = self._prefill(), self._decode()
                prefill._on_pull_req(self._pull_req_frames(decode, 7000, 2))
                with patch(
                    "secrets.randbelow",
                    side_effect=lambda bound: bound - 1 if upper_bound else 0,
                ):
                    prefill._announce_chunk(7000, 0, [1], total_pages=2)

                chunk = Chunk.from_frames(
                    self.inboxes[decode.endpoint].take(MessageTag.CHUNK)
                )
                self.assertGreater(chunk.uuid, 0)
                self.assertEqual(Int64Value(value=chunk.uuid).value, chunk.uuid)

    def test_chunks_from_different_prefills_have_distinct_transfer_ids(self):
        decode = self._decode()
        transfer_ids = []
        for room in (7000, 7001):
            prefill = self._prefill(attn_tp_rank=0)
            prefill._on_pull_req(self._pull_req_frames(decode, room, 2))
            prefill._announce_chunk(room, 0, [1], total_pages=2)

            chunk = Chunk.from_frames(
                self.inboxes[decode.endpoint].take(MessageTag.CHUNK)
            )
            self.assertEqual(chunk.req_id, f"{room}#0")
            self.assertEqual(prefill._engine.reads, [(chunk.req_id, chunk.uuid, [1])])
            transfer_ids.append(chunk.uuid)

        # The receiving native engine keys all peers' reads by UUID alone.
        self.assertNotEqual(*transfer_ids)

    def test_decode_request_emits_a_well_formed_pull_req(self):
        """The decode names itself and its page count so the prefill can size
        the reply; the typed message schema is the wire contract."""
        prefill, decode = self._prefill(), self._decode()
        room = 7001
        local_blocks = [4, 9, 2]

        decode.register_decode_request(
            room,
            local_blocks,
            [],
            aux_index=None,
            bootstrap_infos=self._bootstrap_info(prefill),
        )

        frames = self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ)
        req = PullReq.from_frames(frames)
        self.assertEqual(req.room, room)
        self.assertEqual(req.decode_ip, decode.local_ip)
        self.assertEqual(req.decode_port, decode.rank_port)
        self.assertEqual(req.decode_pages, len(local_blocks))

    def test_reply_waits_for_both_the_request_and_the_finalize(self):
        """Neither half alone may trigger BLOCKS: the prefill needs the decode's
        address, and the decode must not be told to read a partial page set."""
        prefill, decode = self._prefill(), self._decode()
        room, blocks = 7002, [1, 2, 3]

        prefill.append_send_blocks(room, blocks)
        prefill._maybe_reply(room)
        self.assertEqual(
            self.inboxes[decode.endpoint].frames,
            [],
            "replied before the decode asked",
        )

        prefill._on_pull_req(self._pull_req_frames(decode, room, len(blocks)))
        prefill.finalize_send(room, aux_index=None)
        self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)

    def test_reply_is_sent_once_when_the_request_arrives_last(self):
        """The two halves race in production, so the reverse order must behave
        identically -- and must not reply twice."""
        prefill, decode = self._prefill(), self._decode()
        room, blocks = 7003, [5, 6]

        prefill.append_send_blocks(room, blocks)
        prefill.finalize_send(room, aux_index=None)
        self.assertEqual(self.inboxes[decode.endpoint].frames, [])

        prefill._on_pull_req(self._pull_req_frames(decode, room, len(blocks)))
        self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)

        prefill._maybe_reply(room)
        prefill._on_pull_req(self._pull_req_frames(decode, room, len(blocks)))
        self.assertEqual(
            self.inboxes[decode.endpoint].frames, [], "room replied more than once"
        )

    def test_reply_carries_the_blocks_in_accumulation_order(self):
        """Block ids index KV pages in token order; reordering them silently
        corrupts the decode's cache, so the reply must preserve the order the
        chunks were appended in."""
        prefill, decode = self._prefill(), self._decode()
        room = 7004

        self.assertEqual(prefill.append_send_blocks(room, [11, 12]), 0)
        self.assertEqual(prefill.append_send_blocks(room, [13]), 2)
        prefill._on_pull_req(self._pull_req_frames(decode, room, 3))
        prefill.finalize_send(room, aux_index=None)

        frames = self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)
        reply = Blocks.from_frames(frames)
        self.assertEqual(reply.room, room)
        self.assertEqual(reply.blocks, [11, 12, 13])
        self.assertEqual(reply.endpoint, prefill.raiden_endpoint)

    def test_reply_registers_exactly_the_blocks_it_announced(self):
        """Raiden blocks the consumer's read until a matching producer
        registration exists, so the reply and the registration must agree."""
        prefill, decode = self._prefill(), self._decode()
        room, blocks = 7005, [21, 22, 23]

        prefill.append_send_blocks(room, blocks)
        prefill._on_pull_req(self._pull_req_frames(decode, room, len(blocks)))
        prefill.finalize_send(room, aux_index=None)

        frames = self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)
        self.assertEqual(prefill._engine.reads, [(str(room), room, blocks)])
        self.assertEqual(Blocks.from_frames(frames).blocks, blocks)

    def test_malformed_pull_req_is_rejected_before_it_is_indexed(self):
        """A missing JSON payload must produce a protocol validation error."""
        prefill, decode = self._prefill(), self._decode()
        frames = self._pull_req_frames(decode, 7006, 1)[:-1]

        with self.assertRaises(ValueError):
            prefill._on_pull_req(frames)


class TestRaidenAuxSidecar(RaidenRendezvousTestCase):
    """The aux row (first sampled token, logprobs, bootstrap room) rides the
    rendezvous rather than raiden, so it is SGLang's to get right."""

    def test_aux_row_round_trips_from_prefill_slot_to_decode_slot(self):
        prefill, decode = self._prefill(), self._decode()
        room, src_index, dst_index = 7101, 6, 3
        prefill.aux.tensor[src_index] = torch.arange(16, dtype=torch.int32) + 100

        decode.register_decode_request(
            room,
            [1],
            [],
            aux_index=dst_index,
            bootstrap_infos=self._bootstrap_info(prefill),
        )
        prefill._on_pull_req(self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ))
        prefill.append_send_blocks(room, [1])
        prefill.finalize_send(room, aux_index=src_index)

        decode._on_blocks(self.inboxes[decode.endpoint].take(MessageTag.BLOCKS))

        self.assertTrue(
            torch.equal(decode.aux.tensor[dst_index], prefill.aux.tensor[src_index]),
            "aux row did not land in the decode's slot",
        )

    def test_untouched_decode_aux_rows_stay_zero(self):
        """Guards the slot arithmetic: writing row N must not spill into N+-1."""
        prefill, decode = self._prefill(), self._decode()
        room, dst_index = 7102, 3
        prefill.aux.tensor[0] = torch.arange(16, dtype=torch.int32) + 1

        decode.register_decode_request(
            room,
            [1],
            [],
            aux_index=dst_index,
            bootstrap_infos=self._bootstrap_info(prefill),
        )
        prefill._on_pull_req(self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ))
        prefill.append_send_blocks(room, [1])
        prefill.finalize_send(room, aux_index=0)
        decode._on_blocks(self.inboxes[decode.endpoint].take(MessageTag.BLOCKS))

        for row in range(decode.aux.tensor.shape[0]):
            if row == dst_index:
                continue
            self.assertTrue(
                torch.equal(decode.aux.tensor[row], torch.zeros(16, dtype=torch.int32)),
                f"aux write spilled into row {row}",
            )

    def test_a_room_without_aux_sends_an_empty_blob(self):
        """Not every request carries aux; the reply must still be well formed."""
        prefill, decode = self._prefill(), self._decode()
        room = 7103

        prefill.append_send_blocks(room, [1])
        prefill._on_pull_req(self._pull_req_frames(decode, room, 1))
        prefill.finalize_send(room, aux_index=None)

        frames = self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)
        self.assertEqual(Blocks.from_frames(frames).aux_blob, b"")


class TestRaidenDecodeBookkeeping(RaidenRendezvousTestCase):
    def test_decode_drains_completed_receive(self):
        """Decode must process engine completions without prefill-only state."""
        decode = self._decode()
        room = 7200
        decode._engine._stats = ([], [str(room)], [])

        decode._drain_engine()

        self.assertTrue(decode.is_recv_done(room))

    def test_blocks_reply_records_the_remote_side_for_the_read(self):
        """The decode issues its read against these, so they must survive the
        hop unchanged and in order."""
        prefill, decode = self._prefill(), self._decode()
        room, blocks = 7201, [31, 32, 33]

        decode.register_decode_request(
            room,
            [1, 2, 3],
            [],
            aux_index=None,
            bootstrap_infos=self._bootstrap_info(prefill),
        )
        prefill._on_pull_req(self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ))
        prefill.append_send_blocks(room, blocks)
        prefill.finalize_send(room, aux_index=None)

        decode._on_blocks(self.inboxes[decode.endpoint].take(MessageTag.BLOCKS))

        self.assertEqual(decode._decode.remote_blocks[room], blocks)
        self.assertEqual(decode._decode.remote_endpoint[room], prefill.raiden_endpoint)

    def test_decode_records_its_own_pages_for_the_pull(self):
        prefill, decode = self._prefill(), self._decode()
        room, local_blocks = 7202, [41, 42]

        decode.register_decode_request(
            room,
            local_blocks,
            [],
            aux_index=None,
            bootstrap_infos=self._bootstrap_info(prefill),
        )

        self.assertEqual(decode._decode.local_blocks[room], local_blocks)

    def test_prefill_records_the_decode_address_and_page_count(self):
        """_maybe_reply gates on both, and the page count sizes the pull
        window for chunked transfers."""
        prefill, decode = self._prefill(), self._decode()
        room, pages, swa_pages = 7203, 5, 2

        prefill._on_pull_req(self._pull_req_frames(decode, room, pages, swa_pages))

        self.assertEqual(
            prefill._prefill.decode_addr[room], ("127.0.0.1", decode.rank_port)
        )
        self.assertEqual(prefill._prefill.decode_pages[room], pages)
        self.assertEqual(prefill._prefill.decode_swa_pages[room], swa_pages)

    def test_chunk_arriving_after_blocks_is_dropped(self):
        """When BLOCKS overtakes CHUNK across a ZMQ disconnect/reconnect,
        Decode must drop the late chunk so it never queues a duplicate pull or
        reads from already-settled/freed prefill KV memory (issue #656)."""
        prefill, decode = self._prefill(), self._decode()
        decode.chunked_transfer = True
        # Inspects the queue, so reads must wait for the poll.
        decode._listener_start_read = False
        room = 7204
        local_blocks = [101, 102, 103, 104]

        decode.register_decode_request(
            room,
            local_blocks,
            [],
            aux_index=None,
            bootstrap_infos=self._bootstrap_info(prefill),
        )
        prefill._on_pull_req(self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ))

        # Prefill announces chunk 0 covering 2 pages [11, 12] out of 4 total pages.
        prefill.append_send_blocks(room, [11, 12])
        prefill._announce_chunk(room, 0, [11, 12], total_pages=4)
        chunk_frames = self.inboxes[decode.endpoint].take(MessageTag.CHUNK)

        # Prefill finalizes send with all 4 pages [11, 12, 13, 14].
        prefill.append_send_blocks(room, [13, 14])
        prefill.finalize_send(room, aux_index=None)
        blocks_frames = self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)

        # Reordering simulation: BLOCKS arrives first at decode.
        decode._dispatch(blocks_frames)
        self.assertIn(room, decode._decode.final_seen)
        # Because chunk 0 was not received yet, BLOCKS treated all 4 pages as missing.
        self.assertEqual(len(decode._decode.chunk_queue[room]), 1)
        req_id, _, missing, missing_blocks, _ = decode._decode.chunk_queue[room][0]
        self.assertEqual(req_id, str(room))
        self.assertEqual(missing, [0, 1, 2, 3])
        self.assertEqual(missing_blocks, [11, 12, 13, 14])

        # Decode starts pull for the closing read (str(room)).
        self.assertTrue(decode._start_chunked_pull(room))

        # Complete the transfer for str(room) on both sides.
        decode._engine._stats = ([], [str(room)], [])
        decode._drain_engine()
        prefill._engine._stats = ([str(room)], [], [])
        prefill._drain_engine()

        self.assertTrue(decode.is_recv_done(room))
        self.assertTrue(prefill.is_send_done(room))

        # Late CHUNK 0 arrives after BLOCKS has already completed.
        decode._dispatch(chunk_frames)

        # The late chunk must be discarded: chunk_queue remains empty and no new read is started.
        self.assertEqual(decode._decode.chunk_queue.get(room, []), [])
        self.assertEqual(len(decode._engine.started), 1)

    def _announced_chunked_room(self, room):
        """A chunked room with chunk 0 announced (pages 0-1 of 4) and prefill
        finalized. Returns (prefill, decode, CHUNK frames, BLOCKS frames), both
        still undelivered to the decode."""
        prefill, decode = self._prefill(), self._decode()
        decode.chunked_transfer = True
        decode.register_decode_request(
            room,
            [101, 102, 103, 104],
            [],
            aux_index=None,
            bootstrap_infos=self._bootstrap_info(prefill),
        )
        prefill._on_pull_req(self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ))
        prefill.append_send_blocks(room, [11, 12])
        prefill._announce_chunk(room, 0, [11, 12], total_pages=4)
        chunk_frames = self.inboxes[decode.endpoint].take(MessageTag.CHUNK)
        prefill.append_send_blocks(room, [13, 14])
        prefill.finalize_send(room, aux_index=None)
        blocks_frames = self.inboxes[decode.endpoint].take(MessageTag.BLOCKS)
        return prefill, decode, chunk_frames, blocks_frames

    def test_duplicate_chunk_is_queued_once(self):
        """A CHUNK delivered twice must queue one read, not a second pull of
        the same pages."""
        _, decode, chunk_frames, _ = self._announced_chunked_room(7205)
        # Inspects the queue, so reads must wait for the poll.
        decode._listener_start_read = False

        decode._dispatch(chunk_frames)
        decode._dispatch(chunk_frames)

        self.assertEqual(len(decode._decode.chunk_queue[7205]), 1)
        self.assertEqual(decode._decode.announced_spans[7205], [(0, 2)])

    def test_duplicate_blocks_does_not_requeue_the_closing_read(self):
        """A BLOCKS delivered again after the closing read was issued must not
        queue a second closing read of the uncovered pages."""
        room = 7207
        _, decode, chunk_frames, blocks_frames = self._announced_chunked_room(room)
        decode._dispatch(chunk_frames)
        decode._dispatch(blocks_frames)
        # Chunk 0 read, then the closing read of pages 2-3.
        self.assertTrue(decode._start_chunked_pull(room))
        decode._engine._stats = ([], [f"{room}#0"], [])
        decode._drain_engine()
        self.assertTrue(decode._start_chunked_pull(room))
        self.assertEqual(decode._decode.chunk_reads[room], [f"{room}#0", str(room)])

        decode._dispatch(blocks_frames)

        self.assertEqual(decode._decode.chunk_queue.get(room, []), [])
        self.assertEqual(len(decode._engine.started), 2)

    def test_superseded_chunk_read_protocol_driven(self):
        """Drive prefill through _announce_chunk and finalize_send, settle the closing
        read, and verify superseded chunk timeout is safely discarded."""
        prefill, decode = self._prefill(), self._decode()
        prefill.chunked_transfer = True
        room = 7206

        prefill._on_pull_req(self._pull_req_frames(decode, room, 4))
        prefill.append_send_blocks(room, [11, 12])
        prefill._announce_chunk(room, 0, [11, 12], total_pages=4)
        self.assertIn(f"{room}#0", prefill._send_reads[room])

        prefill.append_send_blocks(room, [13, 14])
        prefill.finalize_send(room, aux_index=None)
        self.assertIn(str(room), prefill._send_reads[room])

        # Closing read completes on native engine
        prefill._engine._stats = ([str(room)], [], [])
        prefill._drain_engine()

        self.assertNotIn(room, prefill._send_reads)
        self.assertIn(f"{room}#0", prefill._superseded_sender_reads)
        self.assertFalse(prefill.sender_has_active_io(room))
        self.assertTrue(prefill.is_send_done(room))

        # Chunk 0 times out on native engine
        prefill._engine._stats = ([], [], [f"{room}#0"])
        prefill._drain_engine()

        self.assertNotIn(f"{room}#0", prefill._superseded_sender_reads)
        self.assertNotIn(f"{room}#0", prefill._failed)
        self.assertFalse(prefill.is_failed(room))


def _fake_tpu_sync(engine_cls) -> dict:
    """sys.modules entries that resolve raiden's deferred KVCacheManager import."""
    api = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
    api.KVCacheManager = engine_cls
    return {
        "tpu_sync": types.ModuleType("tpu_sync"),
        "tpu_sync.api": types.ModuleType("tpu_sync.api"),
        "tpu_sync.api.torch": types.ModuleType("tpu_sync.api.torch"),
        "tpu_sync.api.torch.kv_cache_manager": api,
    }


def _mamba_state_indices(slot: int) -> list:
    """state_indices as core builds them for a [MAMBA] state_types list."""
    return [[np.array(slot)]]


class RaidenStateTestCase(RaidenRendezvousTestCase):
    """Managers with a GDN state engine; a slot's committed state is its own row."""

    def _with_state(self, m):
        m._state_engine = _FakeRaidenEngine()
        m._state_engine_max_blocks = 1
        m._committed_state_row = lambda slot: slot
        m.state_raiden_endpoint = f"raiden-state://127.0.0.1:{m.rank_port}"
        m.kv_args.state_types = [StateType.MAMBA]
        m._pulling = StagingAdmission(256, None, 8)
        return m

    def _state_pair(self, *, chunked: bool = False):
        prefill = self._with_state(self._prefill())
        decode = self._with_state(self._decode())
        prefill.chunked_transfer = decode.chunked_transfer = chunked
        # Reads are issued from the poll so tests see each step.
        decode._listener_start_read = False
        return prefill, decode

    def _hand_off(self, prefill, decode, room, *, local_blocks, prefill_slot=7):
        """Drive the handshake up to the decode holding BLOCKS; decode slot 5."""
        decode.register_decode_request(
            room,
            local_blocks,
            [],
            aux_index=None,
            bootstrap_infos=self._bootstrap_info(prefill),
            local_state_rows=decode._state_rows_from_state_indices(
                _mamba_state_indices(5)
            ),
        )
        prefill._on_pull_req(self.inboxes[prefill.endpoint].take(MessageTag.PULL_REQ))
        prefill.append_send_blocks(
            room,
            [31, 32, 33],
            state_block_ids=prefill._state_rows_from_state_indices(
                _mamba_state_indices(prefill_slot)
            ),
        )
        prefill.finalize_send(room, aux_index=None)
        decode._on_blocks(self.inboxes[decode.endpoint].take(MessageTag.BLOCKS))


class TestRaidenStateTransfer(RaidenStateTestCase):
    def test_state_read_moves_each_sides_committed_row(self):
        """Prefill slot 7 is row 7 and decode slot 5 is row 5, whatever the width."""
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                prefill, decode = self._state_pair(chunked=chunked)
                room = 7300 + chunked

                self._hand_off(prefill, decode, room, local_blocks=[1, 2, 3])

                self.assertEqual(
                    prefill._state_engine.reads, [(f"state:{room}", room, [7])]
                )
                self.assertTrue(decode.maybe_start_pull(room))
                self.assertEqual(
                    [args for args, _ in decode._state_engine.started],
                    [(f"state:{room}", room, prefill.state_raiden_endpoint, [7], [5])],
                )
                self.assertEqual(
                    [args[0] for args, _ in decode._engine.started], [str(room)]
                )

    def test_full_prefix_hit_still_moves_state(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                prefill, decode = self._state_pair(chunked=chunked)
                room = 7310 + chunked

                self._hand_off(prefill, decode, room, local_blocks=[])

                self.assertEqual(
                    [r[0] for r in prefill._engine.reads + prefill._state_engine.reads],
                    [f"state:{room}"],
                )
                self.assertTrue(decode.maybe_start_pull(room))
                self.assertEqual(decode._engine.started, [])
                self.assertEqual(
                    [args for args, _ in decode._state_engine.started],
                    [(f"state:{room}", room, prefill.state_raiden_endpoint, [7], [5])],
                )
                # Polling again must not issue the read twice.
                self.assertTrue(decode.maybe_start_pull(room))
                self.assertEqual(len(decode._state_engine.started), 1)
                self.assertFalse(decode.is_recv_done(room))
                decode._state_engine._stats = ([], [f"state:{room}"], [])
                decode._drain_engine()
                self.assertTrue(decode.is_recv_done(room))

    def test_send_completes_only_once_the_state_read_is_served(self):
        prefill, decode = self._state_pair()
        room = 7320
        self._hand_off(prefill, decode, room, local_blocks=[1, 2, 3])

        prefill._engine._stats = ([str(room)], [], [], 1_000_000_000)
        prefill._drain_engine()
        self.assertFalse(prefill.is_send_done(room))
        self.assertTrue(prefill.sender_has_active_io(room))

        prefill._state_engine._stats = ([f"state:{room}"], [], [], 2_000_000_000)
        prefill._drain_engine()
        self.assertTrue(prefill.is_send_done(room))
        self.assertFalse(prefill.sender_has_active_io(room))
        self.assertEqual(prefill.send_done_time(room), 2.0)

    def test_receive_completes_only_once_the_state_read_lands(self):
        prefill, decode = self._state_pair()
        room = 7321
        self._hand_off(prefill, decode, room, local_blocks=[1, 2, 3])
        self.assertTrue(decode.maybe_start_pull(room))

        decode._engine._stats = ([], [str(room)], [])
        decode._drain_engine()
        self.assertFalse(decode.is_recv_done(room))
        self.assertTrue(decode.room_has_active_io(room))
        self.assertEqual(decode._pulling.state_pages(), 1)

        decode._state_engine._stats = ([], [f"state:{room}"], [])
        decode._drain_engine()
        self.assertTrue(decode.is_recv_done(room))
        self.assertFalse(decode.room_has_active_io(room))
        self.assertEqual(len(decode._pulling), 0)

    def test_failed_state_read_fails_the_room(self):
        prefill, decode = self._state_pair()
        room = 7322
        # send_metadata opens the room; only a live room keeps its failure reason.
        decode.request_status[room] = KVPoll.WaitingForInput
        self._hand_off(prefill, decode, room, local_blocks=[1, 2, 3])
        self.assertTrue(decode.maybe_start_pull(room))
        self.assertEqual(RaidenKVManager._room_of(f"state:{room}"), room)

        decode._state_engine._stats = ([], [], [f"state:{room}"])
        decode._drain_engine()

        self.assertTrue(decode.is_failed(room))
        self.assertIn(f"state:{room}", decode.failure_records[room])
        # The full read still holds the room's staging.
        self.assertTrue(decode.room_has_active_io(room))

    def test_clearing_a_room_holds_its_state_read_until_it_settles(self):
        prefill, decode = self._state_pair()
        room = 7323
        self._hand_off(prefill, decode, room, local_blocks=[1, 2, 3])
        self.assertTrue(decode.maybe_start_pull(room))
        decode._engine._stats = ([], [str(room)], [])
        decode._drain_engine()

        decode.clear_room(room)
        self.assertTrue(decode.room_has_active_io(room))
        self.assertNotIn(room, decode._decode.remote_state_rows)
        self.assertNotIn(room, decode._decode.local_state_rows)

        decode._state_engine._stats = ([], [f"state:{room}"], [])
        decode._drain_engine()
        self.assertEqual(len(decode._pulling), 0)
        self.assertFalse(decode.room_has_active_io(room))
        self.assertEqual(decode._completed_recv, set())

        prefill.clear_room(room)
        self.assertNotIn(room, prefill._prefill.send_state_rows)

    def test_push_resharding_with_state_fails_closed(self):
        prefill, decode = self._state_pair()
        infos = self._bootstrap_info(prefill) * 2
        with self.assertRaisesRegex(NotImplementedError, "GDN"):
            decode.register_decode_request(
                7330, [1], [], aux_index=None, bootstrap_infos=infos
            )
        with self.assertRaisesRegex(NotImplementedError, "GDN"):
            prefill._on_reshard_req(
                ReshardReq(
                    room=7330,
                    decode_ip="127.0.0.1",
                    decode_port=decode.rank_port,
                    push_endpoint="tcp://push",
                    decode_blocks=[1],
                    dst_attn_tp=1,
                    decode_swa_blocks=[],
                    swa_push_endpoint="",
                ).to_frames()
            )

    def test_pipeline_transfer_with_state_fails_closed(self):
        decode = self._decode()
        decode._pipeline_enabled = True
        groups = RaidenKVCacheGroups(
            full=[torch.zeros(4, 2)],
            swa=[],
            state=[torch.zeros(3, 2)],
            committed_state_row=lambda slot: slot,
        )
        with self.assertRaisesRegex(NotImplementedError, "GDN"):
            decode.set_kv_caches(groups)


class TestRaidenStateEngine(RaidenRendezvousTestCase):
    def _build(self, *, attn_tp_rank, swa, state):
        m = self._prefill(attn_tp_rank=attn_tp_rank, attn_tp_size=2)
        built = []

        class Engine:
            def __init__(self, **kwargs):
                built.append(kwargs)

        m._engine = None
        m._kv_caches = [torch.zeros(9, 4)]
        m._swa_kv_caches = [torch.zeros(5, 4)] if swa else []
        m._state_kv_caches = (torch.zeros(6, 3), torch.zeros(6, 7)) if state else ()
        m._max_request_blocks = lambda: 8
        m._swa_max_request_blocks = lambda: 2
        m._num_slots = lambda kv_caches, max_blocks, engines=1, **_: 10 * engines
        with patch.dict(sys.modules, _fake_tpu_sync(Engine)):
            m._build_engine()
        return m, built

    def test_state_engine_is_built_only_with_state_tensors(self):
        m, built = self._build(attn_tp_rank=0, swa=False, state=False)
        self.assertEqual(len(built), 1)
        self.assertIsNone(m._state_engine)
        self.assertIsNone(m._pulling.state_capacity)

        m, built = self._build(attn_tp_rank=0, swa=False, state=True)
        self.assertEqual(len(built), 2)
        state = built[1]
        self.assertEqual([t.shape[0] for t in state["kv_caches"]], [6, 6])
        self.assertEqual((state["max_blocks"], state["num_slots"]), (1, 20))
        self.assertIsNotNone(m._state_engine)
        self.assertEqual(m._state_engine_max_blocks, 1)
        self.assertEqual(m._pulling.state_capacity, 20)

    def test_each_rank_reserves_six_ports(self):
        """Full, SWA and state engines at +0/+2/+4 of base + 6*local_rank."""
        m, built = self._build(attn_tp_rank=1, swa=True, state=True)
        self.assertEqual(m._control_port, 32006)
        self.assertEqual(
            [kw["local_control_port"] for kw in built], [32006, 32008, 32010]
        )
        self.assertEqual(m.raiden_endpoint, "127.0.0.1:32006")
        self.assertEqual(m.swa_raiden_endpoint, "127.0.0.1:32008")
        self.assertEqual(m.state_raiden_endpoint, "127.0.0.1:32010")


class TestRaidenStateGroups(CustomTestCase):
    def _hybrid_pool(self, *, mamba_pool):
        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        pool.start_layer = 0
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
        kv = {3: torch.zeros(9, 2), 7: torch.zeros(9, 2)}
        pool.get_fused_kv_buffer = lambda layer_id: kv[layer_id]
        pool.mamba_pool = mamba_pool
        return pool, kv

    def test_linear_attention_pool_contributes_its_state(self):
        mamba = TpuMambaPool.__new__(TpuMambaPool)
        mamba.checkpoint_width = 3
        state = [torch.zeros(12, 2, 1, 4), torch.zeros(12, 4, 2, 2)]
        mamba.get_state = lambda: state
        pool, kv = self._hybrid_pool(mamba_pool=mamba)

        groups = get_raiden_kv_caches(pool)

        self.assertEqual(groups.full, [kv[3], kv[7]])
        self.assertEqual(groups.full_layer_ids, [3, 7])
        self.assertEqual(groups.state, state)
        self.assertEqual(groups.committed_state_row(2), 2)

    def test_non_tpu_mamba_pool_contributes_no_state(self):
        pool, _ = self._hybrid_pool(mamba_pool=object())

        groups = get_raiden_kv_caches(pool)

        self.assertEqual(groups.state, [])
        self.assertIsNone(groups.committed_state_row)

    def test_state_tensors_must_share_one_row_count(self):
        mamba = TpuMambaPool.__new__(TpuMambaPool)
        mamba.checkpoint_width = 1
        mamba.get_state = lambda: [torch.zeros(4, 2), torch.zeros(5, 2)]
        pool, _ = self._hybrid_pool(mamba_pool=mamba)

        with self.assertRaisesRegex(RuntimeError, "uniform GDN state"):
            get_raiden_kv_caches(pool)


class TestRaidenMessageValidation(CustomTestCase):
    def test_all_message_types_round_trip(self):
        messages = [
            PullReq(1, "127.0.0.1", 1234, 2),
            Blocks(1, [2, 3], "tcp://kv", [4], "tcp://swa", b"aux"),
            Blocks(1, [2], "tcp://kv", [], "", b"", [8], "tcp://state"),
            Chunk(1, "req", 2, 0, 2, [3, 4], "tcp://kv"),
            ReshardReq(
                room=1,
                decode_ip="127.0.0.1",
                decode_port=1234,
                push_endpoint="tcp://push",
                decode_blocks=[2, 3],
                dst_attn_tp=2,
                decode_swa_blocks=[4],
                swa_push_endpoint="tcp://swa",
            ),
            ReshardBlocks(1, [2, 3], 0, b"aux", [4]),
            ReshardReady(1, [0, 2]),
            ReshardChunk(1, 0, 0, 2, [3, 4], 0),
            ReshardChunkReady(1, 0, 2),
        ]

        for message in messages:
            with self.subTest(message_type=type(message).__name__):
                self.assertEqual(
                    type(message).from_frames(message.to_frames()), message
                )

    def test_rejects_wrong_frame_count(self):
        with self.assertRaisesRegex(ValueError, "expected 2 frames, got 1"):
            PullReq.from_frames([MessageTag.PULL_REQ])

    def test_rejects_wrong_tag(self):
        message = PullReq(1, "127.0.0.1", 1234, 2)
        with self.assertRaisesRegex(ValueError, "expected tag"):
            PullReq.from_frames([MessageTag.BLOCKS, message.to_frames()[1]])

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            PullReq.from_frames([MessageTag.PULL_REQ, b'{"unexpected":1}'])

    def test_constructor_validates_ranges(self):
        with self.assertRaisesRegex(ValueError, "decode_port"):
            PullReq(room=1, decode_ip="127.0.0.1", decode_port=65536, decode_pages=1)

    def test_decode_validates_ranges_with_message_context(self):
        payload = (
            b'{"room":1,"decode_ip":"127.0.0.1","decode_port":65536,"decode_pages":1}'
        )
        with self.assertRaisesRegex(ValueError, "malformed PullReq payload"):
            PullReq.from_frames([MessageTag.PULL_REQ, payload])


if __name__ == "__main__":
    unittest.main()
