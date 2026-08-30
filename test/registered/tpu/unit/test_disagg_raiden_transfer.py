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
(``test_raiden_slot_admission``).
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit")

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.raiden.conn import RaidenKVManager
from sglang.srt.disaggregation.raiden.serde import MessageTag, unpack_blocks
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.test_utils import CustomTestCase

# Frame counts the handlers enforce via _require_frames. Duplicated from the
# production side on purpose: these are the wire contract, so a change on one
# side should have to be made deliberately on the other.
PULL_REQ_FRAMES = 5
BLOCKS_FRAMES = 7


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

    def start_read(self, *args, **kwargs):
        self.started.append((args, kwargs))

    def register_active_plan(self, *args, **kwargs):
        self.plans.append((args, kwargs))

    def unregister_active_plan(self, *args, **kwargs):
        self.plans.append((args, kwargs))

    def push_registered_plan(self, *args, **kwargs):
        self.pushes.append((args, kwargs))

    def poll_stats(self):
        stats, self._stats = self._stats, []
        return stats


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
            m.local_ip = "127.0.0.1"
            m.rank_port = port
            m.server_socket = inbox
            m._socket_cache = {}
            m._socket_lock = threading.Lock()
            m._socket_send_locks = {}
            m.request_status = {}

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
        )
        server_args = SimpleNamespace(context_length=1024)

        # _listen is stubbed to a no-op: its daemon thread would otherwise race
        # the tests for the inbox. Handlers are driven directly instead. The
        # decode heartbeat checker is stubbed the same way -- the stubbed
        # common init carries none of the fields its thread reads.
        with patch.object(CommonKVManager, "__init__", _fake_common_init), patch.object(
            RaidenKVManager, "_listen", lambda self: None
        ), patch.object(
            CommonKVManager, "_start_heartbeat_checker_thread", lambda self: None
        ):
            m = RaidenKVManager(kv_args, mode, server_args)

        m._engine = _FakeRaidenEngine()
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
    def _pull_req_frames(decode, room, pages):
        """The PULL_REQ a decode rank sends, as _on_pull_req expects it."""
        return [
            MessageTag.PULL_REQ,
            str(room).encode("ascii"),
            decode.local_ip.encode("ascii"),
            str(decode.rank_port).encode("ascii"),
            str(pages).encode("ascii"),
        ]


class TestRaidenPullHandshake(RaidenRendezvousTestCase):
    def test_decode_request_emits_a_well_formed_pull_req(self):
        """The decode names itself and its page count so the prefill can size
        the reply; the frame count is the contract _require_frames enforces."""
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
        self.assertEqual(len(frames), PULL_REQ_FRAMES)
        self.assertEqual(int(frames[1]), room)
        self.assertEqual(frames[2].decode(), decode.local_ip)
        self.assertEqual(int(frames[3]), decode.rank_port)
        self.assertEqual(int(frames[4]), len(local_blocks))

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
        self.assertEqual(len(frames), BLOCKS_FRAMES)
        self.assertEqual(int(frames[1]), room)
        self.assertEqual(unpack_blocks(frames[2]), [11, 12, 13])
        self.assertEqual(frames[3].decode(), prefill.raiden_endpoint)

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
        self.assertEqual(unpack_blocks(frames[2]), blocks)

    def test_malformed_pull_req_is_rejected_before_it_is_indexed(self):
        """A short header must raise rather than IndexError deep in the handler."""
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
        self.assertEqual(len(frames), BLOCKS_FRAMES)
        self.assertEqual(frames[6], b"")


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

        self.assertEqual(decode._remote_blocks[room], blocks)
        self.assertEqual(decode._remote_endpoint[room], prefill.raiden_endpoint)

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

        self.assertEqual(decode._local_blocks[room], local_blocks)

    def test_prefill_records_the_decode_address_and_page_count(self):
        """_maybe_reply gates on both, and the page count sizes the pull
        window for chunked transfers."""
        prefill, decode = self._prefill(), self._decode()
        room, pages = 7203, 5

        prefill._on_pull_req(self._pull_req_frames(decode, room, pages))

        self.assertEqual(prefill._decode_addr[room], ("127.0.0.1", decode.rank_port))
        self.assertEqual(prefill._decode_pages[room], pages)


if __name__ == "__main__":
    unittest.main()
