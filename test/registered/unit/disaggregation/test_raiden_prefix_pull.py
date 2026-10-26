"""Unit tests for the raiden decode-held prefix pull: the decode offers the
prompt prefix pages it holds and the prefill pulls the ones it lacks."""

import errno
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVSender
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.prefix_offer import (
    deepest_window_prefix,
    lock_window_prefix,
    window_tokens,
)
from sglang_tpu.disaggregation.raiden.prefix_pull import DecodePrefixPuller
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    PrefixOffer,
    PrefixPull,
    PrefixReady,
    PullReq,
    ReshardReq,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7
OTHER_ROOM = 8
PAGE = 4
PX = f"{ROOM}#px"
PREFILL_INFO = {"rank_ip": "10.0.0.1", "rank_port": 4444}
PREFILL_ADDR = _tcp_endpoint(ip="10.0.0.1", port=4444)
DECODE_ADDR = _tcp_endpoint(ip="10.0.0.2", port=5555)
DECODE_RAIDEN = "10.0.0.2:31020"
# SettledTransfer's failure fields, unset.
_UNREPORTED = dict(elapsed_ns=None, peer=None, code=None, reason=None, phase=None)


def _polled(sent=(), received=(), failed=(), completed_ns=None, details=None):
    """Fake poll_stats_with_details() result for the given req ids. ``details``
    maps a req id to the failure fields raiden reports for it."""

    def settled(ids):
        return [
            SimpleNamespace(
                req_id=r,
                **{
                    "completed_ns": completed_ns,
                    **_UNREPORTED,
                    **(details or {}).get(r, {}),
                },
            )
            for r in ids
        ]

    return SimpleNamespace(
        sent=settled(sent), received=settled(received), failed=settled(failed)
    )


def _sent_frames(mgr):
    """(endpoint, frames) of every message the manager sent."""
    return [call.args[:2] for call in mgr._send_multipart_locked.call_args_list]


def _activate_prefix_pull(mgr):
    """What prefix_pull_active reads: the pull enabled on one full-attention
    KV pool with a radix cache."""
    mgr.prefix_pull_enabled = True
    mgr._kv_caches = [object()]
    mgr._swa_kv_caches = []
    mgr._state_kv_caches = ()
    mgr.server_args = SimpleNamespace(disable_radix_cache=False)


def _prefill_manager():
    """A prefill-mode RaidenKVManager with the pull active, a mock engine and
    one host staging pool. A real RaidenKVSender can be built on it."""
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
    mgr._engine = Mock()
    mgr._engine.poll_stats_with_details.return_value = _polled()
    mgr._swa_engine = None
    mgr._reshard_slot_rooms = [None]
    mgr._reshard_slot_of = {}
    mgr._aborted = {}
    mgr._dynamic_host_staging = True
    mgr._full_engine_max_blocks = 16
    mgr._pulling = StagingAdmission(16)
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
    _activate_prefix_pull(mgr)
    mgr._send_multipart_locked = Mock()
    mgr._message_handlers = {
        PullReq.TAG: mgr._on_pull_req,
        ReshardReq.TAG: mgr._on_reshard_req,
        PrefixOffer.TAG: mgr._on_prefix_offer,
        PrefixReady.TAG: mgr._on_prefix_ready,
    }
    return mgr


def _sender(mgr, room=ROOM):
    with get_context().override_server_args(dp_size=1):
        return RaidenKVSender(mgr, "unused", room)


def _pull_req(prefix_tokens=8, prefix_offer=True):
    return PullReq(
        room=ROOM,
        decode_ip="10.0.0.2",
        decode_port=5555,
        decode_pages=4,
        prefix_tokens=prefix_tokens,
        prefix_offer=prefix_offer,
    ).to_frames()


def _decode_manager():
    """A decode-mode RaidenKVManager with the pull active and a mock engine."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr._engine = Mock()
    mgr._engine.register_read.return_value = True
    mgr._engine.poll_stats_with_details.return_value = _polled()
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
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._send_reads = {}
    mgr._superseded_sender_reads = set()
    mgr._pulling_cleared = set()
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._timeout_s = 120.0
    mgr._dynamic_host_staging = True
    mgr.chunked_transfer = False
    mgr.waiting_timeout = 300
    _activate_prefix_pull(mgr)
    mgr.kv_args = SimpleNamespace(page_size=PAGE)
    mgr.local_ip = "10.0.0.2"
    mgr.rank_port = 5555
    mgr.raiden_endpoint = DECODE_RAIDEN
    mgr._send_multipart_locked = Mock()
    mgr._message_handlers = {PrefixPull.TAG: mgr._on_prefix_pull}
    return mgr


def _register(mgr, decode_prefix_len):
    mgr.register_decode_request(
        ROOM,
        [21, 22, 23],
        [],
        None,
        [PREFILL_INFO],
        decode_prefix_len=decode_prefix_len,
    )


def _refusal(first_page):
    return PrefixReady(
        room=ROOM,
        req_id=PX,
        uuid=0,
        endpoint=DECODE_RAIDEN,
        first_page=first_page,
        num_pages=0,
    )


class TestPrefixPullWire(CustomTestCase):
    def test_pull_req_without_prefix_fields_promises_no_offer(self):
        # A decode without the feature encodes no prefix fields.
        legacy = PullReq(
            room=ROOM, decode_ip="10.0.0.2", decode_port=5555, decode_pages=4
        )
        req = PullReq.from_frames(legacy.to_frames())
        self.assertEqual((req.prefix_tokens, req.prefix_offer), (0, False))

    def test_messages_round_trip(self):
        for msg in (
            PullReq(
                room=ROOM,
                decode_ip="10.0.0.2",
                decode_port=5555,
                decode_pages=4,
                prefix_tokens=12,
                prefix_offer=True,
            ),
            PrefixOffer(room=ROOM, prefix_tokens=12, blocks=[3, 4, 5]),
            PrefixPull(room=ROOM, first_page=1, num_pages=2),
            PrefixReady(
                room=ROOM,
                req_id=PX,
                uuid=99,
                endpoint=DECODE_RAIDEN,
                first_page=1,
                num_pages=2,
            ),
        ):
            with self.subTest(message=type(msg).__name__):
                self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)

    def test_invalid_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            PrefixPull(room=ROOM, first_page=0, num_pages=0)
        with self.assertRaises(ValueError):
            PrefixOffer(room=ROOM, prefix_tokens=-1, blocks=[])
        with self.assertRaises(ValueError):
            PullReq(
                room=ROOM,
                decode_ip="10.0.0.2",
                decode_port=5555,
                decode_pages=4,
                prefix_tokens=-1,
            )

    def test_unknown_fields_are_rejected(self):
        payload = b'{"room": 7, "prefix_tokens": 4, "blocks": [1], "extra": 1}'
        with self.assertRaises(ValueError):
            PrefixOffer.from_frames([MessageTag.PREFIX_OFFER, payload])


class TestDecodeOffersItsPrefix(CustomTestCase):
    def _served(self, mgr, first_page, num_pages):
        """The decode's PREFIX_READY to a PREFIX_PULL of the given range."""
        mgr._dispatch(
            PrefixPull(
                room=ROOM, first_page=first_page, num_pages=num_pages
            ).to_frames()
        )
        endpoint, frames = _sent_frames(mgr)[-1]
        self.assertEqual(endpoint, PREFILL_ADDR)
        return PrefixReady.from_frames(frames)

    def test_page_long_prefix_promises_an_offer(self):
        mgr = _decode_manager()
        _register(mgr, decode_prefix_len=8)
        ((endpoint, frames),) = _sent_frames(mgr)
        req = PullReq.from_frames(frames)
        self.assertEqual(endpoint, PREFILL_ADDR)
        self.assertEqual((req.prefix_tokens, req.prefix_offer), (8, True))
        self.assertTrue(mgr.prefix_offer_owed(ROOM))

    def test_disabled_or_short_prefix_promises_nothing(self):
        for enabled, prefix_len in ((False, 8), (True, PAGE - 1)):
            with self.subTest(enabled=enabled, prefix_len=prefix_len):
                mgr = _decode_manager()
                mgr.prefix_pull_enabled = enabled
                _register(mgr, decode_prefix_len=prefix_len)
                ((_, frames),) = _sent_frames(mgr)
                req = PullReq.from_frames(frames)
                self.assertEqual(
                    (req.prefix_tokens, req.prefix_offer), (prefix_len, False)
                )
                self.assertFalse(mgr.prefix_offer_owed(ROOM))

    def test_inactive_engine_promises_nothing(self):
        def window_pool(mgr):
            mgr._swa_kv_caches = [object()]

        def state_pool(mgr):
            mgr._state_kv_caches = (object(),)

        def no_radix_cache(mgr):
            mgr.server_args = SimpleNamespace(disable_radix_cache=True)

        def no_pool_yet(mgr):
            mgr._kv_caches = None

        for arm in (window_pool, state_pool, no_radix_cache, no_pool_yet):
            with self.subTest(arm.__name__):
                mgr = _decode_manager()
                arm(mgr)
                _register(mgr, decode_prefix_len=8)
                ((_, frames),) = _sent_frames(mgr)
                self.assertFalse(PullReq.from_frames(frames).prefix_offer)
                self.assertFalse(mgr.prefix_offer_owed(ROOM))

    def test_offer_is_sent_once_and_records_the_pages(self):
        mgr = _decode_manager()
        _register(mgr, decode_prefix_len=8)
        mgr.offer_resident_prefix(ROOM, [3, 4])
        mgr.offer_resident_prefix(ROOM, [5, 6])
        ((endpoint, frames),) = _sent_frames(mgr)[1:]
        self.assertEqual(endpoint, PREFILL_ADDR)
        self.assertEqual(
            PrefixOffer.from_frames(frames),
            PrefixOffer(room=ROOM, prefix_tokens=8, blocks=[3, 4]),
        )
        self.assertEqual(mgr._decode.prefix_blocks[ROOM], [3, 4])
        self.assertFalse(mgr.prefix_offer_owed(ROOM))

    def test_unreachable_prefill_costs_only_the_offer(self):
        mgr = _decode_manager()
        _register(mgr, decode_prefix_len=8)
        mgr._send_multipart_locked.side_effect = zmq.ZMQError(errno.EHOSTUNREACH)
        mgr.offer_resident_prefix(ROOM, [3, 4])
        self.assertFalse(mgr.prefix_offer_owed(ROOM))
        self.assertFalse(mgr.is_failed(ROOM))
        # Any other errno is not one room's problem.
        mgr._send_multipart_locked.side_effect = zmq.ZMQError(zmq.ETERM)
        with self.assertRaises(zmq.ZMQError):
            mgr._dispatch(PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames())

    def test_pull_registers_the_asked_slice_and_replies_ready(self):
        mgr = _decode_manager()
        _register(mgr, decode_prefix_len=16)
        mgr.offer_resident_prefix(ROOM, [3, 4, 5, 6])
        ready = self._served(mgr, 1, 2)
        mgr._engine.register_read.assert_called_once()
        req_id, uuid, blocks = mgr._engine.register_read.call_args.args
        self.assertEqual((req_id, blocks), (PX, [4, 5]))
        self.assertGreater(uuid, 0)
        self.assertEqual(
            ready,
            PrefixReady(
                room=ROOM,
                req_id=PX,
                uuid=uuid,
                endpoint=DECODE_RAIDEN,
                first_page=1,
                num_pages=2,
            ),
        )
        self.assertEqual(len(_sent_frames(mgr)), 3)

    def test_unservable_pull_is_refused_with_no_pages(self):
        def withdrawn(mgr):
            mgr.offer_resident_prefix(ROOM, [])

        def past_the_offer(mgr):
            mgr.offer_resident_prefix(ROOM, [3, 4])

        def aborted(mgr):
            mgr.offer_resident_prefix(ROOM, [3, 4])
            mgr._aborted[ROOM] = time.time()

        for arm in (withdrawn, past_the_offer, aborted):
            with self.subTest(arm.__name__):
                mgr = _decode_manager()
                _register(mgr, decode_prefix_len=8)
                arm(mgr)
                sent = len(_sent_frames(mgr))
                self.assertEqual(self._served(mgr, 1, 2), _refusal(first_page=1))
                self.assertEqual(len(_sent_frames(mgr)), sent + 1)
                mgr._engine.register_read.assert_not_called()
                self.assertFalse(mgr.room_has_active_io(ROOM))
                self.assertFalse(mgr.is_failed(ROOM))

    def test_rejected_registration_is_refused_without_failing_the_room(self):
        for name, raises in (("raises", True), ("declines", False)):
            with self.subTest(name):
                mgr = _decode_manager()
                _register(mgr, decode_prefix_len=8)
                mgr.offer_resident_prefix(ROOM, [3, 4])
                if raises:
                    mgr._engine.register_read.side_effect = RuntimeError("injected")
                else:
                    mgr._engine.register_read.return_value = False
                self.assertEqual(self._served(mgr, 0, 2), _refusal(first_page=0))
                mgr._engine.register_read.assert_called_once()
                self.assertEqual(mgr._send_reads, {})
                self.assertFalse(mgr.room_has_active_io(ROOM))
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})

    def test_lease_holds_the_pages_until_the_read_settles(self):
        for outcome in ("sent", "failed"):
            with self.subTest(outcome=outcome):
                mgr = _decode_manager()
                _register(mgr, decode_prefix_len=8)
                mgr.offer_resident_prefix(ROOM, [3, 4])
                mgr._dispatch(
                    PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames()
                )
                self.assertTrue(mgr.room_has_active_io(ROOM))
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    **{outcome: [PX]}
                )
                mgr._drain_engine()
                self.assertFalse(mgr.room_has_active_io(ROOM))
                # A read the prefill could not finish costs the decode nothing.
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})

    def test_served_read_of_a_cleared_room_is_forgotten(self):
        for live in (True, False):
            with self.subTest(live=live):
                mgr = _decode_manager()
                _register(mgr, decode_prefix_len=8)
                mgr.offer_resident_prefix(ROOM, [3, 4])
                mgr._dispatch(
                    PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames()
                )
                if live:
                    mgr.request_status[ROOM] = KVPoll.WaitingForInput
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    sent=[PX], completed_ns=5_000_000_000
                )
                mgr._drain_engine()
                self.assertFalse(mgr.room_has_active_io(ROOM))
                self.assertEqual(PX in mgr._completed_send, live)
                self.assertEqual(PX in mgr._send_done_at, live)


class TestPrefillWaitsForTheOffer(CustomTestCase):
    def test_promised_offer_keeps_the_room_bootstrapping(self):
        mgr = _prefill_manager()
        sender = _sender(mgr)
        mgr._dispatch(_pull_req())
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))
        self.assertFalse(mgr.prefix_pull_resolved(ROOM))
        mgr.resolve_prefix_pull(ROOM)
        self.assertTrue(mgr.prefix_pull_resolved(ROOM))
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_no_promise_opens_the_gate(self):
        def no_offer(mgr):
            pass

        def disabled(mgr):
            mgr.prefix_pull_enabled = False

        def window_pool(mgr):
            mgr._swa_kv_caches = [object()]

        for arm, offer in ((no_offer, False), (disabled, True), (window_pool, True)):
            with self.subTest(arm.__name__):
                mgr = _prefill_manager()
                arm(mgr)
                sender = _sender(mgr)
                mgr._dispatch(_pull_req(prefix_offer=offer))
                self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
                # An offer that was never promised is dropped.
                mgr._dispatch(
                    PrefixOffer(room=ROOM, prefix_tokens=8, blocks=[3, 4]).to_frames()
                )
                self.assertFalse(mgr.prefix_offer_available(ROOM))

    def test_offer_is_handed_out_once(self):
        mgr = _prefill_manager()
        _sender(mgr)
        mgr._dispatch(_pull_req())
        self.assertFalse(mgr.prefix_offer_available(ROOM))
        self.assertIsNone(mgr.take_prefix_offer(ROOM))
        mgr._dispatch(
            PrefixOffer(room=ROOM, prefix_tokens=8, blocks=[3, 4]).to_frames()
        )
        self.assertTrue(mgr.prefix_offer_available(ROOM))
        self.assertEqual(mgr.take_prefix_offer(ROOM), (8, [3, 4]))
        self.assertFalse(mgr.prefix_offer_available(ROOM))
        self.assertIsNone(mgr.take_prefix_offer(ROOM))
        # The gate stays shut until the puller settles the room.
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))

    def test_withdrawn_offer_resolves_the_room(self):
        mgr = _prefill_manager()
        sender = _sender(mgr)
        mgr._dispatch(_pull_req())
        mgr._dispatch(PrefixOffer(room=ROOM, prefix_tokens=8, blocks=[]).to_frames())
        self.assertFalse(mgr.prefix_offer_available(ROOM))
        self.assertIsNone(mgr.take_prefix_offer(ROOM))
        self.assertTrue(mgr.prefix_pull_resolved(ROOM))
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_unsettled_pull_is_given_up_after_the_bootstrap_timeout(self):
        mgr = _prefill_manager()
        mgr.bootstrap_timeout = 0.5
        sender = _sender(mgr)
        mgr._dispatch(_pull_req())
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        mgr._prefill.prefix_offer_since[ROOM] = time.time() - 1
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertTrue(mgr.prefix_pull_resolved(ROOM))
        self.assertNotIn(ROOM, mgr.failure_records)


class TestPrefillPullsThePrefix(CustomTestCase):
    def _armed(self):
        """A prefill manager whose decode offered pages [3, 4, 5, 6] for ROOM."""
        mgr = _prefill_manager()
        _sender(mgr)
        mgr._dispatch(_pull_req(prefix_tokens=16))
        mgr._dispatch(
            PrefixOffer(room=ROOM, prefix_tokens=16, blocks=[3, 4, 5, 6]).to_frames()
        )
        self.assertEqual(mgr.take_prefix_offer(ROOM), (16, [3, 4, 5, 6]))
        return mgr

    @staticmethod
    def _ready(first_page=1, num_pages=2):
        return PrefixReady(
            room=ROOM,
            req_id=PX,
            uuid=99,
            endpoint=DECODE_RAIDEN,
            first_page=first_page,
            num_pages=num_pages,
        )

    @staticmethod
    def _start(mgr):
        return mgr.start_prefix_pull(ROOM, PX, 99, DECODE_RAIDEN, [4, 5], [40, 41])

    def test_ask_sends_prefix_pull_to_the_decode(self):
        mgr = self._armed()
        self.assertTrue(mgr.ask_prefix_pull(ROOM, 1, 2))
        ((endpoint, frames),) = _sent_frames(mgr)
        self.assertEqual(endpoint, DECODE_ADDR)
        self.assertEqual(
            PrefixPull.from_frames(frames),
            PrefixPull(room=ROOM, first_page=1, num_pages=2),
        )

    def test_ready_is_kept_only_after_a_pull_was_asked(self):
        mgr = self._armed()
        mgr._dispatch(self._ready().to_frames())
        self.assertIsNone(mgr.prefix_pull_ready(ROOM))
        mgr.ask_prefix_pull(ROOM, 1, 2)
        mgr._dispatch(self._ready().to_frames())
        self.assertEqual(mgr.prefix_pull_ready(ROOM), (PX, 99, DECODE_RAIDEN, 1, 2))

    def test_start_admits_staging_and_reads_until_raiden_reports_it(self):
        mgr = self._armed()
        mgr.ask_prefix_pull(ROOM, 1, 2)
        mgr._dispatch(self._ready().to_frames())
        self.assertEqual(self._start(mgr), "started")
        mgr._engine.start_read.assert_called_once_with(
            PX, 99, DECODE_RAIDEN, [4, 5], [40, 41]
        )
        self.assertEqual(mgr.prefix_read_state(PX), "pending")
        self.assertIsNone(mgr.prefix_pull_ready(ROOM))
        # Asking again while the read runs issues nothing new.
        self.assertEqual(self._start(mgr), "started")
        mgr._engine.start_read.assert_called_once()
        mgr._engine.poll_stats_with_details.return_value = _polled(received=[PX])
        mgr._drain_engine()
        self.assertEqual(mgr.prefix_read_state(PX), "done")
        self.assertNotIn(PX, mgr._pulling)
        self.assertFalse(mgr.is_failed(ROOM))

    def test_start_retries_while_staging_is_full(self):
        mgr = self._armed()
        mgr._pulling = StagingAdmission(2)
        self.assertTrue(mgr._pulling.admit("other", 2))
        self.assertEqual(self._start(mgr), "retry")
        mgr._engine.start_read.assert_not_called()
        self.assertEqual(mgr.prefix_read_state(PX), "unknown")
        mgr._pulling.release("other")
        self.assertEqual(self._start(mgr), "started")

    def test_forgotten_pull_gives_up_its_place_in_line(self):
        mgr = self._armed()
        mgr._pulling = StagingAdmission(3)
        self.assertTrue(mgr._pulling.admit("other", 2))
        self.assertEqual(self._start(mgr), "retry")
        # The pull that did not fit holds the head of the line.
        self.assertEqual(mgr._pulling.waiting, PX)
        self.assertFalse(mgr._pulling.admit("third", 1))
        mgr.forget_prefix_pull(ROOM)
        self.assertIsNone(mgr._pulling.waiting)
        self.assertTrue(mgr._pulling.admit("third", 1))

    def test_raising_start_read_fails_the_pull_not_the_room(self):
        mgr = self._armed()
        mgr._engine.start_read.side_effect = RuntimeError("injected start_read")
        self.assertEqual(self._start(mgr), "failed")
        self.assertEqual(mgr.prefix_read_state(PX), "failed")
        self.assertNotIn(PX, mgr._pulling)
        self.assertFalse(mgr.is_failed(ROOM))
        self.assertEqual(mgr.failure_records, {})
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)

    def test_native_failure_of_the_pull_does_not_fail_the_room(self):
        mgr = self._armed()
        self.assertEqual(self._start(mgr), "started")
        mgr._engine.poll_stats_with_details.return_value = _polled(
            failed=[PX],
            details={PX: dict(code="DEADLINE_EXCEEDED", phase="read", reason="x")},
        )
        mgr._drain_engine()
        self.assertEqual(mgr.prefix_read_state(PX), "failed")
        self.assertIn(PX, mgr._native_failed)
        self.assertNotIn(PX, mgr._failed)
        self.assertFalse(mgr.is_failed(ROOM))
        self.assertEqual(mgr.failure_records, {})
        self.assertNotIn(PX, mgr._pulling)


# ----- DecodePrefixPuller against a scripted manager, tree and allocator -----


class _FakeManager:
    """The puller's view of the prefill manager, with scripted replies."""

    def __init__(self):
        self.offers = {}
        self.resolved = set()
        # Rooms settled by another rank, as the reduce would report them.
        self.rank_resolved = set()
        self.forgot = []
        self.asked = []
        self.ask_ok = True
        self.ready = {}
        self.started = []
        self.start_outcome = "started"
        self.read_states = {}
        # Hybrid sliding-window offers and READY replies, and the window part
        # of each ask and start.
        self.swa_offers = {}
        self.swa_ready = {}
        self.asked_swa = []
        self.started_swa = []

    def _drain_engine(self) -> None:
        pass

    def prefix_offer_available(self, room):
        return room in self.offers

    def take_prefix_offer(self, room):
        return self.offers.pop(room, None)

    def resolve_prefix_pull(self, room):
        self.resolved.add(room)

    def prefix_pull_resolved(self, room):
        return room in self.resolved or room in self.rank_resolved

    def forget_prefix_pull(self, room):
        self.forgot.append(room)

    def prefix_swa_offer(self, room):
        return self.swa_offers.get(room, (0, []))

    def ask_prefix_pull(
        self, room, first_page, num_pages, *, swa_first_page=0, swa_num_pages=0
    ):
        self.asked.append((room, first_page, num_pages))
        self.asked_swa.append((swa_first_page, swa_num_pages))
        return self.ask_ok

    def prefix_pull_ready(self, room):
        return self.ready.get(room)

    def prefix_swa_ready(self, room):
        return self.swa_ready.get(room)

    def start_prefix_pull(
        self,
        room,
        req_id,
        uuid,
        endpoint,
        remote,
        local,
        *,
        swa_req_id="",
        swa_uuid=0,
        swa_endpoint="",
        swa_remote=(),
        swa_local=(),
    ):
        self.started.append((room, req_id, uuid, endpoint, remote, local))
        self.started_swa.append(
            (swa_req_id, swa_uuid, swa_endpoint, list(swa_remote), list(swa_local))
        )
        return self.start_outcome

    def prefix_read_state(self, req_id):
        return self.read_states.get(req_id, "unknown")


class _FakeTree:
    """A radix tree holding the first ``have_tokens`` of every prompt."""

    def __init__(self, have_tokens):
        self.device_indices = torch.arange(100, 100 + have_tokens, dtype=torch.int64)
        self.node = object()
        self.lock_refs = 0
        # Commit locks the node its insert ends at.
        self.inserted_node = object()
        self.pin_refs = 0
        self.lock_events = []
        # Replaced by a cache flush, which drops every lock with the old tree.
        self.root = object()
        self.lock_refs_at_evict = []
        self.evicted = []
        self.inserts = []
        self.insert_prefix_len = 0
        self.matched = []
        # Tokens past the device match that this tree holds in its host tier.
        self.host_hit_length = 0
        # Device tokens the tree could evict. Ample unless a test shrinks it.
        self.evictable_tokens = 1 << 20

    def match_prefix(self, params):
        self.matched.append(list(params.key.token_ids))
        return SimpleNamespace(
            device_indices=self.device_indices,
            last_device_node=self.node,
            host_hit_length=self.host_hit_length,
        )

    def inc_lock_ref(self, node):
        self.lock_events.append(("inc", node))
        if node is self.inserted_node:
            self.pin_refs += 1
        else:
            assert node is self.node
            self.lock_refs += 1
        return SimpleNamespace(to_dec_params=lambda: ("receipt", node))

    def dec_lock_ref(self, node, receipt):
        assert receipt == ("receipt", node)
        self.lock_events.append(("dec", node))
        if node is self.inserted_node:
            assert self.pin_refs > 0, "released a lock this tree does not hold"
            self.pin_refs -= 1
        else:
            assert node is self.node and self.lock_refs > 0
            self.lock_refs -= 1

    def root_node_handle(self):
        return self.root

    def flush(self):
        self.root = object()
        self.lock_refs = self.pin_refs = 0

    def evict(self, params):
        self.evicted.append(params.num_tokens)
        self.lock_refs_at_evict.append(self.lock_refs)

    def full_evictable_size(self):
        return self.evictable_tokens

    def insert(self, params):
        self.inserts.append(params)
        return InsertResult(
            prefix_len=self.insert_prefix_len, last_device_node=self.inserted_node
        )


class _FakeAllocator:
    page_size = PAGE
    # Device pool size in tokens. Large enough that the in-flight cap never binds.
    size = 1 << 20

    def __init__(self, free_tokens):
        self.free_tokens = free_tokens
        self.next_index = 200
        self.fail = False
        self.allocs = []
        self.freed = []

    def available_size(self):
        return self.free_tokens

    def alloc(self, num_tokens):
        self.allocs.append(num_tokens)
        if self.fail:
            return None
        out = torch.arange(
            self.next_index, self.next_index + num_tokens, dtype=torch.int32
        )
        self.next_index += num_tokens
        return out

    def free(self, indices):
        self.freed.append(indices.tolist())


NEW_INDICES = list(range(200, 212))  # the allocator's first 12 tokens (3 pages)
NEXT_INDICES = list(range(212, 224))
NEW_PAGES = [50, 51, 52]


def _req(num_tokens=20, room=ROOM):
    return SimpleNamespace(
        bootstrap_room=room,
        origin_input_ids=list(range(1000, 1000 + num_tokens)),
        extra_key=None,
        cache_salt=None,
    )


def _offered(
    have_tokens, offer_tokens=16, blocks=(3, 4, 5, 6), free_tokens=64, allocator=None
):
    """A puller whose decode offered ``blocks`` of a ``offer_tokens`` prefix
    while the prefill's tree holds ``have_tokens`` of it."""
    tree = _FakeTree(have_tokens)
    allocator = allocator if allocator is not None else _FakeAllocator(free_tokens)
    mgr = _FakeManager()
    mgr.offers[ROOM] = (offer_tokens, list(blocks))
    scheduler = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=allocator,
        # Tree values follow this table's device (see DecodePrefixPuller).
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(1, dtype=torch.int32)
        ),
        attn_tp_cpu_group=None,
        attn_cp_cpu_group=None,
    )
    puller = DecodePrefixPuller(kv_manager=mgr, scheduler=scheduler)
    return puller, tree, allocator, mgr


def _other_rank_gave_up(rows, op):
    """A reduce in which some rank reported every room as given up."""
    return [row[:3] + [0] + row[4:] if len(row) == 5 else list(row) for row in rows]


def _other_rank_lacks_ready(rows, op):
    """A reduce in which some rank has not received the decode's READY yet."""
    return [row[:4] + [0] if len(row) == 5 else list(row) for row in rows]


def _other_rank_planned_more(rows, op):
    """A reduce in which some rank planned one page more for every room."""
    if op != dist.ReduceOp.MAX:
        return [list(row) for row in rows]
    return [[1, row[1], row[2] + 1] if len(row) == 3 else list(row) for row in rows]


def _asked(have_tokens=4, **kwargs):
    """A puller that asked the decode for pages [1, 4) of ROOM's offer."""
    puller, tree, allocator, mgr = _offered(have_tokens=have_tokens, **kwargs)
    req = _req()
    puller.advance([req])
    return puller, tree, allocator, mgr, req


def _allocated(**kwargs):
    """A puller holding ROOM's destination pages, its read not started yet
    (no host staging free)."""
    puller, tree, allocator, mgr, req = _asked(**kwargs)
    mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
    mgr.start_outcome = "retry"
    puller.advance([req])
    assert allocator.allocs == [12] and puller._plans[ROOM].state == "asked"
    return puller, tree, allocator, mgr, req


class TestPullerPlans(CustomTestCase):
    def test_prefix_the_prefill_holds_is_not_pulled(self):
        puller, tree, allocator, mgr = _offered(have_tokens=16)
        puller.advance([_req()])
        self.assertEqual(tree.matched, [list(range(1000, 1016))])
        self.assertEqual((mgr.asked, mgr.resolved), ([], {ROOM}))
        self.assertEqual((allocator.allocs, tree.lock_refs), ([], 0))
        self.assertEqual(puller.stats["skipped_have"], 1)

    def test_prefix_in_the_prefill_host_tier_is_restored_locally(self):
        # A local host-to-device load is cheaper than a network pull, so pages
        # the prefill keeps in its own host tier are never pulled.
        puller, tree, allocator, mgr = _offered(have_tokens=4)
        tree.host_hit_length = 12
        puller.advance([_req()])
        self.assertEqual((mgr.asked, mgr.resolved), ([], {ROOM}))
        self.assertEqual((allocator.allocs, tree.lock_refs), ([], 0))
        self.assertEqual(puller.stats["skipped_local_host"], 1)
        # A decode offer longer than the local copy is still pulled.
        puller, tree, allocator, mgr = _offered(have_tokens=4)
        tree.host_hit_length = 4
        puller.advance([_req()])
        self.assertEqual(mgr.asked, [(ROOM, 1, 3)])

    def test_inflight_pulls_never_take_more_than_their_pool_share(self):
        # Pulled pages sit outside admission accounting until commit. Holding
        # the whole pool starved admission (prefill OOM on Qwen3-8B, 4K prefixes).
        with tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MAX_POOL_FRACTION.override(0.25):
            allocator = _FakeAllocator(1 << 20)
            allocator.size = 40
            puller, tree, allocator, mgr = _offered(have_tokens=4, allocator=allocator)
        puller.advance([_req()])
        self.assertEqual((mgr.asked, mgr.resolved), ([], {ROOM}))
        self.assertEqual((allocator.allocs, tree.lock_refs), ([], 0))
        self.assertEqual(puller.stats["over_budget"], 1)

    def test_asked_pulls_count_toward_the_cap_before_allocating(self):
        # Pages are allocated only at READY, so the cap must count what asked
        # pulls will take, or every offer in one step passes it.
        with tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MAX_POOL_FRACTION.override(0.5):
            allocator = _FakeAllocator(1 << 20)
            allocator.size = 40
            puller, tree, allocator, mgr = _offered(have_tokens=4, allocator=allocator)
        mgr.offers[OTHER_ROOM] = (16, [13, 14, 15, 16])
        puller.advance([_req(), _req(room=OTHER_ROOM)])
        self.assertEqual(mgr.asked, [(ROOM, 1, 3)])
        self.assertEqual(allocator.allocs, [])
        self.assertEqual(mgr.resolved, {OTHER_ROOM})
        self.assertEqual((puller.stats["over_budget"], tree.lock_refs), (1, 1))

    def test_missing_pages_are_asked_for_and_locked(self):
        puller, tree, allocator, mgr, req = _asked()
        self.assertEqual(mgr.asked, [(ROOM, 1, 3)])
        self.assertEqual(tree.lock_refs, 1)
        self.assertEqual(mgr.resolved, set())
        plan = puller._plans[ROOM]
        self.assertEqual(plan.remote, [4, 5, 6])
        self.assertEqual(plan.have_indices.tolist(), [100, 101, 102, 103])
        # Nothing new is asked while the decode has not answered.
        puller.advance([req])
        self.assertEqual(mgr.asked, [(ROOM, 1, 3)])
        self.assertEqual(mgr.resolved, set())

    def test_pages_are_allocated_in_the_step_ready_arrives(self):
        """Destination pages are allocated only once the decode's READY is in:
        allocated at ask time they sat idle through the decode's round trip,
        outside admission accounting."""
        puller, _, allocator, mgr, req = _asked()
        puller.advance([req])
        self.assertEqual(allocator.allocs, [])
        self.assertIsNone(puller._plans[ROOM].new_indices)
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller.advance([req])
        self.assertEqual(allocator.allocs, [12])
        self.assertEqual(
            mgr.started, [(ROOM, PX, 99, DECODE_RAIDEN, [4, 5, 6], NEW_PAGES)]
        )

    def test_allocation_waits_for_ready_on_every_rank(self):
        # The allocation may evict tree nodes, so every rank must take it in
        # the same step.
        puller, _, allocator, mgr, req = _asked()
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller._all_ranks_reduce = _other_rank_lacks_ready
        puller.advance([req])
        self.assertEqual((allocator.allocs, mgr.started, mgr.resolved), ([], [], set()))
        self.assertEqual(puller._plans[ROOM].state, "asked")
        del puller._all_ranks_reduce
        puller.advance([req])
        self.assertEqual(allocator.allocs, [12])
        self.assertEqual(puller._plans[ROOM].state, "pulling")

    def test_prompt_shorter_than_the_offer_bounds_the_pull(self):
        puller, _, _, mgr = _offered(have_tokens=0)
        puller.advance([_req(num_tokens=10)])
        self.assertEqual(mgr.asked, [(ROOM, 0, 2)])

    def test_matched_prefix_is_locked_before_pages_are_evicted(self):
        # Eviction takes any unlocked leaf, the matched prefix included.
        puller, tree, _, mgr, req = _asked(free_tokens=8)
        self.assertEqual(tree.evicted, [])
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller.advance([req])
        self.assertEqual(tree.evicted, [4])
        self.assertEqual(tree.lock_refs_at_evict, [1])

    def test_no_memory_at_ready_abandons_the_pull(self):
        puller, tree, allocator, mgr, req = _asked()
        allocator.fail = True
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller.advance([req])
        self.assertEqual(puller.stats["no_memory"], 1)
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual((mgr.started, allocator.freed), ([], []))
        self.assertEqual(
            (tree.lock_refs, mgr.resolved, mgr.forgot), (0, {ROOM}, [ROOM])
        )

    def test_pull_leaves_a_reserve_for_admission(self):
        # Pull pages bypass admission: taking the last free pages starved the
        # admitted requests (prefill OOM on Qwen3-8B with 4K-token prefixes).
        puller, tree, allocator, mgr, req = _asked()
        allocator.size = 64
        allocator.free_tokens = 12
        tree.evictable_tokens = 12
        puller._min_free_tokens = 16
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller.advance([req])
        self.assertEqual((allocator.allocs, tree.evicted), ([], []))
        self.assertEqual(puller.stats["no_memory"], 1)
        self.assertEqual((tree.lock_refs, mgr.resolved), (0, {ROOM}))

    def test_pulls_under_the_minimum_are_skipped(self):
        with tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MIN_TOKENS.override(12):
            puller, _, _, mgr = _offered(have_tokens=8)
            puller.advance([_req()])
            self.assertEqual((mgr.asked, mgr.resolved), ([], {ROOM}))
            self.assertEqual(puller.stats["skipped_small"], 1)
            puller, _, _, mgr = _offered(have_tokens=4)
            puller.advance([_req()])
            self.assertEqual(mgr.asked, [(ROOM, 1, 3)])

    def test_failed_ask_unlocks_and_resolves(self):
        puller, tree, allocator, mgr = _offered(have_tokens=4)
        mgr.ask_ok = False
        puller.advance([_req()])
        self.assertEqual((allocator.allocs, allocator.freed), ([], []))
        self.assertEqual(tree.lock_refs, 0)
        self.assertEqual(mgr.resolved, {ROOM})
        self.assertNotIn(ROOM, puller._plans)

    def test_room_given_up_on_another_rank_is_given_up_here(self):
        puller, tree, allocator, mgr = _offered(have_tokens=4)
        mgr.rank_resolved.add(ROOM)
        puller.advance([_req()])
        self.assertEqual(mgr.resolved, {ROOM})
        # The offer is left untaken: no rank may plan a pull alone.
        self.assertIn(ROOM, mgr.offers)
        self.assertEqual((mgr.asked, allocator.allocs, tree.lock_refs), ([], [], 0))
        # A plan already in flight is dropped the same way, pages and all.
        puller, tree, allocator, mgr, req = _allocated()
        puller._all_ranks_reduce = _other_rank_gave_up
        puller.advance([req])
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(allocator.freed, [NEW_INDICES])
        self.assertEqual(
            (tree.lock_refs, mgr.resolved, mgr.forgot), (0, {ROOM}, [ROOM])
        )

    def test_ranks_planning_different_ranges_give_up(self):
        puller, tree, allocator, mgr = _offered(have_tokens=4)
        puller._all_ranks_reduce = _other_rank_planned_more
        puller.advance([_req()])
        self.assertEqual(mgr.asked, [(ROOM, 1, 3)])
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual((allocator.allocs, allocator.freed), ([], []))
        self.assertEqual(
            (tree.lock_refs, mgr.resolved, mgr.forgot), (0, {ROOM}, [ROOM])
        )


class TestPullerReads(CustomTestCase):
    def _pulling(self):
        puller, tree, allocator, mgr, req = _asked()
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller.advance([req])
        self.assertEqual(puller._plans[ROOM].state, "pulling")
        self.assertEqual(mgr.resolved, set())
        return puller, tree, allocator, mgr, req

    def test_ready_for_another_range_abandons_the_pull(self):
        puller, tree, allocator, mgr, req = _asked()
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 0, 4)
        puller.advance([req])
        self.assertEqual(mgr.started, [])
        self.assertEqual(allocator.freed, [NEW_INDICES])
        self.assertEqual(
            (tree.lock_refs, mgr.resolved, mgr.forgot), (0, {ROOM}, [ROOM])
        )
        self.assertNotIn(ROOM, puller._plans)

    def test_retry_asks_again_next_step(self):
        puller, _, allocator, mgr, req = _allocated()
        puller.advance([req])
        self.assertEqual(len(mgr.started), 2)
        self.assertEqual(puller._plans[ROOM].state, "asked")
        mgr.start_outcome = "started"
        puller.advance([req])
        self.assertEqual(puller._plans[ROOM].state, "pulling")
        # The pages allocated at READY are reused, not allocated again.
        self.assertEqual(allocator.allocs, [12])
        self.assertTrue(all(call[5] == NEW_PAGES for call in mgr.started))

    def test_failed_start_resolves_the_room(self):
        puller, tree, allocator, mgr, req = _asked()
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        mgr.start_outcome = "failed"
        puller.advance([req])
        self.assertEqual(allocator.freed, [NEW_INDICES])
        self.assertEqual((tree.lock_refs, mgr.resolved), (0, {ROOM}))
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(puller.stats["failed"], 1)

    def test_settled_read_resolves_the_room_and_commit_inserts(self):
        puller, tree, allocator, mgr, req = self._pulling()
        mgr.read_states[PX] = "done"
        puller.advance([req])
        self.assertEqual(mgr.resolved, {ROOM})
        self.assertIs(puller._plans[ROOM].agreed_done, True)
        self.assertEqual(tree.inserts, [])
        tree.insert_prefix_len = 8
        puller.commit(req)
        (params,) = tree.inserts
        self.assertEqual(list(params.key.token_ids), list(range(1000, 1016)))
        self.assertEqual(params.value.tolist(), [100, 101, 102, 103] + NEW_INDICES)
        self.assertEqual(params.value.dtype, torch.int64)
        self.assertEqual((params.prev_prefix_len, params.inserted_len), (4, 16))
        self.assertEqual((tree.lock_refs, allocator.freed), (0, []))
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(
            (puller.stats["pulls"], puller.stats["pulled_tokens"]), (1, 12)
        )
        self.assertEqual(puller.stats["dedup_tokens"], 4)

    def test_failed_read_resolves_the_room_and_commit_computes(self):
        puller, tree, allocator, mgr, req = self._pulling()
        mgr.read_states[PX] = "failed"
        puller.advance([req])
        self.assertEqual(mgr.resolved, {ROOM})
        self.assertIs(puller._plans[ROOM].agreed_done, False)
        puller.commit(req)
        self.assertEqual(tree.inserts, [])
        self.assertEqual(allocator.freed, [NEW_INDICES])
        self.assertEqual((tree.lock_refs, mgr.forgot), (0, [ROOM]))
        self.assertEqual(puller.stats["failed"], 1)
        self.assertNotIn(ROOM, puller._plans)

    def test_commit_before_the_read_lands_keeps_the_pages_until_it_settles(self):
        puller, tree, allocator, mgr, req = self._pulling()
        mgr.read_states[PX] = "pending"
        puller.commit(req)
        self.assertEqual((tree.inserts, tree.lock_refs), ([], 0))
        self.assertEqual(puller.stats["failed"], 1)
        self.assertNotIn(ROOM, puller._plans)
        puller.advance([])
        self.assertEqual(allocator.freed, [])
        mgr.read_states[PX] = "done"
        puller.advance([])
        self.assertEqual(allocator.freed, [NEW_INDICES])

    def test_commit_without_a_read_computes_the_prompt(self):
        puller, tree, allocator, mgr, req = _asked()
        puller.commit(req)
        self.assertEqual(tree.inserts, [])
        self.assertEqual((allocator.allocs, allocator.freed), ([], []))
        self.assertEqual((tree.lock_refs, mgr.forgot), (0, [ROOM]))
        self.assertEqual(puller.stats["failed"], 1)

    def test_release_frees_the_plan(self):
        for name, setup in (("asked", _asked), ("allocated", _allocated)):
            with self.subTest(name):
                puller, tree, allocator, mgr, req = setup()
                freed = [NEW_INDICES] if allocator.allocs else []
                puller.release(req)
                self.assertEqual(allocator.freed, freed)
                self.assertEqual((tree.lock_refs, mgr.forgot), (0, [ROOM]))
                self.assertNotIn(ROOM, puller._plans)
                puller.release(req)
                self.assertEqual((allocator.freed, mgr.forgot), (freed, [ROOM]))

    def test_release_all_frees_every_plan(self):
        puller, tree, allocator, mgr = _offered(have_tokens=4)
        mgr.offers[OTHER_ROOM] = (16, [13, 14, 15, 16])
        reqs = [_req(), _req(room=OTHER_ROOM)]
        puller.advance(reqs)
        # ROOM's pages are allocated, OTHER_ROOM still waits for READY.
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        mgr.start_outcome = "retry"
        puller.advance(reqs)
        self.assertEqual(sorted(puller._plans), [ROOM, OTHER_ROOM])
        self.assertEqual((tree.lock_refs, allocator.allocs), (2, [12]))
        puller.release_all()
        self.assertEqual(puller._plans, {})
        self.assertEqual(tree.lock_refs, 0)
        self.assertEqual(allocator.freed, [NEW_INDICES])
        self.assertEqual(sorted(mgr.forgot), [ROOM, OTHER_ROOM])


def _committed(**kwargs):
    """A puller that inserted ROOM's pulled pages [4, 16) into the tree."""
    puller, tree, allocator, mgr, req = _asked(**kwargs)
    mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
    puller.advance([req])
    mgr.read_states[PX] = "done"
    puller.advance([req])
    puller.commit(req)
    assert len(tree.inserts) == 1 and tree.pin_refs == 1
    return puller, tree, allocator, mgr, req


class TestPulledPathLock(CustomTestCase):
    """Nothing but the puller locks the pulled path between the insert and the
    request's own admission match, so without its lock the SWA LRU or a full
    eviction drops the pulled pages first and the prompt is recomputed."""

    def test_path_is_locked_until_the_request_leaves_both_queues(self):
        puller, tree, _, _, req = _committed()
        # Locked before the matched prefix's lock goes: [0, M) is never unlocked.
        self.assertEqual(
            tree.lock_events[-2:], [("inc", tree.inserted_node), ("dec", tree.node)]
        )
        self.assertEqual(tree.lock_refs, 0)
        # Still in the bootstrap queue (finalize deferred), then waiting.
        puller.advance([req])
        puller.advance([], waiting_reqs=[req])
        self.assertEqual(tree.pin_refs, 1)
        # Admission matched it and took its own lock, so it left the queue.
        puller.advance([], waiting_reqs=[_req(room=OTHER_ROOM)])
        self.assertEqual(tree.pin_refs, 0)
        # Nothing releases it again (the tree asserts on a double release).
        puller.advance([])
        puller.release(req)
        puller.release_all()
        self.assertEqual(puller._pins, {})

    def test_failed_aborted_or_released_request_unlocks_its_path_once(self):
        # Failed in the bootstrap queue after commit.
        puller, tree, _, _, req = _committed()
        puller.release(req)
        self.assertEqual(tree.pin_refs, 0)
        puller.advance([])
        puller.release(req)
        # Aborted from the waiting queue, which tells the puller nothing.
        puller, tree, _, _, req = _committed()
        puller.advance([], waiting_reqs=[req])
        puller.advance([])
        self.assertEqual(tree.pin_refs, 0)
        # Memory occupation released.
        puller, tree, _, _, req = _committed()
        puller.release_all()
        self.assertEqual((tree.pin_refs, puller._pins), (0, {}))
        puller.advance([], waiting_reqs=[req])
        puller.release(req)

    def test_flushed_tree_is_not_unlocked(self):
        """A flush can run once both queues are empty, before the next step
        unlocks, and releasing the old tree's node then corrupts the new tree."""
        puller, tree, _, _, _ = _committed()
        tree.flush()
        puller.advance([])
        self.assertEqual((tree.pin_refs, puller._pins), (0, {}))

    def test_locked_paths_count_toward_the_cap_until_admitted(self):
        # A locked path is as unevictable as an in-flight pull's pages.
        with tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MAX_POOL_FRACTION.override(0.5):
            allocator = _FakeAllocator(1 << 20)
            allocator.size = 40
            puller, tree, allocator, mgr, req = _committed(allocator=allocator)
        third_room = OTHER_ROOM + 1
        mgr.offers[OTHER_ROOM] = (16, [13, 14, 15, 16])
        mgr.offers[third_room] = (16, [23, 24, 25, 26])
        puller.advance([_req(room=OTHER_ROOM)], waiting_reqs=[req])
        self.assertEqual(puller.stats["over_budget"], 1)
        puller.advance([_req(room=third_room)])
        self.assertEqual(mgr.asked[-1], (third_room, 1, 3))


class TestBootstrapQueueKeepsThePathLocked(CustomTestCase):
    def _pop(self, queue):
        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *_: [s.poll() for s in senders],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.get_disagg",
                return_value=SimpleNamespace(
                    optimistic_prefill_attempts=0,
                    disaggregation_decode_allocation_policy="early",
                ),
            ),
        ):
            return queue.pop_bootstrapped()

    def test_request_waiting_for_admission_keeps_its_pulled_path_locked(self):
        puller, tree, _, mgr, req = _asked()
        mgr.ready[ROOM] = (PX, 99, DECODE_RAIDEN, 1, 3)
        puller.advance([req])
        mgr.read_states[PX] = "done"
        req.rid = "req-7"
        req.prefill_attempt_count = 0
        req.is_retracted = False
        req.time_stats = Mock()
        req.disagg_kv_sender = SimpleNamespace(poll=lambda: KVPoll.WaitingForInput)
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler_stage_metrics = None
        queue.pp_size = 1
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            processed_tokens_counter=0,
            waiting_queue=[],
        )
        queue.finalize_bootstrap = Mock(return_value=True)
        queue.decode_prefix_puller = puller
        queue.queue = [req]
        # One step settles the read, commits it and pops the request.
        self.assertEqual(self._pop(queue), [req])
        self.assertEqual(tree.pin_refs, 1)
        queue.scheduler.waiting_queue = [req]
        self.assertEqual(self._pop(queue), [])
        self.assertEqual(tree.pin_refs, 1)
        queue.scheduler.waiting_queue = []
        self._pop(queue)
        self.assertEqual(tree.pin_refs, 0)


# ----- hybrid sliding-window models: window pages ride on the SWA engine -----

PXS = f"{ROOM}#pxs"
WINDOW = 8  # tokens: two pages
DECODE_SWA_RAIDEN = "10.0.0.2:31022"


def _hybrid(mgr):
    """A sliding-window pool with its own engine up, as on Muse Spark."""
    mgr._swa_kv_caches = [object()]
    mgr._swa_engine = Mock()
    mgr._swa_engine.register_read.return_value = True
    mgr._swa_engine.poll_stats_with_details.return_value = _polled()
    mgr._swa_engine_max_blocks = 16
    mgr.swa_raiden_endpoint = DECODE_SWA_RAIDEN
    mgr.kv_args = SimpleNamespace(page_size=PAGE)
    return mgr


class TestWindowPrefixWire(CustomTestCase):
    def test_window_fields_round_trip_and_default_to_none(self):
        for msg in (
            PrefixOffer(
                room=ROOM,
                prefix_tokens=16,
                blocks=[3, 4, 5, 6],
                swa_first_page=2,
                swa_blocks=[62, 63],
            ),
            PrefixPull(
                room=ROOM, first_page=1, num_pages=3, swa_first_page=2, swa_num_pages=2
            ),
            PrefixReady(
                room=ROOM,
                req_id=PX,
                uuid=9,
                endpoint=DECODE_RAIDEN,
                first_page=1,
                num_pages=3,
                swa_req_id=PXS,
                swa_uuid=10,
                swa_endpoint=DECODE_SWA_RAIDEN,
                swa_first_page=2,
                swa_num_pages=2,
            ),
        ):
            with self.subTest(message=type(msg).__name__):
                self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)
        # A full-only peer's payload decodes with no window.
        pull = PrefixPull.from_frames(
            [MessageTag.PREFIX_PULL, b'{"room": 7, "first_page": 1, "num_pages": 2}']
        )
        self.assertEqual((pull.swa_first_page, pull.swa_num_pages), (0, 0))
        with self.assertRaises(ValueError):
            PrefixPull(room=ROOM, first_page=0, num_pages=1, swa_num_pages=-1)


class TestWindowPrefixOffer(CustomTestCase):
    def test_deepest_prefix_whose_window_has_no_hole(self):
        cases = (
            # (matched, holes, expected)
            (32, [], 32),  # all live from the root
            (32, [(4, 12)], 32),  # the hole ends before the window
            (32, [(4, 28)], 4),  # window [24, 32) hit: back off to the hole
            (32, [(0, 4), (12, 28)], 12),  # [4, 12) holds a whole window
            (32, [(0, 4), (8, 28)], 0),  # no whole window anywhere
            (6, [], 6),  # shorter than a window, live from the root
            (6, [(0, 4)], 0),
        )
        for matched, holes, expected in cases:
            with self.subTest(matched=matched, holes=holes):
                self.assertEqual(
                    deepest_window_prefix(matched=matched, holes=holes, window=WINDOW),
                    expected,
                )
        self.assertEqual(window_tokens(window_size=5, page_size=PAGE), 8)

    @staticmethod
    def _tree(holes, matched=20, freed_slot=None, is_eagle=False):
        """A decode tree holding full KV for the first ``matched`` tokens with SWA
        missing over ``holes``. Window ids are full ids + 1000, 0 if freed."""
        full = torch.arange(100, 100 + matched, dtype=torch.int64)
        calls = SimpleNamespace(keys=[], locks=[], unlocks=[])
        tree_core = SimpleNamespace(
            is_eagle=is_eagle,
            match_full_device_prefix=lambda key: (
                calls.keys.append(key) or (min(matched, len(key)), 1, 0)
            ),
            swa_tombstone_ranges=lambda key, start, end: [
                (lo, min(hi, end)) for lo, hi in holes if lo < end
            ],
        )

        def match_prefix(params):
            calls.keys.append(params.key)
            return SimpleNamespace(
                device_indices=full[: len(params.key)], last_device_node="node"
            )

        def translate(indices):
            out = indices + 1000
            if freed_slot is not None:
                out[indices == freed_slot] = 0
            return out

        tree = SimpleNamespace(
            tree_core=tree_core,
            match_prefix=match_prefix,
            inc_lock_ref=lambda node: (
                calls.locks.append(node)
                or SimpleNamespace(to_dec_params=lambda: "receipt")
            ),
            dec_lock_ref=lambda node, receipt: calls.unlocks.append((node, receipt)),
        )
        allocator = SimpleNamespace(
            page_size=PAGE, translate_swa_indices_for_transfer=translate
        )
        return tree, allocator, calls

    def _offer(self, tree, allocator, prompt_len=21):
        return lock_window_prefix(
            tree_cache=tree,
            allocator=allocator,
            req=_req(num_tokens=prompt_len),
            window_size=WINDOW,
        )

    def test_offer_ends_at_the_deepest_live_window_and_locks_it(self):
        # Full KV to 20, SWA lost over [8, 12): the window [12, 20) is whole.
        tree, allocator, calls = self._tree(holes=[(8, 12)])
        offer = self._offer(tree, allocator)
        self.assertEqual(offer.blocks, [25, 26, 27, 28, 29])
        self.assertEqual((offer.swa_first_page, offer.swa_blocks), (3, [278, 279]))
        self.assertEqual((offer.lock_node, offer.lock_receipt), ("node", "receipt"))
        self.assertEqual((calls.locks, calls.unlocks), (["node"], []))
        # The prefill computes the last prompt token: a 20-token prompt offers 16.
        tree, allocator, _ = self._tree(holes=[])
        self.assertEqual(len(self._offer(tree, allocator, prompt_len=20).blocks), 4)

    def test_eagle_tree_is_matched_with_one_more_token(self):
        tree, allocator, calls = self._tree(holes=[], is_eagle=True)
        offer = self._offer(tree, allocator)
        self.assertEqual(len(offer.blocks), 5)
        lock_key = calls.keys[-1]
        self.assertTrue(lock_key.is_bigram)
        self.assertEqual((len(lock_key), len(lock_key.token_ids)), (20, 21))

    def test_no_whole_window_withdraws_and_holds_no_lock(self):
        for name, kwargs in (
            ("window lost", dict(holes=[(0, 4), (8, 20)])),
            ("window slot freed", dict(holes=[], freed_slot=117)),
        ):
            with self.subTest(name):
                tree, allocator, calls = self._tree(**kwargs)
                self.assertIsNone(self._offer(tree, allocator))
                self.assertEqual(len(calls.locks), len(calls.unlocks))


class _OfferingReceiver:
    """The transfer queue's view of a raiden receiver on a hybrid decode."""

    def __init__(self):
        self.window_offers = []
        self.full_offers = []
        self.clear_called = False

    @property
    def prefix_offer_pending(self):
        return not (self.window_offers or self.full_offers)

    def offer_window_prefix(self, *, tree_cache, allocator, req, window_size):
        self.window_offers.append(window_size)
        return ("node", "receipt")

    def offer_resident_prefix(self, page_ids):
        self.full_offers.append(page_ids)

    def failure_exception(self):
        return None

    def clear(self):
        self.clear_called = True


class TestDecodeHoldsTheOfferLock(CustomTestCase):
    @staticmethod
    def _queue(polls):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.req_to_metadata_buffer_idx_allocator = Mock()
        queue.tp_rank = 0
        queue.tree_cache = Mock()
        queue.tree_cache.supports_swa.return_value = True
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue._clean_hicache_prefetch_resources = Mock()
        queue._release_request = Mock()
        queue._poll_with_metadata_gate = lambda: [polls.pop(0)]
        queue.scheduler = Mock(enable_decode_hicache=False, enable_hisparse=False)
        queue.scheduler.sliding_window_size = WINDOW
        queue.scheduler.metrics_reporter.enable_metrics = False
        receiver = _OfferingReceiver()
        decode_req = DecodeRequest(
            req=SimpleNamespace(
                rid="r", bootstrap_room=ROOM, return_logprob=False, origin_input_ids=[]
            ),
            kv_receiver=receiver,
            metadata_buffer_index=3,
            prefix_match=SimpleNamespace(),
        )
        queue.queue = [decode_req]
        return queue, decode_req, receiver

    def test_offer_lock_is_held_in_the_queue_and_released_on_leaving(self):
        queue, decode_req, receiver = self._queue([KVPoll.Transferring, KVPoll.Failed])
        with patch("sglang.srt.disaggregation.decode.prepare_abort"):
            queue.pop_transferred()
            self.assertEqual((receiver.window_offers, receiver.full_offers), ([8], []))
            self.assertEqual(decode_req.prefix_offer_lock, ("node", "receipt"))
            queue.tree_cache.dec_lock_ref.assert_not_called()
            queue.pop_transferred()
        self.assertEqual(queue.queue, [])
        queue.tree_cache.dec_lock_ref.assert_called_once_with("node", "receipt")
        self.assertIsNone(decode_req.prefix_offer_lock)
        self.assertEqual(receiver.window_offers, [8])


class TestDecodeServesTheWindow(CustomTestCase):
    def _offered(self):
        mgr = _hybrid(_decode_manager())
        _register(mgr, decode_prefix_len=8)
        mgr.offer_resident_prefix(
            ROOM, [3, 4, 5, 6], swa_first_page=2, swa_page_ids=[62, 63]
        )
        return mgr

    def _pull(self, mgr, first_page=1, num_pages=3, swa_first_page=2, swa_num_pages=2):
        mgr._dispatch(
            PrefixPull(
                room=ROOM,
                first_page=first_page,
                num_pages=num_pages,
                swa_first_page=swa_first_page,
                swa_num_pages=swa_num_pages,
            ).to_frames()
        )
        return PrefixReady.from_frames(_sent_frames(mgr)[-1][1])

    def test_offer_covers_exactly_its_pages_and_names_the_window(self):
        mgr = self._offered()
        self.assertEqual(
            PrefixOffer.from_frames(_sent_frames(mgr)[1][1]),
            PrefixOffer(
                room=ROOM,
                prefix_tokens=16,
                blocks=[3, 4, 5, 6],
                swa_first_page=2,
                swa_blocks=[62, 63],
            ),
        )

    def test_pull_registers_full_and_window_pages_on_their_engines(self):
        mgr = self._offered()
        ready = self._pull(mgr, swa_first_page=3, swa_num_pages=1)
        req_id, uuid, blocks = mgr._engine.register_read.call_args.args
        swa_req_id, swa_uuid, swa_blocks = mgr._swa_engine.register_read.call_args.args
        self.assertEqual(
            (req_id, blocks, swa_req_id, swa_blocks), (PX, [4, 5, 6], PXS, [63])
        )
        self.assertEqual(
            ready,
            PrefixReady(
                room=ROOM,
                req_id=PX,
                uuid=uuid,
                endpoint=DECODE_RAIDEN,
                first_page=1,
                num_pages=3,
                swa_req_id=PXS,
                swa_uuid=swa_uuid,
                swa_endpoint=DECODE_SWA_RAIDEN,
                swa_first_page=3,
                swa_num_pages=1,
            ),
        )
        self.assertEqual(mgr._send_reads, {ROOM: {PX, PXS}})
        # A failed window read costs the decode nothing either.
        mgr._swa_engine.poll_stats_with_details.return_value = _polled(failed=[PXS])
        mgr._engine.poll_stats_with_details.return_value = _polled(sent=[PX])
        mgr._drain_engine()
        self.assertFalse(mgr.room_has_active_io(ROOM))
        self.assertFalse(mgr.is_failed(ROOM))
        self.assertEqual(mgr.failure_records, {})

    def test_any_refusal_serves_neither_range(self):
        def window_refused(mgr):
            mgr._swa_engine.register_read.return_value = False

        def window_raises(mgr):
            mgr._swa_engine.register_read.side_effect = RuntimeError("injected")

        for arm in (window_refused, window_raises):
            with self.subTest(arm.__name__):
                mgr = self._offered()
                arm(mgr)
                ready = self._pull(mgr)
                self.assertEqual((ready.num_pages, ready.swa_num_pages), (0, 0))
                self.assertEqual((ready.uuid, ready.swa_uuid), (0, 0))
                # The full registration raiden cannot withdraw leaves no lease,
                # and its eventual expiry fails nothing.
                self.assertFalse(mgr.room_has_active_io(ROOM))
                mgr._engine.poll_stats_with_details.return_value = _polled(failed=[PX])
                mgr._drain_engine()
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})
        # A window range the offer does not cover is refused unregistered.
        for swa_first_page, swa_num_pages in ((1, 2), (3, 2)):
            with self.subTest(window=(swa_first_page, swa_num_pages)):
                mgr = self._offered()
                ready = self._pull(
                    mgr, swa_first_page=swa_first_page, swa_num_pages=swa_num_pages
                )
                self.assertEqual((ready.num_pages, ready.swa_num_pages), (0, 0))
                mgr._engine.register_read.assert_not_called()
                mgr._swa_engine.register_read.assert_not_called()


class TestPrefillPullsTheWindow(CustomTestCase):
    def _armed(self):
        mgr = _hybrid(_prefill_manager())
        mgr._pulling = StagingAdmission(16, 16)
        _sender(mgr)
        mgr._dispatch(_pull_req(prefix_tokens=8))
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))
        mgr._dispatch(
            PrefixOffer(
                room=ROOM,
                prefix_tokens=16,
                blocks=[3, 4, 5, 6],
                swa_first_page=2,
                swa_blocks=[62, 63],
            ).to_frames()
        )
        self.assertEqual(mgr.take_prefix_offer(ROOM), (16, [3, 4, 5, 6]))
        self.assertEqual(mgr.prefix_swa_offer(ROOM), (2, [62, 63]))
        return mgr

    @staticmethod
    def _start(mgr):
        return mgr.start_prefix_pull(
            ROOM,
            PX,
            99,
            DECODE_RAIDEN,
            [4, 5, 6],
            [40, 41, 42],
            swa_req_id=PXS,
            swa_uuid=98,
            swa_endpoint=DECODE_SWA_RAIDEN,
            swa_remote=[62, 63],
            swa_local=[70, 71],
        )

    def test_ask_and_ready_carry_the_window_range(self):
        mgr = self._armed()
        mgr.ask_prefix_pull(ROOM, 1, 3, swa_first_page=2, swa_num_pages=2)
        self.assertEqual(
            PrefixPull.from_frames(_sent_frames(mgr)[-1][1]),
            PrefixPull(
                room=ROOM, first_page=1, num_pages=3, swa_first_page=2, swa_num_pages=2
            ),
        )
        mgr._dispatch(
            PrefixReady(
                room=ROOM,
                req_id=PX,
                uuid=99,
                endpoint=DECODE_RAIDEN,
                first_page=1,
                num_pages=3,
                swa_req_id=PXS,
                swa_uuid=98,
                swa_endpoint=DECODE_SWA_RAIDEN,
                swa_first_page=2,
                swa_num_pages=2,
            ).to_frames()
        )
        self.assertEqual(mgr.prefix_pull_ready(ROOM), (PX, 99, DECODE_RAIDEN, 1, 3))
        self.assertEqual(mgr.prefix_swa_ready(ROOM), (PXS, 98, DECODE_SWA_RAIDEN, 2, 2))

    def test_both_reads_share_one_admission_until_both_settle(self):
        mgr = self._armed()
        self.assertEqual(self._start(mgr), "started")
        mgr._engine.start_read.assert_called_once_with(
            PX, 99, DECODE_RAIDEN, [4, 5, 6], [40, 41, 42]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            PXS, 98, DECODE_SWA_RAIDEN, [62, 63], [70, 71]
        )
        self.assertEqual((mgr._pulling.pages(), mgr._pulling.swa_pages()), (3, 2))
        mgr._engine.poll_stats_with_details.return_value = _polled(received=[PX])
        mgr._drain_engine()
        # The window may still be landing: its staging and pages stay held.
        self.assertEqual(
            (mgr.prefix_read_state(PX), mgr.prefix_read_state(PXS)), ("done", "pending")
        )
        self.assertIn(PX, mgr._pulling)
        mgr._engine.poll_stats_with_details.return_value = _polled()
        mgr._swa_engine.poll_stats_with_details.return_value = _polled(failed=[PXS])
        mgr._drain_engine()
        self.assertEqual(mgr.prefix_read_state(PXS), "failed")
        self.assertNotIn(PX, mgr._pulling)
        self.assertFalse(mgr.is_failed(ROOM))
        self.assertEqual(mgr.failure_records, {})

    def test_raising_window_start_leaves_the_full_read_to_settle(self):
        mgr = self._armed()
        mgr._swa_engine.start_read.side_effect = RuntimeError("injected")
        self.assertEqual(self._start(mgr), "started")
        self.assertEqual(
            (mgr.prefix_read_state(PX), mgr.prefix_read_state(PXS)),
            ("pending", "failed"),
        )
        self.assertFalse(mgr.is_failed(ROOM))


class _FakeHybridAllocator(_FakeAllocator):
    """A static SWA allocator: full and window pools sized apart. Window ids
    are full ids + 1000."""

    device = "cpu"
    size_full = 1 << 20
    size_swa = 1 << 20

    def __init__(self, free_tokens, free_swa_tokens=64):
        super().__init__(free_tokens)
        self.free_swa_tokens = free_swa_tokens
        self.swa_tail_allocs = []

    def full_available_size(self):
        return self.free_tokens

    def swa_available_size(self):
        return self.free_swa_tokens

    def alloc(self, num_tokens):
        raise AssertionError("a paged static SWA allocator cannot alloc()")

    def alloc_extend_swa_tail(self, **kwargs):
        self.swa_tail_allocs.append(
            {
                k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            }
        )
        return super().alloc(kwargs["extend_num_tokens"])

    def translate_swa_indices_for_transfer(self, indices):
        return indices + 1000


class _FakeHybridTree(_FakeTree):
    def __init__(self, have_tokens, is_eagle=False):
        super().__init__(have_tokens)
        self.tree_core = SimpleNamespace(is_eagle=is_eagle)
        self.swa_evictable_tokens = 1 << 20
        self.evicted_for_alloc = []

    def swa_evictable_size(self):
        return self.swa_evictable_tokens

    def evict_for_alloc(self, params):
        self.evicted_for_alloc.append((params.num_tokens, params.swa_num_tokens))


HYBRID_READY = (PX, 99, DECODE_RAIDEN, 1, 3)
HYBRID_SWA_READY = (PXS, 98, DECODE_SWA_RAIDEN, 2, 2)


def _hybrid_offered(have_tokens=4, swa_offer=(2, [62, 63]), is_eagle=False, **kwargs):
    """A hybrid puller whose decode offered pages [3, 4, 5, 6] (16 tokens) with
    window pages [62, 63] for pages [2, 4), the prefill holding ``have_tokens``."""
    tree = _FakeHybridTree(have_tokens, is_eagle=is_eagle)
    allocator = _FakeHybridAllocator(**{"free_tokens": 64, **kwargs})
    mgr = _FakeManager()
    mgr.offers[ROOM] = (16, [3, 4, 5, 6])
    mgr.swa_offers[ROOM] = swa_offer
    scheduler = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(1, dtype=torch.int32)
        ),
        attn_tp_cpu_group=None,
        attn_cp_cpu_group=None,
    )
    puller = DecodePrefixPuller(kv_manager=mgr, scheduler=scheduler, swa_window=WINDOW)
    return puller, tree, allocator, mgr


def _hybrid_pulling(**kwargs):
    puller, tree, allocator, mgr = _hybrid_offered(**kwargs)
    req = _req()
    puller.advance([req])
    mgr.ready[ROOM] = HYBRID_READY
    mgr.swa_ready[ROOM] = HYBRID_SWA_READY
    puller.advance([req])
    assert puller._plans[ROOM].state == "pulling"
    return puller, tree, allocator, mgr, req


class TestHybridPuller(CustomTestCase):
    def test_plan_takes_the_whole_offer_and_the_window_before_it(self):
        for have_tokens, asked, asked_swa in (
            (4, (ROOM, 1, 3), (2, 2)),  # window [8, 16) past what the prefill has
            (12, (ROOM, 3, 1), (3, 1)),  # the prefill's own window covers [8, 12)
        ):
            with self.subTest(have_tokens=have_tokens):
                puller, tree, _, mgr = _hybrid_offered(have_tokens=have_tokens)
                puller.advance([_req()])
                self.assertEqual((mgr.asked, mgr.asked_swa), ([asked], [asked_swa]))
                self.assertEqual(tree.matched, [list(range(1000, 1016))])
                plan = puller._plans[ROOM]
                self.assertEqual(plan.swa_remote, [62, 63][asked_swa[0] - 2 :])

    def test_offer_not_ending_in_its_window_or_the_prompt_is_not_pulled(self):
        for name, swa_offer, num_tokens in (
            ("window ends early", (1, [61, 62]), 20),
            ("window starts late", (3, [63]), 20),
            ("offer reaches the last prompt token", (2, [62, 63]), 16),
        ):
            with self.subTest(name):
                puller, tree, _, mgr = _hybrid_offered(swa_offer=swa_offer)
                puller.advance([_req(num_tokens=num_tokens)])
                self.assertEqual((mgr.asked, mgr.resolved), ([], {ROOM}))
                self.assertEqual((tree.matched, tree.lock_refs), ([], 0))

    def test_eagle_tree_keys_the_offer_with_one_more_token(self):
        puller, tree, _, mgr = _hybrid_offered(is_eagle=True)
        puller.advance([_req()])
        self.assertEqual(tree.matched, [list(range(1000, 1017))])
        self.assertEqual(mgr.asked, [(ROOM, 1, 3)])

    def test_allocation_binds_window_slots_for_the_tail_only(self):
        puller, tree, allocator, mgr, _ = _hybrid_pulling(
            free_tokens=4, free_swa_tokens=2
        )
        self.assertEqual(tree.evicted_for_alloc, [(8, 6)])
        self.assertEqual(tree.evicted, [])
        (alloc,) = allocator.swa_tail_allocs
        self.assertEqual(
            alloc,
            dict(
                prefix_lens=[4],
                prefix_lens_cpu=[4],
                seq_lens=[16],
                seq_lens_cpu=[16],
                last_loc=[103],
                extend_num_tokens=12,
                swa_tail_len=8,
            ),
        )
        self.assertEqual(
            mgr.started_swa,
            [(PXS, 98, DECODE_SWA_RAIDEN, [62, 63], [301, 302])],
        )

    def test_window_reserve_is_kept_for_admission(self):
        puller, tree, allocator, mgr = _hybrid_offered()
        puller._min_free_swa_tokens = 4
        allocator.free_swa_tokens = 0
        tree.swa_evictable_tokens = 10
        req = _req()
        puller.advance([req])
        mgr.ready[ROOM] = HYBRID_READY
        mgr.swa_ready[ROOM] = HYBRID_SWA_READY
        puller.advance([req])
        self.assertEqual((allocator.swa_tail_allocs, mgr.started), ([], []))
        self.assertEqual(puller.stats["no_memory"], 1)
        self.assertEqual((tree.lock_refs, mgr.resolved), (0, {ROOM}))

    def test_ready_for_another_window_abandons_the_pull(self):
        puller, tree, allocator, mgr = _hybrid_offered()
        req = _req()
        puller.advance([req])
        mgr.ready[ROOM] = HYBRID_READY
        mgr.swa_ready[ROOM] = (PXS, 98, DECODE_SWA_RAIDEN, 2, 0)
        puller.advance([req])
        self.assertEqual(mgr.started, [])
        self.assertEqual(allocator.freed, [NEW_INDICES])
        self.assertEqual((tree.lock_refs, mgr.resolved), (0, {ROOM}))

    def test_done_needs_both_reads_and_commit_marks_the_window_start(self):
        puller, tree, allocator, mgr, req = _hybrid_pulling()
        mgr.read_states[PX] = "done"
        mgr.read_states[PXS] = "pending"
        puller.advance([req])
        self.assertIsNone(puller._plans[ROOM].agreed_done)
        self.assertEqual(mgr.resolved, set())
        mgr.read_states[PXS] = "done"
        puller.advance([req])
        self.assertIs(puller._plans[ROOM].agreed_done, True)
        puller.commit(req)
        (params,) = tree.inserts
        self.assertEqual(list(params.key.token_ids), list(range(1000, 1016)))
        self.assertEqual(params.value.tolist(), [100, 101, 102, 103] + NEW_INDICES)
        self.assertEqual((params.prev_prefix_len, params.inserted_len), (4, 16))
        self.assertEqual(params.component_evicted_seqlens, {ComponentType.SWA: 8})
        self.assertEqual((tree.lock_refs, allocator.freed), (0, []))
        # The locked window stays out of the SWA pool until admission.
        self.assertEqual((tree.pin_refs, puller._inflight_swa_tokens()), (1, 8))
        puller.advance([])
        self.assertEqual((tree.pin_refs, puller._inflight_swa_tokens()), (0, 0))

    def test_failed_window_read_computes_and_frees_once_both_settle(self):
        puller, tree, allocator, mgr, req = _hybrid_pulling()
        mgr.read_states[PX] = "pending"
        mgr.read_states[PXS] = "failed"
        puller.advance([req])
        self.assertIsNone(puller._plans[ROOM].agreed_done)
        mgr.read_states[PX] = "done"
        puller.advance([req])
        self.assertIs(puller._plans[ROOM].agreed_done, False)
        puller.commit(req)
        self.assertEqual((tree.inserts, allocator.freed), ([], [NEW_INDICES]))
        # Dropped while the window still lands: the pages wait for it.
        puller, tree, allocator, mgr, req = _hybrid_pulling()
        mgr.read_states[PX] = "done"
        mgr.read_states[PXS] = "pending"
        puller.release(req)
        self.assertEqual(allocator.freed, [])
        self.assertEqual(puller._inflight_swa_tokens(), 8)
        mgr.read_states[PXS] = "done"
        puller.advance([])
        self.assertEqual(allocator.freed, [NEW_INDICES])

    def test_ranks_planning_different_windows_give_up(self):
        def other_rank_planned_another_window(rows, op):
            if op != dist.ReduceOp.MAX or len(rows[0]) != 5:
                return [list(row) for row in rows]
            return [row[:3] + [row[3] + 1, row[4] - 1] for row in rows]

        puller, tree, allocator, mgr = _hybrid_offered()
        puller._all_ranks_reduce = other_rank_planned_another_window
        puller.advance([_req()])
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual((tree.lock_refs, mgr.resolved), (0, {ROOM}))


if __name__ == "__main__":
    unittest.main()
