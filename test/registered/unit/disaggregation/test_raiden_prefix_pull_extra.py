"""Lower-priority unit tests of the recall handshake: wire round trips and validation, refusals of unservable pulls, the bootstrap gate's less common paths, and host reads that settle after their room was cleared. Fixtures come from test_raiden_prefix_pull.py."""

import errno
import importlib.util
import sys
import time
import unittest
from pathlib import Path

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    PrefixPull,
    PrefixReady,
    PullReq,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_main = _load_sibling("test_raiden_prefix_pull")
DECODE_RAIDEN = _main.DECODE_RAIDEN
PX = _main.PX
ROOM = _main.ROOM
_DecodeOffersItsPrefixFixture = _main._DecodeOffersItsPrefixFixture
_PrefillRecallsIntoItsHostPagesFixture = _main._PrefillRecallsIntoItsHostPagesFixture
_decode_manager = _main._decode_manager
_full = _main._full
_prefill_manager = _main._prefill_manager
_pull_req = _main._pull_req
_register = _main._register
_sender = _main._sender
_sent_frames = _main._sent_frames


class TestPrefixPullWire(CustomTestCase):
    def test_pull_req_without_prefix_fields_offers_nothing(self):
        # A decode without the feature encodes no prefix fields.
        legacy = PullReq(
            room=ROOM, decode_ip="10.0.0.2", decode_port=5555, decode_pages=4
        )
        req = PullReq.from_frames(legacy.to_frames())
        self.assertEqual((req.prefix_blocks, req.kv_offload), ([], False))

    def test_messages_round_trip(self):
        for msg in (
            PullReq(
                room=ROOM,
                decode_ip="10.0.0.2",
                decode_port=5555,
                decode_pages=4,
                prefix_blocks=[3, 4, 5],
                kv_offload=True,
            ),
            PrefixPull(room=ROOM, first_page=1, num_pages=2),
            PrefixReady(room=ROOM, uuid=99, endpoint=DECODE_RAIDEN),
        ):
            with self.subTest(message=type(msg).__name__):
                self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)

    def test_invalid_and_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            PrefixPull(room=ROOM, first_page=0, num_pages=0)
        with self.assertRaises(ValueError):
            PrefixPull(room=-1, first_page=0, num_pages=1)
        payload = b'{"room": 7, "first_page": 0, "num_pages": 1, "extra": 1}'
        with self.assertRaises(ValueError):
            PrefixPull.from_frames([MessageTag.PREFIX_PULL, payload])


class TestDecodeOffersItsPrefixExtra(_DecodeOffersItsPrefixFixture, CustomTestCase):
    def test_unreachable_prefill_costs_only_the_recall(self):
        mgr = _decode_manager()
        _register(mgr, offered=[3, 4])
        mgr._send_multipart_locked.side_effect = zmq.ZMQError(errno.EHOSTUNREACH)
        mgr._dispatch(PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames())
        self.assertFalse(mgr.is_failed(ROOM))
        # Any other errno is not one room's problem.
        mgr._send_multipart_locked.side_effect = zmq.ZMQError(zmq.ETERM)
        with self.assertRaises(zmq.ZMQError):
            mgr._dispatch(PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames())

    def test_unservable_pull_is_refused_with_no_pages(self):
        def past_the_offer(mgr):
            pass

        def aborted(mgr):
            mgr._aborted[ROOM] = time.time()

        def rejected_registration(mgr):
            _full(mgr).register_read.side_effect = RuntimeError("injected")

        for arm, first_page in (
            (past_the_offer, 1),
            (aborted, 0),
            (rejected_registration, 0),
        ):
            with self.subTest(arm.__name__):
                mgr = _decode_manager()
                _register(mgr, offered=[3, 4])
                arm(mgr)
                sent = len(_sent_frames(mgr))
                ready = self._served(mgr, first_page, 2)
                self.assertEqual((ready.uuid, ready.endpoint), (0, ""))
                self.assertEqual(len(_sent_frames(mgr)), sent + 1)
                self.assertEqual(mgr.decode_dram_offload.host_reads, {})
                self.assertFalse(mgr.room_has_active_io(ROOM))
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})


class TestPrefillWaitsForTheRecallExtra(CustomTestCase):
    def test_engine_without_kv_pools_takes_no_offer(self):
        mgr = _prefill_manager()
        mgr._kv_caches = None
        sender = _sender(mgr)
        mgr._dispatch(_pull_req(prefix_blocks=(3, 4)))
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertFalse(mgr.decode_dram_offload.prefix_offer_available(ROOM))

    def test_offer_is_handed_out_once(self):
        mgr = _prefill_manager()
        _sender(mgr)
        mgr._dispatch(_pull_req(prefix_blocks=(3, 4)))
        self.assertTrue(mgr.decode_dram_offload.prefix_offer_available(ROOM))
        self.assertEqual(mgr.decode_dram_offload.take_prefix_offer(ROOM), [3, 4])
        self.assertFalse(mgr.decode_dram_offload.prefix_offer_available(ROOM))
        self.assertIsNone(mgr.decode_dram_offload.take_prefix_offer(ROOM))
        # The gate stays shut until the puller settles the room.
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))

    def test_unsettled_pull_is_given_up_after_the_bootstrap_timeout(self):
        mgr = _prefill_manager()
        mgr.bootstrap_timeout = 0.5
        sender = _sender(mgr)
        mgr._dispatch(_pull_req())
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        mgr.decode_dram_offload.offer_since[ROOM] = time.time() - 1
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertTrue(mgr.decode_dram_offload.prefix_pull_resolved(ROOM))
        self.assertNotIn(ROOM, mgr.failure_records)


class TestPrefillRecallsIntoItsHostPagesExtra(
    _PrefillRecallsIntoItsHostPagesFixture, CustomTestCase
):
    def test_ready_is_kept_only_after_a_pull_was_asked(self):
        mgr = self._armed()
        mgr._dispatch(self._ready().to_frames())
        self.assertIsNone(mgr.decode_dram_offload.prefix_pull_ready(ROOM))

    def test_read_is_refused_after_abort_or_without_a_host_pool(self):
        mgr = self._armed()
        mgr._aborted[ROOM] = time.time()
        with self.assertRaisesRegex(RuntimeError, "after abort"):
            mgr.decode_dram_offload.start_host_prefix_read(
                PX, 99, DECODE_RAIDEN, [4, 5], [40, 41]
            )
        mgr = self._armed()
        mgr.decode_dram_offload.bind_prefix_host_pool(None)
        with self.assertRaisesRegex(RuntimeError, "HiCache pool"):
            mgr.decode_dram_offload.start_host_prefix_read(
                PX, 99, DECODE_RAIDEN, [4, 5], [40, 41]
            )
        self.assertEqual(mgr.decode_dram_offload.host_reads, {})

    def test_a_read_settling_after_its_room_was_cleared_leaves_no_record(self):
        for report in ("received", "failed"):
            with self.subTest(report=report):
                mgr = self._armed()
                offload = mgr.decode_dram_offload
                offload.start_host_prefix_read(PX, 99, DECODE_RAIDEN, [4, 5], [40, 41])
                # As the sender's clear() does.
                mgr.clear_status(ROOM)
                mgr.clear_room(ROOM)
                # The native read outlives the room until it reports.
                self.assertEqual(offload.prefix_read_state(PX), "pending")
                _full(mgr).poll_transfer.return_value = report
                offload.drain_host_reads()
                _full(mgr).forget_transfer.assert_called_once_with(PX)
                self.assertEqual(offload.prefix_read_state(PX), "unknown")
                self.assertEqual(offload._read_results, {})


if __name__ == "__main__":
    unittest.main()
