"""The Raiden handshake of the KV offload's recall: the decode offers the host
pages of a pushed prefix it holds (in its PULL_REQ), the prefill asks for the
range it lacks (PREFIX_PULL), the decode registers those host pages for a
native read (PREFIX_READY). Also the pair-level refusals: a peer with the
switch in the other state, and a decode that needs a resharded transfer.
Manager fixtures come from test_raiden_failure_handling.py."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.disaggregation.raiden.decode_dram_offload import DecodeDramOffload
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    PrefixReady,
    PullReq,
    ReshardReq,
    RoomFailed,
    VictimAck,
    VictimDone,
    VictimPush,
)
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_sibling("test_raiden_failure_handling")
ROOM = _base.ROOM
_sent_frames = _base._sent_frames
SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM
PAGE = 4
# Request id of the prefill's read of the decode-held prefix.
PX = f"{ROOM}#px"
PREFILL_INFO = {"rank_ip": "10.0.0.1", "rank_port": 4444}
PREFILL_ADDR = _tcp_endpoint(ip="10.0.0.1", port=4444)
DECODE_ADDR = _tcp_endpoint(ip="10.0.0.2", port=5555)
# Control endpoint of the decode's HiCache host pool.
DECODE_RAIDEN = "10.0.0.2:7001"


def _with_offload(mgr):
    """Switch the offload on for a fixture manager: KV pools bound and a
    HiCache host pool whose control endpoint is DECODE_RAIDEN."""
    transfer = Mock()
    transfer.host_control_address.side_effect = lambda ip: f"{ip}:7001"
    transfer.poll_transfer.return_value = None
    mgr._kv_caches = [object()]
    mgr.local_ip = "10.0.0.2"
    mgr.rank_port = 5555
    mgr._send_multipart_locked = Mock()
    mgr.decode_dram_offload = DecodeDramOffload(mgr)
    mgr.decode_dram_offload.bind_prefix_host_pool(SimpleNamespace(transfer=transfer))
    mgr._message_handlers.update(mgr.decode_dram_offload.message_handlers())
    return mgr


def _full(mgr):
    return mgr.decode_dram_offload._host_pool.transfer


def _prefill_manager():
    return _with_offload(_base._gate_manager())


def _sender(mgr):
    return _base._gate_sender(mgr, ROOM)


def _decode_manager():
    mgr = _with_offload(_base._decode_manager())
    mgr.kv_args = SimpleNamespace(page_size=PAGE)
    return mgr


def _pull_req(prefix_blocks=(3, 4, 5, 6), kv_offload=False):
    return PullReq(
        room=ROOM,
        decode_ip="10.0.0.2",
        decode_port=5555,
        decode_pages=4,
        prefix_blocks=list(prefix_blocks),
        kv_offload=kv_offload,
    ).to_frames()


def _register(mgr, offered=()):
    """Send the room's PULL_REQ, after the decode offered ``offered`` pages."""
    if mgr.decode_dram_offload is not None:
        mgr.decode_dram_offload.offer_resident_prefix(ROOM, list(offered))
    mgr.register_decode_request(ROOM, [21, 22, 23], [], None, [PREFILL_INFO])


class _DecodeOffersItsPrefixFixture:
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


class TestDecodeOffersItsPrefix(_DecodeOffersItsPrefixFixture, CustomTestCase):
    def test_pull_req_names_the_offered_pages(self):
        mgr = _decode_manager()
        _register(mgr, offered=[3, 4])
        ((endpoint, frames),) = _sent_frames(mgr)
        self.assertEqual(endpoint, PREFILL_ADDR)
        req = PullReq.from_frames(frames)
        # The offer covers exactly the pages it names.
        self.assertEqual((req.prefix_blocks, req.kv_offload), ([3, 4], True))
        self.assertEqual(mgr.decode_dram_offload.prefill_addr[ROOM], PREFILL_ADDR)

    def test_request_without_an_offer_names_no_pages(self):
        mgr = _decode_manager()
        _register(mgr)
        ((_, frames),) = _sent_frames(mgr)
        self.assertEqual(PullReq.from_frames(frames).prefix_blocks, [])
        self.assertNotIn(ROOM, mgr.decode_dram_offload.prefill_addr)

    def test_decode_with_the_switch_off_says_so(self):
        mgr = _decode_manager()
        mgr.decode_dram_offload = None
        _register(mgr)
        ((_, frames),) = _sent_frames(mgr)
        req = PullReq.from_frames(frames)
        self.assertEqual((req.prefix_blocks, req.kv_offload), ([], False))

    def test_pull_registers_the_asked_host_pages_and_replies_ready(self):
        mgr = _decode_manager()
        _register(mgr, offered=[3, 4, 5, 6])
        ready = self._served(mgr, 1, 2)
        _full(mgr).register_read.assert_called_once()
        req_id, uuid, blocks = _full(mgr).register_read.call_args.args
        self.assertEqual(_full(mgr).register_read.call_args.kwargs, {"source": "host"})
        self.assertEqual((req_id, blocks), (PX, [4, 5]))
        self.assertGreater(uuid, 0)
        # The prefill reads from the host pool's control endpoint.
        self.assertEqual(
            ready, PrefixReady(room=ROOM, uuid=uuid, endpoint=DECODE_RAIDEN)
        )
        self.assertEqual(len(_sent_frames(mgr)), 2)

    def test_pull_after_the_queue_released_the_offer_is_refused(self):
        # The request left the transfer queue: its host pin is going, so a
        # late PULL must not register the pages.
        mgr = _decode_manager()
        _register(mgr, offered=[3, 4])
        mgr.decode_dram_offload.release_prefix_offer(ROOM)
        ready = self._served(mgr, 0, 2)
        self.assertEqual((ready.uuid, ready.endpoint), (0, ""))
        _full(mgr).register_read.assert_not_called()
        self.assertEqual(mgr.decode_dram_offload.host_reads, {})
        self.assertFalse(mgr.is_failed(ROOM))

    def test_registered_pages_stay_owned_until_the_read_settles(self):
        for outcome in ("sent", "failed"):
            with self.subTest(outcome=outcome):
                mgr = _decode_manager()
                _register(mgr, offered=[3, 4])
                mgr._dispatch(
                    PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames()
                )
                self.assertTrue(mgr.room_has_active_io(ROOM))
                mgr.decode_dram_offload.drain_host_reads()
                self.assertTrue(mgr.decode_dram_offload.host_prefix_pending(ROOM))
                _full(mgr).poll_transfer.return_value = outcome
                mgr.decode_dram_offload.drain_host_reads()
                self.assertFalse(mgr.decode_dram_offload.host_prefix_pending(ROOM))
                self.assertFalse(mgr.room_has_active_io(ROOM))
                _full(mgr).forget_transfer.assert_called_once_with(PX)
                # A read the prefill could not finish costs the decode nothing.
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})


class TestPrefillWaitsForTheRecall(CustomTestCase):
    def test_offered_pages_keep_the_room_bootstrapping(self):
        mgr = _prefill_manager()
        sender = _sender(mgr)
        mgr._dispatch(_pull_req())
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))
        mgr.decode_dram_offload.resolve_prefix_pull(ROOM)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_pull_req_without_pages_does_not_hold_the_room(self):
        mgr = _prefill_manager()
        sender = _sender(mgr)
        mgr._dispatch(_pull_req(prefix_blocks=()))
        self.assertFalse(mgr.decode_dram_offload.prefix_offer_available(ROOM))
        self.assertIsNone(mgr.decode_dram_offload.take_prefix_offer(ROOM))
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)


class _PrefillRecallsIntoItsHostPagesFixture:
    def _armed(self):
        """A prefill manager whose decode offered pages [3, 4, 5, 6] for ROOM."""
        mgr = _prefill_manager()
        _sender(mgr)
        mgr._dispatch(_pull_req())
        self.assertEqual(mgr.decode_dram_offload.take_prefix_offer(ROOM), [3, 4, 5, 6])
        return mgr

    @staticmethod
    def _ready():
        return PrefixReady(room=ROOM, uuid=99, endpoint=DECODE_RAIDEN)


class TestPrefillRecallsIntoItsHostPages(
    _PrefillRecallsIntoItsHostPagesFixture, CustomTestCase
):
    def test_ask_sends_prefix_pull_to_the_decode(self):
        mgr = self._armed()
        self.assertTrue(mgr.decode_dram_offload.ask_prefix_pull(ROOM, 1, 2))
        ((endpoint, frames),) = _sent_frames(mgr)
        self.assertEqual(endpoint, DECODE_ADDR)
        self.assertEqual(
            PrefixPull.from_frames(frames),
            PrefixPull(room=ROOM, first_page=1, num_pages=2),
        )
        mgr._dispatch(self._ready().to_frames())
        self.assertEqual(
            mgr.decode_dram_offload.prefix_pull_ready(ROOM), (PX, 99, DECODE_RAIDEN)
        )

    def test_read_lands_in_host_pages_and_its_outcome_fails_no_room(self):
        for report, state in (("received", "done"), ("failed", "failed")):
            with self.subTest(report=report):
                mgr = self._armed()
                offload = mgr.decode_dram_offload
                offload.start_host_prefix_read(PX, 99, DECODE_RAIDEN, [4, 5], [40, 41])
                _full(mgr).start_read.assert_called_once_with(
                    PX,
                    99,
                    DECODE_RAIDEN,
                    [4, 5],
                    None,
                    local_host_block_ids=[40, 41],
                    destination="host",
                )
                offload.drain_host_reads()
                self.assertEqual(offload.prefix_read_state(PX), "pending")
                _full(mgr).poll_transfer.return_value = report
                offload.drain_host_reads()
                self.assertEqual(offload.prefix_read_state(PX), state)
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})
                # The record goes with the room.
                mgr.clear_room(ROOM)
                self.assertEqual(offload.prefix_read_state(PX), "unknown")


class TestVictimMessages(CustomTestCase):
    def test_push_ack_and_done_reach_the_other_role(self):
        prefill, decode = _prefill_manager(), _decode_manager()
        prefill.decode_dram_offload.victim_namespace = "ab"
        peer = ("10.0.0.2", 5555)
        self.assertTrue(
            prefill.decode_dram_offload.send_victim_push(
                decode=peer, seq=3, tokens=[1, 2, 3, 4], blocks=[9], uuid=5, address="a"
            )
        )
        prefill.decode_dram_offload.send_victim_done(decode=peer, seq=3, ok=True)
        for endpoint, frames in _sent_frames(prefill):
            self.assertEqual(endpoint, DECODE_ADDR)
            decode._dispatch(frames)
        push = decode.decode_dram_offload.take_victim_message(VictimPush, ("ab", 3))
        self.assertEqual((push.tokens, push.blocks, push.uuid), ([1, 2, 3, 4], [9], 5))
        self.assertIsNone(
            decode.decode_dram_offload.take_victim_message(VictimPush, ("ab", 3))
        )
        self.assertTrue(
            decode.decode_dram_offload.victim_messages(VictimDone)[("ab", 3)].ok
        )
        decode.decode_dram_offload.send_victim_ack(
            endpoint=push.ack_endpoint, ack=VictimAck(seq=3, stored=True)
        )
        prefill._dispatch(_sent_frames(decode)[-1][1])
        self.assertTrue(
            prefill.decode_dram_offload.take_victim_message(VictimAck, 3).stored
        )

    def test_ack_of_a_push_that_is_no_longer_open_is_dropped(self):
        prefill = _prefill_manager()
        prefill.decode_dram_offload.forget_victim(3)
        prefill._dispatch(VictimAck(seq=3, stored=True).to_frames())
        self.assertEqual(prefill.decode_dram_offload.victim_messages(VictimAck), {})


class TestPairRefusals(CustomTestCase):
    """A pair that cannot offload must fail its requests with the reason, not
    serve them as if the switch were off."""

    def _failure(self, mgr):
        self.assertTrue(mgr.is_failed(ROOM))
        endpoint, frames = _sent_frames(mgr)[-1]
        self.assertEqual(endpoint, DECODE_ADDR)
        failed = RoomFailed.from_frames(frames)
        self.assertEqual(failed.reason, mgr.failure_records[ROOM])
        return failed.reason

    def test_switch_on_for_one_role_only_fails_the_room(self):
        switch = SWITCH
        for prefill_on, decode_on in ((True, False), (False, True)):
            with self.subTest(prefill_on=prefill_on, decode_on=decode_on):
                mgr = _prefill_manager()
                _sender(mgr)
                with switch.override(prefill_on):
                    mgr._dispatch(_pull_req(kv_offload=decode_on))
                reason = self._failure(mgr)
                self.assertIn(switch.name, reason)
                self.assertIn("same value on both roles", reason)
                self.assertNotIn(ROOM, mgr._prefill.decode_addr)

    def test_matching_switch_serves_the_room(self):
        switch = SWITCH
        for value in (True, False):
            with self.subTest(switch=value):
                mgr = _prefill_manager()
                _sender(mgr)
                with switch.override(value):
                    mgr._dispatch(_pull_req(prefix_blocks=(), kv_offload=value))
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertIn(ROOM, mgr._prefill.decode_addr)

    def test_resharded_transfer_is_refused_while_the_switch_is_on(self):
        req = ReshardReq(
            room=ROOM,
            decode_ip="10.0.0.2",
            decode_port=5555,
            push_endpoint="10.0.0.2:31030",
            decode_blocks=[1, 2],
            dst_attn_tp=2,
            decode_swa_blocks=[],
            swa_push_endpoint="",
        ).to_frames()
        mgr = _prefill_manager()
        _sender(mgr)
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            mgr._dispatch(req)
        reason = self._failure(mgr)
        self.assertIn("resharding", reason)
        self.assertNotIn(ROOM, mgr._prefill.reshard_req)


if __name__ == "__main__":
    unittest.main()
