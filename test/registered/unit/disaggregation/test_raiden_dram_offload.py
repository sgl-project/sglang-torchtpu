"""The Raiden handshake of the KV offload's recall: the decode offers the host
pages of a pushed prefix it holds (in its PULL_REQ), the prefill asks for the
range it lacks (PREFIX_PULL), the decode registers those host pages for a
native read (PREFIX_READY). Also the pair-level refusals: a peer with the
switch in the other state, and a decode that needs a resharded transfer.
Manager fixtures come from test_raiden_failure_handling.py."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    PrefixReady,
    PullReq,
    ReshardReq,
    RoomFailed,
    VictimAck,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.testing.decode_dram_offload import (
    DECODE_ADDR,
    DECODE_RAIDEN,
    PREFILL_ADDR,
    PX,
    ROOM,
    SWITCH,
    DecodeOffersItsPrefixFixture,
    PrefillRecallsIntoItsHostPagesFixture,
    decode_manager,
    host_transfer_of,
    prefill_manager,
    pull_req,
    register_decode,
    sender_on,
    sent_frames,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDecodeOffersItsPrefix(DecodeOffersItsPrefixFixture, CustomTestCase):
    def test_pull_req_names_the_offered_pages(self):
        mgr = decode_manager()
        register_decode(mgr, offered=[3, 4])
        ((endpoint, frames),) = sent_frames(mgr)
        self.assertEqual(endpoint, PREFILL_ADDR)
        req = PullReq.from_frames(frames)
        # The offer covers exactly the pages it names.
        self.assertEqual((req.prefix_blocks, req.kv_offload), ([3, 4], True))
        self.assertEqual(
            mgr.dram_offload_channel._offers[ROOM].prefill_addr, PREFILL_ADDR
        )

    def test_request_without_an_offer_names_no_pages(self):
        mgr = decode_manager()
        register_decode(mgr)
        ((_, frames),) = sent_frames(mgr)
        self.assertEqual(PullReq.from_frames(frames).prefix_blocks, [])
        self.assertNotIn(ROOM, mgr.dram_offload_channel._offers)

    def test_decode_with_the_switch_off_says_so(self):
        mgr = decode_manager()
        mgr.dram_offload_channel = None
        register_decode(mgr)
        ((_, frames),) = sent_frames(mgr)
        req = PullReq.from_frames(frames)
        self.assertEqual((req.prefix_blocks, req.kv_offload), ([], False))

    def test_pull_registers_the_asked_host_pages_and_replies_ready(self):
        mgr = decode_manager()
        register_decode(mgr, offered=[3, 4, 5, 6])
        ready = self._served(mgr, 1, 2)
        host_transfer_of(mgr).register_read.assert_called_once()
        req_id, uuid, blocks = host_transfer_of(mgr).register_read.call_args.args
        self.assertEqual(
            host_transfer_of(mgr).register_read.call_args.kwargs, {"source": "host"}
        )
        self.assertEqual((req_id, blocks), (PX, [4, 5]))
        self.assertGreater(uuid, 0)
        # The prefill reads from the host pool's control endpoint.
        self.assertEqual(
            ready, PrefixReady(room=ROOM, uuid=uuid, endpoint=DECODE_RAIDEN)
        )
        self.assertEqual(len(sent_frames(mgr)), 2)

    def test_pull_after_the_queue_released_the_offer_is_refused(self):
        # The request left the transfer queue: its host pin is going, so a
        # late PULL must not register the pages.
        mgr = decode_manager()
        register_decode(mgr, offered=[3, 4])
        mgr.dram_offload_channel.release_prefix_offer(ROOM)
        ready = self._served(mgr, 0, 2)
        self.assertEqual((ready.uuid, ready.endpoint), (0, ""))
        host_transfer_of(mgr).register_read.assert_not_called()
        self.assertFalse(mgr.dram_offload_channel.host_prefix_pending(ROOM))
        self.assertFalse(mgr.is_failed(ROOM))

    def test_registered_pages_stay_owned_until_the_read_settles(self):
        for outcome in ("sent", "failed"):
            with self.subTest(outcome=outcome):
                mgr = decode_manager()
                register_decode(mgr, offered=[3, 4])
                mgr._dispatch(
                    PrefixPull(room=ROOM, first_page=0, num_pages=2).to_frames()
                )
                self.assertTrue(mgr.room_has_active_io(ROOM))
                mgr.dram_offload_channel.drain_host_reads()
                self.assertTrue(mgr.dram_offload_channel.host_prefix_pending(ROOM))
                host_transfer_of(mgr).poll_transfer.return_value = outcome
                mgr.dram_offload_channel.drain_host_reads()
                self.assertFalse(mgr.dram_offload_channel.host_prefix_pending(ROOM))
                self.assertFalse(mgr.room_has_active_io(ROOM))
                host_transfer_of(mgr).forget_transfer.assert_called_once_with(PX)
                # A read the prefill could not finish costs the decode nothing.
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})


class TestPrefillWaitsForTheRecall(CustomTestCase):
    def test_offered_pages_keep_the_room_bootstrapping(self):
        mgr = prefill_manager()
        sender = sender_on(mgr)
        mgr._dispatch(pull_req())
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))
        mgr.dram_offload_channel.resolve_prefix_pull(ROOM)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_pull_req_without_pages_does_not_hold_the_room(self):
        mgr = prefill_manager()
        sender = sender_on(mgr)
        mgr._dispatch(pull_req(prefix_blocks=()))
        self.assertIsNone(mgr.dram_offload_channel.prefix_ready(ROOM))
        self.assertIsNone(mgr.dram_offload_channel.take_prefix_offer(ROOM))
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)


class TestPrefillRecallsIntoItsHostPages(
    PrefillRecallsIntoItsHostPagesFixture, CustomTestCase
):
    def test_ask_sends_prefix_pull_to_the_decode(self):
        mgr = self._armed()
        self.assertTrue(mgr.dram_offload_channel.ask_prefix_pull(ROOM, 1, 2))
        ((endpoint, frames),) = sent_frames(mgr)
        self.assertEqual(endpoint, DECODE_ADDR)
        self.assertEqual(
            PrefixPull.from_frames(frames),
            PrefixPull(room=ROOM, first_page=1, num_pages=2),
        )
        mgr._dispatch(self._ready().to_frames())
        self.assertEqual(
            mgr.dram_offload_channel.prefix_ready(ROOM), (99, DECODE_RAIDEN)
        )

    def test_read_lands_in_host_pages_and_its_outcome_fails_no_room(self):
        for report, done in (("received", True), ("failed", False)):
            with self.subTest(report=report):
                mgr = self._armed()
                offload = mgr.dram_offload_channel
                self.assertTrue(
                    offload.start_host_read(PX, 99, DECODE_RAIDEN, [4, 5], [40, 41])
                )
                host_transfer_of(mgr).start_read.assert_called_once_with(
                    PX,
                    99,
                    DECODE_RAIDEN,
                    [4, 5],
                    [40, 41],
                    destination="host",
                )
                offload.drain_host_reads()
                self.assertIsNone(offload.poll_host_read(PX))
                host_transfer_of(mgr).poll_transfer.return_value = report
                offload.drain_host_reads()
                self.assertIs(offload.poll_host_read(PX), done)
                self.assertFalse(mgr.is_failed(ROOM))
                self.assertEqual(mgr.failure_records, {})
                # The record goes with the room.
                mgr.clear_room(ROOM)
                self.assertIs(offload.poll_host_read(PX), False)
                self.assertEqual(offload._reads, {})


class TestVictimMessages(CustomTestCase):
    def test_channel_is_built_before_the_manager_has_its_state(self):
        # The manager creates the channel while its __init__ registers the
        # message handlers, before its locks and room tables exist.
        channel = DramOffloadChannel(SimpleNamespace())
        self.assertTrue(channel.message_handlers())

    def test_push_and_its_answer_reach_the_other_role(self):
        prefill, decode = prefill_manager(), decode_manager()
        prefill.dram_offload_channel.victim_namespace = "ab"
        self.assertTrue(
            prefill.dram_offload_channel.send_victim_push(
                decode=("10.0.0.2", 5555),
                seq=3,
                tokens=[1, 2, 3, 4],
                blocks=[9],
                uuid=5,
                extra_key=None,
                cache_salt=None,
            )
        )
        ((endpoint, frames),) = sent_frames(prefill)
        self.assertEqual(endpoint, DECODE_ADDR)
        decode._dispatch(frames)
        push = decode.dram_offload_channel.take_push(("ab", 3))
        self.assertEqual((push.tokens, push.blocks, push.uuid), ([1, 2, 3, 4], [9], 5))
        # The decode reads from the prefill's host pool.
        self.assertEqual(push.address, DECODE_RAIDEN)
        self.assertIsNone(decode.dram_offload_channel.take_push(("ab", 3)))
        self.assertIsNone(prefill.dram_offload_channel.victim_reply(3))
        decode.dram_offload_channel.send_victim_ack(
            endpoint=push.ack_endpoint, seq=3, stored=True
        )
        prefill._dispatch(sent_frames(decode)[-1][1])
        self.assertTrue(prefill.dram_offload_channel.victim_reply(3))

    def test_answer_to_a_push_that_is_no_longer_open_is_dropped(self):
        prefill = prefill_manager()
        prefill.dram_offload_channel.forget_victim(3)
        prefill._dispatch(VictimAck(seq=3, stored=True).to_frames())
        self.assertIsNone(prefill.dram_offload_channel.victim_reply(3))
        self.assertEqual(prefill.dram_offload_channel._replies, {})


class TestPairRefusals(CustomTestCase):
    """A pair that cannot offload must fail its requests with the reason, not
    serve them as if the switch were off."""

    def _failure(self, mgr):
        self.assertTrue(mgr.is_failed(ROOM))
        endpoint, frames = sent_frames(mgr)[-1]
        self.assertEqual(endpoint, DECODE_ADDR)
        failed = RoomFailed.from_frames(frames)
        self.assertEqual(failed.reason, mgr.failure_records[ROOM])
        return failed.reason

    def test_switch_on_for_one_role_only_fails_the_room(self):
        switch = SWITCH
        for prefill_on, decode_on in ((True, False), (False, True)):
            with self.subTest(prefill_on=prefill_on, decode_on=decode_on):
                mgr = prefill_manager()
                sender_on(mgr)
                with switch.override(prefill_on):
                    mgr._dispatch(pull_req(kv_offload=decode_on))
                reason = self._failure(mgr)
                self.assertIn(switch.name, reason)
                self.assertIn("same value on both roles", reason)
                self.assertNotIn(ROOM, mgr._prefill.decode_addr)

    def test_matching_switch_serves_the_room(self):
        switch = SWITCH
        for value in (True, False):
            with self.subTest(switch=value):
                mgr = prefill_manager()
                sender_on(mgr)
                with switch.override(value):
                    mgr._dispatch(pull_req(kv_offload=value))
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
        mgr = prefill_manager()
        sender_on(mgr)
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            mgr._dispatch(req)
        reason = self._failure(mgr)
        self.assertIn("resharding", reason)
        self.assertNotIn(ROOM, mgr._prefill.reshard_req)


if __name__ == "__main__":
    unittest.main()
