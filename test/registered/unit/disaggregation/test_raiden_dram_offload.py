"""The Raiden handshake of the KV offload's recall: the decode offers the host
pages of a pushed prefix it holds (in its PULL_REQ), the prefill asks for the
range it lacks (PREFIX_PULL), the decode registers those host pages for a
native read (PREFIX_READY). Also the pair-level refusals: a peer with the
switch in the other state, and a decode that needs a resharded transfer.
Manager fixtures come from test_raiden_failure_handling.py."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.disaggregation.raiden.serde import (
    ReshardReq,
    RoomFailed,
    VictimAck,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.testing.decode_dram_offload import (
    DECODE_ADDR,
    DECODE_RAIDEN,
    ROOM,
    SWITCH,
    decode_manager,
    prefill_manager,
    pull_req,
    sender_on,
    sent_frames,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
