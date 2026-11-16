"""Raiden channel of the offload for a pipeline-parallel prefill
(sglang_tpu/disaggregation/raiden/dram_offload_channel.py): a decode rank
collects a pushed branch's header and the parts of the stages it hosts, reads
each stage's pages into the pool of that stage, and registers a stage's share
of an offered prefix when that stage asks. A real channel on a bare manager;
the stage pools are fake native transfers."""

import os
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload.decode_stage_branches import (
    stage_transfer_id,
)
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    PrefixReady,
    VictimPush,
)
from sglang_tpu.testing.decode_dram_offload import (
    FakeHostTransfer,
    _bare_decode_manager,
    sent_frames,
    with_offload,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYOUT = [["bfloat16", [4, 16, 8]]]


class _Pools:
    """decode_stage_branches.StagePoolSet as the channel uses it."""

    def __init__(self, stages):
        self.transfers = {
            (stage, swa): FakeHostTransfer()
            for stage in stages
            for swa in (False, True)
        }

    def transfer(self, stage, *, swa=False):
        return self.transfers[(stage, swa)]


def _push(stage, *, tokens=(), uuid=9, seq=3, **fields):
    values = dict(
        prefill="ab12",
        seq=seq,
        ack_endpoint=f"tcp://10.0.0.1:{7000 + stage}",
        tokens=list(tokens),
        blocks=[40 + stage, 50 + stage] if uuid else [],
        uuid=uuid,
        address=f"10.0.0.1:{9000 + stage}" if uuid else "",
        stage=stage,
        stages=4,
        layout=LAYOUT if uuid else [],
        page_bytes=1024 if uuid else 0,
    )
    values.update(fields)
    return VictimPush(**values)


class TestStageWire(CustomTestCase):
    def test_stage_fields_round_trip_and_stay_off_a_tensor_parallel_push(self):
        staged = _push(
            2,
            window_pages=2,
            swa_blocks=[7],
            swa_address="10.0.0.1:9100",
            swa_layout=LAYOUT,
            swa_page_bytes=512,
        )
        self.assertEqual(VictimPush.from_frames(staged.to_frames()), staged)
        plain = VictimPush(
            prefill="ab12", seq=3, ack_endpoint="e", tokens=[1], blocks=[4], uuid=9
        )
        decoded = VictimPush.from_frames(plain.to_frames())
        self.assertEqual((decoded.stage, decoded.stages, decoded.layout), (-1, 0, []))
        pull = PrefixPull(
            room=5, first_page=0, num_pages=2, stage=3, reply_endpoint="tcp://x:1"
        )
        self.assertEqual(PrefixPull.from_frames(pull.to_frames()), pull)
        self.assertEqual(PrefixPull(room=5, first_page=0, num_pages=2).stage, -1)


class TestStageChannel(CustomTestCase):
    def setUp(self):
        self.mgr = with_offload(_bare_decode_manager())
        self.channel = self.mgr.dram_offload_channel
        # This rank hosts stages 1 and 3 of a four-stage prefill.
        self.pools = _Pools((1, 3))
        self.channel.bind_stage_pools(self.pools)

    def _receive(self, message):
        self.mgr._message_handlers[type(message).TAG](message.to_frames())

    def test_a_branch_is_collected_from_its_header_and_hosted_parts(self):
        self._receive(_push(1))
        (entry,) = self.channel.received_stage_pushes().values()
        self.assertIsNone(entry.header)
        self._receive(_push(0, tokens=[1, 2, 3, 4, 5, 6, 7, 8], uuid=0))
        self._receive(_push(3))
        self.assertEqual(self.channel.received_pushes(), {})
        entry = self.channel.take_stage_push(("ab12", 3))
        self.assertEqual(entry.header.tokens, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(sorted(entry.parts), [1, 3])
        part = entry.parts[3]
        self.assertEqual((part.blocks, part.address), ([43, 53], "10.0.0.1:9003"))
        self.assertEqual(part.layout.tensors, (("bfloat16", (4, 16, 8)),))
        self.assertEqual((part.layout.stage, part.layout.page_bytes), (3, 1024))
        self.assertNotEqual(
            part.layout.fingerprint(), entry.parts[1].layout.fingerprint()
        )
        self.assertIsNone(self.channel.take_stage_push(("ab12", 3)))

    def test_a_stage_read_lands_in_that_stages_pool(self):
        read_id = stage_transfer_id("ab12", 3, 3)
        started = self.channel.start_stage_read(
            read_id, 9, "10.0.0.1:9003", [43, 53], [0, 1], stage=3
        )
        self.assertTrue(started)
        self.assertEqual(
            self.pools.transfer(3).reads[read_id]["local_block_ids"], [0, 1]
        )
        self.assertEqual(self.pools.transfer(1).reads, {})
        self.assertIsNone(self.channel.poll_host_reads([read_id]))
        self.pools.transfer(3).complete("received")
        self.channel.drain_host_reads()
        self.assertTrue(self.channel.poll_host_reads([read_id]))
        # A read that cannot start is reported, not raised.
        self.pools.transfer(1, swa=True).read_error = RuntimeError("no endpoint")
        self.assertFalse(
            self.channel.start_stage_read(
                "victim:ab12:3:s1:swa", 9, "x", [1], [0], stage=1, swa=True
            )
        )

    def _pull(self, stage, *, first_page=0, num_pages=2):
        self._receive(
            PrefixPull(
                room=5,
                first_page=first_page,
                num_pages=num_pages,
                stage=stage,
                reply_endpoint=f"tcp://10.0.0.1:{7000 + stage}",
            )
        )
        endpoint, frames = sent_frames(self.mgr)[-1]
        return endpoint, PrefixReady.from_frames(frames)

    def test_a_stage_pulls_its_share_of_an_offered_prefix(self):
        self.channel.offer_stage_prefix(5, [0, 1, 2], [6, 7])
        endpoint, ready = self._pull(3, first_page=1)
        self.assertEqual(endpoint, "tcp://10.0.0.1:7003")
        self.assertNotEqual(ready.uuid, 0)
        self.assertNotEqual(ready.swa_uuid, 0)
        full = self.pools.transfer(3).registrations
        self.assertEqual(list(full.values()), [(ready.uuid, [1, 2])])
        window = self.pools.transfer(3, swa=True).registrations
        self.assertEqual(list(window.values()), [(ready.swa_uuid, [6, 7])])
        # Stage 1 reads the same page ids from its own pool.
        _, ready = self._pull(1, first_page=1)
        self.assertEqual(
            list(self.pools.transfer(1).registrations.values()), [(ready.uuid, [1, 2])]
        )
        self.assertTrue(self.channel.host_prefix_pending(5))
        for transfer in self.pools.transfers.values():
            transfer.complete("sent")
        self.channel.drain_host_reads()
        self.assertFalse(self.channel.host_prefix_pending(5))

    def test_a_pull_that_cannot_be_served_is_refused(self):
        # Nothing offered.
        self.assertEqual(self._pull(3)[1].uuid, 0)
        self.channel.offer_stage_prefix(5, [0, 1])
        # A stage this rank does not host, and a range past the offer.
        self.assertEqual(self._pull(2)[1].uuid, 0)
        self.assertEqual(self._pull(3, first_page=1)[1].uuid, 0)
        # The queue let go of the pages: a late pull registers nothing.
        self.channel.release_prefix_offer(5)
        self.assertEqual(self._pull(3)[1].uuid, 0)
        self.assertEqual(self.pools.transfer(3).registrations, {})
        # A native refusal.
        self.channel.offer_stage_prefix(6, [0, 1])
        self.pools.transfer(3).registration_error = RuntimeError("full")
        self._receive(
            PrefixPull(
                room=6, first_page=0, num_pages=2, stage=3, reply_endpoint="tcp://x:1"
            )
        )
        self.assertEqual(PrefixReady.from_frames(sent_frames(self.mgr)[-1][1]).uuid, 0)

    def test_a_tensor_parallel_pull_is_not_served_from_stage_pools(self):
        self.channel.offer_stage_prefix(5, [0, 1])
        before = len(sent_frames(self.mgr))
        self._receive(PrefixPull(room=5, first_page=0, num_pages=2))
        # No tensor-parallel offer for the room: the pull is dropped as before.
        self.assertEqual(len(sent_frames(self.mgr)), before)
        self.assertEqual(self.pools.transfer(3).registrations, {})


class TestStagePoolBuilder(CustomTestCase):
    def test_a_stage_pool_is_built_without_dynamic_host_staging(self):
        """The transfer manager turns dynamic host staging on for the engines
        it builds later in the process; a manager built that way refuses host
        reads (seen live: every landing failed to start)."""
        import torch

        from sglang_tpu.disaggregation.decode_dram_offload import stage_pool_builder
        from sglang_tpu.disaggregation.decode_dram_offload.stage_layout_store import (
            StagePoolLayout,
        )

        seen = {}

        def transfer(pool, num_host_pages):
            seen["env"] = os.environ.get("TPU_RAIDEN_DYNAMIC_HOST_STAGING")
            seen["buffers"] = pool.page_buffers()
            seen["pages"] = num_host_pages
            return "pool"

        layout = StagePoolLayout(
            3, 8, (("bfloat16", (4, 2, 8)), ("uint8", (4, 2))), 4 * 2 * 8 * 2 + 8
        )
        with (
            patch.dict(os.environ, {"TPU_RAIDEN_DYNAMIC_HOST_STAGING": "1"}),
            patch("sglang_tpu.raiden_transfer.RaidenKVTransfer", transfer),
        ):
            built = stage_pool_builder.build_stage_pool(layout, 64, device="cpu")
            self.assertEqual(os.environ["TPU_RAIDEN_DYNAMIC_HOST_STAGING"], "1")
        self.assertEqual((built, seen["env"], seen["pages"]), ("pool", "0", 64))
        self.assertEqual(
            [(str(t.dtype), tuple(t.shape)) for t in seen["buffers"]],
            [("torch.bfloat16", (1, 4, 2, 8)), ("torch.uint8", (1, 4, 2))],
        )
        self.assertEqual(torch.count_nonzero(seen["buffers"][0]).item(), 0)
        # A layout that does not add up to its page bytes is refused.
        wrong = StagePoolLayout(3, 8, (("bfloat16", (4, 2, 8)),), 1)
        with self.assertRaisesRegex(ValueError, "stage 3 pages are 1 bytes"):
            stage_pool_builder.build_stage_pool(wrong, 64, device="cpu")


if __name__ == "__main__":
    unittest.main()
