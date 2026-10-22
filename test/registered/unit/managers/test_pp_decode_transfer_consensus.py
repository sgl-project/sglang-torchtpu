"""Pipeline decode transfer outcomes, failure propagation, and page ownership."""

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base.conn import KVPoll  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.utils.common import point_to_point_pyobj  # noqa: E402
from sglang_tpu.pp_scheduler import (  # noqa: E402
    PPDecodeTransferStatus,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Receiver:
    group_failure_requires_drain = True

    def __init__(self, failed=False, active=False):
        self.failed = failed
        self.active = active
        self.quiesces = 0
        self.aborts = 0
        self.metadata_ready = True

    def poll(self):
        if self.active:
            return KVPoll.Transferring
        return KVPoll.Failed if self.failed else KVPoll.Success

    def quiesce_group_failure(self):
        self.quiesces += 1
        self.failed = True

    def abort(self):
        self.aborts += 1
        self.failed = True


class TransferQueue:
    """Record terminal releases without allocating model or KV tensors."""

    def __init__(self, receivers):
        self.queue = [
            SimpleNamespace(req=SimpleNamespace(rid=rid), kv_receiver=receiver)
            for rid, receiver in receivers.items()
        ]
        self.failed_rids = []
        self.resolve_deferred_releases = Mock()
        self.enable_staging = False
        self._poll_with_staging = Mock(side_effect=self._poll_with_metadata_gate)

    def _poll_with_metadata_gate(self):
        return [
            (
                KVPoll.Transferring
                if item.kv_receiver.poll() == KVPoll.Success
                and not item.kv_receiver.metadata_ready
                else item.kv_receiver.poll()
            )
            for item in self.queue
        ]

    def pop_transferred(self, rids):
        released, remaining = [], []
        for item, status in zip(self.queue, self._poll_with_metadata_gate()):
            if item.req.rid not in rids or status == KVPoll.Transferring:
                remaining.append(item)
            elif status == KVPoll.Failed:
                self.failed_rids.append(item.req.rid)
            else:
                released.append(item.req)
        self.queue = remaining
        return released


def scheduler(rank, receivers, size=3):
    result = SchedulerPPMixin()
    # The per-stage poll and the failure union are the TPU wave's path; its
    # consensus travels on the pipeline group's tagged channel.
    from sglang_tpu.pp_scheduler import TpuPipelineTransport

    result._pp_wave = TpuPipelineTransport(None, 1)
    result.pp_group = SimpleNamespace(
        rank_in_group=rank,
        world_size=size,
        is_first_rank=rank == 0,
        is_last_rank=rank == size - 1,
        send_object=Mock(return_value=[]),
    )
    result.attn_cp_cpu_group = object()
    result.attn_tp_cpu_group = object()
    result.disagg_decode_transfer_queue = TransferQueue(receivers)
    result.waiting_queue = []
    result.enable_hisparse = False
    result._pp_recv_pyobj_from_prev_stage = Mock()
    result._pp_send_pyobj_to_next_stage = Mock(return_value=[])
    return result


def consensus(stages):
    payload = None
    for stage in stages:
        stage._pp_recv_pyobj_from_prev_stage.return_value = payload
        payload = stage._pp_pd_get_decode_transferred_ids()
    return payload


class TestDecodeTransferConsensus(unittest.TestCase):
    def release(self, stages, payload):
        forwarded = payload
        for stage in stages:
            forwarded = stage.process_decode_transfer_queue(forwarded)
            self.assertIs(forwarded, payload)
            if not stage.pp_group.is_last_rank:
                _, sent = stage._pp_pd_send_consensus_release_ids(
                    [payload], 0, forwarded, payload
                )
                self.assertIs(sent, payload)
                self.assertIs(stage.pp_group.send_object.call_args.args[0], payload)

    def test_failure_at_any_stage_aborts_every_copy_and_preserves_other_requests(self):
        for failing_rank in range(3):
            with self.subTest(failing_rank=failing_rank):
                stages = [
                    scheduler(
                        rank,
                        {
                            "bad": Receiver(failed=rank == failing_rank),
                            "good": Receiver(),
                            "pending": Receiver(active=rank == 1),
                        },
                    )
                    for rank in range(3)
                ]
                payload = consensus(stages)
                self.assertEqual(set(payload.terminal_rids), {"bad", "good"})
                self.assertEqual(payload.failed_rids, ["bad"])
                self.release(stages, payload)
                for stage in stages:
                    queue = stage.disagg_decode_transfer_queue
                    self.assertEqual(queue.failed_rids, ["bad"])
                    self.assertEqual([req.rid for req in stage.waiting_queue], ["good"])
                    self.assertEqual(
                        [item.req.rid for item in queue.queue], ["pending"]
                    )

    def test_failure_quiesces_pending_receivers_and_waits_for_every_stage_to_drain(
        self,
    ):
        receivers = [Receiver(failed=True), Receiver(active=True), Receiver()]
        stages = [
            scheduler(rank, {"bad": receiver})
            for rank, receiver in enumerate(receivers)
        ]
        payload = consensus(stages)
        self.assertEqual(payload, PPDecodeTransferStatus([], ["bad"]))
        self.release(stages, payload)
        self.assertTrue(receivers[1].active)
        for receiver, stage in zip(receivers, stages):
            self.assertEqual(receiver.quiesces, 1)
            self.assertTrue(receiver.failed)
            self.assertEqual(stage.waiting_queue, [])
            self.assertEqual(len(stage.disagg_decode_transfer_queue.queue), 1)
            self.assertEqual(stage.disagg_decode_transfer_queue.failed_rids, [])

        receivers[1].active = False
        payload = consensus(stages)
        self.assertEqual(payload, PPDecodeTransferStatus(["bad"], ["bad"]))
        self.release(stages, payload)
        for stage in stages:
            self.assertEqual(stage.disagg_decode_transfer_queue.failed_rids, ["bad"])
            self.assertEqual(stage.disagg_decode_transfer_queue.queue, [])
            self.assertEqual(stage.waiting_queue, [])

    def test_failure_union_survives_a_stage_that_has_not_registered_the_request(self):
        stages = [
            scheduler(0, {"bad": Receiver(failed=True)}),
            scheduler(1, {}),
            scheduler(2, {"bad": Receiver(active=True)}),
        ]
        payload = consensus(stages)
        self.assertEqual(payload, PPDecodeTransferStatus([], ["bad"]))
        self.release(stages, payload)
        receiver = stages[2].disagg_decode_transfer_queue.queue[0].kv_receiver
        self.assertTrue(receiver.failed)
        self.assertTrue(receiver.active)

    def test_backend_without_a_drain_hook_receives_abort(self):
        receiver = Receiver()
        receiver.group_failure_requires_drain = False
        stage = scheduler(0, {"bad": receiver})
        payload = PPDecodeTransferStatus(["bad"], ["bad"])
        self.assertIs(stage.process_decode_transfer_queue(payload), payload)
        self.assertEqual(receiver.aborts, 1)
        self.assertEqual(receiver.quiesces, 0)
        self.assertEqual(stage.disagg_decode_transfer_queue.failed_rids, ["bad"])
        self.assertEqual(stage.waiting_queue, [])

    def test_successful_transfers_are_admitted_on_every_stage(self):
        stages = [scheduler(rank, {"ok": Receiver()}) for rank in range(3)]
        payload = consensus(stages)
        self.assertEqual(payload, PPDecodeTransferStatus(["ok"], []))
        self.release(stages, payload)
        for stage in stages:
            self.assertEqual([req.rid for req in stage.waiting_queue], ["ok"])
            self.assertEqual(stage.disagg_decode_transfer_queue.queue, [])

    def test_metadata_readiness_is_required_on_every_stage_before_release(self):
        stages = [scheduler(rank, {"ok": Receiver()}) for rank in range(3)]
        receiver = stages[1].disagg_decode_transfer_queue.queue[0].kv_receiver
        receiver.metadata_ready = False
        payload = consensus(stages)
        self.assertEqual(payload, PPDecodeTransferStatus([], []))
        self.release(stages, payload)
        for stage in stages:
            self.assertEqual(stage.waiting_queue, [])
            self.assertEqual(len(stage.disagg_decode_transfer_queue.queue), 1)
        receiver.metadata_ready = True
        self.release(stages, consensus(stages))
        for stage in stages:
            self.assertEqual([req.rid for req in stage.waiting_queue], ["ok"])

    def test_staging_readiness_participates_in_the_consensus(self):
        stage = scheduler(0, {"ok": Receiver()})
        queue = stage.disagg_decode_transfer_queue
        queue.enable_staging = True
        queue._poll_with_staging.side_effect = None
        queue._poll_with_staging.return_value = [KVPoll.Transferring]
        self.assertEqual(consensus([stage]), PPDecodeTransferStatus([], []))
        queue._poll_with_staging.assert_called_once()

    def test_deferred_releases_progress_without_a_consensus_packet(self):
        stage = scheduler(0, {})
        self.assertIsNone(stage.process_decode_transfer_queue(None))
        stage.disagg_decode_transfer_queue.resolve_deferred_releases.assert_called_once()

    def test_empty_consensus_retains_its_type_over_the_pipeline_transport(self):
        payload = PPDecodeTransferStatus([], [])
        sent = []
        with patch(
            "torch.distributed.send",
            side_effect=lambda tensor, *args, **kwargs: sent.append(tensor.clone()),
        ):
            point_to_point_pyobj(payload, rank=0, src=0, dst=1)
        self.assertEqual(len(sent), 2)
        decoded = pickle.loads(bytes(sent[1].numpy()))
        self.assertIsInstance(decoded, PPDecodeTransferStatus)
        self.assertEqual(decoded, payload)


if __name__ == "__main__":
    unittest.main()
