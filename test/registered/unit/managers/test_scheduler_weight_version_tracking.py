import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    PluginWeightUpdateReqInput,
    PluginWeightUpdateReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ServingStub:
    def __init__(self, weight_version: str):
        self.weight_version = weight_version


class _ContextStub:
    def __init__(self, serving: _ServingStub):
        self.serving = serving

    def override(self, source, **fields):
        self.serving.weight_version = fields["weight_version"]


class TestSchedulerRecordWeightVersionChange(CustomTestCase):
    def _serving(self, version: str) -> _ServingStub:
        serving = _ServingStub(version)
        for name, value in (
            ("get_serving", serving),
            ("get_context", _ContextStub(serving)),
        ):
            patcher = patch(f"sglang.srt.managers.scheduler.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return serving

    def _scheduler(
        self, *, inflight=(), waiting=(), chunked=None, staging=()
    ) -> SimpleNamespace:
        return SimpleNamespace(
            collect_inflight_reqs=lambda: set(inflight),
            waiting_queue=list(waiting),
            chunked_req=chunked,
            hisparse_coordinator=(
                SimpleNamespace(
                    ack_staging_queue=[SimpleNamespace(req=req) for req in staging]
                )
                if staging
                else None
            ),
        )

    def test_a_new_version_is_adopted(self):
        """The scheduler has to end up on the version it was told about, or nothing downstream can read it."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version="v2")

        self.assertEqual(serving.weight_version, "v2")

    def test_same_version_is_a_noop(self):
        """Re-announcing the current version must not be treated as a change."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version="v1")

        self.assertEqual(serving.weight_version, "v1")

    def test_none_version_is_a_noop(self):
        """An update that carries no version must leave the recorded one alone."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version=None)

        self.assertEqual(serving.weight_version, "v1")

    def test_every_source_of_live_requests_is_stamped(self):
        """A request missed here keeps attributing its next tokens to the superseded version."""
        self._serving("v1")
        inflight, queued, chunked, staged = (object() for _ in range(4))
        scheduler = self._scheduler(
            inflight=[inflight], waiting=[queued], chunked=chunked, staging=[staged]
        )

        with patch(
            "sglang.srt.managers.scheduler.record_weight_version_events",
            return_value=0,
        ) as recorder:
            Scheduler.record_weight_version_change(scheduler, new_version="v2")

        self.assertEqual(
            set(recorder.call_args.args[0]), {inflight, queued, chunked, staged}
        )


def _runner(result=(True, "ok")):
    runner = Mock()
    for method in (
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
        "load_weights_from_distributed",
    ):
        getattr(runner.weight_updater, method).return_value = result
    runner.weight_updater.load_rejection.return_value = None
    return runner


def _request(**fields):
    return SimpleNamespace(
        **{
            "weight_version": "v2",
            "flush_cache": True,
            "torch_empty_cache": False,
            "model_path": "m",
            "recapture_cuda_graph": False,
            "load_format": None,
            "selector": "all",
            "names": ["model.layers.0.weight"],
            "dtypes": ["float32"],
            "shapes": [[1]],
            "group_name": "g",
            "serialized_named_tensors": [b""],
            **fields,
        }
    )


class _WeightUpdaterManagerTestBase(CustomTestCase):
    def setUp(self):
        patcher = patch("torch.distributed.barrier")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recorded = []

    def _build_manager(self, target, draft):
        return SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=target,
                weight_update_runners=lambda: [("target", target)],
                deserialize_own_rank=lambda payloads: [],
            ),
            draft_worker=(
                None
                if draft is None
                else SimpleNamespace(weight_update_runners=lambda: [("draft", draft)])
            ),
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                )
            ),
        )

    def _manager(self, target, draft=None, *, session=True):
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = self._build_manager(target, draft)
        if session:
            # update_weights_from_* require an open session
            manager._session = _WeightUpdateSession(selector="all")
        return manager


class TestRecordWeightVersionAfterUpdate(_WeightUpdaterManagerTestBase):
    def test_successful_update_records_the_version(self):
        """A refit that reports success advances the scheduler-side version."""
        output = self._manager(_runner()).update_weights_from_disk(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_update_does_not_record_the_version(self):
        """A refit that fails must leave the version alone, or later tokens are mislabelled."""
        output = self._manager(_runner((False, "boom"))).update_weights_from_disk(
            _request()
        )

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_draft_failure_does_not_record_the_version(self):
        """The target succeeding is not enough: a failed draft refit leaves the engine mixed."""
        manager = self._manager(_runner(), draft=_runner((False, "draft boom")))

        output = manager.update_weights_from_disk(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_distributed_update_records_the_version(self):
        """The distributed refit is the path an RL trainer actually drives, so it must record too."""
        output = self._manager(_runner()).update_weights_from_distributed(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_distributed_update_does_not_record_the_version(self):
        """A failed distributed refit leaves the version alone, exactly like the disk path."""
        output = self._manager(
            _runner((False, "boom"))
        ).update_weights_from_distributed(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_tensor_update_records_the_version(self):
        """The tensor refit records the version once the load reports success."""
        output = self._manager(_runner()).update_weights_from_tensor(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_successful_ipc_update_records_the_version(self):
        """The checkpoint-engine IPC refit records the version like every other path."""
        output = self._manager(_runner()).update_weights_from_ipc(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_ipc_update_does_not_record_the_version(self):
        """The IPC path branches on success separately from the cache flush, so failure must record nothing."""
        output = self._manager(_runner((False, "boom"))).update_weights_from_ipc(
            _request()
        )

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])


class TestWeightUpdateSession(_WeightUpdaterManagerTestBase):
    def test_distributed_update_receives_once_on_target_loads_into_each(self):
        """A draft runner that received its own broadcast would deadlock the update group."""
        target, draft = _runner(), _runner()
        weights = target.weight_updater.receive_weights_from_distributed.return_value
        req = _request()

        output = self._manager(target, draft).update_weights_from_distributed(req)

        self.assertTrue(output.success)
        target.weight_updater.receive_weights_from_distributed.assert_called_once_with(
            names=req.names,
            dtypes=req.dtypes,
            shapes=req.shapes,
            group_name=req.group_name,
            load_format=req.load_format,
        )
        target.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.receive_weights_from_distributed.assert_not_called()

    def test_distributed_update_target_only_selector_skips_draft(self):
        """selector="target" must not load into the draft."""
        target, draft = _runner(), _runner()

        output = self._manager(target, draft).update_weights_from_distributed(
            _request(selector="target")
        )

        self.assertTrue(output.success)
        target.weight_updater.load_weights_from_distributed.assert_called_once()
        draft.weight_updater.load_weights_from_distributed.assert_not_called()

    def test_end_runs_post_load_on_both_when_load_was_bypassed(self):
        """A P2P/RDMA session never calls load_weights, so end must run post_load_weights."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)

        output = manager.end_weight_update(EndWeightUpdateReqInput())

        self.assertTrue(output.success)
        for runner in (target, draft):
            runner.weight_updater.end_weight_update.assert_called_once_with(
                run_post_load=True
            )
        self.assertIsNone(manager._session)

    def test_end_skips_post_load_on_both_when_weights_loaded(self):
        """load_weights already ran post_load_weights; running it twice would double-apply."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)
        manager._session = _WeightUpdateSession(selector="all", loaded_weights=True)

        manager.end_weight_update(EndWeightUpdateReqInput())

        for runner in (target, draft):
            runner.weight_updater.end_weight_update.assert_called_once_with(
                run_post_load=False
            )

    def test_session_selector_confines_begin_and_end_to_selected_runners(self):
        """end finalizing a runner begin never restored would repack unrestored weights."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft, session=False)

        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
        manager.end_weight_update(EndWeightUpdateReqInput())

        target.weight_updater.begin_weight_update.assert_not_called()
        target.weight_updater.end_weight_update.assert_not_called()
        draft.weight_updater.begin_weight_update.assert_called_once_with()
        draft.weight_updater.end_weight_update.assert_called_once()

    def test_begin_rejects_reentry(self):
        """A second begin would leave the first session's runners unfinalized."""
        target = _runner()

        output = self._manager(target).begin_weight_update(BeginWeightUpdateReqInput())

        self.assertFalse(output.success)
        self.assertIn("already open", output.message)
        target.weight_updater.begin_weight_update.assert_not_called()

    def test_end_without_session_is_rejected(self):
        """Finalizing runners begin never restored would repack weights twice."""
        target = _runner()

        output = self._manager(target, session=False).end_weight_update(
            EndWeightUpdateReqInput()
        )

        self.assertFalse(output.success)
        self.assertIn("begin_weight_update", output.message)
        target.weight_updater.end_weight_update.assert_not_called()

    def test_update_without_session_is_rejected_without_loading(self):
        """A caller that skips begin gets an error back instead of crashing the scheduler."""
        target = _runner()

        output = self._manager(target, session=False).update_weights_from_distributed(
            _request()
        )

        self.assertFalse(output.success)
        self.assertIn("begin_weight_update", output.message)
        target.weight_updater.receive_weights_from_distributed.assert_not_called()


class TestPluginWeightUpdate(_WeightUpdaterManagerTestBase):
    def setUp(self):
        super().setUp()
        gather = patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda outputs, obj, group: outputs.__setitem__(0, obj),
        )
        gather.start()
        self.addCleanup(gather.stop)
        override = get_parallel().override(tp_rank=0, tp_size=1)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        self.flushes = []

    def _manager_serving(self, handlers, *, target=None, draft=None, session=True):
        manager = self._manager(target or _runner(), draft, session=session)
        manager.flush_cache = lambda **kwargs: self.flushes.append(kwargs) or True
        platform = patch(
            "sglang.srt.managers.scheduler_components.weight_updater.current_platform",
            SimpleNamespace(get_weight_update_handlers=lambda: handlers),
        )
        platform.start()
        self.addCleanup(platform.stop)
        return manager

    def test_received_tensors_load_into_every_selected_runner(self):
        """Like update_weights_from_distributed: the transport receives, the core loads target and draft."""
        weights = [("w", object())]
        target, draft = _runner(), _runner()
        manager = self._manager_serving(
            {"ici": lambda *, context, payload: weights}, target=target, draft=draft
        )

        output = manager.plugin_weight_update(
            PluginWeightUpdateReqInput(handler="ici", weight_version="3")
        )

        self.assertTrue(output.success)
        self.assertEqual(output.received_tensor_counts, [1])
        target.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        self.assertEqual(self.recorded, ["3"])
        self.assertEqual(len(self.flushes), 1)

    def test_end_runs_post_load_only_after_writes_that_bypassed_load_weights(self):
        """A transport that writes parameters itself returns no tensors, like p2p peer writes."""
        for received, run_post_load in (([("w", object())], False), ([], True)):
            with self.subTest(received=received):
                target = _runner()
                manager = self._manager_serving(
                    {"ici": lambda *, context, payload: received}, target=target
                )
                manager.plugin_weight_update(PluginWeightUpdateReqInput(handler="ici"))
                manager.end_weight_update(EndWeightUpdateReqInput())

                target.weight_updater.end_weight_update.assert_called_once_with(
                    run_post_load=run_post_load
                )

    def test_a_failed_receive_loads_nothing_on_its_rank_and_keeps_the_cache_and_version(
        self,
    ):
        """A rank that failed to receive loads nothing; the request fails and the old KV and version stay."""

        def broken(*, context, payload):
            raise ValueError("no receive buffers")

        target = _runner()
        output = self._manager_serving(
            {"ici": broken}, target=target
        ).plugin_weight_update(
            PluginWeightUpdateReqInput(handler="ici", weight_version="3")
        )

        self.assertFalse(output.success)
        self.assertIn("no receive buffers", output.message)
        target.weight_updater.load_weights_from_distributed.assert_not_called()
        self.assertEqual((self.flushes, self.recorded), ([], []))

    def test_rank_0_reports_another_ranks_failure_from_the_gather(self):
        """Only rank 0 replies: a failure gathered from rank 1 fails the request after rank 0 loaded."""
        gathered = [(1, None), (0, "TP rank 1: no receive buffers")]
        target = _runner()
        manager = self._manager_serving(
            {"ici": lambda *, context, payload: [("w", object())]}, target=target
        )

        with (
            patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda outputs, obj, group: outputs.__setitem__(
                    slice(None), gathered
                ),
            ),
            get_parallel().override(tp_rank=0, tp_size=2),
        ):
            output = manager.plugin_weight_update(
                PluginWeightUpdateReqInput(handler="ici", weight_version="3")
            )

        self.assertFalse(output.success)
        self.assertIn("TP rank 1", output.message)
        self.assertIn("other TP ranks have loaded", output.message)
        self.assertEqual(output.received_tensor_counts, [])
        target.weight_updater.load_weights_from_distributed.assert_called_once()
        self.assertEqual((self.flushes, self.recorded), ([], []))

    def test_a_runner_that_rejects_loads_fails_the_request_before_the_transport_runs(
        self,
    ):
        """The weight-cache and derived-weight guards cover direct writes too, so they run first."""
        called = []
        target = _runner()
        target.weight_updater.load_rejection.return_value = (
            "[weight_cache] plugin_weight_update is not supported"
        )

        output = self._manager_serving(
            {"ici": lambda *, context, payload: called.append(1) or []}, target=target
        ).plugin_weight_update(PluginWeightUpdateReqInput(handler="ici"))

        self.assertFalse(output.success)
        self.assertIn("weight_cache", output.message)
        self.assertEqual(called, [])

    def test_transport_state_persists_across_calls(self):
        """A transport that registers buffers in one call must find them in the next."""

        def receive(*, context, payload):
            context.state.setdefault("calls", 0)
            context.state["calls"] += 1
            return []

        manager = self._manager_serving({"ici": receive})
        for _ in range(2):
            manager.plugin_weight_update(PluginWeightUpdateReqInput(handler="ici"))

        self.assertEqual(manager._plugin_state["calls"], 2)

    def test_requests_outside_a_session_or_for_unknown_handlers_are_rejected(self):
        """A caller that skips begin, or names a missing transport, gets an error, not a crash."""
        called = []
        handlers = {"ici": lambda *, context, payload: called.append(1) or []}

        outside = self._manager_serving(handlers, session=False).plugin_weight_update(
            PluginWeightUpdateReqInput(handler="ici")
        )
        unknown = self._manager_serving(handlers).plugin_weight_update(
            PluginWeightUpdateReqInput(handler="nccl")
        )

        self.assertFalse(outside.success)
        self.assertIn("begin_weight_update", outside.message)
        self.assertFalse(unknown.success)
        self.assertIn("nccl", unknown.message)
        self.assertEqual(called, [])


class TestPluginWeightUpdateReply(CustomTestCase):
    def test_attention_dp_leaders_replies_merge_like_the_distributed_route(self):
        """Under attention DP the tokenizer gets one reply per leader; one failure fails the request."""
        ok = PluginWeightUpdateReqOutput(
            success=True, message="Success", received_tensor_counts=[3, 3]
        )
        failed = PluginWeightUpdateReqOutput(
            success=False, message="TP rank 1: no receive buffers"
        )

        merged_ok = PluginWeightUpdateReqOutput.merged([ok, ok])
        merged_failed = PluginWeightUpdateReqOutput.merged([ok, failed])

        self.assertEqual(
            (merged_ok.success, merged_ok.received_tensor_counts), (True, [3, 3])
        )
        self.assertFalse(merged_failed.success)
        self.assertIn("no receive buffers", merged_failed.message)
        self.assertEqual(merged_failed.received_tensor_counts, [])


if __name__ == "__main__":
    unittest.main()
