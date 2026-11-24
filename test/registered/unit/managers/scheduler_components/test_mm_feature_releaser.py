import mmap
import os
import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import (
    batch_result_processor as processor_module,
)
from sglang.srt.managers.scheduler_components import (
    mm_feature_releaser,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.mm_feature_releaser import (
    MMFeatureReleaser,
)
from sglang.srt.managers.scheduler_components.request_list import (
    KeyedReqList,
    ListKind,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Stop(Exception):
    pass


def _shm_feature(nbytes: int = 1 << 16) -> torch.Tensor:
    """A feature like ShmPointerMMData.materialize: a private mapping of an
    unlinked file."""
    shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    with tempfile.TemporaryFile(dir=shm_dir) as f:
        f.truncate(nbytes)
        owned = mmap.mmap(f.fileno(), nbytes, access=mmap.ACCESS_COPY)
    return torch.frombuffer(owned, dtype=torch.uint8).reshape(-1, 256)


def _mm_inputs(num_items: int):
    features = [_shm_feature() for _ in range(num_items)]
    refs = [weakref.ref(feature) for feature in features]
    items = [MultimodalDataItem(modality=Modality.IMAGE, feature=f) for f in features]
    return MultimodalInputs(mm_items=items), refs


def _event(ready: bool):
    return SimpleNamespace(query=lambda: ready)


def _alive(refs) -> int:
    return sum(ref() is not None for ref in refs)


class TestMMFeatureReleaser(CustomTestCase):
    def test_off_releases_features_immediately(self):
        for mode in ("", "0"):
            with self.subTest(mode=mode):
                releaser = MMFeatureReleaser(mode)
                mm_inputs, refs = _mm_inputs(3)

                releaser.release(mm_inputs)

                self.assertFalse(releaser.enabled)
                self.assertTrue(all(i.feature is None for i in mm_inputs.mm_items))
                self.assertEqual(_alive(refs), 0)
                self.assertEqual(len(releaser._pending), 0)

    def test_unsupported_mode_raises(self):
        for mode in ("1", "thread"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                MMFeatureReleaser(mode)

    @patch.object(mm_feature_releaser, "_DRAIN_BUDGET_NS", 10**9)
    def test_incremental_parks_until_a_pending_device_wait(self):
        releaser = MMFeatureReleaser("incremental")
        mm_inputs, refs = _mm_inputs(4)

        releaser.release(mm_inputs)
        self.assertTrue(all(i.feature is None for i in mm_inputs.mm_items))
        self.assertEqual(_alive(refs), 4)

        # A ready result leaves no device time to hide the unmaps behind.
        releaser.drain_before_wait(_event(ready=True))
        self.assertEqual(_alive(refs), 4)

        releaser.drain_before_wait(_event(ready=False))
        self.assertEqual(_alive(refs), 0)
        self.assertEqual(len(releaser._pending), 0)

    @patch.object(mm_feature_releaser, "_DRAIN_BUDGET_NS", 0)
    def test_idle_drain_frees_what_the_budget_left(self):
        releaser = MMFeatureReleaser("incremental")
        mm_inputs, refs = _mm_inputs(2)
        releaser.release(mm_inputs)

        releaser.drain_before_wait(_event(ready=False))
        self.assertEqual(_alive(refs), 2)

        releaser.drain_all()
        self.assertEqual(_alive(refs), 0)

    @patch.object(mm_feature_releaser, "_MAX_PENDING", 2)
    def test_backlog_is_bounded(self):
        releaser = MMFeatureReleaser("incremental")
        mm_inputs, refs = _mm_inputs(3)

        releaser.release(mm_inputs)

        self.assertEqual(len(releaser._pending), 2)
        self.assertEqual([ref() is None for ref in refs], [True, False, False])

    def test_transport_release_failure_still_clears_feature(self):
        releaser = MMFeatureReleaser("incremental")
        mm_inputs, refs = _mm_inputs(1)

        with patch.object(
            MultimodalDataItem,
            "release_transport_proxies",
            side_effect=RuntimeError("release failed"),
        ):
            releaser.release(mm_inputs)

        self.assertIsNone(mm_inputs.mm_items[0].feature)
        self.assertEqual(len(releaser._pending), 1)
        releaser.drain_all()
        self.assertEqual(_alive(refs), 0)

    def test_release_goes_through_release_features(self):
        original = MultimodalInputs.release_features
        for mode, alive_after in (("", 0), ("incremental", 2)):
            with self.subTest(mode=mode):
                releaser = MMFeatureReleaser(mode)
                mm_inputs, refs = _mm_inputs(2)

                with patch.object(
                    MultimodalInputs,
                    "release_features",
                    autospec=True,
                    side_effect=original,
                ) as release_features:
                    releaser.release(mm_inputs)

                release_features.assert_called_once_with(mm_inputs)
                self.assertTrue(all(i.feature is None for i in mm_inputs.mm_items))
                self.assertEqual(_alive(refs), alive_after)
                releaser.drain_all()
                self.assertEqual(_alive(refs), 0)

    def test_second_release_of_the_same_inputs_parks_nothing(self):
        releaser = MMFeatureReleaser("incremental")
        mm_inputs, _ = _mm_inputs(2)

        releaser.release(mm_inputs)
        releaser.release(mm_inputs)

        self.assertEqual(len(releaser._pending), 2)


class TestSchedulerHooks(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.releaser = MMFeatureReleaser("incremental")
        for module in (scheduler_module, processor_module):
            patcher = patch.object(module, "MM_FEATURE_RELEASER", self.releaser)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_finished_non_session_requests_are_parked(self):
        def req(finished, session):
            mm_inputs, _ = _mm_inputs(1)
            return SimpleNamespace(
                finished=lambda: finished,
                finished_reason=None,
                multimodal_inputs=mm_inputs,
                session=session,
            )

        done, in_session, running = (
            req(True, None),
            req(True, object()),
            req(False, None),
        )
        session_inputs = in_session.multimodal_inputs
        running_inputs = running.multimodal_inputs

        Scheduler._maybe_clear_mm_inputs(
            Scheduler.__new__(Scheduler),
            SimpleNamespace(reqs=[done, in_session, running]),
        )

        self.assertIsNone(done.multimodal_inputs)
        self.assertEqual(len(self.releaser._pending), 1)
        self.assertIsNotNone(session_inputs.mm_items[0].feature)
        self.assertIsNotNone(running_inputs.mm_items[0].feature)

    def test_finish_path_parks_finished_non_session_features(self):
        processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            model_config=SimpleNamespace(think_end_ids=None),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
            hisparse_coordinator=None,
            req_to_token_pool=None,
            kv_weight_version_tracker=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=MagicMock(),
            draft_worker=None,
            model_worker=MagicMock(),
            logprob_result_processor=None,
            output_streamer=MagicMock(),
            beam_coordinator=MagicMock(),
            request_retirer=MagicMock(),
            abort_request=lambda *args, **kwargs: None,
            still_in_batch=lambda req: True,
            prefill_launched_reqs=KeyedReqList(ListKind.PREFILL_LAUNCHED),
        )
        batch = SimpleNamespace(mamba_track_mask_cpu=None)

        def finished_req(session):
            mm_inputs, refs = _mm_inputs(1)
            req = SimpleNamespace(
                rid="req",
                finished=lambda: True,
                finished_reason=None,
                multimodal_inputs=mm_inputs,
                session=session,
                kv=SimpleNamespace(mamba_ping_pong_track_buffer=None),
                return_routed_experts=False,
                time_stats=MagicMock(),
            )
            return req, refs

        done, done_refs = finished_req(session=None)
        in_session, session_refs = finished_req(session=object())

        result = SimpleNamespace(num_correct_drafts_per_req_cpu=None)
        with get_context().override_server_args():
            for req in (done, in_session):
                processor._handle_finish_state_updated_req(req, batch, result, 0, None)

        self.assertEqual(processor.request_retirer.release.call_count, 2)
        self.assertIsNone(done.multimodal_inputs.mm_items[0].feature)
        self.assertEqual(len(self.releaser._pending), 1)
        self.assertEqual(_alive(done_refs), 1)
        self.assertIsNotNone(in_session.multimodal_inputs.mm_items[0].feature)
        self.assertEqual(_alive(session_refs), 1)

        self.releaser.drain_all()
        self.assertEqual(_alive(done_refs), 0)

    def test_idle_drops_parked_features(self):
        mm_inputs, refs = _mm_inputs(2)
        self.releaser.release(mm_inputs)
        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            dp_attn_adapter=MagicMock(),
            deferred_responses=MagicMock(),
            admit_deferred_health_probes=MagicMock(side_effect=_Stop),
        )

        with self.assertRaises(_Stop):
            Scheduler.on_idle(scheduler)

        self.assertEqual(_alive(refs), 0)

    @patch.object(mm_feature_releaser, "_DRAIN_BUDGET_NS", 10**9)
    def test_result_processing_drains_before_the_copy_wait(self):
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        # Frozen dataclass: set only the fields read before the copy wait.
        object.__setattr__(processor, "is_generation", True)
        object.__setattr__(processor, "token_to_kv_pool_allocator", MagicMock())
        object.__setattr__(processor, "still_in_batch", lambda req: True)
        methods = (
            processor.process_batch_result_prefill,
            processor.process_batch_result_decode,
        )
        for method in methods:
            for ready, parked_at_wait in ((False, 0), (True, 2)):
                with self.subTest(method=method.__name__, ready=ready):
                    self.releaser.drain_all()
                    mm_inputs, _ = _mm_inputs(2)
                    self.releaser.release(mm_inputs)
                    observed = []

                    def synchronize():
                        observed.append(len(self.releaser._pending))
                        raise _Stop

                    copy_done = SimpleNamespace(
                        query=lambda: ready, synchronize=synchronize
                    )
                    with self.assertRaises(_Stop):
                        method(
                            SimpleNamespace(reqs=[]),
                            SimpleNamespace(copy_done=copy_done),
                        )

                    self.assertEqual(observed, [parked_at_wait])


if __name__ == "__main__":
    unittest.main()
