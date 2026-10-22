# SPDX-License-Identifier: Apache-2.0
"""In-tree defaults of the scheduler extension a platform may replace."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler_components.backend_extension import (
    SchedulerExtension,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerExtensionDefaults(CustomTestCase):
    def test_platforms_default_to_the_in_tree_extension(self):
        self.assertIs(
            SRTPlatform.get_scheduler_extension_cls(object()), SchedulerExtension
        )
        self.assertIsNone(
            SRTPlatform.initialize_scheduler_worker(object(), None, tp_rank=0)
        )

    def test_defaults_keep_the_scheduler_behavior(self):
        ext = SchedulerExtension()
        algorithm = Mock()
        self.assertIs(
            ext.create_future_map(algorithm, "cpu", "pool", extra=1),
            algorithm.create_future_map.return_value,
        )
        algorithm.create_future_map.assert_called_once_with("cpu", "pool", extra=1)

        result, copy_stream, forward_stream = Mock(), Mock(), Mock()
        ext.copy_result_to_cpu(
            result,
            return_logprob=True,
            return_hidden_states=False,
            use_overlap_streams=True,
            copy_stream=copy_stream,
            forward_stream=forward_stream,
            copy_stream_ctx=nullcontext(),
        )
        copy_stream.wait_stream.assert_called_once_with(forward_stream)
        result.copy_to_cpu.assert_called_once_with(
            return_logprob=True, return_hidden_states=False
        )

        self.assertEqual(
            ext.prefill_adder_options(
                waiting_queue=[],
                chunked_req=None,
                mbs=None,
                last_mbs=None,
                tree_cache=None,
            ),
            (PrefillAdder, {}),
        )
        senders = [Mock(**{"drained.return_value": d}) for d in (True, False)]
        self.assertEqual(ext.group_drained(senders, None), [True, False])
        self.assertEqual(list(ext.poll_drained(senders, None)), [True, False])
        for sender in senders:
            sender.poll.assert_called_once_with()
        self.assertFalse(
            ext.needs_kv_free_fence(
                enable_overlap=True,
                uses_raiden_hicache=False,
                disaggregation_mode="null",
            )
        )
        self.assertIsNone(
            ext.create_pipeline_transport(
                model_runner=None,
                pp_group=None,
                stage_group=None,
                world_group=None,
                depth=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
