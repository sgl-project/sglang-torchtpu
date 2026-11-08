"""A PD prefill chunk honours the chunk-insert stride unless it is the prompt's last."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _req(end, prompt_len=10):
    return SimpleNamespace(
        pending_bootstrap=False,
        disagg_kv_sender=SimpleNamespace(),
        extend_range=SimpleNamespace(start=0, end=end),
        origin_input_ids=list(range(prompt_len)),
    )


class TestPdChunkCheckpointStride(unittest.TestCase):
    def _checkpoint(self, req):
        scheduler = SimpleNamespace(tree_cache=object())
        with patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache") as ckpt:
            SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(
                scheduler, req
            )
        return ckpt

    def test_chunk_before_the_last_is_mid_prefill(self):
        req = _req(end=4)
        ckpt = self._checkpoint(req)
        ckpt.assert_called_once()
        self.assertTrue(ckpt.call_args.kwargs["mid_prefill"])

    def test_last_chunk_always_publishes(self):
        req = _req(end=10)
        ckpt = self._checkpoint(req)
        ckpt.assert_called_once()
        self.assertFalse(ckpt.call_args.kwargs["mid_prefill"])


if __name__ == "__main__":
    unittest.main()
