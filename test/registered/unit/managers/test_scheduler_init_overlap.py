"""Unit tests for Scheduler.init_overlap stream gating.

Overlap scheduling on a backend with real multi-stream execution (CUDA/ROCm)
uses CUDA-style forward/copy/schedule streams. A single-queue backend such as
torch_tpu serialises all device work on one PJRT queue, so cross-stream ordering
is a no-op and a non-default stream only relabels async-copy fences — there
overlap comes from async dispatch plus a scoped completion event instead.

``Platform.supports_overlap_streams()`` is the seam that selects between the two,
and ``Scheduler.init_overlap`` consumes it via ``self.use_overlap_streams``. These
tests pin that gating in isolation (no real worker / device), since the rest of
the overlap path is now shared between TPU and GPU.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_SCHED_MOD = "sglang.srt.managers.scheduler"


def _make_scheduler(*, enable_overlap: bool) -> Scheduler:
    """A bare Scheduler with only the attributes init_overlap touches."""
    s = Scheduler.__new__(Scheduler)
    s.device = "cpu"
    s.forward_stream = object()  # sentinel; only used on the stream path
    s.enable_overlap = enable_overlap
    s.spec_algorithm = MagicMock()
    # FutureMap is always-on; its factory reads the draft worker's attn backends
    # (target-only when there is no draft worker) and the req->token pool.
    s.draft_worker = None
    s.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(attn_backend=object()))
    s.req_to_token_pool = object()
    # Passed to decide_needs_cpu_seq_lens (patched in the runner) and to
    # decide_needs_confidence_relay, which reads speculative_algorithm.
    s.server_args = SimpleNamespace(speculative_algorithm=None)
    return s


def _run_init_overlap(s: Scheduler, *, use_mlx: bool, supports_streams: bool):
    """Invoke init_overlap with use_mlx() and the platform capability stubbed.

    Returns the MagicMock standing in for ``torch.get_device_module(device)`` so
    callers can assert whether stream factories were touched. ``needs_cpu_seq_lens``
    (True) and ``needs_confidence_relay`` (False) resolution is pinned so these
    stay focused on stream gating.
    """
    mock_platform = MagicMock()
    mock_platform.supports_overlap_streams.return_value = supports_streams
    mock_device_module = MagicMock()
    with patch(f"{_SCHED_MOD}.use_mlx", return_value=use_mlx), patch(
        f"{_SCHED_MOD}.current_platform", mock_platform
    ), patch(f"{_SCHED_MOD}.decide_needs_cpu_seq_lens", return_value=True), patch(
        f"{_SCHED_MOD}.decide_needs_confidence_relay", return_value=False
    ), patch(
        "torch.get_device_module", return_value=mock_device_module
    ):
        Scheduler.init_overlap(s)
    return mock_device_module


class TestInitOverlapStreamGating(CustomTestCase):
    def test_single_queue_skips_streams(self):
        """supports_overlap_streams=False -> nullcontext stand-ins, no real
        forward/copy/schedule streams, but the FutureMap is still built."""
        s = _make_scheduler(enable_overlap=True)
        dm = _run_init_overlap(s, use_mlx=False, supports_streams=False)

        self.assertFalse(s.use_overlap_streams)
        self.assertIsInstance(s.forward_stream_ctx, nullcontext)
        self.assertIsInstance(s.copy_stream_ctx, nullcontext)
        # No non-default streams are created on a single-queue backend.
        self.assertFalse(hasattr(s, "copy_stream"))
        self.assertFalse(hasattr(s, "schedule_stream"))
        dm.stream.assert_not_called()
        dm.Stream.assert_not_called()
        # Overlap bookkeeping is shared with the GPU path. FutureMap is always
        # built from (device, req_to_token_pool, needs_cpu_seq_lens).
        self.assertIs(s.future_map, s.spec_algorithm.create_future_map.return_value)
        s.spec_algorithm.create_future_map.assert_called_once_with(
            s.device,
            s.req_to_token_pool,
            needs_cpu_seq_lens=True,
            needs_confidence_relay=False,
        )
        self.assertEqual(s.batch_record_buf, [None, None])
        self.assertEqual(s.batch_record_ct, 0)

    def test_multi_stream_creates_streams(self):
        """supports_overlap_streams=True -> CUDA-style forward/copy streams from
        the device module (unchanged GPU behavior)."""
        s = _make_scheduler(enable_overlap=True)
        dm = _run_init_overlap(s, use_mlx=False, supports_streams=True)

        self.assertTrue(s.use_overlap_streams)
        dm.stream.assert_any_call(s.forward_stream)
        dm.Stream.assert_called_once_with()
        self.assertIs(s.copy_stream, dm.Stream.return_value)
        self.assertIs(s.forward_stream_ctx, dm.stream(s.forward_stream))
        self.assertIs(s.future_map, s.spec_algorithm.create_future_map.return_value)
        self.assertEqual(s.batch_record_buf, [None, None])

    def test_single_queue_overlap_disabled_still_builds_future_map(self):
        """With overlap disabled, the single-queue path still avoids streams, but
        FutureMap is always-on now (input_ids relay used in both modes), so it is
        built before the enable_overlap early-return."""
        s = _make_scheduler(enable_overlap=False)
        dm = _run_init_overlap(s, use_mlx=False, supports_streams=False)

        self.assertFalse(s.use_overlap_streams)
        self.assertIsInstance(s.forward_stream_ctx, nullcontext)
        self.assertIs(s.future_map, s.spec_algorithm.create_future_map.return_value)
        self.assertFalse(hasattr(s, "copy_stream"))
        dm.Stream.assert_not_called()

    def test_mlx_is_untouched(self):
        """MLX keeps its own early-return path: no streams and an empty
        result_queue, and it returns before forward_stream_ctx is set. FutureMap
        is still built first (always-on input_ids relay). use_mlx() wins
        regardless of the stream capability."""
        s = _make_scheduler(enable_overlap=True)
        dm = _run_init_overlap(s, use_mlx=True, supports_streams=True)

        self.assertFalse(s.use_overlap_streams)
        self.assertIs(s.future_map, s.spec_algorithm.create_future_map.return_value)
        self.assertEqual(len(s.result_queue), 0)
        self.assertFalse(hasattr(s, "forward_stream_ctx"))
        dm.stream.assert_not_called()
        dm.Stream.assert_not_called()


class TestLaunchBatchSampleStreamGating(CustomTestCase):
    """``launch_batch_sample_if_needed`` runs inside ``forward_stream_ctx`` and
    cross-waits the schedule stream — but that wait is only valid when
    ``use_overlap_streams`` is set. On a single-queue backend ``schedule_stream``
    is never created (see ``test_single_queue_skips_streams``), so an unguarded
    ``wait_stream`` raises AttributeError on the delayed-sample (spec-V2) path.
    Basic inference returns early (``delay_sample_func is None``) and never hits
    it, so this path needs its own coverage."""

    @staticmethod
    def _make_batch_result():
        br = SimpleNamespace()
        br.future_indices = object()
        br.logits_output = None
        br.copy_to_cpu = MagicMock()
        br.delay_sample_func = lambda: br  # must return the same result
        return br

    @staticmethod
    def _make_self(*, use_overlap_streams: bool) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.use_overlap_streams = use_overlap_streams
        s.forward_stream_ctx = nullcontext()
        s.copy_stream_ctx = nullcontext()
        # The relay itself (FutureMap stash vs ngram skip) is covered elsewhere;
        # here we only assert it is invoked with this iter's payload.
        s._relay_forward_payload = MagicMock()
        return s

    @staticmethod
    def _make_cur_batch() -> SimpleNamespace:
        return SimpleNamespace(return_logprob=False, return_hidden_states=False)

    def test_single_queue_skips_wait_stream(self):
        # Regression: with overlap streams off, schedule_stream is never created,
        # so the method must not touch it. Both stream attrs are left unset — an
        # unguarded wait_stream would AttributeError here.
        s = self._make_self(use_overlap_streams=False)
        br = self._make_batch_result()

        Scheduler.launch_batch_sample_if_needed(
            s, br, self._make_cur_batch()
        )  # must not raise

        br.copy_to_cpu.assert_called_once()
        s._relay_forward_payload.assert_called_once_with(br.future_indices, br)
        self.assertIsNone(br.delay_sample_func)  # closure released

    def test_multi_stream_waits_on_schedule_stream(self):
        # The multi-stream (GPU) path still cross-waits the schedule stream.
        s = self._make_self(use_overlap_streams=True)
        s.forward_stream = MagicMock()
        s.schedule_stream = MagicMock()
        s.copy_stream = MagicMock()
        br = self._make_batch_result()

        Scheduler.launch_batch_sample_if_needed(s, br, self._make_cur_batch())

        s.forward_stream.wait_stream.assert_called_once_with(s.schedule_stream)
        s.copy_stream.wait_stream.assert_called_once_with(s.forward_stream)
        br.copy_to_cpu.assert_called_once()

    def test_delayed_sample_releases_global_top_k_tensors(self):
        s = self._make_self(use_overlap_streams=False)
        br = self._make_batch_result()
        br.logits_output = SimpleNamespace(
            next_token_logits=object(),
            next_token_global_top_k_logits=object(),
            next_token_global_top_k_token_ids=object(),
        )

        Scheduler.launch_batch_sample_if_needed(s, br, self._make_cur_batch())

        self.assertIsNone(br.logits_output.next_token_logits)
        self.assertIsNone(br.logits_output.next_token_global_top_k_logits)
        self.assertIsNone(br.logits_output.next_token_global_top_k_token_ids)

    def test_no_delay_sample_func_is_noop(self):
        s = self._make_self(use_overlap_streams=False)
        # cur_batch=None proves the early return never touches the batch.
        out = Scheduler.launch_batch_sample_if_needed(
            s, SimpleNamespace(delay_sample_func=None), None
        )
        self.assertIsNone(out)
        s._relay_forward_payload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
