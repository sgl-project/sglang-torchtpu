import sys
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.result_copy_worker import (
    TpuAsyncCopyEvent,
    TPUAsyncCopyWorker,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class DelayedCopyEvent:
    def __init__(self):
        self.recorded = Event()
        self.waiting = Event()
        self.transferred = Event()

    def record(self):
        self.recorded.set()

    def synchronize(self):
        assert self.recorded.is_set(), "consumer waited on an unrecorded event"
        self.waiting.set()
        assert self.transferred.wait(timeout=5)

    def query(self):
        # Like a device event, an unrecorded event reports done.
        return not self.recorded.is_set() or self.transferred.is_set()


def test_consumer_waits_for_submission_and_transfer():
    event = DelayedCopyEvent()
    submitted = Future()
    completion = TpuAsyncCopyEvent(event, submitted)
    consumer_started = Event()

    def consume():
        consumer_started.set()
        completion.synchronize()

    with ThreadPoolExecutor(max_workers=1) as pool:
        consumer = pool.submit(consume)
        try:
            assert consumer_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                consumer.result(timeout=0.05)
            assert not event.waiting.is_set()
            completion.record()
            submitted.set_result(None)
            assert event.waiting.wait(timeout=5)
            assert not consumer.done()
            event.transferred.set()
            consumer.result(timeout=5)
        finally:
            if not submitted.done():
                submitted.set_exception(RuntimeError("test cleanup"))
            event.transferred.set()


def test_query_tracks_submission_then_transfer():
    """The prefill's continuous input polling calls query() on copy_done."""
    event = DelayedCopyEvent()
    submitted = Future()
    completion = TpuAsyncCopyEvent(event, submitted)

    # Not submitted yet: the device event would report done, query must not.
    assert not completion.query()
    event.record()
    assert not completion.query()
    submitted.set_result(None)
    assert not completion.query()
    event.transferred.set()
    assert completion.query()
    assert not event.waiting.is_set()


def test_query_reports_failed_submission_as_done():
    event = DelayedCopyEvent()
    submitted = Future()
    submitted.set_exception(RuntimeError("copy submission failed"))
    completion = TpuAsyncCopyEvent(event, submitted)

    assert completion.query()
    with pytest.raises(RuntimeError, match="copy submission failed"):
        completion.synchronize()
    assert not event.waiting.is_set()


def test_worker_error_reaches_consumer_before_event_wait():
    event = DelayedCopyEvent()
    submitted = Future()
    submitted.set_exception(RuntimeError("copy submission failed"))
    completion = TpuAsyncCopyEvent(event, submitted)

    with pytest.raises(RuntimeError, match="copy submission failed"):
        completion.synchronize()
    assert not event.waiting.is_set()


def test_worker_preserves_context_and_serializes_copies(monkeypatch):
    scheduler_thread = get_ident()
    private_stream = object()
    contexts = []
    first_started = Event()
    release_first = Event()
    second_started = Event()

    @contextmanager
    def context(kind, value):
        contexts.append((kind, value))
        try:
            yield
        finally:
            contexts.pop()

    device_module = SimpleNamespace(
        current_device=lambda: 3,
        Stream=Mock(return_value=private_stream),
        device=lambda value: context("device", value),
        stream=lambda value: context("stream", value),
    )

    get_device_module = Mock(return_value=device_module)
    monkeypatch.setattr(torch, "get_device_module", get_device_module)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(
        torch.accelerator, "device_index", lambda value: context("device", value)
    )

    def make_result(first):
        event = DelayedCopyEvent()
        event.transferred.set()
        result = SimpleNamespace(copy_done=event, next_draft_input=None)

        def copy_to_cpu(return_logprob, return_hidden_states):
            assert get_ident() != scheduler_thread
            assert torch.is_inference_mode_enabled()
            assert contexts == [("device", 3), ("stream", private_stream)]
            assert (return_logprob, return_hidden_states) == (True, False)
            if first:
                first_started.set()
                assert release_first.wait(timeout=5)
            else:
                second_started.set()
            result.copy_done.record()

        result.copy_to_cpu = copy_to_cpu
        return result, event

    first, first_event = make_result(True)
    second, second_event = make_result(False)
    worker = TPUAsyncCopyWorker()
    try:
        get_device_module.assert_called_once_with("tpu")
        device_module.Stream.assert_called_once_with(device=3)
        worker.submit(first, return_logprob=True, return_hidden_states=False)
        assert first_started.wait(timeout=5)
        assert not first_event.recorded.is_set()
        worker.submit(second, return_logprob=True, return_hidden_states=False)
        assert not second_started.is_set()
        release_first.set()
        first.copy_done.synchronize()
        second.copy_done.synchronize()
        assert first_event.waiting.is_set()
        assert second_event.waiting.is_set()
    finally:
        release_first.set()
        worker.close()
    assert contexts == []


@pytest.mark.parametrize("copy_draft_tail", [True, False])
def test_copy_done_covers_spec_draft_extend(monkeypatch, copy_draft_tail):
    """Draft-extend writes draft KV after the verify outputs are produced, so
    under the KV free fence copy_done must also wait for one of its outputs;
    otherwise the extra copy would only delay result processing."""
    device_module = SimpleNamespace(
        Stream=Mock(),
        stream=lambda value: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(torch, "get_device_module", Mock(return_value=device_module))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.accelerator,
        "device_index",
        lambda value: contextmanager(lambda: (yield))(),
    )
    issued = []
    draft_tail = object()
    topk_index = Mock()
    topk_index.to.side_effect = lambda *args, **kwargs: (
        issued.append("draft") or draft_tail
    )
    event = Mock()
    event.record.side_effect = lambda: issued.append("record")
    result = SimpleNamespace(
        copy_done=event, next_draft_input=EagleDraftInput(topk_index=topk_index)
    )
    result.copy_to_cpu = lambda *args: (
        issued.append("copy"),
        result.copy_done.record(),
    )

    worker = TPUAsyncCopyWorker(copy_draft_tail=copy_draft_tail)
    try:
        worker.submit(result, return_logprob=False, return_hidden_states=False)
        tail = result.copy_done.submitted.result(timeout=5)
    finally:
        worker.close()
    if copy_draft_tail:
        assert tail is draft_tail
        assert issued == ["draft", "copy", "record"]
    else:
        assert tail is None
        assert issued == ["copy", "record"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
