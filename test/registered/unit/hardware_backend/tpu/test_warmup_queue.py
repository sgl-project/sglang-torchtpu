"""Unit tests for global TPU warmup collection."""

import contextvars
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.warmup_queue import (
    collect_warmups,
    current_warmup_queue,
    eager_warmup,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_collect_warmups_nests_and_preserves_context():
    value = contextvars.ContextVar("value", default="outside")
    events = []
    with collect_warmups() as queue:
        value.set("captured")
        with collect_warmups() as nested:
            assert nested is queue
            nested.calls.append(PendingCall(lambda: events.append(value.get())))
        value.set("changed")
        queue.completions.append(lambda: events.append("complete"))
        assert current_warmup_queue() is queue

    assert events == ["captured", "complete"]
    assert current_warmup_queue() is None
    assert queue.calls == []
    assert queue.completions == []


def test_collect_warmups_releases_inputs_between_calls():
    events = []

    def enqueue(queue):
        tensor = torch.ones(8)
        queue.calls.append(PendingCall(lambda: events.append(tensor.numel())))
        return weakref.ref(tensor)

    with collect_warmups() as queue:
        reference = enqueue(queue)
        queue.calls.append(PendingCall(lambda: events.append(reference() is None)))
        assert reference() is not None

    assert events == [8, True]


def test_collect_warmups_clears_after_failure():
    completed = []

    def fail():
        raise RuntimeError("warmup failed")

    with pytest.raises(RuntimeError, match="warmup failed"):
        with collect_warmups() as queue:
            queue.calls.append(PendingCall(fail))
            queue.completions.append(lambda: completed.append(True))

    assert current_warmup_queue() is None
    assert completed == []
    assert queue.calls == []
    assert queue.completions == []


def test_pending_call_restores_forward_context():
    first_backend = object()
    second_backend = object()
    with forward_context(ForwardContext(attn_backend=first_backend)):
        pending = PendingCall(lambda: get_forward_context().attn_backend)

    with forward_context(ForwardContext(attn_backend=second_backend)):
        assert pending.wait() is first_backend
        assert get_forward_context().attn_backend is second_backend


@pytest.mark.parametrize("fail", [False, True])
def test_eager_warmup_joins_before_releasing_destinations(monkeypatch, fail):
    events = []
    destinations = []

    class Source:
        def to(self, device, *, non_blocking):
            assert device == "cpu" and non_blocking
            events.append("submit")
            destination = torch.empty(1)
            destinations.append(weakref.ref(destination))
            return destination

    def synchronize():
        assert events == ["submit", "submit"]
        assert all(reference() is not None for reference in destinations)
        events.append("join")

    stream = SimpleNamespace(synchronize=synchronize)
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(current_stream=lambda: stream), raising=False
    )

    def warmup():
        with eager_warmup() as materialize:
            materialize(Source())
            materialize(Source())
            assert events == ["submit", "submit"]
            if fail:
                raise ValueError("failed after submission")
        assert all(reference() is None for reference in destinations)

    if fail:
        with pytest.raises(ValueError, match="failed after submission"):
            warmup()
    else:
        warmup()
    assert events == ["submit", "submit", "join"]
    assert all(reference() is None for reference in destinations)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
