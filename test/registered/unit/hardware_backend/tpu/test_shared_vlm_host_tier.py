"""CPU checks for the VLM adapter; real TPU uploads require device validation."""

import contextlib
import threading

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.mm_embedding import EmbeddingSpan, PaddedEmbedding
from sglang_tpu.mm_schedule_tpu import _VlmHostTier
from sglang_tpu.shared_vlm_host_tier import VlmSharedHostTier

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class DeviceModule:
    class Stream:
        def __init__(self, device=None):
            self.device = device

    class Event:
        def record(self):
            return None

        def synchronize(self):
            return None

    @staticmethod
    def stream(stream):
        return contextlib.nullcontext()


@pytest.fixture
def make_tier(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "get_device_module", lambda device: DeviceModule)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.accelerator, "device_index", lambda i: contextlib.nullcontext()
    )
    original = _VlmHostTier._to_host

    def host_copy(embedding):
        entry = original(embedding)
        entry.buffers = [b.clone() for b in entry.buffers]
        return entry

    monkeypatch.setattr(_VlmHostTier, "_to_host", staticmethod(host_copy))
    made = []

    def create(cap=4096, verify=False):
        tier = VlmSharedHostTier(cap, verify, 1e9, tmp_path, "encoder-v1")
        made.append(tier)
        return tier

    yield create
    for tier in made:
        tier._pool.shutdown(wait=True)
        tier._uploads.shutdown(wait=True)


def drain(pool):
    pool.submit(lambda: None).result(timeout=10)


def test_second_engine_restores_all_padding_and_valid_spans(make_tier):
    a, b = make_tier(), make_tier(verify=True)
    buffer = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    buffer[5] = float("nan")
    fresh = PaddedEmbedding((EmbeddingSpan(buffer, 1, 3),))
    a.write_back((42, 3), fresh)
    drain(a._pool)
    restored, entry = b.lookup((42, 3), "cpu")
    assert b.c["host_hit"] == 1
    assert torch.equal(
        restored.to_host().view(torch.uint8), fresh.to_host().view(torch.uint8)
    )
    assert b.verify_pair((42, 3), fresh, entry)
    assert entry.lease.fd == -1
    assert b.c["shared_h2d_complete"] == 1


def test_source_remains_leased_until_upload_event_completes(make_tier, monkeypatch):
    a, b = make_tier(cap=32), make_tier(cap=32)
    buffer = torch.ones(4, 2)
    a.write_back((42, 4), buffer)
    drain(a._pool)
    gate = threading.Event()

    class DelayedEvent(DeviceModule.Event):
        def synchronize(self):
            if not gate.wait(10):
                raise TimeoutError("Test upload was not released")

    monkeypatch.setattr(DeviceModule, "Event", DelayedEvent)
    try:
        b.lookup((42, 4), "cpu")
        assert a.shared.put((43, 4), [buffer]) == "busy"
        gate.set()
        drain(b._uploads)
        assert a.shared.put((43, 4), [buffer]) == "inserted"
    finally:
        gate.set()


def test_background_storage_failure_is_visible_on_the_scheduler(make_tier, monkeypatch):
    tier = make_tier()

    def failed_put(*args):
        raise OSError("storage failed")

    monkeypatch.setattr(tier.shared, "put", failed_put)
    tier.write_back((42, 4), torch.ones(4, 2))
    drain(tier._pool)
    with pytest.raises(RuntimeError, match="cache worker failed"):
        tier.lookup((42, 4), "cpu")
    assert tier.c["wb_error"] == 1


def test_verification_mismatch_raises_and_releases_the_lease(make_tier):
    a, b = make_tier(), make_tier(verify=True)
    a.write_back((42, 4), torch.ones(4, 2))
    drain(a._pool)
    restored, entry = b.lookup((42, 4), "cpu")
    with pytest.raises(RuntimeError, match="differs from a fresh encode"):
        b.verify_pair((42, 4), torch.zeros(4, 2), entry)
    assert entry.lease.fd == -1


def test_padding_mismatch_fails_even_when_valid_rows_match(make_tier):
    a, b = make_tier(), make_tier(verify=True)
    buffer = torch.ones(6, 2)
    fresh = PaddedEmbedding((EmbeddingSpan(buffer, 0, 4),))
    a.write_back((42, 4), fresh)
    drain(a._pool)
    restored, entry = b.lookup((42, 4), "cpu")
    buffer[5] = 7
    with pytest.raises(RuntimeError, match="differs from a fresh encode"):
        b.verify_pair((42, 4), fresh, entry)
    assert entry.lease.fd == -1
    assert b.c["verify_full_diff"] == 1
