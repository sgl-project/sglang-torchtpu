"""CPU multiprocess coverage for shared embedding publication and reader leases."""

import multiprocessing
import os
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.shared_host_cache import SharedHostCache

register_cpu_ci(est_time=20, suite="stage-a-test-cpu-intel")


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def _reader(directory, queue):
    cache = SharedHostCache(directory, 4096, "encoder-v1")
    with cache.get((42, 5)) as entry:
        queue.put((entry.buffers[0].view(torch.uint16).tolist(), entry.spans))


def _paused_writer(directory, reserved, resume, queue):
    original = torch.from_file

    def paused(*args, **kwargs):
        reserved.set()
        if not resume.wait(20):
            raise TimeoutError("Test did not release the writer")
        return original(*args, **kwargs)

    torch.from_file = paused
    cache = SharedHostCache(directory, 4096, "encoder-v1")
    queue.put(cache.put((42, 5), [torch.arange(16).reshape(8, 2).to(torch.bfloat16)]))


def _crashed_writer(directory):
    cache = SharedHostCache(directory, 4096, "encoder-v1")
    torch.from_file = lambda *args, **kwargs: os._exit(17)
    cache.put((42, 5), [torch.ones(8, 2)])


def test_buffers_are_shared_between_processes_without_changing_bits(tmp_path):
    cache = SharedHostCache(tmp_path, 4096, "encoder-v1")
    raw = torch.tensor([0, 1, 0x8000, 0x7FC1, 0xFFFF, 0x3F80], dtype=torch.uint16)
    buffer = raw.view(torch.bfloat16).reshape(3, 2)
    spans = ((0, 0, 2), (0, 2, 1))
    assert cache.put((42, 5), [buffer], spans) == "inserted"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_reader, args=(str(tmp_path), queue))
    process.start()
    value, received_spans = queue.get(timeout=30)
    process.join(30)
    assert process.exitcode == 0
    assert value == raw.reshape(3, 2).tolist()
    assert received_spans == spans


def test_lru_skips_leased_entries_and_charges_them_to_capacity(tmp_path):
    cache = SharedHostCache(tmp_path, 32, "encoder-v1")
    a = torch.arange(4, dtype=torch.int32)
    assert cache.put((1, 4), [a]) == "inserted"
    assert cache.put((2, 4), [a + 1]) == "inserted"
    with cache.get((1, 4)) as first, cache.get((2, 4)) as second:
        assert cache.put((3, 4), [a + 2]) == "busy"
        assert cache.stats()["nbytes"] == 32
        assert torch.equal(first.buffers[0], a)
        assert torch.equal(second.buffers[0], a + 1)
    with cache.get((1, 4)):
        pass
    assert cache.put((3, 4), [a + 2]) == "inserted"
    assert cache.get((2, 4)) is None
    with cache.get((1, 4)) as entry:
        assert torch.equal(entry.buffers[0], a)
    assert cache.stats()["nbytes"] == 32


def test_no_partial_publication_and_duplicate_writer_does_not_overwrite(tmp_path):
    cache = SharedHostCache(tmp_path, 4096, "encoder-v1")
    ctx = multiprocessing.get_context("spawn")
    reserved, resume, queue = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(
        target=_paused_writer, args=(str(tmp_path), reserved, resume, queue)
    )
    process.start()
    try:
        assert reserved.wait(30)
        assert cache.get((42, 5)) is None
        assert cache.stats()["pending"] == 1
        assert cache.put((42, 5), [torch.zeros(8, 2)]) == "pending"
        resume.set()
        assert queue.get(timeout=30) == "inserted"
    finally:
        resume.set()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert process.exitcode == 0
    with cache.get((42, 5)) as entry:
        assert torch.equal(
            entry.buffers[0], torch.arange(16).reshape(8, 2).to(torch.bfloat16)
        )


def test_dead_writer_is_recovered_without_publishing_partial_bytes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_crashed_writer, args=(str(tmp_path),))
    process.start()
    process.join(30)
    assert process.exitcode == 17
    cache = SharedHostCache(tmp_path, 4096, "encoder-v1")
    assert cache.get((42, 5)) is None
    assert cache.stats() == dict(entries=0, nbytes=0, pending=0)
    assert cache.put((42, 5), [torch.ones(8, 2)]) == "inserted"


def test_namespace_and_capacity_must_be_explicit(tmp_path):
    cache = SharedHostCache(tmp_path, 4096, "encoder-v1")
    cache.put((42, 5), [torch.ones(8, 2)])
    other = SharedHostCache(tmp_path, 4096, "encoder-v2")
    assert other.get((42, 5)) is None
    with pytest.raises(ValueError, match="configuration differs"):
        SharedHostCache(tmp_path, 8192, "encoder-v1")
    with pytest.raises(ValueError, match="positive budget and namespace"):
        SharedHostCache(tmp_path, 4096, "")
    with pytest.raises(ValueError, match="content identity"):
        cache.get((None, 5))


def test_multiple_buffers_strides_padding_and_oversized_entry(tmp_path):
    cache = SharedHostCache(tmp_path, 4096, "encoder-v1")
    a = torch.arange(48, dtype=torch.int16).reshape(8, 6).transpose(0, 1)
    b = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    spans = ((0, 1, 3), (1, 0, 4))
    assert cache.put((42, 7), [a, b], spans) == "inserted"
    with cache.get((42, 7)) as entry:
        assert torch.equal(bits(entry.buffers[0]), bits(a))
        assert torch.equal(bits(entry.buffers[1]), bits(b))
        assert entry.spans == spans
    assert cache.put((43, 1), [torch.ones(4097, dtype=torch.uint8)]) == "too_big"
    assert cache.stats()["entries"] == 1
    orphan = Path(tmp_path) / ("a" * 64 + ".bin")
    orphan.write_bytes(b"partial orphan")
    SharedHostCache(tmp_path, 4096, "encoder-v1")
    assert not orphan.exists()


def test_corrupted_buffer_length_fails_loudly(tmp_path):
    cache = SharedHostCache(tmp_path, 4096, "encoder-v1")
    cache.put((42, 5), [torch.ones(8, 2)])
    path = next(Path(tmp_path).glob("*.bin"))
    with path.open("r+b") as file:
        file.truncate(1)
    with pytest.raises(RuntimeError, match="invalid length"):
        cache.get((42, 5))
