# SPDX-License-Identifier: Apache-2.0
"""CPU tests: SGLANG_TPU_DSV41_ENGRAM_PACK_THREADS gathers large selections on torch's
intra-op threads with output byte-identical to the NumPy path; small selections and the
default keep the NumPy path; the thread count is capped by the CPU affinity set."""

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41 import engram_host as eh

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HEADS, DIM, BLOCK = 4, 64, 32


@pytest.fixture(autouse=True)
def _reset():
    threads = torch.get_num_threads()
    eh._PACK_THREADS_STATE.update(threads=None, calls=0, ms=0.0)
    yield
    torch.set_num_threads(threads)
    eh._PACK_THREADS_STATE.update(threads=None, calls=0, ms=0.0)


def _table(rows, seed):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (rows, DIM), dtype=torch.uint8, generator=g)
    scale = torch.randint(0, 255, (rows, DIM // BLOCK), dtype=torch.uint8, generator=g)
    return weight, scale


def _pack(monkeypatch, weight, scale, idx, padded, threads, min_rows, row_start=0):
    monkeypatch.setenv("SGLANG_TPU_DSV41_ENGRAM_PACK_THREADS", str(threads))
    monkeypatch.setenv("SGLANG_TPU_DSV41_ENGRAM_PACK_THREADS_MIN_ROWS", str(min_rows))
    out = torch.full((eh.packed_row_words(HEADS, DIM, padded),), -3, dtype=torch.int32)
    return eh.pack_rows(
        weight, scale, idx, row_start=row_start, padded_tokens=padded, out=out
    )


@pytest.mark.parametrize(
    "tokens,padded,row_start", [(40, 48, 0), (33, 33, 0), (20, 32, 50)]
)
def test_threaded_gather_is_byte_identical(monkeypatch, tokens, padded, row_start):
    weight, scale = _table(200, tokens)
    rng = np.random.default_rng(tokens)
    idx = torch.from_numpy(rng.integers(0, 260, size=(tokens, HEADS)).astype(np.int64))
    ref = _pack(monkeypatch, weight, scale, idx, padded, 0, 8192, row_start)
    got = _pack(monkeypatch, weight, scale, idx, padded, 2, 1, row_start)
    assert torch.equal(got, ref)
    assert eh._PACK_THREADS_STATE["calls"] == 1


def test_small_gathers_keep_numpy(monkeypatch):
    weight, scale = _table(100, 1)
    idx = torch.randint(0, 100, (8, HEADS))
    _pack(monkeypatch, weight, scale, idx, 8, 4, 8192)
    assert eh._PACK_THREADS_STATE["calls"] == 0
    assert eh._PACK_THREADS_STATE["threads"] is None


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_TPU_DSV41_ENGRAM_PACK_THREADS", raising=False)
    weight, scale = _table(100, 2)
    idx = torch.randint(0, 100, (64, HEADS))
    out = torch.zeros(eh.packed_row_words(HEADS, DIM, 64), dtype=torch.int32)
    eh.pack_rows(weight, scale, idx, row_start=0, padded_tokens=64, out=out)
    assert eh._PACK_THREADS_STATE["calls"] == 0


def test_thread_count_capped_by_affinity(monkeypatch):
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert eh._engage_pack_threads(8) == 2
    assert torch.get_num_threads() == 2
    assert eh._engage_pack_threads(8) == 2  # set once per process
