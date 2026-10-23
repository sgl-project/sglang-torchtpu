from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace as NS

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import pd_swa_transfer_window_start
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

PAGE, WINDOW, ROWS = 128, 128, 32768


class Allocator:
    def __init__(self):
        self.freed = []

    def free_swa_segment(self, indices, *, start_pos):
        self.freed.append(indices.clone())


def scheduler(swa=True, window=WINDOW, page=PAGE):
    return NS(
        tree_cache=NS(
            supports_prefix_sharing=lambda: False,
            supports_swa=lambda: swa,
            sliding_window_size=window,
            page_size=page,
        ),
        req_to_token_pool=NS(
            req_to_token=torch.arange(ROWS, dtype=torch.int64).view(1, ROWS)
        ),
        token_to_kv_pool_allocator=Allocator(),
    )


def req(tokens):
    return NS(
        origin_input_ids=[0] * tokens,
        extend_range=NS(end=tokens),
        kv=ReqKvInfo(req_pool_idx=0, kv_committed_len=tokens, kv_allocated_len=tokens),
    )


def release(sched, request):
    SchedulerDisaggregationPrefillMixin._release_swa_beyond_window(sched, request)
    return sched.token_to_kv_pool_allocator.freed


def payload_window_start(seq_len, window=WINDOW, page=PAGE):
    # The SWA payload sent to decode starts at this page boundary.
    return pd_swa_transfer_window_start(seq_len, window, page)


def test_finished_prefill_keeps_only_the_window():
    sched, request = scheduler(), req(16384)
    freed = release(sched, request)
    assert len(freed) == 1 and freed[0].tolist() == list(range(16384 - WINDOW))
    assert (
        request.kv.get_evicted_seqlen(ComponentType.SWA)
        == payload_window_start(16384)
        == 16256
    )
    assert release(sched, request) == freed  # a second release frees nothing more


def test_partial_page_stops_at_the_payload_boundary():
    sched, request = scheduler(), req(16300)
    freed = release(sched, request)
    assert freed[0].numel() == payload_window_start(16300) == 16128
    assert request.kv.get_evicted_seqlen(ComponentType.SWA) == 16128


def test_window_not_a_page_multiple_keeps_the_page_the_payload_starts_in():
    # gpt-oss: a 127-token window at 64-token pages. For L % 64 == 63 the
    # payload starts a page before page_floor(L - W); the release must keep it.
    window, page = 127, 64
    for seq_len, start in ((16383, 16192), (16384, 16256), (16320, 16192)):
        sched, request = scheduler(window=window, page=page), req(seq_len)
        freed = release(sched, request)
        assert start == payload_window_start(seq_len, window, page)
        assert len(freed) == 1 and freed[0].tolist() == list(range(start))
        assert request.kv.get_evicted_seqlen(ComponentType.SWA) == start
        assert (seq_len // page) * page - start >= window


def test_release_keeps_exactly_the_transfer_window_for_every_length():
    # Sweep window/page shapes: the release frees exactly [0, start) where
    # start is the transfer window start, never a page the payload sends, and
    # what it keeps holds a full window before the page-aligned boundary.
    for window, page in (
        (127, 64),
        (65, 64),
        (64, 64),
        (50, 128),
        (2047, 512),
        (1000, 512),
    ):
        for seq_len in range(1, 6 * page + window):
            sched, request = scheduler(window=window, page=page), req(seq_len)
            freed = release(sched, request)
            start = payload_window_start(seq_len, window, page)
            if start == 0:
                assert freed == [], (window, page, seq_len)
                assert request.kv.get_evicted_seqlen(ComponentType.SWA) == 0
                continue
            assert len(freed) == 1, (window, page, seq_len)
            assert freed[0].tolist() == list(range(start)), (window, page, seq_len)
            assert request.kv.get_evicted_seqlen(ComponentType.SWA) == start
            assert (seq_len // page) * page - start >= window
            # Releasing again frees nothing more.
            assert release(sched, request) == freed


def test_window_smaller_than_a_page_keeps_at_least_the_last_full_page():
    window, page = 50, 128
    # 300 tokens: boundary 256, 256 - 50 = 206 -> page 128. Page 1 stays
    # because the window reaches into it from the boundary.
    sched, request = scheduler(window=window, page=page), req(300)
    freed = release(sched, request)
    assert payload_window_start(300, window, page) == 128
    assert freed[0].tolist() == list(range(128))
    # 383 tokens: same boundary, same start, same release.
    sched, request = scheduler(window=window, page=page), req(383)
    assert release(sched, request)[0].tolist() == list(range(128))
    # 384 tokens: boundary 384 -> start 256.
    sched, request = scheduler(window=window, page=page), req(384)
    assert release(sched, request)[0].tolist() == list(range(256))


def test_window_equal_to_a_page_releases_like_before():
    window = page = 64
    for seq_len, start in ((127, 0), (128, 64), (191, 64), (192, 128), (1000, 896)):
        sched, request = scheduler(window=window, page=page), req(seq_len)
        freed = release(sched, request)
        assert payload_window_start(seq_len, window, page) == start
        assert (max(0, seq_len - window) // page) * page == start  # unchanged
        assert [t.tolist() for t in freed] == ([list(range(start))] if start else [])


def test_prompt_shorter_than_a_window_plus_a_page_frees_nothing():
    window, page = 127, 64
    # start is 0 until page_floor(L) - W reaches a page: page_floor(L) >= 191,
    # i.e. L >= 192 (not L >= W + P = 191, since 191 - 127 = 64 only on the
    # page-aligned boundary 192).
    for seq_len in range(1, 192):
        sched, request = scheduler(window=window, page=page), req(seq_len)
        assert release(sched, request) == [], seq_len
    for seq_len in (192, 200, 255):
        sched, request = scheduler(window=window, page=page), req(seq_len)
        assert release(sched, request)[0].tolist() == list(range(64)), seq_len


def test_prompt_within_the_window_frees_nothing():
    sched, request = scheduler(), req(100)
    assert release(sched, request) == []
    assert request.kv.get_evicted_seqlen(ComponentType.SWA) == 0


def test_only_sliding_window_chunk_caches_release():
    sched, request = scheduler(swa=False), req(16384)
    assert release(sched, request) == []


def test_released_kv_is_left_alone():
    sched, request = scheduler(), req(16384)
    request.kv.req_pool_idx = None
    assert release(sched, request) == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
