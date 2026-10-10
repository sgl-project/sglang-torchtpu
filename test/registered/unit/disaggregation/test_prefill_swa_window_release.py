from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace as NS

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

PAGE, WINDOW, ROWS = 128, 128, 32768


class Allocator:
    def __init__(self):
        self.freed = []

    def free_swa_segment(self, indices, *, start_pos):
        self.freed.append(indices.clone())


def scheduler(swa=True):
    return NS(
        tree_cache=NS(
            supports_prefix_sharing=lambda: False,
            supports_swa=lambda: swa,
            sliding_window_size=WINDOW,
            page_size=PAGE,
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


def payload_window_start(seq_len):
    # The SWA payload sent to decode starts at this page boundary.
    return (max(0, seq_len - WINDOW) // PAGE) * PAGE


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
