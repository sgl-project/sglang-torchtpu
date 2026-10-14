"""The PP disaggregated prefill hands the Raiden pipeline transfer only real
Raiden requests' final chunks: health-check requests carry a FakeKVSender with
no Raiden room, and handing one over crashed the prefill scheduler."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _raiden_sender(room):
    sender = RaidenKVSender.__new__(RaidenKVSender)
    sender.bootstrap_room = room
    return sender


def _req(sender, n_input, extend_end, pool_idx=0):
    return SimpleNamespace(
        disagg_kv_sender=sender,
        origin_input_ids=list(range(n_input)),
        extend_range=SimpleNamespace(end=extend_end),
        kv=SimpleNamespace(req_pool_idx=pool_idx),
    )


def _scheduler(pipeline, state_types=()):
    kv_manager = SimpleNamespace(
        _pipeline=pipeline, kv_args=SimpleNamespace(state_types=list(state_types))
    )
    return SimpleNamespace(
        transfer_backend=TransferBackend.RAIDEN,
        disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=kv_manager),
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=4, translate_swa_indices_for_transfer=lambda t: t
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(4 * 64, dtype=torch.int32).reshape(4, 64)
        ),
        sliding_window_size=8,
    )


def test_only_final_chunks_of_raiden_requests_are_handed_over():
    calls, fences = [], []
    pipeline = SimpleNamespace(
        stage_parts=True,
        stage_ready=lambda *args: calls.append(args),
        make_fence=lambda output: fences.append(output) or "fence",
        advance_stages=lambda: None,
    )
    fake = FakeKVSender.__new__(FakeKVSender)
    fake.bootstrap_room = None
    batch = SimpleNamespace(
        reqs=[
            _req(fake, 16, 16, 1),
            _req(_raiden_sender(77), 16, 8, 2),  # not the final chunk
            _req(_raiden_sender(77), 16, 16, 3),
        ]
    )
    output = torch.zeros(1)
    result = SimpleNamespace(
        pp_hidden_states_proxy_tensors=SimpleNamespace(tensors={"h": output})
    )
    SchedulerPPMixin._pp_pd_stage_ready(
        _scheduler(pipeline, [StateType.SWA]), batch, result
    )
    assert fences == [output]
    # Row 3 holds tokens 192..255: pages 48..51, the 8-token window pages 50..51.
    assert calls == [(77, [48, 49, 50, 51], [50, 51], "fence")]


def test_no_fence_without_a_final_chunk():
    fences = []
    pipeline = SimpleNamespace(
        stage_parts=True,
        stage_ready=None,
        make_fence=fences.append,
        advance_stages=lambda: None,
    )
    batch = SimpleNamespace(reqs=[_req(_raiden_sender(5), 16, 8)])
    SchedulerPPMixin._pp_pd_stage_ready(_scheduler(pipeline), batch, None)
    assert fences == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
