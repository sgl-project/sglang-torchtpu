"""The PP disaggregated prefill hands the Raiden pipeline transfer each request's
KV on a stage before the stage's result point: only real Raiden requests' final
chunks (health-check requests carry a FakeKVSender with no Raiden room, and
handing one over crashed the prefill scheduler), and the result point keeps
the pages it exposed out of the radix cache."""

import sys
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import pipeline_final_chunks
from sglang_tpu.disaggregation.raiden.pipeline_final_chunks import FinalChunks
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _raiden_sender(room):
    sender = RaidenKVSender.__new__(RaidenKVSender)
    sender.bootstrap_room = room
    return sender


def _req(sender, n_input, extend_end, pool_idx=0, **fields):
    return SimpleNamespace(
        disagg_kv_sender=sender,
        origin_input_ids=list(range(n_input)),
        extend_range=SimpleNamespace(end=extend_end),
        kv=SimpleNamespace(req_pool_idx=pool_idx),
        **fields,
    )


def _first_token(row, **overrides):
    """Fields that decide whether the first token leaves before the result point."""
    return (
        dict(
            return_logprob=False,
            return_sampling_mask=False,
            grammar=None,
            pending_bootstrap=False,
            metadata_buffer_index=row,
        )
        | overrides
    )


def _scheduler(rows=None):
    """Request rows of 64 tokens in 4-token pages; metadata rows keep the token."""
    rows = {} if rows is None else rows
    return SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=4, translate_swa_indices_for_transfer=lambda t: t
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(4 * 64, dtype=torch.int32).reshape(4, 64)
        ),
        sliding_window_size=8,
        kv_checksum_computer=None,
        disagg_metadata_buffers=SimpleNamespace(
            set_buf_with_first_token=lambda req, token: rows.__setitem__(
                req.metadata_buffer_index, token
            )
        ),
    )


def _transfer(calls, *, pp_rank=0, stage_parts=False, early_close=False, swa=False):
    return SimpleNamespace(
        stage_parts=stage_parts,
        early_close=early_close,
        manager=SimpleNamespace(
            pp_rank=pp_rank,
            pp_size=2,
            kv_args=SimpleNamespace(
                state_types=[StateType.SWA] if swa else [],
                aux_data_ptrs=None,
                aux_item_lens=None,
            ),
        ),
        advance_stages=lambda: None,
        make_fence=lambda output: calls.append(("fence", output)) or "fence",
        stage_ready=lambda *args: calls.append(("ready",) + args),
        close_early=lambda *args: calls.append(("close",) + args),
    )


def test_only_final_chunks_of_raiden_requests_are_handed_over():
    calls = []
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
    transfer = _transfer(calls, stage_parts=True, swa=True)
    FinalChunks(transfer).launched(_scheduler(), 0, batch, result)
    # Row 3 holds tokens 192..255: pages 48..51, the 8-token window pages 50..51.
    assert calls == [
        ("fence", output),
        ("ready", 77, [48, 49, 50, 51], [50, 51], "fence"),
    ]


def test_no_fence_without_a_final_chunk():
    calls = []
    batch = SimpleNamespace(reqs=[_req(_raiden_sender(5), 16, 8)])
    FinalChunks(_transfer(calls, stage_parts=True)).launched(
        _scheduler(), 0, batch, None
    )
    assert calls == []


def test_early_close_closes_at_launch_and_on_the_last_stages_token_copy(monkeypatch):
    """Stages before the last hand their share over at launch. The last stage
    waits for the PP loop's copy of that forward's sampled tokens, and sends
    the first token with its share unless the result point must (logprobs)."""
    rows = {}
    monkeypatch.setattr(
        pipeline_final_chunks,
        "read_aux_blob",
        lambda ptrs, lens, row: b"%d:%d" % (row, rows[row]),
    )
    plain = _req(_raiden_sender(5), 16, 16, 1, **_first_token(3))
    logprobs = _req(
        _raiden_sender(6), 16, 16, 2, **_first_token(4, return_logprob=True)
    )
    batch = SimpleNamespace(reqs=[plain, logprobs])
    calls = []
    FinalChunks(_transfer(calls, early_close=True)).launched(
        _scheduler(rows), 1, batch, None
    )
    # Row 1 holds tokens 64..127: pages 16..19; row 2: pages 32..35.
    first, second = [16, 17, 18, 19], [32, 33, 34, 35]
    assert calls == [("close", 5, first, []), ("close", 6, second, [])]
    calls.clear()
    last = FinalChunks(_transfer(calls, pp_rank=1, early_close=True))
    last.launched(_scheduler(rows), 1, batch, None)
    assert calls == []
    last.tokens_copied(1, torch.tensor([11, 12]))
    assert calls == [("close", 5, first, [], b"3:11"), ("close", 6, second, [], b"")]
    assert not last.held


def test_last_stage_matches_tokens_to_the_batch_as_launched(monkeypatch):
    """Copied tokens are matched by each request's position when its batch was
    launched, health checks included. A token count that does not match that
    batch leaves every first token to the result point, logged once."""
    rows = {}
    monkeypatch.setattr(
        pipeline_final_chunks,
        "read_aux_blob",
        lambda ptrs, lens, row: b"%d:%d" % (row, rows[row]),
    )
    fake = FakeKVSender.__new__(FakeKVSender)
    fake.bootstrap_room = None
    health = _req(fake, 16, 16, 0)
    plain = _req(_raiden_sender(5), 16, 16, 1, **_first_token(3))
    batch = SimpleNamespace(reqs=[health, plain])
    calls = []
    last = FinalChunks(_transfer(calls, pp_rank=1, early_close=True))
    last.launched(_scheduler(rows), 0, batch, None)
    batch.reqs.remove(health)
    last.tokens_copied(0, torch.tensor([7, 11]))
    assert calls == [("close", 5, [16, 17, 18, 19], [], b"3:11")]
    calls.clear()
    logger = Mock()
    monkeypatch.setattr(pipeline_final_chunks, "logger", logger)
    for _ in range(2):
        last.launched(_scheduler(rows), 0, SimpleNamespace(reqs=[plain]), None)
        last.tokens_copied(0, torch.tensor([7, 11]))
    assert calls == [("close", 5, [16, 17, 18, 19], [], b"")] * 2
    logger.warning.assert_called_once()


def test_prefill_loop_reports_launches_and_the_tokens_its_send_copied():
    """The KV manager learns of each launch, and of the sampled tokens as the
    copy the deferred send ships, so they leave the device once; a slot that
    launched nothing reports nothing."""
    copies = []

    class DeviceTokens:
        def cpu(self):
            copies.append(torch.tensor([11, 12]))
            return copies[-1]

    shipped = []

    def ship(tensors, **kwargs):
        # The wave sends each tensor as a host copy.
        shipped.append({key: value.cpu() for key, value in tensors.items()})
        return ["work"]

    kv_manager = Mock()
    scheduler = SchedulerPPMixin()
    scheduler.pp_comm_stream_ctx = nullcontext()
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(kv_manager=kv_manager)
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, reqs=[])
    scheduler._pp_pd_stage_ready(1, batch, "result")
    kv_manager.on_forward_launched.assert_called_once_with(
        scheduler, 1, batch, "result"
    )
    scheduler.pp_group = SimpleNamespace(is_last_rank=True)
    scheduler._pp_wave = object()
    scheduler.mbs = [None, batch]
    scheduler.pp_outputs = None
    scheduler.last_rank_comm_queue = deque(
        [(None, PPProxyTensors({"next_token_ids": DeviceTokens()}))]
    )
    scheduler._pp_send_dict_to_next_stage = ship
    scheduler._pp_pd_send_deferred_output(0)
    kv_manager.on_outputs_copied.assert_not_called()
    scheduler._pp_pd_send_deferred_output(1)
    (copy,) = copies
    assert shipped[0]["next_token_ids"] is copy
    kv_manager.on_outputs_copied.assert_called_once_with(1, copy)
    assert scheduler.send_output_work == ["work"]


def test_result_point_keeps_pages_exposed_before_it_out_of_the_radix_cache():
    """With stage groups or early close, a transfer may read a stage's pages
    before the request's final result. Inserting them there freed the ones the
    tree already held for the same tokens, under that transfer; the final
    chunk's insert waits for release_kv_cache, earlier chunks insert as before."""
    pipeline = SimpleNamespace(stage_parts=False, early_close=False)
    sender = _raiden_sender(5)
    sender.kv_mgr = SimpleNamespace(_pipeline=pipeline)
    final, middle = (_req(sender, 16, end, pending_bootstrap=False) for end in (16, 8))
    scheduler = SimpleNamespace(tree_cache=Mock())
    checkpointed = []
    with patch.object(
        prefill, "checkpoint_kv_cache", lambda r, c: checkpointed.append(r)
    ):
        for stage_parts, early_close in ((False, False), (True, False), (False, True)):
            pipeline.stage_parts, pipeline.early_close = stage_parts, early_close
            for req in (final, middle):
                prefill.SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(
                    scheduler, req
                )
    assert checkpointed == [final, middle, middle, middle]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
