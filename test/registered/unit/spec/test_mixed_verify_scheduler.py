"""Scheduler side of mixed prefill + verify DP steps: step formation, budget,
MLP-sync counts, relay of the verify part and result routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.dp_attn import _mixed_verify_num_tokens
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import mixed_verify as mv
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS = 6


class FakeBatch:
    def __init__(self, mode, bs=0, extend_tokens=0, prefill_only=False):
        self.forward_mode = mode
        self._bs = bs
        self.extend_num_tokens = extend_tokens
        self.is_prefill_only = prefill_only
        self.mixed_verify = None
        self.mixed_verify_batch = None
        self.global_num_tokens = [99]
        self.global_num_tokens_for_logprob = [99]
        self.req_pool_indices = object()
        self.encoder_swa_reset = None

    def is_empty(self):
        return self._bs == 0

    def batch_size(self):
        return self._bs


def _sched(retract_to=None, fused=True, row_cap=None, updated=None, skip=False):
    def update_running_batch(batch):
        if updated is not None:
            updated.append(batch)
        if skip:
            return None  # KV full while finishing requests free theirs
        if retract_to is not None:
            batch._bs = retract_to
        return batch

    idle = FakeBatch(ForwardMode.IDLE)

    def sync(ret, need_sync):
        assert need_sync
        return ret if ret is not None else idle

    return SimpleNamespace(
        update_running_batch=update_running_batch,
        mixed_verify_fused=fused,
        _mixed_verify_row_cap=row_cap,
        num_verify_rows=ROWS,
        dp_attn_adapter=SimpleNamespace(
            maybe_prepare_mlp_sync_batch=sync,
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(folds_encoder_swa_replay=False)
            ),
        ),
    )


def _step(gathered, new_batch, running, retract_to=None, **kw):
    with patch.object(sched_mod, "gather_mixed_counts", return_value=gathered):
        return Scheduler._mixed_verify_next_batch(
            _sched(retract_to, **kw), new_batch, running
        )


def test_decode_step_when_no_rank_prefills():
    running = FakeBatch(ForwardMode.DECODE, bs=3)
    ret, run = _step(([0, 0], [3, 2]), None, running)
    assert ret is running and run is running and ret.mixed_verify is None


def test_extend_step_when_no_rank_verifies():
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=1024)
    empty = FakeBatch(ForwardMode.DECODE, bs=0)
    ret, run = _step(([1024, 0], [0, 0]), new, empty)
    assert ret is new and ret.mixed_verify is None and run is empty


def test_mixed_prompt_and_verify_container():
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=1000)
    running = FakeBatch(ForwardMode.DECODE, bs=4)
    ret, run = _step(([1000, 0], [4, 2]), new, running)
    assert ret is new and run is running
    plan = ret.mixed_verify
    assert plan == mv.MixedVerifyPlan(1000, 4, ROWS, (1000, 0), (4, 2))
    assert ret.mixed_verify_batch is running and running.mixed_verify is plan
    # The verify part skips the MLP sync; stale counts must not be read.
    assert running.global_num_tokens is None
    assert mv.mixed_parts(ret) == (new, running)


def test_mixed_verify_only_rank_runs_its_decode_batch():
    running = FakeBatch(ForwardMode.DECODE, bs=2)
    ret, _ = _step(([1024, 0], [0, 2]), None, running)
    assert ret is running and ret.mixed_verify.n_verify == 2
    assert ret.mixed_verify.n_prompt_rows == 0
    assert mv.mixed_parts(ret) == (None, running)


def test_mixed_prompt_only_rank():
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=512)
    empty = FakeBatch(ForwardMode.DECODE, bs=0)
    ret, _ = _step(([512, 0], [0, 3]), new, empty)
    assert ret is new and ret.mixed_verify_batch is None
    assert ret.mixed_verify.n_verify == 0
    assert mv.mixed_parts(ret) == (new, None)


def test_mixed_idle_rank_gets_the_plan():
    empty = FakeBatch(ForwardMode.DECODE, bs=0)
    ret, _ = _step(([1024, 0], [3, 0]), None, empty)
    assert ret.forward_mode.is_idle() and ret.mixed_verify is not None
    assert mv.mixed_parts(ret) == (None, None)


def test_gather_bounds_verify_by_running_count_then_retracts():
    running = FakeBatch(ForwardMode.DECODE, bs=5)
    with patch.object(
        sched_mod, "gather_mixed_counts", return_value=([0], [5])
    ) as gather:
        ret, _ = Scheduler._mixed_verify_next_batch(_sched(retract_to=2), None, running)
    # The bound goes out before decode prep; what runs is the retracted batch.
    gather.assert_called_once_with(0, 5)
    assert ret is running and ret.batch_size() == 2


def test_fused_step_without_slack_runs_the_prompt_alone():
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=1024)
    running = FakeBatch(ForwardMode.DECODE, bs=4)
    updated = []
    # 1024 + 4 x 6 rows exceed the 1024 bucket: no mixing, decode waits.
    ret, run = _step(
        ([1024, 0], [4, 2]), new, running, row_cap=lambda p: 1024, updated=updated
    )
    assert ret is new and ret.mixed_verify is None and run is running
    assert updated == []  # the running batch is not prepared for decode


def test_fused_step_mixes_into_slack():
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=768)
    running = FakeBatch(ForwardMode.DECODE, bs=4)
    ret, _ = _step(([768, 0], [4, 2]), new, running, row_cap=lambda p: 1024)
    assert ret is new and ret.mixed_verify is not None
    assert ret.mixed_verify_batch is running and ret.mixed_verify.n_verify == 4


def test_prefill_only_running_batch_does_not_verify():
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=256)
    running = FakeBatch(ForwardMode.EXTEND, bs=3, prefill_only=True)
    with patch.object(
        sched_mod, "gather_mixed_counts", return_value=([256], [0])
    ) as gather:
        Scheduler._mixed_verify_next_batch(_sched(), new, running)
    gather.assert_called_once_with(256, 0)


@pytest.mark.parametrize(
    "mixed_chunk,stage_b,fused,expect",
    [
        (False, False, False, 0),
        (True, False, False, 4),
        # Fused verify rows use only the prompt step's slack.
        (True, True, True, 0),
        # Verify forward then extend forward: the prompt keeps the whole chunk.
        (True, True, False, 0),
    ],
)
def test_mixed_decode_tokens_budget(mixed_chunk, stage_b, fused, expect):
    self = SimpleNamespace(
        is_mixed_chunk=mixed_chunk,
        num_verify_rows=ROWS,
        enable_mixed_verify=stage_b,
        mixed_verify_fused=fused,
    )
    assert Scheduler._mixed_decode_tokens(self, 4) == expect


def _plan(n_verify, prompt=0, fused=True):
    return mv.MixedVerifyPlan(
        prompt, n_verify, ROWS, (prompt,), (n_verify,), fused=fused
    )


def test_mlp_sync_tokens():
    ext = FakeBatch(ForwardMode.EXTEND, bs=2, extend_tokens=1000)
    assert _mixed_verify_num_tokens(ext, 1000) == 1000
    ext.mixed_verify = _plan(4, 1000)
    assert _mixed_verify_num_tokens(ext, 1000 + 128) == 1000 + 128 + 24
    dec = FakeBatch(ForwardMode.DECODE, bs=3)
    dec.mixed_verify = _plan(3)
    assert _mixed_verify_num_tokens(dec, 3) == 18
    idle = FakeBatch(ForwardMode.IDLE)
    idle.mixed_verify = _plan(0)
    assert _mixed_verify_num_tokens(idle, 0) == 0
    assert _mixed_verify_num_tokens(None, 0) == 0


def test_mlp_sync_tokens_unfused():
    # The count sizes the extend half: prompt tokens only, 0 on a verify-only rank.
    ext = FakeBatch(ForwardMode.EXTEND, bs=2, extend_tokens=1000)
    ext.mixed_verify = _plan(4, 1000, fused=False)
    assert _mixed_verify_num_tokens(ext, 1000 + 128) == 1000 + 128
    dec = FakeBatch(ForwardMode.DECODE, bs=3)
    dec.mixed_verify = _plan(3, fused=False)
    assert _mixed_verify_num_tokens(dec, 3) == 0


def test_relay_and_copy_of_the_verify_part():
    verify = FakeBatch(ForwardMode.DECODE, bs=2)
    verify.input_ids = object()
    draft = SimpleNamespace(dsa_topk_indices=None)
    result = SimpleNamespace(next_draft_input=draft, copy_done=None)
    self = SimpleNamespace(
        device_module=SimpleNamespace(Event=lambda: "event"),
        _relay_forward_payload=MagicMock(),
        _copy_batch_result_to_cpu=MagicMock(),
    )
    Scheduler._relay_and_copy_mixed_verify(self, verify, result)
    self._relay_forward_payload.assert_called_once_with(
        verify, verify.req_pool_indices, result
    )
    self._copy_batch_result_to_cpu.assert_called_once_with(verify, result)
    assert result.copy_done == "event" and verify.input_ids is None
    assert verify.spec_info is draft and draft.future_indices is verify.req_pool_indices
    with pytest.raises(AssertionError):
        Scheduler._relay_and_copy_mixed_verify(self, verify, None)


def test_fused_mixed_steps_do_not_arm_the_prefill_decode_interval():
    def sched():
        return SimpleNamespace(
            prefill_decode_interval=4,
            require_mlp_sync=True,
            _prefill_decode_interval_remaining=0,
        )

    ext = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=512)
    ext.is_extend_in_batch = True
    s = sched()
    Scheduler._arm_prefill_decode_interval(s, ext)
    assert s._prefill_decode_interval_remaining == 4  # a plain extend arms it
    ext.mixed_verify = _plan(2, 512, fused=True)
    s = sched()
    Scheduler._arm_prefill_decode_interval(s, ext)
    assert s._prefill_decode_interval_remaining == 0  # a fused mixed step does not
    ext.mixed_verify = _plan(2, 512, fused=False)
    s = sched()
    Scheduler._arm_prefill_decode_interval(s, ext)
    assert s._prefill_decode_interval_remaining == 4  # Stage A halves still arm it


def test_update_running_batch_none_skips_this_steps_verify():
    """update_running_batch returns None (KV full, finishing requests free theirs
    at the end of the step): this rank verifies nothing this step, its running
    batch stays the scheduler's running batch with its requests, and it still
    enters the step (DECODE: idle; MIXED: prompt part only or idle)."""
    # DECODE step.
    running = FakeBatch(ForwardMode.DECODE, bs=3)
    ret, run = _step(([0, 0], [3, 2]), None, running, skip=True)
    assert run is running and run.batch_size() == 3
    assert ret.forward_mode.is_idle()
    # MIXED step, a rank with a prompt.
    new = FakeBatch(ForwardMode.EXTEND, bs=1, extend_tokens=512)
    running = FakeBatch(ForwardMode.DECODE, bs=4)
    ret, run = _step(([512, 0], [4, 2]), new, running, skip=True, fused=False)
    assert run is running and ret is new and ret.mixed_verify_batch is None
    assert ret.mixed_verify.n_verify == 0 and running.mixed_verify is None
    # MIXED step, a verify-only rank.
    running = FakeBatch(ForwardMode.DECODE, bs=2)
    ret, run = _step(([1024, 0], [0, 2]), None, running, skip=True, fused=False)
    assert run is running and ret.forward_mode.is_idle()
    assert ret.mixed_verify is not None and ret.mixed_verify.n_verify == 0
