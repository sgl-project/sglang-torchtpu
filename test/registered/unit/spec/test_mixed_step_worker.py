"""TpuSpecWorker mixed prefill + verify DP steps, run as a verify half then a
prompt half: routing per rank kind, collective order and result assembly."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.mixed_verify import MixedVerifyPlan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.speculative import mixed_step
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PLAN = MixedVerifyPlan(512, 3, 6, (512, 0, 0), (3, 2, 0))


class Batch(SimpleNamespace):
    def __init__(self, mode, name, **kw):
        fields = dict(
            forward_mode=mode,
            name=name,
            mixed_verify=PLAN,
            mixed_verify_batch=None,
            has_grammar=False,
            global_num_tokens=[530, 12, 0],
            global_num_tokens_for_logprob=[530, 12, 0],
            is_extend_in_batch=True,
        )
        super().__init__(**{**fields, **kw})


def _result(tag):
    return SimpleNamespace(
        tag=tag, new_seq_lens=f"{tag}-lens", mixed_verify_result=None
    )


class FakeWorker:
    def __init__(self):
        self.calls = []
        self._mixed_steps = 0

    def _check_supported(self, *, batch, pp_proxy_tensors):
        pass

    def _forward_decode(self, batch, *, grammar_barrier=None):
        self.calls.append(("decode", batch.name, list(batch.global_num_tokens)))
        assert batch.is_extend_in_batch is False
        return _result("verify")

    def _forward_idle_decode(self, batch):
        self.calls.append(("idle_decode", batch.name, list(batch.global_num_tokens)))

    def extend(self, batch, *, pp_proxy_tensors):
        self.calls.append(("extend", batch.name, list(batch.global_num_tokens)))
        return _result("prompt") if batch.name == "prompt" else _result("idle-extend")

    def _idle_result(self):
        return _result("idle")


def _idle_like(batch, **overrides):
    idle = Batch(ForwardMode.IDLE, f"idle-like-{batch.name}")
    idle.global_num_tokens = list(batch.global_num_tokens)
    for k, v in overrides.items():
        setattr(idle, k, v)
    return idle


def _run(batch):
    worker = FakeWorker()
    published = []
    with patch.object(mixed_step, "idle_like", _idle_like):
        result = TpuSpecWorker._forward_mixed(
            worker,
            batch,
            grammar_barrier=None,
            on_publish_verify=published.append,
        )
    return worker.calls, result, published


VERIFY_COUNTS = [3, 2, 0]
SYNC_COUNTS = [530, 12, 0]


def test_prompt_and_verify_rank():
    prompt = Batch(ForwardMode.EXTEND, "prompt")
    verify = Batch(ForwardMode.DECODE, "verify", global_num_tokens=None)
    prompt.mixed_verify_batch = verify
    calls, result, published = _run(prompt)
    assert calls == [
        ("decode", "verify", VERIFY_COUNTS),
        ("extend", "prompt", SYNC_COUNTS),
    ]
    assert result.tag == "prompt" and result.mixed_verify_result.tag == "verify"
    assert published == ["verify-lens"]
    # The verify part's own fields are restored after its half.
    assert verify.global_num_tokens is None and verify.is_extend_in_batch is True


def test_verify_only_rank_enters_an_idle_extend():
    verify = Batch(ForwardMode.DECODE, "verify")
    calls, result, published = _run(verify)
    assert calls == [
        ("decode", "verify", VERIFY_COUNTS),
        ("extend", "idle-like-verify", SYNC_COUNTS),
    ]
    assert result.tag == "verify" and published == []
    assert verify.global_num_tokens == SYNC_COUNTS


def test_prompt_only_rank_enters_an_idle_verify():
    prompt = Batch(ForwardMode.EXTEND, "prompt")
    calls, result, _ = _run(prompt)
    assert calls == [
        ("idle_decode", "idle-like-prompt", VERIFY_COUNTS),
        ("extend", "prompt", SYNC_COUNTS),
    ]
    assert result.tag == "prompt" and result.mixed_verify_result is None


def test_idle_rank_enters_both_halves():
    idle = Batch(ForwardMode.IDLE, "idle")
    calls, result, _ = _run(idle)
    assert calls == [
        ("idle_decode", "idle-like-idle", VERIFY_COUNTS),
        ("extend", "idle", SYNC_COUNTS),
    ]
    assert result.tag == "idle"


def test_verify_sync_fields_give_the_plan_bucket():
    fields = mixed_step.verify_sync_fields(PLAN)
    assert fields["global_num_tokens"] == VERIFY_COUNTS
    assert fields["is_extend_in_batch"] is False


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE])
def test_dispatch_routes_planned_batches_first(mode):
    batch = Batch(mode, "b")
    worker = SimpleNamespace(
        _check_supported=lambda **kw: None,
        _forward_mixed=lambda b, **kw: SimpleNamespace(
            next_draft_input=None, new_seq_lens="x", kw=kw
        ),
    )
    with patch(
        "sglang_tpu.experimental.speculative.base_spec_worker.HostPaddedDraftInput",
        type(None),
    ):
        batch.reqs = []
        result = TpuSpecWorker.forward_batch_generation(
            worker, batch, on_publish_verify="cb"
        )
    assert result.kw["on_publish_verify"] == "cb"


def test_idle_like_carries_the_dp_sync_fields():
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    made = SimpleNamespace(
        forward_mode=None, prepare_for_idle=lambda: setattr(made, "idle", True)
    )
    src = Batch(
        ForwardMode.DECODE,
        "src",
        req_to_token_pool=1,
        token_to_kv_pool_allocator=2,
        tree_cache=3,
        model_config=4,
        enable_overlap=True,
        spec_algorithm=5,
        tbo_split_seq_index=None,
        global_decoder_rows=[0, 128, 0],
    )
    with patch.object(ScheduleBatch, "init_new", return_value=made) as init:
        idle = mixed_step.idle_like(src, global_num_tokens=[3, 2, 0])
    init.assert_called_once_with([], 1, 2, 3, 4, True, 5)
    assert idle.idle and idle.is_extend_in_batch is True
    assert idle.global_num_tokens == [3, 2, 0]
    assert idle.global_num_tokens_for_logprob == SYNC_COUNTS
    # The extend's decoder-row bucket comes from this column on every rank
    # (a missing column made an idle stand-in pick the token bucket).
    assert idle.global_decoder_rows == [0, 128, 0]


def test_row_cap_slack_and_fine_bucket():
    t = SimpleNamespace(
        max_val=2048,
        next_bucket=lambda n: next(b for b in (512, 1024, 1152, 1280, 2048) if b >= n),
    )
    slack = mixed_step.row_cap_from_bucketer(t, fine=False)
    fine = mixed_step.row_cap_from_bucketer(t, fine=True)
    assert slack(896) == 1024 and slack(300) == 512
    assert fine(896) == 1152  # the next bucket is 128 rows wider
    assert fine(1100) == 1280  # 1152 -> 1280 is 128 wider too
    coarse = SimpleNamespace(
        max_val=2048, next_bucket=lambda n: next(b for b in (512, 1024, 2048) if b >= n)
    )
    assert mixed_step.row_cap_from_bucketer(coarse, fine=True)(896) == 1024
