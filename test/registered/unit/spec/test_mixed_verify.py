"""Unit tests for the mixed prefill + speculative-verify DP step rules."""

from unittest.mock import patch

import pytest

from sglang.srt.speculative import mixed_verify as mv
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BUCKETS = (8, 16, 32, 64)


def b_bucket(n):
    return next(b for b in BUCKETS if b >= n)


@pytest.mark.parametrize(
    "prompt,verify,kind",
    [
        ([0, 0, 0], [3, 0, 5], mv.StepKind.DECODE),
        ([0, 0, 0], [0, 0, 0], mv.StepKind.DECODE),
        ([1024, 0, 0], [0, 0, 0], mv.StepKind.EXTEND),
        ([1024, 0, 0], [0, 2, 0], mv.StepKind.MIXED),
        ([0, 512, 0], [4, 0, 0], mv.StepKind.MIXED),
    ],
)
def test_global_step_kind(prompt, verify, kind):
    assert mv.global_step_kind(prompt, verify) is kind


def test_every_rank_derives_the_same_kind_and_bucket():
    prompt, verify = [1018, 0, 0, 512, 0, 0, 0, 0], [1, 3, 0, 0, 9, 0, 0, 2]
    seen = {
        (
            mv.global_step_kind(prompt, verify),
            mv.verify_bucket(verify, attn_tp_size=1, b_bucket=b_bucket),
        )
        for _rank in range(8)
    }
    assert seen == {(mv.StepKind.MIXED, 16)}


def test_verify_bucket():
    assert mv.verify_bucket([0, 0], attn_tp_size=1, b_bucket=b_bucket) == 0
    assert mv.verify_bucket([1, 0], attn_tp_size=1, b_bucket=b_bucket) == 8
    assert mv.verify_bucket([9, 2], attn_tp_size=1, b_bucket=b_bucket) == 16
    # Aligned to attention TP first, as the decode bucket is.
    assert mv.verify_bucket([9], attn_tp_size=4, b_bucket=b_bucket) == 16
    assert mv.verify_bucket([17], attn_tp_size=4, b_bucket=b_bucket) == 32


def test_mixed_rows_and_budget():
    assert mv.mixed_rows(1000, 4, 6) == 1024
    # Verify rows take the first share of a rank's budget, so the step stays in
    # the 1024 bucket; never below the floor.
    assert mv.prompt_token_budget(1024, 4, 6) == 1000
    assert mv.mixed_rows(mv.prompt_token_budget(1024, 4, 6), 4, 6) == 1024


def test_plan_rows_and_parts():
    plan = mv.MixedVerifyPlan(
        n_prompt_rows=1000,
        n_verify=4,
        num_verify_rows=6,
        global_prompt_rows=(1000, 0),
        global_verify_bs=(4, 2),
    )
    assert plan.local_rows == 1024

    class Mode:
        def __init__(self, kind):
            self.kind = kind

        def is_extend(self):
            return self.kind == "extend"

        def is_decode(self):
            return self.kind == "decode"

    verify = type("B", (), {"forward_mode": Mode("decode"), "mixed_verify": plan})()
    prompt = type(
        "B",
        (),
        {
            "forward_mode": Mode("extend"),
            "mixed_verify": plan,
            "mixed_verify_batch": verify,
        },
    )()
    idle = type("B", (), {"forward_mode": Mode("idle"), "mixed_verify": plan})()
    plain = type("B", (), {"forward_mode": Mode("extend"), "mixed_verify": None})()
    assert mv.mixed_parts(prompt) == (prompt, verify)
    assert mv.mixed_parts(verify) == (None, verify)
    assert mv.mixed_parts(idle) == (None, None)
    assert mv.mixed_parts(plain) == (None, None)
    assert mv.mixed_parts(None) == (None, None)


def _enabled(**over):
    mv.mixed_verify_enabled.cache_clear()
    cfg = dict(mixed=True, algo="DSPARK", dp=True, platform=True)
    cfg.update(over)
    sched = type("S", (), {"enable_mixed_chunk": cfg["mixed"]})()
    spec = type("P", (), {"speculative_algorithm": cfg["algo"]})()
    par = type("D", (), {"attn_dp_enabled": cfg["dp"]})()
    plat = type(
        "X", (), {"supports_spec_mixed_verify_dp": lambda self, a: cfg["platform"]}
    )()
    with (
        patch("sglang.srt.runtime_context.get_schedule", return_value=sched),
        patch("sglang.srt.runtime_context.get_spec", return_value=spec),
        patch("sglang.srt.runtime_context.get_parallel", return_value=par),
        patch("sglang.srt.platforms.current_platform", plat),
    ):
        out = mv.mixed_verify_enabled()
    mv.mixed_verify_enabled.cache_clear()
    return out


def test_gate_needs_every_condition():
    assert _enabled()
    assert not _enabled(mixed=False)
    assert not _enabled(algo=None)
    assert not _enabled(dp=False)
    assert not _enabled(platform=False)


def test_startup_limits():
    mv.check_mixed_verify_limits(chunk_budget=1024, overlap=True)
    for bad, match in (
        (dict(overlap=False), "overlap"),
        (dict(chunk_budget=None), "chunked prefill"),
    ):
        with pytest.raises(ValueError, match=match):
            mv.check_mixed_verify_limits(
                **{"chunk_budget": 1024, "overlap": True, **bad}
            )


def test_mixed_step_fits():
    cap = lambda p: 1024 if p <= 1024 else 2048  # noqa: E731
    assert mv.mixed_step_fits([768, 0], [4, 2], 6, cap)  # slack behind replay
    assert mv.mixed_step_fits([0, 512], [16, 0], 6, cap)  # verify-only rank: 96 rows
    assert not mv.mixed_step_fits([1024, 0], [4, 0], 6, cap)  # full chunk: no slack
    assert not mv.mixed_step_fits([1000, 0], [0, 200], 6, cap)  # 1200 verify rows
    assert mv.mixed_step_fits([1024, 0], [4, 0], 6, None)  # no cap
