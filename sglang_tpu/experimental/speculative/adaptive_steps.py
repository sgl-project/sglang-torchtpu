"""Draft length chosen per decode step from the number of running requests.

SGLANG_TPU_SPEC_STEPS_BY_RUNNING="1:7,5:3" drafts 7 steps while 1-4 requests
run and 3 from 5 up. The launch's speculative_num_steps is the largest count:
it sizes the verify window, KV reservation and linear-attention checkpoints.
Every (batch bucket, width) pair a live size reaches is warmed; a bucket whose
live sizes straddle a tier boundary gets both widths.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import msgspec

from sglang_tpu.envs import envs as tpu_envs


class DraftStepSchedule(msgspec.Struct, frozen=True):
    """Draft steps by running requests: tier i covers running >= min_running[i]."""

    min_running: tuple[int, ...]
    steps: tuple[int, ...]

    def steps_for(self, running: int) -> int:
        return self.steps[bisect.bisect_right(self.min_running, running) - 1]

    @property
    def max_steps(self) -> int:
        return max(self.steps)

    def widths(self) -> tuple[int, ...]:
        """Every verify width (steps + 1) the schedule reaches, ascending."""
        return tuple(sorted({steps + 1 for steps in self.steps}))

    def widths_for_bucket(
        self, padded_bs: int, buckets: Sequence[int]
    ) -> tuple[int, ...]:
        """Verify widths of the live sizes that pad to padded_bs."""
        ordered = sorted(buckets)
        index = ordered.index(padded_bs)
        low = ordered[index - 1] + 1 if index else 1
        return tuple(
            sorted({self.steps_for(bs) + 1 for bs in range(low, padded_bs + 1)})
        )


def parse_steps_by_running(text: str | None) -> DraftStepSchedule | None:
    """ "R:K,..." -> schedule; unset or empty -> None."""
    text = (text or "").strip()
    if not text:
        return None
    pairs = sorted(_int_pairs(text, "SGLANG_TPU_SPEC_STEPS_BY_RUNNING"))
    min_running = tuple(r for r, _ in pairs)
    steps = tuple(k for _, k in pairs)
    if min_running[0] != 1 or len(set(min_running)) != len(min_running):
        raise ValueError(
            "SGLANG_TPU_SPEC_STEPS_BY_RUNNING needs distinct running counts "
            f"starting at 1, got {text!r}."
        )
    if min(steps) < 1:
        raise ValueError(f"Draft step counts must be >= 1, got {text!r}.")
    return DraftStepSchedule(min_running=min_running, steps=steps)


def parse_acc_len_by_steps(text: str | None) -> dict[int, float]:
    """ "K:AL,..." -> {K: AL}; unset or empty -> {}."""
    out: dict[int, float] = {}
    for item in filter(None, (x.strip() for x in (text or "").split(","))):
        steps, _, acc_len = item.partition(":")
        if not acc_len:
            raise ValueError(f"Expected K:AL, got {item!r}.")
        out[int(steps)] = float(acc_len)
    return out


def draft_step_schedule() -> DraftStepSchedule | None:
    """The configured schedule, or None for a fixed draft length."""
    return parse_steps_by_running(tpu_envs.SGLANG_TPU_SPEC_STEPS_BY_RUNNING.get())


def adaptive_verify_widths() -> tuple[int, ...]:
    """Verify widths the schedule reaches; () without one."""
    schedule = draft_step_schedule()
    return () if schedule is None else schedule.widths()


def check_draft_step_schedule(
    schedule: DraftStepSchedule,
    *,
    launch_steps: int,
    simulate_acc_len: float,
    dp_attention: bool,
) -> dict[int, float]:
    """Raise on a schedule the worker cannot serve; return the simulated AL per step count."""
    if schedule.max_steps != launch_steps:
        raise ValueError(
            f"SGLANG_TPU_SPEC_STEPS_BY_RUNNING reaches {schedule.max_steps} draft "
            f"steps; launch with --speculative-num-steps {schedule.max_steps}."
        )
    if dp_attention:
        # DP ranks run different live sizes but must run the same programs.
        raise NotImplementedError(
            "SGLANG_TPU_SPEC_STEPS_BY_RUNNING does not support DP attention."
        )
    if simulate_acc_len <= 0:
        return {}
    acc_lens = parse_acc_len_by_steps(
        tpu_envs.SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS.get()
    )
    missing = sorted(set(schedule.steps) - set(acc_lens))
    if missing:
        raise ValueError(
            "SGLANG_SIMULATE_ACC_LEN with a draft-step schedule needs "
            f"SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS for steps {missing}."
        )
    for steps, acc_len in acc_lens.items():
        if not 1.0 <= acc_len <= steps + 1:
            raise ValueError(f"Acceptance length {acc_len} at {steps} steps.")
    if acc_lens[launch_steps] != simulate_acc_len:
        raise ValueError(
            f"SGLANG_SIMULATE_ACC_LEN={simulate_acc_len} differs from the "
            f"{launch_steps}-step entry {acc_lens[launch_steps]}."
        )
    return {steps: acc_lens[steps] for steps in schedule.steps}


def _int_pairs(text: str, name: str) -> list[tuple[int, int]]:
    pairs = []
    for item in filter(None, (x.strip() for x in text.split(","))):
        left, _, right = item.partition(":")
        if not right:
            raise ValueError(f"{name}: expected R:K, got {item!r}.")
        pairs.append((int(left), int(right)))
    if not pairs:
        raise ValueError(f"{name}: no entries in {text!r}.")
    return pairs
