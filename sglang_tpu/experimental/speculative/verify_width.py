"""Verify width per decode batch bucket: a static table (SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS)
or a modeled choice among compiled candidates (SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES).

Either way a host-side choice: no device reads, and every reachable (bucket, width)
pair is warmed at startup, so switching never compiles.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence

import msgspec

from sglang_tpu.envs import envs as tpu_envs


class VerifyWidthTable(msgspec.Struct, frozen=True):
    """Ascending (min_bs, width) steps; a bucket takes the last step at or below it."""

    min_bs: tuple[int, ...]
    widths: tuple[int, ...]

    def width_for(self, padded_bs: int) -> int:
        index = bisect.bisect_right(self.min_bs, padded_bs) - 1
        return self.widths[max(index, 0)]

    def reachable_widths(self, buckets: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted({self.width_for(bs) for bs in buckets}))


def parse_verify_width_table(spec: str, max_width: int) -> VerifyWidthTable:
    """Parse "min_bs:width,..." with every width in [2, max_width]."""
    steps: dict[int, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            min_bs_str, width_str = item.split(":")
            min_bs, width = int(min_bs_str), int(width_str)
        except ValueError as e:
            raise ValueError(
                f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS item {item!r} is not min_bs:width."
            ) from e
        if min_bs < 1 or min_bs in steps:
            raise ValueError(
                f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS needs distinct min_bs >= 1, got {spec!r}."
            )
        if not 2 <= width <= max_width:
            raise ValueError(
                f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS width {width} is outside "
                f"[2, gamma + 1 = {max_width}]."
            )
        steps[min_bs] = width
    if not steps:
        raise ValueError("SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS is empty.")
    ordered = sorted(steps.items())
    return VerifyWidthTable(
        min_bs=tuple(bs for bs, _ in ordered),
        widths=tuple(width for _, width in ordered),
    )


def verify_width_table(max_width: int) -> VerifyWidthTable | None:
    """The configured table, or None to verify max_width rows at every bucket."""
    spec = tpu_envs.SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS.get()
    if spec is None:
        return None
    return parse_verify_width_table(spec, max_width)


def verify_width_candidates(max_width: int) -> tuple[int, ...] | None:
    """SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES as ascending widths, or None when unset."""
    spec = tpu_envs.SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES.get()
    if spec is None:
        return None
    try:
        widths = sorted({int(item) for item in spec.split(",") if item.strip()})
    except ValueError as e:
        raise ValueError(
            f"SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES={spec!r} is not a list of widths."
        ) from e
    if not widths or not all(2 <= w <= max_width for w in widths):
        raise ValueError(
            f"SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES={spec!r} needs widths in "
            f"[2, gamma + 1 = {max_width}]."
        )
    return tuple(widths)


class DraftSurvival:
    """Per-position draft survival h_j = P(d_j accepted | d_1..d_{j-1} accepted, d_j verified).

    Kaplan-Meier over every request of every step, whatever width it verified: a
    request that verified k drafts and accepted c of them is at risk at positions
    1..min(c + 1, k) and survives 1..c; c == k is censored at k, not a failure at
    k + 1. Each position keeps about its last `memory` at-risk requests, so a
    deep position, which sees few requests when acceptance is low, does not stay
    pinned by the many it saw when acceptance was high. Counts also decay by
    `decay` per step, and a position with little recent weight (one deeper than
    the widths recently verified) shrinks toward the pooled survival of all
    positions with `prior_weight` pseudo-requests, so stale history fades into
    current traffic.
    """

    def __init__(
        self, num_drafts: int, *, memory: float, decay: float, prior_weight: float
    ) -> None:
        self._memory = memory
        self._decay = decay
        self._prior_weight = prior_weight
        self._at_risk = [0.0] * num_drafts
        self._survived = [0.0] * num_drafts

    def record(
        self, num_correct_drafts: Sequence[int], num_verify_drafts: Sequence[int]
    ) -> None:
        """One step's per-request accepted and verified draft counts."""
        num_drafts = len(self._at_risk)
        # reached[n]: requests at risk at exactly positions 1..n; correct[n]: survived 1..n.
        reached = [0] * (num_drafts + 1)
        correct = [0] * (num_drafts + 1)
        for c, k in zip(num_correct_drafts, num_verify_drafts):
            # Completions pair with steps by batch size, so a stray pairing can
            # report more accepted drafts than the paired step verified.
            c = min(c, k)
            reached[min(c + 1, k)] += 1
            correct[c] += 1
        at_risk = survived = 0
        keep = 1.0 - self._decay
        for j in range(num_drafts, 0, -1):
            at_risk += reached[j]
            survived += correct[j]
            old_at_risk = keep * self._at_risk[j - 1]
            old_survived = keep * self._survived[j - 1]
            if old_at_risk + at_risk > self._memory:
                # Old counts give way so that `memory` requests stay at risk.
                # No history yet (a first batch above memory): nothing to scale.
                scale = (
                    max(0.0, self._memory - at_risk) / old_at_risk
                    if old_at_risk > 0
                    else 0.0
                )
                old_at_risk, old_survived = scale * old_at_risk, scale * old_survived
            self._at_risk[j - 1] = old_at_risk + at_risk
            self._survived[j - 1] = old_survived + survived

    def hazards(self) -> list[float] | None:
        """h_1..h_N, or None before any draft was verified."""
        total_at_risk = sum(self._at_risk)
        if total_at_risk == 0:
            return None
        pooled = sum(self._survived) / total_at_risk
        return [
            (survived + self._prior_weight * pooled) / (at_risk + self._prior_weight)
            if at_risk + self._prior_weight > 0
            else pooled
            for at_risk, survived in zip(self._at_risk, self._survived)
        ]

    def tokens_per_req(self, width: int) -> float | None:
        """Expected committed tokens per request verifying width - 1 drafts each."""
        hazards = self.hazards()
        if hazards is None:
            return None
        tokens, survival = 1.0, 1.0
        for hazard in hazards[: width - 1]:
            survival *= hazard
            tokens += survival
        return tokens


class AdaptiveVerifyWidth:
    """Per batch bucket, the compiled verify width with the best predicted goodput.

    Goodput = predicted tokens per request / step seconds, with the two learned
    separately. Step time per (bucket, width) is a near-constant hardware cost,
    from the interval between consecutive decode completions. Tokens per request
    come from one live DraftSurvival profile, fed by every step whatever its
    width, so a width that has not run lately is judged on current acceptance,
    not on what it accepted when it last ran. With per-request caps (compact /
    cap-accept) the uniform-width prediction is a lower bound: confidence ranking
    spends the budget where acceptance is likely; no per-width correction is
    learned, since it would go stale with traffic exactly as measured tokens did.

    The width changes only at window boundaries, where rank 0's choice is
    broadcast, so every TP rank verifies at the same shape. The first windows of
    a bucket try each candidate in turn to measure its step time; afterwards one
    window in explore_every revisits the others, round robin, only to refresh
    their step times. A bucket leaves its current width only for one at least
    switch_margin better.

    seed() takes a startup profile of each (bucket, width)'s verify forward. A
    width not measured yet is then priced at its profiled time plus a per-bucket
    offset, learned online from the widths that do run, for what the profile
    leaves out (the draft, host work, sampling; about the same at every width).
    A seeded bucket skips the trial windows and explores 4x less often.
    """

    def __init__(
        self,
        candidates: tuple[int, ...],
        *,
        num_drafts: int,
        window_steps: int = 32,
        explore_every: int = 64,
        step_time_smoothing: float = 0.1,
        acceptance_memory: float = 1024.0,
        acceptance_decay: float = 0.03,
        prior_weight: float = 32.0,
        switch_margin: float = 0.03,
    ) -> None:
        # Arbitrary defaults: acceptance remembers ~acceptance_memory requests per
        # position and ~1 / acceptance_decay steps; step times are refreshed once
        # per explore_every windows.
        assert candidates[-1] - 1 <= num_drafts, (candidates, num_drafts)
        self.candidates = candidates
        self._window_steps = window_steps
        self._explore_every = explore_every
        self._step_time_smoothing = step_time_smoothing
        self._switch_margin = switch_margin
        self._survival = DraftSurvival(
            num_drafts,
            memory=acceptance_memory,
            decay=acceptance_decay,
            prior_weight=prior_weight,
        )
        self._settled: dict[int, int] = {}
        self._steps: dict[int, int] = {}
        self._width: dict[int, int] = {}
        self._explored: dict[int, int] = {}
        self._seconds: dict[tuple[int, int], float] = {}
        self._profiled: dict[tuple[int, int], float] = {}
        self._offset: dict[int, float] = {}

    def seed(self, profile: Mapping[tuple[int, int], float]) -> None:
        """Profiled verify-forward seconds per (bucket, width), as priors for step time."""
        self._profiled.update(profile)

    def seeded(self, bucket: int) -> bool:
        return all((bucket, width) in self._profiled for width in self.candidates)

    def width_for_step(self, bucket: int, *, agree: Callable[[int], int]) -> int:
        """The width for this bucket's next step; agree() makes rank 0's choice everyone's."""
        step = self._steps.get(bucket, 0)
        self._steps[bucket] = step + 1
        if step % self._window_steps == 0:
            self._width[bucket] = agree(
                self._propose(bucket, step // self._window_steps)
            )
        return self._width[bucket]

    def record(
        self,
        bucket: int,
        width: int,
        *,
        num_correct_drafts: Sequence[int],
        num_verify_drafts: Sequence[int],
        seconds: float | None,
    ) -> None:
        """One completed step: per-request accepted / verified drafts and, if clean, its duration."""
        self._survival.record(num_correct_drafts, num_verify_drafts)
        if seconds is None:
            return
        key = (bucket, width)
        self._seconds[key] = self._smooth(self._seconds.get(key), seconds)
        profiled = self._profiled.get(key)
        if profiled is not None:
            self._offset[bucket] = self._smooth(
                self._offset.get(bucket), seconds - profiled
            )

    def _smooth(self, old: float | None, sample: float) -> float:
        return (
            sample if old is None else old + self._step_time_smoothing * (sample - old)
        )

    def tokens_per_req(self, width: int) -> float | None:
        return self._survival.tokens_per_req(width)

    def step_seconds(self, bucket: int, width: int) -> float | None:
        """Measured step seconds, else the profiled verify plus the bucket's offset."""
        measured = self._seconds.get((bucket, width))
        if measured is not None:
            return measured
        profiled = self._profiled.get((bucket, width))
        if profiled is None:
            return None
        return profiled + self._offset.get(bucket, 0.0)

    def goodput(self, bucket: int, width: int) -> float | None:
        tokens = self.tokens_per_req(width)
        seconds = self.step_seconds(bucket, width)
        if tokens is None or seconds is None:
            return None
        return tokens / seconds

    def best(self, bucket: int) -> int:
        """The predicted best width; the widest candidate until any is scored."""
        scored = [
            (goodput, width)
            for width in self.candidates
            if (goodput := self.goodput(bucket, width)) is not None
        ]
        return max(scored)[1] if scored else self.candidates[-1]

    def summary(self, bucket: int) -> dict[int, tuple[float | None, float | None]]:
        """Each candidate's (predicted tokens per request, step seconds), for logs."""
        return {
            width: (self.tokens_per_req(width), self.step_seconds(bucket, width))
            for width in self.candidates
        }

    def _settle(self, bucket: int) -> int:
        """The best width, kept at the current one unless another clears the margin."""
        best = self.best(bucket)
        current = self._settled.get(bucket)
        if current is not None and current != best:
            current_goodput = self.goodput(bucket, current)
            best_goodput = self.goodput(bucket, best)
            if (
                current_goodput is not None
                and best_goodput is not None
                and best_goodput < current_goodput * (1 + self._switch_margin)
            ):
                best = current
        self._settled[bucket] = best
        return best

    def _propose(self, bucket: int, window: int) -> int:
        seeded = self.seeded(bucket)
        if not seeded and window < len(self.candidates):
            return self.candidates[window]
        best = self._settle(bucket)
        others = [w for w in self.candidates if w != best]
        explore_every = self._explore_every * (4 if seeded else 1)
        if others and window > 0 and window % explore_every == 0:
            explored = self._explored.get(bucket, 0)
            self._explored[bucket] = explored + 1
            return others[explored % len(others)]
        return best
