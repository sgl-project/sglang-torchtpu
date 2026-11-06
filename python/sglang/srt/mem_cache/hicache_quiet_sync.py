"""Rank-consistent back-off for the per-step HiCache ack poll.

Every scheduler step, each TP rank MIN-reduces how many of its HiCache write and load
acks are ready, so all ranks reap the same acks in lockstep. When no rank has a ready ack
the reduce returns zeros and the step only paid for the collective.

QuietSyncGate lets a caller skip that poll on some steps without ever letting ranks
disagree on whether to enter the collective: its decision depends only on the previous
reduce's result, which every rank receives identically, and on how many times it was
asked, which is the same on every rank because each rank asks once per scheduler step.
After a poll in which no rank had a ready ack, it skips the next `every - 1` steps and
polls on the `every`-th; any poll that finds a ready ack returns it to polling every step.
A ready ack therefore waits at most `every - 1` steps and is still reaped by all ranks in
the same step.
"""


class QuietSyncGate:
    def __init__(self, every: int):
        # every <= 1 keeps the poll on every step.
        self.every = max(int(every), 1)
        self.quiet = False
        self.skipped = 0

    def should_skip(self) -> bool:
        """Call once per scheduler step, on every rank, before the poll."""
        if self.every <= 1 or not self.quiet:
            return False
        self.skipped += 1
        if self.skipped < self.every:
            return True
        self.skipped = 0
        return False

    def record(self, found_ready: bool) -> None:
        """Call after each poll with whether the reduced counts were nonzero."""
        self.quiet = not found_ready
        self.skipped = 0
