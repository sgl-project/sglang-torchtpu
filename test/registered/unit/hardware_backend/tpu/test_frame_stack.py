"""Unit tests for the frame-stack reservation used around model tracing."""

import resource
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.frame_stack import (
    call_with_reserved_frame_stack,
)

# Pure Python, no torch_tpu needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CALLS_PER_DEPTH = 2000
# Sweep enough depths to cross several ordinary stack chunks.
_DEPTHS = 300


def _leaf():
    return None


# Make the leaf large enough to cross a boundary during the depth sweep.
_leaf.__code__ = _leaf.__code__.replace(co_stacksize=64)


def _call_leaf_at_depth(depth, calls):
    if depth:
        return _call_leaf_at_depth(depth - 1, calls)
    for _ in range(calls):
        _leaf()


def _sweep_depths():
    for depth in range(_DEPTHS):
        _call_leaf_at_depth(depth, _CALLS_PER_DEPTH)


def _minor_faults(fn):
    # Repeated chunk allocations show up as minor page faults.
    before = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt
    fn()
    return resource.getrusage(resource.RUSAGE_THREAD).ru_minflt - before


@unittest.skipUnless(
    sys.platform == "linux" and sys.version_info[:2] == (3, 12),
    "the reservation is applied on CPython 3.12 only; needs per-thread rusage",
)
class TestCallWithReservedFrameStack(CustomTestCase):
    def test_no_chunk_churn_at_any_depth(self):
        """A callee must not map and unmap a stack chunk on every call."""
        faults = _minor_faults(lambda: call_with_reserved_frame_stack(_sweep_depths))

        self.assertLess(faults, _CALLS_PER_DEPTH)

    def test_nested_reservation_keeps_its_own_room(self):
        """An inner reservation must not use up the outer one's remainder."""

        def nested():
            call_with_reserved_frame_stack(_sweep_depths)

        faults = _minor_faults(lambda: call_with_reserved_frame_stack(nested))

        self.assertLess(faults, _CALLS_PER_DEPTH)


if __name__ == "__main__":
    unittest.main()
