"""Slot bookkeeping of the sparse MLA DCP prefill scratch.

A wrong slot serves a request another request's history, so every transition
that must fall back to the gather is pinned here.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dcp_prefill_scratch import plan_scratch_slots

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _plan(state, rows):
    """rows: (rid, prefix_len, seq_len) per prefill row."""
    return plan_scratch_slots(
        state,
        rids=[r[0] for r in rows],
        prefix_lens=[r[1] for r in rows],
        seq_lens=[r[2] for r in rows],
    )


def test_a_request_keeps_its_slot_across_chunks():
    state = [None, None]
    assert _plan(state, [("a", 0, 4096)]) == [0]
    assert _plan(state, [("a", 4096, 8192)]) == [0]
    # A chunk may restart below the stored length (it rewrites those rows).
    assert _plan(state, [("a", 8000, 12288)]) == [0]
    assert state[0] == ("a", 12288)


def test_a_prefix_beyond_the_stored_tokens_gathers_from_then_on():
    """A radix prefix hit: the slot never held those tokens."""
    state = [None]
    assert _plan(state, [("a", 2048, 6144)]) is None
    assert _plan(state, [("a", 6144, 10240)]) is None
    state = [("a", 4096)]
    assert _plan(state, [("a", 8192, 12288)]) is None
    assert state == [None]


def test_a_slot_is_reused_only_once_its_request_left_the_batch():
    state = [None, None]
    assert _plan(state, [("a", 0, 4096), ("b", 0, 4096)]) == [0, 1]
    # c starts while a and b still prefill: no slot, the whole batch gathers
    # and a and b lose theirs.
    assert _plan(state, [("a", 4096, 8192), ("b", 4096, 8192), ("c", 0, 100)]) is None
    assert state == [None, None]
    state = [("a", 4096), ("b", 4096)]
    assert _plan(state, [("b", 4096, 8192), ("c", 0, 4096)]) == [1, 0]
    assert state == [("c", 4096), ("b", 8192)]


def test_a_stolen_slot_sends_its_old_request_to_the_gather():
    state = [("a", 4096)]
    assert _plan(state, [("b", 0, 4096)]) == [0]
    assert _plan(state, [("a", 4096, 8192)]) is None
