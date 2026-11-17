"""Slot bookkeeping of the sparse MLA DCP prefill scratch.

A wrong slot serves a request another request's history, so every transition
that must fall back to the gather is pinned here.
"""

from dataclasses import dataclass

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dcp_prefill_scratch import plan_scratch_slots

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _plan(state, rows):
    """rows: (ownership token, prefix_len, seq_len) per prefill row."""
    return plan_scratch_slots(
        state,
        owners=[r[0] for r in rows],
        prefix_lens=[r[1] for r in rows],
        seq_lens=[r[2] for r in rows],
    )


def test_a_request_keeps_its_slot_across_chunks():
    a = _Handle("a")
    state = [None, None]
    assert _plan(state, [(a, 0, 4096)]) == [0]
    assert _plan(state, [(a, 4096, 8192)]) == [0]
    # A chunk may restart below the stored length (it rewrites those rows).
    assert _plan(state, [(a, 8000, 12288)]) == [0]
    assert state[0] == (a, 12288)


def test_a_prefix_beyond_the_stored_tokens_gathers_from_then_on():
    """A radix prefix hit: the slot never held those tokens."""
    a = _Handle("a")
    state = [None]
    assert _plan(state, [(a, 2048, 6144)]) is None
    assert _plan(state, [(a, 6144, 10240)]) is None
    state = [(a, 4096)]
    assert _plan(state, [(a, 8192, 12288)]) is None
    assert state == [None]


def test_a_slot_is_reused_only_once_its_request_left_the_batch():
    a, b, c = _Handle("a"), _Handle("b"), _Handle("c")
    state = [None, None]
    assert _plan(state, [(a, 0, 4096), (b, 0, 4096)]) == [0, 1]
    # c starts while a and b still prefill: no slot, the whole batch gathers
    # and a and b lose theirs.
    assert _plan(state, [(a, 4096, 8192), (b, 4096, 8192), (c, 0, 100)]) is None
    assert state == [None, None]
    state = [(a, 4096), (b, 4096)]
    assert _plan(state, [(b, 4096, 8192), (c, 0, 4096)]) == [1, 0]
    assert state == [(c, 4096), (b, 8192)]


def test_a_stolen_slot_sends_its_old_request_to_the_gather():
    a, b = _Handle("a"), _Handle("b")
    state = [(a, 4096)]
    assert _plan(state, [(b, 0, 4096)]) == [0]
    assert _plan(state, [(a, 4096, 8192)]) is None


@dataclass(frozen=True)
class _Handle:
    rid: str
    attempt_id: int = 0


def test_reused_public_id_does_not_reuse_completed_request_scratch():
    old = _Handle("reused")
    new = _Handle("reused")
    assert old == new and old is not new
    state = [None]
    assert _plan(state, [(old, 0, 7222)]) == [0]
    # B hits its own radix prefix after A finished with the same public ID.
    # A's scratch contains different tokens, despite equal-valued handles.
    assert _plan(state, [(new, 7168, 7223)]) is None
    assert _plan(state, [(new, 7223, 10000)]) is None


def test_equal_valued_new_owner_can_start_cold_without_aliasing_old_owner():
    old, new = _Handle("reused"), _Handle("reused")
    state = [(old, 8192)]
    assert _plan(state, [(new, 0, 4096)]) == [0]
    assert state[0][0] is new
    assert _plan(state, [(new, 4096, 8192)]) == [0]
    # A late or retracted old request cannot consume the new owner's slot.
    assert _plan(state, [(old, 8192, 12288)]) is None


def test_new_attempt_and_cancelled_request_cannot_inherit_old_scratch():
    old = _Handle("request", 0)
    state = [(old, 8192)]
    assert _plan(state, [(_Handle("request", 1), 4096, 8192)]) is None
    # Cancellation can happen mid-prefill without a decode/completion callback.
    assert _plan(state, [(_Handle("request", 0), 4096, 8192)]) is None


def test_equal_valued_owners_take_distinct_slots_in_one_batch():
    a, b = _Handle("same"), _Handle("same")
    state = [None, None]
    assert _plan(state, [(a, 0, 4096), (b, 0, 4096)]) == [0, 1]
    assert _plan(state, [(b, 4096, 8192), (a, 4096, 8192)]) == [1, 0]
