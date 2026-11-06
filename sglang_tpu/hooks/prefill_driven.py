# SPDX-License-Identifier: Apache-2.0
"""Prefill-driven decode KV reservation (SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN).

Why: decodes reserve KV in arrival order, so a long prompt queued on the
     prefill holds a decode's KV while shorter prompts wait behind it. Here the
     prefill picks which requests the decodes reserve for, in its own schedule
     order, on top of --disaggregation-decode-allocation-policy early. A decode
     that cannot admit a request now says so, and the prefill asks for another
     one meanwhile.
How: expose all unstarted requests to the real scheduler. Its default-true
     PrefillAdder gate defers unreserved candidates before KV commitment and
     records feasible candidates. After the pass, a count-only slot controller
     agrees on reservations across TP/CP and Raiden sends RESERVE/BUSY/PULL_REQ.
     Canonical bootstrap polling and root policy order keep rank order aligned.
     Mamba matching allocates one candidate at a time during controlled passes.
Upstream PR or issue: none.
Remove when: core PD queues expose a reservation seam.
"""

from __future__ import annotations

import weakref
from contextvars import ContextVar
from http import HTTPStatus
from types import SimpleNamespace
from typing import TYPE_CHECKING

from sglang_tpu._logging import get_logger
from sglang_tpu.disaggregation.prefill_driven import is_enabled
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodePreallocQueue
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sglang.srt.managers.schedule_batch import Req

logger = get_logger(__name__)

# Plugin-side state, keyed weakly so it dies with its queue or request.
_controllers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_active_pass: ContextVar = ContextVar("prefill_reservation_pass", default=None)
# Preserve aging from initial intake into the bootstrap queue.
_enqueued_at: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


# ----- prefill ---------------------------------------------------------------


def attach_batch_reservations(result, queue: PrefillBootstrapQueue, *args, **kwargs):
    from sglang_tpu.disaggregation.reservation_slots import (
        SlotReservations,
        effective_slot_limit,
    )

    scheduler = queue.scheduler
    capacity = min(scheduler.max_running_requests, scheduler.req_to_token_pool.size)
    limit = effective_slot_limit(
        tpu_envs.SGLANG_DISAGG_PREFILL_DRIVEN_SLOTS.get(), capacity
    )
    logger.info(
        "prefill_driven reservation slots: limit=%d capacity=%d", limit, capacity
    )
    _controllers[queue] = SlotReservations(
        limit=limit,
        rank_groups=[scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group],
        log_stats=queue.tp_rank == 0,
    )
    return result


def stamp_enqueue(result, queue: PrefillBootstrapQueue, req: Req, *args, **kwargs):
    # add() can reject/retire an old request (for example, late grammar
    # readiness after abort). Only a successful synchronous append owns state.
    if not queue.queue or queue.queue[-1] is not req:
        return result
    req.arrival_processed_tokens = _enqueued_at[req] = (
        queue.scheduler.processed_tokens_counter
    )
    _controllers[queue].track(req)
    return result


def pop_bootstrapped_with_reservations(
    original, queue: PrefillBootstrapQueue, *args, **kwargs
):
    from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender

    result = original(queue, *args, **kwargs)
    bootstrapped = result[0] if isinstance(result, tuple) else result
    # Intake only: pending requests enter the ordinary policy queue with no
    # metadata or decode allocation. Core resolves all-reduced READY there.
    pending = [r for r in queue.queue if isinstance(r.disagg_kv_sender, RaidenKVSender)]
    queue.queue = [
        r for r in queue.queue if not isinstance(r.disagg_kv_sender, RaidenKVSender)
    ]
    for req in pending:
        req.time_stats.set_wait_queue_entry_time()
    bootstrapped.extend(pending)
    for req in bootstrapped:
        req.arrival_processed_tokens = _enqueued_at.pop(
            req, req.arrival_processed_tokens
        )
    return result


def canonical_bootstrap_poll(original, scheduler, *args, **kwargs):
    queue = scheduler.disagg_prefill_bootstrap_queue
    if queue in _controllers:
        scheduler.waiting_queue[:] = _controllers[queue].canonical_order(
            scheduler.waiting_queue
        )
    return original(scheduler, *args, **kwargs)


def observe_transport_ready(original, queue, req):
    # Core calls finalize only after an all-reduced WaitingForInput poll.
    # Record ownership even when metadata exhaustion makes finalize fail.
    if queue in _controllers:
        _controllers[queue].mark_ready(req)
    return original(queue, req)


def reservation_pass(original, scheduler, prefill_delayer_single_pass, running_batch):
    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    if queue not in _controllers:
        return original(scheduler, prefill_delayer_single_pass, running_batch)
    controller = _controllers[queue]
    if not scheduler.waiting_queue and controller.is_idle():
        return original(scheduler, prefill_delayer_single_pass, running_batch)
    slots = min(
        scheduler.get_num_allocatable_reqs(
            len(running_batch.reqs), running_batch=running_batch
        ),
        scheduler.req_to_token_pool.available_size(),
    )
    from sglang.srt.runtime_context import get_schedule

    for limit in (scheduler.max_prefill_bs, get_schedule().prefill_max_requests):
        if limit:
            slots = min(slots, limit)
    controller.begin(scheduler.waiting_queue, capacity=slots)
    token = _active_pass.set(SimpleNamespace(controller=controller))
    try:
        result = original(scheduler, prefill_delayer_single_pass, running_batch)
    finally:
        _active_pass.reset(token)
    controller.finish(scheduler.waiting_queue)
    return result


def agree_policy_order(result, policy, waiting_queue, *args, **kwargs):
    context = _active_pass.get()
    if context is not None:
        waiting_queue[:] = context.controller.agree_order(waiting_queue)
    return result


def gate_admission(original, adder, req, extend_len):
    from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender

    context = _active_pass.get()
    if context is None or not isinstance(req.disagg_kv_sender, RaidenKVSender):
        return original(adder, req, extend_len)
    # The ignore-EOS fast path checks tiles after the generic readiness seam.
    # Check here before proposing a reservation, without consuming any budget.
    if req.sampling_params.ignore_eos and adder.tree_cache.disable:
        if adder._check_prefill_tile_budget(extend_len) is not None:
            return False
    prefix_len = len(req.prefix_indices) + req.host_hit_length
    if not adder._kv_shard_reserve_scratch(prefix_len, extend_len, commit=False):
        return False
    context.controller.consider(req)
    return not req.pending_bootstrap and original(adder, req, extend_len)


def match_group_size(original, adder, num_candidates):
    return 0 if _active_pass.get() is not None else original(adder, num_candidates)


# ----- decode ----------------------------------------------------------------


def answer_busy_for_unadmitted(result, queue: DecodePreallocQueue, *args, **kwargs):
    """A room the prefill asked for that this pass left in the queue could not
    be admitted, or waits behind one that could not. Its decode keeps no
    reservation and tells the prefill, which asks for other requests now and
    this one again later. Every rank runs the same pass on the same queue, so
    they answer alike."""
    from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver

    # Core's current callers scan all queued rooms. Any future filtered scan
    # must keep unexamined reservations held rather than answering BUSY here.
    for decode_req in queue.queue:
        if decode_req.waiting_for_input and isinstance(
            decode_req.kv_receiver, RaidenKVReceiver
        ):
            from sglang.srt.mem_cache.common import release_kv_cache

            req = decode_req.req
            # Decode prefix matching can allocate Mamba COW state before the
            # later KV feasibility check fails. BUSY must return that state.
            req.kv.mamba_cow_src_index = None
            req.kv.mamba_needs_clear = False
            if req.kv.holds_mamba and not req.kv.holds_kv:
                release_kv_cache(req, queue.tree_cache, is_insert=False)
            decode_req.kv_receiver.reject_reservation()
            decode_req.waiting_for_input = False
    return result


def fail_rebootstrap(
    original,
    queue: DecodePreallocQueue,
    req: Req,
    is_retracted: bool = False,
    is_rebootstrap: bool = False,
):
    if not is_rebootstrap:
        return original(
            queue, req, is_retracted=is_retracted, is_rebootstrap=is_rebootstrap
        )
    from sglang.srt.disaggregation.utils import prepare_abort

    # The recompute would wait for a new decode reservation with no
    # lifecycle guard, so a retract-mode pause fails these requests instead.
    message = (
        f"Request {req.rid} was retracted by a pause and cannot be "
        "recomputed under prefill_driven reservation"
    )
    logger.warning(message)
    prepare_abort(req, message, status_code=HTTPStatus.SERVICE_UNAVAILABLE)
    queue.scheduler.output_streamer.stream_output([req], req.return_logprob)


def hook_specs() -> list:
    if not is_enabled():
        return []
    from sglang.srt.plugins.hook_registry import HookType

    prefill = "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue"
    decode = "sglang.srt.disaggregation.decode.DecodePreallocQueue"
    return [
        (f"{prefill}.__init__", attach_batch_reservations, HookType.AFTER),
        (f"{prefill}.add", stamp_enqueue, HookType.AFTER),
        (f"{prefill}.finalize_bootstrap", observe_transport_ready, HookType.AROUND),
        (
            f"{prefill}.pop_bootstrapped",
            pop_bootstrapped_with_reservations,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw",
            reservation_pass,
            HookType.AROUND,
        ),
        (
            "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.resolve_waiting_queue_bootstrap",
            canonical_bootstrap_poll,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.schedule_policy.SchedulePolicy.calc_priority",
            agree_policy_order,
            HookType.AFTER,
        ),
        (
            "sglang.srt.managers.schedule_policy.PrefillAdder.is_admission_ready",
            gate_admission,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.schedule_policy.PrefillAdder.get_match_group_size",
            match_group_size,
            HookType.AROUND,
        ),
        (f"{decode}.pop_preallocated", answer_busy_for_unadmitted, HookType.AFTER),
        (f"{decode}.add", fail_rebootstrap, HookType.AROUND),
    ]
