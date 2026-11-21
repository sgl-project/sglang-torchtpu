# SPDX-License-Identifier: Apache-2.0
"""Install DP routing by pending uncached prompt tokens on the DP controller.

Why: the DataParallelController balances on request counts (or raw prompt
     tokens) and sends affinity-routed requests to their rank however much
     prefill is queued there; under DP attention a rank's prefill stalls every
     rank (sglang_tpu/dp_prompt_backlog.py has the rule and the evidence).
How: pass-through unless SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN > 0.
  * AROUND DataParallelController.__init__ builds the PromptBacklogRouter once
    the schedulers have reported their KV pool, and turns on the controller's
    per-dispatch load refresh;
  * AROUND DPBudget.update_budget feeds the router the same load snapshots;
  * AROUND DataParallelController.dispatching_with_trace scopes one request's
    dispatch (thread-local);
  * AROUND the controller module's sock_send (the registry propagates it to
    every importer; outside a scoped dispatch it is a pass-through) sends the
    request to the router's rank instead of the stock policy's, and moves the
    budget increments the stock policy made;
  * AROUND load_snapshot.zmq_reader_owner gives the controller the ZMQ load
    snapshots when backlog routing needs them under round_robin.
  The controller process loads plugins at start (run_data_parallel_controller_process).
Upstream seam issue: not filed (a DP-dispatch extension point would replace the
     sock_send redirect).
Remove when: the core DP controller exposes a rank-choice hook.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from sglang.srt.plugins.hook_registry import HookType
from sglang_tpu.dp_prompt_backlog import PromptBacklogRouter, backlog_routing_enabled

logger = logging.getLogger(__name__)

_DPC = "sglang.srt.managers.data_parallel_controller"

# Budget increments each stock policy adds to its rank before sending:
# (requests, tokens per prompt token). Routed requests add none.
_COUNTED = {
    "ROUND_ROBIN": (0, 0),
    "TOTAL_REQUESTS": (1, 0),
    "TOTAL_TOKENS": (1, 1),
    "PREFIX_AWARE": (1, 1),
}


def _margin() -> float:
    from sglang_tpu.envs import envs

    return envs.SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN.get()


def _elastic(max_dp: Optional[int], dp: int) -> bool:
    return (max_dp or dp) > dp


def make_router(controller: Any) -> Optional[PromptBacklogRouter]:
    """The router for this controller, or None when backlog routing is off."""
    from sglang.srt.runtime_context import get_disagg, get_observability
    from sglang_tpu.envs import envs

    margin = _margin()
    if margin <= 0:
        return None
    if not backlog_routing_enabled(
        margin,
        controller.load_balance_method.name,
        get_disagg().disaggregation_mode,
        elastic=_elastic(controller.max_dp_size, controller.launch_dp_size),
    ):
        logger.warning(
            "SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN is ignored under PD "
            "disaggregation, follow_bootstrap_room and elastic DP."
        )
        return None
    # Like the prefix-aware index: a rank's radix cache holds at most its KV
    # pool; err on the large side.
    pool = getattr(controller, "max_total_num_tokens", 0) or 0
    router = PromptBacklogRouter(
        num_ranks=controller.launch_dp_size,
        margin=margin,
        override_routed=envs.SGLANG_TPU_DP_PROMPT_BACKLOG_ROUTED.get(),
        continuation_margin=envs.SGLANG_TPU_DP_PROMPT_BACKLOG_CONTINUATION_MARGIN.get(),
        capacity_blocks=max(1 << 10, 4 * pool // 1024),
        log_interval=get_observability().decode_log_interval,
    )
    logger.info(
        "DP prompt backlog routing: margin %.2f x prompt tokens, routed ranks are "
        "%s, continuation margin %.2f",
        margin,
        "affinity hints" if router.override_routed else "hard routes",
        router.continuation_margin,
    )
    return router


def init_controller(original, self, *args, **kwargs):
    """AROUND DataParallelController.__init__."""
    original(self, *args, **kwargs)
    router = make_router(self)
    self._tpu_backlog_router = router
    if router is not None:
        self.refresh_load_budget_on_dispatch = True
        self.dp_budget._tpu_backlog_router = router


def update_budget(original, self, loads):
    """AROUND DPBudget.update_budget: the router reads the same snapshots."""
    router = getattr(self, "_tpu_backlog_router", None)
    if router is None:
        return original(self, loads)
    loads = list(loads)
    result = original(self, loads)
    router.update(loads)
    return result


@dataclass
class _Dispatch:
    controller: Any
    req: Any
    keys: list = field(default_factory=list)
    held_before: Optional[list] = None
    stock: Optional[int] = None
    sent: Optional[int] = None


_LOCAL = threading.local()


def dispatching_with_trace(original, self, req, *args, **kwargs):
    """AROUND DataParallelController.dispatching_with_trace: scope one request."""
    if getattr(self, "_tpu_backlog_router", None) is None:
        return original(self, req, *args, **kwargs)
    ctx = _Dispatch(self, req)
    selector = getattr(self, "prefix_selector", None)
    # Only requests that take the stock prefix selection can be redirected
    # after select() recorded them; routed and pinned ones never call it or
    # never move.
    if (
        selector is not None
        and getattr(req, "routed_dp_rank", None) is None
        and not _pinned(req)
    ):
        # select() records the prompt on its own choice; keep what that rank
        # held before, to undo the claim if the request goes elsewhere.
        ctx.keys = selector.index.block_keys(req.input_ids or [])
        ctx.held_before = selector.index.match(ctx.keys)
    outer = getattr(_LOCAL, "ctx", None)
    _LOCAL.ctx = ctx
    try:
        return original(self, req, *args, **kwargs)
    finally:
        _LOCAL.ctx = outer
        if (
            ctx.held_before is not None
            and ctx.sent is not None
            and ctx.sent != ctx.stock
            and ctx.stock is not None
        ):
            _forget(selector.index, ctx.stock, ctx.keys[ctx.held_before[ctx.stock] :])
            selector.index.insert(ctx.sent, ctx.keys)


def _forget(index, rank: int, keys) -> None:
    """Drop ``keys`` from ``rank`` in a PrefixAffinityIndex."""
    blocks = index._blocks[rank]
    for key in keys:
        blocks.pop(key, None)


def _prompt_tokens(req) -> tuple[list, int]:
    input_ids = req.input_ids or []
    if input_ids:
        return input_ids, len(input_ids)
    return [], len(getattr(req, "input_embeds", None) or [])


def _pinned(req) -> bool:
    """A stateful session's history lives on its rank, and the router's prefix
    index can't tell cache namespaces apart."""
    return any(
        getattr(req, name, None) is not None
        for name in (
            "session_id",
            "session_params",
            "lora_id",
            "extra_key",
            "cache_salt",
        )
    )


def sock_send(original, sock, obj, *args, **kwargs):
    """AROUND the controller module's sock_send: redirect a scoped dispatch."""
    ctx = getattr(_LOCAL, "ctx", None)
    if ctx is None or obj is not ctx.req or ctx.sent is not None:
        return original(sock, obj, *args, **kwargs)
    ctl = ctx.controller
    stock = next((i for i, w in enumerate(ctl.workers) if w is sock), None)
    router = ctl._tpu_backlog_router
    if stock is None or router is None:
        return original(sock, obj, *args, **kwargs)
    input_ids, n = _prompt_tokens(obj)
    routed = getattr(obj, "routed_dp_rank", None) is not None
    active = [
        r
        for r in ctl._active_workers
        if r < router.num_ranks and ctl.workers[r] is not None and ctl.status[r]
    ]
    chosen = router.route(input_ids, stock, active, routed, n, _pinned(obj))
    if chosen != stock:
        reqs, per_token = (
            (0, 0) if routed else _COUNTED.get(ctl.load_balance_method.name, (0, 0))
        )
        # The stock policies count token ids, not embedding positions.
        n_ids = len(obj.input_ids or [])
        budget = ctl.dp_budget
        if stock < budget.num_dp_ranks:
            budget.total_requests[stock] -= reqs
            budget.total_tokens[stock] -= per_token * n_ids
        if chosen < budget.num_dp_ranks:
            budget.total_requests[chosen] += reqs
            budget.total_tokens[chosen] += per_token * n_ids
    result = original(ctl.workers[chosen], obj, *args, **kwargs)
    # Only after a successful send does the prefix index follow the request.
    ctx.stock, ctx.sent = stock, chosen
    return result


def zmq_reader_owner(original, caller: str) -> bool:
    """AROUND load_snapshot.zmq_reader_owner: under round_robin the controller
    normally never reads loads, so the tokenizer side owns the ZMQ socket;
    backlog routing reads them on every dispatch, so the controller must."""
    from sglang.srt.managers import load_snapshot
    from sglang.srt.runtime_context import get_disagg, get_parallel

    parallel = get_parallel()
    method = parallel.load_balance_method.lower()
    if (
        load_snapshot.should_use_zmq()
        and parallel.node_rank == 0
        and parallel.num_dp_ranks > 1
        and method not in load_snapshot._LOAD_AWARE_METHODS
        and backlog_routing_enabled(
            _margin(),
            method,
            get_disagg().disaggregation_mode,
            elastic=_elastic(
                getattr(parallel, "max_ep_size", None), parallel.num_dp_ranks
            ),
        )
    ):
        return caller == "DataParallelController"
    return original(caller)


def hook_specs():
    return [
        (f"{_DPC}.DataParallelController.__init__", init_controller, HookType.AROUND),
        (f"{_DPC}.DPBudget.update_budget", update_budget, HookType.AROUND),
        (
            f"{_DPC}.DataParallelController.dispatching_with_trace",
            dispatching_with_trace,
            HookType.AROUND,
        ),
        (f"{_DPC}.sock_send", sock_send, HookType.AROUND),
        (
            "sglang.srt.managers.load_snapshot.zmq_reader_owner",
            zmq_reader_owner,
            HookType.AROUND,
        ),
    ]
