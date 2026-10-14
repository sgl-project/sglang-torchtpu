# SPDX-License-Identifier: Apache-2.0
"""P/D /health_generate judged in the HTTP process from its own request bookkeeping.

``sglang_tpu.patches`` wraps four ``TokenizerManager`` methods to call the
functions below. State lives in a weak map keyed by the manager; the verdict
stays live when the scheduler or the detokenizer stops.
"""

from __future__ import annotations

import weakref
from collections import Counter
from typing import Optional

from fastapi import Request

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang_tpu.disaggregation.health_generate import (
    PDHealthState,
    PDPeerHealth,
    PDProgressHealth,
    build_pd_health_tracker,
    build_pd_peer_tracker,
)

HEALTH_GENERATE_PATH = "/health_generate"


class _HttpHealth:
    """One tokenizer manager's trackers, in-flight rids and their peers."""

    __slots__ = ("tracker", "peers", "in_flight", "rid_peer")

    def __init__(self, tracker: PDProgressHealth, peers: Optional[PDPeerHealth] = None):
        self.tracker = tracker
        self.peers = peers
        self.in_flight: set[str] = set()
        self.rid_peer: dict[str, str] = {}


_STATE: weakref.WeakKeyDictionary[object, _HttpHealth] = weakref.WeakKeyDictionary()
# The core /health_generate handler, kept for managers without a tracker.
_ORIGINAL_HANDLER: dict[str, object] = {}


def state_of(tokenizer_manager) -> Optional[_HttpHealth]:
    return _STATE.get(tokenizer_manager)


def attach_tracker(tokenizer_manager) -> None:
    """One tracker per P/D manager, and the route swap."""
    mode = tokenizer_manager.disaggregation_mode
    if mode not in (DisaggregationMode.DECODE, DisaggregationMode.PREFILL):
        return
    tracker = build_pd_health_tracker(mode.value)
    if tracker is None:
        return
    _STATE[tokenizer_manager] = _HttpHealth(tracker, build_pd_peer_tracker(mode.value))
    install_health_generate_route()


# ----- bookkeeping, each before the core method runs -----


def _observe(tokenizer_manager, state: _HttpHealth) -> None:
    # Rids whose state core dropped (finish, cancel, disconnect) stop being outstanding.
    state.in_flight.intersection_update(tokenizer_manager.rid_to_state)
    state.tracker.observe(len(state.in_flight))
    if state.peers is not None:
        for rid in list(state.rid_peer):
            if rid not in state.in_flight:
                del state.rid_peer[rid]
        state.peers.observe(Counter(state.rid_peer.values()))


def _peer_of(tokenizer_manager, rid: str) -> Optional[str]:
    """The prefill this decode request pairs with, as the router stamped it."""
    req_state = tokenizer_manager.rid_to_state.get(rid)
    if req_state is None or not isinstance(req_state.obj, GenerateReqInput):
        return None
    host, port = req_state.obj.bootstrap_host, req_state.obj.bootstrap_port
    if isinstance(host, list):
        host = host[0] if host else None
    if isinstance(port, list):
        port = port[0] if port else None
    if host is None:
        return None
    return f"{host}:{port}" if port is not None else str(host)


def track_dispatched(tokenizer_manager, rid: str) -> None:
    """A real request went out; probes never count."""
    state = _STATE.get(tokenizer_manager)
    if state is None or rid.startswith(HEALTH_CHECK_RID_PREFIX):
        return
    state.in_flight.add(rid)
    if state.peers is not None:
        peer = _peer_of(tokenizer_manager, rid)
        if peer is not None:
            state.rid_peer[rid] = peer
    _observe(tokenizer_manager, state)


def track_outputs(tokenizer_manager, recv_obj) -> None:
    """Chunks came back: progress, or an engine-side abort."""
    state = _STATE.get(tokenizer_manager)
    if state is None:
        return
    decode = tokenizer_manager.disaggregation_mode == DisaggregationMode.DECODE
    touched = False
    for rid, finish_reason in zip(recv_obj.rids, recv_obj.finished_reasons):
        if rid not in state.in_flight:
            continue
        touched = True
        if finish_reason is not None:
            state.in_flight.discard(rid)
        peer = state.rid_peer.get(rid) if state.peers is not None else None
        if finish_reason is not None and finish_reason.get("type") == "abort":
            state.tracker.record_failure()
            if peer is not None:
                state.peers.record_failure(peer)
        elif finish_reason is not None or decode:
            # A decode streams a chunk per generated token; a prefill's only
            # output is its final one, sent after the KV release that follows
            # a completed transfer.
            state.tracker.record_progress()
            if peer is not None:
                state.peers.record_progress(peer)
    if touched:
        _observe(tokenizer_manager, state)


def track_cancelled(tokenizer_manager, *, rid: str, abort_all: bool) -> None:
    """A client gave up: no longer outstanding, records nothing."""
    state = _STATE.get(tokenizer_manager)
    if state is None:
        return
    if abort_all:
        state.in_flight.clear()
    else:
        state.in_flight.discard(rid)
    _observe(tokenizer_manager, state)


# ----- the /health_generate route -----


def health_response(tracker: PDProgressHealth, peers: Optional[PDPeerHealth] = None):
    """503 with the records when the engine or its peers are unhealthy, else a
    bare 200: no probe request."""
    import orjson
    from fastapi.responses import ORJSONResponse, Response

    engine_unhealthy = tracker.health_state() == PDHealthState.UNHEALTHY.value
    peers_unhealthy = (
        peers is not None and peers.health_state() == PDHealthState.UNHEALTHY.value
    )
    if not engine_unhealthy and not peers_unhealthy:
        return Response(status_code=200)
    record = orjson.loads(tracker.health_json())
    peer_record = orjson.loads(peers.health_json()) if peers is not None else None
    reason = record["reason"] if engine_unhealthy else peer_record["reason"]
    return ORJSONResponse(
        status_code=503,
        content={
            "role": record.get("role"),
            "state": PDHealthState.UNHEALTHY.value,
            "reason": reason,
            "pd_health": record,
            "pd_peers": peer_record,
        },
    )


# Unannotated, FastAPI would read ``request`` as a required query field (422).
async def health_generate_with_tracker(request: Request):
    from fastapi.responses import Response

    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers.tokenizer_manager import ServerStatus

    tokenizer_manager = http_server.get_global_state().tokenizer_manager
    state = _STATE.get(tokenizer_manager)
    if state is None:
        return await _ORIGINAL_HANDLER["handler"](request)
    if tokenizer_manager.gracefully_exit:
        return Response(status_code=503)
    if tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)
    return health_response(state.tracker, state.peers)


def install_health_generate_route() -> bool:
    """Swap only GET /health_generate; /health keeps the core handler. Idempotent."""
    from fastapi.routing import APIRoute

    from sglang.srt.entrypoints import http_server

    routes = http_server.app.router.routes
    for route in routes:
        if isinstance(route, APIRoute) and route.path == HEALTH_GENERATE_PATH:
            if route.endpoint is health_generate_with_tracker:
                return False
            _ORIGINAL_HANDLER["handler"] = route.endpoint
            routes.remove(route)
            break
    else:
        return False
    http_server.app.add_api_route(
        HEALTH_GENERATE_PATH, health_generate_with_tracker, methods=["GET"]
    )
    return True
