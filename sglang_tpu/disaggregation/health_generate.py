"""P/D health from completed work, the core probe and confirmed worker crashes."""

from __future__ import annotations

import math
import time
from typing import Optional

import msgspec
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import HealthCheckOutput
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.srt.observability.req_time_stats import real_time
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs

logger = get_logger(__name__)

HEALTH_GENERATE_PATH = "/health_generate"
# The scheduler reports a completed prefill chunk or decode step as a core
# HealthCheckOutput carrying this rid; core's own probe replies carry none.
COMPLETED_WORK_RID = f"{HEALTH_CHECK_RID_PREFIX}_tpu_completed_work"
# The core handler, which runs the probe; kept to call it from the swapped route.
_ORIGINAL_HANDLER: dict[str, object] = {}


class GenerationHealth(msgspec.Struct, frozen=True):
    ok: bool
    probe_ok: Optional[bool]
    stalled: bool
    held: int
    waiting: int
    oldest_wait_s: float
    idle_s: Optional[float]
    window_s: Optional[float]
    reason: str
    worker_failures: tuple[str, ...] = ()


def configure_window() -> None:
    """Validate a progress window before transport aborts remove requests.

    The rule only sees requests still held in the table, and the transport
    timeouts abort stuck requests, so a window at or above them never fires.
    """
    bootstrap_s = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
    waiting_s = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
    transport_s = min(bootstrap_s, waiting_s)
    window_s = tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.get()
    if window_s == 0:
        return
    if not (0 < window_s < transport_s) or not math.isfinite(window_s):
        raise ValueError(
            f"SGLANG_TPU_PD_HEALTH_WINDOW_S={window_s} must be positive and below "
            f"min(SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT={bootstrap_s}, "
            f"SGLANG_DISAGGREGATION_WAITING_TIMEOUT={waiting_s})={transport_s}; "
            "set a lower value; 0 disables the progress window"
        )


def window_s() -> Optional[float]:
    return tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.get() or None


def generation_stall(tokenizer_manager, window: float):
    """Read the request table: (stalled, held, waiting, oldest_wait_s, idle_s)."""
    now = time.perf_counter()
    held = waiting = 0
    oldest_wait = oldest_dispatch = 0.0
    # Keep progress across request cleanup, independently of metrics.
    last_produced = getattr(tokenizer_manager, "_tpu_pd_last_progress_time", 0.0)
    for rid, state in list(tokenizer_manager.rid_to_state.items()):
        if not state.dispatched or rid.startswith(HEALTH_CHECK_RID_PREFIX):
            continue
        held += 1
        stats = state.time_stats
        since = stats.api_server_dispatch_time or stats.created_time
        dispatch_age = now - since
        oldest_dispatch = max(oldest_dispatch, dispatch_age)
        if stats.first_token_time == 0.0:
            waiting += 1
            oldest_wait = max(oldest_wait, dispatch_age)
        else:
            last_produced = max(last_produced, stats.last_time)
    idle_s = now - last_produced if last_produced else None
    stalled = oldest_dispatch > window and (idle_s is None or idle_s > window)
    return stalled, held, waiting, oldest_wait, idle_s


def _worker_failures(tokenizer_manager) -> tuple[str, ...]:
    """Non-zero exit codes of the HTTP parent's child processes; no watchdog means none."""
    watchdog = getattr(tokenizer_manager, "_subprocess_watchdog", None)
    failures = []
    names = getattr(watchdog, "_names", ())
    for i, process in enumerate(getattr(watchdog, "_processes", ())):
        try:
            exitcode = getattr(process, "exitcode", None)
        except (AssertionError, OSError, ValueError):
            continue
        if isinstance(exitcode, int) and exitcode != 0:
            name = names[i] if i < len(names) else f"process_{i}"
            failures.append(f"{name} (exitcode {exitcode})")
    return tuple(failures)


def verdict(tokenizer_manager, *, probe_ok: Optional[bool], window: Optional[float]):
    """Reject crashes or expired progress; allow active work its full window."""
    stalled, held, waiting, oldest_wait, idle_s = generation_stall(
        tokenizer_manager, window if window else math.inf
    )
    worker_failures = _worker_failures(tokenizer_manager)
    reasons = []
    if worker_failures:
        reasons.append("worker crash: " + ", ".join(worker_failures))
    if probe_ok is False:
        reasons.append(
            "probe: no scheduler output within the probe timeout, last message "
            f"{real_time() - tokenizer_manager.last_receive_tstamp:.0f}s ago"
        )
    if stalled:
        produced = (
            "no completed chunk or token since dispatch"
            if idle_s is None
            else f"no completed chunk or token for {idle_s:.0f}s"
        )
        reasons.append(
            f"stalled: {held} requests held, oldest waiting {oldest_wait:.0f}s "
            f"for a first token, {produced} (window {window:.0f}s)"
        )
    return GenerationHealth(
        # Active work gets the full progress window, even if the probe times out.
        ok=not worker_failures
        and not stalled
        and (probe_ok is True or bool(window and held)),
        probe_ok=probe_ok,
        stalled=stalled,
        held=held,
        waiting=waiting,
        oldest_wait_s=round(oldest_wait, 3),
        idle_s=None if idle_s is None else round(idle_s, 3),
        window_s=window,
        reason="; ".join(reasons),
        worker_failures=worker_failures,
    )


def mark_progress(tokenizer_manager, recv_obj) -> None:
    """Record chunks or tokens before cleanup, without changing metric clocks."""
    if isinstance(recv_obj, HealthCheckOutput):
        completed = recv_obj.rid == COMPLETED_WORK_RID
    else:
        completed = any(
            not rid.startswith(HEALTH_CHECK_RID_PREFIX)
            and not _is_abort(finished)
            and (state := tokenizer_manager.rid_to_state.get(rid)) is not None
            and state.dispatched
            for rid, finished in zip(
                getattr(recv_obj, "rids", ()), getattr(recv_obj, "finished_reasons", ())
            )
        )
    if completed:
        tokenizer_manager._tpu_pd_last_progress_time = time.perf_counter()


def _is_abort(finished_reason) -> bool:
    # Abort results end requests; they are not completed work.
    return isinstance(finished_reason, dict) and finished_reason.get("type") == "abort"


def report_completed_work(scheduler, batch) -> None:
    """Publish completed real compute, including buffered decode tokens."""
    mode = batch.forward_mode
    if scheduler.disaggregation_mode == DisaggregationMode.NULL or not window_s():
        return
    if not (
        mode.is_extend_without_speculative()
        or mode.is_decode()
        or mode.is_target_verify()
    ):
        return
    routes = {
        req.http_worker_ipc
        for req in batch.reqs
        if not req.rid.startswith(HEALTH_CHECK_RID_PREFIX)
        and not isinstance(getattr(req, "finished_reason", None), FINISH_ABORT)
        and not isinstance(getattr(req, "to_finish", None), FINISH_ABORT)
    }
    for route in routes:
        scheduler.ipc_channels.send_to_tokenizer.send_output(
            HealthCheckOutput(rid=COMPLETED_WORK_RID, http_worker_ipc=route)
        )


async def health_generate(request: Request) -> Response:
    """GET /health_generate: progress window, core probe and worker failures."""
    from sglang.srt.entrypoints import http_server

    tokenizer_manager = http_server.get_global_state().tokenizer_manager
    probe_ok = None
    if (
        tokenizer_manager.gracefully_exit
        or tokenizer_manager.server_status == ServerStatus.Starting
        or not _worker_failures(tokenizer_manager)
    ):
        core = await _ORIGINAL_HANDLER["handler"](request)
        if tokenizer_manager.gracefully_exit or (
            tokenizer_manager.server_status == ServerStatus.Starting
        ):
            return core
        probe_ok = core.status_code == 200
    health = verdict(tokenizer_manager, probe_ok=probe_ok, window=window_s())
    if health.ok:
        tokenizer_manager.server_status = ServerStatus.Up
        if health.reason:
            logger.info("P/D health diagnostics (reporting healthy): %s", health.reason)
            return JSONResponse(msgspec.to_builtins(health), status_code=200)
        return Response(status_code=200)
    logger.warning("P/D health check failed: %s", health.reason)
    tokenizer_manager.server_status = ServerStatus.UnHealthy
    return JSONResponse(msgspec.to_builtins(health), status_code=503)


def install_health_generate_route() -> bool:
    """Swap only GET /health_generate; /health keeps the core handler. Idempotent."""
    from fastapi.routing import APIRoute

    from sglang.srt.entrypoints import http_server

    routes = http_server.app.router.routes
    for route in routes:
        if not isinstance(route, APIRoute) or route.path != HEALTH_GENERATE_PATH:
            continue
        if route.endpoint is health_generate:
            return False
        _ORIGINAL_HANDLER["handler"] = route.endpoint
        routes.remove(route)
        http_server.app.add_api_route(
            HEALTH_GENERATE_PATH, health_generate, methods=["GET"]
        )
        return True
    return False
