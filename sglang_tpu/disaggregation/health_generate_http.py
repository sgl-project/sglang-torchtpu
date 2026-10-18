# SPDX-License-Identifier: Apache-2.0
"""GET /health_generate answered from the scheduler ranks' slot files.

``sglang_tpu.patches`` installs the route after ``TokenizerManager.init_disaggregation``.
An unready rank gives 503; otherwise the core probe handler runs as before.
"""

from __future__ import annotations

import time
import weakref
from typing import Optional

from fastapi import Request

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang_tpu.disaggregation.health_generate import PDHealthState, read_slots
from sglang_tpu.envs import envs as tpu_envs

HEALTH_GENERATE_PATH = "/health_generate"

# TokenizerManager -> the HTTP port whose slot directory to read.
_PORT: weakref.WeakKeyDictionary[object, int] = weakref.WeakKeyDictionary()
# The core /health_generate handler, kept for ready engines and untracked ones.
_ORIGINAL_HANDLER: dict[str, object] = {}


def port_of(tokenizer_manager) -> Optional[int]:
    return _PORT.get(tokenizer_manager)


def attach_slots(tokenizer_manager) -> None:
    mode = tokenizer_manager.disaggregation_mode
    if mode not in (DisaggregationMode.DECODE, DisaggregationMode.PREFILL):
        return
    if not tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.get():
        return
    _PORT[tokenizer_manager] = tokenizer_manager.server_args.port
    install_health_generate_route()


def unready_slots(port: int, now: Optional[float] = None) -> list[dict]:
    """Records of ranks that are unhealthy, or frozen with requests held."""
    import orjson

    now = time.time() if now is None else now
    unready = []
    for slot in read_slots(port):
        record = orjson.loads(slot.record) if slot.record else {}
        if slot.state == PDHealthState.UNHEALTHY.value:
            unready.append(record)
            continue
        age = now - slot.written_at
        if (
            slot.requests_present > 0
            and slot.stale_after_s > 0
            and age >= slot.stale_after_s
        ):
            unready.append(
                {
                    **record,
                    "state": PDHealthState.UNHEALTHY.value,
                    "reason": (
                        f"scheduler stopped judging: slot {age:.0f}s old with "
                        f"{slot.requests_present} requests held "
                        f"(limit {slot.stale_after_s:.0f}s)"
                    ),
                    "slot_age_s": round(age, 1),
                }
            )
    return unready


# Unannotated, FastAPI would read ``request`` as a required query field (422).
async def health_generate_with_slots(request: Request):
    from fastapi.responses import ORJSONResponse

    from sglang.srt.entrypoints import http_server

    port = _PORT.get(http_server.get_global_state().tokenizer_manager)
    unready = unready_slots(port) if port is not None else []
    if unready:
        return ORJSONResponse(
            status_code=503,
            content={
                "role": unready[0].get("role"),
                "state": PDHealthState.UNHEALTHY.value,
                "reason": unready[0].get("reason"),
                "pd_health": unready,
            },
        )
    return await _ORIGINAL_HANDLER["handler"](request)


def install_health_generate_route() -> bool:
    """Swap only GET /health_generate; /health keeps the core handler. Idempotent."""
    from fastapi.routing import APIRoute

    from sglang.srt.entrypoints import http_server

    routes = http_server.app.router.routes
    for route in routes:
        if isinstance(route, APIRoute) and route.path == HEALTH_GENERATE_PATH:
            if route.endpoint is health_generate_with_slots:
                return False
            _ORIGINAL_HANDLER["handler"] = route.endpoint
            routes.remove(route)
            break
    else:
        return False
    http_server.app.add_api_route(
        HEALTH_GENERATE_PATH, health_generate_with_slots, methods=["GET"]
    )
    return True
