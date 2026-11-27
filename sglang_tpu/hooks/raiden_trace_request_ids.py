# SPDX-License-Identifier: Apache-2.0
"""Bind scheduler request IDs onto RaidenKVManager for transfer trace spans.

Why: ``PrefillBootstrapQueue.create_sender`` and
     ``DecodePreallocQueue._create_receiver_and_enqueue`` pass
     ``bootstrap_room`` but not ``req.rid`` when creating KVSender/KVReceiver.
     Binding ``bootstrap_room -> req.rid`` lets Raiden transfer spans emit the
     same ``request_id`` as step spans on the same host while ``uuid`` links
     prefill and decode hosts.
How: ``HookType.AFTER`` on both queue methods forwards
     ``(req.bootstrap_room, req.rid)`` to
     ``kv_manager.register_trace_request_id`` when supported.
Upstream PR or issue: none.
Remove when: core passes ``req.rid`` into KVSender / KVReceiver construction.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.plugins.hook_registry import HookType

_TARGETS = (
    "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue.create_sender",
    "sglang.srt.disaggregation.decode.DecodePreallocQueue._create_receiver_and_enqueue",
)


def bind_trace_request_id(result: Any, self: Any, req: Any, *args: Any, **kwargs: Any):
    register = getattr(
        getattr(self, "kv_manager", None), "register_trace_request_id", None
    )
    if register is not None:
        room = getattr(req, "bootstrap_room", None)
        rid = getattr(req, "rid", None)
        if room is not None and rid is not None:
            register(room, rid)
    return result


def hook_specs() -> list[tuple[str, Any, HookType]]:
    return [(target, bind_trace_request_id, HookType.AFTER) for target in _TARGETS]
