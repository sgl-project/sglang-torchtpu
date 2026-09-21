"""Opt-in instrumentation for TPU Sync calls and Raiden transfer states."""

from __future__ import annotations

import functools
import inspect
import math
import random
import time
from collections.abc import Callable
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.raiden_runtime_verification.runtime_monitor_driver import (
    get_runtime_monitor_driver,
)

_METHOD_HOOKS_MARKER = "_sglang_raiden_runtime_verification_hooks_registered"


@functools.cache
def _concurrency_fuzz_max_delay_seconds() -> float:
    max_delay_ms = envs.SGLANG_TPU_RAIDEN_RUNTIME_VERIFICATION_FUZZ_MAX_DELAY_MS.get()
    if max_delay_ms < 0 or not math.isfinite(max_delay_ms):
        raise ValueError(
            "SGLANG_TPU_RAIDEN_RUNTIME_VERIFICATION_FUZZ_MAX_DELAY_MS must be a "
            f"non-negative finite number, got {max_delay_ms!r}"
        )
    return max_delay_ms / 1000.0


def inject_concurrency_fuzz_delay() -> None:
    max_delay_seconds = _concurrency_fuzz_max_delay_seconds()
    if max_delay_seconds > 0:
        time.sleep(random.uniform(0.0, max_delay_seconds))


def _register_method_hooks(
    owner: type,
    method_name: str,
    *,
    on_call: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
    on_return: Callable[[tuple[Any, ...], dict[str, Any], Any], None] | None = None,
    on_error: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
) -> None:
    try:
        function = vars(owner)[method_name]
    except KeyError:
        raise AttributeError(
            f"{owner.__qualname__} does not define {method_name!r}"
        ) from None

    if not inspect.isfunction(function):
        raise TypeError(
            f"cannot instrument non-instance method "
            f"{owner.__qualname__}.{method_name}"
        )

    if getattr(function, _METHOD_HOOKS_MARKER, False):
        return

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if on_call is not None:
            on_call(args, kwargs)

        try:
            result = function(*args, **kwargs)
        except Exception:
            if on_error is not None:
                on_error(args, kwargs)
            raise

        if on_return is not None:
            on_return(args, kwargs, result)
        return result

    setattr(wrapped, _METHOD_HOOKS_MARKER, True)
    setattr(owner, method_name, wrapped)


def install_tpu_sync_api_instrumentation() -> None:
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    def publish_call(
        method_name: str,
        phase: str,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        _result: Any = None,
    ) -> None:
        get_runtime_monitor_driver().step(method_name, phase)

    def publish_call_and_delay(
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        publish_call(method_name, "call_started", args, kwargs)
        inject_concurrency_fuzz_delay()

    for method_name in (
        "register_read",
        "start_read",
        "register_active_plan",
        "push_registered_plan",
        "unregister_active_plan",
    ):
        _register_method_hooks(
            KVCacheManager,
            method_name,
            on_call=functools.partial(publish_call_and_delay, method_name),
            on_return=functools.partial(publish_call, method_name, "call_completed"),
            on_error=functools.partial(publish_call, method_name, "call_failed"),
        )


def install_kv_transfer_state_instrumentation() -> None:
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.disaggregation.common.conn import CommonKVManager
    from sglang.srt.disaggregation.raiden.conn import (
        RaidenKVReceiver,
        RaidenKVSender,
    )

    def state_ap(state: int | None) -> str:
        if state is None:
            return "kv_poll_no_entry"
        return {
            KVPoll.Failed: "kv_poll_failed",
            KVPoll.Bootstrapping: "kv_poll_bootstrapping",
            KVPoll.WaitingForInput: "kv_poll_waiting_for_input",
            KVPoll.Transferring: "kv_poll_transferring",
            KVPoll.Success: "kv_poll_success",
        }.get(state, f"kv_poll_unknown_{state}")

    def publish_state(
        method_name: str,
        phase: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        _result: Any = None,
    ) -> None:
        # Instance methods receive self as args[0]; bootstrap_room may follow
        # positionally or be passed by keyword.
        manager = args[0]
        assert isinstance(manager, CommonKVManager)
        bootstrap_room = args[1] if len(args) > 1 else kwargs["bootstrap_room"]
        state = manager.request_status.get(bootstrap_room)
        monitor = get_runtime_monitor_driver()
        monitor.step(
            state_ap(state),
            method_name,
            phase,
            bootstrap_room=bootstrap_room,
        )
        if method_name == "clear_status" and phase == "on_return":
            monitor.discard(bootstrap_room)

    def publish_state_and_delay(
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        publish_state(method_name, "on_call", args, kwargs)
        inject_concurrency_fuzz_delay()

    for method_name in ("update_status", "clear_status"):
        _register_method_hooks(
            CommonKVManager,
            method_name,
            on_call=functools.partial(publish_state_and_delay, method_name),
            on_return=functools.partial(publish_state, method_name, "on_return"),
        )

    def publish_poll_result(
        method_ap: str,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        result: Any,
    ) -> None:
        poller = args[0]
        get_runtime_monitor_driver().step(
            state_ap(result),
            method_ap,
            "on_return",
            bootstrap_room=poller.bootstrap_room,
        )

    def publish_poll_error(
        method_ap: str,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        poller = args[0]
        get_runtime_monitor_driver().step(
            method_ap,
            "call_failed",
            bootstrap_room=poller.bootstrap_room,
        )

    def delay_poll(
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        inject_concurrency_fuzz_delay()

    _register_method_hooks(
        RaidenKVSender,
        "poll",
        on_call=delay_poll,
        on_return=functools.partial(publish_poll_result, "raiden_sender_poll"),
        on_error=functools.partial(publish_poll_error, "raiden_sender_poll"),
    )
    _register_method_hooks(
        RaidenKVReceiver,
        "poll",
        on_call=delay_poll,
        on_return=functools.partial(publish_poll_result, "raiden_receiver_poll"),
        on_error=functools.partial(publish_poll_error, "raiden_receiver_poll"),
    )


def install_runtime_verification() -> bool:
    """Install the process-local hooks when explicitly enabled."""
    if not envs.SGLANG_TPU_ENABLE_RAIDEN_RUNTIME_VERIFICATION.get():
        return False

    install_tpu_sync_api_instrumentation()
    install_kv_transfer_state_instrumentation()
    return True
