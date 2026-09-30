"""Process-local runtime monitor for Raiden transfer state."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_ENABLED = envs.SGLANG_TPU_ENABLE_RAIDEN_RUNTIME_VERIFICATION.get()


class RuntimeVerificationError(RuntimeError):
    pass


_VALID_STATES: set[int | None] = {
    KVPoll.Failed,
    KVPoll.Bootstrapping,
    KVPoll.WaitingForInput,
    KVPoll.Transferring,
    KVPoll.Success,
    None,
}


def _state_label(state: int | None) -> str:
    if state is None:
        return "kv_poll_none"
    return {
        KVPoll.Failed: "kv_poll_failed",
        KVPoll.Bootstrapping: "kv_poll_bootstrapping",
        KVPoll.WaitingForInput: "kv_poll_waiting_for_input",
        KVPoll.Transferring: "kv_poll_transferring",
        KVPoll.Success: "kv_poll_success",
    }.get(state, f"kv_poll_unknown_{state}")


@dataclass(frozen=True)
class _Transition:
    src_state: int | None
    label: str
    dst_state: int | None
    label_param: int | None = None

    @property
    def label_str(self) -> str:
        if self.label_param is None:
            return self.label
        return f"{self.label}({_state_label(self.label_param)})"


_VALID_SENDER_POLL_RESULTS = {
    KVPoll.WaitingForInput: {
        KVPoll.WaitingForInput,
        KVPoll.Transferring,
        KVPoll.Success,
        KVPoll.Failed,
    },
    KVPoll.Transferring: {
        KVPoll.WaitingForInput,
        KVPoll.Transferring,
        KVPoll.Success,
        KVPoll.Failed,
    },
    KVPoll.Success: {KVPoll.Success},
    KVPoll.Failed: {
        KVPoll.WaitingForInput,
        KVPoll.Transferring,
        KVPoll.Success,
        KVPoll.Failed,
    },
}

_VALID_RECEIVER_POLL_RESULTS = {
    KVPoll.Bootstrapping: {
        KVPoll.Bootstrapping,
    },
    KVPoll.WaitingForInput: {
        KVPoll.WaitingForInput,
        KVPoll.Success,
    },
    KVPoll.Transferring: {
        KVPoll.WaitingForInput,
        KVPoll.Transferring,
        KVPoll.Success,
    },
    KVPoll.Failed: {
        KVPoll.WaitingForInput,
        KVPoll.Transferring,
        KVPoll.Success,
        KVPoll.Failed,
    },
}


_VALID_SELF_TRANSITIONS: set[_Transition] = {
    *(
        _Transition(state, "update_status", state)
        for state in {
            KVPoll.Bootstrapping,
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
            KVPoll.Success,
            KVPoll.Failed,
        }
    ),
    _Transition(None, "clear_status", None),
    *(
        _Transition(KVPoll.WaitingForInput, label, KVPoll.WaitingForInput)
        for label in {
            "append_send_blocks",
            "finalize_send",
            "maybe_start_pull",
            "offer_chunk",
            "register_decode_request",
        }
    ),
    _Transition(KVPoll.Transferring, "maybe_start_pull", KVPoll.Transferring),
    _Transition(KVPoll.Bootstrapping, "maybe_start_pull", KVPoll.Bootstrapping),
    _Transition(KVPoll.Failed, "fail_decode_silence", KVPoll.Failed),
    _Transition(KVPoll.Failed, "maybe_start_pull", KVPoll.Failed),
    _Transition(None, "clear_room", None),
    *(
        _Transition(
            state,
            "raiden_sender_poll",
            state,
            label_param=observed_state,
        )
        for state, observed_states in _VALID_SENDER_POLL_RESULTS.items()
        for observed_state in observed_states
    ),
    *(
        _Transition(
            state,
            "raiden_receiver_poll",
            state,
            label_param=observed_state,
        )
        for state, observed_states in _VALID_RECEIVER_POLL_RESULTS.items()
        for observed_state in observed_states
    ),
}


_VALID_TRANSITIONS: set[_Transition] = {
    _Transition(
        None,
        "update_status",
        KVPoll.Bootstrapping,
    ),
    _Transition(
        None,
        "update_status",
        KVPoll.WaitingForInput,
    ),
    _Transition(
        KVPoll.Bootstrapping,
        "update_status",
        KVPoll.WaitingForInput,
    ),
    _Transition(
        KVPoll.Bootstrapping,
        "update_status",
        KVPoll.Failed,
    ),
    _Transition(
        KVPoll.WaitingForInput,
        "update_status",
        KVPoll.Transferring,
    ),
    _Transition(
        KVPoll.WaitingForInput,
        "update_status",
        KVPoll.Success,
    ),
    _Transition(
        KVPoll.WaitingForInput,
        "update_status",
        KVPoll.Failed,
    ),
    _Transition(
        KVPoll.Transferring,
        "update_status",
        KVPoll.Failed,
    ),
    _Transition(
        KVPoll.Success,
        "update_status",
        KVPoll.Failed,
    ),
    *(
        _Transition(state, "clear_status", None)
        for state in _VALID_STATES
        if state is not None
    ),
    *_VALID_SELF_TRANSITIONS,
}


class RuntimeMonitor:
    """Validate Raiden state transitions independently for each bootstrap room."""

    def __init__(self) -> None:
        self.process_id = os.getpid()
        self._states: dict[int, int | None] = {}
        self._violated = False
        self._lock = threading.Lock()

    def _raise_violation(
        self,
        bootstrap_room: int,
        transition_label: str,
        reason: str,
    ) -> None:
        self._violated = True
        message = (
            "RuntimeMonitor detected a violation in bootstrap_room=%s "
            "transition_label=%s: %s accepted=False. The full observed trace is "
            "available in the preceding RuntimeMonitor logs."
        ) % (
            bootstrap_room,
            transition_label,
            reason,
        )
        logger.error(message)
        raise RuntimeVerificationError(message)

    def _assert_transition_locked(
        self,
        label: str,
        bootstrap_room: int,
        next_state: int | None,
        label_param: int | None = None,
    ) -> None:
        previous = self._states.get(bootstrap_room)
        transition = _Transition(previous, label, next_state, label_param)
        if next_state not in _VALID_STATES:
            self._raise_violation(
                bootstrap_room,
                transition.label_str,
                f"unknown state {_state_label(next_state)}",
            )
        if transition not in _VALID_TRANSITIONS:
            self._raise_violation(
                bootstrap_room,
                transition.label_str,
                f"invalid state transition "
                f"{_state_label(previous)}->{_state_label(next_state)}",
            )
        self._states[bootstrap_room] = next_state
        logger.info(
            "RuntimeMonitor bootstrap_room=%s transition_label=%s "
            "state=%s->%s accepted=True",
            bootstrap_room,
            transition.label_str,
            _state_label(previous),
            _state_label(next_state),
        )

    def assert_transition(
        self,
        next_state: int | None,
        label: str,
        bootstrap_room: int,
        label_param: int | None = None,
    ) -> None:
        with self._lock:
            self._assert_transition_locked(
                label, bootstrap_room, next_state, label_param
            )

    def assert_self_transition(
        self,
        label: str,
        bootstrap_room: int,
        label_param: int | None = None,
    ) -> None:
        with self._lock:
            self._assert_transition_locked(
                label,
                bootstrap_room,
                self._states.get(bootstrap_room),
                label_param,
            )

    def discard(self, bootstrap_room: int) -> None:
        with self._lock:
            self._states.pop(bootstrap_room, None)

    @property
    def violated(self) -> bool:
        with self._lock:
            return self._violated


_monitor_lock = threading.Lock()
_monitor: RuntimeMonitor | None = None


def get_runtime_monitor() -> RuntimeMonitor:
    """Return one runtime monitor per OS process, including after a fork."""
    global _monitor
    process_id = os.getpid()
    with _monitor_lock:
        if _monitor is None or _monitor.process_id != process_id:
            _monitor = RuntimeMonitor()
        return _monitor


def assert_raiden_transition(
    bootstrap_room: int,
    label: str,
    next_state: int | None,
    *,
    label_param: int | None = None,
) -> None:
    """Assert one Raiden state transition when verification is enabled."""
    if not _ENABLED:
        return

    get_runtime_monitor().assert_transition(
        next_state,
        label,
        bootstrap_room=bootstrap_room,
        label_param=label_param,
    )


def assert_raiden_self_transition(
    bootstrap_room: int,
    label: str,
    *,
    label_param: int | None = None,
) -> None:
    """Assert one labeled Raiden observation without changing state."""
    if not _ENABLED:
        return

    get_runtime_monitor().assert_self_transition(
        label,
        bootstrap_room=bootstrap_room,
        label_param=label_param,
    )


def discard_raiden_trace(bootstrap_room: int) -> None:
    if _ENABLED:
        get_runtime_monitor().discard(bootstrap_room)
