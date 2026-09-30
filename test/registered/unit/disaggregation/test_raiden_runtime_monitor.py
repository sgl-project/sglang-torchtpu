import sys
import threading

import pytest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden import runtime_monitor
from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVManager,
    RaidenKVReceiver,
    RaidenKVSender,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def monitored_manager(monkeypatch):
    monitor = runtime_monitor.RuntimeMonitor()
    manager = RaidenKVManager.__new__(RaidenKVManager)
    manager.request_status = {}
    manager._status_lock = threading.Lock()
    manager._drop_room_state = lambda room, *, clear_failure_record: None

    monkeypatch.setattr(runtime_monitor, "_ENABLED", True)
    monkeypatch.setattr(runtime_monitor, "_monitor", monitor)
    return manager, monitor


def test_runtime_monitor_accepts_room_lifecycle(monitored_manager):
    manager, monitor = monitored_manager
    bootstrap_room = 7
    sender = RaidenKVSender.__new__(RaidenKVSender)
    sender.bootstrap_room = bootstrap_room
    sender._poll_impl = lambda: KVPoll.Success

    manager.update_status(bootstrap_room, KVPoll.Bootstrapping)
    assert monitor._states[bootstrap_room] == KVPoll.Bootstrapping

    for state in (
        KVPoll.WaitingForInput,
        KVPoll.WaitingForInput,
        KVPoll.Transferring,
    ):
        manager.update_status(bootstrap_room, state)

    assert sender.poll() == KVPoll.Success
    assert monitor._states[bootstrap_room] == KVPoll.Transferring

    manager.clear_status(bootstrap_room)
    assert monitor._states[bootstrap_room] is None

    manager.clear_room(bootstrap_room)
    assert bootstrap_room not in monitor._states

    zero_page_room = 8
    receiver = RaidenKVReceiver.__new__(RaidenKVReceiver)
    receiver.bootstrap_room = zero_page_room
    receiver._poll_impl = lambda: KVPoll.Success
    manager.update_status(zero_page_room, KVPoll.WaitingForInput)

    assert receiver.poll() == KVPoll.Success
    manager.clear_status(zero_page_room)
    manager.clear_room(zero_page_room)
    assert zero_page_room not in monitor._states

    dummy_cp_room = 9
    manager.update_status(dummy_cp_room, KVPoll.WaitingForInput)
    manager.update_status(dummy_cp_room, KVPoll.Success)
    assert monitor._states[dummy_cp_room] == KVPoll.Success
    manager.clear_status(dummy_cp_room)
    manager.clear_room(dummy_cp_room)


def test_runtime_monitor_rejects_invalid_room_lifecycle(monitored_manager):
    manager, monitor = monitored_manager
    bootstrap_room = 7

    manager.update_status(bootstrap_room, KVPoll.Bootstrapping)

    with pytest.raises(runtime_monitor.RuntimeVerificationError):
        manager.update_status(bootstrap_room, KVPoll.Transferring)

    assert monitor.violated

    receiver = RaidenKVReceiver.__new__(RaidenKVReceiver)
    receiver.bootstrap_room = 8
    receiver._poll_impl = lambda: KVPoll.Bootstrapping
    manager.update_status(receiver.bootstrap_room, KVPoll.WaitingForInput)

    with pytest.raises(runtime_monitor.RuntimeVerificationError):
        receiver.poll()


def test_runtime_monitor_checks_manager_api_states(monitored_manager):
    manager, monitor = monitored_manager
    bootstrap_room = 7

    receiver = RaidenKVReceiver.__new__(RaidenKVReceiver)
    receiver.bootstrap_room = bootstrap_room
    receiver._poll_impl = lambda: KVPoll.Bootstrapping
    manager.update_status(bootstrap_room, KVPoll.Bootstrapping)
    manager.chunked_transfer = False
    manager._start_whole_pull = lambda room: False
    assert not manager.maybe_start_pull(bootstrap_room)
    assert receiver.poll() == KVPoll.Bootstrapping

    manager.clear_status(bootstrap_room)
    manager.update_status(bootstrap_room, KVPoll.WaitingForInput)
    for label in (
        "append_send_blocks",
        "offer_chunk",
        "finalize_send",
        "register_decode_request",
    ):
        monitor.assert_self_transition(label, bootstrap_room)

    manager.chunked_transfer = False
    manager._start_whole_pull = lambda room: False
    assert not manager.maybe_start_pull(bootstrap_room)

    manager.update_status(bootstrap_room, KVPoll.Transferring)
    manager._start_whole_pull = lambda room: True
    assert manager.maybe_start_pull(bootstrap_room)
    manager.update_status(bootstrap_room, KVPoll.Failed)
    monitor.assert_self_transition("fail_decode_silence", bootstrap_room)

    manager.clear_status(bootstrap_room)
    monitor.assert_self_transition("clear_room", bootstrap_room)

    invalid_room = 8
    manager.update_status(invalid_room, KVPoll.Bootstrapping)
    with pytest.raises(runtime_monitor.RuntimeVerificationError):
        monitor.assert_self_transition("append_send_blocks", invalid_room)


def test_cleared_room_trace_is_not_recreated_by_late_failure(monitored_manager):
    manager, monitor = monitored_manager
    bootstrap_room = 7
    sender = RaidenKVSender.__new__(RaidenKVSender)
    sender.bootstrap_room = bootstrap_room

    manager.update_status(bootstrap_room, KVPoll.Bootstrapping)
    manager.update_status(bootstrap_room, KVPoll.WaitingForInput)
    manager.update_status(bootstrap_room, KVPoll.Transferring)
    manager.update_status(bootstrap_room, KVPoll.Failed)
    assert monitor._states[bootstrap_room] == KVPoll.Failed

    sender._poll_impl = lambda: KVPoll.Transferring
    assert sender.poll() == KVPoll.Transferring
    sender._poll_impl = lambda: KVPoll.Failed
    assert sender.poll() == KVPoll.Failed

    manager.clear_status(bootstrap_room)
    manager.clear_room(bootstrap_room)
    assert bootstrap_room not in monitor._states

    manager.update_status(bootstrap_room, KVPoll.Failed)
    assert bootstrap_room not in manager.request_status
    assert bootstrap_room not in monitor._states


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
