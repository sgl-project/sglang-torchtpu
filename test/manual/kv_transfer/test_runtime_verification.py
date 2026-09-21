import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang.srt.raiden_runtime_verification import (
    instrumentation,
    runtime_monitor_driver,
)


@pytest.fixture(autouse=True)
def _clear_concurrency_fuzz_config():
    instrumentation._concurrency_fuzz_max_delay_seconds.cache_clear()
    yield
    instrumentation._concurrency_fuzz_max_delay_seconds.cache_clear()


@pytest.fixture
def observed(monkeypatch):
    events = []

    def step(*aps, bootstrap_room=None):
        events.append((bootstrap_room, aps))

    monitor = SimpleNamespace(step=step, discard=lambda bootstrap_room: None)
    monkeypatch.setattr(
        instrumentation,
        "get_runtime_monitor_driver",
        lambda: monitor,
    )
    return events


def test_tpu_sync_api_instrumentation(monkeypatch, observed):
    class FakeKVCacheManager:
        def register_read(self):
            return True

        def start_read(self):
            raise RuntimeError("expected")

        def register_active_plan(self):
            pass

        def push_registered_plan(self):
            pass

        def unregister_active_plan(self):
            pass

        def unrelated(self):
            pass

    original_unrelated = FakeKVCacheManager.unrelated
    module = ModuleType("tpu_sync.api.torch.kv_cache_manager")
    module.KVCacheManager = FakeKVCacheManager
    monkeypatch.setitem(sys.modules, module.__name__, module)

    instrumentation.install_tpu_sync_api_instrumentation()
    manager = FakeKVCacheManager()
    assert manager.register_read()
    with pytest.raises(RuntimeError, match="expected"):
        manager.start_read()

    assert FakeKVCacheManager.unrelated is original_unrelated
    assert observed == [
        (None, ("register_read", "call_started")),
        (None, ("register_read", "call_completed")),
        (None, ("start_read", "call_started")),
        (None, ("start_read", "call_failed")),
    ]


def test_instrumented_methods_support_random_delay(monkeypatch, observed):
    class FakeKVCacheManager:
        def register_read(self):
            return True

        def start_read(self):
            pass

        def register_active_plan(self):
            pass

        def push_registered_plan(self):
            pass

        def unregister_active_plan(self):
            pass

    module = ModuleType("tpu_sync.api.torch.kv_cache_manager")
    module.KVCacheManager = FakeKVCacheManager
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(
        "SGLANG_TPU_RAIDEN_RUNTIME_VERIFICATION_FUZZ_MAX_DELAY_MS",
        "10",
    )
    uniform_bounds = []

    def choose_delay(lower, upper):
        uniform_bounds.append((lower, upper))
        return 0.004

    monkeypatch.setattr(instrumentation.random, "uniform", choose_delay)
    sleeps = []
    monkeypatch.setattr(instrumentation.time, "sleep", sleeps.append)

    instrumentation.install_tpu_sync_api_instrumentation()
    manager = FakeKVCacheManager()
    assert manager.register_read()
    monkeypatch.setenv(
        "SGLANG_TPU_RAIDEN_RUNTIME_VERIFICATION_FUZZ_MAX_DELAY_MS",
        "20",
    )
    assert manager.register_read()

    assert sleeps == [0.004, 0.004]
    assert uniform_bounds == [(0.0, 0.01), (0.0, 0.01)]
    assert observed == [
        (None, ("register_read", "call_started")),
        (None, ("register_read", "call_completed")),
        (None, ("register_read", "call_started")),
        (None, ("register_read", "call_completed")),
    ]


def test_runtime_monitor_driver_reuses_room_monitor(monkeypatch):
    monitors = []

    class FakeMonitor:
        def __init__(self, *, trace_id=None):
            monitors.append(trace_id)

        def step(self, _valuation):
            return True

    monkeypatch.setattr(runtime_monitor_driver, "GeneratedRuntimeMonitor", FakeMonitor)
    driver = runtime_monitor_driver.RuntimeMonitorDriver()

    driver.step("kv_poll_bootstrapping", bootstrap_room=7)
    driver.step("kv_poll_waiting_for_input", bootstrap_room=7)
    driver.step("kv_poll_bootstrapping", bootstrap_room=8)

    assert monitors == [7, 8]


def test_kv_transfer_state_instrumentation(monkeypatch, observed):
    class FakeKVPoll:
        Failed = 0
        Bootstrapping = 1
        WaitingForInput = 2
        Transferring = 3
        Success = 4

    class FakeCommonKVManager:
        def __init__(self):
            self.request_status = {}

        def update_status(self, bootstrap_room, status):
            self.request_status[bootstrap_room] = status

        def clear_status(self, bootstrap_room):
            self.request_status.pop(bootstrap_room, None)

    class FakeRaidenKVSender:
        bootstrap_room = 8

        def poll(self):
            return FakeKVPoll.Success

    class FakeRaidenKVReceiver:
        bootstrap_room = 9

        def poll(self):
            return FakeKVPoll.Transferring

    base = ModuleType("sglang.srt.disaggregation.base.conn")
    base.KVPoll = FakeKVPoll
    common = ModuleType("sglang.srt.disaggregation.common.conn")
    common.CommonKVManager = FakeCommonKVManager
    raiden = ModuleType("sglang.srt.disaggregation.raiden.conn")
    raiden.RaidenKVSender = FakeRaidenKVSender
    raiden.RaidenKVReceiver = FakeRaidenKVReceiver
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, raiden.__name__, raiden)

    instrumentation.install_kv_transfer_state_instrumentation()
    manager = FakeCommonKVManager()
    manager.update_status(7, FakeKVPoll.Bootstrapping)
    manager.clear_status(7)
    assert FakeRaidenKVSender().poll() == FakeKVPoll.Success
    assert FakeRaidenKVReceiver().poll() == FakeKVPoll.Transferring

    assert observed == [
        (7, ("kv_poll_no_entry", "update_status", "on_call")),
        (7, ("kv_poll_bootstrapping", "update_status", "on_return")),
        (7, ("kv_poll_bootstrapping", "clear_status", "on_call")),
        (7, ("kv_poll_no_entry", "clear_status", "on_return")),
        (8, ("kv_poll_success", "raiden_sender_poll", "on_return")),
        (9, ("kv_poll_transferring", "raiden_receiver_poll", "on_return")),
    ]
