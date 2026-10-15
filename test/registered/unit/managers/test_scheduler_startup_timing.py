from types import SimpleNamespace

import pytest

from sglang.srt.managers import scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("device,peer_wait", [("tpu", 0), ("tpu", 300), ("cuda", 300)])
def test_model_startup_excludes_wait_for_peers(monkeypatch, device, peer_wait):
    now = 0.0
    barriers = []

    def barrier():
        nonlocal now
        barriers.append(now)
        now += peer_wait

    monkeypatch.setattr(scheduler.time, "perf_counter", lambda: now)
    monkeypatch.setattr(scheduler, "get_device", lambda: SimpleNamespace(device=device))
    monkeypatch.setattr(
        scheduler, "get_world_group", lambda: SimpleNamespace(barrier=barrier)
    )
    instance = scheduler.Scheduler.__new__(scheduler.Scheduler)
    instance.init_startup_timing_begin()
    now += 40
    instance.init_model_startup_timing_begin()
    now += 20
    instance.tp_worker = SimpleNamespace(
        weight_load_time=10, graph_time_usage=None, compile_timing_steps=[]
    )
    instance.draft_worker = None
    instance.kv_cache_allocation_time = 1
    instance.init_startup_timing_summary()

    assert instance.startup_time["scheduler_e2e"] == now
    if device == "tpu":
        assert barriers == [40]
        assert instance.startup_time["model_startup"] == 20
    else:
        assert barriers == []
        assert "model_startup" not in instance.startup_time
