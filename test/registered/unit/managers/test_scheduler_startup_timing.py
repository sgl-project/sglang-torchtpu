import sys
from types import SimpleNamespace

import pytest

from sglang.srt.managers import scheduler
from sglang.srt.model_executor import model_runner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("device,peer_wait", [("tpu", 0), ("tpu", 300), ("cuda", 300)])
def test_post_sync_startup_excludes_existing_pre_load_sync(
    monkeypatch, device, peer_wait
):
    now = 0.0
    memory_checks = []

    def measure_pre_model_load_memory(*, device, is_draft_worker):
        nonlocal now
        memory_checks.append((device, is_draft_worker))
        if not is_draft_worker:
            now += peer_wait
        return 100.0

    monkeypatch.setattr(scheduler.time, "perf_counter", lambda: now)
    monkeypatch.setattr(
        model_runner.bootstrap,
        "measure_pre_model_load_memory",
        measure_pre_model_load_memory,
    )
    monkeypatch.setattr(
        model_runner, "get_parallel", lambda: SimpleNamespace(pp_group=None)
    )
    instance = scheduler.Scheduler.__new__(scheduler.Scheduler)
    instance.init_startup_timing_begin()
    instance.distributed_init_time = 30
    now += 40

    target = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
    target.device = device
    target.is_draft_worker = False
    target.init_startup_observability()
    target.init_torch_distributed()
    now += 10

    draft = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
    draft.device = device
    draft.is_draft_worker = True
    draft.init_startup_observability()
    draft.init_torch_distributed()
    assert draft.post_sync_startup_begin is None
    now += 10

    instance.tp_worker = SimpleNamespace(
        model_runner=target,
        weight_load_time=10,
        graph_time_usage=None,
        compile_timing_steps=[],
    )
    instance.draft_worker = SimpleNamespace(
        model_runner=draft,
        weight_load_time=5,
        graph_time_usage=None,
        compile_timing_steps=[],
    )
    instance.kv_cache_allocation_time = 1
    instance.init_startup_timing_summary()

    assert memory_checks == [(device, False), (device, True)]
    assert instance.startup_time["scheduler_e2e"] == now
    assert instance.startup_time["distributed_init"] == 30
    if device == "tpu":
        assert instance.startup_time["post_sync_startup"] == 20
    else:
        assert target.post_sync_startup_begin is None
        assert "post_sync_startup" not in instance.startup_time


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
