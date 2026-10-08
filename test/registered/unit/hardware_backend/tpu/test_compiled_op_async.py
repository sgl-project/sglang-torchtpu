"""Compilation submission, deferred execution, and serving-wrapper reuse."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import spec_op_precompile
from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall
from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import collect_warmups
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def backend(monkeypatch):
    events = []
    wrappers = []
    module = ModuleType("sglang.srt.hardware_backend.tpu.compilation.direct_compile")

    class Compiled:
        def __init__(self, fn, specialization_key, **options):
            self.fn = fn
            self.key = specialization_key
            self.cache = set()
            wrappers.append(options)

        def __call__(self, x):
            key = (self.fn.__name__, x.numel())
            events.append(("execute", key))
            return self.fn(x)

        def warmup_async(self, x):
            key = self.key(x)
            if key not in self.cache:
                self.cache.add(key)
                events.append(("submit", (self.fn.__name__, x.numel())))
            return PendingCall(lambda: module.finish(self, x))

    module.direct_compile = Compiled
    module.finish = lambda compiled, x: compiled(x)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with get_context().override_server_args(tpu_compile_tracer="direct"):
        yield SimpleNamespace(events=events, wrappers=wrappers, module=module)


def test_registered_ops_submit_before_join_and_reuse_serving_wrapper(
    backend, monkeypatch
):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "_compiled_ops", {})
    monkeypatch.setattr(registry, "is_tpu", lambda: True)
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda: backend.events.append("sync"),
    )
    monkeypatch.setattr(
        spec_op_precompile,
        "snapshot_recompilation_baseline",
        lambda: backend.events.append("snapshot"),
    )

    @registry.compiled_op(warmup=lambda ctx: (torch.zeros(ctx.physical_bs),))
    def add_one(x):
        return x + 1

    @registry.compiled_op(warmup=lambda ctx: (torch.zeros(ctx.physical_bs),))
    def add_two(x):
        return x + 2

    with (
        envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
    ):
        with collect_warmups() as queue:
            queue.calls.append(
                PendingCall(lambda: backend.events.append("model execute"))
            )
            spec_op_precompile.precompile_spec_v2_ops(
                device=SimpleNamespace(type="tpu"),
                physical_batch_buckets=(1, 2),
                num_verify_tokens=3,
                vocab_size=8,
                draft_vocab_size=8,
                hidden_size=4,
                hidden_dtype=torch.float32,
            )
            assert backend.events == [
                ("submit", ("add_one", 1)),
                ("submit", ("add_two", 1)),
                ("submit", ("add_one", 2)),
                ("submit", ("add_two", 2)),
            ]
        assert backend.events[4] == "model execute"
        assert backend.events[-2:] == ["sync", "snapshot"]
        assert len(backend.wrappers) == 2
        for op, offset in ((add_one, 1), (add_two, 2)):
            assert torch.equal(
                op(torch.arange(2, dtype=torch.float32)), torch.arange(2) + offset
            )
        assert (
            sum(
                event[0] == "submit"
                for event in backend.events
                if isinstance(event, tuple)
            )
            == 4
        )


def test_cold_call_and_synchronous_mode_return_real_results(backend, monkeypatch):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "is_tpu", lambda: True)
    for asynchronous in (False, True):
        with (
            envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
            envs.SGLANG_DIRECT_COMPILE_ASYNC.override(asynchronous),
        ):
            op = compiled_op()(lambda x: x * 3)
            x = torch.arange(3)
            with collect_warmups() as queue:
                assert torch.equal(op(x), x * 3)
                assert queue.calls == []
            assert warmup_compiled_op(op, x) is None
            assert torch.equal(op(x), x * 3)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_default_dynamo_warmup_runs_inline_and_reuses_serving_graph(
    backend, monkeypatch, asynchronous
):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "is_tpu", lambda: True)
    original_compile = torch.compile
    wrappers = []
    graphs = []

    def record_graph(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_on_cpu(fn, **options):
        wrappers.append(options)
        return original_compile(fn, backend=record_graph, fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile_on_cpu)
    op = compiled_op()(lambda x: x * 3)
    x = torch.arange(3)
    with (
        get_context().override_server_args(),
        envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        envs.SGLANG_DIRECT_COMPILE_ASYNC.override(asynchronous),
    ):
        with collect_warmups() as queue:
            warmup_compiled_op(op, x)
            assert len(graphs) == 1
            assert queue.calls == []
        with torch.compiler.set_stance("fail_on_recompile"):
            assert torch.equal(op(x + 1), (x + 1) * 3)
    assert wrappers == [dict(backend="tpu", fullgraph=True, dynamic=False)]
    assert len(graphs) == 1
    assert backend.wrappers == []


def test_pending_compile_failure_prevents_completion(backend, monkeypatch):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "is_tpu", lambda: True)

    def fail(*_):
        raise RuntimeError("native compile failed")

    monkeypatch.setattr(backend.module, "finish", fail)
    op = compiled_op()(lambda x: x + 1)
    with (
        envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
        pytest.raises(RuntimeError, match="native compile failed"),
    ):
        with collect_warmups() as queue:
            warmup_compiled_op(op, torch.ones(2))
            queue.completions.append(lambda: backend.events.append("completed"))
    assert "completed" not in backend.events


def test_disabled_compilation_uses_original_helper(backend, monkeypatch):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "_compiled_ops", {})
    monkeypatch.setattr(registry, "is_tpu", lambda: True)
    with envs.SGLANG_ENABLE_TORCH_COMPILE.override(False):

        @registry.compiled_op(warmup=lambda _: (torch.ones(2),))
        def op(x):
            return x + 4

        assert torch.equal(op(torch.ones(2)), torch.full((2,), 5.0))
        (registered,) = registry.get_compiled_ops()
        assert warmup_compiled_op(registered.op, torch.ones(2)) is None
        assert backend.wrappers == []


def test_owner_warmups_share_serving_callable_without_global_registration(
    backend, monkeypatch
):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "_compiled_ops", {})
    monkeypatch.setattr(registry, "is_tpu", lambda: True)

    def make_op(offset):
        @registry.compiled_op()
        def add(x):
            return x + offset

        return add

    first, second = make_op(1), make_op(2)
    with (
        envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
    ):
        with collect_warmups() as queue:
            warmup_compiled_op(first, torch.zeros(2))
            warmup_compiled_op(first, torch.ones(2))
            warmup_compiled_op(second, torch.zeros(3))
            assert len(queue.calls) == 2
            assert not any(event[0] == "execute" for event in backend.events)
        assert torch.equal(first(torch.ones(2)), torch.full((2,), 2.0))
        assert torch.equal(second(torch.ones(3)), torch.full((3,), 3.0))
        warmup_compiled_op(first, torch.zeros(4))
    assert len(backend.wrappers) == 2
    assert registry.get_compiled_ops() == ()


def test_compiled_helper_inlines_inside_outer_dynamo_graph():
    @compiled_op()
    def add_one(x):
        return x + 1

    def outer(x):
        return add_one(x) * 2

    compiled = torch.compile(outer, backend="eager", fullgraph=True)
    x = torch.arange(4)
    assert torch.equal(compiled(x), (x + 1) * 2)


@pytest.mark.parametrize(
    "changed",
    [
        (torch.ones(3), 2, True),
        (torch.ones(2, dtype=torch.int32), 2, True),
        (torch.ones(2), 3, True),
        (torch.ones(2), 2, False),
        (torch.ones(2), 2, None),
    ],
)
def test_helper_key_covers_tensor_metadata_and_static_values(changed):
    from sglang.srt.compilation.compiled_op import _specialization_key

    key = _specialization_key(torch.zeros(2), 2, True)
    assert key == _specialization_key(torch.ones(2), 2, True)
    assert key != _specialization_key(*changed)
    assert key != _specialization_key(torch.zeros(2), 2, flag=True)


@pytest.mark.parametrize("shape", [(1, 4), (3, 4)])
def test_helper_normalizes_views_without_copying_canonical_inputs(shape):
    from sglang.srt.compilation.compiled_op import _canonical_tensor

    canonical = torch.arange(shape[0] * shape[1]).reshape(shape)
    assert _canonical_tensor(canonical) is canonical
    backing = torch.arange((shape[0] + 1) * shape[1] * 2).reshape(-1, shape[1] * 2)
    view = backing[1:, ::2]
    normalized = _canonical_tensor(view)
    assert torch.equal(normalized, view)
    assert normalized.storage_offset() == 0
    assert normalized.stride() == canonical.stride()
    assert _canonical_tensor(normalized) is normalized


def test_scheduler_submits_helpers_before_model_join(monkeypatch):
    from sglang.srt.hardware_backend.tpu.compilation import spec_decode_compile_policy
    from sglang.srt.managers import scheduler

    events = []
    policy = SimpleNamespace(
        b_bucketer=SimpleNamespace(buckets=(1, 2)), draft_token_num=3
    )
    monkeypatch.setattr(
        spec_decode_compile_policy, "BaseSpecDecodeCompilePolicyMixin", SimpleNamespace
    )
    monkeypatch.setattr(
        scheduler, "current_platform", SimpleNamespace(is_tpu=lambda: True)
    )
    monkeypatch.setattr(
        scheduler, "get_draft_recurrent_hidden_state_spec", lambda _: (4, torch.float32)
    )

    def submit_model():
        from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import (
            current_warmup_queue,
        )

        events.append("model submit")
        current_warmup_queue().calls.append(
            PendingCall(lambda: events.append("model execute"))
        )

    def submit_helpers(**kwargs):
        from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import (
            current_warmup_queue,
        )

        assert kwargs["device"] == torch.device("tpu")
        events.append("helper submit")
        current_warmup_queue().calls.append(
            PendingCall(lambda: events.append("helper execute"))
        )

    monkeypatch.setattr(spec_op_precompile, "precompile_spec_v2_ops", submit_helpers)
    # Scheduler.device is assigned only after init_all_cuda_graphs returns.
    instance = SimpleNamespace(
        server_args=SimpleNamespace(
            unsafe_save_torch_compile_bundle=None, unsafe_load_torch_compile_bundle=None
        ),
        tp_worker=SimpleNamespace(
            init_cuda_graphs=submit_model,
            model_runner=SimpleNamespace(device="tpu", compile_policy=policy),
        ),
        draft_worker=SimpleNamespace(
            init_cuda_graphs=submit_model,
            draft_worker=SimpleNamespace(
                draft_runner=SimpleNamespace(
                    model=SimpleNamespace(config=SimpleNamespace(vocab_size=8))
                )
            ),
        ),
        model_config=SimpleNamespace(vocab_size=8),
    )
    instance.precompile_tpu_spec_ops = lambda: (
        scheduler.Scheduler.precompile_tpu_spec_ops(instance)
    )
    with envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True):
        scheduler.Scheduler.init_all_cuda_graphs(instance)
    assert events == [
        "model submit",
        "model submit",
        "helper submit",
        "model execute",
        "model execute",
        "helper execute",
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
