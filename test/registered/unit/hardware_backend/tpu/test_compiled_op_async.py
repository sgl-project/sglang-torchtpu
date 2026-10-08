"""Compilation submission, deferred execution, and serving-wrapper reuse."""

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, local
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import spec_op_precompile
from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall
from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import collect_warmups
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_async_compile_workers_preserve_graph_mode_without_serializing(monkeypatch):
    from sglang.srt.hardware_backend.tpu import patches

    class ModeState(local):
        mode = "eager"

    state = ModeState()
    barrier = Barrier(2)

    @contextmanager
    def set_mode(mode):
        previous = state.mode
        state.mode = mode
        try:
            yield
        finally:
            state.mode = previous

    class StaticCompiler:
        def _compile(self, fail=False):
            assert state.mode == "graph"
            barrier.wait(timeout=10)
            if fail:
                raise RuntimeError("compile failed")
            return "compiled"

    execution_mode = SimpleNamespace(
        EagerMode=SimpleNamespace(INTERNAL_COMPILE_FX_GRAPH="graph"),
        set_eager_mode=set_mode,
    )
    internal = ModuleType("torch_tpu._internal")
    internal.execution_mode = execution_mode
    compiler = ModuleType("torch_tpu._internal.compile.compiler")
    compiler.StaticCompiler = StaticCompiler
    monkeypatch.setitem(sys.modules, internal.__name__, internal)
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    monkeypatch.setattr(patches, "_async_compile_context_patched", False)
    patches._patch_async_compile_context()
    patched_compile = StaticCompiler._compile
    patches._patch_async_compile_context()
    assert StaticCompiler._compile is patched_compile

    def run(fail):
        state.mode = "eager"
        try:
            return StaticCompiler()._compile(fail=fail)
        finally:
            assert state.mode == "eager"

    with set_mode("caller"), ThreadPoolExecutor(max_workers=2) as pool:
        success = pool.submit(run, False)
        failure = pool.submit(run, True)
        assert success.result(timeout=15) == "compiled"
        with pytest.raises(RuntimeError, match="compile failed"):
            failure.result(timeout=15)
        assert state.mode == "caller"


@pytest.fixture
def backend(monkeypatch):
    events = []
    wrappers = []
    module = ModuleType("torch_tpu._internal.compile")

    class Submitted(Exception):
        def __init__(self, artifact):
            self.artifact = artifact

    def resolve(artifacts):
        for artifact in artifacts:
            events.append(("resolve", artifact))

    def compile_fn(fn, **options):
        shapes = set()
        wrappers.append(options)

        def call(x):
            key = (fn.__name__, x.numel())
            if key not in shapes:
                shapes.add(key)
                events.append(("submit", key))
                if options["options"]:
                    raise Submitted(key)
            events.append(("execute", key))
            return fn(x)

        return call

    module.AsyncCompilationSubmitted = Submitted
    module.resolve_compilations = resolve
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch, "compile", compile_fn)
    return SimpleNamespace(events=events, wrappers=wrappers, module=module)


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
            assert torch.equal(op(torch.arange(2)), torch.arange(2) + offset)
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


def test_pending_compile_failure_prevents_completion(backend, monkeypatch):
    registry = importlib.import_module("sglang.srt.compilation.compiled_op")
    monkeypatch.setattr(registry, "is_tpu", lambda: True)

    def fail(_):
        raise RuntimeError("native compile failed")

    monkeypatch.setattr(backend.module, "resolve_compilations", fail)
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
