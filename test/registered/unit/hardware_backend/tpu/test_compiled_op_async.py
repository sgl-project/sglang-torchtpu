"""Compilation submission, deferred execution, and serving-wrapper reuse."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import spec_op_precompile
from sglang.srt.hardware_backend.tpu.compilation.dynamo_compile import (
    DynamoCompiledFunction,
)
from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall
from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import collect_warmups
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def test_cold_call_and_synchronous_mode_return_real_results(backend):
    for asynchronous in (False, True):
        op = DynamoCompiledFunction(lambda x: x * 3, async_compile=asynchronous)
        x = torch.arange(3)
        assert torch.equal(op(x), x * 3)
        assert op.warmup_async(x) is None
        assert torch.equal(op(x), x * 3)


def test_pending_compile_failure_prevents_completion(backend, monkeypatch):
    def fail(_):
        raise RuntimeError("native compile failed")

    monkeypatch.setattr(backend.module, "resolve_compilations", fail)
    op = DynamoCompiledFunction(lambda x: x + 1, async_compile=True)
    with pytest.raises(RuntimeError, match="native compile failed"):
        with collect_warmups() as queue:
            queue.calls.append(op.warmup_async(torch.ones(2)))
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
        assert registered.warmup_async(torch.ones(2)) is None
        assert backend.wrappers == []


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
