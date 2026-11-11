"""direct_compile rebuilds artifacts from stored lowered graphs without tracing."""

import dataclasses
import pickle
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import zstandard
from jax import tree_util

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import direct_compile, lowered_graph, spmd_compiler
from sglang_tpu.compilation.graph_identity import graph_name

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")

_SERIALIZED = b"serialized-executable"


class _Executable(str):
    """A fake executable: its name selects outputs; every one has one fingerprint."""

    origin = None

    def fingerprint_executable(self):
        return "fp"

    def subslice_origin(self):
        return self.origin


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("bias", torch.full((2,), 3.0))


class _KVPool:
    def __init__(self):
        self.value = torch.zeros(2)

    def get_state(self):
        return [self.value]

    def set_state(self, state):
        self.value = state[0]


class _MemoryStore:
    """Requests are frozen dataclasses, so equal requests share an entry."""

    def __init__(self):
        self.graphs = {}
        self.lookups = []
        self.records = []

    def lookup(self, request):
        self.lookups.append(request)
        return self.graphs.get(request)

    def wants_record(self):
        return True

    def record(self, request, graph):
        self.records.append(graph)
        self.graphs[request] = graph


def _compiled(
    model,
    pool,
    traces,
    *,
    name="test/fn",
    callback=None,
    uses_rng=False,
    spmd_safe=False,
):
    def fn(value):
        traces.append(tuple(value.shape))
        pool.value.add_(model.bias)
        if callback is not None:
            direct_compile.insert_callback(value + 1, callback)
        return {"out": value + model.weight}

    return direct_compile.direct_compile(
        fn,
        specialization_key=lambda value: tuple(value.shape),
        model=model,
        kv_pool=pool,
        name=name,
        spmd_safe=spmd_safe,
        uses_rng=uses_rng,
    )


def _raw(serialized):
    """Stored executables use shared compilation's zstd serialization."""
    return zstandard.ZstdDecompressor().decompress(serialized)


def _shared_compiler(rank=0, size=2):
    """An SPMDCompiler without collectives, as rank ``rank`` of ``size``."""
    compiler = object.__new__(spmd_compiler.SPMDCompiler)
    compiler.group = None
    compiler.rank = rank
    compiler.size = size
    compiler.verify_spmd_safe = False
    compiler.submissions = {}
    compiler.mlirs = {}
    compiler.argument_layouts = {}
    compiler._estimate = None
    compiler._estimated_load = [0.0] * size
    compiler.trace_seconds = {}
    compiler.directory = None
    compiler._serializer = None
    return compiler


def _execute(executable, inputs):
    weight, bias, kv, value, *_ = inputs
    outputs = [value + weight, kv + bias]
    if executable.endswith("callback"):
        outputs.append(value + 1)
    return outputs


class TestDirectCompileLoweredGraph(CustomTestCase):
    @contextmanager
    def _cpu_compile(self, store, *, executable="traced", origin=None):
        compiler = direct_compile.tpu_torch_compile
        executable = _Executable(executable)
        executable.origin = origin
        pending = SimpleNamespace(get=lambda: executable)
        with ExitStack() as stack:
            enter = stack.enter_context
            enter(
                patch.object(
                    direct_compile, "current_lowered_graph_store", return_value=store
                )
            )
            enter(
                patch.object(
                    direct_compile,
                    "_is_tpu_tensor",
                    side_effect=lambda value: isinstance(value, torch.Tensor),
                )
            )
            enter(
                patch.object(
                    direct_compile.execution_mode,
                    "set_eager_mode",
                    return_value=nullcontext(),
                )
            )
            enter(patch.object(compiler, "placeholder_like", side_effect=torch.clone))
            enter(
                patch.object(
                    compiler,
                    "traverse_and_compile",
                    return_value=SimpleNamespace(executable=executable),
                )
            )
            enter(
                patch.object(
                    compiler, "async_traverse_and_compile", return_value=pending
                )
            )
            mocks = SimpleNamespace(
                execute=enter(patch.object(compiler, "execute", side_effect=_execute)),
                serialize=enter(
                    patch.object(
                        compiler, "serialize_executable", return_value=_SERIALIZED
                    )
                ),
                load=enter(
                    patch.object(
                        compiler,
                        "load_serialized_executable",
                        return_value=_Executable("reused"),
                    )
                ),
            )
            yield mocks

    def _seed(self, store):
        """Trace once, as the first boot would, and return the recorded graph."""
        traces = []
        with self._cpu_compile(store):
            _compiled(_Model(), _KVPool(), traces)(torch.full((2,), 2.0))
        self.assertEqual(traces, [(2,)])
        (graph,) = store.records
        return graph

    def test_stored_graph_replaces_tracing(self):
        for async_compile in (False, True):
            with self.subTest(async_compile=async_compile):
                store = _MemoryStore()
                graph = self._seed(store)
                self.assertEqual(_raw(graph.executable), _SERIALIZED)
                self.assertEqual(graph.fingerprint, "fp")
                interface = graph.interface
                self.assertEqual(interface.model_indices, (0, 1))
                self.assertEqual(
                    [spec.shape for spec in interface.input_specs], [(2,)] * 4
                )
                self.assertEqual(interface.argument_layouts, [[]] * 4)
                self.assertEqual(
                    interface.output_treespec, tree_util.tree_structure({"out": 0})
                )
                self.assertEqual(
                    interface.callback_factories, lowered_graph.NO_CALLBACKS
                )

                # A new process: fresh model, pool and compiled function.
                model, pool, traces = _Model(), _KVPool(), []
                compiled = _compiled(model, pool, traces)
                value = torch.full((2,), 5.0)
                with self._cpu_compile(store) as mocks:
                    if async_compile:
                        result = compiled.warmup_async(value).wait()
                    else:
                        result = compiled(value)

                self.assertEqual(traces, [])
                self.assertEqual(len(store.lookups), 2)
                self.assertEqual(len(store.records), 1)
                torch.testing.assert_close(result["out"], torch.full((2,), 6.0))
                torch.testing.assert_close(pool.value, torch.full((2,), 3.0))
                mocks.load.assert_called_once_with(_SERIALIZED, subslice_origin=None)
                executable, inputs = mocks.execute.call_args.args
                self.assertEqual(executable, "reused")
                self.assertIs(inputs[0], model.weight)
                self.assertIs(inputs[1], model.bias)
                self.assertIs(inputs[3], value)

    def test_stored_graph_loads_with_its_subslice_origin(self):
        # Serialized executables omit a subslice TP group's chip coordinates.
        store = _MemoryStore()
        with self._cpu_compile(store, origin=(1, 0, 0)):
            _compiled(_Model(), _KVPool(), [])(torch.ones(2))
        (graph,) = store.records
        self.assertEqual(graph.subslice_origin, (1, 0, 0))
        traces = []
        with self._cpu_compile(store) as mocks:
            _compiled(_Model(), _KVPool(), traces)(torch.ones(2))
        self.assertEqual(traces, [])
        mocks.load.assert_called_once_with(_SERIALIZED, subslice_origin=(1, 0, 0))

    def test_async_compiles_are_recorded_once_at_join(self):
        store, traces = _MemoryStore(), []
        compiled = _compiled(_Model(), _KVPool(), traces)
        with self._cpu_compile(store) as mocks:
            pending = compiled.warmup_async(torch.ones(2))
            self.assertEqual(store.records, [])
            pending.wait()
            compiled(torch.ones(2))
        self.assertEqual(traces, [(2,)])
        (graph,) = store.records
        self.assertEqual(_raw(graph.executable), _SERIALIZED)
        mocks.serialize.assert_called_once()

    def test_short_output_list_fails_instead_of_skipping_kv(self):
        store = _MemoryStore()
        self._seed(store)
        pool = _KVPool()
        compiled = _compiled(_Model(), pool, [])
        with self._cpu_compile(store) as mocks:
            mocks.execute.side_effect = lambda executable, inputs: _execute(
                executable, inputs
            )[:1]
            with self.assertRaisesRegex(RuntimeError, "output count changed"):
                compiled(torch.ones(2))
        torch.testing.assert_close(pool.value, torch.zeros(2))

    def test_unmaterialized_buffers_key_with_their_default_layout(self):
        compiler = direct_compile.tpu_torch_compile
        default = ([1, 0], [[8, 128], [2, 1]], 0)
        tensor = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), dtype=torch.bfloat16, shape=(64, 2048)
        )
        with (
            patch.object(compiler, "get_default_layout", return_value=default),
            patch.object(
                compiler, "get_device_layout_if_materialized", return_value=None
            ),
        ):
            unmaterialized = direct_compile._keyed_layout(tensor)
        with patch.object(
            compiler, "get_device_layout_if_materialized", return_value=default
        ):
            materialized = direct_compile._keyed_layout(tensor)
        self.assertEqual(unmaterialized, materialized)
        self.assertIsNone(direct_compile._keyed_layout(torch.ones(2)))

    def test_store_requires_name(self):
        store, traces = _MemoryStore(), []
        with self._cpu_compile(store):
            _compiled(_Model(), _KVPool(), traces, name=None)(torch.ones(2))
        self.assertEqual((traces, store.lookups, store.records), ([(2,)], [], []))

    def test_key_without_identity_is_traced_and_not_stored(self):
        store, traces = _MemoryStore(), []
        unencodable = TypeError("unencodable key")
        with (
            self._cpu_compile(store),
            patch.object(direct_compile, "graph_name", side_effect=unencodable),
        ):
            _compiled(_Model(), _KVPool(), traces)(torch.ones(2))
        self.assertEqual((traces, store.lookups, store.records), ([(2,)], [], []))

    def test_rng_functions_are_always_traced(self):
        # Stored graphs carry no generator-state input or output.
        store, traces = _MemoryStore(), []
        state = torch.zeros(1)
        compiler = direct_compile.tpu_torch_compile
        tpu = SimpleNamespace(default_generators=[object()])

        def execute(executable, inputs):
            return [*_execute(executable, inputs), inputs[-1] + 1]

        with ExitStack() as stack:
            enter = stack.enter_context
            mocks = enter(self._cpu_compile(store))
            mocks.execute.side_effect = execute
            enter(patch.object(torch, "tpu", tpu, create=True))
            for name in (
                "make_export_mode_device_generator",
                "set_export_mode_default_device_generator",
                "clear_export_mode_default_device_generators",
                "set_device_state_tensor",
            ):
                enter(patch.object(compiler, name, create=True))
            enter(
                patch.object(
                    compiler, "get_device_state_tensor", create=True, return_value=state
                )
            )
            enter(
                patch.object(
                    compiler,
                    "MultiGeneratorLocker",
                    create=True,
                    side_effect=lambda generators: nullcontext(),
                )
            )
            compiled = _compiled(_Model(), _KVPool(), traces, uses_rng=True)
            result = compiled(torch.ones(2))
        torch.testing.assert_close(result["out"], torch.full((2,), 2.0))
        self.assertEqual((traces, store.lookups, store.records), ([(2,)], [], []))

    def test_graphs_with_callbacks_are_not_recorded(self):
        store, traces, seen = _MemoryStore(), [], []
        with self._cpu_compile(store, executable="traced-callback"):
            _compiled(_Model(), _KVPool(), traces, callback=seen.append)(torch.ones(2))
        self.assertEqual(len(seen), 1)
        self.assertEqual(store.records, [])

    def test_unusable_entries_log_an_error_and_fall_back_to_tracing(self):
        def failing_lookup(store):
            def lookup(request):
                raise OSError("unreadable")

            store.lookup = lookup

        def damage_interface(**changes):
            def damage(store):
                for request, graph in store.graphs.items():
                    interface = dataclasses.replace(graph.interface, **changes)
                    store.graphs[request] = dataclasses.replace(
                        graph, interface=interface
                    )

            return damage

        def wrong_input_count(store):
            for graph in store.graphs.values():
                specs = graph.interface.input_specs[:3]
            damage_interface(input_specs=specs)(store)

        cases = {
            "lookup-error": failing_lookup,
            "input-count": wrong_input_count,
            "retained-inputs": damage_interface(model_indices=(0,)),
            # The checks a shared-compile consumer applies to its owner's
            # interface: input order, and physical layouts.
            "input-digest": damage_interface(input_digest="other"),
            "layouts": damage_interface(argument_layouts=[[0]] * 4),
        }
        for name, damage in cases.items():
            with self.subTest(name):
                store = _MemoryStore()
                self._seed(store)
                damage(store)
                traces = []
                with (
                    self._cpu_compile(store) as mocks,
                    self.assertLogs(direct_compile.logger, "ERROR") as logs,
                ):
                    result = _compiled(_Model(), _KVPool(), traces)(torch.ones(2))
                self.assertEqual(traces, [(2,)])
                torch.testing.assert_close(result["out"], torch.full((2,), 2.0))
                self.assertEqual(mocks.execute.call_args.args[0], "traced")
                self.assertIn("test/fn", logs.output[0])

        store = _MemoryStore()
        self._seed(store)
        failures = {
            "load": RuntimeError("incompatible executable"),
            "fingerprint": _Executable("other"),
        }
        for name, failure in failures.items():
            with self.subTest(name), self._cpu_compile(store) as mocks:
                if isinstance(failure, Exception):
                    mocks.load.side_effect = failure
                else:
                    mocks.load.return_value = failure
                    failure.fingerprint_executable = lambda: "other-fp"
                traces = []
                _compiled(_Model(), _KVPool(), traces)(torch.ones(2))
                self.assertEqual(traces, [(2,)])
                self.assertEqual(mocks.execute.call_args.args[0], "traced")

    def test_recording_errors_never_fail_the_compile(self):
        for error in (OSError("disk full"), RuntimeError("store failed")):
            with self.subTest(error=type(error).__name__):
                store = _MemoryStore()

                def record(request, graph, error=error):
                    raise error

                store.record = record
                compiled = _compiled(_Model(), _KVPool(), [])
                with self._cpu_compile(store):
                    result = compiled(torch.ones(2))
                torch.testing.assert_close(result["out"], torch.full((2,), 2.0))

    @contextmanager
    def _shared(self, store, compiler):
        with ExitStack() as stack:
            enter = stack.enter_context
            mocks = enter(self._cpu_compile(store))
            enter(
                patch.object(
                    direct_compile,
                    "current_warmup_queue",
                    return_value=SimpleNamespace(spmd_compiler=compiler, verify=False),
                )
            )
            enter(patch.object(direct_compile.sync, "synchronize"))
            yield mocks

    def test_shared_compile_owner_reuses_and_distributes_a_stored_graph(self):
        store = _MemoryStore()
        self._seed(store)
        compiler = _shared_compiler(rank=0)
        traces = []
        compiled = _compiled(_Model(), _KVPool(), traces, spmd_safe=True)
        with self._shared(store, compiler) as mocks:
            pending = compiled.warmup_async(torch.ones(2))
            (submission,) = compiler.submissions.values()
            self.assertEqual(traces, [])
            mocks.load.assert_called_once_with(_SERIALIZED, subslice_origin=None)
            # Distribution serializes the owner's executable as for a compile.
            executable, serialized = spmd_compiler._prepare_executable(
                submission, serialize=True
            )
            self.assertEqual(executable, "reused")
            self.assertEqual(_raw(serialized), _SERIALIZED)
            submission.executable = executable
            result = pending.wait()
        interface = submission.interface
        self.assertEqual(interface.model_indices, (0, 1))
        self.assertEqual(
            interface.output_treespec, tree_util.tree_structure({"out": 0})
        )
        self.assertEqual(pickle.loads(interface.callback_factories), ())
        self.assertEqual(len(interface.input_specs), 4)
        torch.testing.assert_close(result["out"], torch.full((2,), 2.0))
        self.assertEqual(len(store.records), 1)

    def test_shared_compile_owner_records_the_distribution_bytes(self):
        store, traces = _MemoryStore(), []
        compiler = _shared_compiler(rank=0)
        compiled = _compiled(_Model(), _KVPool(), traces, spmd_safe=True)
        with self._shared(store, compiler) as mocks:
            compiled.warmup_async(torch.ones(2))
            (submission,) = compiler.submissions.values()
            self.assertEqual(traces, [(2,)])
            self.assertEqual(store.records, [])
            _, serialized = spmd_compiler._prepare_executable(
                submission, serialize=True
            )
            mocks.serialize.assert_called_once()
        (graph,) = store.records
        self.assertIs(graph.executable, serialized)
        self.assertEqual(_raw(graph.executable), _SERIALIZED)

    def test_shared_compile_consumers_never_consult_the_store(self):
        store = _MemoryStore()
        self._seed(store)
        lookups = len(store.lookups)
        compiler = _shared_compiler(rank=1)
        traces = []
        compiled = _compiled(_Model(), _KVPool(), traces, spmd_safe=True)
        with self._shared(store, compiler):
            compiled.warmup_async(torch.ones(2))
        (submission,) = compiler.submissions.values()
        self.assertEqual(submission.owner, 0)
        self.assertIsNotNone(submission.bind)
        self.assertEqual((traces, len(store.lookups)), ([], lookups))

    def test_store_requests_use_the_shared_compile_graph_identity(self):
        store = _MemoryStore()
        self._seed(store)
        (request,) = store.lookups
        self.assertEqual(request.name, "test/fn")
        self.assertEqual(request.graph, graph_name("test/fn", (2,)))
        self.assertEqual(request.graph, "test/fn:[2]")
        self.assertEqual(
            [spec for _, spec in request.model_inputs],
            [spmd_compiler.InputSpec((2,), "torch.float32", (1,), 0)] * 2,
        )

        # The owner of a shared graph looks up the string the gang agreed on.
        compiler = _shared_compiler(rank=0)
        compiled = _compiled(_Model(), _KVPool(), [], spmd_safe=True)
        with self._shared(store, compiler):
            compiled.warmup_async(torch.ones(2))
        self.assertEqual(list(compiler.submissions), [store.lookups[-1].graph])
        self.assertEqual(store.lookups[-1], request)

    def test_graph_identity_falls_back_to_repr_for_non_json_keys(self):
        self.assertEqual(
            graph_name("helper", (torch.float32, (2, 3), None)),
            'helper:["torch.float32",[2,3],null]',
        )
        device = torch.device("cpu")
        self.assertEqual(graph_name("helper", (device,)), f'helper:["{device!r}"]')
        # A repr with a memory address would differ in every process.
        with self.assertRaisesRegex(TypeError, "no stable identity"):
            graph_name("helper", (object(),))

    def test_store_hits_and_consumers_share_one_interface_check(self):
        store = _MemoryStore()
        interface = self._seed(store).interface
        compiler = _shared_compiler(rank=1)
        compiled = _compiled(_Model(), _KVPool(), [], spmd_safe=True)
        with self._shared(store, compiler):
            compiled.warmup_async(torch.ones(2))
        (submission,) = compiler.submissions.values()
        # A stored interface binds on a consumer like its owner's would.
        submission.bind(interface)
        changes = {
            "digest": dict(input_digest="other"),
            "specs": dict(input_specs=interface.input_specs[:3]),
            "layouts": dict(argument_layouts=[[0]] * 4),
        }
        for field, change in changes.items():
            with self.subTest(field):
                with self.assertRaisesRegex(
                    ValueError, f"owner_{field}=.*local_{field}=.*owner=0, rank=1"
                ):
                    submission.bind(dataclasses.replace(interface, **change))

    def test_compiled_for_default_layout_matches_a_default_layout_buffer(self):
        compiler = direct_compile.tpu_torch_compile
        default = ([1, 0], [[8, 128], [2, 1]], 0)
        tensor = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), dtype=torch.bfloat16, shape=(64, 2048)
        )
        with patch.object(compiler, "get_default_layout", return_value=default):
            for materialized, expected, matches in (
                (default, [], True),
                (None, [1, 0], True),
                (None, [], True),
                (([0, 1], [], 0), [], False),
                (([0, 1], [], 0), [1, 0], False),
            ):
                with patch.object(
                    compiler,
                    "get_device_layout_if_materialized",
                    return_value=materialized,
                ):
                    self.assertEqual(
                        direct_compile._layouts_match([expected], [tensor]), matches
                    )
        self.assertTrue(direct_compile._layouts_match([[]], [torch.ones(2)]))
        self.assertFalse(direct_compile._layouts_match([[0]], [torch.ones(2)]))
        self.assertFalse(direct_compile._layouts_match([], [torch.ones(2)]))


if __name__ == "__main__":
    unittest.main()
