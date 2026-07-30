"""Unit tests for TPU backend shape specialization."""

import contextlib
import pickle
import unittest
from unittest.mock import patch

import torch
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch.fx.experimental import _config as symbolic_shapes_config

from sglang.srt.hardware_backend.tpu.compilation.backend_shape_specialization import (
    _ArtifactDispatcher,
    _DoneCompiling,
    _compile_context,
    delay_aot_cache_writes,
    mark_dynamic_size,
    shape_specialization_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Artifact:
    def __init__(self, graph, compiler):
        self.graph = graph
        self.compiler = compiler

    def __call__(self, *args):
        self.compiler.artifact_executions += 1
        return self.graph(*args)


class _Compiler:
    def __init__(self):
        self.frontend_graphs = 0
        self.static_compiles = []
        self.artifact_executions = 0

    def __call__(self, graph, args, _is_fwd):
        self.static_compiles.append(dict(_compile_context.get()))
        return _Artifact(graph, self)

    def execute_pre_grad_passes(self, _graph):
        self.frontend_graphs += 1


@contextlib.contextmanager
def _use_compiler(compiler, *, enable_aot_cache=False):
    with (
        patch("torch_tpu._internal.compile.compiler.StaticCompiler"),
        patch(
            "torch_tpu._internal.compile.split_compiler.SplitCompiler",
            return_value=compiler,
        ),
        torch._functorch.config.patch(enable_autograd_cache=enable_aot_cache),
        symbolic_shapes_config.patch(backed_size_oblivious=True),
    ):
        yield


class TestBackendShapeSpecialization(unittest.TestCase):
    def test_missing_compile_context_rejects_backend_compilation(self):
        graph = torch.fx.symbolic_trace(lambda value: value.sin())
        with self.assertRaisesRegex(RuntimeError, "_compile_context"):
            shape_specialization_backend(graph, [torch.randn(4)])

    @staticmethod
    def _extend(tokens, positions, batch):
        return tokens.sin() + positions[0].cos(), batch.cos()

    @staticmethod
    def _decode(tokens, positions, batch):
        return tokens.cos() + positions[0].sin(), batch.sin()

    def _precompile(self, compiled, num_tokens, bs):
        tokens = torch.randn(num_tokens)
        positions = torch.randn(3, num_tokens)
        batch = torch.randn(bs)

        torch._dynamo.mark_dynamic(tokens, 0)
        mark_dynamic_size(positions, 1, "num_tokens")
        mark_dynamic_size(batch, 0, "bs")

        context = _compile_context.set({"num_tokens": num_tokens, "bs": bs})
        try:
            with self.assertRaises(_DoneCompiling):
                compiled(tokens, positions, batch)
        finally:
            _compile_context.reset(context)
        return tokens, positions, batch

    def test_one_frontend_graph_dispatches_static_specializations(self):
        torch._dynamo.reset()
        compiler = _Compiler()
        shape_specs = [(8, 6), (8, 4), (4, 4)]

        with _use_compiler(compiler):
            compiled = torch.compile(
                self._extend,
                backend=shape_specialization_backend,
                fullgraph=True,
                dynamic=False,
            )
            inputs = [self._precompile(compiled, *spec) for spec in shape_specs]

        expected_compiles = [
            {"num_tokens": num_tokens, "bs": bs} for num_tokens, bs in shape_specs
        ]
        self.assertEqual(compiler.frontend_graphs, 1)
        self.assertEqual(compiler.static_compiles, expected_compiles)
        self.assertEqual(compiler.artifact_executions, 0)

        for args in inputs:
            torch.testing.assert_close(compiled(*args), self._extend(*args))

        self.assertEqual(compiler.static_compiles, expected_compiles)
        self.assertEqual(compiler.artifact_executions, len(shape_specs))

    def test_defers_two_complete_aot_cache_entries(self):
        torch._dynamo.reset()
        compiler = _Compiler()
        saved_payloads = []

        def save_entry(_key, entry, remote):
            entry.pre_save()
            payload = AOTAutogradCache._pickle_entry(entry, remote)
            self.assertIsNotNone(payload)
            saved_payloads.append(payload)

        with (
            _use_compiler(compiler, enable_aot_cache=True),
            patch.object(AOTAutogradCache, "_lookup", return_value=None),
            patch.object(AOTAutogradCache, "save", new=staticmethod(save_entry)),
        ):
            compiled_extend = torch.compile(
                self._extend,
                backend=shape_specialization_backend,
                fullgraph=True,
                dynamic=False,
            )
            compiled_decode = torch.compile(
                self._decode,
                backend=shape_specialization_backend,
                fullgraph=True,
                dynamic=False,
            )

            with delay_aot_cache_writes():
                self._precompile(compiled_extend, 8, 2)
                self._precompile(compiled_extend, 4, 2)
                self._precompile(compiled_decode, 4, 4)
                self._precompile(compiled_decode, 2, 2)
                self._precompile(compiled_decode, 1, 1)
                self.assertEqual(saved_payloads, [])

        self.assertEqual(compiler.frontend_graphs, 2)
        self.assertEqual(compiler.artifact_executions, 0)
        self.assertEqual(len(saved_payloads), 2)

        dispatchers = [
            pickle.loads(payload).compiled_fw.result for payload in saved_payloads
        ]
        self.assertTrue(
            all(isinstance(item, _ArtifactDispatcher) for item in dispatchers)
        )
        artifact_keys = {frozenset(item._artifacts) for item in dispatchers}
        self.assertEqual(
            artifact_keys,
            {
                frozenset({(4, 2), (8, 2)}),
                frozenset({(1, 1), (2, 2), (4, 4)}),
            },
        )


if __name__ == "__main__":
    unittest.main()
