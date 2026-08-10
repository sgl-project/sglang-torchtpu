"""Tests for direct TPU compilation and artifact dispatch."""

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import patch

import torch
from jax import tree_util

from sglang.srt.hardware_backend.tpu.compilation import direct_compile
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


@tree_util.register_dataclass
@dataclass
class _Inputs:
    value: torch.Tensor
    ignored: torch.Tensor
    metadata: object


@tree_util.register_dataclass
@dataclass
class _Result:
    value: torch.Tensor


class _ModelState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("bias", torch.ones(2))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = _ModelState()
        self.state_alias = self.state


class _KVPool:
    def __init__(self, value):
        self.value = value

    def get_state(self):
        return [self.value]

    def set_state(self, state):
        self.value = state[0]


class TestDirectCompile(CustomTestCase):
    def test_artifact_dispatches_and_rebuilds_outputs(self):
        model = _Model()
        kv_pool = _KVPool(torch.zeros(2))
        pool_input = kv_pool.value
        traced_model_state = []
        traced_pool_state = []
        traced_compile_state = []

        def fn(*, inputs):
            traced_compile_state.append(torch.compiler.is_compiling())
            traced_model_state.append((model.state.weight, model.state_alias.weight))
            traced_pool_state.append(kv_pool.value)
            kv_pool.value.add_(model.state.weight)
            return _Result(
                inputs.value
                + model.state.weight
                + model.state_alias.weight
                + model.state.bias
            )

        compiled = direct_compile.direct_compile(
            fn,
            specialization_key=lambda *, inputs: tuple(inputs.value.shape),
            model=model,
            kv_pool=kv_pool,
            ignored_argnames=("inputs.ignored",),
        )
        self.assertIsInstance(compiled, direct_compile.CompiledFunction)

        first = _Inputs(
            value=torch.full((2,), 2.0),
            ignored=torch.full((2,), 100.0),
            metadata="warmup",
        )
        second = _Inputs(
            value=torch.full((2,), 3.0),
            ignored=torch.full((2,), 200.0),
            metadata=["live", "structure"],
        )
        first_result = torch.full((2,), 10.0)
        first_pool = torch.full((2,), 12.0)
        second_result = torch.full((2,), 20.0)
        second_pool = torch.full((2,), 22.0)

        compiler = direct_compile.tpu_torch_compile
        with (
            patch.object(
                direct_compile,
                "_is_tpu_tensor",
                side_effect=lambda value: (
                    isinstance(value, torch.Tensor)
                    and all(
                        value is not ignored
                        for ignored in (first.ignored, second.ignored)
                    )
                ),
            ),
            patch.object(
                direct_compile.execution_mode,
                "set_eager_mode",
                return_value=nullcontext(),
            ),
            patch.object(
                compiler, "placeholder_like", side_effect=lambda tensor: tensor.clone()
            ),
            patch.object(compiler, "build_mlir", return_value="module") as build_mlir,
            patch.object(
                compiler, "compile_mlir", return_value="executable"
            ) as compile_mlir,
            patch.object(
                compiler,
                "execute",
                side_effect=(
                    [first_result, first_pool],
                    [second_result, second_pool],
                ),
            ) as execute,
        ):
            output = compiled(inputs=first)
            self.assertFalse(torch.compiler.is_compiling())
            self.assertTrue(torch.equal(output.value, first_result))
            self.assertTrue(torch.equal(kv_pool.value, first_pool))

            output = compiled(inputs=second)
            self.assertTrue(torch.equal(output.value, second_result))
            self.assertIs(kv_pool.value, pool_input)
            self.assertTrue(torch.equal(kv_pool.value, second_pool))

        self.assertEqual(len(compiled.cache), 1)
        build_mlir.assert_called_once()
        compile_mlir.assert_called_once_with("module")
        runtime_inputs = execute.call_args_list[1].args[1]
        self.assertEqual(len(runtime_inputs), 4)
        self.assertIs(runtime_inputs[0], model.state.weight)
        self.assertIs(runtime_inputs[1], model.state.bias)
        self.assertIs(runtime_inputs[2], pool_input)
        self.assertIs(runtime_inputs[3], second.value)
        self.assertIs(traced_model_state[0][0], traced_model_state[0][1])
        self.assertIsNot(traced_model_state[0][0], model.state.weight)
        self.assertIsNot(traced_pool_state[0], pool_input)
        self.assertEqual(traced_compile_state, [True])

        unexpected = torch.ones(2)
        invalid = direct_compile.direct_compile(
            lambda value, metadata: value,
            specialization_key=lambda *_args: None,
        )
        with (
            patch.object(
                direct_compile,
                "_is_tpu_tensor",
                side_effect=lambda value: (
                    isinstance(value, torch.Tensor) and value is not unexpected
                ),
            ),
            self.assertRaisesRegex(TypeError, "Unsupported.*Tensor"),
        ):
            invalid(torch.ones(2), unexpected)


if __name__ == "__main__":
    unittest.main()
