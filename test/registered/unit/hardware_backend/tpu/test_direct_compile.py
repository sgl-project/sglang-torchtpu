"""Tests for direct TPU compilation and artifact dispatch."""

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from jax import tree_util

from sglang.srt.hardware_backend.tpu.compilation import direct_compile
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit")

_COMPILE_RESULT = SimpleNamespace(executable="executable")


def _make_artifact(traced_shapes):
    """A minimal single-input artifact reading its tensor from args[0][0]."""
    return direct_compile._CompiledArtifact(
        executable=object(),
        model_inputs=(),
        kv_pool=None,
        tensor_inputs_key_paths=((tree_util.SequenceKey(0), tree_util.SequenceKey(0)),),
        tensor_inputs_traced_shapes=traced_shapes,
        output_treespec=tree_util.tree_structure(()),
        output_count=0,
    )


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


class _ModelWithRegisteredAlias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("weight_alias", self.weight, persistent=False)
        self.register_buffer("second_weight_alias", self.weight, persistent=False)


class _ObjectWithTensorState:
    def __init__(self, value, weight_alias):
        self.value = value
        self.weight_alias = weight_alias


class _ModelWithObjectTensorState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.object_state = _ObjectWithTensorState(torch.full((2,), 2.0), self.weight)


class _KVPool:
    def __init__(self, value):
        self.value = value

    def get_state(self):
        return [self.value]

    def set_state(self, state):
        self.value = state[0]


class TestDirectCompile(CustomTestCase):
    def test_fail_on_recompile_stance(self):
        compiled = direct_compile.direct_compile(
            lambda value: value,
            specialization_key=lambda value: value,
        )
        artifact = MagicMock(return_value="output")

        with (
            patch.object(compiled, "_compile", return_value=artifact) as compile_fn,
            torch.compiler.set_stance("fail_on_recompile"),
        ):
            self.assertEqual(compiled("warmup"), "output")
            self.assertEqual(compiled("warmup"), "output")
            with self.assertRaisesRegex(RuntimeError, "Detected recompile"):
                compiled("new specialization")

        compile_fn.assert_called_once()

    @unittest.skipUnless(torch.tpu.is_available(), "requires TPU")
    def test_torch_tpu_cache_hit_for_equivalent_compilations(self):
        def fn(value):
            return value + value

        def new_compiled_fn():
            return direct_compile.direct_compile(
                fn,
                specialization_key=lambda value: tuple(value.shape),
            )

        value = torch.ones(2, device="tpu")
        expected = torch.full((2,), 2.0)
        torch.tpu._clear_cache()
        requests_before = torch.tpu._get_cache_requests()
        hits_before = torch.tpu._get_cache_hits()
        misses_before = torch.tpu._get_cache_misses()

        torch.testing.assert_close(new_compiled_fn()(value).cpu(), expected)
        self.assertEqual(torch.tpu._get_cache_requests(), requests_before + 1)
        self.assertEqual(torch.tpu._get_cache_hits(), hits_before)
        self.assertEqual(torch.tpu._get_cache_misses(), misses_before + 1)

        requests_before = torch.tpu._get_cache_requests()
        hits_before = torch.tpu._get_cache_hits()
        misses_before = torch.tpu._get_cache_misses()

        torch.testing.assert_close(new_compiled_fn()(value).cpu(), expected)
        self.assertEqual(torch.tpu._get_cache_requests(), requests_before + 1)
        self.assertEqual(torch.tpu._get_cache_hits(), hits_before + 1)
        self.assertEqual(torch.tpu._get_cache_misses(), misses_before)

    def test_model_extra_tensor_attrs_trace_user_object_state_and_aliases(self):
        model = _ModelWithObjectTensorState()
        original_value = model.object_state.value
        original_alias = model.object_state.weight_alias
        traced_state = []

        def fn(value):
            traced_state.append(
                (
                    model.weight,
                    model.object_state.value,
                    model.object_state.weight_alias,
                )
            )
            return (
                value
                + model.weight
                + model.object_state.value
                + model.object_state.weight_alias
            )

        compiled = direct_compile.direct_compile(
            fn,
            specialization_key=lambda value: tuple(value.shape),
            model=model,
            model_extra_tensor_attrs=(
                "object_state.value",
                "object_state.weight_alias",
            ),
        )
        compiler = direct_compile.tpu_torch_compile
        value = torch.zeros(2)
        result = torch.full((2,), 6.0)
        with (
            patch.object(
                direct_compile,
                "_is_tpu_tensor",
                side_effect=lambda item: isinstance(item, torch.Tensor),
            ),
            patch.object(
                direct_compile.execution_mode,
                "set_eager_mode",
                return_value=nullcontext(),
            ),
            patch.object(
                compiler, "placeholder_like", side_effect=lambda tensor: tensor.clone()
            ),
            patch.object(
                compiler, "traverse_and_compile", return_value=_COMPILE_RESULT
            ),
            patch.object(compiler, "execute", return_value=[result]) as execute,
        ):
            output = compiled(value)

        self.assertTrue(torch.equal(output, result))
        traced_weight, traced_value, traced_alias = traced_state[0]
        self.assertIsNot(traced_weight, model.weight)
        self.assertIsNot(traced_value, original_value)
        self.assertIs(traced_alias, traced_weight)
        self.assertIs(model.object_state.value, original_value)
        self.assertIs(model.object_state.weight_alias, original_alias)
        runtime_inputs = execute.call_args.args[1]
        self.assertEqual(len(runtime_inputs), 3)
        self.assertIs(runtime_inputs[0], model.weight)
        self.assertIs(runtime_inputs[1], original_value)
        self.assertIs(runtime_inputs[2], value)

    def test_model_extra_tensor_attrs_require_model_and_tensor_values(self):
        with self.assertRaisesRegex(ValueError, "requires model"):
            direct_compile.direct_compile(
                lambda value: value,
                specialization_key=lambda value: tuple(value.shape),
                model_extra_tensor_attrs=("state.value",),
            )

        model = _ModelWithObjectTensorState()
        model.object_state.not_a_tensor = "static"
        compiled = direct_compile.direct_compile(
            lambda value: value,
            specialization_key=lambda value: tuple(value.shape),
            model=model,
            model_extra_tensor_attrs=("object_state.not_a_tensor",),
        )
        with self.assertRaisesRegex(TypeError, "not Tensor"):
            compiled(torch.ones(2))

    def test_registered_tensor_aliases_are_traced(self):
        model = _ModelWithRegisteredAlias()
        traced_state = []

        def fn(value):
            traced_state.append(
                (model.weight, model.weight_alias, model.second_weight_alias)
            )
            return value + model.weight + model.weight_alias + model.second_weight_alias

        compiled = direct_compile.direct_compile(
            fn,
            specialization_key=lambda value: tuple(value.shape),
            model=model,
        )
        compiler = direct_compile.tpu_torch_compile
        result = torch.full((2,), 3.0)
        with (
            patch.object(
                direct_compile,
                "_is_tpu_tensor",
                side_effect=lambda value: isinstance(value, torch.Tensor),
            ),
            patch.object(
                direct_compile.execution_mode,
                "set_eager_mode",
                return_value=nullcontext(),
            ),
            patch.object(
                compiler, "placeholder_like", side_effect=lambda tensor: tensor.clone()
            ),
            patch.object(
                compiler, "traverse_and_compile", return_value=_COMPILE_RESULT
            ),
            patch.object(compiler, "execute", return_value=[result]) as execute,
        ):
            output = compiled(torch.zeros(2))

        self.assertTrue(torch.equal(output, result))
        self.assertIsNot(traced_state[0][0], model.weight)
        self.assertIs(traced_state[0][0], traced_state[0][1])
        self.assertIs(traced_state[0][0], traced_state[0][2])
        runtime_inputs = execute.call_args.args[1]
        self.assertIs(runtime_inputs[0], model.weight)
        self.assertEqual(len(runtime_inputs), 2)

    def test_eagle_draft_pytrees_only_expose_hidden_states(self):
        draft_hidden_states = torch.ones((1, 2))
        draft_extend_hidden_states = torch.ones((1, 3))
        inputs = (
            EagleDraftInput(
                topk_p=torch.ones((1, 1)),
                hidden_states=draft_hidden_states,
            ),
            EagleDraftExtendInput(
                hidden_states=draft_extend_hidden_states,
                input_ids=torch.ones(1, dtype=torch.long),
            ),
        )

        leaves, treespec = tree_util.tree_flatten(inputs)
        rebuilt = tree_util.tree_unflatten(treespec, leaves)

        self.assertEqual(len(leaves), 2)
        self.assertIs(leaves[0], draft_hidden_states)
        self.assertIs(leaves[1], draft_extend_hidden_states)
        self.assertIs(rebuilt[0].hidden_states, draft_hidden_states)
        self.assertIsNone(rebuilt[0].topk_p)
        self.assertIs(rebuilt[1].hidden_states, draft_extend_hidden_states)
        self.assertIsNone(rebuilt[1].input_ids)

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
            patch.object(
                compiler, "traverse_and_compile", return_value=_COMPILE_RESULT
            ),
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

    def test_artifact_reports_runtime_tensor_structure_changes(self):
        value = torch.ones(2)
        artifact = _make_artifact(traced_shapes=((2,),))

        with (
            patch.object(direct_compile, "_is_tpu_tensor", return_value=False),
            self.assertRaisesRegex(TypeError, r"\[0\]\[0\].*now Tensor"),
        ):
            artifact(((value,), {}))

    def test_artifact_rejects_runtime_tensor_shape_changes(self):
        """Bug regression: a pytree leaf not covered by the specialization key
        (e.g. an unpadded batch-metadata tensor) used to be bound into the
        executable at its runtime shape, corrupting the launch device-side
        (XLA e0102). The artifact must instead fail loudly, naming the leaf."""
        artifact = _make_artifact(traced_shapes=((64,),))

        with (
            patch.object(direct_compile, "_is_tpu_tensor", return_value=True),
            self.assertRaisesRegex(
                ValueError, r"\[0\]\[0\].*traced as \(64,\).*now \(63,\)"
            ),
        ):
            artifact(((torch.ones(63),), {}))

    def test_artifact_accepts_traced_shapes(self):
        """Negative branch of the shape guard: a traced-shape match must not
        raise (a guard degraded to always-raise would pass the case above)."""
        artifact = _make_artifact(traced_shapes=((2,),))
        executed = []

        with (
            patch.object(direct_compile, "_is_tpu_tensor", return_value=True),
            patch.object(
                direct_compile.tpu_torch_compile,
                "execute",
                side_effect=lambda _executable, inputs: executed.append(inputs) or [],
            ),
        ):
            artifact(((torch.ones(2),), {}))

        self.assertEqual(len(executed), 1)


if __name__ == "__main__":
    unittest.main()
