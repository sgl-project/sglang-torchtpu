"""Tests for direct TPU compilation and artifact dispatch."""

import unittest
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from jax import tree_util
from torch_tpu._internal.device_utils.annotations import LayoutContext, TpuLayout

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.srt.environ import envs
from sglang.srt.layers import dp_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import direct_compile
from sglang_tpu.compilation.attention_rank_inputs import (
    use_attention_rank_tensors,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.warmup_queue import collect_warmups

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

_COMPILE_RESULT = SimpleNamespace(executable="executable")
_TPU_AVAILABLE = hasattr(torch, "accelerator") and torch.accelerator.is_available()


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


class _ModelWithUnusedInputs(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, device=device))
        self.unused_weight = torch.nn.Parameter(torch.full((2,), 9.0, device=device))
        self.register_buffer("weight_alias", self.weight, persistent=False)
        self.register_buffer(
            "unused_buffer",
            torch.view_as_complex(torch.zeros((2, 2), device=device)),
            persistent=False,
        )
        self.register_buffer("kv_bias", torch.full((2,), 5.0, device=device))
        self.object_state = _ObjectWithTensorState(
            torch.full((2,), 3.0, device=device), self.weight
        )


def _pruning_mlir_artifact():
    """Model inputs 1 and 2 are unused; 3 and 4 feed only KV/callback outputs."""
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo

    with mlir.make_ir_context():
        module = ir.Module.parse("""module {
              func.func @main(
                  %weight: tensor<2xf32>, %unused_weight: tensor<2xf32>,
                  %unused_buffer: tensor<2xcomplex<f32>>, %kv_bias: tensor<2xf32>,
                  %callback_bias: tensor<2xf32>, %kv: tensor<2xf32>,
                  %value: tensor<2xf32>, %unused_runtime: tensor<2xf32>
              ) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
                %first = stablehlo.add %value, %weight : tensor<2xf32>
                %result = stablehlo.add %first, %weight : tensor<2xf32>
                %updated_kv = stablehlo.add %kv, %kv_bias : tensor<2xf32>
                %callback = stablehlo.add %value, %callback_bias : tensor<2xf32>
                return %result, %updated_kv, %callback
                    : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
              }
            }""")
        return stablehlo.serialize_portable_artifact(
            module, stablehlo.get_current_version()
        )


class _KVPool:
    def __init__(self, value):
        self.value = value

    def get_state(self):
        return [self.value]

    def set_state(self, state):
        self.value = state[0]


class TestPPProxyPytree(CustomTestCase):
    def test_proxy_flattens_to_dict_key_paths_and_rebuilds(self):
        hidden = torch.zeros(2, 3)
        residual = torch.ones(2, 3)
        proxy = PPProxyTensors({"hidden_states": hidden, "residual": residual})

        path_leaves, treespec = tree_util.tree_flatten_with_path(proxy)

        self.assertEqual(
            [path for path, _ in path_leaves],
            [(tree_util.DictKey("hidden_states"),), (tree_util.DictKey("residual"),)],
        )
        self.assertIs(path_leaves[0][1], hidden)
        self.assertIs(path_leaves[1][1], residual)
        for path, leaf in path_leaves:
            self.assertIs(direct_compile._get_path(proxy, path), leaf)

        rebuilt = tree_util.tree_unflatten(treespec, [residual, hidden])
        self.assertIsInstance(rebuilt, PPProxyTensors)
        self.assertIs(rebuilt["hidden_states"], residual)
        self.assertIs(rebuilt["residual"], hidden)

    def test_proxy_inside_call_kwargs_flattens(self):
        proxy = PPProxyTensors({"hidden_states": torch.zeros(1)})
        path_leaves, _ = tree_util.tree_flatten_with_path(
            ((), {"pp_proxy_tensors": proxy})
        )
        self.assertEqual(len(path_leaves), 1)
        path, leaf = path_leaves[0]
        self.assertIs(leaf, proxy["hidden_states"])
        self.assertEqual(path[-1], tree_util.DictKey("hidden_states"))


class TestAuxHiddenStatePytree(CustomTestCase):
    def test_aux_list_output_flattens_to_tensor_leaves_and_rebuilds(self):
        """A target verify that returns (hidden, aux) for DFlash capture must trace.

        Unregistered, the aux list is one opaque leaf and direct_compile rejects
        the output as a non-tensor.
        """
        hidden, first, second = torch.zeros(2), torch.ones(2), torch.full((2,), 2.0)
        leaves, treespec = tree_util.tree_flatten(
            (hidden, AuxHiddenStateList([first, second]))
        )
        self.assertEqual(
            [id(leaf) for leaf in leaves], [id(hidden), id(first), id(second)]
        )
        _, rebuilt = tree_util.tree_unflatten(treespec, [hidden, second, first])
        self.assertIsInstance(rebuilt, AuxHiddenStateList)
        self.assertEqual([id(t) for t in rebuilt], [id(second), id(first)])


def _make_pruning_function(model, kv_pool, callback_values, **compile_options):
    def fn(value, unused_runtime):
        kv_pool.value.add_(model.kv_bias)
        direct_compile.insert_callback(
            value + model.object_state.value, callback_values.append
        )
        return value + model.weight_alias + model.object_state.weight_alias

    return direct_compile.direct_compile(
        fn,
        specialization_key=lambda value, unused_runtime: tuple(value.shape),
        model=model,
        model_extra_tensor_attrs=("object_state.value", "object_state.weight_alias"),
        kv_pool=kv_pool,
        **compile_options,
    )


class TestAuxHiddenStatePytree(CustomTestCase):
    def test_aux_states_flatten_to_tensor_leaves_and_rebuild(self):
        """An EAGLE3 target's (hidden, aux) output must reach direct_compile as
        tensor leaves; an opaque list leaf fails its all-TPU-tensors check."""
        hidden, first, second = torch.zeros(2), torch.ones(2), torch.full((2,), 2.0)
        leaves, treespec = tree_util.tree_flatten(
            (hidden, AuxHiddenStateList([first, second]))
        )
        self.assertEqual(len(leaves), 3)
        self.assertTrue(all(isinstance(leaf, torch.Tensor) for leaf in leaves))
        _, aux = tree_util.tree_unflatten(treespec, leaves)
        self.assertIsInstance(aux, AuxHiddenStateList)
        self.assertIs(aux[1], second)


class TestDirectCompile(CustomTestCase):
    def test_random_compiled_op_warmup_preserves_rng(self):
        torch_compile = torch.compile

        def compile_on_cpu(fn, **kwargs):
            return torch_compile(fn, backend="eager", fullgraph=True)

        def draw(*, value):
            return torch.rand_like(value)

        value = torch.ones(4)
        with torch.random.fork_rng(devices=[]):
            for enabled, tpu in ((False, True), (True, False), (True, True)):
                with (
                    self.subTest(enabled=enabled, tpu=tpu),
                    envs.SGLANG_ENABLE_TORCH_COMPILE.override(enabled),
                    patch(
                        "sglang.srt.compilation.compiled_op.is_tpu", return_value=tpu
                    ),
                    patch("torch.compile", side_effect=compile_on_cpu) as compiler,
                    get_context().override_server_args(tpu_compile_tracer="dynamo"),
                ):
                    eager_calls = []

                    def eager(*, value):
                        eager_calls.append(value)
                        return draw(value=value)

                    op = compiled_op(uses_rng=True)(draw if enabled and tpu else eager)
                    state = torch.random.get_rng_state()
                    with collect_warmups() as queue:
                        warmup_compiled_op(op, value=value)
                        self.assertTrue(
                            torch.equal(torch.random.get_rng_state(), state)
                        )
                        self.assertEqual(queue.calls, [])
                    warmup_compiled_op(op, value=value)
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
                    if not enabled or not tpu:
                        self.assertEqual(eager_calls, [])
                        compiler.assert_not_called()
                    actual = op(value=value)
                    after = torch.random.get_rng_state()
                    self.assertFalse(torch.equal(after, state))
                    torch.random.set_rng_state(state)
                    self.assertTrue(torch.equal(actual, draw(value=value)))
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), after))

            @compiled_op(uses_rng=True)
            def fail(value):
                torch.rand_like(value)
                raise RuntimeError("warmup failed")

            state = torch.random.get_rng_state()
            with (
                envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
                patch("sglang.srt.compilation.compiled_op.is_tpu", return_value=True),
                patch("torch.compile", side_effect=lambda fn, **kwargs: fn),
                get_context().override_server_args(tpu_compile_tracer="dynamo"),
                self.assertRaisesRegex(RuntimeError, "warmup failed"),
            ):
                warmup_compiled_op(fail, value)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_cold_call_and_synchronous_mode_return_real_results(self):
        def compile_helper(fn, **kwargs):
            compiled = MagicMock(side_effect=fn)
            compiled.cache = {}
            compiled.precompile.side_effect = lambda *args, **kwargs: PendingCall(
                lambda: None
            )
            return compiled

        with (
            patch("sglang.srt.compilation.compiled_op.is_tpu", return_value=True),
            patch(
                "sglang.srt.compilation.compiled_op.direct_compile",
                new=compile_helper,
            ),
            get_context().override_server_args(tpu_compile_tracer="direct"),
            envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        ):
            for asynchronous in (False, True):
                with (
                    self.subTest(asynchronous=asynchronous),
                    envs.SGLANG_DIRECT_COMPILE_ASYNC.override(asynchronous),
                ):
                    op = compiled_op()(lambda x: x * 3)
                    x = torch.arange(3)
                    with collect_warmups() as queue:
                        self.assertTrue(torch.equal(op(x), x * 3))
                        self.assertEqual(queue.calls, [])
                    self.assertIsNone(warmup_compiled_op(op, x))
                    self.assertTrue(torch.equal(op(x), x * 3))

    def test_precompile_releases_inputs_and_joins_without_execution(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                compiled = direct_compile.direct_compile(
                    lambda value: value + 1,
                    specialization_key=lambda value: tuple(value.shape),
                )
                artifact = _make_artifact(((2,),))
                future = MagicMock()
                if asynchronous:
                    artifact.executable = None
                    artifact.pending_compile = future

                def compile_helper(
                    args, kwargs, *, async_compile, spmd_compiler, graph
                ):
                    self.assertEqual(async_compile, asynchronous)
                    self.assertIsNone(spmd_compiler)
                    self.assertIsNone(graph)
                    return artifact

                with (
                    patch.object(compiled, "_compile", compile_helper),
                    patch.object(
                        direct_compile.tpu_torch_compile, "execute"
                    ) as execute,
                ):
                    inputs = torch.ones(2)
                    reference = weakref.ref(inputs)
                    with collect_warmups() as queue:
                        queue.calls.append(
                            compiled.precompile(inputs, async_compile=asynchronous)
                        )
                        del inputs
                        self.assertIsNone(reference())
                        future.get.assert_not_called()
                    self.assertIsNone(artifact.pending_compile)
                    self.assertEqual(future.get.call_count, int(asynchronous))
                    execute.assert_not_called()

    def test_compiled_helper_inlines_inside_outer_dynamo_graph(self):
        @compiled_op()
        def add_one(x):
            return x + 1

        def outer(x):
            return add_one(x) * 2

        compiled = torch.compile(outer, backend="eager", fullgraph=True)
        x = torch.arange(4)
        self.assertTrue(torch.equal(compiled(x), (x + 1) * 2))

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

    def test_compiled_op_keyword_call_reuses_positional_warmup(self):
        """A keyword-argument call must hit the specialization warmed positionally."""
        artifact = MagicMock(return_value="output")

        @compiled_op()
        def op(a, b, scale=2):
            return a * scale + b

        with (
            patch("sglang.srt.compilation.compiled_op.is_tpu", return_value=True),
            patch.object(
                direct_compile.CompiledFunction, "_compile", return_value=artifact
            ) as compile_fn,
            get_context().override_server_args(tpu_compile_tracer="direct"),
            envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        ):
            a, b = torch.ones(16, 6), torch.zeros(16, 6)
            op(a, b)
            with torch.compiler.set_stance("fail_on_recompile"):
                op(a=a, b=b)
                op(b=b, a=a, scale=2)

        compile_fn.assert_called_once()

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_torch_tpu_cache_hit_for_equivalent_compilations(self):
        def fn(value):
            return value + value

        def new_compiled_fn():
            return direct_compile.direct_compile(
                fn,
                specialization_key=lambda value: tuple(value.shape),
            )

        def run(compiled_fn, value, async_compile):
            if async_compile:
                return compiled_fn.warmup_async(value).wait()
            return compiled_fn(value)

        for async_compile in (False, True):
            with self.subTest(async_compile=async_compile):
                value = torch.ones(2, device="tpu")
                expected = torch.full((2,), 2.0)
                torch.tpu._clear_cache()
                requests_before = torch.tpu._get_cache_requests()
                hits_before = torch.tpu._get_cache_hits()
                misses_before = torch.tpu._get_cache_misses()

                result = run(new_compiled_fn(), value, async_compile)
                torch.testing.assert_close(result.cpu(), expected)
                self.assertEqual(torch.tpu._get_cache_requests(), requests_before + 1)
                self.assertEqual(torch.tpu._get_cache_hits(), hits_before)
                self.assertEqual(torch.tpu._get_cache_misses(), misses_before + 1)

                requests_before = torch.tpu._get_cache_requests()
                hits_before = torch.tpu._get_cache_hits()
                misses_before = torch.tpu._get_cache_misses()

                result = run(new_compiled_fn(), value, async_compile)
                torch.testing.assert_close(result.cpu(), expected)
                self.assertEqual(torch.tpu._get_cache_requests(), requests_before + 1)
                self.assertEqual(torch.tpu._get_cache_hits(), hits_before + 1)
                self.assertEqual(torch.tpu._get_cache_misses(), misses_before)

    @unittest.skipUnless(torch.accelerator.is_available(), "requires TPU")
    def test_native_async_grid_preserves_model_kv_and_callbacks(self):
        model = _ModelWithObjectTensorState().to("tpu")
        model.object_state.value = model.object_state.value.to("tpu")
        model.object_state.weight_alias = model.weight
        pool = _KVPool(torch.zeros(2, device="tpu"))
        original_pool = pool.value
        callbacks = []

        def fn(value):
            pool.value.add_(value.sum(dim=0))
            out = value + model.weight + model.object_state.value
            direct_compile.insert_callback(out[0], callbacks.append)
            return out

        compiled = direct_compile.direct_compile(
            fn,
            specialization_key=lambda value: tuple(value.shape),
            model=model,
            model_extra_tensor_attrs=(
                "object_state.value",
                "object_state.weight_alias",
            ),
            kv_pool=pool,
        )
        values = [torch.full((bs, 2), float(bs), device="tpu") for bs in (1, 2, 4)]
        compiler = direct_compile.tpu_torch_compile
        with (
            torch.no_grad(),
            patch.object(compiler, "execute", wraps=compiler.execute) as execute,
        ):
            pending = [compiled.warmup_async(value) for value in values]
            execute.assert_not_called()
            self.assertEqual(len(compiled.cache), 3)
            self.assertEqual(callbacks, [])
            self.assertIs(pool.value, original_pool)
            self.assertTrue(torch.equal(pool.value.cpu(), torch.zeros(2)))

            expected_pool = torch.zeros(2)
            for warmup, value in zip(pending, values):
                actual = warmup.wait()
                expected = value.cpu() + 3
                expected_pool += value.cpu().sum(dim=0)
                self.assertTrue(torch.equal(actual.cpu(), expected))
                self.assertTrue(torch.equal(pool.value.cpu(), expected_pool))
                self.assertTrue(torch.equal(callbacks[-1].cpu(), expected[0]))
                self.assertIs(pool.value, original_pool)

            replacement = torch.full((2, 2), 7.0, device="tpu")
            self.assertTrue(
                torch.equal(compiled(replacement).cpu(), torch.full((2, 2), 10.0))
            )
            self.assertEqual(len(compiled.cache), 3)
            self.assertEqual(execute.call_count, 4)
            self.assertEqual(len(callbacks), 4)
            self.assertTrue(torch.equal(pool.value.cpu(), expected_pool + 14))
            self.assertTrue(
                all(
                    artifact.pending_compile is None
                    for artifact in compiled.cache.values()
                )
            )

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_random_helper_preserves_trace_state_and_advances_on_replay(self):
        value = torch.ones((2, 7), device="tpu")
        original_state = torch.tpu.get_rng_state()
        try:
            for asynchronous in (False, True):

                def draw(x):
                    return torch.rand_like(x)

                compiled = direct_compile.direct_compile(
                    draw,
                    specialization_key=lambda x: tuple(x.shape),
                    uses_rng=True,
                )
                torch.manual_seed(113)
                before = torch.tpu.get_rng_state()
                if asynchronous:
                    pending = compiled.warmup_async(value)
                    self.assertTrue(torch.equal(before, torch.tpu.get_rng_state()))
                    first = pending.wait().cpu()
                else:
                    first = compiled(value).cpu()
                after_first = torch.tpu.get_rng_state()
                with torch.compiler.set_stance("fail_on_recompile"):
                    second = compiled(value).cpu()
                after_second = torch.tpu.get_rng_state()
                torch.tpu.set_rng_state(before)
                self.assertTrue(torch.equal(first, draw(value).cpu()))
                self.assertTrue(torch.equal(after_first, torch.tpu.get_rng_state()))
                self.assertTrue(torch.equal(second, draw(value).cpu()))
                self.assertTrue(torch.equal(after_second, torch.tpu.get_rng_state()))
        finally:
            torch.tpu.set_rng_state(original_state)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_precompile_first_call_preserves_rng_without_recompilation(self):
        value = torch.ones((2, 7), device="tpu")
        original_state = torch.tpu.get_rng_state()

        def draw(x):
            return x + torch.rand_like(x)

        try:
            for asynchronous in (False, True):
                with (
                    self.subTest(asynchronous=asynchronous),
                    torch.no_grad(),
                    get_context().override_server_args(tpu_compile_tracer="direct"),
                    envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
                    envs.SGLANG_DIRECT_COMPILE_ASYNC.override(asynchronous),
                ):
                    op = compiled_op(uses_rng=True)(draw)
                    before = torch.tpu.get_rng_state()
                    compiler = direct_compile.tpu_torch_compile
                    with (
                        patch.object(
                            compiler, "execute", wraps=compiler.execute
                        ) as execute,
                        patch.object(
                            compiler,
                            "get_device_layout_if_materialized",
                            wraps=compiler.get_device_layout_if_materialized,
                        ) as layout,
                    ):
                        with collect_warmups():
                            warmup_compiled_op(op, value)
                        execute.assert_not_called()
                        if asynchronous:
                            self.assertTrue(layout.called)
                            self.assertTrue(
                                all(
                                    call.args[0] is not value
                                    for call in layout.call_args_list
                                )
                            )
                    self.assertTrue(torch.equal(before, torch.tpu.get_rng_state()))
                    torch.accelerator.synchronize()
                    misses = torch.tpu._get_cache_misses()
                    with torch.compiler.set_stance("fail_on_recompile"):
                        first = op(value)
                        torch.accelerator.synchronize()
                    self.assertEqual(torch.tpu._get_cache_misses(), misses)
                    after = torch.tpu.get_rng_state()
                    torch.tpu.set_rng_state(before)
                    self.assertTrue(torch.equal(first.cpu(), draw(value).cpu()))
                    self.assertTrue(torch.equal(after, torch.tpu.get_rng_state()))
        finally:
            torch.tpu.set_rng_state(original_state)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_registered_helpers_compile_directly_and_match_eager(self):
        from sglang.srt.compilation.compiled_op import (
            get_compiled_ops,
        )
        from sglang_tpu import spec_op_precompile  # noqa: F401
        from sglang_tpu.spec_op_warmup import SpecOpWarmupContext

        cases = []
        with (
            torch.no_grad(),
            get_context().override_server_args(tpu_compile_tracer="direct"),
            envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
            envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
        ):
            with collect_warmups():
                for width in (1, 3):
                    context = SpecOpWarmupContext(
                        device=torch.device("tpu"),
                        physical_bs=2,
                        num_verify_tokens=width,
                        vocab_size=16,
                        draft_vocab_size=16,
                        hidden_size=8,
                        hidden_dtype=torch.float32,
                    )
                    for registered in get_compiled_ops():
                        args = registered.make_warmup_args(context)
                        warmup_compiled_op(registered.op, *args)
                        cases.append((registered.op, args))
            self.assertEqual(len(cases), 12)
            with torch.compiler.set_stance("fail_on_recompile"):
                for op, args in cases:
                    cpu_args = tree_util.tree_map(
                        lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, args
                    )
                    expected = op.__wrapped__(*cpu_args)
                    torch.accelerator.synchronize()
                    misses = torch.tpu._get_cache_misses()
                    actual = op(*args)
                    torch.accelerator.synchronize()
                    self.assertEqual(torch.tpu._get_cache_misses(), misses, op.__name__)
                    self.assertEqual(
                        tree_util.tree_structure(actual),
                        tree_util.tree_structure(expected),
                    )
                    for a, b in zip(
                        tree_util.tree_leaves(actual), tree_util.tree_leaves(expected)
                    ):
                        self.assertTrue(torch.equal(a.cpu(), b), op.__name__)

    def test_native_async_failure_propagates_before_execution(self):
        artifact = _make_artifact(((2,),))
        artifact.pending_compile = MagicMock()
        artifact.pending_compile.get.side_effect = RuntimeError("native compile failed")
        with patch.object(direct_compile.tpu_torch_compile, "execute") as execute:
            complete = MagicMock()
            with self.assertRaisesRegex(RuntimeError, "native compile failed"):
                with collect_warmups() as queue:
                    queue.calls.append(PendingCall(artifact.wait_for_compile))
                    queue.completions.append(complete)
            complete.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "native compile failed"):
                artifact(((torch.ones(2),), {}))
            execute.assert_not_called()

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_preserves_model_and_argument_layouts(self):
        """Compile against live buffer layouts, not placeholder defaults."""
        cache_cpu = torch.arange(256 * 64, dtype=torch.float32).reshape(256, 64)
        values_cpu = cache_cpu * 2
        compiler = direct_compile.tpu_torch_compile
        execute = compiler.execute

        def execute_with_layout_check(executable, inputs):
            input_layouts = [
                compiler.get_device_layout_if_materialized(tensor) for tensor in inputs
            ]
            self.assertTrue(all(layout is not None for layout in input_layouts))
            # Inspect the real executable before launching it: a silently ignored
            # layout must fail even if execution would relayout the inputs.
            self.assertEqual(
                executable.get_parameter_layouts(),
                input_layouts,
            )
            return execute(executable, inputs)

        # Swap the layouts without changing the graph. An unconstrained compiler
        # cannot accidentally choose the required layouts for both cases.
        for cache_order, values_order in (([1, 0], [0, 1]), ([0, 1], [1, 0])):
            with self.subTest(cache_order=cache_order, values_order=values_order):
                with LayoutContext(TpuLayout(minor_to_major=cache_order)):
                    cache = cache_cpu.to("tpu")
                with LayoutContext(TpuLayout(minor_to_major=values_order)):
                    values = values_cpu.to("tpu")
                for tensor, order in ((cache, cache_order), (values, values_order)):
                    layout = compiler.get_device_layout_if_materialized(tensor)
                    self.assertIsNotNone(layout)
                    self.assertEqual(layout[0], order)
                model = torch.nn.Module()
                model.register_buffer("cache", cache)

                def fn(positions, values):
                    cache_rows = model.cache.index_select(0, positions)
                    return cache_rows + values.index_select(0, positions)

                compiled = direct_compile.direct_compile(
                    fn, model=model, specialization_key=lambda positions, values: ()
                )
                with patch.object(
                    compiler, "execute", side_effect=execute_with_layout_check
                ):
                    for indices in ([0, 127, 128, 255], [255, 128, 0, 255]):
                        positions = torch.tensor(indices)
                        expected = (cache_cpu + values_cpu).index_select(0, positions)
                        actual = compiled(positions.to("tpu"), values)
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )
                self.assertEqual(len(compiled.cache), 1)

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
                direct_compile, "_live_model_input_indices", return_value=(0, 1)
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

    def test_forward_batch_routing_values_reuse_compilation(self):
        routing_inputs = [
            torch.full((1, 2, 2), variant, dtype=torch.int32) for variant in range(32)
        ]
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=torch.zeros(2, dtype=torch.int64),
            req_pool_indices=torch.zeros(2, dtype=torch.int64),
            seq_lens=torch.ones(2, dtype=torch.int64),
            out_cache_loc=torch.zeros(2, dtype=torch.int64),
            seq_lens_sum=2,
        )
        compiled = direct_compile.direct_compile(
            lambda forward_batch: forward_batch.forced_routing_ids,
            specialization_key=lambda forward_batch: forward_batch.batch_size,
        )
        compiler = direct_compile.tpu_torch_compile
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
            ) as compile_fn,
            patch.object(
                compiler,
                "execute",
                side_effect=lambda _executable, inputs: [inputs[-1]],
            ) as execute,
            torch.compiler.set_stance("fail_on_recompile"),
        ):
            for routing in routing_inputs:
                output = compiled(replace(batch, forced_routing_ids=routing))
                self.assertIs(output, routing)
                self.assertIs(execute.call_args.args[1][-1], routing)

        compile_fn.assert_called_once()
        self.assertEqual(execute.call_count, len(routing_inputs))

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
                direct_compile, "_live_model_input_indices", return_value=(0,)
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

    def test_model_input_pruning_preserves_runtime_state(self):
        self._check_model_input_pruning(prune_unused_inputs=True)

    def test_preserves_all_model_inputs_by_default(self):
        self._check_model_input_pruning()

    def _check_model_input_pruning(self, **compile_options):
        prune_unused_inputs = compile_options.get("prune_unused_inputs", False)
        model = _ModelWithUnusedInputs()
        kv_pool = _KVPool(torch.zeros(2))
        pool_input = kv_pool.value
        callback_values = []
        compiled = _make_pruning_function(
            model, kv_pool, callback_values, **compile_options
        )
        compiler = direct_compile.tpu_torch_compile

        def execute(_executable, inputs):
            if prune_unused_inputs:
                weight, kv_bias, callback_bias, kv, value, _unused_runtime = inputs
            else:
                (
                    weight,
                    _unused_weight,
                    _unused_buffer,
                    kv_bias,
                    callback_bias,
                    kv,
                    value,
                    _unused_runtime,
                ) = inputs
            return [value + weight + weight, kv + kv_bias, value + callback_bias]

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
                direct_compile,
                "_live_model_input_indices",
                wraps=direct_compile._live_model_input_indices,
            ) as prune,
            patch.object(compiler, "build_mlir", return_value=object()) as lower,
            patch.object(
                compiler,
                "serialize_mlir_portable_artifact",
                side_effect=lambda _: _pruning_mlir_artifact(),
            ) as serialize,
            patch.object(
                compiler, "traverse_and_compile", return_value=_COMPILE_RESULT
            ) as traverse,
            patch.object(compiler, "execute", side_effect=execute) as run,
        ):
            for step, value in enumerate((2.0, 7.0), start=1):
                result = compiled(torch.full((2,), value), torch.zeros(2))
                torch.testing.assert_close(result, torch.full((2,), value + 2))
                torch.testing.assert_close(kv_pool.value, torch.full((2,), 5.0 * step))
                torch.testing.assert_close(
                    callback_values[-1], torch.full((2,), value + 3)
                )

            # Explicit runtime arguments keep their shape guard even when the
            # model does not read them. Pruning is limited to immutable state.
            with self.assertRaisesRegex(ValueError, "tensor input changed shape"):
                compiled(torch.ones(2), torch.zeros(3))

        traverse.assert_called_once()
        self.assertEqual(run.call_count, 2)
        retained_graph_inputs = traverse.call_args.args[1]
        if prune_unused_inputs:
            prune.assert_called_once()
            self.assertEqual(lower.call_count, 2)
            serialize.assert_called_once()
            original_graph_inputs = lower.call_args_list[0].args[1]
            self.assertEqual(len(original_graph_inputs), 8)
            expected_indices = (0, 3, 4, 5, 6, 7)
            self.assertEqual(len(retained_graph_inputs), len(expected_indices))
            for tensor, index in zip(retained_graph_inputs, expected_indices):
                self.assertIs(tensor, original_graph_inputs[index])
            for tensor, validated in zip(
                retained_graph_inputs, lower.call_args_list[1].args[1], strict=True
            ):
                self.assertIs(tensor, validated)
            expected_model_inputs = (
                model.weight,
                model.kv_bias,
                model.object_state.value,
            )
        else:
            prune.assert_not_called()
            lower.assert_not_called()
            serialize.assert_not_called()
            self.assertEqual(len(retained_graph_inputs), 8)
            expected_model_inputs = (
                model.weight,
                model.unused_weight,
                model.unused_buffer,
                model.kv_bias,
                model.object_state.value,
            )
        self.assertEqual(
            traverse.call_args.kwargs["argument_layouts"],
            [[]] * len(retained_graph_inputs),
        )
        model_inputs = next(iter(compiled.cache.values())).model_inputs
        self.assertEqual(len(model_inputs), len(expected_model_inputs))
        for tensor, expected in zip(model_inputs, expected_model_inputs):
            self.assertIs(tensor, expected)
        self.assertIs(model.weight_alias, model.weight)
        self.assertIs(model.object_state.weight_alias, model.weight)
        self.assertIs(kv_pool.value, pool_input)

    def test_model_input_pruning_checks_exported_signature(self):
        compiler = direct_compile.tpu_torch_compile
        with patch.object(compiler, "build_mlir") as lower:
            self.assertEqual(direct_compile._live_model_input_indices([], [], 0), ())
        lower.assert_not_called()

        with (
            patch.object(compiler, "build_mlir", return_value=object()),
            patch.object(
                compiler,
                "serialize_mlir_portable_artifact",
                return_value=_pruning_mlir_artifact(),
            ),
            self.assertRaisesRegex(RuntimeError, "expected 7, got 8"),
        ):
            direct_compile._live_model_input_indices([], [torch.zeros(2)] * 7, 5)

    def test_model_input_pruning_preserves_dependencies_discarded_by_lowering(self):
        compiler = direct_compile.tpu_torch_compile
        value = compiler.placeholder([2], torch.int64, False)
        with direct_compile.execution_mode.set_eager_mode(
            direct_compile.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            result = torch.isnan(value)

        # Integer isnan exports a constant false, so its MLIR argument has no
        # uses. The original deferred graph still needs that argument to lower.
        self.assertEqual(
            direct_compile._live_model_input_indices([result], [value], 1), (0,)
        )

    def test_model_input_pruning_with_native_topk_export(self):
        compiler = direct_compile.tpu_torch_compile
        for stable in (False, True):
            with self.subTest(stable=stable):
                unused = compiler.placeholder([2, 8], torch.float32, False)
                weight = compiler.placeholder([2, 8], torch.float32, False)
                value = compiler.placeholder([2, 8], torch.float32, False)
                with direct_compile.execution_mode.set_eager_mode(
                    direct_compile.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
                ):
                    scores = value + weight
                    outputs = (
                        torch.sort(scores, stable=True)
                        if stable
                        else torch.topk(scores, 2)
                    )

                # Exercise TorchTPU's real exporter and JAXlib's real reader:
                # a fixture produced by JAXlib cannot catch CHLO version skew.
                self.assertEqual(
                    direct_compile._live_model_input_indices(
                        list(outputs), [unused, weight, value], 2
                    ),
                    (1,),
                )

    def test_model_input_pruning_propagates_unrelated_export_errors(self):
        compiler = direct_compile.tpu_torch_compile
        with (
            patch.object(
                compiler,
                "build_mlir",
                side_effect=[object(), RuntimeError("unrelated lowering failure")],
            ),
            patch.object(
                compiler,
                "serialize_mlir_portable_artifact",
                return_value=_pruning_mlir_artifact(),
            ),
            self.assertRaisesRegex(RuntimeError, "unrelated lowering failure"),
        ):
            direct_compile._live_model_input_indices([], [torch.zeros(2)] * 8, 5)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_pruned_topk_model_on_tpu(self):
        model = torch.nn.Module()
        model.register_buffer(
            "bias", torch.arange(8, dtype=torch.float32, device="tpu")
        )
        model.register_buffer("unused", torch.ones(8, device="tpu"))
        compiled = direct_compile.direct_compile(
            lambda value: tuple(torch.topk(value + model.bias, 2)),
            specialization_key=lambda value: tuple(value.shape),
            model=model,
            prune_unused_inputs=True,
        )

        for rows in (1, 4, 1):
            value = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
            expected = torch.topk(value + torch.arange(8), 2)
            actual = compiled(value.to("tpu"))
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result.cpu(), reference)

        self.assertEqual(len(compiled.cache), 2)
        for artifact in compiled.cache.values():
            self.assertEqual(len(artifact.model_inputs), 1)
            self.assertIs(artifact.model_inputs[0], model.bias)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_constant_folded_model_input_on_tpu(self):
        model = torch.nn.Module()
        model.register_buffer(
            "integers", torch.ones(2, dtype=torch.int64, device="tpu")
        )
        compiled = direct_compile.direct_compile(
            lambda: torch.isnan(model.integers),
            specialization_key=lambda: None,
            model=model,
        )

        for _ in range(2):
            torch.testing.assert_close(
                compiled().cpu(), torch.zeros(2, dtype=torch.bool)
            )
        artifact = compiled.cache[None]
        self.assertEqual(len(artifact.model_inputs), 1)
        self.assertIs(artifact.model_inputs[0], model.integers)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_model_input_pruning_on_tpu_across_specializations(self):
        self._check_model_input_pruning_on_tpu(prune_unused_inputs=True)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
    def test_preserves_all_model_inputs_by_default_on_tpu(self):
        self._check_model_input_pruning_on_tpu()

    def _check_model_input_pruning_on_tpu(self, **compile_options):
        prune_unused_inputs = compile_options.get("prune_unused_inputs", False)
        model = _ModelWithUnusedInputs(device="tpu")
        kv_pool = _KVPool(torch.zeros(2, device="tpu"))
        pool_input = kv_pool.value
        callback_values = []
        compiled = _make_pruning_function(
            model, kv_pool, callback_values, **compile_options
        )
        compiler = direct_compile.tpu_torch_compile

        with (
            patch.object(compiler, "build_mlir", wraps=compiler.build_mlir) as lower,
            patch.object(compiler, "execute", wraps=compiler.execute) as run,
        ):
            for step, rows in enumerate((2, 4, 2), start=1):
                value = torch.full((rows, 2), float(step), device="tpu")
                result = compiled(value, torch.zeros(2, device="tpu"))
                torch.testing.assert_close(
                    result.cpu(), torch.full((rows, 2), float(step + 2))
                )
                torch.testing.assert_close(
                    kv_pool.value.cpu(), torch.full((2,), 5.0 * step)
                )
                torch.testing.assert_close(
                    callback_values[-1].cpu(), torch.full((rows, 2), float(step + 3))
                )

        self.assertEqual(lower.call_count, 4 if prune_unused_inputs else 0)
        self.assertEqual(len(compiled.cache), 2)
        self.assertIs(kv_pool.value, pool_input)
        for artifact in compiled.cache.values():
            self.assertEqual(
                len(artifact.model_inputs), 3 if prune_unused_inputs else 5
            )
        for call in run.call_args_list:
            self.assertEqual(len(call.args[1]), 6 if prune_unused_inputs else 8)
            self.assertEqual(
                any(t is model.unused_buffer for t in call.args[1]),
                not prune_unused_inputs,
            )
            self.assertEqual(
                any(t is model.unused_weight for t in call.args[1]),
                not prune_unused_inputs,
            )

    def test_forward_hook_dispatches_inserted_callback(self):
        model = torch.nn.Identity()
        inserted = []
        callback_values = []

        def hook(_module, _args, output):
            inserted.append(
                direct_compile.insert_callback(
                    output,
                    lambda value: callback_values.append(value.clone()),
                )
            )

        model.register_forward_hook(hook)
        self.assertFalse(
            direct_compile.insert_callback(torch.ones(2), callback_values.append)
        )
        self.assertEqual(callback_values, [])

        compiled = direct_compile.direct_compile(
            model,
            specialization_key=lambda value: tuple(value.shape),
            model=model,
        )
        result = torch.full((2,), 3.0)
        callback_output = torch.full((2,), 4.0)
        compiler = direct_compile.tpu_torch_compile
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
            patch.object(compiler, "execute", return_value=[result, callback_output]),
        ):
            output = compiled(torch.zeros(2))

        self.assertTrue(torch.equal(output, result))
        self.assertEqual(inserted, [True])
        self.assertEqual(len(callback_values), 1)
        self.assertTrue(torch.equal(callback_values[0], callback_output))

    def test_artifact_dispatches_and_rebuilds_outputs(self):
        model = _Model()
        kv_pool = _KVPool(torch.zeros(2))
        pool_input = kv_pool.value
        traced_model_state = []
        traced_pool_state = []
        traced_compile_state = []
        captured_callback_values = []

        def fn(*, inputs):
            traced_compile_state.append(torch.compiler.is_compiling())
            traced_model_state.append((model.state.weight, model.state_alias.weight))
            traced_pool_state.append(kv_pool.value)
            kv_pool.value.add_(model.state.weight)
            direct_compile.insert_callback(
                inputs.value + 4,
                lambda value: captured_callback_values.append(value.clone()),
            )
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
        first_callback_output = torch.full((2,), 14.0)
        second_callback_output = torch.full((2,), 24.0)

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
                direct_compile, "_live_model_input_indices", return_value=(0, 1)
            ),
            patch.object(
                compiler, "traverse_and_compile", return_value=_COMPILE_RESULT
            ),
            patch.object(
                compiler,
                "execute",
                side_effect=(
                    [first_result, first_pool, first_callback_output],
                    [second_result, second_pool, second_callback_output],
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

        self.assertEqual(len(captured_callback_values), 2)
        self.assertTrue(torch.equal(captured_callback_values[0], first_callback_output))
        self.assertTrue(
            torch.equal(captured_callback_values[1], second_callback_output)
        )

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

    def test_mutation_only_function_compiles_without_output_leaves(self):
        """Bug regression: the DFlash draft KV appender returns None and mutates
        only the KV pool. An empty output-leaf list was rejected by the
        must-be-TPU-tensor guard, aborting draft warmup."""
        kv_pool = _KVPool(torch.zeros(2))
        pool_input = kv_pool.value

        def fn(*, value):
            kv_pool.value.add_(value)

        compiled = direct_compile.direct_compile(
            fn,
            specialization_key=lambda *, value: tuple(value.shape),
            kv_pool=kv_pool,
        )
        pool_output = torch.full((2,), 5.0)

        compiler = direct_compile.tpu_torch_compile
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
            ) as traverse,
            patch.object(compiler, "execute", return_value=[pool_output]),
        ):
            output = compiled(value=torch.ones(2))

        self.assertIsNone(output)
        self.assertIs(kv_pool.value, pool_input)
        self.assertTrue(torch.equal(kv_pool.value, pool_output))
        self.assertEqual(len(traverse.call_args.args[0]), 1)

    def test_rejects_function_with_neither_outputs_nor_mutations(self):
        """Negative branch: without a KV pool, a fn returning None traces an
        empty graph and must still be rejected."""
        compiled = direct_compile.direct_compile(
            lambda value: None,
            specialization_key=lambda value: tuple(value.shape),
        )

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
                direct_compile.tpu_torch_compile,
                "placeholder_like",
                side_effect=lambda tensor: tensor.clone(),
            ),
            self.assertRaisesRegex(
                TypeError, "TPU tensor outputs or KV-pool mutations"
            ),
        ):
            compiled(torch.ones(2))

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


class _RankModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros((), device="tpu"))

    def forward(self, input_ids, positions, forward_batch):
        mask = dp_attention._dp_rank_slot_mask(2, input_ids.device)
        scale = dp_attention.attn_rank0_scale(input_ids)
        return torch.where(mask[:, None], input_ids[None, :] * scale, self.bias)


class _RankPolicy(DecoderTpuCompilePolicy):
    def _direct_compile_specialization_key(self, metadata, input_ids, *args, **kwargs):
        return tuple(input_ids.shape)


class TestAttentionRankTensorScope(unittest.TestCase):
    def test_nested_scopes_restore_rank_state_after_error(self):
        original = {("attn_tp", torch.device("cpu")): torch.tensor(7)}
        outer = (torch.tensor(0), torch.tensor(1))
        inner = (torch.tensor(2), torch.tensor(3))
        with patch.object(dp_attention, "_RANK_TENSORS", original):
            with use_attention_rank_tensors(outer):
                self.assertIs(
                    dp_attention._rank_tensor("attn_dp", torch.device("cpu")), outer[1]
                )
                with self.assertRaisesRegex(RuntimeError, "trace failed"):
                    with use_attention_rank_tensors(inner):
                        self.assertIs(
                            dp_attention._rank_tensor("attn_tp", torch.device("cpu")),
                            inner[0],
                        )
                        raise RuntimeError("trace failed")
                self.assertIs(
                    dp_attention._rank_tensor("attn_dp", torch.device("cpu")), outer[1]
                )
            self.assertIs(dp_attention._RANK_TENSORS, original)


@unittest.skipUnless(torch.accelerator.is_available(), "requires TPU")
class TestDirectCompileDpRanks(unittest.TestCase):
    def test_rank_selectors_are_runtime_inputs(self):
        model = _RankModel()
        policy = _RankPolicy(
            model_runner=SimpleNamespace(
                model=model, token_to_kv_pool=None, device="tpu"
            ),
            owner=model,
            tracer="direct",
        )
        ranks = {
            axis: torch.tensor(0, dtype=torch.int64, device="tpu")
            for axis in ("attn_tp", "attn_dp")
        }
        cache = {
            (axis, device): value
            for axis, value in ranks.items()
            for device in (torch.device("tpu"), value.device)
        }
        values = torch.tensor([2.0, 3.0], device="tpu")
        backend = AttentionBackend()
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=values,
            req_pool_indices=None,
            seq_lens=None,
            out_cache_loc=None,
            seq_lens_sum=0,
        )
        inputs = (values, None, forward_batch)
        with (
            patch.object(dp_attention, "_RANK_TENSORS", cache),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            compiled, warmup_async = policy._direct_compile(model.forward)
            torch.testing.assert_close(
                warmup_async(*inputs).wait().cpu(),
                model(*inputs).cpu(),
            )
            # Reuse one executable while changing both scalar rank buffers.
            # Constants baked during tracing would return the first rank's result.
            for tp_rank, dp_rank in ((0, 0), (0, 1), (1, 0), (1, 1)):
                ranks["attn_tp"].fill_(tp_rank)
                ranks["attn_dp"].fill_(dp_rank)
                expected = model(*inputs).cpu()
                actual = compiled(*inputs).cpu()
                torch.testing.assert_close(actual, expected)
                self.assertIs(dp_attention._RANK_TENSORS, cache)


if __name__ == "__main__":
    unittest.main()
