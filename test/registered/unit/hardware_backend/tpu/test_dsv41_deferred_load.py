# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 deferred weight load.

Guards that the raw parameters recorded before the post-load pass come back
(shape, dtype, loader attributes, module attributes) after packing, that
parameters the pass added are dropped, that the dummy initialization leaves
host-resident tensors alone, which configurations may defer, and that every
rank makes the same single collective call whichever rank fails.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.environ import envs as core_envs
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.models.deepseek_v41 import deferred_load


def _loader(param, loaded):
    param.data.copy_(loaded)


class _QuantLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method = object()
        self.weight_block_size = (128, 128)
        self.weight = nn.Parameter(
            torch.zeros(8, 4, dtype=torch.uint8), requires_grad=False
        )
        self.weight.weight_loader = _loader
        self.weight_scale_inv = nn.Parameter(torch.ones(2, 1), requires_grad=False)


def _post_load_pass(layer):
    layer.weight = nn.Parameter(
        torch.zeros(2, 4, 4, dtype=torch.uint8), requires_grad=False
    )
    del layer._parameters["weight_scale_inv"]
    layer.weight_scale = nn.Parameter(torch.ones(4), requires_grad=False)
    layer.weight_block_size = None


class TestRecordRestore(CustomTestCase):
    def test_restore_brings_back_the_raw_parameters(self):
        model = nn.Module()
        model.layer = _QuantLinear()
        deferred_load.record_raw_params(model)
        _post_load_pass(model.layer)

        deferred_load.restore_raw_params(model)
        layer = model.layer
        self.assertEqual(tuple(layer.weight.shape), (8, 4))
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertIs(layer.weight.weight_loader, _loader)
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (2, 1))
        self.assertNotIn("weight_scale", layer._parameters)
        self.assertEqual(layer.weight_block_size, (128, 128))

    def test_same_shape_rebind_gets_the_loader_attributes_back(self):
        model = nn.Module()
        model.layer = _QuantLinear()
        deferred_load.record_raw_params(model)
        # A post-load pass that rebinds at the same shape drops weight_loader.
        model.layer.weight = nn.Parameter(
            torch.zeros(8, 4, dtype=torch.uint8), requires_grad=False
        )
        self.assertFalse(hasattr(model.layer.weight, "weight_loader"))
        deferred_load.restore_raw_params(model)
        self.assertIs(model.layer.weight.weight_loader, _loader)

    def test_parameters_on_later_submodules_are_dropped(self):
        model = nn.Module()
        model.layer = _QuantLinear()
        deferred_load.record_raw_params(model)
        # post_load_weights-style fusion on a submodule created after init.
        model.layer.fused = nn.Module()
        model.layer.fused.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        deferred_load.restore_raw_params(model)
        self.assertNotIn("weight", model.layer.fused._parameters)
        self.assertTrue(
            all(
                hasattr(p, "weight_loader")
                for n, p in model.named_parameters()
                if n == "layer.weight"
            )
        )

    def test_parameter_subclass_and_its_loader_property_survive(self):
        from sglang.srt.layers.parameter import BasevLLMParameter

        model = nn.Module()
        model.layer = _QuantLinear()
        model.layer.gate_up = BasevLLMParameter(
            data=torch.zeros(6, 4), weight_loader=_loader
        )
        deferred_load.record_raw_params(model)
        # The post-load pass rebinds it as a plain Parameter of a new shape.
        model.layer.gate_up = nn.Parameter(torch.zeros(4, 6), requires_grad=False)
        deferred_load.restore_raw_params(model)
        restored = model.layer.gate_up
        self.assertIsInstance(restored, BasevLLMParameter)
        self.assertEqual(tuple(restored.shape), (6, 4))
        self.assertIs(restored.weight_loader, _loader)

    def test_unchanged_parameters_are_kept(self):
        model = nn.Module()
        model.layer = _QuantLinear()
        kept = model.layer.weight_scale_inv
        deferred_load.record_raw_params(model)
        deferred_load.restore_raw_params(model)
        self.assertIs(model.layer.weight_scale_inv, kept)


class TestDummyInitSkipsHostTensors(CustomTestCase):
    def test_device_only_view_drops_host_tensors(self):
        model = nn.Module()
        model.host_table = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.assertEqual(list(deferred_load.device_only(model).state_dict()), [])


def _v41(is_draft=False):
    model = nn.Module()
    model.config = SimpleNamespace(model_type="deepseek_v41")
    model.w = nn.Parameter(torch.zeros(2), requires_grad=False)
    return model, SimpleNamespace(
        is_draft_model=is_draft, hf_config=SimpleNamespace(model_type="deepseek_v41")
    )


def _spmd(enabled=True, tracer="direct", torch_compile=True):
    graph = SimpleNamespace(
        use_spmd_compile=enabled,
        tpu_compile_tracer=tracer,
        enable_torch_compile=torch_compile,
    )
    return mock.patch(
        "sglang.srt.runtime_context.get_exec", lambda: SimpleNamespace(graph=graph)
    )


class TestShouldDefer(CustomTestCase):
    def setUp(self):
        patcher = _spmd()
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        deferred_load._DEFERRED.clear()

    def _should(self, cfg, load_format="runai_streamer", switch=True):
        load_config = SimpleNamespace(load_format=load_format, draft_model_idx=None)
        with (
            envs.SGLANG_TPU_DSV41_DEFERRED_LOAD.override(switch),
            mock.patch.object(deferred_load, "_deferred_in_process", False),
        ):
            return deferred_load.should_defer(cfg, load_config)

    def test_defers_the_target(self):
        self.assertTrue(self._should(_v41()[1]))

    def test_dummy_has_nothing_to_defer(self):
        with self.assertRaises(ValueError):
            self._should(_v41()[1], load_format="dummy")

    def test_drafts_load_normally(self):
        self.assertFalse(self._should(_v41(is_draft=True)[1]))
        self.assertFalse(self._should(_v41(is_draft=True)[1], load_format="dummy"))

    def test_only_the_first_load_in_a_process(self):
        with (
            envs.SGLANG_TPU_DSV41_DEFERRED_LOAD.override(True),
            mock.patch.object(deferred_load, "_deferred_in_process", True),
        ):
            load_config = SimpleNamespace(
                load_format="runai_streamer", draft_model_idx=None
            )
            self.assertFalse(deferred_load.should_defer(_v41()[1], load_config))

    def test_inert_without_the_switch_or_for_other_models(self):
        self.assertFalse(self._should(_v41()[1], switch=False))
        other = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
        self.assertFalse(self._should(other))

    def test_async_warmup_is_required(self):
        with core_envs.SGLANG_DIRECT_COMPILE_ASYNC.override(False):
            with self.assertRaises(ValueError):
                self._should(_v41()[1])

    def test_compile_settings_are_required(self):
        for patch, flag in (
            (_spmd(False), "--use-spmd-compile"),
            (_spmd(tracer="dynamo"), "--tpu-compile-tracer direct"),
            (_spmd(torch_compile=False), "--enable-torch-compile"),
        ):
            with patch, self.assertRaises(ValueError) as ctx:
                self._should(_v41()[1])
            self.assertIn(flag, str(ctx.exception))

    def test_defer_keeps_the_real_loader(self):
        model, _ = _v41()
        with mock.patch.object(deferred_load, "_deferred_in_process", False):
            deferred_load.defer(model, "real-loader")
            self.assertTrue(deferred_load._deferred_in_process)
        self.assertEqual(deferred_load._DEFERRED[id(model)].loader, "real-loader")


class TestCoordinatedLoad(CustomTestCase):
    def tearDown(self):
        deferred_load._DEFERRED.clear()

    def test_uncompiled_model_fails_every_rank(self):
        model, _ = _v41()
        deferred_load.record_raw_params(model)  # never registered by the tracer
        with self.assertRaises(RuntimeError) as ctx:
            deferred_load.load_pending(lambda v: [v, None])
        self.assertIn("never compiled", str(ctx.exception))
        self.assertEqual(deferred_load._DEFERRED, {})

    def test_register_keeps_the_runner_and_its_compiled_functions(self):
        model, _ = _v41()
        deferred_load.record_raw_params(model)
        runner = SimpleNamespace(model=model)
        compiler = SimpleNamespace(before_distribute=[])
        queue = SimpleNamespace(spmd_compiler=compiler)
        with mock.patch(
            "sglang_tpu.compilation.warmup_queue.current_warmup_queue", lambda: queue
        ):
            deferred_load.register(runner, "fn0")
            deferred_load.register(runner, "fn1")
            # Another model's compile is not tracked.
            deferred_load.register(SimpleNamespace(model=nn.Module()), "other")
        state = deferred_load._DEFERRED[id(model)]
        self.assertIs(state.runner, runner)
        self.assertEqual(state.functions, ["fn0", "fn1"])
        self.assertEqual(compiler.before_distribute, [deferred_load.load_pending])

    def test_register_outside_an_spmd_scope_fails(self):
        model, _ = _v41()
        deferred_load.record_raw_params(model)
        with mock.patch(
            "sglang_tpu.compilation.warmup_queue.current_warmup_queue", lambda: None
        ):
            with self.assertRaises(RuntimeError):
                deferred_load.register(SimpleNamespace(model=model), "fn")

    def test_peer_failure_fails_this_rank_too(self):
        # Nothing pending here, but another rank reports a failure.
        with self.assertRaises(RuntimeError) as ctx:
            deferred_load.load_pending(lambda v: [v, "OSError: gcs read failed"])
        self.assertIn("gcs read failed", str(ctx.exception))


class _Table:
    layout = "shared"

    def __init__(self, barriers):
        self.group = SimpleNamespace(barrier=lambda: barriers.append(1))


class _Runner:
    def __init__(self, barriers):
        self.model = nn.Module()
        for name in ("a", "b", "c"):
            module = nn.Module()
            module.host_table = _Table(barriers)
            self.model.add_module(name, module)


class TestSingleCollective(CustomTestCase):
    """Every rank makes exactly one collective call, the final gather, whatever
    happens: the shared Engram tables' own barriers are deferred to it."""

    def setUp(self):
        self.calls, self.barriers = [], []
        self.runner = _Runner(self.barriers)
        model, _ = _v41()
        deferred_load.record_raw_params(model)
        state = next(iter(deferred_load._DEFERRED.values()))
        state.runner = self.runner
        self.groups = [m.host_table.group for m in self.runner.model.children()]

    def tearDown(self):
        deferred_load._DEFERRED.clear()

    def _gather(self, peer_error=None):
        def gather(value):
            self.calls.append(value)
            return [value, peer_error]

        return gather

    def _finish(self, tables, fail_after=None):
        # What Engram's finish_load does: barrier on dirty shared tables only.
        def load(state):
            for i, module in enumerate(self.runner.model.children()):
                if i in tables:
                    module.host_table.group.barrier()
                if i == fail_after:
                    raise RuntimeError("post-process failed")

        return load

    def _assert_one_gather_and_restored(self):
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.barriers, [])  # no real barrier ever ran
        self.assertEqual(
            [m.host_table.group for m in self.runner.model.children()], self.groups
        )

    def test_success(self):
        with mock.patch.object(deferred_load, "_load", self._finish({0, 1, 2})):
            deferred_load.load_pending(self._gather())
        self.assertEqual(self.calls, [None])
        self._assert_one_gather_and_restored()
        self.assertEqual(deferred_load._DEFERRED, {})

    def test_clean_table_then_local_failure(self):
        # Table 1 was never dirtied, so it skips its barrier; then rank fails.
        with mock.patch.object(deferred_load, "_load", self._finish({0, 2}, 2)):
            with self.assertRaises(RuntimeError):
                deferred_load.load_pending(self._gather())
        self._assert_one_gather_and_restored()
        self.assertIn("post-process failed", self.calls[0])

    def test_failure_before_any_table(self):
        def fail(state):
            raise OSError("gcs read failed")

        with mock.patch.object(deferred_load, "_load", fail):
            with self.assertRaises(RuntimeError):
                deferred_load.load_pending(self._gather())
        self._assert_one_gather_and_restored()

    def test_peer_failure_fails_this_rank(self):
        with mock.patch.object(deferred_load, "_load", self._finish({0, 1, 2})):
            with self.assertRaises(RuntimeError) as ctx:
                deferred_load.load_pending(self._gather("OSError: read"))
        self.assertIn("OSError: read", str(ctx.exception))
        self._assert_one_gather_and_restored()


if __name__ == "__main__":
    unittest.main()
