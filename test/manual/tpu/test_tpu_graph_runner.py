"""Unit tests for TpuGraphRunner.

Mocks torch.compile and the tpu dynamo backend registration so the
tests run on any machine without TPU hardware.
"""

import inspect
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from sglang.srt.compilation.support_torch_compile import (
    CompilePolicy,
    support_torch_compile,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    SpecDecodeCompilePolicyMixin,
)
from sglang.srt.model_executor.tpu_graph_runner import (
    TpuGraphRunner,
    _augment_spec_policy,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase

DECODER_POLICY_PRECOMPILE = (
    "sglang.srt.hardware_backend.tpu.compilation."
    "decoder_compile_policy.DecoderTpuCompilePolicy.precompile"
)


def _model_with(**attrs: Any) -> Any:
    """Return a bare object with the given attributes.

    A bare object is preferable here over MagicMock because MagicMock
    auto-creates attributes — so getattr(m, "forward_split_prefill",
    None) returns a Mock instead of None, and we can't distinguish
    "method present" from "method absent."
    """

    class _M:
        pass

    m = _M()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class TestTpuGraphRunner(CustomTestCase):
    """TpuGraphRunner orchestrates annotated compile policies."""

    def setUp(self):
        # Stub torch.compile so the test doesn't need the tpu dynamo backend.
        # Return a MagicMock that records the wrapped fn/kwargs and lets
        # tpu_graph_runner assign ``__signature__`` on the result (a tuple
        # would raise AttributeError on attribute set).
        def _fake_compile(fn: Any, **kwargs: Any) -> MagicMock:
            m = MagicMock(name="compiled")
            m._wrapped_fn = fn
            m._wrapped_kwargs = kwargs
            return m

        compile_patcher = patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "decoder_compile_policy.torch.compile",
            side_effect=_fake_compile,
        )
        self.mock_compile = compile_patcher.start()
        self.addCleanup(compile_patcher.stop)
        # Stub the backend-registration import so the test doesn't need torch_tpu.
        register_patcher = patch(
            "sglang.srt.hardware_backend.tpu.patches._register_tpu_compile_backend"
        )
        self.mock_register = register_patcher.start()
        self.addCleanup(register_patcher.stop)
        # Stub the in-process precompile so the test doesn't need a real
        # attn_backend / model_runner.device. Decoder policy warmup behavior
        # is covered separately in test_decoder_tpu_compile_policy.py.
        precompile_patcher = patch(DECODER_POLICY_PRECOMPILE)
        self.mock_precompile = precompile_patcher.start()
        self.addCleanup(precompile_patcher.stop)
        platform_patcher = patch(
            "sglang.srt.model_executor.tpu_graph_runner.current_platform",
            SimpleNamespace(device_type="tpu"),
        )
        platform_patcher.start()
        self.addCleanup(platform_patcher.stop)
        backend_patcher = patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "decoder_compile_policy.current_platform",
            SimpleNamespace(get_compile_backend=lambda: "tpu"),
        )
        backend_patcher.start()
        self.addCleanup(backend_patcher.stop)

    def _runner_for(
        self, model: Any, enable_torch_compile: bool = True
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=2048),
            max_running_requests=1,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            server_args=SimpleNamespace(
                enable_torch_compile=enable_torch_compile,
                tpu_compile_tokens=None,
                tpu_compile_bs=None,
                chunked_prefill_size=None,
                max_prefill_tokens=2048,
            ),
        )

    def _annotate(self, model: Any) -> Any:
        return support_torch_compile(
            model,
            policy=DecoderTpuCompilePolicy,
        )

    def test_wraps_annotated_forward_when_enabled(self):
        def original(
            input_ids: Any,
            positions: Any,
            forward_batch: Any,
            input_embeds: Any | None = None,
        ) -> None:
            pass

        model = self._annotate(_model_with(forward=original))
        runner = self._runner_for(model)
        TpuGraphRunner(runner)
        self.assertIsNot(model.forward, original)
        self.assertIn("input_embeds", inspect.signature(model.forward).parameters)
        self.assertIsInstance(runner.compile_policy, DecoderTpuCompilePolicy)
        self.mock_compile.assert_called_once()
        _, kwargs = self.mock_compile.call_args
        # Pinned compile kwargs — these are correctness-critical for the
        # TPU backend (dynamic=True triggers recompiles per shape;
        # fullgraph=False allows torch.compile to fall back to eager on
        # unsupported ops, which silently breaks the XLA pipeline).
        self.assertEqual(kwargs["backend"], "tpu")
        self.assertTrue(kwargs["fullgraph"])
        self.assertFalse(kwargs["dynamic"])
        self.mock_register.assert_called_once()

    def test_module_annotation_wraps_forward_only(self):
        model = self._annotate(
            _model_with(
                forward=MagicMock(),
                forward_split_prefill=MagicMock(),
            ),
        )
        original_split = model.forward_split_prefill
        TpuGraphRunner(self._runner_for(model))
        self.assertEqual(self.mock_compile.call_count, 1)
        self.assertIs(model.forward_split_prefill, original_split)

    def test_respects_can_torch_compile_false(self):
        original = MagicMock()
        model = _model_with(forward=original)
        model._can_torch_compile = False
        runner = self._runner_for(model)
        TpuGraphRunner(runner)
        # forward was not replaced; torch.compile and registration both skipped.
        self.assertIs(model.forward, original)
        self.mock_compile.assert_not_called()
        self.mock_register.assert_not_called()
        # The skip is reflected back so downstream warmup doesn't try
        # to specialize over compile-only buckets.
        self.assertFalse(runner.server_args.enable_torch_compile)

    def test_annotation_drives_compile_when_can_torch_compile_default(self):
        # Models that don't set _can_torch_compile should compile only when
        # they declare a boundary.
        model = self._annotate(_model_with(forward=MagicMock()))
        TpuGraphRunner(self._runner_for(model))
        self.mock_compile.assert_called_once()

    def test_no_annotation_falls_back_to_decoder_policy(self):
        # Most text decoder models should not need an explicit annotation.
        # If no annotated boundary is installed, TPU compile falls back to
        # the generic decoder policy on the model forward.
        model = _model_with(forward=MagicMock())
        runner = self._runner_for(model)
        TpuGraphRunner(runner)
        self.mock_compile.assert_called_once()
        self.assertIsInstance(runner.compile_policy, DecoderTpuCompilePolicy)
        # Backend registration still happens — the gate is _can_torch_compile.
        self.mock_register.assert_called_once()

    def test_augment_spec_policy_mixes_spec_over_base_for_eagle3(self):
        # EAGLE3 spec composes the spec mixin over the resolved policy (both the
        # fallback and a model's own annotated policy go through this).
        mr = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="EAGLE3"
            ),
        )
        cls = _augment_spec_policy(DecoderTpuCompilePolicy, mr)
        self.assertTrue(issubclass(cls, SpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, DecoderTpuCompilePolicy))
        # Idempotent: a policy that already implements spec is left untouched.
        self.assertIs(_augment_spec_policy(cls, mr), cls)

    def test_augment_spec_policy_noop_without_spec(self):
        # No speculative decoding -> plain (unmixed) policy.
        no_spec = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NONE, server_args=SimpleNamespace()
        )
        self.assertIs(
            _augment_spec_policy(DecoderTpuCompilePolicy, no_spec),
            DecoderTpuCompilePolicy,
        )

    def test_augment_spec_policy_is_algorithm_agnostic(self):
        # Any spec run gets the spec mixin regardless of algorithm; the mixin
        # enforces the supported algorithm at construction (non-EAGLE3 raises,
        # covered in test_decoder_tpu_compile_policy), rather than silently
        # falling back to a non-spec policy here.
        ngram = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NGRAM,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="NGRAM"
            ),
        )
        self.assertTrue(
            issubclass(
                _augment_spec_policy(DecoderTpuCompilePolicy, ngram),
                SpecDecodeCompilePolicyMixin,
            )
        )

    def test_precompile_called_for_installed_policy(self):
        model = self._annotate(_model_with(forward=MagicMock()))
        TpuGraphRunner(self._runner_for(model))
        self.mock_precompile.assert_called_once()

    def test_discovers_annotated_module(self):
        class _VisionPolicy(CompilePolicy):
            platform = "tpu"

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

            def install(self) -> bool:
                self.owner.installed = True
                return True

            def precompile(self) -> None:
                self.owner.precompiled = True

            def unpad_outputs(self, out: Any, context: Any) -> Any:
                return out

        class _VisionModule:
            def forward(self):
                pass

        support_torch_compile(
            _VisionModule,
            policy=_VisionPolicy,
        )
        vision_module = _VisionModule()
        model = _model_with()
        model.modules = lambda: (model, vision_module)

        TpuGraphRunner(self._runner_for(model))

        self.assertTrue(vision_module.installed)
        self.assertTrue(vision_module.precompiled)
        self.mock_compile.assert_not_called()

    def test_ignores_policy_for_another_platform(self):
        class _CudaPolicy(CompilePolicy):
            platform = "cuda"

            def install(self) -> bool:
                self.owner.installed = True
                return True

            def precompile(self) -> None:
                self.owner.precompiled = True

            def unpad_outputs(self, out: Any, context: Any) -> Any:
                return out

        foreign_module = _model_with(forward=MagicMock())
        support_torch_compile(foreign_module, policy=_CudaPolicy)
        model = _model_with(forward=MagicMock())
        model.modules = lambda: (model, foreign_module)
        runner = self._runner_for(model)

        TpuGraphRunner(runner)

        self.assertFalse(hasattr(foreign_module, "installed"))
        self.assertIsInstance(runner.compile_policy, DecoderTpuCompilePolicy)
        self.mock_compile.assert_called_once()

    def test_dynamic_compile_unwrap_updates_repetition_penalty_alias(self):
        from sglang.srt.hardware_backend.tpu import patches as tpu_patches

        repetition_path = "sglang.srt.sampling.penaltylib.repetition_penalty"
        sampling_path = "sglang.srt.sampling.sampling_batch_info"
        target = "apply_scaling_penalties"

        def eager(logits, penalties):
            return logits, penalties

        compiled = MagicMock()
        compiled.__wrapped__ = eager
        modules = {
            repetition_path: SimpleNamespace(apply_scaling_penalties=compiled),
            sampling_path: SimpleNamespace(apply_scaling_penalties=compiled),
        }

        def import_module(path):
            return modules.get(path, SimpleNamespace())

        with patch.object(
            tpu_patches.importlib, "import_module", side_effect=import_module
        ), patch.object(tpu_patches, "_dynamic_compile_unwrapped", False):
            tpu_patches._unwrap_dynamic_compile_fns()

        self.assertIs(getattr(modules[repetition_path], target), eager)
        self.assertIs(getattr(modules[sampling_path], target), eager)

    def test_can_run_is_always_false(self):
        # ModelRunner.forward branches on `graph_runner.can_run(...)`
        # to decide whether to replay a captured graph. TPU has no
        # captured graph (PJRT caches HLO internally), so this must
        # be False for every forward mode so the replay path stays off.
        model = self._annotate(_model_with(forward=MagicMock()))
        runner = TpuGraphRunner(self._runner_for(model))
        self.assertFalse(runner.can_run(MagicMock()))


if __name__ == "__main__":
    unittest.main()
