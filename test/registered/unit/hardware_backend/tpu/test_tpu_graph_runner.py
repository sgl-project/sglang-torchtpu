"""Unit tests for TpuGraphRunner.

Mocks torch.compile and the tpu dynamo backend registration so the
tests run on any machine without TPU hardware.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
    Eagle3CompilePolicyMixin,
    SpecDecodeCompilePolicyMixin,
)
from sglang.srt.model_executor.tpu_graph_runner import (
    TpuGraphRunner,
    _augment_spec_policy,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase

DECODER_POLICY_PRECOMPILE = (
    "sglang.srt.hardware_backend.tpu.compilation."
    "decoder_compile_policy.DecoderTpuCompilePolicy.precompile"
)


class _ServerArgs(SimpleNamespace):
    def override(self, _source: str, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


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
        # Publish a dummy config: init warms the CP singleton (parallel leaf read).
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        # Flush init's lazy imports before the torch.compile patch: their
        # import-time @torch.compile decorators must not hit the recorder.
        import sglang.kernels.ops.moe.virtual_experts  # noqa: F401
        import sglang.kernels.ops.speculative.cache_locs  # noqa: F401
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe  # noqa: F401
        import sglang.srt.layers.sampler  # noqa: F401
        import sglang.srt.layers.vocab_parallel_embedding  # noqa: F401
        import sglang.srt.managers.overlap_utils  # noqa: F401
        import sglang.srt.sampling.penaltylib.repetition_penalty  # noqa: F401
        import sglang.srt.speculative.spec_utils  # noqa: F401
        from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx

        get_cp_decode_attn_tp_ctx()

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
        direct_patcher = patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "decoder_compile_policy.DecoderTpuCompilePolicy._direct_compile",
            return_value=MagicMock(name="direct_compiled"),
        )
        self.mock_direct_compile = direct_patcher.start()
        self.addCleanup(direct_patcher.stop)
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
        # install() consults is_dp_attention_enabled() -> get_flags(), which
        # needs global server args set. No DP in these unit tests, so stub it.
        dp_patcher = patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "decoder_compile_policy.is_dp_attention_enabled",
            return_value=False,
        )
        dp_patcher.start()
        self.addCleanup(dp_patcher.stop)

    def _runner_for(
        self,
        model: Any,
        enable_torch_compile: bool = True,
        tpu_compile_tracer: str = "dynamo",
    ) -> SimpleNamespace:
        server_args = SimpleNamespace(
            enable_torch_compile=enable_torch_compile,
            tpu_compile_tracer=tpu_compile_tracer,
            unsafe_save_torch_compile_bundle=None,
            unsafe_load_torch_compile_bundle=None,
            tpu_compile_tokens=None,
            tpu_compile_bs=None,
            torch_compile_max_bs=8,
            prefill_max_requests=None,
            chunked_prefill_size=None,
            max_prefill_tokens=2048,
        )

        # Mirror ServerArgs.override: the _can_torch_compile=False path calls it
        # to flip enable_torch_compile off.
        def _override(source: str, **fields: Any) -> None:
            for k, v in fields.items():
                setattr(server_args, k, v)

        server_args.override = _override
        return SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=2048),
            max_running_requests=1,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            server_args=server_args,
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

    def test_prepare_forward_batch_composes_model_and_policy_hooks(self):
        model_prepare = MagicMock()
        model = self._annotate(
            _model_with(
                forward=MagicMock(),
                prepare_forward_batch=model_prepare,
            )
        )
        runner = self._runner_for(model)
        TpuGraphRunner(runner)
        policy_prepare = MagicMock()
        runner.compile_policy.prepare_forward_batch = policy_prepare
        forward_batch = object()

        model.prepare_forward_batch(forward_batch)

        model_prepare.assert_called_once_with(forward_batch)
        policy_prepare.assert_called_once_with(forward_batch)

    def test_respects_can_torch_compile_false(self):
        from sglang.srt.runtime_context import get_flags

        original = MagicMock()
        model = _model_with(forward=original)
        model._can_torch_compile = False
        runner = self._runner_for(model)
        with get_flags().capture.override(enable_torch_compile=True):
            TpuGraphRunner(runner)
            # forward was not replaced; compile and registration both skipped.
            self.assertIs(model.forward, original)
            self.mock_compile.assert_not_called()
            self.mock_register.assert_not_called()
            # The skip is reflected into the capture flag for downstream warmup.
            self.assertFalse(get_flags().capture.enable_torch_compile)

    def test_annotation_drives_compile_when_can_torch_compile_default(self):
        # Models that don't set _can_torch_compile should compile only when
        # they declare a boundary.
        model = self._annotate(_model_with(forward=MagicMock()))
        TpuGraphRunner(self._runner_for(model))
        self.mock_compile.assert_called_once()

    def test_no_annotation_uses_direct_tracer(self):
        # Most text decoder models should not need an explicit annotation.
        # If no annotated boundary is installed, TPU compile falls back to
        # the generic decoder policy on the model forward.
        model = _model_with(forward=MagicMock())
        runner = self._runner_for(model, tpu_compile_tracer="direct")
        TpuGraphRunner(runner)
        self.mock_direct_compile.assert_called_once()
        self.mock_compile.assert_not_called()
        self.assertIsInstance(runner.compile_policy, DecoderTpuCompilePolicy)
        self.assertEqual(runner.compile_policy.tracer, "direct")
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
        self.assertTrue(issubclass(cls, Eagle3CompilePolicyMixin))
        self.assertTrue(issubclass(cls, SpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, DecoderTpuCompilePolicy))
        # Idempotent: a policy that already implements spec is left untouched.
        self.assertIs(_augment_spec_policy(cls, mr), cls)

    def test_augment_spec_policy_noop_without_spec(self):
        # No speculative decoding -> plain (unmixed) policy.
        no_spec = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NONE,
            server_args=SimpleNamespace(),
        )
        self.assertIs(
            _augment_spec_policy(DecoderTpuCompilePolicy, no_spec),
            DecoderTpuCompilePolicy,
        )

    def test_augment_spec_policy_preserves_explicit_spec_policy(self):
        class _ExplicitSpecPolicy(
            SpecDecodeCompilePolicyMixin, DecoderTpuCompilePolicy
        ):
            pass

        mr = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="EAGLE3"
            ),
        )
        self.assertIs(
            _augment_spec_policy(_ExplicitSpecPolicy, mr), _ExplicitSpecPolicy
        )

    def test_augment_spec_policy_noop_for_other_algorithm(self):
        ngram = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NGRAM,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="NGRAM"
            ),
        )
        self.assertIs(
            _augment_spec_policy(DecoderTpuCompilePolicy, ngram),
            DecoderTpuCompilePolicy,
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

        foreign_module = _model_with(forward=MagicMock())
        support_torch_compile(foreign_module, policy=_CudaPolicy)
        model = _model_with(forward=MagicMock())
        model.modules = lambda: (model, foreign_module)
        runner = self._runner_for(model)

        TpuGraphRunner(runner)

        self.assertFalse(hasattr(foreign_module, "installed"))
        self.assertIsInstance(runner.compile_policy, DecoderTpuCompilePolicy)
        self.mock_compile.assert_called_once()
        self.mock_direct_compile.assert_not_called()

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
        # ModelRunner.forward branches on `graph_runner.can_run_graph(...)`
        # to decide whether to replay a captured graph. TPU has no
        # captured graph (PJRT caches HLO internally), so this must
        # be False for every forward mode so the replay path stays off.
        model = self._annotate(_model_with(forward=MagicMock()))
        runner = TpuGraphRunner(self._runner_for(model))
        self.assertFalse(runner.can_run_graph(MagicMock()))


class TestDraftTorchCompileInstall(CustomTestCase):
    """A runner that skips decode-graph capture must still be compiled on TPU.

    Speculative draft runners pass ``capture_decode_cuda_graph=False`` (they
    capture their own decode-style graphs). On TPU ``capture_decode_graph``
    only installs torch.compile, so that flag must not switch compilation off
    for the draft model (capture_cuda_graphs' tpu+compile elif branch).
    """

    _SETUP_MOD = "sglang.srt.model_executor.model_runner_components.cuda_graph_setup"

    @staticmethod
    def _runner(device):
        return SimpleNamespace(
            device=device,
            is_draft_worker=True,
            model=object(),
            model_config=SimpleNamespace(quantization=None),
            server_args=SimpleNamespace(
                speculative_moe_runner_backend=None,
                moe_runner_backend="triton",
                speculative_moe_a2a_backend=None,
                moe_a2a_backend="none",
                forward_hooks=[],
                enable_symm_mem=False,
            ),
            forward_stream=None,
            canary_manager=None,
        )

    def _call(self, device, enable_torch_compile):
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            capture_cuda_graphs,
        )
        from sglang.srt.runtime_context import get_flags

        runner = self._runner(device)
        with patch(
            f"{self._SETUP_MOD}.GraphSharedOutput.create_for_model_runner"
        ), patch(f"{self._SETUP_MOD}.EagerRunner"), patch(
            f"{self._SETUP_MOD}.capture_prefill_graph"
        ), patch(
            f"{self._SETUP_MOD}.prealloc_symmetric_memory_pool"
        ), patch(
            f"{self._SETUP_MOD}.capture_decode_graph"
        ) as decode_capture, get_flags().capture.override(
            enable_torch_compile=enable_torch_compile
        ):
            capture_cuda_graphs(model_runner=runner, capture_decode_cuda_graph=False)
        return decode_capture

    def test_installs_on_tpu_when_compile_enabled(self):
        decode_capture = self._call("tpu", True)
        decode_capture.assert_called_once()

    def test_noop_when_torch_compile_disabled(self):
        decode_capture = self._call("tpu", False)
        decode_capture.assert_not_called()

    def test_noop_off_tpu(self):
        decode_capture = self._call("cuda", True)
        decode_capture.assert_not_called()


class TestDraftPrecompileOrdering(CustomTestCase):
    """Draft precompile must run after the policy is installed.

    The scheduler runs every worker's ``init_attention_backends()`` before any
    ``init_cuda_graphs()``, and the draft policy is installed by the latter.
    Warming draft shapes from ``init_attention_backend`` would therefore always
    hit ``NoopCompilePolicy``.
    """

    def test_precompile_runs_after_draft_init_cuda_graphs(self):
        import contextlib

        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        events = []
        worker = object.__new__(EagleDraftWorker)
        worker.speculative_num_steps = 3
        worker.server_args = SimpleNamespace(tpu_compile_tracer="dynamo")
        worker.draft_tp_context = lambda _group: contextlib.nullcontext()
        worker.draft_runner = SimpleNamespace(
            tp_group=object(),
            canary_manager=None,
            init_prefill_cuda_graph=MagicMock(),
            compile_policy=SimpleNamespace(
                precompile_draft=MagicMock(
                    side_effect=lambda *_args: events.append("precompile")
                )
            ),
        )
        worker.draft_attn_backend = object()
        worker.draft_extend_attn_backend = object()
        worker.draft_worker = SimpleNamespace(
            init_cuda_graphs=MagicMock(
                side_effect=lambda **_kwargs: events.append("install")
            )
        )
        worker._capture_cuda_graphs = MagicMock(
            side_effect=lambda: events.append("capture")
        )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2." "speculative_moe_backend_context",
            side_effect=contextlib.nullcontext,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2."
            "speculative_moe_a2a_backend_context",
            side_effect=contextlib.nullcontext,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.check_cuda_graph_backend",
            return_value=False,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2._is_tpu",
            True,
        ):
            worker.init_cuda_graphs()

        self.assertEqual(events, ["install", "precompile", "capture"])
        worker.draft_worker.init_cuda_graphs.assert_called_once_with(
            capture_decode_cuda_graph=False
        )
        worker.draft_runner.compile_policy.precompile_draft.assert_called_once_with(
            worker.draft_attn_backend,
            worker.draft_extend_attn_backend,
            worker._fused_draft_forward,
        )


if __name__ == "__main__":
    unittest.main()
