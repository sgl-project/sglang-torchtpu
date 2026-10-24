"""Unit tests for TpuGraphRunner.

Mocks torch.compile and the tpu dynamo backend registration so the
tests run on any machine without TPU hardware.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from sglang.srt.compilation.support_torch_compile import (
    CompilePolicy,
    support_torch_compile,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy import (
    BaseSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compilation.spec_decode_compile_policy_eagle3 import (
    Eagle3SpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compilation.warmup_queue import collect_warmups
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.compilation.block_compile_policy import (
    TpuBlockSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    TpuChainSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.graph_runner import tpu_graph_runner
from sglang_tpu.graph_runner.tpu_graph_runner import (
    TpuGraphRunner,
    _augment_spec_policy,
)

DECODER_POLICY_PRECOMPILE = (
    "sglang_tpu.compilation.decoder_compile_policy.DecoderTpuCompilePolicy.precompile"
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
        # Publish a dummy config: init warms the CP singleton and policy install() reads the
        # compile grid off the bags (chunked_prefill_size sets max_prefill_buffer_tokens to 2048).
        override = get_context().override_server_args(
            torch_compile_max_bs=8,
            chunked_prefill_size=2048,
            max_prefill_tokens=2048,
        )
        override.install()
        self.addCleanup(override.restore)
        # Flush init's lazy imports before the torch.compile patch: their
        # import-time @torch.compile decorators must not hit the recorder.
        import sglang.kernels.ops.lora.moe.virtual_experts  # noqa: F401
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
            "sglang_tpu.compilation.decoder_compile_policy.torch.compile",
            side_effect=_fake_compile,
        )
        self.mock_compile = compile_patcher.start()
        self.addCleanup(compile_patcher.stop)
        direct_patcher = patch(
            "sglang_tpu.compilation."
            "decoder_compile_policy.DecoderTpuCompilePolicy._direct_compile",
            return_value=(
                MagicMock(name="direct_compiled"),
                MagicMock(name="direct_warmup_async"),
            ),
        )
        self.mock_direct_compile = direct_patcher.start()
        self.addCleanup(direct_patcher.stop)
        # Stub the backend-registration import so the test doesn't need torch_tpu.
        register_patcher = patch("sglang_tpu.patches._register_tpu_compile_backend")
        self.mock_register = register_patcher.start()
        self.addCleanup(register_patcher.stop)
        self.mock_cache_key = enter_scope(
            self,
            patch(
                "sglang_tpu.graph_runner.tpu_graph_runner."
                "add_tpu_rank_assignment_to_compile_cache_key"
            ),
        )
        # Stub the in-process precompile so the test doesn't need a real
        # attn_backend / model_runner.device. Decoder policy warmup behavior
        # is covered separately in test_decoder_tpu_compile_policy.py.
        precompile_patcher = patch(DECODER_POLICY_PRECOMPILE)
        self.mock_precompile = precompile_patcher.start()
        self.addCleanup(precompile_patcher.stop)
        dispatch_patcher = patch(
            "sglang_tpu.graph_runner.tpu_graph_runner._resolve_fused_op_dispatch"
        )
        dispatch_patcher.start()
        self.addCleanup(dispatch_patcher.stop)
        platform_patcher = patch(
            "sglang_tpu.graph_runner.tpu_graph_runner.current_platform",
            SimpleNamespace(device_type="tpu"),
        )
        platform_patcher.start()
        self.addCleanup(platform_patcher.stop)
        backend_patcher = patch(
            "sglang_tpu.compilation.decoder_compile_policy.current_platform",
            SimpleNamespace(get_compile_backend=lambda: "tpu"),
        )
        backend_patcher.start()
        self.addCleanup(backend_patcher.stop)
        # install() consults is_dp_attention_enabled() -> get_flags(), which
        # needs global server args set. No DP in these unit tests, so stub it.
        dp_patcher = patch(
            "sglang_tpu.compilation.decoder_compile_policy.is_dp_attention_enabled",
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
        # Only what production still reads off the handed instance.
        server_args = SimpleNamespace(
            enable_torch_compile=enable_torch_compile,
            tpu_compile_tracer=tpu_compile_tracer,
            unsafe_save_torch_compile_bundle=None,
            unsafe_load_torch_compile_bundle=None,
            tpu_compile_tokens=None,
            tpu_compile_bs=None,
            prefill_max_requests=None,
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
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            server_args=server_args,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
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

    def test_native_mtp_discovers_named_root_only(self):
        child = self._annotate(_model_with(forward=MagicMock()))
        child_forward = child.forward
        model = self._annotate(_model_with(forward=MagicMock()))
        model.named_modules = lambda: [("", model), ("backbone", child)]
        runner = self._runner_for(model)
        runner.is_draft_worker = True
        runner.spec_algorithm = SpeculativeAlgorithm.EAGLE
        # Exercise discovery independently of device-specific speculative tracing.
        with patch(
            "sglang_tpu.graph_runner.tpu_graph_runner._augment_spec_policy",
            side_effect=lambda cls, _: cls,
        ):
            TpuGraphRunner(runner)
        self.assertIs(runner.compile_policy.owner, model)
        self.assertEqual(runner.compile_policy.name, "draft/forward")
        self.assertIs(child.forward, child_forward)

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

    def test_queues_all_modules_before_waiting_for_warmups(self):
        events = []
        modules = [self._annotate(_model_with(forward=MagicMock())) for _ in range(2)]
        model = _model_with(
            named_modules=lambda: [(str(i), module) for i, module in enumerate(modules)]
        )

        def precompile():
            index = len(events)
            events.append(("submit", index))
            return [PendingCall(lambda: events.append(("wait", index)))]

        self.mock_precompile.side_effect = precompile
        TpuGraphRunner(self._runner_for(model, tpu_compile_tracer="direct"))
        self.assertEqual(
            events, [("submit", 0), ("submit", 1), ("wait", 0), ("wait", 1)]
        )

    def test_global_queue_submits_all_runners_before_waiting(self):
        events = []

        def precompile():
            index = len(events)
            events.append(("submit", index))
            return [PendingCall(lambda: events.append(("wait", index)))]

        self.mock_precompile.side_effect = precompile
        with collect_warmups() as queue:
            for _ in range(3):
                module = self._annotate(_model_with(forward=MagicMock()))
                TpuGraphRunner(
                    self._runner_for(
                        _model_with(named_modules=lambda: [("decoder", module)])
                    )
                )
            self.assertEqual(events, [("submit", i) for i in range(3)])
        self.assertEqual(
            events, [("submit", i) for i in range(3)] + [("wait", i) for i in range(3)]
        )
        self.assertEqual(queue.calls, [])
        self.assertEqual(queue.completions, [])

    def test_global_queue_defers_recompilation_checker_install(self):
        events = []
        self.mock_precompile.side_effect = lambda: [
            PendingCall(lambda: events.append("wait"))
        ]
        checker = MagicMock()
        checker.snapshot_baseline.side_effect = lambda: events.append("snapshot")
        model = self._annotate(_model_with(forward=MagicMock()))
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch(
                "sglang_tpu.compilation.recompilation_checker."
                "_shared_recompilation_checker",
                return_value=checker,
            ),
            patch(
                "sglang_tpu.compilation."
                "decoder_compile_policy.DecoderTpuCompilePolicy."
                "set_recompilation_checker",
                side_effect=lambda *a, **k: events.append("install"),
            ),
        ):
            with collect_warmups():
                TpuGraphRunner(self._runner_for(model))
                self.assertEqual(events, [])
        self.assertEqual(events, ["wait", "snapshot", "install"])

    def test_precompile_done_marker_follows_warmup_completion(self):
        for global_queue in (False, True):
            for async_compile in (False, True):
                with self.subTest(
                    global_queue=global_queue, async_compile=async_compile
                ):
                    events = []

                    def precompile():
                        if async_compile:
                            return [PendingCall(lambda: events.append("execute"))]
                        events.append("execute")
                        return []

                    def record_log(message, *args):
                        if "precompile done" in message:
                            events.append("done")

                    self.mock_precompile.side_effect = precompile
                    model = self._annotate(_model_with(forward=MagicMock()))
                    scope = collect_warmups() if global_queue else nullcontext()
                    with patch(
                        "sglang_tpu.graph_runner.tpu_graph_runner.logger.info",
                        side_effect=record_log,
                    ):
                        with scope:
                            TpuGraphRunner(self._runner_for(model))
                            if global_queue:
                                self.assertNotIn("done", events)
                    self.assertEqual(events, ["execute", "done"])

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
            self.mock_cache_key.assert_called_once_with()
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

    def test_experimental_worker_takes_eagle_family_to_the_chain_policy(self):
        for algo, name in (
            (SpeculativeAlgorithm.EAGLE, "EAGLE"),
            (SpeculativeAlgorithm.EAGLE3, "EAGLE3"),
        ):
            mr = SimpleNamespace(
                spec_algorithm=algo,
                server_args=SimpleNamespace(
                    speculative_num_steps=3, speculative_algorithm=name
                ),
            )
            cls = _augment_spec_policy(DecoderTpuCompilePolicy, mr)
            self.assertTrue(issubclass(cls, TpuChainSpecDecodeCompilePolicyMixin))
            self.assertFalse(issubclass(cls, Eagle3SpecDecodeCompilePolicyMixin))

    def test_experimental_worker_takes_dflash_to_the_block_policy(self):
        mr = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="DFLASH"
            ),
        )
        cls = _augment_spec_policy(DecoderTpuCompilePolicy, mr)
        self.assertTrue(issubclass(cls, TpuBlockSpecDecodeCompilePolicyMixin))
        self.assertFalse(issubclass(cls, TpuChainSpecDecodeCompilePolicyMixin))

    def test_augment_spec_policy_mixes_spec_over_base_for_eagle3(self):
        # EAGLE3 spec composes the Eagle3 spec mixin over the resolved policy
        # when the experimental spec worker is disabled.
        mr = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="EAGLE3"
            ),
        )
        with tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            cls = _augment_spec_policy(DecoderTpuCompilePolicy, mr)
        self.assertTrue(issubclass(cls, Eagle3SpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, BaseSpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, DecoderTpuCompilePolicy))
        # Idempotent: a policy that already implements spec is left untouched.
        self.assertIs(_augment_spec_policy(cls, mr), cls)

    def test_augment_spec_policy_mixes_spec_over_base_for_native_mtp(self):
        # EAGLE spec composes the Eagle3 spec mixin over the resolved policy.
        mr = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="EAGLE"
            ),
        )
        with tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            cls = _augment_spec_policy(DecoderTpuCompilePolicy, mr)
        self.assertTrue(issubclass(cls, Eagle3SpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, BaseSpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, DecoderTpuCompilePolicy))
        # Idempotent: a policy that already implements spec is left untouched.
        self.assertIs(_augment_spec_policy(cls, mr), cls)

    def test_augment_spec_policy_mixes_spec_over_base_for_dflash(self):
        # DFLASH spec composes the DFlash spec mixin over the resolved policy
        # when the experimental spec worker is disabled.
        mr = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            server_args=SimpleNamespace(
                speculative_num_steps=3, speculative_algorithm="DFLASH"
            ),
        )
        with tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            cls = _augment_spec_policy(DecoderTpuCompilePolicy, mr)
        self.assertTrue(issubclass(cls, DFlashSpecDecodeCompilePolicyMixin))
        self.assertFalse(issubclass(cls, TpuBlockSpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, BaseSpecDecodeCompilePolicyMixin))
        self.assertTrue(issubclass(cls, DecoderTpuCompilePolicy))
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
            BaseSpecDecodeCompilePolicyMixin, DecoderTpuCompilePolicy
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

    def test_checker_baselines_after_model_precompile(self):
        """Model warmup is excluded before the checker is attached."""
        events = []
        self.mock_precompile.side_effect = lambda *a, **k: events.append("precompile")
        checker = MagicMock()
        checker.snapshot_baseline.side_effect = lambda: events.append("snapshot")
        model = self._annotate(_model_with(forward=MagicMock()))
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch(
                "sglang_tpu.compilation.recompilation_checker."
                "_shared_recompilation_checker",
                return_value=checker,
            ),
            patch(
                "sglang_tpu.compilation."
                "decoder_compile_policy.DecoderTpuCompilePolicy.set_recompilation_checker",
                side_effect=lambda *a, **k: events.append("install"),
            ),
        ):
            TpuGraphRunner(self._runner_for(model))
        self.assertEqual(events, ["precompile", "snapshot", "install"])

    def test_checker_is_not_armed_unless_opted_in(self):
        model = self._annotate(_model_with(forward=MagicMock()))
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(False),
            patch(
                "sglang_tpu.compilation.recompilation_checker."
                "_shared_recompilation_checker"
            ) as shared,
        ):
            TpuGraphRunner(self._runner_for(model))
        shared.assert_not_called()

    def test_draft_runner_leaves_arming_to_its_deferred_precompile(self):
        """The draft precompiles after init, so arming here would report it."""
        model = self._annotate(_model_with(forward=MagicMock()))
        runner_args = self._runner_for(model)
        runner_args.is_draft_worker = True
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch(
                "sglang_tpu.compilation.recompilation_checker."
                "_shared_recompilation_checker"
            ) as shared,
        ):
            TpuGraphRunner(runner_args)
        shared.assert_not_called()

    def test_precompile_called_for_installed_policy(self):
        model = self._annotate(_model_with(forward=MagicMock()))
        TpuGraphRunner(self._runner_for(model))
        self.mock_precompile.assert_called_once()

    def test_discovers_annotated_module(self):
        events = []
        self.mock_cache_key.side_effect = lambda: events.append("cache-key")

        class _VisionPolicy(CompilePolicy):
            platform = "tpu"

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

            def install(self) -> bool:
                events.append("install")
                self.owner.installed = True
                return True

            def precompile(self) -> None:
                events.append("precompile")
                self.owner.precompiled = True

        class _VisionModule:
            def forward(self):
                pass

        support_torch_compile(
            _VisionModule,
            policy=_VisionPolicy,
        )
        for is_draft_worker in (False, True):
            vision_module = _VisionModule()
            model = _model_with()
            model.named_modules = lambda: (("", model), ("vision", vision_module))
            runner = self._runner_for(model)
            runner.is_draft_worker = is_draft_worker
            TpuGraphRunner(runner)
            self.assertTrue(vision_module.installed)
            self.assertTrue(vision_module.precompiled)
        self.assertEqual(events, ["cache-key", "install", "precompile"] * 2)
        self.mock_compile.assert_not_called()

    def test_warns_when_only_non_decoder_policies_install(self):
        # A vision-only annotation skips the whole-model fallback, so the
        # decoder runs eager; the runner must say so instead of staying silent.
        class _VisionPolicy(CompilePolicy):
            platform = "tpu"

            def install(self) -> bool:
                return True

            def precompile(self) -> None:
                pass

        class _VisionModule:
            def forward(self):
                pass

        support_torch_compile(_VisionModule, policy=_VisionPolicy)
        model = _model_with()
        model.named_modules = lambda: (("", model), ("vision", _VisionModule()))
        with self.assertLogs(tpu_graph_runner.logger, level="WARNING") as logs:
            TpuGraphRunner(self._runner_for(model))
        self.assertTrue(
            any("decode steps run eager" in line for line in logs.output), logs.output
        )
        self.mock_compile.assert_not_called()

    def test_decoder_policy_install_does_not_warn(self):
        model = self._annotate(_model_with(forward=MagicMock()))
        with self.assertNoLogs(tpu_graph_runner.logger, level="WARNING"):
            TpuGraphRunner(self._runner_for(model))

    def test_invalid_assignment_prevents_compilation(self):
        self.mock_cache_key.side_effect = RuntimeError("Invalid TPU rank assignment")
        model = self._annotate(_model_with(forward=MagicMock()))
        with self.assertRaisesRegex(RuntimeError, "Invalid TPU rank assignment"):
            TpuGraphRunner(self._runner_for(model))
        self.mock_compile.assert_not_called()
        self.mock_direct_compile.assert_not_called()
        self.mock_precompile.assert_not_called()

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
        model.named_modules = lambda: (("", model), ("foreign", foreign_module))
        runner = self._runner_for(model)

        TpuGraphRunner(runner)

        self.assertFalse(hasattr(foreign_module, "installed"))
        self.assertIsInstance(runner.compile_policy, DecoderTpuCompilePolicy)
        self.mock_compile.assert_called_once()
        self.mock_direct_compile.assert_not_called()

    def test_dynamic_compile_unwrap_updates_repetition_penalty_alias(self):
        from sglang_tpu import patches as tpu_patches

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

        with (
            patch.object(
                tpu_patches.importlib, "import_module", side_effect=import_module
            ),
            patch.object(tpu_patches, "_dynamic_compile_unwrapped", False),
        ):
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

    def setUp(self):
        # capture_cuda_graphs reads the draft MoE backends and the symmetric-memory switch
        # off the bags, so the case runs against a published config.
        super().setUp()
        override = get_context().override_server_args(moe_runner_backend="triton")
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _runner(device):
        return SimpleNamespace(
            device=device,
            is_draft_worker=True,
            model=object(),
            model_config=SimpleNamespace(quantization=None),
            # Only what production still reads off the handed record.
            server_args=SimpleNamespace(forward_hooks=[]),
            forward_stream=None,
            canary_manager=None,
        )

    def _call(self, device, enable_torch_compile):
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            capture_cuda_graphs,
        )
        from sglang.srt.runtime_context import get_flags

        runner = self._runner(device)
        with (
            patch(f"{self._SETUP_MOD}.GraphSharedOutput.create_for_model_runner"),
            patch(f"{self._SETUP_MOD}.EagerRunner"),
            patch(f"{self._SETUP_MOD}.capture_prefill_graph"),
            patch(f"{self._SETUP_MOD}.prealloc_symmetric_memory_pool"),
            patch(f"{self._SETUP_MOD}.capture_decode_graph") as decode_capture,
            get_flags().capture.override(enable_torch_compile=enable_torch_compile),
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

    def test_registers_tpu_forward_hooks_before_compilation(self):
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            capture_cuda_graphs,
        )
        from sglang.srt.runtime_context import get_flags

        runner = self._runner("tpu")
        events = []

        with (
            patch(f"{self._SETUP_MOD}.GraphSharedOutput.create_for_model_runner"),
            patch(f"{self._SETUP_MOD}.EagerRunner"),
            patch(
                f"{self._SETUP_MOD}.register_forward_hooks",
                side_effect=lambda *_: events.append("hooks"),
            ),
            patch(
                f"{self._SETUP_MOD}.capture_prefill_graph",
                side_effect=lambda **_: events.append("prefill") or MagicMock(),
            ),
            patch(f"{self._SETUP_MOD}.prealloc_symmetric_memory_pool"),
            patch(
                f"{self._SETUP_MOD}.capture_decode_graph",
                side_effect=lambda **_: events.append("decode") or MagicMock(),
            ),
            get_context().override_server_args(forward_hooks=[{"name": "capture"}]),
            get_flags().capture.override(enable_torch_compile=True),
        ):
            capture_cuda_graphs(model_runner=runner)

        self.assertEqual(events, ["hooks", "prefill", "decode"])


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
        from sglang.srt.runtime_context import get_flags

        # ExitStack rather than enterContext, which is 3.11+.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_flags().capture.override(enable_torch_compile=True))
        worker.server_args = SimpleNamespace(tpu_compile_tracer="dynamo")
        worker.draft_tp_context = lambda _group, **_kwargs: contextlib.nullcontext()
        worker.draft_owns_attention = False
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

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2."
                "speculative_moe_backend_context",
                side_effect=contextlib.nullcontext,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2."
                "speculative_moe_a2a_backend_context",
                side_effect=contextlib.nullcontext,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.check_cuda_graph_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2._is_tpu",
                True,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.draft_pp_context",
                side_effect=contextlib.nullcontext,
            ),
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

    def test_fused_draft_refreshes_dp_metadata_after_spec_padding(self):
        """The direct fused call must publish its final padded DP shape."""
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        events = []
        worker = object.__new__(EagleDraftWorker)
        worker.draft_runner = SimpleNamespace(
            compile_policy=SimpleNamespace(
                prepare_spec_forward_batch=lambda _batch: events.append("spec"),
                refresh_fused_draft_dp_metadata=lambda _batch: events.append("dp"),
            ),
        )
        worker.draft_attn_backend = SimpleNamespace(
            init_forward_metadata=lambda _batch: events.append("metadata")
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            mark_forward_metadata_ready=lambda: events.append("ready"),
        )

        worker._prepare_draft_forward_batch(batch, refresh_dp_metadata=True)

        self.assertEqual(events, ["spec", "dp", "metadata", "ready"])


if __name__ == "__main__":
    unittest.main()
