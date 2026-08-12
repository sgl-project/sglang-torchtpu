"""Unit tests for the unquant MoE module on TPU
(:class:`sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod`).

Covers the two TPU-specific hooks -- load-time and run-time halves of
the same contract:

  1. ``process_weights_after_loading`` pre-transposes ``w13`` / ``w2``
     into the gmm_v2 layout. ``TestProcessWeightsAfterLoading``.
  2. ``forward_tpu`` routes the inference call through the Pallas
     fused-MoE kernel (``fused_moe_call``). ``TestForwardTpuContract``
     (direct) and ``TestForwardTpuViaDispatch`` (via
     ``MultiPlatformOp.forward``).

The kernel is mocked via the ``_MoeMockRunner`` harness so these tests
run on any host; TPU-side kernel numerics live in
``test_pallas_fused_moe.py``.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.test_utils import CustomTestCase


class _MethodWithFakeRunnerConfig(UnquantizedFusedMoEMethod):
    """``UnquantizedFusedMoEMethod`` with an injected ``moe_runner_config``.

    The real method gets its config via ``create_moe_runner`` during full
    ``FusedMoE`` layer setup; here we inject a SimpleNamespace so unit
    tests can vary ``activation`` / ``apply_router_weight_on_input``
    without standing up the whole MoE pipeline.
    """

    def __init__(self, *, activation="silu", apply_router_weight_on_input=False):
        # MultiPlatformOp.__init__ runs dispatch_forward, which resolves
        # off the *module-level* _is_* flags. The harness patches those
        # flags before constructing the method so the dispatch cache
        # reflects the intended platform.
        super().__init__()
        self.moe_runner_config = SimpleNamespace(
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            no_combine=False,
            num_fused_shared_experts=0,
        )


class _MoeMockRunner:
    """Test harness ("mock runner") for ``UnquantizedFusedMoEMethod.forward_tpu``.

    Encapsulates the setup that every forward_tpu test shares:

      * Patch ``fused_moe_call`` with a capturing fake so tests run on
        any host (no torch_tpu, no JAX).
      * Optionally pin ``_platform_key`` so ``BaseFusedOp`` resolves
        ``forward_tpu`` at ``__init__`` time. This enables tests that
        exercise the full dispatch -> forward_tpu chain via
        ``method.forward(...)``.
      * Build ``layer`` / ``dispatch_output`` / ``method`` on demand.

    Two invocation paths are exposed:

      * ``tpu_forward(layer, d_out)``: direct ``method.forward_tpu(...)``
        call. Isolates the method's contract from BaseFusedOp dispatch.
      * ``dispatched_forward(layer, d_out)``: goes through
        ``method.forward(...)`` which routes via
        ``BaseFusedOp._forward_method``. Confirms the dispatch wiring on
        top of the method's contract.

    Usage as a context manager (clean for variant-per-test configs):

        with _MoeMockRunner(activation="gelu") as r:
            layer = r.make_layer()
            d_out = r.make_dispatch_output()
            r.tpu_forward(layer, d_out)
            self.assertEqual(r.kernel_kwargs["activation"], "gelu")

    Usage from setUp/tearDown (clean for shared per-class config):

        def setUp(self):
            self.runner = _MoeMockRunner()
            self.runner.start()
        def tearDown(self):
            self.runner.stop()
    """

    def __init__(
        self,
        *,
        force_tpu_dispatch: bool = False,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        self.force_tpu_dispatch = force_tpu_dispatch
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self._capture: dict = {}
        self._fake_kernel_out: torch.Tensor | None = None
        self._patchers: list = []
        self.method: _MethodWithFakeRunnerConfig | None = None

    # ---- lifecycle ----------------------------------------------------

    def start(self):
        def fake_kernel(**kwargs):
            self._capture["kwargs"] = kwargs
            m, k = kwargs["hidden_states"].shape
            self._fake_kernel_out = torch.zeros(
                m, k, dtype=kwargs["hidden_states"].dtype
            )
            return self._fake_kernel_out

        # ``forward_tpu`` does ``from sglang.srt.layers.moe.pallas.fused_moe_op
        # import fused_moe_call`` inside the function body, so we shadow
        # the whole module in sys.modules with one that only exposes the
        # fake. Tests then run on any host.
        patchers = [
            patch.dict(
                "sys.modules",
                {
                    "sglang.srt.layers.moe.pallas.fused_moe_op": SimpleNamespace(
                        fused_moe_call=fake_kernel
                    ),
                },
            )
        ]
        if self.force_tpu_dispatch:
            # BaseFusedOp resolves the platform forward at __init__ time via
            # _platform_key() (the former MultiPlatformOp module-level _is_*
            # globals are gone, RFC #29630). Pin the key so ``method.forward``
            # routes to ``forward_tpu`` regardless of the test host, and pin
            # the OOT key to None so it does not short-circuit ahead of it.
            patchers.extend(
                [
                    patch(
                        "sglang.kernels.fused_op._platform_key",
                        return_value="tpu",
                    ),
                    patch(
                        "sglang.kernels.fused_op._oot_dispatch_key",
                        return_value=None,
                    ),
                ]
            )
        for p in patchers:
            p.start()
        self._patchers = patchers

        # Method must be built *after* patches are live so its
        # MultiPlatformOp.__init__ sees the patched flags.
        self.method = _MethodWithFakeRunnerConfig(
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        return self

    def stop(self):
        for p in reversed(self._patchers):
            p.stop()
        self._patchers = []
        self.method = None
        self._capture = {}
        self._fake_kernel_out = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---- input factories ---------------------------------------------

    def make_layer(
        self,
        *,
        num_experts: int = 4,
        hidden: int = 32,
        inter: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Build a minimal ``layer`` stand-in with gmm_v2-layout weights.

        Mirrors the post-``process_weights_after_loading`` state on TPU:
        ``w13_weight`` is (g, hidden, 2*inter) and ``w2_weight`` is
        (g, inter, hidden). ``forward_tpu`` consumes these unchanged, so
        we don't need a real ``nn.Module``.
        """
        layer = SimpleNamespace()
        layer.num_local_experts = num_experts
        layer.w13_weight = torch.nn.Parameter(
            torch.randn(num_experts, hidden, 2 * inter, dtype=dtype) * 0.05,
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randn(num_experts, inter, hidden, dtype=dtype) * 0.05,
            requires_grad=False,
        )
        return layer

    def make_dispatch_output(
        self,
        *,
        m: int = 8,
        hidden: int = 32,
        num_experts: int = 4,
        topk: int = 2,
        hidden_dtype: torch.dtype = torch.bfloat16,
        weights_dtype: torch.dtype = torch.float32,
        ids_dtype: torch.dtype = torch.int64,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> StandardDispatchOutput:
        """Build a StandardDispatchOutput with deliberately *mismatched* dtypes.

        Production routing produces fp32 ``topk_weights`` and int64
        ``topk_ids`` even when hidden_states is bf16; ``forward_tpu``
        owns the casts. Pre-building inputs in those dtypes ensures the
        cast logic is actually exercised.
        """
        hidden_t = torch.randn(m, hidden, dtype=hidden_dtype) * 0.5
        topk_w = torch.softmax(torch.randn(m, topk, dtype=weights_dtype), dim=-1)
        topk_ids = torch.randint(0, num_experts, (m, topk), dtype=ids_dtype)
        return StandardDispatchOutput(
            hidden_states=hidden_t,
            hidden_states_scale=hidden_states_scale,
            topk_output=(topk_w, topk_ids, None),
        )

    # ---- invocation paths --------------------------------------------

    def tpu_forward(self, layer, d_out):
        """Direct ``method.forward_tpu(layer, d_out)`` call."""
        # Single-rank EP: no distributed process group in a unit test.
        with patch(
            "sglang.srt.layers.quantization.unquant.get_moe_expert_parallel_rank",
            lambda: 0,
        ):
            return self.method.forward_tpu(layer, d_out)

    def dispatched_forward(self, layer, d_out):
        """``method.forward(layer, d_out)`` — routes through
        ``MultiPlatformOp._forward_method`` (the dispatch cache).
        Use this with ``force_tpu_dispatch=True`` to integration-test
        the dispatch -> forward_tpu chain.
        """
        with patch(
            "sglang.srt.layers.quantization.unquant.get_moe_expert_parallel_rank",
            lambda: 0,
        ):
            return self.method.forward(layer, d_out)

    # ---- captured state ----------------------------------------------

    @property
    def kernel_kwargs(self) -> dict:
        """The kwargs the fake kernel received on the most recent call."""
        return self._capture["kwargs"]

    @property
    def kernel_was_called(self) -> bool:
        return "kwargs" in self._capture


class TestProcessWeightsAfterLoading(CustomTestCase):
    """Load-time half of the MoE module contract on TPU.

    ``UnquantizedFusedMoEMethod.process_weights_after_loading`` swaps
    expert weights from SGLang storage layout to the layout
    ``gmm_v2`` consumes:

      * ``w13``: (g, 2*intermediate, hidden) -> (g, hidden, 2*intermediate)
      * ``w2``:  (g, hidden, intermediate)   -> (g, intermediate, hidden)

    The swap runs once at weight-load time so ``forward_tpu`` can hand
    ``layer.w13_weight`` / ``layer.w2_weight`` straight to
    ``fused_moe_call`` with no per-step transpose. A regression in this
    hook (wrong axes, missing ``.contiguous()``, accidentally a no-op)
    produces garbage numerics that only surface in model inference --
    this unit pin catches it earlier.
    """

    def test_tpu_branch_swaps_to_gmm_v2_layout(self):
        from sglang.srt.layers.quantization import unquant

        torch.manual_seed(0)
        # SGLang storage layout.
        # w13: (g, 2*intermediate, hidden); w2: (g, hidden, intermediate).
        sglang_w13 = torch.randn(4, 128, 32, dtype=torch.bfloat16)
        sglang_w2 = torch.randn(4, 32, 64, dtype=torch.bfloat16)

        layer = SimpleNamespace()
        layer.w13_weight = torch.nn.Parameter(sglang_w13.clone(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(sglang_w2.clone(), requires_grad=False)

        # The TPU branch is gated on the module-level ``_is_tpu`` flag;
        # patch it so the test runs on any host.
        with patch.object(unquant, "_is_tpu", True):
            UnquantizedFusedMoEMethod().process_weights_after_loading(layer)

        # gmm_v2 layout: (g, k, n) for both w1 and w2.
        # w13: (g, hidden, 2*intermediate);  w2: (g, intermediate, hidden).
        torch.testing.assert_close(
            layer.w13_weight.data,
            sglang_w13.transpose(1, 2).contiguous(),
        )
        torch.testing.assert_close(
            layer.w2_weight.data,
            sglang_w2.transpose(1, 2).contiguous(),
        )
        # ``forward_tpu`` hands these to the kernel without further
        # transposition; both must be contiguous so gmm_v2's BlockSpec /
        # reshape paths (gmm_v2.py:624-625) work.
        self.assertTrue(layer.w13_weight.data.is_contiguous())
        self.assertTrue(layer.w2_weight.data.is_contiguous())
        # Inference weights -- the hook must preserve no-grad.
        self.assertFalse(layer.w13_weight.requires_grad)
        self.assertFalse(layer.w2_weight.requires_grad)

    def test_no_swap_when_not_on_tpu(self):
        """The layout swap is platform-specific. With ``_is_tpu=False``
        the hook must not modify the weights -- a regression that
        always swapped would silently break the CUDA / CPU paths."""
        from sglang.srt.layers.quantization import unquant

        torch.manual_seed(1)
        sglang_w13 = torch.randn(4, 128, 32, dtype=torch.bfloat16)
        sglang_w2 = torch.randn(4, 32, 64, dtype=torch.bfloat16)

        layer = SimpleNamespace()
        layer.w13_weight = torch.nn.Parameter(sglang_w13.clone(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(sglang_w2.clone(), requires_grad=False)

        with patch.object(unquant, "_is_tpu", False):
            UnquantizedFusedMoEMethod().process_weights_after_loading(layer)

        # On a vanilla CPU host none of the non-TPU branches (aiter,
        # deep_gemm, flashinfer, npu) light up, so the hook returns
        # without modifying the weights.
        torch.testing.assert_close(layer.w13_weight.data, sglang_w13)
        torch.testing.assert_close(layer.w2_weight.data, sglang_w2)


class TestForwardTpuContract(CustomTestCase):
    """Direct ``forward_tpu(layer, d_out)`` -- isolates the method's
    contract from MultiPlatformOp dispatch. Numerics are covered by the
    end-to-end Pallas test; here we just pin the call shape.
    """

    def setUp(self):
        self.runner = _MoeMockRunner()
        self.runner.start()

    def tearDown(self):
        self.runner.stop()

    def test_routes_through_fused_moe_call(self):
        """Headline: ``forward_tpu`` must invoke the Pallas fused MoE
        kernel (``fused_moe_call``). A regression that dropped to
        ``forward_native``, raised early, or otherwise skipped the
        kernel would silently disable Pallas on TPU -- the contract
        tests below would all error out, but the failure mode would
        be confusing. This explicit routing check produces a clean
        diagnostic ahead of those secondary failures.
        """
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output()
        self.runner.tpu_forward(layer, d_out)

        self.assertTrue(
            self.runner.kernel_was_called,
            "forward_tpu did not route through fused_moe_call",
        )

    def test_passes_w13_and_w2_through_unchanged(self):
        """``process_weights_after_loading`` already swapped axes -- the
        forward must not transpose again, must not call ``.contiguous()``
        on already-contiguous tensors, and must hand the kernel the same
        tensor objects."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output()
        self.runner.tpu_forward(layer, d_out)

        kw = self.runner.kernel_kwargs
        self.assertIs(kw["w1"], layer.w13_weight)
        self.assertIs(kw["w2"], layer.w2_weight)

    def test_casts_topk_weights_to_hidden_dtype(self):
        """fused_moe_call's JaxCallable specialises on dtype; if topk_weights
        leaks in as fp32 while hidden_states is bf16, the JAX trace cache
        explodes per layer and we silently regress prefill throughput."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output(
            hidden_dtype=torch.bfloat16,
            weights_dtype=torch.float32,
        )
        self.runner.tpu_forward(layer, d_out)

        kw = self.runner.kernel_kwargs
        self.assertEqual(kw["topk_weights"].dtype, torch.bfloat16)
        self.assertEqual(kw["hidden_states"].dtype, torch.bfloat16)

    def test_casts_topk_ids_to_int32(self):
        """gmm_v2's argsort expects int32. Production routing produces
        int64 by default -- the cast happens here.
        """
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output(ids_dtype=torch.int64)
        self.runner.tpu_forward(layer, d_out)

        self.assertEqual(self.runner.kernel_kwargs["topk_ids"].dtype, torch.int32)

    def test_topk_is_taken_from_topk_ids_last_dim(self):
        """``topk`` is a static int compiled into the JAX trace -- it
        must come from the tensor shape, not from anywhere mutable like
        ``moe_runner_config``. Otherwise per-layer topk overrides would
        be silently ignored at trace time."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output(topk=3)
        self.runner.tpu_forward(layer, d_out)

        self.assertEqual(self.runner.kernel_kwargs["topk"], 3)
        self.assertIsInstance(self.runner.kernel_kwargs["topk"], int)

    def test_activation_is_forwarded_from_runner_config(self):
        """Both ``silu`` and ``gelu`` must reach the kernel verbatim --
        a silent rewrite (e.g. "gelu" -> "silu" via some compat shim)
        would break GeluAndMul-using models on TPU.
        """
        for activation in ("silu", "gelu"):
            with self.subTest(activation=activation), _MoeMockRunner(
                activation=activation
            ) as r:
                layer = r.make_layer()
                d_out = r.make_dispatch_output()
                r.tpu_forward(layer, d_out)
                self.assertEqual(r.kernel_kwargs["activation"], activation)

    def test_apply_router_weight_on_input_raises(self):
        """Explicit guard: this path is not yet implemented on TPU and
        silently falling through would produce wrong numerics with no
        warning. The error message must remain specific to the TPU
        backend so an upstream user knows where to look."""
        with _MoeMockRunner(apply_router_weight_on_input=True) as r:
            layer = r.make_layer()
            d_out = r.make_dispatch_output()
            with self.assertRaises(NotImplementedError) as cm:
                r.tpu_forward(layer, d_out)
            self.assertIn("TPU", str(cm.exception))

    def test_hidden_states_passed_through_unchanged(self):
        """No layout / dtype mutation: ``hidden_states`` must reach the
        kernel as the exact tensor object out of dispatch_output."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output()
        self.runner.tpu_forward(layer, d_out)

        self.assertIs(self.runner.kernel_kwargs["hidden_states"], d_out.hidden_states)

    def test_return_value_reports_standard_format(self):
        """The returned ``CombineInput`` must report ``format == STANDARD``.

        The downstream combiner uses ``CombineInputFormat`` to decide
        between standard / deepep / flashinfer combiners. A return value
        wrapped in (e.g.) ``FlashinferCombineInput`` would silently
        mis-route on TPU. Pinning the format closes that hole."""
        from sglang.srt.layers.moe.token_dispatcher.base import (
            CombineInputFormat,
        )

        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output()
        result = self.runner.tpu_forward(layer, d_out)

        self.assertEqual(result.format, CombineInputFormat.STANDARD)

    def test_return_value_satisfies_combine_input_protocol(self):
        """``CombineInput`` is a ``@runtime_checkable`` Protocol that the
        MoE pipeline uses with ``isinstance(x, CombineInput)`` to
        dispatch combine variants. A return value that fails this check
        would crash the combine step with a confusing TypeError far from
        the actual mistake."""
        from sglang.srt.layers.moe.token_dispatcher.base import CombineInput

        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output()
        result = self.runner.tpu_forward(layer, d_out)

        self.assertIsInstance(result, CombineInput)

    def test_hidden_states_scale_is_ignored(self):
        """``StandardDispatchOutput`` has a ``hidden_states_scale`` field
        used by quant paths. The unquant ``forward_tpu`` must ignore it
        -- threading a non-None scale into ``fused_moe_call`` would
        change the JAX trace cache key per layer and silently regress
        prefill throughput, and downstream the kernel would mis-quantise."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output(
            hidden_states_scale=torch.ones(8, dtype=torch.float32),
        )
        self.runner.tpu_forward(layer, d_out)

        kw = self.runner.kernel_kwargs
        # No quantization scales may leak into the kernel call, and for a
        # bias-less layer the bias kwargs (which unquant forward_tpu now
        # forwards explicitly) must resolve to None.
        for forbidden in ("w1_scale", "w2_scale"):
            self.assertNotIn(
                forbidden,
                kw,
                f"unquant forward_tpu must not pass {forbidden}",
            )
        for bias_kw in ("w1_bias", "w2_bias"):
            self.assertIn(
                bias_kw,
                kw,
                f"unquant forward_tpu must forward {bias_kw} explicitly",
            )
            self.assertIsNone(
                kw[bias_kw],
                f"{bias_kw} must be None for a bias-less layer",
            )


class TestForwardTpuViaDispatch(CustomTestCase):
    """Integration: ``BaseFusedOp.forward`` -> dispatch -> ``forward_tpu``
    -> kernel. Pins ``_platform_key()`` to ``"tpu"`` so dispatch resolution
    (lazy, on first call / ``dispatch_forward()``) lands on ``forward_tpu``.
    Confirms the dispatch wiring is intact on top of the method's
    contract.
    """

    def setUp(self):
        self.runner = _MoeMockRunner(force_tpu_dispatch=True)
        self.runner.start()

    def tearDown(self):
        self.runner.stop()

    def test_dispatch_resolves_forward_tpu(self):
        """``dispatch_forward()`` must resolve to ``forward_tpu``. Pin its
        identity so an accidental rename in ``BaseFusedOp`` (or a
        regression in the platform-key branch ordering) catches here --
        the most common failure mode would be silently dropping back to
        ``forward_native``.
        """
        self.assertEqual(
            self.runner.method.dispatch_forward().__func__.__name__,
            "forward_tpu",
        )

    def test_dispatched_forward_reaches_kernel(self):
        """End-to-end: a ``method.forward(layer, d_out)`` call must
        thread through the dispatch cache, into ``forward_tpu``, and
        ultimately into the kernel. If dispatch routes to
        ``forward_native`` (or any other branch), the kernel mock
        would never be invoked -- this assertion catches that."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output()
        self.runner.dispatched_forward(layer, d_out)

        self.assertTrue(self.runner.kernel_was_called)

    def test_dispatched_path_preserves_kernel_kwargs_contract(self):
        """The contract that direct ``forward_tpu`` tests pin (dtype
        casts, weight pass-through, topk extraction) must also hold
        when the call arrives via ``method.forward(...)``. This guards
        against a future refactor that adds a wrapper around dispatch
        which silently mutates the args before they reach
        ``forward_tpu``."""
        layer = self.runner.make_layer()
        d_out = self.runner.make_dispatch_output(
            hidden_dtype=torch.bfloat16,
            weights_dtype=torch.float32,
            ids_dtype=torch.int64,
            topk=3,
        )
        self.runner.dispatched_forward(layer, d_out)

        kw = self.runner.kernel_kwargs
        # Same invariants as the direct contract tests above.
        self.assertIs(kw["w1"], layer.w13_weight)
        self.assertIs(kw["w2"], layer.w2_weight)
        self.assertEqual(kw["topk_weights"].dtype, torch.bfloat16)
        self.assertEqual(kw["topk_ids"].dtype, torch.int32)
        self.assertEqual(kw["topk"], 3)


if __name__ == "__main__":
    unittest.main()
