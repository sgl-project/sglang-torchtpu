"""Kimi-K3 attention-residual kernel hooks (sglang_tpu/hooks/attn_residual.py),
applied through HookRegistry, and the aggregation they register with
attn_residual, with a stand-in for the JAX kernel module."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import contextlib
import inspect
import pkgutil
import sys
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import AttnResidual, aggregate_stream_torch, get_cw
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models import kimi_k3
from sglang.srt.platforms import current_platform
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_context
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import attn_residual as hooks
from sglang_tpu.platform import TpuOOTPlatform

_KERNEL_MODULE = "sglang_tpu.kimi_attn_residual"
_FLAG = tpu_envs.SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL
_PREFILL_FLAG = tpu_envs.SGLANG_TPU_ENABLE_K3_ATTN_RES_PREFILL_KERNEL
_ATTN_RES_HOOKS = [h for h in HOOKS if h[1].__module__ == hooks.__name__]
_KEY = current_platform.get_dispatch_key_name()


@contextlib.contextmanager
def _hooks_applied(kernel=None, prefill=False):
    """Apply the attention-residual HOOKS entries through HookRegistry, as the
    plugin does on TPU, with `kernel` behind them, a scratch aggregation
    registry and the TPU platform's post-load call; restore every target on
    exit. A given `kernel` is registered as register_attn_residual_ops
    registers it."""
    with contextlib.ExitStack() as stack:
        for target, _, _ in _ATTN_RES_HOOKS:
            obj_path, attr = target.rsplit(".", 1)
            obj = pkgutil.resolve_name(obj_path)
            # Re-sets the original now, so exit undoes whatever apply_hooks installs.
            stack.enter_context(patch.object(obj, attr, getattr(obj, attr)))
        stack.enter_context(patch.object(HookRegistry, "_hooks", defaultdict(list)))
        stack.enter_context(patch.object(HookRegistry, "_patched", set()))
        stack.enter_context(patch.object(hooks, "_kernel", kernel))
        stack.enter_context(patch.object(hooks, "_prefill", prefill))
        stack.enter_context(patch.object(attn_residual, "_AGGREGATE_IMPLS", {}))
        stack.enter_context(patch.object(kimi_k3, "current_platform", TpuOOTPlatform()))
        for target, hook, hook_type in _ATTN_RES_HOOKS:
            HookRegistry.register(target, hook, hook_type)
        HookRegistry.apply_hooks()
        # apply_hooks logs and moves on when a target fails to resolve.
        assert HookRegistry._patched == {t for t, _, _ in _ATTN_RES_HOOKS}
        if kernel is not None:
            attn_residual.register_aggregate_impl(_KEY, hooks.aggregate)
        yield


def _stand_in_kernel(supported: bool = True, prefill: bool = False) -> types.ModuleType:
    """The kernel module's interface without JAX. The decode stand-in returns
    (normed, prefix) like the kernel, with a constant normed to tell it apart
    (8.0 from the prefill stand-in)."""
    kernel = types.ModuleType(_KERNEL_MODULE)
    kernel.register_kimi_attn_res_decode_ops = MagicMock()
    kernel.supports_attn_res_decode = MagicMock(return_value=supported)
    kernel.supports_attn_res_prefill = MagicMock(return_value=prefill)
    kernel.prefill_tokens_supported = lambda t: t >= 256 and t % 32 == 0
    kernel.get_norm_weight_f32 = MagicMock(side_effect=lambda n: n.weight.float())
    for name, value in (("decode", 7.0), ("prefill", 8.0)):
        setattr(
            kernel,
            f"attention_residual_{name}_tpu",
            MagicMock(
                side_effect=lambda a, b, *_, v=value: (
                    torch.full_like(a, v),
                    a if b is None else a + b,
                )
            ),
        )
    return kernel


class TestAttnResidualHookWiring(CustomTestCase):
    def test_model_init_and_bank_hooks(self):
        """Aggregation goes through attn_residual's registry and preparation
        through the platform's post-load call; the only core hook besides
        model init sets the bank layout."""
        self.assertEqual(
            [h[:1] + h[2:] for h in _ATTN_RES_HOOKS],
            [
                (
                    "sglang.srt.models.kimi_k3.KimiK3LinearModel.__init__",
                    HookType.AFTER,
                ),
                (
                    "sglang.srt.layers.attn_residual.AttnResidual.__init__",
                    HookType.AFTER,
                ),
            ],
        )

    def test_aggregate_mirrors_the_registry_protocol(self):
        """A core signature change fails here instead of misrouting arguments."""
        params = list(inspect.signature(hooks.aggregate).parameters)
        protocol = inspect.signature(attn_residual.AggregateImpl.__call__)
        self.assertEqual(params, list(protocol.parameters)[1:])

    def test_model_without_attn_residuals_keeps_kernel_off(self):
        # None in sys.modules makes importing the kernel module raise.
        model = SimpleNamespace(config=SimpleNamespace(attn_res_block_size=None))
        with (
            patch.dict(sys.modules, {_KERNEL_MODULE: None}),
            _FLAG.override(True),
            patch.object(hooks, "_kernel", None),
            patch.object(attn_residual, "_AGGREGATE_IMPLS", {}),
        ):
            hooks.register_attn_residual_ops(None, model)
            self.assertIsNone(hooks._kernel)
            self.assertEqual(attn_residual._AGGREGATE_IMPLS, {})


@unittest.skipIf(torch.cuda.is_available(), "the original takes the TPU torch path")
class TestAttnResidualHookRouting(CustomTestCase):
    T, H, NB, NVB = 4, 1024, 3, 2

    def setUp(self):
        tpu = patch.object(attn_residual, "_is_tpu", True)
        tpu.start()
        self.addCleanup(tpu.stop)
        torch.manual_seed(0)
        self.score_proj = ReplicatedLinear(self.H, 1, bias=False)
        self.score_norm = RMSNorm(self.H)
        self.out_norm = RMSNorm(self.H, eps=1e-5)
        with torch.no_grad():
            self.score_proj.weight.normal_()
        self.modules = (self.score_proj, self.score_norm, self.out_norm)
        self.a, self.b = torch.randn(self.T, self.H), torch.randn(self.T, self.H)
        self.ar = AttnResidual(self.a, self.NB)
        self.ar.block_residual.copy_(torch.randn(self.T, self.NB, self.H))
        self.ar.num_valid_blocks = self.NVB

    def _reference(self, prefix):
        bank = self.ar.block_residual
        mixed = aggregate_stream_torch(
            prefix, bank, self.NVB, self.score_proj, self.score_norm
        )
        return self.out_norm(mixed)

    def test_without_kernel_both_entry_points_run_the_original(self):
        with _hooks_applied(kernel=None):
            normed, prefix = self.ar.forward(self.a, None, *self.modules)
            self.assertIs(prefix, self.a)
            torch.testing.assert_close(normed, self._reference(self.a))
            # Pending add: hidden_states is the addend, prefix_sum the prefix.
            normed, prefix = self.ar.forward(self.b, self.a, *self.modules)
            torch.testing.assert_close(prefix, self.a + self.b)
            torch.testing.assert_close(normed, self._reference(self.a + self.b))

    def test_decode_calls_take_the_kernel(self):
        kernel = _stand_in_kernel(supported=True)
        with _hooks_applied(kernel=kernel):
            normed, prefix = self.ar.forward(self.a, None, *self.modules)
            self.assertTrue(torch.all(normed == 7))
            self.assertIs(prefix, self.a)
            normed, prefix = self.ar.forward(self.b, self.a, *self.modules)
            self.assertTrue(torch.all(normed == 7))
            torch.testing.assert_close(prefix, self.a + self.b)

        bank = self.ar.block_residual
        gate = kernel.supports_attn_res_decode
        decode = kernel.attention_residual_decode_tpu
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(decode.call_count, 2)
        for i, addend in enumerate((None, self.b)):
            a, b, bk, score_norm, out_norm = gate.call_args_list[i].args
            self.assertTrue(a is self.a and b is addend and bk is bank)
            self.assertIs(score_norm, self.score_norm)
            self.assertIs(out_norm, self.out_norm)
            a, b, bk, nvb, cw, eps, norm_weight, out_eps = decode.call_args_list[i].args
            self.assertTrue(a is self.a and b is addend and bk is bank)
            self.assertEqual(nvb, self.NVB)
            torch.testing.assert_close(cw, get_cw(self.score_proj, self.score_norm))
            self.assertEqual(eps, self.score_norm.variance_epsilon)
            torch.testing.assert_close(norm_weight, self.out_norm.weight.float())
            self.assertEqual(out_eps, self.out_norm.variance_epsilon)

    def test_other_calls_run_the_original(self):
        """A closed shape gate or a fused bank write leaves the call to the
        original."""
        kernel = _stand_in_kernel(supported=False)
        with _hooks_applied(kernel=kernel):
            normed, _ = self.ar.forward(self.a, None, *self.modules)
            torch.testing.assert_close(normed, self._reference(self.a))
            normed, prefix = self.ar.forward(self.b, self.a, *self.modules)
            torch.testing.assert_close(normed, self._reference(self.a + self.b))
            # The kernel has no bank write: the original owns that contract
            # (fast path only), even when the shape gate holds.
            kernel.supports_attn_res_decode.return_value = True
            with self.assertRaisesRegex(AssertionError, "fast-path only"):
                attn_residual._aggregate(
                    self.a,
                    self.ar.block_residual,
                    self.NVB,
                    *self.modules,
                    write_bank_row=True,
                )
        kernel.attention_residual_decode_tpu.assert_not_called()


@unittest.skipIf(torch.cuda.is_available(), "the original takes the TPU torch path")
class TestAttnResidualPrefillBank(CustomTestCase):
    """SGLANG_TPU_ENABLE_K3_ATTN_RES_PREFILL_KERNEL: prefill-sized banks are
    stored block-major behind core's [T, NB, H] view, and only those reach the
    prefill kernel."""

    H, NB, NVB = 256, 3, 2

    def setUp(self):
        tpu = patch.object(attn_residual, "_is_tpu", True)
        tpu.start()
        self.addCleanup(tpu.stop)
        torch.manual_seed(0)
        self.score_proj = ReplicatedLinear(self.H, 1, bias=False)
        self.score_norm = RMSNorm(self.H)
        self.out_norm = RMSNorm(self.H, eps=1e-5)
        with torch.no_grad():
            self.score_proj.weight.normal_()
        self.modules = (self.score_proj, self.score_norm, self.out_norm)

    @staticmethod
    def _block_major(bank):
        return bank.permute(1, 0, 2).is_contiguous()

    def _filled(self, tokens, **hooks_kwargs):
        """An AttnResidual built under the hooks, with NVB blocks written."""
        rows = [torch.randn(tokens, self.H) for _ in range(self.NVB)]
        with _hooks_applied(**hooks_kwargs):
            ar = AttnResidual(torch.randn(tokens, self.H), self.NB)
        for row in rows:
            ar.write(row)
        return ar, rows

    def test_prefill_sized_bank_is_block_major(self):
        kernel = _stand_in_kernel()
        ar, rows = self._filled(256, kernel=kernel, prefill=True)
        bank = ar.block_residual
        self.assertEqual(tuple(bank.shape), (256, self.NB, self.H))
        self.assertTrue(self._block_major(bank))
        for j, row in enumerate(rows):  # core's writes land in block j's slab
            self.assertTrue(torch.equal(bank[:, j, :], row))
            self.assertTrue(torch.equal(bank.permute(1, 0, 2)[j], row))

    def test_other_banks_stay_token_major(self):
        kernel = _stand_in_kernel()
        # Decode-sized (16, 64), not whole tiles (272), flag off (256).
        for tokens, prefill in ((16, True), (64, True), (272, True), (256, False)):
            with self.subTest(tokens=tokens, prefill=prefill):
                ar, _ = self._filled(tokens, kernel=kernel, prefill=prefill)
                self.assertTrue(ar.block_residual.is_contiguous())

    def test_inherited_rows_are_kept(self):
        """A bank handed over by the previous PP rank keeps its rows."""
        inherited = torch.randn(256, self.NVB, self.H)
        with _hooks_applied(kernel=_stand_in_kernel(), prefill=True):
            ar = AttnResidual(torch.randn(256, self.H), self.NB, inherited)
        self.assertTrue(self._block_major(ar.block_residual))
        self.assertEqual(ar.num_valid_blocks, self.NVB)
        self.assertTrue(torch.equal(ar.block_residual[:, : self.NVB], inherited))

    def test_block_major_bank_keeps_the_original_results(self):
        """With the prefill gate closed, the original reads the block-major view
        and returns what it returns on a token-major bank."""
        kernel = _stand_in_kernel(supported=False, prefill=False)
        ar, rows = self._filled(256, kernel=kernel, prefill=True)
        ref = AttnResidual(torch.randn(256, self.H), self.NB)
        for row in rows:
            ref.write(row)
        a, b = torch.randn(256, self.H), torch.randn(256, self.H)
        with _hooks_applied(kernel=kernel, prefill=True):
            got = ar.forward(b, a, *self.modules)
        want = ref.forward(b, a, *self.modules)
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0)
        kernel.attention_residual_prefill_tpu.assert_not_called()

    def test_prefill_calls_take_the_prefill_kernel(self):
        kernel = _stand_in_kernel(supported=False, prefill=True)
        ar, _ = self._filled(256, kernel=kernel, prefill=True)
        a, b = torch.randn(256, self.H), torch.randn(256, self.H)
        with _hooks_applied(kernel=kernel, prefill=True):
            normed, prefix = ar.forward(b, a, *self.modules)
        self.assertTrue(torch.all(normed == 8))
        torch.testing.assert_close(prefix, a + b)
        args = kernel.attention_residual_prefill_tpu.call_args.args
        self.assertIs(args[2], ar.block_residual)
        self.assertEqual(args[3], self.NVB)
        kernel.attention_residual_decode_tpu.assert_not_called()

    def test_prefill_flag_off_keeps_prefill_on_the_original(self):
        kernel = _stand_in_kernel(supported=False, prefill=True)
        ar, _ = self._filled(256, kernel=kernel, prefill=False)
        with _hooks_applied(kernel=kernel, prefill=False):
            ar.forward(torch.randn(256, self.H), None, *self.modules)
        kernel.supports_attn_res_prefill.assert_not_called()
        kernel.attention_residual_prefill_tpu.assert_not_called()


def _tiny_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=2048,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        moe_intermediate_size=256,
        num_experts_per_token=2,
        num_experts=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        attn_res_block_size=2,
        linear_attn_config={
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
            "short_conv_kernel_size": 4,
            "head_dim": 64,
            "num_heads": 8,
            "num_key_value_heads": 8,
            "q_lora_rank": None,
            "use_full_rank_gate": True,
        },
    )


def _aggregation_points(model):
    """(score_proj, out_norm) of every AttnResidual.forward call site."""
    points = []
    for layer in model.model.layers:
        if isinstance(layer, PPMissingLayer) or not layer.use_attn_residuals:
            continue
        points.append((layer.self_attention_res_proj, layer.input_layernorm))
        points.append((layer.mlp_res_proj, layer.post_attention_layernorm))
    points.append((model.model.output_attn_res_proj, model.model.norm))
    return points


@unittest.skipIf(torch.cuda.is_available(), "builds the model on CPU")
class TestKimiK3AttnResHookModelSetup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        # The model's TPU construction path; off TPU it builds CUDA streams.
        tpu = patch.object(kimi_k3, "_is_tpu", True)
        tpu.start()
        cls.addClassCleanup(tpu.stop)

    def _build(self, flag: bool, kernel):
        """Construct and post-load the model with the hooks applied. Returns the
        model, the hooks' kernel right after construction, and the aggregation
        registry then."""
        with (
            patch.dict(sys.modules, {_KERNEL_MODULE: kernel}),
            _FLAG.override(flag),
            _hooks_applied(),
        ):
            model = kimi_k3.KimiK3LinearForCausalLM(
                config=_tiny_config(), quant_config=None
            )
            kernel_after_init = hooks._kernel
            registered = dict(attn_residual._AGGREGATE_IMPLS)
            model.post_load_weights()
        return model, kernel_after_init, registered

    def test_flag_on_prepares_the_kernel_before_the_trace(self):
        """Model init registers the decode ops; post-load builds each aggregation
        point's fp32 out-norm weight next to its fp32 score weight, and publishes
        the weight updater's guard."""
        kernel = _stand_in_kernel()
        model, kernel_after_init, registered = self._build(flag=True, kernel=kernel)
        self.assertIs(kernel_after_init, kernel)
        kernel.register_kimi_attn_res_decode_ops.assert_called_once_with(prefill=False)
        self.assertEqual(registered, {_KEY: hooks.aggregate})

        points = _aggregation_points(model)
        self.assertEqual(len(points), 2 * 4 + 1)
        for proj, _ in points:
            self.assertIn(torch.float32, proj._attn_res_cw_cache)
        # Modules compare by identity: each point's own out norm, in order.
        built = [c.args for c in kernel.get_norm_weight_f32.call_args_list]
        self.assertEqual(built, [(out_norm,) for _, out_norm in points])

        # Checked on the module prepare_attn_residual_kernel marks: the TPU KDA
        # layers publish a guard of their own, so the updater rejects updates
        # either way.
        reason = model.model._derived_weight_cache_error
        self.assertIn("SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL", reason)

    def test_prefill_flag_registers_the_prefill_ops(self):
        kernel = _stand_in_kernel()
        with _PREFILL_FLAG.override(True):
            self._build(flag=True, kernel=kernel)
        kernel.register_kimi_attn_res_decode_ops.assert_called_once_with(prefill=True)

    def test_flag_off_never_imports_the_kernel(self):
        # None in sys.modules makes importing the kernel module raise.
        model, kernel_after_init, registered = self._build(flag=False, kernel=None)
        self.assertIsNone(kernel_after_init)
        self.assertEqual(registered, {})
        self.assertIsNone(getattr(model.model, "_derived_weight_cache_error", None))


if __name__ == "__main__":
    unittest.main()
