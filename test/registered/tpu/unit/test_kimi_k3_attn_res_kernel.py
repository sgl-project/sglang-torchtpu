"""Kimi-K3 attention-residual Pallas decode kernel vs the fp32 torch reference."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

import contextlib
import importlib.util
import itertools
import unittest
from collections import defaultdict
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import (
    AttnResidual,
    aggregate_stream_torch,
    get_cw,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import attn_residual as hooks

H = 7168  # Kimi-K3 hidden size
NB = 9  # bank rows allocated; nvb <= NB
EPS = 1e-6

_HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None


@contextlib.contextmanager
def _aggregation_hooks(kernel):
    """The plugin's AROUND hooks on attn_residual._aggregate and
    _aggregate_fused_add, applied through HookRegistry as on a server with
    SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL=1, and undone on exit."""
    prefix = attn_residual.__name__ + "."
    arounds = [h for h in HOOKS if h[0].startswith(prefix)]
    with contextlib.ExitStack() as stack:
        for target, _, _ in arounds:
            name = target[len(prefix) :]
            # Re-sets the original now, so exit undoes whatever apply_hooks installs.
            stack.enter_context(
                patch.object(attn_residual, name, getattr(attn_residual, name))
            )
        stack.enter_context(patch.object(HookRegistry, "_hooks", defaultdict(list)))
        stack.enter_context(patch.object(HookRegistry, "_patched", set()))
        stack.enter_context(patch.object(hooks, "_kernel", kernel))
        for target, hook, hook_type in arounds:
            HookRegistry.register(target, hook, hook_type)
        HookRegistry.apply_hooks()
        assert HookRegistry._patched == {t for t, _, _ in arounds}
        yield


def _modules(seed: int, eps: float = EPS):
    g = torch.Generator().manual_seed(seed)
    score_proj = ReplicatedLinear(H, 1, bias=False, params_dtype=torch.bfloat16)
    score_norm = RMSNorm(H, eps=eps)
    out_norm = RMSNorm(H, eps=eps)
    with torch.no_grad():
        # |cw| ~ 1.7 so the softmax is neither uniform nor one-hot.
        score_proj.weight.copy_(torch.randn(1, H, generator=g) * 0.02)
        score_norm.weight.copy_(1 + 0.1 * torch.randn(H, generator=g))
        out_norm.weight.copy_(1 + 0.1 * torch.randn(H, generator=g))
    for m in (score_proj, score_norm, out_norm):
        m.to(torch.bfloat16)
    return score_proj, score_norm, out_norm


def _reference(prefix, bank, nvb, score_proj, score_norm, out_norm):
    """fp32 aggregate_stream_torch + forward_native out norm on CPU. The mixture
    is rounded to bf16 before the norm, as on every production path."""
    eps = score_norm.variance_epsilon
    proj32 = ReplicatedLinear(H, 1, bias=False, params_dtype=torch.float32)
    norm32, out32 = RMSNorm(H, eps=eps), RMSNorm(H, eps=out_norm.variance_epsilon)
    with torch.no_grad():
        proj32.weight.copy_(score_proj.weight.float())
        norm32.weight.copy_(score_norm.weight.float())
        out32.weight.copy_(out_norm.weight.float())
    mixed = aggregate_stream_torch(prefix.float(), bank.float(), nvb, proj32, norm32)
    mixed = mixed.to(torch.bfloat16).float()
    return out32.forward_native(mixed).to(torch.bfloat16)


def _to_tpu(*modules):
    for m in modules:
        m.to(torch.device("tpu"))
    return modules


@unittest.skipUnless(_HAS_TORCH_TPU, "Requires torch_tpu")
class TestKimiK3AttnResKernel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.layers import kimi_attn_residual

        kimi_attn_residual.register_kimi_attn_res_decode_ops()
        cls.kernel = kimi_attn_residual

    def test_shape_gate(self):
        """Decode-sized bf16 batches of 128-lane rows whose bank fits one VMEM
        block, with default RMSNorms; a pending addend must match the prefix."""
        supports = self.kernel.supports_attn_res_decode
        cap = self.kernel._DECODE_MAX_TOKENS
        norm = RMSNorm(H)
        hf_norm = RMSNorm(H, cast_x_before_out_mul=True)

        def rows(tokens, dtype=torch.bfloat16, hidden=H):
            return torch.empty(tokens, hidden, dtype=dtype)

        def bank(tokens, blocks=NB, hidden=H):
            return torch.empty(tokens, blocks, hidden, dtype=torch.bfloat16)

        def gate(a, b=None, blocks=NB, score=norm, out=norm):
            return supports(a, b, bank(a.shape[0], blocks, a.shape[1]), score, out)

        self.assertTrue(gate(rows(1)))
        self.assertTrue(gate(rows(cap)))  # the bank bound: 16 rows at NB = 9
        self.assertTrue(gate(rows(cap), rows(cap)))
        self.assertFalse(gate(rows(0)))
        self.assertFalse(gate(rows(cap + 1)))
        self.assertFalse(gate(rows(cap + 1), blocks=4))  # the token cap alone
        self.assertFalse(gate(rows(cap), blocks=NB + 1))  # bank past VMEM
        self.assertFalse(gate(rows(1, hidden=H + 64)))
        self.assertFalse(gate(rows(1, torch.float32)))
        self.assertFalse(gate(rows(1), score=hf_norm))
        self.assertFalse(gate(rows(1), out=hf_norm))
        self.assertFalse(gate(rows(1), rows(1, torch.float32)))
        self.assertFalse(gate(rows(1), rows(2)))

    def test_norm_weight_f32_is_cached(self):
        """The fp32 out-norm weight is built once, at post-load."""
        out_norm = RMSNorm(256).to(torch.bfloat16)
        with torch.no_grad():
            out_norm.weight.normal_()
        w32 = self.kernel.get_norm_weight_f32(out_norm)
        self.assertEqual(w32.dtype, torch.float32)
        self.assertTrue(w32.is_contiguous())
        torch.testing.assert_close(w32, out_norm.weight.float())
        self.assertIs(self.kernel.get_norm_weight_f32(out_norm), w32)
        # Building it inside a trace would bake a guarded constant: refuse.
        with patch("torch.compiler.is_compiling", return_value=True):
            self.assertIs(self.kernel.get_norm_weight_f32(out_norm), w32)
            with self.assertRaisesRegex(RuntimeError, "before tracing"):
                self.kernel.get_norm_weight_f32(RMSNorm(256))

    def _bank_and_rows(self, tokens, seed):
        g = torch.Generator().manual_seed(seed)
        # Per-row scales stress the per-row RMS in the score.
        scale = torch.rand(tokens, NB, 1, generator=g) * 4 + 0.25
        bank = (torch.randn(tokens, NB, H, generator=g) * scale).to(torch.bfloat16)
        prefix_a = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
        prefix_b = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
        return bank, prefix_a, prefix_b

    def _check(self, tokens: int, nvb: int, has_addend: bool):
        bank, prefix_a, prefix_b = self._bank_and_rows(tokens, tokens * 100 + nvb)
        score_proj, score_norm, out_norm = _modules(nvb)
        prefix = prefix_a + prefix_b if has_addend else prefix_a
        expected = _reference(prefix, bank, nvb, score_proj, score_norm, out_norm)

        dev = torch.device("tpu")
        _to_tpu(score_proj, score_norm, out_norm)
        ar = AttnResidual(prefix_a.to(dev), NB)
        ar.block_residual.copy_(bank.to(dev))
        ar.num_valid_blocks = nvb
        a_dev, b_dev = prefix_a.to(dev), prefix_b.to(dev)
        modules = (score_proj, score_norm, out_norm)
        self.assertTrue(
            self.kernel.supports_attn_res_decode(
                a_dev,
                b_dev if has_addend else None,
                ar.block_residual,
                score_norm,
                out_norm,
            )
        )
        with _aggregation_hooks(self.kernel):
            if has_addend:
                # Pending add: hidden_states is the addend, prefix_sum the prefix.
                normed, prefix_out = ar.forward(b_dev, a_dev, *modules)
            else:
                normed, prefix_out = ar.forward(a_dev, None, *modules)
        # Hooks off: the torch reference path this kernel replaces.
        torch_path, _ = ar.forward(prefix.to(dev), None, *modules)
        normed, torch_path = normed.cpu(), torch_path.cpu()
        err = (normed.float() - expected.float()).abs().max().item()
        torch_err = (torch_path.float() - expected.float()).abs().max().item()
        print(
            f"T={tokens} nvb={nvb} addend={has_addend} max|kernel-ref|={err:.3g} "
            f"max|torch-ref|={torch_err:.3g} "
            f"kernel==torch: {torch.equal(normed, torch_path)}"
        )

        self.assertEqual(normed.dtype, torch.bfloat16)
        if has_addend:
            # The kernel materializes the pending add like a torch bf16 add.
            torch.testing.assert_close(prefix_out.cpu(), prefix, rtol=0, atol=0)
        else:
            self.assertIs(prefix_out, a_dev)
        # Both sides score in f32 and round the output to bf16; the tolerance
        # covers a few bf16 ulps (measured max abs error 0.0156).
        torch.testing.assert_close(
            normed.float(), expected.float(), rtol=1.6e-2, atol=1e-2
        )

    def test_matches_fp32_reference(self):
        for tokens, nvb, has_addend in itertools.product(
            sorted({1, 4, self.kernel._DECODE_MAX_TOKENS}), (1, 4, NB), (False, True)
        ):
            with self.subTest(tokens=tokens, nvb=nvb, has_addend=has_addend):
                self._check(tokens, nvb, has_addend)

    def test_compiled_stack_matches_eager(self):
        """A K3-like stack (bank writes, pending adds, nvb 0..4, two epsilons)
        compiles through the plugin's hooks to one graph that only looks up the
        ops registered at init."""
        kimi_attn_residual = self.kernel
        tokens = kimi_attn_residual._DECODE_MAX_TOKENS
        dev = torch.device("tpu")
        # (score_proj, score_norm, out_norm) per aggregation point: two per layer
        # with a bank write every other layer, then the final aggregation.
        points = [
            _to_tpu(*_modules(100 + i, eps=1e-6 if i % 3 else 1e-5))
            for i in range(2 * (NB - 1) + 1)
        ]
        for proj, norm, out in points:  # post_load_weights state
            get_cw(proj, norm)
            kimi_attn_residual.get_norm_weight_f32(out)
        _, x, _ = self._bank_and_rows(tokens, 7)

        def stack(hidden):
            ar = AttnResidual(hidden, NB)
            prefix_sum = None
            outs = []
            for i in range(0, len(points) - 1, 2):
                write = i % 4 == 0
                normed, prefix = ar.forward(hidden, prefix_sum, *points[i], write=write)
                prefix_sum = None if write else prefix
                normed2, prefix2 = ar.forward(normed * 0.5, prefix_sum, *points[i + 1])
                hidden, prefix_sum = normed2 * 0.25 + prefix2, None
                outs += [normed, normed2]
            out, _ = ar.forward(hidden, prefix_sum, *points[-1])
            return torch.stack(outs + [out])

        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return torch._dynamo.lookup_backend("tpu")(gm, example_inputs)

        with (
            _aggregation_hooks(kimi_attn_residual),
            patch.object(
                kimi_attn_residual,
                "cpu_export_jax_op",
                side_effect=AssertionError("op built after model init"),
            ),
        ):
            stack = torch.no_grad()(stack)
            eager = stack(x.to(dev)).cpu()
            torch._dynamo.reset()
            compiled = torch.compile(
                stack, backend=backend, fullgraph=True, dynamic=False
            )
            out = compiled(x.to(dev)).cpu()
            out2 = compiled(x.to(dev) * 2).cpu()
        self.assertEqual(len(graphs), 1)
        self.assertTrue(torch.isfinite(out2.float()).all())
        torch.testing.assert_close(out.float(), eager.float(), rtol=1.6e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
