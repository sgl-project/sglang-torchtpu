"""Kimi-K3 attention-residual Pallas prefill kernel vs the fp32 torch reference.

The bank is stored block-major by the plugin's AttnResidual hook, as on a
server with SGLANG_TPU_ENABLE_K3_ATTN_RES_PREFILL_KERNEL=1.
"""

from sglang.test.ci.ci_register import register_tpu_ci

# Behind the default-off SGLANG_TPU_ENABLE_K3_ATTN_RES_PREFILL_KERNEL: nightly,
# outside the stage-a budget.
register_tpu_ci(
    est_time=150,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "kimi-k3"],
)

import contextlib
import importlib.util
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import AttnResidual, aggregate_stream_torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.platforms import current_platform
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import attn_residual as hooks

H = 7168  # Kimi-K3 hidden size
NB = 8  # Kimi-K3.1: 93 layers in blocks of 12
EPS = 1e-6

_HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None


def _modules(seed: int):
    g = torch.Generator().manual_seed(seed)
    score_proj = ReplicatedLinear(H, 1, bias=False, params_dtype=torch.bfloat16)
    score_norm, out_norm = RMSNorm(H, eps=EPS), RMSNorm(H, eps=EPS)
    with torch.no_grad():
        # |cw| ~ 1.7 so the softmax is neither uniform nor one-hot.
        score_proj.weight.copy_(torch.randn(1, H, generator=g) * 0.02)
        score_norm.weight.copy_(1 + 0.1 * torch.randn(H, generator=g))
        out_norm.weight.copy_(1 + 0.1 * torch.randn(H, generator=g))
    return tuple(m.to(torch.bfloat16) for m in (score_proj, score_norm, out_norm))


def _reference(prefix, bank, nvb, score_proj, score_norm, out_norm):
    """fp32 aggregate_stream_torch + forward_native out norm on CPU, the mixture
    rounded to bf16 before the norm."""
    proj32 = ReplicatedLinear(H, 1, bias=False, params_dtype=torch.float32)
    norm32, out32 = RMSNorm(H, eps=EPS), RMSNorm(H, eps=EPS)
    with torch.no_grad():
        proj32.weight.copy_(score_proj.weight.float())
        norm32.weight.copy_(score_norm.weight.float())
        out32.weight.copy_(out_norm.weight.float())
    mixed = aggregate_stream_torch(prefix.float(), bank.float(), nvb, proj32, norm32)
    return out32.forward_native(mixed.to(torch.bfloat16).float()).to(torch.bfloat16)


@unittest.skipUnless(_HAS_TORCH_TPU, "Requires torch_tpu")
class TestKimiK3AttnResPrefillKernel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu import kimi_attn_residual

        kimi_attn_residual.register_kimi_attn_res_decode_ops(prefill=True)
        cls.kernel = kimi_attn_residual

    @contextlib.contextmanager
    def _plugin(self):
        """The plugin's aggregation registered and the prefill flag's globals
        set, as register_attn_residual_ops sets them."""
        key = current_platform.get_dispatch_key_name()
        with (
            patch.object(attn_residual, "_AGGREGATE_IMPLS", {key: hooks.aggregate}),
            patch.object(hooks, "_kernel", self.kernel),
            patch.object(hooks, "_prefill", True),
        ):
            yield

    def _check(self, tokens: int, nvb: int, has_addend: bool):
        g = torch.Generator().manual_seed(tokens * 100 + nvb)
        # Per-row scales stress the per-row RMS in the score.
        scale = torch.rand(tokens, NB, 1, generator=g) * 4 + 0.25
        bank = (torch.randn(tokens, NB, H, generator=g) * scale).to(torch.bfloat16)
        prefix_a = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
        prefix_b = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
        modules = _modules(nvb)
        prefix = prefix_a + prefix_b if has_addend else prefix_a
        expected = _reference(prefix, bank, nvb, *modules)

        dev = torch.device("tpu")
        for m in modules:
            m.to(dev)
        a_dev, b_dev = prefix_a.to(dev), prefix_b.to(dev)
        with self._plugin():
            ar = AttnResidual(a_dev, NB)
            hooks.block_major_bank(None, ar, a_dev, NB)
            self.assertTrue(ar.block_residual.permute(1, 0, 2).is_contiguous())
            for j in range(nvb):
                ar.write(bank[:, j, :].to(dev))
            spy = patch.object(
                self.kernel,
                "attention_residual_prefill_tpu",
                wraps=self.kernel.attention_residual_prefill_tpu,
            )
            with spy as prefill:
                if has_addend:
                    # Pending add: hidden_states is the addend, prefix_sum the prefix.
                    normed, prefix_out = ar.forward(b_dev, a_dev, *modules)
                else:
                    normed, prefix_out = ar.forward(a_dev, None, *modules)
            self.assertEqual(prefill.call_count, 1)
        # The decode kernel on 16-token slices runs the same per-token math.
        decode = [
            self.kernel.attention_residual_decode_tpu(
                a_dev[i : i + 16],
                b_dev[i : i + 16] if has_addend else None,
                ar.block_residual[i : i + 16].contiguous(),
                nvb,
                attn_residual.get_cw(modules[0], modules[1]),
                EPS,
                self.kernel.get_norm_weight_f32(modules[2]),
                EPS,
            )[0]
            for i in range(0, min(tokens, 64), 16)
        ]
        normed = normed.cpu()
        decode = torch.cat([d.cpu() for d in decode])
        err = (normed.float() - expected.float()).abs().max().item()
        head = normed[: decode.shape[0]]
        flips = head != decode
        print(
            f"T={tokens} nvb={nvb} addend={has_addend} max|kernel-ref|={err:.3g} "
            f"elements != decode kernel: {int(flips.sum())}"
        )
        # The decode kernel's math in its order: equal per token, up to a rare
        # one-ulp rounding flip of the bf16 output.
        self.assertLessEqual(flips.float().mean().item(), 1e-4)
        ulp = 2.0**-7 * torch.maximum(head.float().abs(), decode.float().abs())
        self.assertTrue(bool(((head.float() - decode.float()).abs() <= ulp).all()))
        self.assertEqual(normed.dtype, torch.bfloat16)
        if has_addend:
            # The kernel materializes the pending add like a torch bf16 add.
            torch.testing.assert_close(prefix_out.cpu(), prefix, rtol=0, atol=0)
        else:
            self.assertIs(prefix_out, a_dev)
        # Both sides score in f32 and round the output to bf16 (as the decode
        # kernel's test: a few bf16 ulps).
        torch.testing.assert_close(
            normed.float(), expected.float(), rtol=1.6e-2, atol=1e-2
        )

    def test_matches_fp32_reference(self):
        for tokens, nvb, has_addend in (
            (256, 1, False),  # the smallest prefill-sized batch
            (288, 3, True),
            (288, NB, False),
            (4096, NB, True),  # Kimi-K3.1's prefill chunk at the full bank
        ):
            with self.subTest(tokens=tokens, nvb=nvb, has_addend=has_addend):
                self._check(tokens, nvb, has_addend)

    def test_shape_gate(self):
        """Batches of at least 256 tokens in whole 32-token tiles, with a
        block-major bf16 bank of matching shape, and the decode gate's row,
        addend and RMSNorm conditions."""
        supports = self.kernel.supports_attn_res_prefill
        norm, hf_norm = RMSNorm(H), RMSNorm(H, cast_x_before_out_mul=True)

        def rows(tokens, dtype=torch.bfloat16, hidden=H):
            return torch.empty(tokens, hidden, dtype=dtype)

        def bank(tokens, hidden=H, dtype=torch.bfloat16, block_major=True):
            if block_major:
                return torch.empty(NB, tokens, hidden, dtype=dtype).permute(1, 0, 2)
            return torch.empty(tokens, NB, hidden, dtype=dtype)

        def gate(a, b=None, bk=None, score=norm, out=norm):
            bk = bank(a.shape[0], a.shape[1]) if bk is None else bk
            return supports(a, b, bk, score, out)

        self.assertTrue(gate(rows(256)))
        self.assertTrue(gate(rows(4096), rows(4096)))
        self.assertFalse(gate(rows(256), bk=bank(256, block_major=False)))
        self.assertFalse(gate(rows(16)))  # decode sizes
        self.assertFalse(gate(rows(64)))
        self.assertFalse(gate(rows(272)))  # not whole tiles
        self.assertFalse(gate(rows(256), bk=bank(256, dtype=torch.float32)))
        self.assertFalse(gate(rows(256, torch.float32), bk=bank(256)))
        self.assertFalse(gate(rows(256), bk=bank(288)))
        self.assertFalse(gate(rows(256), bk=bank(256, hidden=H - 128)))
        self.assertFalse(gate(rows(256, hidden=H + 64), bk=bank(256, hidden=H + 64)))
        self.assertFalse(gate(rows(256), rows(288)))
        self.assertFalse(gate(rows(256), score=hf_norm))
        self.assertFalse(gate(rows(256), out=hf_norm))

    def test_compiled_stack_matches_eager(self):
        """A K3-like stack (block-major bank, bank writes, pending adds, nvb up to
        4) compiles to one graph that calls the prefill op registered at init."""
        tokens = 288
        dev = torch.device("tpu")
        points = [_modules(200 + i) for i in range(2 * (NB - 1) + 1)]
        for proj, norm, out in points:  # post_load_weights state
            for m in (proj, norm, out):
                m.to(dev)
            attn_residual.get_cw(proj, norm)
            self.kernel.get_norm_weight_f32(out)
        x = torch.randn(tokens, H, generator=torch.Generator().manual_seed(7))
        x = x.to(torch.bfloat16)

        def stack(hidden):
            ar = AttnResidual(hidden, NB)
            hooks.block_major_bank(None, ar, hidden, NB)
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
            self._plugin(),
            patch.object(
                self.kernel,
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
        self.assertEqual(len(graphs), 1)
        self.assertIn("kimi_attn_res_prefill", graphs[0].code)
        torch.testing.assert_close(out.float(), eager.float(), rtol=1.6e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
