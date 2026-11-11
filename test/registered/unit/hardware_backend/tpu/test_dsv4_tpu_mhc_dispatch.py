# SPDX-License-Identifier: Apache-2.0
"""DSV4 public mHC dispatch remains traceable at the model's D5120 boundary."""

import functools
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Load the implementation without importing unrelated scheduler hooks (MLX on
# macOS). The device preflight separately exercises normal plugin activation.
_spec = importlib.util.spec_from_file_location(
    "_dsv4_mhc_hooks", Path(__file__).resolve().parents[5] / "sglang_tpu/hooks/dsv4.py"
)
_hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hooks)
tpu_dsv4_hc_combine = _hooks.tpu_dsv4_hc_combine
tpu_dsv4_hc_split_sinkhorn = _hooks.tpu_dsv4_hc_split_sinkhorn

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSV4MHCPublicDispatch(CustomTestCase):
    def test_real_model_methods_fullgraph(self):
        layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.hc_mult = 4
        layer.hc_sinkhorn_iters = 20
        layer.hc_eps = layer.rms_norm_eps = 1e-6
        layer.config = SimpleNamespace(model_type="deepseek_v41")
        norm = RMSNorm(5120, eps=1e-6).to(dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.7, 1.3, 5120))

        def forward(x, pre, fn, scale, base):
            combined = mhc.hc_combine(x.flatten(1), pre, 4, x.dtype)
            normalized = layer._hc_combine(x, pre, norm)
            first = layer._hc_combine(x, None, norm)
            p, q, c = layer._hc_mix_stats_impl(x, fn, scale, base)
            updated = layer.hc_post(normalized, x, q, c)
            public_post = mhc.hc_post(normalized, x.flatten(1), q, c.flatten(1), 4)
            public_input, public_res, public_coeff, _ = mhc.hc_pre(
                x.flatten(1), fn, scale, base, 4, 1e-6, 1e-6, 20
            )
            return (
                combined,
                normalized,
                first,
                p,
                q,
                c,
                updated,
                public_post,
                public_input,
                public_res,
                public_coeff,
            )

        # Resolve the real fused-op platform method before Dynamo, as model
        # loading/warmup does; platform discovery is outside this mHC contract.
        norm(torch.zeros(1, 5120, dtype=torch.bfloat16))

        # Install the real plugin implementations at their public dependency
        # boundary; no model method or mathematical behavior is mocked.
        with (
            patch.object(
                mhc, "hc_combine", functools.partial(tpu_dsv4_hc_combine, None)
            ),
            patch.object(
                mhc,
                "hc_split_sinkhorn",
                functools.partial(tpu_dsv4_hc_split_sinkhorn, None),
            ),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False),
            envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False),
        ):
            compiled = torch.compile(forward, backend="eager", fullgraph=True)
            for tokens in (1, 16):
                with self.subTest(tokens=tokens):
                    torch.manual_seed(137 + tokens)
                    x = torch.randn(tokens, 4, 5120).bfloat16()
                    pre = torch.rand(tokens, 4)
                    fn = torch.randn(24, 20480) / 20480**0.5
                    scale = torch.tensor([0.1, 0.2, 0.3])
                    base = torch.randn(24) * 0.1
                    acc = torch.zeros(tokens, 5120)
                    for k in range(4):
                        acc = acc + pre[:, k : k + 1] * x[:, k, :].float()
                    combined = acc.bfloat16()

                    def reference_norm(y):
                        y = y.float()
                        return (
                            y
                            * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
                            * norm.weight.float()
                        ).bfloat16()

                    normalized = reference_norm(combined)
                    flat = x.flatten(1).float()
                    mixes = (
                        torch.nn.functional.linear(flat, fn)
                        * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
                    ).unsqueeze(1)
                    # Independent gate equations (not the production Sinkhorn helper).
                    raw = mixes[:, 0, :]
                    p = torch.sigmoid(raw[:, :4] * scale[0] + base[:4]) + 1e-6
                    q = 2 * torch.sigmoid(raw[:, 4:8] * scale[1] + base[4:8])
                    logits = (raw[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
                    c = torch.exp(logits - logits.amax(-1, keepdim=True))
                    c = c / c.sum(-1, keepdim=True) + 1e-6
                    c = c / (c.sum(-2, keepdim=True) + 1e-6)
                    for _ in range(19):
                        c = c / (c.sum(-1, keepdim=True) + 1e-6)
                        c = c / (c.sum(-2, keepdim=True) + 1e-6)
                    updated = (
                        q[:, :, None] * normalized.float()[:, None, :]
                        + (c[:, :, :, None] * x.float()[:, :, None, :]).sum(1)
                    ).bfloat16()
                    public_c = logits.softmax(-1) + 1e-6
                    public_c = public_c / (public_c.sum(-2, keepdim=True) + 1e-6)
                    for _ in range(19):
                        public_c = public_c / (public_c.sum(-1, keepdim=True) + 1e-6)
                        public_c = public_c / (public_c.sum(-2, keepdim=True) + 1e-6)
                    expected = (
                        combined,
                        normalized,
                        reference_norm(x[:, 0, :]),
                        p,
                        q,
                        c,
                        updated,
                        updated.flatten(1),
                        (p[:, :, None] * x.float()).sum(1).bfloat16(),
                        public_c.flatten(1),
                        q,
                    )
                    actual = compiled(x, pre, fn, scale, base)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
