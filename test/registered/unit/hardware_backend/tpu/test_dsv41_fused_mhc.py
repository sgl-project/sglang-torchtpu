"""DeepSeek-V4.1 fused mHC post-pre hooks (sglang_tpu/models/deepseek_v41/fused_mhc.py).

Routing runs with a mocked kernel op. The equivalence test runs the Pallas kernel
in interpret mode (needs jax) through the hook and compares it with the torch
path the hook replaces: core post, then the TPU combine (tpu_dsv41_hc_combine)
and RMSNorm, then the next coefficients as _mix_stats_impl computes them off
CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import inspect
import pkgutil
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.plugins.hook_registry import HookType
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks.dsv4 import LiveHcSubLayer
from sglang_tpu.models.deepseek_v41 import fused_mhc
from sglang_tpu.models.deepseek_v41.hooks import (
    fused_mhc_hook_specs,
    tpu_dsv41_hc_combine,
)

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sglang_tpu.kernels.mhc_v41 import param_row, v41_post_pre
except ImportError:
    jax = None

HC, D, ITERS = 4, 256, 20
MIX = (2 + HC) * HC
RMS_EPS, HC_EPS, NORM_EPS = 1e-6, 1e-6, 1e-6
CFG = mhc.HcConfig(
    mult=HC,
    sinkhorn_iters=ITERS,
    eps=HC_EPS,
    rms_eps=RMS_EPS,
    hidden=D,
    pre_from_prev=True,
    cp_prefill=False,
)


def _boundary(tokens, seed=0, fn_parts=True):
    """A boundary's inputs: this sublayer's hc and coefficients, and the next
    boundary whose hc carries the split fn."""
    gen = torch.Generator().manual_seed(seed)
    y = torch.randn(tokens, D, generator=gen).bfloat16()
    residual = torch.randn(tokens, HC, D, generator=gen).bfloat16()
    pre = torch.sigmoid(torch.randn(tokens, HC, generator=gen)) + HC_EPS
    post = 2 * torch.sigmoid(torch.randn(tokens, HC, generator=gen))
    comb = torch.softmax(torch.randn(tokens, HC, HC, generator=gen), dim=-1)
    fn = torch.randn(MIX, HC * D, generator=gen) * 0.02
    norm = RMSNorm(D, eps=NORM_EPS)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(D, generator=gen))
    next_hc = SimpleNamespace(
        cfg=CFG,
        fn=fn,
        scale=torch.tensor([0.7, 1.3, 2.0]),
        base=torch.randn(MIX, generator=gen) * 0.5,
        norm=norm,
        fused_fn_parts=fused_mhc.pack_fn(fused_mhc.split_fn(fn)) if fn_parts else None,
    )
    next = mhc.HcNextBoundary(
        norm=norm, accepts_mxfp8=False, norm_fusable=True, hc=next_hc
    )
    hc = SimpleNamespace(cfg=CFG)
    return hc, y, residual, (pre, post, comb), next


def _interpret_op(y, res, pre, post, comb, fn_parts, base, scale, norm_w, hc, *rest):
    """The custom op's body in Pallas interpret mode on CPU tensors."""
    iters, rms_eps, hc_eps, norm_eps = rest

    def to_jax(t):
        return jnp.asarray(t.detach().float().numpy()).astype(
            jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
        )

    outs = v41_post_pre(
        *map(to_jax, (y, res, pre, post, comb, fn_parts)),
        param_row(to_jax(base), to_jax(scale)),
        to_jax(norm_w),
        hc=hc,
        iters=iters,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        interpret=True,
    )
    dtypes = (y.dtype, y.dtype, torch.float32, torch.float32, torch.float32)
    return tuple(
        torch.from_numpy(np.asarray(o.astype(jnp.float32))).to(dt)
        for o, dt in zip(outs, dtypes)
    )


def _torch_reference(hc, y, residual, coefficients, next):
    """What the next sublayer reads on the torch path: new streams, its normed
    input and its coefficients."""
    from sglang.srt.models.deepseek_v4 import _hc_split_sinkhorn_torch

    state = mhc._post_fusion(hc, y, residual, coefficients, next)
    streams = state.residual
    x_flat = streams.flatten(1)
    u = next.norm.forward_native(
        tpu_dsv41_hc_combine(None, x_flat, state.pre, HC, streams.dtype)
    )
    x32 = x_flat.float()
    rsqrt = torch.rsqrt(x32.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = (F.linear(x32, next.hc.fn) * rsqrt).unsqueeze(1)
    stats = _hc_split_sinkhorn_torch(
        mixes, next.hc.scale, next.hc.base, HC, ITERS, HC_EPS
    )
    return streams, u, tuple(s.squeeze(1) for s in stats)


class TestDsv41FusedMhc(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for env in (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST,
            envs.SGLANG_OPT_USE_FLASHINFER_MHC,
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL,
        ):
            self.stack.enter_context(env.override(False))
        self.stack.enter_context(patch.object(fused_mhc, "_DEVICE_TYPE", "cpu"))

    def _kernel(self, op, min_tokens=128):
        self.stack.enter_context(patch.object(fused_mhc, "_post_pre_op", op))
        self.stack.enter_context(patch.object(fused_mhc, "_min_tokens", min_tokens))

    def test_hooks_registered_only_with_the_flag(self):
        with tpu_envs.SGLANG_TPU_DSV41_FUSED_MHC.override(False):
            self.assertEqual(fused_mhc_hook_specs(), [])
        with tpu_envs.SGLANG_TPU_DSV41_FUSED_MHC.override(True):
            specs = fused_mhc_hook_specs()
        self.assertEqual(
            [(h, t) for _, h, t in specs],
            [
                (fused_mhc.cache_fn_parts, HookType.AFTER),
                (fused_mhc.post_fusion, HookType.AROUND),
            ],
        )
        for target, _, _ in specs:
            pkgutil.resolve_name(target)
        # The AROUND hook names its target's parameters.
        target = pkgutil.resolve_name(specs[1][0])
        params = list(inspect.signature(fused_mhc.post_fusion).parameters)
        self.assertEqual(params[0], "original")
        self.assertEqual(params[1:], list(inspect.signature(target).parameters))

    def test_split_fn_is_exact(self):
        fn = torch.randn(MIX, HC * D) * 0.02
        parts = fused_mhc.split_fn(fn)
        self.assertEqual(parts.shape, (3, MIX, HC * D))
        self.assertEqual(parts.dtype, torch.bfloat16)
        torch.testing.assert_close(parts.float().sum(0), fn, rtol=0, atol=0)

    def test_pack_fn_layout(self):
        parts = fused_mhc.split_fn(torch.randn(MIX, HC * D) * 0.02)
        packed = fused_mhc.pack_fn(parts)
        self.assertEqual(packed.shape, (HC * D, 128))
        self.assertTrue(packed.is_contiguous())
        torch.testing.assert_close(
            packed[:, : 3 * MIX].t().reshape(3, MIX, HC * D), parts, rtol=0, atol=0
        )
        self.assertEqual(int(packed[:, 3 * MIX :].count_nonzero()), 0)

    def test_fn_parts_cached_on_boundary_layers_only(self):
        def layer(boundary_mhc_fusions):
            module = torch.nn.Module()
            module.hc_pre_from_prev_sublayer = True
            module.boundary_mhc_fusions = boundary_mhc_fusions
            module.hc_attn_fn = torch.nn.Parameter(torch.randn(MIX, HC * D))
            module.hc_ffn_fn = torch.nn.Parameter(torch.randn(MIX, HC * D))
            return module

        decoder, draft_stage = layer(True), layer(False)
        with patch.object(fused_mhc, "register_ops") as register:
            fused_mhc.cache_fn_parts(None, decoder)
            fused_mhc.cache_fn_parts(None, draft_stage)
        register.assert_called_once()
        buffers = dict(decoder.named_buffers())
        for attn, fn in ((True, decoder.hc_attn_fn), (False, decoder.hc_ffn_fn)):
            parts = LiveHcSubLayer(decoder, attn=attn).fused_fn_parts
            torch.testing.assert_close(
                parts, fused_mhc.pack_fn(fused_mhc.split_fn(fn.data))
            )
            name = fused_mhc.FN_PARTS_ATTRS[attn]
            self.assertIs(buffers[name], parts)
            self.assertNotIn(name, decoder.state_dict())
        self.assertIsNone(LiveHcSubLayer(draft_stage, attn=True).fused_fn_parts)

    def test_routing(self):
        original = MagicMock(return_value="torch")
        op = MagicMock(
            side_effect=lambda y, res, *a: (
                torch.empty_like(res),
                torch.empty_like(y),
                torch.empty(y.shape[0], HC),
                torch.empty(y.shape[0], HC),
                torch.empty(y.shape[0], HC * HC),
            )
        )

        def run(case):
            original.reset_mock()
            op.reset_mock()
            return fused_mhc.post_fusion(original, *case)

        # Kernel not built (flag off): the torch path.
        self.assertEqual(run(_boundary(128)), "torch")

        self._kernel(op, min_tokens=128)
        hc, y, residual, coefficients, next = _boundary(128)
        declined = {
            "below the row threshold": _boundary(127),
            "no next boundary": (hc, y, residual, coefficients, None),
            "next hc without fn parts": _boundary(128, fn_parts=False),
            "fp32 streams": (hc, y, residual.float(), coefficients, next),
            "fused norm cast": (
                hc,
                y,
                residual,
                coefficients,
                next._replace(
                    norm=RMSNorm(D, eps=NORM_EPS, cast_x_before_out_mul=True)
                ),
            ),
            "same-sublayer pre": (
                SimpleNamespace(cfg=CFG._replace(pre_from_prev=False)),
                y,
                residual,
                coefficients,
                next,
            ),
        }
        for why, case in declined.items():
            with self.subTest(why):
                self.assertEqual(run(case), "torch")
                op.assert_not_called()

        state = run((hc, y, residual, coefficients, next))
        op.assert_called_once()
        original.assert_not_called()
        self.assertIsInstance(state.input, mhc.HcNormed)
        self.assertIs(state.pre, coefficients[0])
        self.assertEqual(state.residual.shape, (128, HC, D))
        shapes = [tuple(s.shape) for s in state.stats]
        self.assertEqual(shapes, [(128, HC), (128, HC), (128, HC, HC)])

    @unittest.skipIf(jax is None, "requires jax")
    def test_matches_torch_path(self):
        self._kernel(_interpret_op, min_tokens=1)
        for tokens in (8, 130):
            with self.subTest(tokens=tokens):
                case = _boundary(tokens, seed=tokens)
                fused = fused_mhc.post_fusion(mhc._post_fusion, *case)
                streams, u, stats = _torch_reference(*case)
                # bf16 outputs: one rounding of the same f32 sums, summed in
                # another order; the coefficients see those streams.
                torch.testing.assert_close(
                    fused.residual.float(), streams.float(), rtol=1e-2, atol=1e-2
                )
                torch.testing.assert_close(
                    fused.input.rows.float(), u.float(), rtol=1e-2, atol=1e-2
                )
                for got, want in zip(fused.stats, stats):
                    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
