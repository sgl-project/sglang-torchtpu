# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 fused QK RMSNorm + RoPE on TPU (minimax_h3_qk_norm_rope).

The kernel is checked against a float32 CPU reference of the CUDA fused
kernel's numerics (RMSNorm in float32 rounded to bf16, then the half-split
rotation of the first 96 of 128 head dims in float32, rounded once) and against
the unfused path it replaces, at H3's 56 heads and at 8, and at row counts that
are not a multiple of the kernel's row block. Head counts that are not a
multiple of 8 (7 heads per rank when tensor parallelism splits H3's 56) keep
the unfused path, which is faster there.
"""

import contextlib
import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.rotary_embedding import RotaryEmbedding
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen import minimax_h3_qk_norm_rope as qk_rope
from sglang_tpu.multimodal_gen import worker

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

EPS = 1e-5  # MiniMaxH3DiTArchConfig.qk_norm_eps
# (rows, q heads, k heads); no row count is a multiple of its row block (32, 224
# and 56 rows).
SHAPES = ((37, 56, 56), (4103, 8, 8), (100, 8, 56))


@contextlib.contextmanager
def _installed(original):
    """install() over ``original``, with the module's binding restored afterwards."""
    with (
        patch.object(dit, "_apply_qk_norm_rope", original),
        patch.object(qk_rope, "_INSTALLED", False),
    ):
        qk_rope.install()
        qk_rope.install()  # idempotent: must not wrap its own wrapper
        yield dit._apply_qk_norm_rope


def _norms(device):
    gen = torch.Generator().manual_seed(1)
    out = []
    for _ in range(2):
        norm = nn.RMSNorm(128, eps=EPS, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + 0.5 * torch.randn(128, generator=gen))
        out.append(norm.to(device))
    return out


def _inputs(rows, q_heads, k_heads, device):
    gen = torch.Generator().manual_seed(rows * 131 + q_heads)
    q = (torch.randn(rows, q_heads, 128, generator=gen) * 4).bfloat16()
    k = (torch.randn(rows, k_heads, 128, generator=gen) * 4).bfloat16()
    # H3's table: 48 frequencies duplicated to 96, as cos | sin in bf16.
    half = torch.rand(rows, 48, generator=gen) * 200
    cache = dit._rope_cos_sin_cache(
        torch.cat((half, half), dim=-1), dtype=torch.bfloat16
    )
    positions = torch.arange(rows)
    return [t.to(device) for t in (q, k, cache, positions)]


def _reference(x, norm, cache):
    """float32 RMSNorm rounded to bf16, then RoPE on dims [0, 96) in float32."""
    x = x.cpu().float()
    w = norm.weight.detach().cpu().float()
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w).bfloat16()
    normed = normed.float()
    cos, sin = cache.cpu().float()[:, None, :].split(48, dim=-1)
    x1, x2, rest = normed.split((48, 48, 32), dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)
    # Largest change one bf16 rounding flip of the normed input can cause.
    partner = torch.cat((x2, x1, torch.zeros_like(rest)), dim=-1)
    flip = (normed.abs() + partner.abs()) * 2**-7
    return out, flip


def _assert_close(test, got, want, flip):
    """``got`` is ``want`` rounded to bf16, up to rare rounding flips of the norm.

    TPU and CPU rsqrt (or two XLA fusions) may round differently, which can flip
    the bf16 rounding of a normed value; beyond that only the final rounding of
    the output may differ.
    """
    got, want = got.cpu().float(), want.cpu().float()
    test.assertEqual(got.shape, want.shape)
    bound = flip + want.abs() * 2**-7 + 2**-126
    test.assertTrue(bool(((got - want).abs() <= bound).all()))
    test.assertLess((got != want.bfloat16().float()).float().mean().item(), 1e-4)


class TestMiniMaxH3QkNormRopeTpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if is_in_ci():
                raise RuntimeError("TPU CI requires torch_tpu and a TPU device")
            raise unittest.SkipTest("torch_tpu is not installed")
        from sglang_tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        # Do not turn initialization/lowering failures into a successful skip.
        torch.zeros(1, device="tpu").cpu()

    def test_kernel_matches_fp32_reference(self):
        q_norm, k_norm = _norms("tpu")
        original = Mock(side_effect=AssertionError("fell back to the unfused path"))
        with _installed(original) as fused:
            for rows, q_heads, k_heads in SHAPES:
                with self.subTest(rows=rows, q_heads=q_heads, k_heads=k_heads):
                    q, k, cache, pos = _inputs(rows, q_heads, k_heads, "tpu")
                    outs = fused(q, k, q_norm, k_norm, 128, cache, pos, None)
                    for out, x, norm in zip(outs, (q, k), (q_norm, k_norm)):
                        _assert_close(self, out, *_reference(x, norm, cache))

    def test_kernel_matches_the_path_it_replaces(self):
        q_norm, k_norm = _norms("tpu")
        rotary = RotaryEmbedding(
            head_size=128,
            rotary_dim=96,
            is_neox_style=True,
            use_precomputed_cache=False,
        )
        original = dit._apply_qk_norm_rope
        with _installed(original) as fused:
            for rows, heads, _ in SHAPES[:2]:
                with self.subTest(rows=rows, heads=heads):
                    q, k, cache, pos = _inputs(rows, heads, heads, "tpu")
                    args = (q, k, q_norm, k_norm, 128, cache, pos, rotary)
                    for got, want, x, norm in zip(
                        fused(*args), original(*args), (q, k), (q_norm, k_norm)
                    ):
                        _assert_close(self, got, want, _reference(x, norm, cache)[1])

    def test_compiled_matches_eager(self):
        # The DiT blocks call the op inside torch.compile.
        import torch_tpu._internal.compile  # noqa: F401

        q_norm, k_norm = _norms("tpu")
        q, k, cache, pos = _inputs(37, 56, 56, "tpu")
        original = Mock(side_effect=AssertionError("fell back to the unfused path"))
        # Inference, as the DiT runs: the op registers no autograd formula.
        with _installed(original) as fused, torch.no_grad():
            compiled = torch.compile(
                lambda q, k: fused(q, k, q_norm, k_norm, 128, cache, pos, None),
                backend="tpu",
                dynamic=False,
            )
            eager = fused(q, k, q_norm, k_norm, 128, cache, pos, None)
            for got, want in zip(compiled(q, k), eager):
                torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)

    def test_ineligible_calls_keep_the_original(self):
        q_norm, k_norm = _norms("tpu")
        q, k, cache, pos = _inputs(37, 8, 8, "tpu")
        other_eps = nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16).to("tpu")
        no_eps = nn.RMSNorm(128, dtype=torch.bfloat16).to("tpu")
        no_weight = nn.RMSNorm(128, eps=EPS, elementwise_affine=False)
        cpu_q_norm, cpu_k_norm = _norms("cpu")
        full_rotation = torch.cat((cache, cache[:, :32]), dim=-1)
        cases = {
            "cpu": (q.cpu(), k.cpu(), cpu_q_norm, cpu_k_norm, 128, cache.cpu()),
            "float32": (q.float(), k.float(), q_norm, k_norm, 128, cache),
            "head_dim_64": (q[..., :64], k[..., :64], q_norm, k_norm, 64, cache),
            "batched": (q[None], k[None], q_norm, k_norm, 128, cache),
            "7_heads": (q[:, :7], k[:, :7], q_norm, k_norm, 128, cache),
            "k_12_heads": (q, torch.cat((k, k[:, :4]), 1), q_norm, k_norm, 128, cache),
            "full_rotation": (q, k, q_norm, k_norm, 128, full_rotation),
            # A positions-indexed table, not one row per token.
            "table_rows": (q, k, q_norm, k_norm, 128, cache[:16]),
            "eps_mismatch": (q, k, q_norm, other_eps, 128, cache),
            "eps_none": (q, k, no_eps, no_eps, 128, cache),
            "no_weight": (q, k, no_weight, no_weight, 128, cache),
        }
        sentinel = object()
        for name, args in cases.items():
            with self.subTest(name), _installed(Mock(return_value=sentinel)) as fn:
                args = (*args, pos, None)
                self.assertIs(fn(*args), sentinel)
        # The same tensors with every condition met take the kernel.
        with _installed(Mock(return_value=sentinel)) as fn:
            self.assertIsNot(fn(q, k, q_norm, k_norm, 128, cache, pos, None), sentinel)

    def test_worker_installs_the_patch(self):
        precision = torch.get_float32_matmul_precision()
        try:
            with (
                patch("sglang_tpu.patches._register_aten_fallbacks"),
                patch("sglang_tpu.multimodal_gen.rope.install"),
                patch("sglang_tpu.multimodal_gen.minimax_h3.install"),
                patch.object(qk_rope, "install") as install,
            ):
                worker.install_runtime_patches()
            install.assert_called_once_with()
        finally:
            torch.set_float32_matmul_precision(precision)


class TestMiniMaxH3QkNormRopeInstall(unittest.TestCase):
    """Host-only: needs neither torch_tpu nor a device."""

    def test_registration_failure_keeps_the_unfused_path(self):
        # Every TPU diffusion worker installs the patch; a failed op
        # registration must not stop a worker, H3 or not.
        original = Mock()
        with (
            patch.object(qk_rope, "_ensure_op", side_effect=RuntimeError("no op")),
            patch.object(qk_rope.logger, "warning") as warning,
            _installed(original) as fn,
        ):
            self.assertIs(fn, original)
        warning.assert_called()


if __name__ == "__main__":
    unittest.main()
