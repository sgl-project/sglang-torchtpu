# SPDX-License-Identifier: Apache-2.0
"""H3's native tensor paths must not require an optional GPU kernel runtime."""

import ast
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch

import sglang
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_MODEL_DIR = Path(sglang.__file__).parent / "multimodal_gen/runtime/models/dits"
_OPS_PATH = _MODEL_DIR / "minimax_h3_ops.py"


def _load_ops():
    # Test the real operation module without starting the unrelated diffusion
    # config/quantization registry in multimodal_gen.__init__.
    spec = importlib.util.spec_from_file_location(
        "minimax_h3_ops_under_test", _OPS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ops = _load_ops()


class TestMiniMaxH3NativeOps(unittest.TestCase):
    def test_indexed_modulation_preserves_values_dtype_and_inputs(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for rows in (0, 5):
                for strided in (False, True):
                    with self.subTest(dtype=dtype, rows=rows, strided=strided):
                        x = torch.randn(rows, 32, dtype=dtype)
                        if strided:
                            x = x[:, ::2]
                        width = x.shape[-1]
                        shift, scale, gate = torch.randn(3, 3, width, dtype=dtype)
                        other = torch.randn_like(x)
                        indices = torch.tensor([2, 0, 2, 1, 0][:rows], dtype=torch.long)
                        before = [t.clone() for t in (x, shift, scale, gate, other)]
                        for out_dtype in (dtype, torch.float32):
                            expected = (x * (1 + scale[indices]) + shift[indices]).to(
                                out_dtype
                            )
                            actual = ops._modulate_scale_shift(
                                x, shift, scale, indices, dtype=out_dtype
                            )
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            for allow_inplace in (False, True):
                                actual = ops._modulate_gate(
                                    x,
                                    gate,
                                    other,
                                    indices,
                                    dtype=out_dtype,
                                    allow_inplace=allow_inplace,
                                )
                                expected = (x + gate[indices] * other).to(out_dtype)
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                        for tensor, saved in zip(
                            (x, shift, scale, gate, other), before
                        ):
                            torch.testing.assert_close(tensor, saved, rtol=0, atol=0)

    def test_silu_reuse_flag_keeps_native_input_unchanged(self):
        torch.manual_seed(1)
        for dtype in (torch.float32, torch.bfloat16):
            for width in (14, 32):
                for reuse in (False, True):
                    hidden = torch.randn(5, width * 2, dtype=dtype)[:, ::2]
                    before = hidden.clone()
                    gate, up = hidden.chunk(2, dim=-1)
                    actual = ops._silu_mul(hidden, reuse_input=reuse)
                    expected = torch.nn.functional.silu(gate) * up
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(hidden, before, rtol=0, atol=0)

    def test_qk_norm_matches_native_modules(self):
        torch.manual_seed(2)
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(5, 3, 64, dtype=dtype)
            k = torch.randn(5, 1, 64, dtype=dtype)
            for parameter_dtype in (dtype, torch.float32):
                q_norm = torch.nn.RMSNorm(64, eps=1e-6, dtype=parameter_dtype)
                k_norm = torch.nn.RMSNorm(64, eps=1e-5, dtype=parameter_dtype)
                with torch.no_grad():
                    q_norm.weight.uniform_(0.5, 1.5)
                    k_norm.weight.uniform_(0.5, 1.5)
                    before = (q.clone(), k.clone())
                    actual = ops._apply_qk_norm(q, k, q_norm, k_norm, 64)
                    for result, expected in zip(actual, (q_norm(q), k_norm(k))):
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    for tensor, saved in zip((q, k), before):
                        torch.testing.assert_close(tensor, saved, rtol=0, atol=0)

    def test_native_paths_capture_as_full_graph(self):
        torch.manual_seed(3)
        norm = torch.nn.RMSNorm(32, eps=1e-6)

        def forward(x, shift, scale, gate, indices):
            modulated = ops._modulate_scale_shift(
                x, shift, scale, indices, dtype=x.dtype
            )
            gated = ops._modulate_gate(x, gate, modulated, indices, dtype=x.dtype)
            activated = ops._silu_mul(torch.cat((x, gated), dim=-1), reuse_input=True)
            return ops._apply_qk_norm(activated, gated, norm, norm, 32)

        inputs = (
            torch.randn(5, 32),
            *torch.randn(3, 3, 32),
            torch.tensor([2, 0, 2, 1, 0]),
        )
        with torch.no_grad():
            expected = forward(*inputs)
            # CPU capture only; this is not evidence of TPU lowering.
            actual = torch.compile(forward, backend="eager", fullgraph=True)(*inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_gpu_imports_are_not_needed_by_native_operations(self):
        code = textwrap.dedent(f"""
            import importlib.abc
            import importlib.util
            import sys
            import torch

            class NoGPUKernels(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    roots = ('triton', 'tvm_ffi', 'sgl_kernel', 'sglang.kernels')
                    if any(fullname == root or fullname.startswith(root + '.') for root in roots):
                        raise AssertionError('GPU dependency imported: ' + fullname)

            sys.meta_path.insert(0, NoGPUKernels())
            spec = importlib.util.spec_from_file_location('h3_ops', {_OPS_PATH.as_posix()!r})
            ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ops)
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.ones(3, 32, dtype=dtype)
                table = torch.ones(2, 32, dtype=dtype)
                indices = torch.tensor([1, 0, 1])
                ops._modulate_scale_shift(x, table, table, indices, dtype=dtype)
                for inplace in (False, True):
                    ops._modulate_gate(x, table, x, indices, dtype=dtype, allow_inplace=inplace)
                ops._silu_mul(x, reuse_input=True)
                norm = torch.nn.RMSNorm(32, eps=1e-6, dtype=dtype)
                ops._apply_qk_norm(x, x, norm, norm, 32)
            """)
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_model_does_not_eagerly_import_gpu_kernel_symbols(self):
        # Also guard the model integration: the helper alone is insufficient if
        # the caller still resolves GPU symbols before checking the platform.
        tree = ast.parse((_MODEL_DIR / "minimax_h3.py").read_text())
        eager = [
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("sglang.kernels.")
        ]
        self.assertEqual(eager, [])


class TestMiniMaxH3CudaRouting(unittest.TestCase):
    def test_existing_cuda_branches_still_resolve_the_same_kernels(self):
        # Routing only, not CUDA numerical validation. No CUDA runtime required.
        x = Mock(is_cuda=True, dtype=torch.bfloat16, shape=(5, 64))
        x.is_contiguous.return_value = True
        x.stride.side_effect = lambda dim: {-1: 1, -2: 64}[dim]
        indices = torch.tensor([0, 1, 0, 1, 0])
        norm = Mock(eps=1e-6, weight=Mock(dtype=torch.bfloat16))
        diffusion = ModuleType("sglang.kernels.ops.diffusion")
        activation = ModuleType("sglang.kernels.ops.activation.activation")
        layernorm = ModuleType("sglang.kernels.ops.layernorm.norm")
        for module, names in (
            (
                diffusion,
                (
                    "indexed_scale_shift_bf16_",
                    "indexed_gate_bf16_",
                    "indexed_gate_bf16",
                ),
            ),
            (activation, ("silu_and_mul_with_activation_rounding_",)),
            (layernorm, ("fused_inplace_qknorm",)),
        ):
            for name in names:
                setattr(module, name, Mock())
        with patch.dict(
            sys.modules, {m.__name__: m for m in (diffusion, activation, layernorm)}
        ):
            result = ops._modulate_scale_shift(x, x, x, indices, dtype=x.dtype)
            diffusion.indexed_scale_shift_bf16_.assert_called_once_with(
                x, x, x, indices
            )
            self.assertIs(result, diffusion.indexed_scale_shift_bf16_.return_value)
            for inplace, name in (
                (True, "indexed_gate_bf16_"),
                (False, "indexed_gate_bf16"),
            ):
                result = ops._modulate_gate(
                    x, x, x, indices, dtype=x.dtype, allow_inplace=inplace
                )
                getattr(diffusion, name).assert_called_once_with(x, x, x, indices)
                self.assertIs(result, getattr(diffusion, name).return_value)
            self.assertIs(
                ops._silu_mul(x, reuse_input=True),
                activation.silu_and_mul_with_activation_rounding_.return_value,
            )
            activation.silu_and_mul_with_activation_rounding_.assert_called_once_with(x)
            q, k = ops._apply_qk_norm(x, x, norm, norm, 64)
            self.assertIs(q, x)
            self.assertIs(k, x)
            layernorm.fused_inplace_qknorm.assert_called_once_with(
                x, x, norm.weight, norm.weight, eps=1e-6, head_dim=64
            )


if __name__ == "__main__":
    unittest.main()
